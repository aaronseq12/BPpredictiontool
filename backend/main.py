"""
Main Flask API application for Blood Pressure Prediction Tool.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging
import traceback

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from marshmallow import ValidationError
import numpy as np
import structlog

# Add backend directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import config
from api.validation_schemas import (
    prediction_request_schema,
    prediction_response_schema,
    error_response_schema,
    health_check_response_schema,
)
from data_processing.advanced_preprocessor import preprocessor
from models.advanced_lstm_model import bp_model
from utils.signal_processing import AdvancedSignalProcessor

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize Flask application
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.api.max_content_length

# Configure CORS
CORS(app, origins=config.api.cors_origins.split(","))

# Global variables for model and preprocessor
model_loaded = False
preprocessor_fitted = False


class BloodPressureCalculator:
    """Calculate blood pressure parameters from ABP waveform."""

    @staticmethod
    def calculate_bp_parameters(abp_waveform: np.ndarray) -> Dict[str, float]:
        """Calculate systolic, diastolic, and MAP from ABP waveform."""
        # Denormalize ABP values (assuming 0-1 normalization for 0-150 mmHg)
        abp_mmhg = abp_waveform * 150

        # Find peaks (systolic) and troughs (diastolic)
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(abp_mmhg, distance=50)
        troughs, _ = find_peaks(-abp_mmhg, distance=50)

        if len(peaks) > 0 and len(troughs) > 0:
            systolic = np.mean(abp_mmhg[peaks])
            diastolic = np.mean(abp_mmhg[troughs])
        else:
            # Fallback to simple min/max
            systolic = np.max(abp_mmhg)
            diastolic = np.min(abp_mmhg)

        # Calculate Mean Arterial Pressure
        mean_arterial_pressure = diastolic + (systolic - diastolic) / 3

        return {
            "systolic": float(systolic),
            "diastolic": float(diastolic),
            "map": float(mean_arterial_pressure),
        }


def load_model():
    """Load the trained model."""
    global model_loaded

    model_path = config.data.models_directory / "advanced_bp_model.keras"
    fallback_path = config.data.models_directory / "blood_pressure_model.keras"

    try:
        if model_path.exists():
            bp_model.load_model(str(model_path))
            logger.info("Advanced model loaded successfully")
        elif fallback_path.exists():
            bp_model.load_model(str(fallback_path))
            logger.info("Fallback model loaded successfully")
        else:
            logger.error("No trained model found. Please train the model first.")
            return False

        model_loaded = True
        return True

    except Exception as e:
        logger.error(
            "Error loading model", error=str(e), traceback=traceback.format_exc()
        )
        return False


def setup_preprocessor():
    """Setup the data preprocessor."""
    global preprocessor_fitted

    try:
        # Check if we have training data to fit the preprocessor
        if config.data.data_directory.exists():
            # For production, we should have pre-fitted scalers saved
            # For now, we'll mark it as fitted for prediction
            preprocessor_fitted = True
            logger.info("Preprocessor setup completed")
            return True
    except Exception as e:
        logger.error("Error setting up preprocessor", error=str(e))
        return False


def initialize_application():
    """Initialize the application components."""
    logger.info("Initializing Blood Pressure Prediction API")

    # Load model
    if not load_model():
        logger.error("Failed to load model")

    # Setup preprocessor
    if not setup_preprocessor():
        logger.error("Failed to setup preprocessor")


@app.before_request
def before_request():
    """Log incoming requests."""
    g.start_time = datetime.utcnow()
    logger.info(
        "Request started",
        method=request.method,
        url=request.url,
        remote_addr=request.remote_addr,
    )


@app.after_request
def after_request(response):
    """Log request completion."""
    duration = datetime.utcnow() - g.start_time
    logger.info(
        "Request completed",
        status_code=response.status_code,
        duration_ms=duration.total_seconds() * 1000,
    )
    return response


@app.errorhandler(ValidationError)
def handle_validation_error(e):
    """Handle validation errors."""
    logger.warning("Validation error", errors=e.messages)

    error_response = error_response_schema.dump(
        {
            "error": "Validation failed",
            "error_code": "VALIDATION_ERROR",
            "details": e.messages,
        }
    )

    return jsonify(error_response), 400


@app.errorhandler(Exception)
def handle_general_error(e):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(e), traceback=traceback.format_exc())

    error_response = error_response_schema.dump(
        {
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "details": {"message": str(e)} if config.api.debug else None,
        }
    )

    return jsonify(error_response), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        health_data = {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "timestamp": datetime.utcnow(),
            "version": "2.0.0",
        }

        response = health_check_response_schema.dump(health_data)
        status_code = 200 if model_loaded else 503

        return jsonify(response), status_code

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_blood_pressure():
    """Predict blood pressure from PPG and ECG signals."""
    try:
        # Check if model is loaded
        if not model_loaded:
            return jsonify(
                error_response_schema.dump(
                    {"error": "Model not loaded", "error_code": "MODEL_NOT_LOADED"}
                )
            ), 503

        # Validate request data
        request_data = prediction_request_schema.load(request.get_json())

        ppg_signal = request_data["ppg_signal"]
        ecg_signal = request_data["ecg_signal"]
        include_quality = request_data.get("include_quality_metrics", False)

        # Preprocess signals
        try:
            input_data, quality_report = preprocessor.preprocess_single_prediction(
                ppg_signal, ecg_signal
            )
        except Exception as e:
            logger.error("Preprocessing failed", error=str(e))
            return jsonify(
                error_response_schema.dump(
                    {
                        "error": "Signal preprocessing failed",
                        "error_code": "PREPROCESSING_ERROR",
                        "details": {"message": str(e)},
                    }
                )
            ), 400

        # Make prediction
        try:
            prediction = bp_model.predict(input_data)
            predicted_abp = prediction[
                0
            ].flatten()  # Remove batch and feature dimensions

            # Calculate blood pressure parameters
            bp_calculator = BloodPressureCalculator()
            bp_params = bp_calculator.calculate_bp_parameters(predicted_abp)

            # Calculate confidence score based on signal quality
            confidence_score = min(1.0, quality_report["overall_score"])

            # Prepare response
            response_data = {
                "predicted_abp": predicted_abp.tolist(),
                "systolic_bp": bp_params["systolic"],
                "diastolic_bp": bp_params["diastolic"],
                "mean_arterial_pressure": bp_params["map"],
                "confidence_score": confidence_score,
            }

            if include_quality:
                response_data["quality_metrics"] = quality_report

            response = prediction_response_schema.dump(response_data)

            logger.info(
                "Prediction completed",
                systolic=bp_params["systolic"],
                diastolic=bp_params["diastolic"],
                confidence=confidence_score,
            )

            return jsonify(response), 200

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return jsonify(
                error_response_schema.dump(
                    {
                        "error": "Prediction failed",
                        "error_code": "PREDICTION_ERROR",
                        "details": {"message": str(e)},
                    }
                )
            ), 500

    except ValidationError as e:
        raise  # Will be handled by the validation error handler
    except Exception as e:
        logger.error("Request handling failed", error=str(e))
        raise  # Will be handled by the general error handler


@app.route("/api/info", methods=["GET"])
def api_info():
    """Get API information and documentation."""
    info = {
        "name": "Blood Pressure Prediction API",
        "version": "2.0.0",
        "description": "Advanced API for predicting blood pressure from PPG and ECG signals",
        "endpoints": {
            "/health": "Health check endpoint",
            "/predict": "Predict blood pressure from signals",
            "/api/info": "API information",
        },
        "model_info": {
            "input_format": "PPG and ECG signals (1000 samples each)",
            "output_format": "ABP waveform and BP parameters",
            "features": [
                "Signal quality assessment",
                "Advanced preprocessing",
                "Attention-based LSTM model",
                "Confidence scoring",
            ],
        },
    }

    return jsonify(info), 200


if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create necessary directories
    config.data.models_directory.mkdir(parents=True, exist_ok=True)
    config.data.logs_directory.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting Blood Pressure Prediction API",
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug,
    )

    # Initialize components
    initialize_application()

    # Run the application
    app.run(host=config.api.host, port=config.api.port, debug=config.api.debug)
