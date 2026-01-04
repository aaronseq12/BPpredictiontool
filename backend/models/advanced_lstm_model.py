"""
Advanced LSTM model for blood pressure prediction with attention mechanism.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    TimeDistributed,
    Attention,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from config import config

logger = logging.getLogger(__name__)


class AttentionBloodPressureModel:
    """Advanced LSTM model with attention mechanism for BP prediction."""

    def __init__(self):
        self.model = None
        self.history = None

    def build_model(self, input_shape: Tuple[int, int] = None) -> Model:
        """Build the advanced LSTM model with attention."""
        if input_shape is None:
            input_shape = (config.model.sequence_length, 2)  # PPG + ECG

        # Input layer
        inputs = Input(shape=input_shape, name="signal_input")

        # Multi-layer LSTM with residual connections
        lstm_1 = LSTM(
            config.model.lstm_units,
            return_sequences=True,
            dropout=config.model.dropout_rate,
            recurrent_dropout=config.model.dropout_rate,
            kernel_regularizer=l2(0.001),
            name="lstm_layer_1",
        )(inputs)

        lstm_1_norm = LayerNormalization(name="lstm_1_norm")(lstm_1)

        lstm_2 = LSTM(
            config.model.lstm_units // 2,
            return_sequences=True,
            dropout=config.model.dropout_rate,
            recurrent_dropout=config.model.dropout_rate,
            kernel_regularizer=l2(0.001),
            name="lstm_layer_2",
        )(lstm_1_norm)

        lstm_2_norm = LayerNormalization(name="lstm_2_norm")(lstm_2)

        # Multi-head attention mechanism
        attention_output = MultiHeadAttention(
            num_heads=8,
            key_dim=config.model.lstm_units // 16,
            name="multi_head_attention",
        )(lstm_2_norm, lstm_2_norm)

        # Residual connection
        attention_residual = Add(name="attention_residual")(
            [lstm_2_norm, attention_output]
        )
        attention_norm = LayerNormalization(name="attention_norm")(attention_residual)

        # Final LSTM layer
        lstm_3 = LSTM(
            config.model.lstm_units // 4,
            return_sequences=True,
            dropout=config.model.dropout_rate,
            name="lstm_layer_3",
        )(attention_norm)

        # Dense layers for sequence prediction
        dense_1 = TimeDistributed(
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            name="time_distributed_dense_1",
        )(lstm_3)

        dense_1_dropout = TimeDistributed(
            Dropout(config.model.dropout_rate), name="time_distributed_dropout_1"
        )(dense_1)

        dense_2 = TimeDistributed(
            Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
            name="time_distributed_dense_2",
        )(dense_1_dropout)

        # Output layer - predicting ABP values
        abp_output = TimeDistributed(
            Dense(1, activation="linear"), name="abp_prediction"
        )(dense_2)

        # Create model
        model = Model(inputs=inputs, outputs=abp_output, name="BloodPressurePredictor")

        return model

    def compile_model(self, model: Model) -> None:
        """Compile the model with appropriate optimizer and loss."""
        optimizer = Adam(
            learning_rate=config.model.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )

        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae", "mse", self._custom_bp_accuracy],
        )

        self.model = model

    @staticmethod
    def _custom_bp_accuracy(y_true, y_pred):
        """Custom accuracy metric for blood pressure prediction."""
        # Consider prediction accurate if within 5 mmHg
        threshold = 5.0 / 150.0  # Normalized threshold
        absolute_error = tf.abs(y_true - y_pred)
        accuracy = tf.reduce_mean(tf.cast(absolute_error <= threshold, tf.float32))
        return accuracy

    def get_callbacks(self, model_path: str) -> List:
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
        ]

        return callbacks

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_save_path: str = None,
    ) -> Dict:
        """Train the model."""
        if self.model is None:
            self.model = self.build_model(input_shape=X_train.shape[1:])
            self.compile_model(self.model)

        if model_save_path is None:
            model_save_path = str(
                config.data.models_directory / "advanced_bp_model.keras"
            )

        logger.info("Starting model training...")
        logger.info(f"Model architecture:\n{self.model.summary()}")

        callbacks = self.get_callbacks(model_save_path)

        # Train the model
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=config.model.batch_size,
            epochs=config.model.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Model training completed")

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        predictions = self.model.predict(X)
        return predictions

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        evaluation_metrics = self.model.evaluate(X_test, y_test, verbose=0)

        # Get predictions for additional metrics
        predictions = self.predict(X_test)

        # Calculate additional metrics
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

        # Blood pressure specific metrics (denormalized)
        mae_mmhg = mae * 150  # Assuming normalization range 0-1 for 0-150 mmHg
        rmse_mmhg = rmse * 150

        metrics_dict = {
            "loss": evaluation_metrics[0],
            "mae": evaluation_metrics[1],
            "mse": evaluation_metrics[2],
            "custom_bp_accuracy": evaluation_metrics[3],
            "mae_mmhg": mae_mmhg,
            "rmse_mmhg": rmse_mmhg,
        }

        return metrics_dict

    def load_model(self, model_path: str) -> None:
        """Load a saved model."""
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={"_custom_bp_accuracy": self._custom_bp_accuracy},
            )
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def save_model(self, model_path: str) -> None:
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")


# Global model instance
bp_model = AttentionBloodPressureModel()
