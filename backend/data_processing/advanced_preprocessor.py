"""
Advanced data preprocessing for blood pressure prediction.
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import logging
from utils.signal_processing import AdvancedSignalProcessor, validate_signal_quality
from config import config

logger = logging.getLogger(__name__)


class BloodPressureDataProcessor:
    """Advanced data processor for blood pressure prediction."""

    def __init__(self):
        self.signal_processor = AdvancedSignalProcessor()
        self.ppg_scaler = MinMaxScaler()
        self.ecg_scaler = MinMaxScaler()
        self.abp_scaler = MinMaxScaler()
        self.is_fitted = False

    def load_dataset_from_mat_files(
        self, data_directory: Path
    ) -> Tuple[List, List, List]:
        """Load PPG, ECG, and ABP data from .mat files."""
        logger.info(f"Loading dataset from {data_directory}")

        ppg_signals = []
        ecg_signals = []
        abp_signals = []

        # Look for .mat files in the directory
        mat_files = list(data_directory.glob("*.mat"))

        if not mat_files:
            # Try the pattern mentioned in original code
            for i in range(1, 13):
                mat_file_path = data_directory / f"part_{i}.mat"
                if mat_file_path.exists():
                    mat_files.append(mat_file_path)

        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {data_directory}")

        logger.info(f"Found {len(mat_files)} .mat files")

        for mat_file in mat_files:
            try:
                mat_data = sio.loadmat(str(mat_file))

                # Assuming structure similar to original code
                if "p" in mat_data:
                    records = mat_data["p"][0]

                    for record in records:
                        if len(record) >= 3:  # Ensure we have PPG, ABP, ECG
                            ppg_signal = record[0][: config.model.sequence_length]
                            abp_signal = record[1][: config.model.sequence_length]
                            ecg_signal = record[2][: config.model.sequence_length]

                            # Ensure all signals have the same length
                            min_length = min(
                                len(ppg_signal), len(abp_signal), len(ecg_signal)
                            )
                            if min_length >= config.model.sequence_length:
                                ppg_signals.append(
                                    ppg_signal[: config.model.sequence_length]
                                )
                                abp_signals.append(
                                    abp_signal[: config.model.sequence_length]
                                )
                                ecg_signals.append(
                                    ecg_signal[: config.model.sequence_length]
                                )

            except Exception as e:
                logger.warning(f"Error loading {mat_file}: {e}")
                continue

        logger.info(f"Loaded {len(ppg_signals)} signal triplets")
        return ppg_signals, abp_signals, ecg_signals

    def preprocess_signals(
        self, ppg_signals: List, ecg_signals: List, abp_signals: List
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess and validate signals."""
        logger.info("Preprocessing signals...")

        processed_ppg = []
        processed_ecg = []
        processed_abp = []
        quality_reports = []

        for i, (ppg, ecg, abp) in enumerate(zip(ppg_signals, ecg_signals, abp_signals)):
            # Convert to numpy arrays
            ppg_array = np.array(ppg, dtype=np.float32)
            ecg_array = np.array(ecg, dtype=np.float32)
            abp_array = np.array(abp, dtype=np.float32)

            # Preprocess signals
            processed_ppg_signal = self.signal_processor.preprocess_ppg_signal(
                ppg_array
            )
            processed_ecg_signal = self.signal_processor.preprocess_ecg_signal(
                ecg_array
            )

            # Validate signal quality
            is_valid, quality_report = validate_signal_quality(
                processed_ppg_signal,
                processed_ecg_signal,
                config.data.min_signal_quality_threshold,
            )

            if is_valid:
                processed_ppg.append(processed_ppg_signal)
                processed_ecg.append(processed_ecg_signal)
                processed_abp.append(abp_array)
                quality_reports.append(quality_report)
            else:
                logger.debug(f"Signal {i} rejected due to poor quality")

        logger.info(
            f"Kept {len(processed_ppg)} high-quality signals out of {len(ppg_signals)}"
        )

        # Convert to numpy arrays
        ppg_array = np.array(processed_ppg)
        ecg_array = np.array(processed_ecg)
        abp_array = np.array(processed_abp)

        # Normalize ABP signals (target)
        if config.data.normalization_method == "minmax":
            abp_normalized = self.abp_scaler.fit_transform(
                abp_array.reshape(-1, 1)
            ).reshape(abp_array.shape)
        else:  # zscore
            abp_normalized = (abp_array - np.mean(abp_array)) / np.std(abp_array)

        # Stack PPG and ECG as input features
        input_data = np.stack([ppg_array, ecg_array], axis=-1)

        self.is_fitted = True

        return input_data, abp_normalized

    def create_train_validation_split(
        self, input_data: np.ndarray, target_data: np.ndarray, test_size: float = None
    ) -> Tuple[np.ndarray, ...]:
        """Create train/validation split."""
        if test_size is None:
            test_size = config.model.validation_split

        return train_test_split(
            input_data,
            target_data,
            test_size=test_size,
            random_state=42,
            stratify=None,  # Can implement stratification based on BP ranges
        )

    def load_and_preprocess_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Complete data loading and preprocessing pipeline."""
        # Load raw data
        ppg_signals, abp_signals, ecg_signals = self.load_dataset_from_mat_files(
            config.data.data_directory
        )

        # Preprocess
        input_data, target_data = self.preprocess_signals(
            ppg_signals, ecg_signals, abp_signals
        )

        # Create train/validation split
        X_train, X_val, y_train, y_val = self.create_train_validation_split(
            input_data, target_data
        )

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")

        return X_train, X_val, y_train, y_val

    def preprocess_single_prediction(
        self, ppg_data: List, ecg_data: List
    ) -> np.ndarray:
        """Preprocess single prediction input."""
        if not self.is_fitted:
            raise ValueError(
                "Preprocessor not fitted. Load and preprocess training data first."
            )

        # Convert to numpy arrays
        ppg_array = np.array(ppg_data, dtype=np.float32)
        ecg_array = np.array(ecg_data, dtype=np.float32)

        # Ensure correct length
        if len(ppg_array) != config.model.sequence_length:
            raise ValueError(
                f"PPG signal must have length {config.model.sequence_length}"
            )
        if len(ecg_array) != config.model.sequence_length:
            raise ValueError(
                f"ECG signal must have length {config.model.sequence_length}"
            )

        # Preprocess
        processed_ppg = self.signal_processor.preprocess_ppg_signal(ppg_array)
        processed_ecg = self.signal_processor.preprocess_ecg_signal(ecg_array)

        # Validate quality
        is_valid, quality_report = validate_signal_quality(processed_ppg, processed_ecg)

        if not is_valid:
            logger.warning("Input signals have poor quality")

        # Stack and reshape for prediction
        input_data = np.stack([processed_ppg, processed_ecg], axis=-1)
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

        return input_data, quality_report


# Global preprocessor instance
preprocessor = BloodPressureDataProcessor()
