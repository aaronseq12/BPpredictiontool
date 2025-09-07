import numpy as np
import pandas as pd
import scipy.io as sio
import os

def load_and_preprocess_data():
    """
    Loads and preprocesses the PPG, ECG, and ABP data from the .mat files.

    Returns:
        tuple: A tuple containing the preprocessed input data and ABP data.
    """
    # Path to the dataset
    data_path = 'data/'
    
    # List to store data from .mat files
    signal_data = []
    for i in range(1, 13):
        mat_file_path = os.path.join(data_path, f'part_{i}.mat')
        signal_data.append(sio.loadmat(mat_file_path)['p'][0])

    # Concatenate all the data into a single array
    concatenated_data = np.concatenate(signal_data)

    # Split data into PPG, ABP, and ECG signals
    ppg_signal = [sample[0][:1000] for sample in concatenated_data]
    abp_signal = [sample[1][:1000] for sample in concatenated_data]
    ecg_signal = [sample[2][:1000] for sample in concatenated_data]

    # Convert to pandas DataFrame
    ppg_df = pd.DataFrame(ppg_signal)
    abp_df = pd.DataFrame(abp_signal)
    ecg_df = pd.DataFrame(ecg_signal)

    # Normalize ABP data
    abp_df = (abp_df - 50) / 150

    # Stack PPG and ECG data
    input_data = np.stack((ppg_df, ecg_df), axis=-1)

    return input_data, abp_df

if __name__ == '__main__':
    # Example of how to use the function
    input_signals, abp_targets = load_and_preprocess_data()
    print("Input data shape:", input_signals.shape)
    print("ABP data shape:", abp_targets.shape)
