import numpy as np
import pandas as pd
import scipy.io as sio
import os

def load_and_preprocess_data():
    """
    Loads and preprocesses the PPG, ECG, and ABP data from the .mat files.
    """
    # Path to the dataset
    data_path = 'data/'
    
    # List to store data from .mat files
    mats_list = []
    for i in range(1, 13):
        current_mat_str = os.path.join(data_path, f'part_{i}.mat')
        mats_list.append(sio.loadmat(current_mat_str)['p'][0])

    data = np.concatenate(mats_list)

    # Split data into PPG, ABP, and ECG
    ppg = [sample[0][:1000] for sample in data]
    abp = [sample[1][:1000] for sample in data]
    ecg = [sample[2][:1000] for sample in data]

    # Convert to pandas DataFrame
    ppg_df = pd.DataFrame(ppg)
    abp_df = pd.DataFrame(abp)
    ecg_df = pd.DataFrame(ecg)

    # Normalize ABP data
    abp_df = np.divide(np.subtract(abp_df, 50), 150)

    # Stack PPG and ECG data
    input_data = np.stack((ppg_df, ecg_df), axis=-1)

    return input_data, abp_df

if __name__ == '__main__':
    # Example of how to use the function
    input_data, abp_data = load_and_preprocess_data()
    print("Input data shape:", input_data.shape)
    print("ABP data shape:", abp_data.shape)
