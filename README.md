# Blood Pressure Prediction Tool from PPG and ECG Signals

This project predicts Arterial Blood Pressure (ABP) from Photoplethysmography (PPG) and Electrocardiogram (ECG) signals using a Long Short-Term Memory (LSTM) model. It also includes a Generative Adversarial Network (GAN) to generate synthetic ABP data. The tool features a dynamic and interactive web interface for users to input PPG and ECG data and receive real-time blood pressure predictions.

## Project Structure

The project is organized into two main directories: `frontend` and `backend`.

- **`frontend`**: Contains the React application for the user interface.
- **`backend`**: Contains the Python scripts for data preprocessing, model training, and the Flask API.

### Backend Files

- **`main.py`**: The main file for the Flask API. It loads the trained LSTM model and provides an endpoint for blood pressure prediction.
- **`data_preprocessing.py`**: A script to load and preprocess the PPG, ECG, and ABP data from the dataset.
- **`train_model.py`**: A script to train the LSTM model using the preprocessed data and save the trained model.
- **`GAN.py`**: A script for the Generative Adversarial Network (GAN) to generate synthetic ABP data.
- **`requirements.txt`**: A file listing all the Python libraries required for the backend.

### Frontend Files

- **`App.js`**: The main component of the React application, containing the UI for data input, signal visualization, and blood pressure prediction.

## How to Run the Code

### Backend Setup

1.  **Navigate to the `backend` directory:**
    ```bash
    cd backend
    ```

2.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the LSTM model:**
    ```bash
    python train_model.py
    ```
    This will preprocess the data and train the LSTM model, saving it as `lstm_model.h5`.

4.  **Run the Flask API:**
    ```bash
    python main.py
    ```
    The API will start running on `http://127.0.0.1:5000`.

### Frontend Setup

1.  **Navigate to the `frontend` directory:**
    ```bash
    cd frontend
    ```

2.  **Install the required Node.js packages:**
    ```bash
    npm install
    ```

3.  **Start the React application:**
    ```bash
    npm start
    ```
    The application will open in your web browser at `http://localhost:3000`.

## Dataset

The dataset used for this project can be found at the following link:
[https://www.kaggle.com/mkachuee/BloodPressureDataset](https://www.kaggle.com/mkachuee/BloodPressureDataset)

Download the dataset and place the `.mat` files in the `backend/data` directory.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
