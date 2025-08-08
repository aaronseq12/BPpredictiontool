from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained LSTM model
model = load_model('lstm_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive PPG and ECG data and return the predicted blood pressure.
    """
    data = request.get_json()
    ppg = np.array(data['ppg']).reshape(1, 1000, 1)
    ecg = np.array(data['ecg']).reshape(1, 1000, 1)

    # Combine PPG and ECG data
    input_data = np.concatenate((ppg, ecg), axis=-1)

    # Make a prediction
    prediction = model.predict(input_data)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
