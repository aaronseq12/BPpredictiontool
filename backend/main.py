from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained LSTM model
model = load_model('blood_pressure_model.keras')

@app.route('/predict', methods=['POST'])
def predict_blood_pressure():
    """
    Endpoint to receive PPG and ECG data and return the predicted blood pressure.
    """
    try:
        data = request.get_json()
        ppg_signal = np.array(data['ppg']).reshape(1, 1000, 1)
        ecg_signal = np.array(data['ecg']).reshape(1, 1000, 1)

        # Combine PPG and ECG data
        input_signals = np.concatenate((ppg_signal, ecg_signal), axis=-1)

        # Make a prediction
        prediction = model.predict(input_signals)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
