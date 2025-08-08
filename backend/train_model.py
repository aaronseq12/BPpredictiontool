from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Model
from data_preprocessing import load_and_preprocess_data

def train_lstm_model():
    """
    Trains the LSTM model for blood pressure prediction and saves it.
    """
    # Load and preprocess the data
    input_data, abp_data = load_and_preprocess_data()

    # Split data into training and validation sets
    X_train = input_data[:9600]
    y_train = abp_data[:9600]
    X_val = input_data[9600:10800]
    y_val = abp_data[9600:10800]

    # Define the LSTM model architecture
    inputs = Input(shape=(1000, 2))
    encoder = LSTM(128, return_sequences=True)(inputs)
    sequence_prediction = TimeDistributed(Dense(1, activation='linear'))(encoder)
    model = Model(inputs, sequence_prediction)

    # Compile and train the model
    model.compile('adam', 'mse')
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

    # Save the trained model
    model.save('lstm_model.h5')
    print("LSTM model trained and saved as lstm_model.h5")

if __name__ == '__main__':
    train_lstm_model()
