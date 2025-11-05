import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import pickle

class AQIPredictor:
    """
    LSTM-based AQI prediction model for time series forecasting.
    """

    def __init__(self, sequence_length=24, epochs=50, batch_size=32):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_path = 'aqi_model.h5'
        self.scaler_path = 'scaler.pkl'

    def create_sequences(self, data):
        """
        Create sequences for LSTM training.
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length), 0])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def build_model(self):
        """
        Build LSTM model architecture.
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model

    def train(self, data):
        """
        Train the LSTM model on historical AQI data.
        """
        # Normalize the data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        # Create sequences
        X, y = self.create_sequences(scaled_data)

        # Reshape X for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build model if not exists
        if self.model is None:
            self.build_model()

        # Early stopping
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # Train the model
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        # Save model and scaler
        self.save_model()

        return history

    def predict(self, data, steps=24):
        """
        Make predictions for future AQI values.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                raise ValueError("Model not trained or loaded")

        # Normalize input data
        scaled_data = self.scaler.transform(data.reshape(-1, 1))

        # Create sequence for prediction
        last_sequence = scaled_data[-self.sequence_length:].reshape((1, self.sequence_length, 1))

        predictions = []

        for _ in range(steps):
            # Make prediction
            pred = self.model.predict(last_sequence, verbose=0)
            predictions.append(pred[0, 0])

            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred[0, 0]

        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)

        return predictions.flatten()

    def evaluate(self, actual, predicted):
        """
        Evaluate model performance.
        """
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)

        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        }

    def save_model(self):
        """
        Save trained model and scaler.
        """
        if self.model:
            self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_model(self):
        """
        Load trained model and scaler.
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

    def plot_predictions(self, actual, predicted, title="AQI Predictions"):
        """
        Plot actual vs predicted AQI values.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual AQI', color='blue')
        plt.plot(predicted, label='Predicted AQI', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('AQI Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt

def prepare_training_data(historical_data):
    """
    Prepare historical AQI data for training.
    """
    # Convert to numpy array
    data = np.array(historical_data).reshape(-1, 1)
    return data

def train_aqi_model(historical_data, sequence_length=24, epochs=50):
    """
    Train AQI prediction model with historical data.
    """
    predictor = AQIPredictor(sequence_length=sequence_length, epochs=epochs)

    # Prepare data
    data = prepare_training_data(historical_data)

    # Train model
    history = predictor.train(data)

    return predictor, history

def make_predictions(predictor, recent_data, prediction_steps=24):
    """
    Make AQI predictions using trained model.
    """
    data = prepare_training_data(recent_data)
    predictions = predictor.predict(data, steps=prediction_steps)
    return predictions
