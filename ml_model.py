import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import pickle

class AQIPredictor:
    """
    Enhanced LSTM-based AQI prediction model with bidirectional layers and attention mechanism.
    """

    def __init__(self, sequence_length=24, epochs=50, batch_size=32, use_attention=True):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_attention = use_attention
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_path = 'aqi_model.h5'
        self.scaler_path = 'scaler.pkl'
        self.feature_scaler_path = 'feature_scaler.pkl'

    def create_sequences(self, data, features=None):
        """
        Create sequences for LSTM training with optional additional features.
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            if features is not None:
                # Combine AQI with weather/pollutant features
                seq_features = []
                for j in range(self.sequence_length):
                    seq_data = [data[i + j, 0]]  # AQI
                    if i + j < len(features):
                        seq_data.extend(features[i + j])  # Weather/pollutants
                    else:
                        seq_data.extend([0] * len(features[0]))  # Pad with zeros
                    seq_features.append(seq_data)
                X.append(seq_features)
            else:
                X.append(data[i:(i + self.sequence_length), 0])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def build_model(self, n_features=1):
        """
        Build enhanced LSTM model architecture with bidirectional layers and attention.
        """
        if self.use_attention and n_features > 1:
            # Multi-feature model with attention
            inputs = Input(shape=(self.sequence_length, n_features))

            # Bidirectional LSTM layers
            lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
            lstm_out = Dropout(0.3)(lstm_out)
            lstm_out = Bidirectional(LSTM(64, return_sequences=True))(lstm_out)
            lstm_out = Dropout(0.3)(lstm_out)

            # Attention mechanism
            attention = Dense(1, activation='tanh')(lstm_out)
            attention = Dense(1, activation='softmax')(attention)
            context = attention * lstm_out
            context = np.sum(context, axis=1)

            # Dense layers
            dense_out = Dense(64, activation='relu')(context)
            dense_out = Dropout(0.3)(dense_out)
            dense_out = Dense(32, activation='relu')(dense_out)
            outputs = Dense(1)(dense_out)

            model = Model(inputs=inputs, outputs=outputs)
        else:
            # Standard LSTM for single feature
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True, input_shape=(self.sequence_length, n_features))),
                Dropout(0.3),
                Bidirectional(LSTM(64, return_sequences=False)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1)
            ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        self.model = model
        return model

    def train(self, data, features=None):
        """
        Train the LSTM model on historical AQI data with optional additional features.
        """
        # Normalize the data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        if features is not None:
            # Normalize features
            scaled_features = self.feature_scaler.fit_transform(features)

            # Create sequences with features
            X, y = self.create_sequences(scaled_data, scaled_features)
            n_features = X.shape[2]
        else:
            # Create sequences without features
            X, y = self.create_sequences(scaled_data)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            n_features = 1

        # Build model if not exists
        if self.model is None:
            self.build_model(n_features)

        # Early stopping
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # Train the model
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1,
            validation_split=0.2
        )

        # Save model and scalers
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
        Evaluate model performance with comprehensive metrics.
        """
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)

        # Additional metrics
        mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100  # Avoid division by zero
        r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

        # Directional accuracy (trend prediction)
        actual_trend = np.diff(actual) > 0
        predicted_trend = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_trend == predicted_trend) * 100

        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }

    def cross_validate(self, data, features=None, n_splits=5):
        """
        Perform time series cross-validation.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        # Prepare data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        if features is not None:
            scaled_features = self.feature_scaler.fit_transform(features)
            X, y = self.create_sequences(scaled_data, scaled_features)
        else:
            X, y = self.create_sequences(scaled_data)
            X = X.reshape((X.shape[0], X.shape[1], 1))

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Build and train model
            n_features_cv = X.shape[2] if len(X.shape) > 2 else 1
            self.build_model(n_features_cv)

            early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                          callbacks=[early_stop], verbose=0, validation_split=0.1)

            # Predict and evaluate
            predictions = self.model.predict(X_test, verbose=0)
            mse = mean_squared_error(y_test, predictions.flatten())
            mae = mean_absolute_error(y_test, predictions.flatten())

            scores.append({'MSE': mse, 'MAE': mae})

        # Calculate average scores
        avg_scores = {}
        for key in scores[0].keys():
            avg_scores[f'CV_{key}_mean'] = np.mean([score[key] for score in scores])
            avg_scores[f'CV_{key}_std'] = np.std([score[key] for score in scores])

        return avg_scores, scores

    def predict_with_confidence(self, data, features=None, steps=24, n_bootstraps=100):
        """
        Make predictions with confidence intervals using bootstrapping.
        """
        predictions_list = []

        # Generate bootstrap predictions
        for _ in range(n_bootstraps):
            # Bootstrap resampling
            indices = np.random.choice(len(data), len(data), replace=True)
            bootstrap_data = data[indices]

            if features is not None:
                bootstrap_features = features[indices]
                # Retrain model on bootstrap sample
                self.train(bootstrap_data, bootstrap_features)
            else:
                self.train(bootstrap_data)

            # Make prediction
            pred = self.predict(bootstrap_data[-self.sequence_length:], steps=steps)
            predictions_list.append(pred)

        # Calculate confidence intervals
        predictions_array = np.array(predictions_list)
        mean_predictions = np.mean(predictions_array, axis=0)
        lower_bound = np.percentile(predictions_array, 2.5, axis=0)
        upper_bound = np.percentile(predictions_array, 97.5, axis=0)

        return mean_predictions, lower_bound, upper_bound

    def save_model(self):
        """
        Save trained model and scalers.
        """
        if self.model:
            self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.feature_scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)

    def load_model(self):
        """
        Load trained model and scalers.
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        if os.path.exists(self.feature_scaler_path):
            with open(self.feature_scaler_path, 'rb') as f:
                self.feature_scaler = pickle.load(f)

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
    Handles both list of numbers and list of dictionaries.
    """
    if isinstance(historical_data[0], dict):
        # Extract AQI values from dictionary format
        aqi_values = [item['aqi'] for item in historical_data]
        data = np.array(aqi_values).reshape(-1, 1)
    else:
        # Handle legacy format (list of numbers)
        data = np.array(historical_data).reshape(-1, 1)
    return data

def prepare_features_data(historical_data):
    """
    Prepare additional features (weather, pollutants) for training.
    """
    if not isinstance(historical_data[0], dict):
        return None

    features_list = []
    for item in historical_data:
        features = []
        # Add pollutant levels
        pollutants = item.get('pollutants', {})
        for pollutant in ['pm25', 'pm10', 'o3', 'no2']:
            if pollutant in pollutants and isinstance(pollutants[pollutant], dict):
                features.append(pollutants[pollutant].get('v', 0))
            else:
                features.append(0)
        features_list.append(features)

    return np.array(features_list) if features_list else None

def prepare_features(historical_weather, historical_pollutants):
    """
    Prepare additional features (weather and pollutants) for training.
    """
    features = []
    for i in range(len(historical_weather)):
        weather = historical_weather[i]
        pollutants = historical_pollutants[i] if i < len(historical_pollutants) else {}

        # Extract relevant features
        feature_vector = [
            weather.get('temperature', 0),
            weather.get('humidity', 0),
            weather.get('wind_speed', 0),
            pollutants.get('pm25', {}).get('v', 0),
            pollutants.get('pm10', {}).get('v', 0),
            pollutants.get('o3', {}).get('v', 0),
            pollutants.get('no2', {}).get('v', 0)
        ]
        features.append(feature_vector)

    return np.array(features)

def train_aqi_model(historical_data, sequence_length=24, epochs=50, features=None):
    """
    Train AQI prediction model with historical data and optional features.
    """
    predictor = AQIPredictor(sequence_length=sequence_length, epochs=epochs)

    # Prepare data
    data = prepare_training_data(historical_data)

    # Train model
    history = predictor.train(data, features)

    return predictor, history

def make_predictions(predictor, recent_data, prediction_steps=24):
    """
    Make AQI predictions using trained model.
    """
    data = prepare_training_data(recent_data)
    predictions = predictor.predict(data, steps=prediction_steps)
    return predictions
