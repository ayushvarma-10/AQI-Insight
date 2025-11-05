# TODO List for Enhancing AQI Dashboard

- [x] Update data_fetch.py: Load WAQI_API_KEY from .env and integrate real WAQI API for AQI data
- [x] Update app.py: Enhance UI with sidebar, background, icons, and improved layout
- [x] Run the app and test AQI accuracy for multiple cities
- [x] Verify UI improvements and overall functionality

## Deep Learning Integration Tasks

- [x] Add ML dependencies to requirements.txt (TensorFlow, scikit-learn, numpy)
- [x] Create ml_model.py with LSTM neural network for AQI forecasting
- [x] Extend data_fetch.py to fetch historical AQI data (last 30 days) for training
- [x] Add prediction visualization functions to analysis.py (predicted vs actual AQI trends)
- [x] Update app.py to display AQI predictions alongside current data
- [x] Install new dependencies and test environment setup
- [x] Fetch historical training data for model training
- [x] Train LSTM model on historical sequences
- [x] Test model predictions and accuracy
- [x] Run enhanced dashboard with ML predictions
