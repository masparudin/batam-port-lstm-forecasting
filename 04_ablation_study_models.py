import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate

# Parameter simulation (aligned with experimental setup)
lookback_window = 12
lstm_units = 50
num_situational_features = 4  # (Eid_Actual, Eid_Lead, School, Xmas)

def build_proposed_multivariate_lstm():
    # ---------------------------------------------------------
    # BRANCH 1: Historical Data (Time Series Input)
    # Shape: (Batch_Size, 12 months, 1 feature/GT_Scaled)
    # ---------------------------------------------------------
    input_history = Input(shape=(lookback_window, 1), name='Historical_Input')
    
    # LSTM extracts temporal patterns from the past 12 months
    lstm_out = LSTM(lstm_units, activation='relu')(input_history) 
    # Output shape from LSTM: (Batch_Size, 50)
    
    # ---------------------------------------------------------
    # BRANCH 2: Situational Features (Calendar Variables)
    # Shape: (Batch_Size, 4 features) - Features EXCLUSIVELY for the target month (t)
    # ---------------------------------------------------------
    input_situational = Input(shape=(num_situational_features,), name='Situational_Input')
    
    # ---------------------------------------------------------
    # MERGE (Concatenation)
    # Merges historical temporal memory (50 neurons) with target month context (4 neurons)
    # ---------------------------------------------------------
    combined = Concatenate()([lstm_out, input_situational])
    # Output shape: (Batch_Size, 54)
    
    # Final Output Layer
    prediction = Dense(1, name='Forecast_Output')(combined)
    
    model = Model(inputs=[input_history, input_situational], outputs=prediction)
    model.compile(optimizer='adam', loss='mse')
    
    return model

# Architecture summary
model = build_proposed_multivariate_lstm()
model.summary()