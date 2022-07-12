import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Sequential
from tensorflow.keras.models import Model

def LSTM_model(num_degree, num_frame, num_pose):

    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=num_degree),
        LSTM(128, activation='relu', return_sequences=True),
        LSTM(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_pose, activation='softmax')
    ])

    return model
