import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

def LSTM_model(num_degree, num_pose):

    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(20,6)),
        LSTM(128, activation='relu', return_sequences=True),
        LSTM(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_pose, activation='softmax')
    ])

    return model
