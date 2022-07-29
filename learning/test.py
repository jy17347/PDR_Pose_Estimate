import os
import time, os
import numpy as np
import pandas as pd
from model import LSTM_model
from get_dataset_new import IMU_dataset
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

frame_len = 20
model_list = os.listdir('./models')
model_dir = './models/'+model_list[-1]+'/model.h5'
model = load_model(model_dir)
dataset_dir = './test'
pose_list = []

data = pd.read_csv(dataset_dir+'/test.csv').to_numpy()
pose_list = os.listdir(dataset_dir)

seq_data = []

print(np.shape(data))


for seq in range(np.shape(data)[0] - frame_len):
    seq_data.append(data[seq:seq+frame_len, 2:8])

test_data = seq_data

for i in range(np.shape(seq_data)[0]):

    print(np.around(model.predict(np.array([test_data[i]])), 2))