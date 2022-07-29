import pandas as pd
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split


class IMU_dataset():
    
    def __init__(self, folder):
        
        data = []
        pose_list = os.listdir('./dataset_new')
        for pose in pose_list:
            data_dir = folder + '/' + pose
            dataset = pd.read_csv(data_dir).to_numpy()
            data.append(dataset)
        
        # if metrics > 3:
        #     for pose in pose_list:
        #         sensor_csv = os.listdir(folder + '/' + pose)
        #         for sensor in sensor_csv:
        #             data_dir = folder + '/' + pose + '/' + sensor   
            
        #         dataset = pd.read_csv(data_dir).to_numpy()
        #         data.append(dataset)
        

        self.data = data


    def __length__(self):

        return len(self.data)

    
    def __getitem__(self, index):

        data = self.data[index]

        return data


    def Sequence_data(self, frame_length):

        data = np.array(self.data[:], dtype=object)
        label_list = os.listdir('./dataset_new')
        seq_data = []
        label = []

        scaler = MaxAbsScaler()

        for pose in range(len(data)):
            for seq in range(np.shape(data[pose])[0] - frame_length):
                seq_data.append(data[pose][seq:seq+frame_length, 2:8])
                label.append(pose)

        for i in range(len(seq_data)):
            scaler.fit(seq_data[i])
            seq_data[i] = scaler.transform(seq_data[i])

        return np.array(seq_data), np.array(label), label_list

