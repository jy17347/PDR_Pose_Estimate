import pandas as pd
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split


class IMU_dataset():
    
    def __init__(self, folder):
        
        data = []
        pose_list = os.listdir('./22_dataset')
        for pose in pose_list:
            data_dir = folder + '/' + pose
            gravity = pd.read_csv(data_dir+'/Gravity.csv').to_numpy()
            gyroscope = pd.read_csv(data_dir+'/Gyroscope.csv').to_numpy()
            orientation = pd.read_csv(data_dir+'/Orientation.csv').to_numpy()
            # data = np.concatenate((gravity, gyroscope), axis = 1)
            pose_data = np.concatenate((gravity[:,2:5], orientation[:,6:9]), axis = 1)
            data.append(pose_data)
            # data.append(gravity[:,2:5:-1])
            # data.append(gyroscope[:,2:5:-1], axis = 1)
            # data.append(orientation[:,6:9:-1], axis = 1)
        
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
                seq_data.append(data[pose][seq:seq+frame_length, :])
                label.append(pose)

        for i in range(len(seq_data)):
            scaler.fit(seq_data[i])
            seq_data[i] = scaler.transform(seq_data[i])

        return np.array(seq_data), np.array(label), label_list

IMU = IMU_dataset('./22_dataset')
x_data, y_data, y_list = IMU.Sequence_data(20)

print(np.shape(x_data), np.shape(y_data), np.shape(y_list))
