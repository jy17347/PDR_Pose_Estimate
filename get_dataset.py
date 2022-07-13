import pandas as pd
import numpy as np
import os


class IMU_dataset():
    
    def __init__(self, folder):
        
        data = []
        pose_list = os.listdir(folder)
        for pose in pose_list:
            data_dir = folder + '/' + pose + '/Accelerometer.csv'
            dataset = pd.read_csv(data_dir).to_numpy()
            data.append(dataset)
        
        self.data = data


    def __length__(self):

        return len(self.data)

    
    def __getitem__(self, index):

        data = self.data[index]

        return data


    def Sequence_data(self, frame_length):

        data = self.data[:]
        label_list = os.listdir('./dataset')
        seq_data = []
        label = []

        for pose in range(len(data)):
            for seq in range(np.shape(data[pose])[0] - frame_length):
                seq_data.append(data[pose][seq:seq+frame_length, 2:5])
                label.append(pose)

        return np.array(seq_data), np.array(label)
