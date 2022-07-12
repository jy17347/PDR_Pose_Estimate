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
            data.append((dataset, pose))

        self.data = data


    def __length__(self):
        return len(self.data)

    
    def __getitem__(self, index):

        data = self.data[index]

        return data
