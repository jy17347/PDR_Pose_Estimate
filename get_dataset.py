import pandas as pd
import numpy as np

data_dir = 'Accelerometer.csv'
dataset = pd.read_csv(data_dir)

print(dataset)