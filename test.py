import os
from tensorflow.keras.models import load_model


model_list = os.listdir('./models')
model_dir = 'models/'+model_list[-1]+'/model.h5'
model = load_model(model_dir)
model.evaluate(x_test, y_test)
print('complete')