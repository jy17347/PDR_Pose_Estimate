import tensorflowjs as tfjs
from tensorflow.keras.models import load_model
import os


models = os.listdir('./models')
model = load_model(f"models/{models[-1]}/model.h5")
tfjs.converters.save_keras_model(model, f'models/{models[-1]}/' + '_model_tfjs_')