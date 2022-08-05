import tensorflowjs as tfjs
from tensorflow.keras.models import load_model
import os


models = os.listdir('./models')
model = load_model(f"models/{models[-2]}/model.h5")
tfjs.converters.save_keras_model(model, f'models_js/' + '_model_tfjs_')
