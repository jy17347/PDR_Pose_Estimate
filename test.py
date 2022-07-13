from tf.keras.models import load_model

model = load_model('models/model.h5')
model.evaluate(x_test, y_test)