from tensorflow.keras.models import load_model
import numpy as np

model = load_model("mpg_model")
x = np.zeros((1, 7))
model.predict(x)