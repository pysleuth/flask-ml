from werkzeug.wrappers import Request, Response
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, world!"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics

df = pd.read_csv("auto-mpg.csv", na_values=['NA', '?'])
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

X = df.drop(["mpg", "name"], axis=1).values
y = df["mpg"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/4, random_state=23
)

model = Sequential(
    [
        Dense(25, input_dim=X_train.shape[1], activation="relu"),
        Dense(10, activation="relu"),
        Dense(1)
    ]
)
model.compile(loss="mean_squared_error", optimizer="adam")

monitor = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-3,
    patience=5,
    verbose=1,
    mode="auto",
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[monitor],
    verbose=2,
    epochs=1_000
)

prediction = model.predict(X_test)
score = np.sqrt(metrics.mean_squared_error(prediction, y_test))
print(score)

model.save("mpg_model")


if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple("localhost", 9000, app)
