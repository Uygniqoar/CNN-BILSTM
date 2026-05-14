import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

X_train = np.random.randn(2000, 5, 25)
y_train = np.random.randn(2000) * 0.05
X_val = np.random.randn(260, 5, 25)
y_val = np.random.randn(260) * 0.05

def test_model(lr, units, epochs, batch_size):
    model = Sequential([
        Input(shape=(5, 25)),
        LSTM(units),
        Dropout(0.9),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = model.predict(X_val, verbose=0).flatten()
    return preds

print("E=4, LR=0.15, B=2048:", np.mean(np.abs(test_model(0.15, 1, 4, 2048))))
