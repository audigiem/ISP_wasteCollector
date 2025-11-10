"""
@file modelBuilding.py
@brief Build LSTM and BiLSTM models for waste bin fill level prediction
       according to the baseline architecture in Ahmed et al. (2022).
"""

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def build_lstm_model(n_steps: int, n_features: int = 1) -> keras.Model:
    """
    Baseline LSTM:
    - LSTM(100)
    - Dropout(0.2)
    - Dense(1)
    Paper training: Adam(lr=5e-4), loss=MSE, batch=70, epochs=20
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(n_steps, n_features)),
            layers.LSTM(100, name="lstm"),
            layers.Dropout(0.2, name="dropout"),
            layers.Dense(1, name="dense"),
        ]
    )
    # Compile with paper settings
    opt = keras.optimizers.legacy.Adam(learning_rate=5e-4)
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="MAE"),
                 tf.keras.metrics.RootMeanSquaredError(name="RMSE")]
    )
    return model


def build_bilstm_model(n_steps: int, n_features: int = 1) -> keras.Model:
    """
    Baseline BiLSTM:
    - Bidirectional(LSTM(100))
    - Dropout(0.2)
    - Dense(1)
    Paper training: Adam(lr=5e-4), loss=MSE, batch=70, epochs=20
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(n_steps, n_features)),
            layers.Bidirectional(layers.LSTM(100), name="bilstm"),
            layers.Dropout(0.2, name="dropout"),
            layers.Dense(1, name="dense"),
        ]
    )
    # Compile with paper settings
    opt = keras.optimizers.legacy.Adam(learning_rate=5e-4)
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="MAE"),
                 tf.keras.metrics.RootMeanSquaredError(name="RMSE")]
    )
    return model