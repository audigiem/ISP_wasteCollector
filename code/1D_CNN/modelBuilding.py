"""
@file modelBuilding.py
@brief Build a 1D CNN model for waste bin fill level prediction
       according to the architecture described in the paper:
       "A Machine Learning Approach to Predicting Waste Bin Fill Levels
        for Smart Waste Management Systems"
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_1d_cnn_model(n_steps: int, n_features: int = 1) -> keras.Model:
    """
    Build 1D CNN model according to the paper's architecture.

    Paper specifications:
    - 1D CNN layer with 200 filters
    - GlobalMaxPooling1D
    - Dense output layer
    - Total params: 6401 (trainable: 6401, non-trainable: 0)
    Args:
        n_steps (int): Number of time steps in the input sequences.
        n_features (int): Number of features per time step. Default is 1.
    Returns:
        keras.Model: Compiled 1D CNN model.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(n_steps, n_features)),
            # 1D CNN layer - the paper uses (None, 1, 200) as output
            # This suggests kernel_size creates (None, 1, 200) output
            layers.Conv1D(
                filters=200,
                kernel_size=30,
                activation="relu",
                padding="valid",
                name="conv1d",
            ),
            # Global Max Pooling
            layers.GlobalMaxPooling1D(name="global_max_pooling"),
            # Dense output layer
            layers.Dense(1, name="dense"),
        ]
    )

    return model
