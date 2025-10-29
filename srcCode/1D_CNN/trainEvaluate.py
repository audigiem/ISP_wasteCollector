"""
@file trainEvaluate.py
@brief Train and evaluate a 1D CNN model for waste bin fill level prediction.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Calculate performance metrics as in the paper:
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - RMSE (Root Mean Squared Error)
    - R² (Coefficient of Determination)
    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
    Returns:
        tuple: (MAE, MAPE, RMSE, R²)
    """
    mae = np.mean(np.abs(y_true - y_pred))

    # MAPE (avoid division by zero)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)))

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))

    return mae, mape, rmse, r2


def plot_results(
    history: keras.callbacks.History,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
) -> None:
    """
    Plot training history and predictions vs actual values.
    Args:
        history (keras.callbacks.History): Training history object.
        y_train (np.ndarray): True training target values.
        y_train_pred (np.ndarray): Predicted training target values.
        y_test (np.ndarray): True test target values.
        y_test_pred (np.ndarray): Predicted test target values.
    Returns:
        None
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Training and validation loss
    axes[0, 0].plot(history.history["loss"], label="Train Loss")
    axes[0, 0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Model Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Training predictions
    axes[0, 1].plot(y_train, label="Actual", alpha=0.7)
    axes[0, 1].plot(y_train_pred, label="Predicted", alpha=0.7)
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Bin Fullness")
    axes[0, 1].set_title("Training Set: Actual vs Predicted")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Test predictions
    axes[1, 0].plot(y_test, label="Actual", alpha=0.7)
    axes[1, 0].plot(y_test_pred, label="Predicted", alpha=0.7)
    axes[1, 0].set_xlabel("Sample")
    axes[1, 0].set_ylabel("Bin Fullness")
    axes[1, 0].set_title("Test Set: Actual vs Predicted")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Scatter plot for test set
    axes[1, 1].scatter(y_test, y_test_pred, alpha=0.5)
    axes[1, 1].plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
    )
    axes[1, 1].set_xlabel("Actual")
    axes[1, 1].set_ylabel("Predicted")
    axes[1, 1].set_title("Test Set: Predicted vs Actual")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("1d_cnn_results.png", dpi=300, bbox_inches="tight")
    plt.show()
