"""
@file trainEvaluate.py
@brief Train/evaluate plotting & metric utilities (reused for LSTM/BiLSTM).
"""

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Calculate MAE, MAPE, RMSE, R^2 on ORIGINAL scale.
    - MAPE ignores values close to 0 to avoid division blow-ups.
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))

    nonzero = np.abs(y_true) > 1e-8
    if np.any(nonzero):
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])))
    else:
        mape = 0.0

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

    return mae, mape, rmse, r2


def plot_results(
    history: keras.callbacks.History,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    model_name: str = "model",
    save_name: str = "training_results.png",
) -> None:
    """
    Plot loss curves and normalized predictions vs. actuals (same style as 1D-CNN script).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Training and validation loss
    axes[0, 0].plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        axes[0, 0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title(f"{model_name} Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Training predictions (normalized)
    axes[0, 1].plot(y_train, label="Actual", alpha=0.7)
    axes[0, 1].plot(y_train_pred, label="Predicted", alpha=0.7)
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Bin Fullness (normalized)")
    axes[0, 1].set_title(f"{model_name} — Train: Actual vs Predicted")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Test predictions for first bin window (first 202 samples)
    axes[1, 0].plot(y_test[:202], label="Actual", alpha=0.7)
    axes[1, 0].plot(y_test_pred[:202], label="Predicted", alpha=0.7)
    axes[1, 0].set_xlabel("Sample")
    axes[1, 0].set_ylabel("Bin Fullness (normalized)")
    axes[1, 0].set_title(f"{model_name} — Test (First Bin Window): Actual vs Predicted")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Test predictions for second bin window (next 202 samples)
    axes[1, 1].plot(y_test[202:404], label="Actual", alpha=0.7)
    axes[1, 1].plot(y_test_pred[202:404], label="Predicted", alpha=0.7)
    axes[1, 1].set_xlabel("Sample")
    axes[1, 1].set_ylabel("Bin Fullness (normalized)")
    axes[1, 1].set_title(f"{model_name} — Test (Second Bin Window): Actual vs Predicted")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    import os
    output_dir = "../../outputs/"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, save_name)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Training results plot saved as '{plot_path}'")

    plt.show()