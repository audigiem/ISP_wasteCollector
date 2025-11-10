"""
@file  : main.py
@brief : Train and evaluate BOTH LSTM and BiLSTM baselines for individual-bin
         fill-level prediction, using the same dataset, preprocessing, and
         visualization pipeline as the 1D-CNN baseline.
"""

import os
import json
import pickle
import numpy as np
import collections
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

from dataLoader import load_cleaned_csv_data, create_individual_bin_sequences
from modelBuilding import build_lstm_model, build_bilstm_model
from trainEvaluate import calculate_metrics, plot_results


# -------------------- Reproducibility --------------------
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
        print("TF op determinism enabled (experimental).")
    except Exception:
        pass


set_seeds(42)


# -------------------- Dataset utilities --------------------
def create_individual_bin_datasets(
    df: pd.DataFrame,
    n_steps: int = 30,
    train_ratio: float = 0.8,
    min_days: int = 365,
):
    """Create temporal train/test splits per bin."""
    print(f"\nCreating individual bin datasets (temporal split) ...")
    print(f"  Sequence length: {n_steps} | Train/Test: {train_ratio:.0%}/{(1-train_ratio):.0%}")

    X_all, y_all, bin_ids_all, scalers_by_bin = create_individual_bin_sequences(
        df, n_steps=n_steps, min_days=min_days
    )

    idxs_by_bin = collections.defaultdict(list)
    for idx, bin_id in enumerate(bin_ids_all):
        idxs_by_bin[bin_id].append(idx)

    X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
    train_bin_ids, test_bin_ids = [], []

    for bin_id, idxs in idxs_by_bin.items():
        indices = np.array(idxs)
        total = len(indices)
        split_idx = int(total * train_ratio)
        if split_idx < 1 or total - split_idx < 1:
            continue

        X_tr_list.append(X_all[indices[:split_idx]])
        y_tr_list.append(y_all[indices[:split_idx]])
        X_te_list.append(X_all[indices[split_idx:]])
        y_te_list.append(y_all[indices[split_idx:]])

        train_bin_ids.extend([bin_id] * split_idx)
        test_bin_ids.extend([bin_id] * (total - split_idx))

    X_train = np.vstack(X_tr_list)
    y_train = np.hstack(y_tr_list)
    X_test = np.vstack(X_te_list)
    y_test = np.hstack(y_te_list)

    print(f"\nFinal dataset: train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test, train_bin_ids, test_bin_ids, scalers_by_bin


def inverse_transform_array(
    y_norm_array: np.ndarray, bin_id_list: list, scalers_by_bin: dict
) -> np.ndarray:
    """Inverse-transform normalized y using per-bin MinMaxScaler."""
    y_orig = []
    for val, bin_id in zip(y_norm_array, bin_id_list):
        scaler: MinMaxScaler = scalers_by_bin[bin_id]
        y_orig.append(scaler.inverse_transform([[val]])[0, 0])
    return np.array(y_orig)


# -------------------- Model training wrapper --------------------
def train_model(
    model,
    model_name: str,
    X_train, y_train, X_test, y_test,
    train_bin_ids, test_bin_ids, scalers_by_bin,
    epochs=20, batch_size=70, n_steps=30,
):
    """Train, evaluate, plot, and save one model (LSTM or BiLSTM)."""
    print("\n" + "=" * 70)
    print(f"TRAINING {model_name.upper()} BASELINE")
    print("=" * 70)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
        shuffle=True,
    )

    # Predictions
    y_train_pred_norm = model.predict(X_train, verbose=0).flatten()
    y_test_pred_norm = model.predict(X_test, verbose=0).flatten()

    # Inverse-transform
    y_train_orig = inverse_transform_array(y_train, train_bin_ids, scalers_by_bin)
    y_train_pred_orig = inverse_transform_array(y_train_pred_norm, train_bin_ids, scalers_by_bin)
    y_test_orig = inverse_transform_array(y_test, test_bin_ids, scalers_by_bin)
    y_test_pred_orig = inverse_transform_array(y_test_pred_norm, test_bin_ids, scalers_by_bin)

    # Metrics (original scale)
    train_mae, train_mape, train_rmse, train_r2 = calculate_metrics(y_train_orig, y_train_pred_orig)
    test_mae, test_mape, test_rmse, test_r2 = calculate_metrics(y_test_orig, y_test_pred_orig)

    # Print
    print(f"\n{model_name.upper()} Results:")
    print(f"Train → MAE={train_mae:.3f}, RMSE={train_rmse:.3f}, R²={train_r2:.3f}")
    print(f"Test  → MAE={test_mae:.3f}, RMSE={test_rmse:.3f}, R²={test_r2:.3f}")

    # Plot (normalized)
    plot_results(
        history,
        y_train,
        y_train_pred_norm,
        y_test,
        y_test_pred_norm,
        model_name=model_name,
        save_name=f"{model_name.lower()}_training_results.png",
    )

    # Save model/scalers/results
    models_dir = "../../outputs/models/"
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, f"{model_name.lower()}_model.keras"))
    with open(os.path.join(models_dir, f"{model_name.lower()}_scalers.pkl"), "wb") as f:
        pickle.dump(scalers_by_bin, f)

    return {
        "model": model_name,
        "train_metrics": {"MAE": train_mae, "MAPE": train_mape, "RMSE": train_rmse, "R2": train_r2},
        "test_metrics": {"MAE": test_mae, "MAPE": test_mape, "RMSE": test_rmse, "R2": test_r2},
    }


# -------------------- MAIN --------------------
def main(csv_path: str, n_steps=30, epochs=20, batch_size=70):
    """
    Train and evaluate both LSTM and BiLSTM on the same dataset.
    Saves models, plots, and a combined comparison JSON.
    """
    print("=" * 70)
    print("LSTM + BiLSTM BASELINES — INDIVIDUAL BIN PREDICTION")
    print("=" * 70)

    # 1. Load
    df = load_cleaned_csv_data(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["latestFullness", "timestamp"])
    df = df.sort_values(["serialNumber", "timestamp"]).reset_index(drop=True)
    print(f"Loaded {len(df)} records from {df['serialNumber'].nunique()} bins")

    # 2. Prepare data
    X_train, y_train, X_test, y_test, train_bin_ids, test_bin_ids, scalers_by_bin = (
        create_individual_bin_datasets(df, n_steps=n_steps, train_ratio=0.8)
    )

    # 3. Train both models
    lstm = build_lstm_model(n_steps=n_steps, n_features=X_train.shape[-1])
    bilstm = build_bilstm_model(n_steps=n_steps, n_features=X_train.shape[-1])

    results = {}
    results["LSTM"] = train_model(
        lstm, "LSTM",
        X_train, y_train, X_test, y_test,
        train_bin_ids, test_bin_ids, scalers_by_bin,
        epochs, batch_size, n_steps
    )
    results["BiLSTM"] = train_model(
        bilstm, "BiLSTM",
        X_train, y_train, X_test, y_test,
        train_bin_ids, test_bin_ids, scalers_by_bin,
        epochs, batch_size, n_steps
    )

    # 4. Print summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON (TEST SET)")
    print("=" * 70)
    for name, res in results.items():
        t = res["test_metrics"]
        print(f"{name:8s} | RMSE={t['RMSE']:.3f} | MAE={t['MAE']:.3f} | R²={t['R2']:.3f} | MAPE={t['MAPE']*100:.2f}%")

    # 5. Save combined JSON
    out_dir = "../../outputs/"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "lstm_bilstm_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCombined results saved to '{out_dir}/lstm_bilstm_comparison.json'")

    print("\nAll models trained and evaluated successfully!")


if __name__ == "__main__":
    CSV_PATH = "../../data/wyndham_waste_data_cleaned.csv"
    try:
        main(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: file not found at {CSV_PATH}")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback; traceback.print_exc()