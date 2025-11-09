"""
@file  : main.py
@brief : Main script to implement the 1D CNN baseline for INDIVIDUAL BIN waste bin fill level prediction,
            replicating the results from the paper:
            "A Machine Learning Approach to Predicting Waste Bin Fill Levels
                for Smart Waste Management Systems"

"""

import numpy as np
import collections
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from dataLoader import (
    load_cleaned_csv_data,
    create_sequences,
    create_individual_bin_sequences,
)
from sklearn.preprocessing import MinMaxScaler
from modelBuilding import build_1d_cnn_model
from trainEvaluate import calculate_metrics, plot_results


# reproducibility helpers
def set_seeds(seed=42):
    np.random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Try to enable deterministic ops in TF (best-effort)
    try:
        tf.config.experimental.enable_op_determinism()
        print("TF op determinism enabled (experimental).")
    except Exception as e:
        print("TF deterministic mode not available:", e)


# call it early in main (before model creation)
set_seeds(42)


def create_individual_bin_datasets(
    df, n_steps=30, train_ratio=0.8, min_days: int = 365
):
    """
    Create training and test datasets following the paper's individual bin prediction methodology.

    Paper approach:
    - Individual bin prediction: each bin's data treated separately
    - Temporal splitting: chronological split (not random)
    - The paper mentions splitting by years, we'll use temporal split per bin

    Args:
        df (pd.DataFrame): Raw dataframe with bin data
        n_steps (int): Number of time steps in each sequence (paper uses 30)
        train_ratio (float): Ratio of data to use for training per bin (0.8 = 80% train, 20% test)
        min_days (int): Minimum number of days of data required per bin

    Returns:
        tuple: (X_train, y_train, X_test, y_test, train_bin_ids, test_bin_ids, scalers_dict)
    """
    print(f"\nCreating individual bin datasets with temporal splitting...")
    print(f"  Sequence length: {n_steps} days")
    print(f"  Train/test split: {train_ratio:.0%}/{(1-train_ratio):.0%}")

    # Create sequences for each bin individually
    X_all, y_all, bin_ids_all, scalers_by_bin = create_individual_bin_sequences(
        df, n_steps=n_steps, min_days=min_days
    )

    # Group sequences by bin ID for temporal splitting
    idxs_by_bin = collections.defaultdict(list)
    for idx, bin_id in enumerate(bin_ids_all):
        idxs_by_bin[bin_id].append(idx)

    X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
    train_bin_ids, test_bin_ids = [], []

    print(f"\nSplitting data per bin (temporal split):")
    for bin_id, idxs in idxs_by_bin.items():
        indices = np.array(idxs)
        total = len(indices)

        # Temporal split: first part for training, last part for testing
        # This maintains the temporal order as sequences are created chronologically
        split_idx = int(total * train_ratio)

        if split_idx < 1 or total - split_idx < 1:
            print(f"  Bin {bin_id}: Skipped (too few sequences: {total})")
            continue

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        X_tr_list.append(X_all[train_indices])
        y_tr_list.append(y_all[train_indices])
        X_te_list.append(X_all[test_indices])
        y_te_list.append(y_all[test_indices])

        train_bin_ids.extend([bin_id] * len(train_indices))
        test_bin_ids.extend([bin_id] * len(test_indices))

        print(
            f"  Bin {bin_id}: {len(train_indices)} train, {len(test_indices)} test sequences"
        )

    if not X_tr_list:
        raise ValueError("No valid sequences after temporal splitting!")

    X_train = np.vstack(X_tr_list)
    y_train = np.hstack(y_tr_list)
    X_test = np.vstack(X_te_list)
    y_test = np.hstack(y_te_list)

    print(f"\nFinal dataset sizes:")
    print(
        f"  Training: {X_train.shape[0]} sequences from {len(set(train_bin_ids))} bins"
    )
    print(f"  Testing: {X_test.shape[0]} sequences from {len(set(test_bin_ids))} bins")

    return X_train, y_train, X_test, y_test, train_bin_ids, test_bin_ids, scalers_by_bin


def inverse_transform_array(
    y_norm_array: np.ndarray, bin_id_list: list, scalers_by_bin: dict
) -> np.ndarray:
    """
    Inverse transform a normalized array of y values using the appropriate scaler per bin.
    Args:
        y_norm_array (np.ndarray): Normalized y values.
        bin_id_list (list): List of bin IDs corresponding to each y value.
        scalers_by_bin (dict): Dictionary mapping bin IDs to their MinMaxScaler.
    Returns:
        np.ndarray: Inverse transformed y values.
    """
    y_orig = []
    for val, bin_id in zip(y_norm_array, bin_id_list):
        scaler = scalers_by_bin[bin_id]
        orig = scaler.inverse_transform([[val]])[0, 0]
        y_orig.append(orig)
    return np.array(y_orig)


def main(csv_path: str) -> tuple[keras.Model, keras.callbacks.History, dict]:
    """
    Main execution function for INDIVIDUAL BIN prediction.

    Args:
        csv_path (str): Path to the cleaned CSV data file.

    Returns:
        tuple: (trained model, training history, scalers dictionary)
    """
    print("=" * 70)
    print("1D CNN BASELINE - INDIVIDUAL BIN PREDICTION")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading data...")
    df = load_cleaned_csv_data(csv_path)
    print(f"   Raw data shape: {df.shape}")
    print(f"   Unique bins: {df['serialNumber'].nunique()}")

    # 2. Preprocess data
    print("\n[2/6] Preprocessing individual bin data...")
    # Sort and clean data
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["serialNumber", "timestamp"]).reset_index(drop=True)
    df = df.dropna(subset=["latestFullness", "timestamp"])

    # 3. Create individual bin sequences with temporal splitting
    print("\n[3/6] Creating individual bin sequences with temporal splitting...")
    n_steps = 30
    X_train, y_train, X_test, y_test, train_bin_ids, test_bin_ids, scalers_by_bin = (
        create_individual_bin_datasets(df, n_steps=n_steps, train_ratio=0.8)
    )

    print(f"   Final dataset sizes:")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 4. Build and compile model
    print("\n[4/6] Building 1D CNN model...")
    model = build_1d_cnn_model(n_steps)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    # 5. Train model
    print("\n[5/6] Training model...")
    # Paper parameters: epochs=20, batch_size=70, optimizer=adam lr=0.001
    # Using validation_data instead of validation_split to maintain temporal order
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=70,
        validation_data=(
            X_test,
            y_test,
        ),  # Use test set for validation to see generalization
        verbose=1,
        shuffle=True,  # Shuffle training data for better generalization
    )

    # 6. Evaluate model
    print("\n[6/6] Evaluating model...")

    # Make predictions
    y_train_pred_norm = model.predict(X_train, verbose=0).flatten()
    y_test_pred_norm = model.predict(X_test, verbose=0).flatten()

    # Build bin_id arrays aligned with y_train / y_test order:
    # train_bin_ids and test_bin_ids were returned from create_individual_bin_datasets
    # they must be in the same sequence order as y_train and y_test (they are by construction)
    y_train_orig = inverse_transform_array(y_train, train_bin_ids, scalers_by_bin)
    y_train_pred_orig = inverse_transform_array(
        y_train_pred_norm, train_bin_ids, scalers_by_bin
    )
    y_test_orig = inverse_transform_array(y_test, test_bin_ids, scalers_by_bin)
    y_test_pred_orig = inverse_transform_array(
        y_test_pred_norm, test_bin_ids, scalers_by_bin
    )
    print(
        f"Y train original scale: min {y_train_orig.min():.2f}, max {y_train_orig.max():.2f}"
    )
    print(
        f"Y test original scale: min {y_test_orig.min():.2f}, max {y_test_orig.max():.2f}"
    )
    # Calculate metrics on ORIGINAL scale (0-10)
    train_mae, train_mape, train_rmse, train_r2 = calculate_metrics(
        y_train_orig, y_train_pred_orig
    )
    test_mae, test_mape, test_rmse, test_r2 = calculate_metrics(
        y_test_orig, y_test_pred_orig
    )

    # Print results (original scale)
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON WITH PAPER (ORIGINAL SCALE)")
    print("=" * 70)

    print("\nTraining Set Metrics (original scale):")
    print(f"  MAE:  {train_mae:.3f} (Paper: 0.667)")
    print(f"  MAPE: {train_mape * 100:.3f}% (Paper: 3.170%)")
    print(f"  RMSE: {train_rmse:.3f} (Paper: 1.128)")
    print(f"  R²:   {train_r2:.3f} (Paper: 0.274)")

    print("\nTest Set Metrics (original scale):")
    print(f"  MAE:  {test_mae:.3f} (Paper: 0.677)")
    print(f"  MAPE: {test_mape * 100:.3f}% (Paper: 3.678%)")
    print(f"  RMSE: {test_rmse:.3f} (Paper: 1.132)")
    print(f"  R²:   {test_r2:.3f} (Paper: 0.269)")

    # Plot results (on normalized scale for now)
    print("\n[7/7] Generating plots...")
    plot_results(history, y_train, y_train_pred_norm, y_test, y_test_pred_norm)

    # Save model in outputs directory
    import os

    # Create outputs/models directory if it doesn't exist
    output_dir = "../../outputs/models/"
    os.makedirs(output_dir, exist_ok=True)

    # Save model in new .keras format
    model_path = os.path.join(output_dir, "1d_cnn_individual_bin_model.keras")
    model.save(model_path)
    print(f"\nModel saved as '{model_path}'")

    # Also save scalers for future use
    import pickle

    scalers_path = os.path.join(output_dir, "bin_scalers.pkl")
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers_by_bin, f)
    print(f"Bin scalers saved as '{scalers_path}'")

    # Save training results and metrics
    results_data = {
        "training_metrics": {
            "MAE": float(train_mae),
            "MAPE": float(train_mape),
            "RMSE": float(train_rmse),
            "R2": float(train_r2),
        },
        "test_metrics": {
            "MAE": float(test_mae),
            "MAPE": float(test_mape),
            "RMSE": float(test_rmse),
            "R2": float(test_r2),
        },
        "training_config": {
            "epochs": 20,
            "batch_size": 70,
            "sequence_length": 30,
            "total_sequences": len(X_train) + len(X_test),
            "train_sequences": len(X_train),
            "test_sequences": len(X_test),
            "num_bins": len(scalers_by_bin),
        },
        "paper_comparison": {
            "paper_train_mae": 0.667,
            "paper_train_mape": 3.17,
            "paper_train_rmse": 1.128,
            "paper_train_r2": 0.274,
            "paper_test_mae": 0.677,
            "paper_test_mape": 3.678,
            "paper_test_rmse": 1.132,
            "paper_test_r2": 0.269,
        },
    }

    import json

    results_path = os.path.join("../../outputs/", "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Training results saved as '{results_path}'")

    return model, history, scalers_by_bin


if __name__ == "__main__":
    # Replace with your actual CSV file path
    CSV_FILEPATH = "../../data/wyndham_waste_data_cleaned.csv"

    try:
        model, history, scalers_dict = main(CSV_FILEPATH)
        print("\n Individual bin 1D CNN implementation completed successfully!")
        print(f" Trained on {len(scalers_dict)} individual bins")
    except FileNotFoundError:
        print(f"\n Error: File '{CSV_FILEPATH}' not found.")
        print("   Please update CSV_FILEPATH with the correct path to your data file.")
    except Exception as e:
        print(f"\n Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
