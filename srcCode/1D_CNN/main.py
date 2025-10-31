"""
@file  : main.py
@brief : Main script to implement the 1D CNN baseline for INDIVIDUAL BIN waste bin fill level prediction,
            replicating the results from the paper:
            "A Machine Learning Approach to Predicting Waste Bin Fill Levels
                for Smart Waste Management Systems"

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from dataCleaner import (
    load_json_data,
    create_sequences,
    create_individual_bin_sequences,
)
from modelBuilding import build_1d_cnn_model
from trainEvaluate import calculate_metrics, plot_results


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def create_individual_bin_datasets(df, n_steps=30, train_ratio=0.8):
    """
    Create training and test datasets for individual bin prediction with proper temporal splitting per bin.

    Args:
        df (pd.DataFrame): Raw dataframe with bin data
        n_steps (int): Number of time steps in each sequence
        train_ratio (float): Ratio of data to use for training per bin

    Returns:
        tuple: (X_train, y_train, X_test, y_test, scalers_dict)
    """
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    scalers_dict = {}

    print("Processing individual bins with temporal splitting...")

    for i, bin_id in enumerate(df["serialNumber"].unique()):
        # Get data for this specific bin
        bin_data = df[df["serialNumber"] == bin_id].copy()
        bin_data = bin_data.sort_values("timestamp")

        # Convert to daily data (take last reading of each day)
        bin_data["date"] = pd.to_datetime(bin_data["timestamp"]).dt.date
        daily_bin_data = bin_data.groupby("date")["latestFullness"].last().reset_index()
        daily_bin_data = daily_bin_data.sort_values("date")

        fullness_values = daily_bin_data["latestFullness"].values.reshape(-1, 1)

        if len(fullness_values) > n_steps + 10:  # Ensure enough data
            # Scale per bin
            scaler = MinMaxScaler()
            fullness_scaled = scaler.fit_transform(fullness_values)
            scalers_dict[bin_id] = scaler

            # Create sequences for this bin
            X_bin, y_bin = create_sequences(fullness_scaled.flatten(), n_steps)
            X_bin = X_bin.reshape(X_bin.shape[0], X_bin.shape[1], 1)

            # Temporal split for this bin (80% train, 20% test)
            bin_train_samples = int(len(X_bin) * train_ratio)

            X_bin_train = X_bin[:bin_train_samples]
            y_bin_train = y_bin[:bin_train_samples]
            X_bin_test = X_bin[bin_train_samples:]
            y_bin_test = y_bin[bin_train_samples:]

            all_X_train.append(X_bin_train)
            all_y_train.append(y_bin_train)
            all_X_test.append(X_bin_test)
            all_y_test.append(y_bin_test)

            if i < 5:  # Print first 5 bins for debugging
                print(f"  Bin {bin_id}: {len(X_bin)} total sequences")
                print(f"    Train: {len(X_bin_train)}, Test: {len(X_bin_test)}")
                print(
                    f"    Fullness range: [{fullness_values.min():.1f}, {fullness_values.max():.1f}]"
                )

    # Combine all bins
    if all_X_train:
        X_train = np.vstack(all_X_train)
        y_train = np.hstack(all_y_train)
        X_test = np.vstack(all_X_test)
        y_test = np.hstack(all_y_test)

        print(f"\nCombined dataset:")
        print(f"  Total train sequences: {len(X_train)}")
        print(f"  Total test sequences: {len(X_test)}")
        print(f"  Unique bins: {len(df['serialNumber'].unique())}")
        print(f"  Sequence shape: {X_train.shape[1:]}")

        return X_train, y_train, X_test, y_test, scalers_dict
    else:
        raise ValueError("No valid sequences created!")


def inverse_transform_predictions(y_true, y_pred, bin_ids, scalers_dict):
    """
    Inverse transform predictions back to original scale using the appropriate scaler for each bin.

    Args:
        y_true (np.ndarray): True values (normalized)
        y_pred (np.ndarray): Predicted values (normalized)
        bin_ids (list): List of bin IDs for each sample
        scalers_dict (dict): Dictionary mapping bin IDs to their scalers

    Returns:
        tuple: (y_true_original, y_pred_original)
    """
    y_true_original = []
    y_pred_original = []

    for i, bin_id in enumerate(bin_ids):
        scaler = scalers_dict[bin_id]
        y_true_original.append(scaler.inverse_transform([[y_true[i]]])[0, 0])
        y_pred_original.append(scaler.inverse_transform([[y_pred[i]]])[0, 0])

    return np.array(y_true_original), np.array(y_pred_original)


def main(json_filepath: str) -> tuple[keras.Model, keras.callbacks.History, dict]:
    """
    Main execution function for INDIVIDUAL BIN prediction.

    Args:
        json_filepath (str): Path to the JSON data file.

    Returns:
        tuple: (trained model, training history, scalers dictionary)
    """
    print("=" * 70)
    print("1D CNN BASELINE - INDIVIDUAL BIN PREDICTION")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading data...")
    df = load_json_data(json_filepath)
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
    X_train, y_train, X_test, y_test, scalers_dict = create_individual_bin_datasets(
        df, n_steps=n_steps, train_ratio=0.8
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
    # Paper parameters: epochs=20, batch_size=70
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=70,
        validation_split=0.2,
        verbose=1,
        shuffle=True,  # Can shuffle since we have proper temporal split per bin
    )

    # 6. Evaluate model
    print("\n[6/6] Evaluating model...")

    # Make predictions
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    y_test_pred = model.predict(X_test, verbose=0).flatten()

    # For evaluation, we need to track which predictions belong to which bins
    # Since we combined all bins, we'll evaluate on normalized scale for simplicity
    # Alternatively, we could create bin_id arrays during dataset creation

    print(f"   Predictions on normalized scale (0-1):")
    print(f"   Train pred range: [{y_train_pred.min():.3f}, {y_train_pred.max():.3f}]")
    print(f"   Test pred range: [{y_test_pred.min():.3f}, {y_test_pred.max():.3f}]")

    # Calculate metrics on NORMALIZED scale for now
    # (Proper inverse scaling would require tracking bin IDs for each sample)
    train_mae, train_mape, train_rmse, train_r2 = calculate_metrics(
        y_train, y_train_pred
    )
    test_mae, test_mape, test_rmse, test_r2 = calculate_metrics(y_test, y_test_pred)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON WITH PAPER")
    print("=" * 70)
    print("NOTE: Metrics calculated on normalized scale (0-1)")
    print("Paper results are on original scale (0-10)")

    print("\nTraining Set Metrics:")
    print(f"  MAE:  {train_mae:.3f} (Paper: 0.667)")
    print(f"  MAPE: {train_mape:.3f}% (Paper: 3.170%)")
    print(f"  RMSE: {train_rmse:.3f} (Paper: 1.128)")
    print(f"  R²:   {train_r2:.3f} (Paper: 0.274)")

    print("\nTest Set Metrics:")
    print(f"  MAE:  {test_mae:.3f} (Paper: 0.677)")
    print(f"  MAPE: {test_mape:.3f}% (Paper: 3.678%)")
    print(f"  RMSE: {test_rmse:.3f} (Paper: 1.132)")
    print(f"  R²:   {test_r2:.3f} (Paper: 0.269)")

    print("\n" + "=" * 70)

    # Plot results (on normalized scale for now)
    print("\n[7/7] Generating plots...")
    plot_results(history, y_train, y_train_pred, y_test, y_test_pred)

    # Save model
    model.save("1d_cnn_individual_bin_model.h5")
    print("\nModel saved as '1d_cnn_individual_bin_model.h5'")

    return model, history, scalers_dict


if __name__ == "__main__":
    # Replace with your actual JSON file path
    JSON_FILEPATH = "/home/matteo/Bureau/FIB/cours/ISP/ISP_Project/data/wyndham_smartbin_filllevel.json"

    try:
        model, history, scalers_dict = main(JSON_FILEPATH)
        print("\n✓ Individual bin 1D CNN implementation completed successfully!")
        print(f"✓ Trained on {len(scalers_dict)} individual bins")
    except FileNotFoundError:
        print(f"\n✗ Error: File '{JSON_FILEPATH}' not found.")
        print("   Please update JSON_FILEPATH with the correct path to your data file.")
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
