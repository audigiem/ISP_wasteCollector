"""
@file  : main.py
@brief : Main script to implement the 1D CNN baseline for waste bin fill level prediction,
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
from sklearn.model_selection import train_test_split
from dataCleaner import load_json_data, preprocess_data, create_sequences
from modelBuilding import build_1d_cnn_model
from trainEvaluate import calculate_metrics, plot_results


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def main(
    json_filepath: str,
) -> tuple[keras.Model, keras.callbacks.History, MinMaxScaler]:
    """
    Main execution function.
    Args:
        json_filepath (str): Path to the JSON data file.
    Returns:
        tuple: (trained model, training history, scaler used)
    """
    print("=" * 70)
    print("1D CNN BASELINE FOR WASTE PREDICTION - REPLICATING PAPER RESULTS")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading data...")
    df = load_json_data(json_filepath)
    print(f"   Raw data shape: {df.shape}")

    # 2. Preprocess data
    print("\n[2/6] Preprocessing data...")
    daily_data = preprocess_data(df)
    print(f"   Daily aggregated data shape: {daily_data.shape}")
    print(f"   Date range: {daily_data['date'].min()} to {daily_data['date'].max()}")

    # 3. Prepare sequences
    print("\n[3/6] Creating sequences...")

    # Extract fullness values
    fullness_values = daily_data["fullness"].values.reshape(-1, 1)

    print(f"   Fullness values shape before scaling: {fullness_values.shape}")
    print(f"   Number of samples: {len(fullness_values)}")

    if len(fullness_values) == 0:
        raise ValueError("No fullness data available! Check data preprocessing.")

    # Normalize data using MinMaxScaler (as mentioned in paper)
    scaler = MinMaxScaler()
    fullness_scaled = scaler.fit_transform(fullness_values)

    # Create sequences (using 30 time steps as a reasonable window)
    n_steps = 30

    if len(fullness_scaled) <= n_steps:
        print(
            f"   WARNING: Not enough data points ({len(fullness_scaled)}) for sequence length {n_steps}"
        )
        print(f"   Reducing sequence length to {len(fullness_scaled) // 2}")
        n_steps = max(5, len(fullness_scaled) // 2)

    X, y = create_sequences(fullness_scaled.flatten(), n_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Add feature dimension

    print(f"   Sequence shape: X={X.shape}, y={y.shape}")

    # 4. Split data (80% train, 20% test as per paper)
    # temporal split 80% first samples for training, rest for testing
    total_samples = len(X)
    train_samples = int(0.8 * total_samples)

    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_train = y[:train_samples]
    y_test = y[train_samples:]

    print(
        f"   Train samples: {len(X_train)} ({len(X_train) / (len(X_train) + len(X_test)) * 100:.1f}%)"
    )
    print(
        f"   Test samples: {len(X_test)} ({len(X_test) / (len(X_train) + len(X_test)) * 100:.1f}%)"
    )

    # 5. Build and compile model
    print("\n[4/6] Building 1D CNN model...")
    model = build_1d_cnn_model(n_steps)

    # Compile with Adam optimizer (as per paper)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    model.summary()

    # 6. Train model
    print("\n[5/6] Training model...")

    # Paper parameters: epochs=20, batch_size=70
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=70,
        validation_split=0.2,
        verbose=1,
        shuffle=False,  # Keep temporal order
    )

    # 7. Evaluate model
    print("\n[6/6] Evaluating model...")

    # Make predictions
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    y_test_pred = model.predict(X_test, verbose=0).flatten()

    # INVERSE TRANSFORM to get back to original scale
    y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_train_pred_original = scaler.inverse_transform(
        y_train_pred.reshape(-1, 1)
    ).flatten()
    y_test_pred_original = scaler.inverse_transform(
        y_test_pred.reshape(-1, 1)
    ).flatten()

    print(
        f"Y train range: [{y_train_original.min():.2f}, {y_train_original.max():.2f}]"
    )
    print(f"Y test range: [{y_test_original.min():.2f}, {y_test_original.max():.2f}]")
    print(
        f"Predictions range: [{y_train_pred_original.min():.2f}, {y_train_pred_original.max():.2f}]"
    )

    # Calculate metrics on ORIGINAL scale
    train_mae, train_mape, train_rmse, train_r2 = calculate_metrics(
        y_train_original, y_train_pred_original
    )
    test_mae, test_mape, test_rmse, test_r2 = calculate_metrics(
        y_test_original, y_test_pred_original
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON WITH PAPER")
    print("=" * 70)

    print("\nTraining Set Metrics:")
    print(f"  MAE:  {train_mae:.3f} (Paper: 0.667)")
    print(f"  MAPE: {train_mape:.3f} (Paper: 3.170)")
    print(f"  RMSE: {train_rmse:.3f} (Paper: 1.128)")
    print(f"  R²:   {train_r2:.3f} (Paper: 0.274)")

    print("\nTest Set Metrics:")
    print(f"  MAE:  {test_mae:.3f} (Paper: 0.677)")
    print(f"  MAPE: {test_mape:.3f} (Paper: 3.678)")
    print(f"  RMSE: {test_rmse:.3f} (Paper: 1.132)")
    print(f"  R²:   {test_r2:.3f} (Paper: 0.269)")

    print("\n" + "=" * 70)

    # Plot results
    print("\n[7/7] Generating plots...")
    plot_results(history, y_train, y_train_pred, y_test, y_test_pred)

    # Save model
    model.save("1d_cnn_waste_model.h5")
    print("\nModel saved as '1d_cnn_waste_model.h5'")

    return model, history, scaler


if __name__ == "__main__":
    # Replace with your actual JSON file path
    JSON_FILEPATH = "/home/matteo/Bureau/FIB/cours/ISP/ISP_Project/data/wyndham_smartbin_filllevel.json"

    try:
        model, history, scaler = main(JSON_FILEPATH)
        print("\n✓ Baseline 1D CNN implementation completed successfully!")
    except FileNotFoundError:
        print(f"\n✗ Error: File '{JSON_FILEPATH}' not found.")
        print("   Please update JSON_FILEPATH with the correct path to your data file.")
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
