import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict
from dataCleaner import timestamp_count_per_serialnumber


def load_cleaned_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load cleaned CSV data into a pandas DataFrame.
    Args:
        filepath (str): Path to the cleaned CSV file.
    Returns:
        pd.DataFrame: Dataframe with relevant fields extracted.
    """
    df = pd.read_csv(filepath)

    # Debug information
    print(f"   Loaded {len(df)} records from {filepath}")
    if len(df) > 0:
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Unique bins (serialNumber): {df['serialNumber'].nunique()}")
        print(f"   Sample record:")
        print(f"     Timestamp: {df.iloc[0]['timestamp']}")
        print(f"     LatestFullness: {df.iloc[0]['latestFullness']}")
        print(f"     SerialNumber: {df.iloc[0]['serialNumber']}")

    # check that every serialNumber has same number of timestamps
    counts = timestamp_count_per_serialnumber(df)
    unique_counts = set(counts.values())
    if len(unique_counts) > 1:
        print("   WARNING: Not all serialNumbers have the same number of timestamps!")
        for bin_id, count in counts.items():
            print(f"     Bin {bin_id}: {count} timestamps")
    else:
        print(
            f"   All serialNumbers have the same number of timestamps: {unique_counts.pop()}"
        )
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the waste data according to the paper's methodology.
    Paper mentions: "data from a total of thirty-two bins are stored every day"
    Dataset: July 2018 to May 2021
    Args:
        df (pd.DataFrame): Raw dataframe with columns including
                           'timestamp', 'latestFullness', 'serialNumber', etc.
    Returns:
        pd.DataFrame: Preprocessed dataframe with daily average fullness.
    """
    print(f"   Initial dataframe shape: {df.shape}")
    print(f"   Unique bins: {df['serialNumber'].nunique()}")

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Handle missing values in fullness and timestamp
    initial_count = len(df)
    df = df.dropna(subset=["latestFullness", "timestamp"])
    print(
        f"   Dropped {initial_count - len(df)} rows with missing latestFullness/timestamp"
    )
    return df


def create_individual_bin_sequences(
    df: pd.DataFrame, n_steps: int = 30, min_days: int = 365
) -> tuple[np.ndarray, np.ndarray, list, Dict[int, MinMaxScaler]]:
    """
    Create sequences for each bin individually following the paper's methodology.

    Paper approach:
    - Individual bin prediction (each bin treated separately)
    - Daily data (fill gaps with forward/backward fill)
    - Normalize per bin using MinMaxScaler (0-1)
    - Sliding window approach for sequences

    Args:
        df (pd.DataFrame): Raw dataframe with columns including
                           'timestamp', 'latestFullness', 'serialNumber', etc.
        n_steps (int): Number of time steps in each input sequence.
        min_days (int): Minimum number of days with data required per bin
    Returns:
        tuple: (X, y, bin_ids, scalers_bin_ids) where X is of shape (num_samples, n_steps, 1),
                y is of shape (num_samples,), bin_ids is a list of bin identifiers and
                scalers_bin_ids is a dict mapping bin_id to its MinMaxScaler.
    """
    all_X, all_y, all_bin_ids = [], [], []
    scalers_by_bin = {}
    print("Processing individual bins with gap filling for continuous daily data...")

    for i, bin_id in enumerate(sorted(df["serialNumber"].unique())):
        bin_data = df[df["serialNumber"] == bin_id].copy().sort_values("timestamp")
        if bin_data.empty:
            continue

        # Create continuous daily index for this bin
        bin_min = bin_data["timestamp"].min().floor("D")
        bin_max = bin_data["timestamp"].max().ceil("D")
        date_index = pd.date_range(bin_min, bin_max, freq="D")

        # Get daily values (take last reading per day if multiple)
        bin_data["date"] = bin_data["timestamp"].dt.floor("D")
        daily = (
            bin_data.groupby("date")["latestFullness"]
            .last()
            .reindex(date_index)
            .astype(float)
        )

        # Fill missing values - forward fill then backward fill
        daily = daily.ffill().bfill()

        # Check if we have enough data after filling
        days_available = len(daily)
        if days_available < min_days:
            if i < 5:
                print(
                    f"  Skipping bin {bin_id}: only {days_available} days (<{min_days})"
                )
            continue

        # Normalize per bin (0-1 scaling as in paper)
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = daily.values.reshape(-1, 1)
        values_scaled = scaler.fit_transform(values).flatten()
        scalers_by_bin[bin_id] = scaler

        # Create sliding window sequences
        X_bin = []
        y_bin = []
        for start in range(len(values_scaled) - n_steps):
            X_bin.append(values_scaled[start : start + n_steps])
            y_bin.append(values_scaled[start + n_steps])

        if len(X_bin) == 0:
            continue

        X_bin = np.array(X_bin).reshape(-1, n_steps, 1)
        y_bin = np.array(y_bin)

        all_X.append(X_bin)
        all_y.append(y_bin)
        all_bin_ids.extend([bin_id] * len(y_bin))

        if i < 5:
            print(f"  Bin {bin_id}: {len(X_bin)} sequences from {days_available} days")
            print(f"    Date range: {bin_min.date()} to {bin_max.date()}")
            print(f"    Fullness range: {daily.min():.1f} to {daily.max():.1f}")

    if not all_X:
        raise ValueError("No valid sequences created!")

    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)

    print(f"\nCombined dataset:")
    print(f"  Total sequences: {X_combined.shape[0]}")
    print(f"  Input shape: {X_combined.shape[1:]}")
    print(f"  Bins used: {len(set(all_bin_ids))}")

    return X_combined, y_combined, all_bin_ids, scalers_by_bin


def create_sequences(data: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    X: past n_steps values
    y: next value to predict
    Args:
        data (np.ndarray): 1D array of data points.
        n_steps (int): Number of time steps in each input sequence.
    Returns:
        tuple: (X, y) where X is of shape (num_samples, n_steps)
               and y is of shape (num_samples,)
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i : i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)
