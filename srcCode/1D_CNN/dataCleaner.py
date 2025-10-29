import numpy as np
import pandas as pd
import json


def load_json_data(filepath: str) -> pd.DataFrame:
    """
    Load and parse the JSON data from Wyndham City Council.
    The data is in GeoJSON format with lowercase field names.
    Args:
        filepath (str): Path to the JSON file.
    Returns:
        pd.DataFrame: Dataframe with relevant fields extracted.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    records = []

    print(f"   JSON type: {data.get('type', 'unknown')}")
    print(f"   JSON name: {data.get('name', 'unknown')}")

    # Parse GeoJSON features
    if "features" in data:
        for feature in data["features"]:
            if "properties" in feature:
                props = feature["properties"]
                # Extract coordinates if needed
                coords = feature.get("geometry", {}).get("coordinates", [None, None])

                record = {
                    "timestamp": props.get("timestamp"),
                    "latestFullness": props.get("latestFullness"),
                    "fullnessThreshold": props.get("fullnessThreshold"),
                    "ageThreshold": props.get("ageThreshold"),
                    "serialNumber": props.get("serialNumber"),
                    "reason": props.get("reason"),
                    "description": props.get("description"),
                    "position": props.get("position"),
                    "longitude": coords[0],
                    "latitude": coords[1],
                }
                records.append(record)

    df = pd.DataFrame(records)

    # Debug information
    print(f"   Loaded {len(df)} records")
    if len(df) > 0:
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Unique bins (serialNumber): {df['serialNumber'].nunique()}")
        print(f"   Sample record:")
        print(f"     Timestamp: {df.iloc[0]['timestamp']}")
        print(f"     LatestFullness: {df.iloc[0]['latestFullness']}")
        print(f"     SerialNumber: {df.iloc[0]['serialNumber']}")

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

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Handle missing values in fullness and timestamp
    initial_count = len(df)
    df = df.dropna(subset=["latestFullness", "timestamp"])
    print(f"   Dropped {initial_count - len(df)} rows with missing values")

    if len(df) == 0:
        raise ValueError("No valid data remains after removing missing values!")

    # Group by date and calculate mean fullness (aggregating multiple bins per day)
    df["date"] = df["timestamp"].dt.date
    daily_data = df.groupby("date")["latestFullness"].mean().reset_index()
    daily_data.columns = ["date", "fullness"]

    # Sort by date
    daily_data = daily_data.sort_values("date").reset_index(drop=True)

    print(f"   Date range: {daily_data['date'].min()} to {daily_data['date'].max()}")
    print(f"   Number of days: {len(daily_data)}")
    print(f"   Fullness statistics:")
    print(f"     Min: {daily_data['fullness'].min():.2f}")
    print(f"     Max: {daily_data['fullness'].max():.2f}")
    print(f"     Mean: {daily_data['fullness'].mean():.2f}")
    print(f"     Std: {daily_data['fullness'].std():.2f}")

    print(f"   Fullness value distribution:")
    print(f"     Values = 0: {(daily_data['fullness'] == 0).sum()}")
    print(f"     Values < 0.1: {(daily_data['fullness'] < 0.1).sum()}")
    print(f"     Values < 1.0: {(daily_data['fullness'] < 1.0).sum()}")

    # Add this diagnostic to see individual bin patterns
    print("\n=== INDIVIDUAL BIN ANALYSIS ===")
    for bin_id in df['serialNumber'].unique()[:5]:  # Check first 5 bins
        bin_data = df[df['serialNumber'] == bin_id]
        print(
            f"Bin {bin_id}: {len(bin_data)} records, fullness range: [{bin_data['latestFullness'].min():.1f}, {bin_data['latestFullness'].max():.1f}]")
    return daily_data


def create_individual_bin_sequences(df: pd.DataFrame, n_steps: int=30) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Create sequences for each bin individually and combine them
    Args:
        df (pd.DataFrame): Raw dataframe with columns including
                           'timestamp', 'latestFullness', 'serialNumber', etc.
        n_steps (int): Number of time steps in each input sequence.
    Returns:
        tuple: (X, y, bin_ids) where X is of shape (num_samples, n_steps, 1),
                y is of shape (num_samples,), and bin_ids is a list of bin identifiers.
    """
    all_X, all_y, all_bin_ids = [], [], []

    print("Processing individual bins...")

    for i, bin_id in enumerate(df['serialNumber'].unique()):
        # Get data for this specific bin
        bin_data = df[df['serialNumber'] == bin_id].copy()
        bin_data = bin_data.sort_values('timestamp')

        # Convert to daily data (in case of multiple readings per day)
        bin_data['date'] = pd.to_datetime(bin_data['timestamp']).dt.date
        daily_bin_data = bin_data.groupby('date')['latestFullness'].last().reset_index()
        daily_bin_data = daily_bin_data.sort_values('date')

        fullness_values = daily_bin_data['fullness'].values.reshape(-1, 1)

        if len(fullness_values) > n_steps + 10:  # Ensure enough data
            # Scale per bin
            scaler = MinMaxScaler()
            fullness_scaled = scaler.fit_transform(fullness_values)

            # Create sequences for this bin
            X_bin, y_bin = create_sequences(fullness_scaled.flatten(), n_steps)
            X_bin = X_bin.reshape(X_bin.shape[0], X_bin.shape[1], 1)

            all_X.append(X_bin)
            all_y.append(y_bin)
            all_bin_ids.extend([bin_id] * len(y_bin))

            if i < 5:  # Print first 5 bins for debugging
                print(
                    f"  Bin {bin_id}: {len(X_bin)} sequences, fullness range: [{fullness_values.min():.1f}, {fullness_values.max():.1f}]")

    # Combine all bins
    if all_X:
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)

        print(f"\nCombined dataset:")
        print(f"  Total sequences: {len(X_combined)}")
        print(f"  Unique bins: {len(df['serialNumber'].unique())}")
        print(f"  Sequence shape: {X_combined.shape}")

        return X_combined, y_combined, all_bin_ids
    else:
        raise ValueError("No valid sequences created!")

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
