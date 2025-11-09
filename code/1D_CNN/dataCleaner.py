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


def convert_json_to_csv(input_filepath: str, output_filepath: str) -> None:
    """
    Convert the JSON data to CSV format.
    Args:
        input_filepath (str): Path to the input JSON file.
        output_filepath (str): Path to the output CSV file.
    """
    df = load_json_data(input_filepath)
    df.to_csv(output_filepath, index=False)
    print(f"   Data saved to CSV at: {output_filepath}")


def count_number_of_timestamps(df: pd.DataFrame) -> pd.Series:
    """
    Count the occurrences of each unique timestamp in the dataframe.
    Args:
        df (pd.DataFrame): Dataframe with a 'timestamp' column.
    Returns:
        pd.Series: Series with timestamps as index and their counts as values.
    """
    return df["timestamp"].value_counts()


def timestamp_count_per_serialnumber(df: pd.DataFrame) -> dict:
    """
    Count the number of unique timestamps for each serialNumber (bin).
    Args:
        df (pd.DataFrame): Dataframe with 'serialNumber' and 'timestamp' columns.
    Returns:
        dict: Dictionary mapping serialNumber to count of unique timestamps.
    """
    counts = {}
    for bin_id in df["serialNumber"].unique():
        bin_data = df[df["serialNumber"] == bin_id]
        unique_timestamps = bin_data["timestamp"].nunique()
        counts[bin_id] = unique_timestamps
    return counts


def inspect_overpresented_timestamps(
    df: pd.DataFrame, expected_count: int = 33
) -> dict:
    """
    Inspect timestamps that have more entries than expected.
    Args:
        df (pd.DataFrame): Dataframe with a 'timestamp' column.
        expected_count (int): Expected number of entries per timestamp.
    Returns:
        dict: key: timestamp, value: count of occurrences exceeding expected_count.
    """
    timestamp_counts = count_number_of_timestamps(df)
    exceeding_timestamps = timestamp_counts[timestamp_counts > expected_count]
    result = exceeding_timestamps.to_dict()
    return result


def remove_overpresented_timestamps(
    exceeding_timestamps: dict, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove entries from the original dataframe that correspond to overpresented timestamps.
    Args:
        exceeding_timestamps (dict): Dictionary of timestamps exceeding expected counts.
        df (pd.DataFrame): Original dataframe.
    Returns:
        pd.DataFrame: Cleaned dataframe with overpresented timestamps removed.
    """
    timestamps_to_remove = set(exceeding_timestamps.keys())
    cleaned_df = df[~df["timestamp"].isin(timestamps_to_remove)].reset_index(drop=True)
    print(
        f"   Removed {len(df) - len(cleaned_df)} entries with overpresented timestamps."
    )
    return cleaned_df


def remove_over_and_undersampled_serialnumbers(
    timestamp_counts: dict, df: pd.DataFrame, expected_count: int = 947
) -> pd.DataFrame:
    """
    Remove serialNumbers (bins) that have too many or too few timestamps.
    Args:
        timestamp_counts (dict): Dictionary mapping serialNumber to count of unique timestamps.
        df (pd.DataFrame): Original dataframe.
        expected_count (int): Expected number of unique timestamps per serialNumber.
    Returns:
        pd.DataFrame: Cleaned dataframe with over- and under-sampled serialNumbers removed.
    """
    serials_to_remove = [
        serial for serial, count in timestamp_counts.items() if count != expected_count
    ]
    cleaned_df = df[~df["serialNumber"].isin(serials_to_remove)].reset_index(drop=True)
    print(
        f"   Removed {len(serials_to_remove)} serialNumbers not matching expected count of {expected_count}."
    )
    print(f"   Cleaned dataframe shape: {cleaned_df.shape}")
    return cleaned_df


def export_df_as_json_and_csv(
    df: pd.DataFrame, json_filepath: str, csv_filepath: str
) -> None:
    """
    Export the dataframe as both JSON and CSV files.
    Args:
        df (pd.DataFrame): Dataframe to export.
        json_filepath (str): Path to save the JSON file.
        csv_filepath (str): Path to save the CSV file.
    """
    df.to_json(json_filepath, orient="records", lines=True)
    df.to_csv(csv_filepath, index=False)
    print(f"   Data exported to JSON at: {json_filepath}")
    print(f"   Data exported to CSV at: {csv_filepath}")


if __name__ == "__main__":
    # Example usage
    input_json = "data/wyndham_smartbin_filllevel.json"
    output_csv = "data/wyndham_waste_data.csv"

    # convert_json_to_csv(input_json, output_csv)
    df = load_json_data(input_json)
    dict = timestamp_count_per_serialnumber(df)

    print(f"\nTimestamp counts per serial number:\n{dict}")

    exceeding_timestamps = inspect_overpresented_timestamps(df, expected_count=33)
    print(f"\nOverpresented timestamps:\n{exceeding_timestamps}")
    cleaned_df = remove_overpresented_timestamps(exceeding_timestamps, df)
    cleaned_df.info()

    new_dict = timestamp_count_per_serialnumber(cleaned_df)
    print(f"\nTimestamp counts per serial number after cleaning:\n{new_dict}")

    final_df = remove_over_and_undersampled_serialnumbers(
        new_dict, cleaned_df, expected_count=947
    )
    final_df.info()

    final_dict = timestamp_count_per_serialnumber(final_df)
    print(f"\nFinal timestamp counts per serial number:\n{final_dict}")

    print("\nFinal dataframe preview:")
    # print(final_df.head())
    print(
        f"Final dataframe as {len(final_dict)} different serial numbers and {list(final_dict.values())[0]} timestamps each."
    )

    export_df_as_json_and_csv(
        final_df,
        "data/wyndham_waste_data_cleaned.json",
        "data/wyndham_waste_data_cleaned.csv",
    )

    # Also save to outputs directory for easy access
    import os

    output_dir = "../../outputs/"
    os.makedirs(output_dir, exist_ok=True)

    export_df_as_json_and_csv(
        final_df,
        os.path.join(output_dir, "wyndham_waste_data_cleaned.json"),
        os.path.join(output_dir, "wyndham_waste_data_cleaned.csv"),
    )
