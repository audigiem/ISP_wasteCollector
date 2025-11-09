"""
Data analysis script to understand the dataset structure and verify alignment with paper methodology.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def analyze_dataset(csv_path: str):
    """
    Analyze the cleaned dataset to understand its structure and alignment with the paper.
    """
    print("=" * 70)
    print("DATASET ANALYSIS FOR 1D CNN BASELINE")
    print("=" * 70)

    # Load data
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"\n1. BASIC STATISTICS:")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique bins: {df['serialNumber'].nunique()}")
    print(
        f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
    )
    print(
        f"   Total days span: {(df['timestamp'].max() - df['timestamp'].min()).days + 1}"
    )

    # Check data distribution per bin
    counts = df["serialNumber"].value_counts()
    print(f"\n2. DATA DISTRIBUTION PER BIN:")
    print(f"   Min samples per bin: {counts.min()}")
    print(f"   Max samples per bin: {counts.max()}")
    print(f"   Mean samples per bin: {counts.mean():.1f}")
    print(f"   All bins have same count: {len(set(counts)) == 1}")

    # Analyze temporal gaps for a sample bin
    sample_bin_id = df["serialNumber"].iloc[0]
    sample_data = df[df["serialNumber"] == sample_bin_id].sort_values("timestamp")

    print(f"\n3. TEMPORAL STRUCTURE (Sample bin {sample_bin_id}):")
    gaps = sample_data["timestamp"].diff().dropna()
    gap_counts = gaps.value_counts().sort_index()

    print(f"   Records: {len(sample_data)}")
    print(
        f"   Date range: {sample_data['timestamp'].min().date()} to {sample_data['timestamp'].max().date()}"
    )
    print(f"   Gap analysis:")
    for gap, count in gap_counts.head(5).items():
        print(f"     {gap}: {count} occurrences")

    # Calculate missing days
    expected_days = (
        sample_data["timestamp"].max() - sample_data["timestamp"].min()
    ).days + 1
    actual_days = len(sample_data)
    missing_days = expected_days - actual_days
    print(
        f"   Missing days: {missing_days} out of {expected_days} ({missing_days/expected_days*100:.1f}%)"
    )

    # Analyze fullness values
    print(f"\n4. FULLNESS VALUE ANALYSIS:")
    print(f"   Min fullness: {df['latestFullness'].min()}")
    print(f"   Max fullness: {df['latestFullness'].max()}")
    print(f"   Mean fullness: {df['latestFullness'].mean():.2f}")
    print(f"   Std fullness: {df['latestFullness'].std():.2f}")

    # Check thresholds
    unique_thresholds = df["fullnessThreshold"].unique()
    print(f"   Unique thresholds: {sorted(unique_thresholds)}")

    # Paper comparison
    print(f"\n5. PAPER COMPARISON:")
    print(f"   Paper dataset: July 2018 to May 2021 (32 bins)")
    print(
        f"   Our dataset: {df['timestamp'].min().strftime('%B %Y')} to {df['timestamp'].max().strftime('%B %Y')} ({df['serialNumber'].nunique()} bins)"
    )

    # Check if we have data from the paper's timeframe
    paper_start = pd.to_datetime("2018-07-01")
    paper_end = pd.to_datetime("2021-05-31")
    our_start = df["timestamp"].min()
    our_end = df["timestamp"].max()

    overlap_start = max(paper_start, our_start)
    overlap_end = min(paper_end, our_end)

    if overlap_start <= overlap_end:
        overlap_days = (overlap_end - overlap_start).days + 1
        print(
            f"   Temporal overlap: {overlap_start.date()} to {overlap_end.date()} ({overlap_days} days)"
        )

        # Filter data to paper timeframe
        paper_data = df[
            (df["timestamp"] >= overlap_start) & (df["timestamp"] <= overlap_end)
        ]
        print(f"   Records in paper timeframe: {len(paper_data):,}")
        print(
            f"   Bins active in paper timeframe: {paper_data['serialNumber'].nunique()}"
        )
    else:
        print(f"   No temporal overlap with paper dataset!")

    return df


def plot_sample_bin_timeline(
    df: pd.DataFrame, bin_id: int = None, save_path: str = None
):
    """
    Plot timeline for a sample bin to visualize data gaps and fullness patterns.
    """
    if bin_id is None:
        bin_id = df["serialNumber"].iloc[0]

    bin_data = df[df["serialNumber"] == bin_id].copy().sort_values("timestamp")

    plt.figure(figsize=(15, 8))

    # Plot fullness over time
    plt.subplot(2, 1, 1)
    plt.plot(
        bin_data["timestamp"], bin_data["latestFullness"], "b-", alpha=0.7, linewidth=1
    )
    plt.title(f"Bin {bin_id}: Fullness Over Time")
    plt.ylabel("Fullness Level")
    plt.grid(True, alpha=0.3)

    # Plot data availability (gaps)
    plt.subplot(2, 1, 2)
    gaps = bin_data["timestamp"].diff()
    gap_days = gaps.dt.days
    gap_days = gap_days.fillna(0)

    plt.plot(bin_data["timestamp"].iloc[1:], gap_days.iloc[1:], "r-", alpha=0.7)
    plt.title(f"Bin {bin_id}: Data Gaps (Days Between Records)")
    plt.ylabel("Gap (Days)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    csv_path = "../../data/wyndham_waste_data_cleaned.csv"

    try:
        df = analyze_dataset(csv_path)

        print(f"\n6. GENERATING VISUALIZATION...")
        # Save to outputs directory
        import os

        output_dir = "../../outputs/"
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, "data_analysis_timeline.png")
        plot_sample_bin_timeline(df, save_path=save_path)
        print(f"   Saved timeline plot as '{save_path}'")

    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        print("Please run this from the 1D_CNN directory or update the path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
