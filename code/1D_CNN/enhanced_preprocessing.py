"""
Improved data preparation for waste bin prediction following paper methodology more closely.
Addresses gaps in data and improves preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List


def improved_gap_filling(series: pd.Series, method: str = "cubic") -> pd.Series:
    """
    Improved gap filling using more sophisticated interpolation methods.

    Args:
        series: Time series with missing values
        method: Interpolation method ('linear', 'cubic', 'seasonal')

    Returns:
        Series with gaps filled
    """
    if method == "seasonal":
        # Use seasonal decomposition for more realistic gap filling
        from statsmodels.tsa.seasonal import seasonal_decompose

        # Need at least 2 full cycles (14 days) for weekly seasonality
        if len(series.dropna()) < 28:
            return (
                series.interpolate(method="cubic")
                .fillna(method="ffill")
                .fillna(method="bfill")
            )

        # Fill small gaps first with linear interpolation
        temp_series = series.interpolate(method="linear", limit=3)

        if temp_series.isna().sum() == 0:
            return temp_series

        # For remaining gaps, use seasonal decomposition if possible
        try:
            decomposition = seasonal_decompose(
                temp_series.dropna(), model="additive", period=7
            )
            # Use trend + seasonal components to fill gaps
            seasonal_filled = decomposition.trend + decomposition.seasonal
            return (
                seasonal_filled.reindex(series.index)
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
        except:
            # Fall back to cubic if seasonal decomposition fails
            return (
                series.interpolate(method="cubic")
                .fillna(method="ffill")
                .fillna(method="bfill")
            )

    else:
        return (
            series.interpolate(method=method)
            .fillna(method="ffill")
            .fillna(method="bfill")
        )


def create_enhanced_sequences(
    df: pd.DataFrame,
    n_steps: int = 30,
    min_days: int = 365,
    gap_fill_method: str = "cubic",
    add_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[int, MinMaxScaler]]:
    """
    Create sequences with enhanced preprocessing following paper methodology.

    Args:
        df: Input dataframe
        n_steps: Sequence length
        min_days: Minimum days required per bin
        gap_fill_method: Method for filling gaps ('linear', 'cubic', 'seasonal')
        add_features: Whether to add temporal features

    Returns:
        X, y, bin_ids, scalers
    """
    print(f"Enhanced sequence creation with {gap_fill_method} gap filling...")

    all_X, all_y, all_bin_ids = [], [], []
    scalers_by_bin = {}
    bin_stats = {}

    # Sort by bin and timestamp
    df = df.sort_values(["serialNumber", "timestamp"]).reset_index(drop=True)

    for i, bin_id in enumerate(sorted(df["serialNumber"].unique())):
        bin_data = df[df["serialNumber"] == bin_id].copy()

        # Create continuous daily index
        bin_min = bin_data["timestamp"].min().floor("D")
        bin_max = bin_data["timestamp"].max().floor("D")
        date_index = pd.date_range(bin_min, bin_max, freq="D")

        # Prepare daily values
        bin_data["date"] = bin_data["timestamp"].dt.floor("D")
        daily = bin_data.groupby("date")["latestFullness"].last().reindex(date_index)

        # Enhanced gap filling
        daily_filled = improved_gap_filling(daily, method=gap_fill_method)

        # Quality check
        if len(daily_filled) < min_days:
            continue

        # Store statistics for analysis
        original_na_count = daily.isna().sum()
        bin_stats[bin_id] = {
            "total_days": len(daily_filled),
            "missing_days": original_na_count,
            "missing_pct": (original_na_count / len(daily_filled)) * 100,
            "min_val": daily_filled.min(),
            "max_val": daily_filled.max(),
            "mean_val": daily_filled.mean(),
            "std_val": daily_filled.std(),
        }

        # Normalize per bin (0-1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = daily_filled.values.reshape(-1, 1)
        values_scaled = scaler.fit_transform(values).flatten()
        scalers_by_bin[bin_id] = scaler

        # Create base sequences
        X_bin = []
        y_bin = []

        for start in range(len(values_scaled) - n_steps):
            sequence = values_scaled[start : start + n_steps]
            target = values_scaled[start + n_steps]

            if add_features:
                # Add temporal features: day of week, month (normalized)
                dates = date_index[start : start + n_steps]
                dow_features = np.sin(2 * np.pi * dates.dayofweek / 7)  # Day of week
                month_features = np.sin(2 * np.pi * dates.month / 12)  # Month

                # Stack features: [fullness, day_of_week, month]
                features = np.column_stack([sequence, dow_features, month_features])
                X_bin.append(features)
            else:
                X_bin.append(sequence.reshape(-1, 1))

            y_bin.append(target)

        if len(X_bin) == 0:
            continue

        X_bin = np.array(X_bin)
        y_bin = np.array(y_bin)

        all_X.append(X_bin)
        all_y.append(y_bin)
        all_bin_ids.extend([bin_id] * len(y_bin))

        if i < 5:
            print(
                f"  Bin {bin_id}: {len(X_bin)} sequences, "
                f"{original_na_count} gaps filled ({original_na_count/len(daily_filled)*100:.1f}%)"
            )

    # Combine all bins
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)

    # Print summary statistics
    print(f"\nDataset Statistics:")
    print(f"  Total sequences: {len(X_combined)}")
    print(f"  Bins processed: {len(bin_stats)}")
    print(
        f"  Average missing days per bin: {np.mean([s['missing_days'] for s in bin_stats.values()]):.1f}"
    )
    print(
        f"  Average missing percentage: {np.mean([s['missing_pct'] for s in bin_stats.values()]):.1f}%"
    )

    return X_combined, y_combined, all_bin_ids, scalers_by_bin, bin_stats


def analyze_bin_patterns(bin_stats: Dict) -> None:
    """
    Analyze patterns in individual bin data for insights.
    """
    print(f"\nIndividual Bin Analysis:")
    print(
        f"{'Bin ID':<8} {'Days':<6} {'Missing':<8} {'Miss%':<6} {'Range':<12} {'Mean±Std':<12}"
    )
    print(f"{'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*12} {'-'*12}")

    for bin_id, stats in sorted(bin_stats.items()):
        range_str = f"{stats['min_val']:.1f}-{stats['max_val']:.1f}"
        mean_std_str = f"{stats['mean_val']:.1f}±{stats['std_val']:.1f}"
        print(
            f"{bin_id:<8} {stats['total_days']:<6} {stats['missing_days']:<8} "
            f"{stats['missing_pct']:<6.1f} {range_str:<12} {mean_std_str:<12}"
        )


if __name__ == "__main__":
    # Test the enhanced preprocessing
    csv_path = "../../data/wyndham_waste_data_cleaned.csv"

    try:
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        print("Testing enhanced data preprocessing...")

        # Test different gap filling methods
        for method in ["linear", "cubic", "seasonal"]:
            print(f"\n{'='*60}")
            print(f"Testing {method.upper()} gap filling:")

            X, y, bin_ids, scalers, stats = create_enhanced_sequences(
                df, n_steps=30, gap_fill_method=method, add_features=False
            )

            analyze_bin_patterns(stats)

            # Basic quality metrics
            print(f"\nQuality Metrics for {method}:")
            print(f"  Total sequences: {len(X)}")
            print(f"  Input shape: {X.shape}")
            print(f"  Target range: {y.min():.3f} to {y.max():.3f}")
            print(f"  Target mean±std: {y.mean():.3f}±{y.std():.3f}")

    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
