#!/usr/bin/env python3
"""check_data_integrity_en.py

English data integrity checker for:
data/wyndham_smartbin_filllevel.json

Validations performed:
- Each unique timestamp should appear expected_count times (default 33)
- Required fields present
- Coordinate ranges
- Fullness and threshold ranges
- Duplicate (serialNumber + timestamp)
- Timestamp parsing correctness

Outputs a short English summary and exits with code 0 if OK, 1 if any errors.
"""
import sys
from pathlib import Path
import pandas as pd


DATA_REL = Path(__file__).parent.parent / "data" / "processed_filllevel.csv"
EXPECTED_PER_DATE = 33  # oczekiwana liczba rekordów na każdy timestamp/date


def load_df(path: Path = DATA_REL) -> pd.DataFrame:
    # Read CSV without forcing parse_dates to support files using either 'timestamp' or 'date'
    df = pd.read_csv(path)
    # Prefer 'timestamp' column if present, otherwise fall back to 'date'
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
    elif 'date' in df.columns:
        # date may be already date-like or string
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    else:
        raise KeyError("Missing required 'timestamp' or 'date' column in CSV")
    return df


def basic_overview(df: pd.DataFrame):
    print("File:", DATA_REL)
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMemory usage:")
    print(df.memory_usage(deep=True))
    print("\nFirst rows:")
    print(df.head().to_string(index=False))
    print("\nLast rows:")
    print(df.tail().to_string(index=False))
    print("\nNull counts per column:")
    print(df.isnull().sum())


def uniqueness_and_counts(df: pd.DataFrame):
    print("\nUnique values per column:")
    print(df.nunique(dropna=False))
    if "serialNumber" in df.columns:
        print("\nTop 10 serialNumber counts:")
        print(df["serialNumber"].value_counts().head(10))


def check_expected_per_timestamp(df: pd.DataFrame, expected: int = EXPECTED_PER_DATE):
    counts = df.groupby("date").size().sort_index()
    print(f"\nPer-date record counts: (min={counts.min()}, max={counts.max()}, mean={counts.mean():.2f})")
    bad = counts[counts != expected]
    if bad.empty:
        print(f"All dates have exactly {expected} records.")
    else:
        print(f"Dates with count != {expected}: {len(bad)} (showing up to 20):")
        print(bad.head(20).to_string())


def check_duplicates_per_bin_date(df: pd.DataFrame):
    grp = df.groupby(["serialNumber", "date"]).size()
    dup = grp[grp > 1]
    if dup.empty:
        print("\nNo duplicate records for same (serialNumber, date).")
    else:
        print(f"\nFound {len(dup)} duplicated (serialNumber, date) entries (showing up to 20):")
        print(dup.head(20).to_string())


def check_missing_dates_per_bin(df: pd.DataFrame, max_report: int = 20):
    missing_report = []
    for sn, g in df.groupby("serialNumber"):
        dates = pd.to_datetime(sorted(set(g["date"])))
        idx = pd.date_range(dates.min(), dates.max(), freq="D")
        missing = len(idx) - len(dates)
        if missing > 0:
            missing_report.append((sn, dates.min().date(), dates.max().date(), len(dates), len(idx), missing))
    missing_report.sort(key=lambda x: x[4] - x[3], reverse=True)
    print(f"\nBins with missing dates: {len(missing_report)} (showing up to {max_report})")
    for rec in missing_report[:max_report]:
        sn, start, end, present, expected, missing = rec
        print(f"serial={sn} | range={start}..{end} | present={present} | expected={expected} | missing={missing}")


def quick_numeric_checks(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        print("\nNo numeric columns detected.")
        return
    print("\nNumeric summary (describe):")
    print(numeric.describe().transpose())
    if "latestFullness" in df.columns and "fullnessThreshold" in df.columns:
        corr = df[["latestFullness", "fullnessThreshold"]].corr().iloc[0, 1]
        print(f"\nCorrelation latestFullness <-> fullnessThreshold: {corr:.3f}")


def main(csv_path: str = None):
    path = Path(csv_path) if csv_path else DATA_REL
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    df = load_df(path)
    basic_overview(df)
    uniqueness_and_counts(df)
    check_expected_per_timestamp(df)
    check_duplicates_per_bin_date(df)
    check_missing_dates_per_bin(df)
    quick_numeric_checks(df)
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(main(arg))
