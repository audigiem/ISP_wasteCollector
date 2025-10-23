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
from pathlib import Path
import json
import sys
from typing import Dict, Any, Tuple
import pandas as pd


DEFAULT_FILENAME = "wyndham_smartbin_filllevel.json"
EXPECTED_COUNT_PER_TIMESTAMP = 33


def load_json(filepath: Path) -> Dict[str, Any]:
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def features_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    features = data.get("features", [])
    records = []
    for feat in features:
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        props = feat.get("properties", {}) or {}

        records.append({
            "longitude": coords[0] if len(coords) > 0 else None,
            "latitude": coords[1] if len(coords) > 1 else None,
            "latestFullness": props.get("latestFullness"),
            "reason": props.get("reason"),
            "serialNumber": props.get("serialNumber"),
            "description": props.get("description"),
            "position": props.get("position"),
            "ageThreshold": props.get("ageThreshold"),
            "fullnessThreshold": props.get("fullnessThreshold"),
            "timestamp_raw": props.get("timestamp"),
        })
    df = pd.DataFrame.from_records(records)
    # convert timestamps; invalid formats will become NaT
    df["timestamp"] = pd.to_datetime(df["timestamp_raw"], errors="coerce")
    return df


def check_timestamp_counts(df: pd.DataFrame, expected_count: int) -> Tuple[bool, pd.DataFrame]:
    counts = df.groupby("timestamp").size().reset_index(name="count")
    bad = counts[counts["count"] != expected_count]
    ok = bad.empty
    return ok, bad


def check_required_fields(df: pd.DataFrame, required_fields=None) -> Tuple[bool, pd.DataFrame]:
    if required_fields is None:
        required_fields = ["serialNumber", "latestFullness", "timestamp_raw", "longitude", "latitude"]
    missing_mask = df[required_fields].isnull().any(axis=1)
    bad = df[missing_mask]
    ok = bad.empty
    return ok, bad


def check_coordinate_ranges(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    cond_lon = df["longitude"].between(-180, 180)
    cond_lat = df["latitude"].between(-90, 90)
    bad = df[~(cond_lon & cond_lat)]
    ok = bad.empty
    return ok, bad


def check_fullness_and_thresholds(df: pd.DataFrame) -> Tuple[bool, Dict[str, pd.DataFrame]]:
    results = {}
    bad_fullness = df[~df["latestFullness"].between(0, 100)]
    results["fullness"] = bad_fullness

    bad_ft = df[~df["fullnessThreshold"].between(0, 100)]
    results["fullnessThreshold"] = bad_ft

    bad_age = df[df["ageThreshold"].notnull() & (df["ageThreshold"] < 0)]
    results["ageThreshold"] = bad_age

    ok = all(tbl.empty for tbl in results.values())
    return ok, results


def check_duplicates(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    subset = ["serialNumber", "timestamp_raw"]
    dup_mask = df.duplicated(subset=subset, keep=False)
    bad = df[dup_mask].sort_values(subset)
    ok = bad.empty
    return ok, bad


def run_validations(filepath: Path, expected_count: int = EXPECTED_COUNT_PER_TIMESTAMP) -> int:
    print(f"Loading file: {filepath}")
    try:
        data = load_json(filepath)
    except Exception as e:
        print(f"ERROR: Could not load file: {e}")
        return 1

    df = features_to_dataframe(data)
    total = len(df)
    print(f"Records: {total}")
    errors_found = False

    # 1) Timestamp parsing
    n_invalid_ts = df["timestamp"].isna().sum()
    if n_invalid_ts > 0:
        print(f"WARN: {n_invalid_ts} records have invalid timestamp format (could not parse).")
        errors_found = True
    else:
        print("OK: All timestamps parsed successfully.")

    # 2) Expected occurrences per timestamp
    ok_counts, bad_counts = check_timestamp_counts(df, expected_count)
    if ok_counts:
        print(f"OK: Each timestamp appears exactly {expected_count} times.")
    else:
        print("ERROR: Found timestamps with incorrect occurrence counts (expected "
              f"{expected_count}). Examples:")
        print(bad_counts.head(10).to_string(index=False))
        errors_found = True

    # 3) Required fields
    ok_required, missing_rows = check_required_fields(df)
    if ok_required:
        print("OK: No missing required fields (serialNumber, latestFullness, timestamp, longitude, latitude).")
    else:
        print(f"ERROR: {len(missing_rows)} records have missing required fields. Showing first 10:")
        print(missing_rows.head(10).to_string(index=False))
        errors_found = True

    # 4) Coordinate ranges
    ok_coords, bad_coords = check_coordinate_ranges(df)
    if ok_coords:
        print("OK: All coordinates are within acceptable ranges.")
    else:
        print(f"ERROR: {len(bad_coords)} records have invalid coordinates (lon/lat). Showing first 10:")
        print(bad_coords.head(10).to_string(index=False))
        errors_found = True

    # 5) Fullness and thresholds
    ok_vals, bad_vals = check_fullness_and_thresholds(df)
    if ok_vals:
        print("OK: latestFullness and thresholds are within expected ranges.")
    else:
        for key, tbl in bad_vals.items():
            if not tbl.empty:
                print(f"ERROR: {len(tbl)} records have invalid values in '{key}'. Showing first 10:")
                print(tbl.head(10).to_string(index=False))
                errors_found = True

    # 6) Duplicates
    ok_dups, bad_dups = check_duplicates(df)
    if ok_dups:
        print("OK: No duplicates found for (serialNumber + timestamp).")
    else:
        print(f"ERROR: Found {len(bad_dups)} duplicate rows for (serialNumber + timestamp). Showing first 20:")
        print(bad_dups.head(20).to_string(index=False))
        errors_found = True

    # Summary
    if errors_found:
        print("\nVALIDATION COMPLETE: issues detected. Please review messages above.")
        return 1
    else:
        print("\nVALIDATION COMPLETE: all checks passed successfully.")
        return 0


def main() -> None:
    data_path = Path(__file__).parent.parent / "data" / DEFAULT_FILENAME
    exit_code = run_validations(data_path, EXPECTED_COUNT_PER_TIMESTAMP)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
