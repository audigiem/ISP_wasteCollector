import pandas as pd
import numpy as np


def preprocess_df(df: pd.DataFrame, resample: str = 'D') -> pd.DataFrame:
    """Preprocess the raw DataFrame from JSON to a clean, per-bin daily timeseries.

    Steps performed:
    - ensure timestamp is datetime
    - keep relevant columns
    - fill missing numeric values per bin (median)
    - encode 'reason' as categorical codes and keep mapping
    - extract date features (year, month, day, weekday)
    - resample/aggregate to daily frequency per bin using mean for fullness
    """
    df = df.copy()
    
    # Ensure timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Keep relevant columns
    keep_cols = ['timestamp', 'serialNumber', 'latestFullness', 'reason', 'longitude', 'latitude', 'fullnessThreshold']
    for c in keep_cols:
        if c not in df.columns:
            raise ValueError(f"Missing expected column: {c}")
    df = df[keep_cols]

    # Fill missing numeric values per bin using median
    numeric_cols = ['latestFullness', 'fullnessThreshold']
    for col in numeric_cols:
        df[col] = df.groupby('serialNumber')[col].transform(lambda s: s.fillna(s.median()))
        # If still NA (all values NA for that bin), fill global median
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Encode reason as categorical codes
    df['reason'] = df['reason'].fillna('UNKNOWN')
    df['reason_code'] = df['reason'].astype('category').cat.codes

    # Extract date features
    df['date'] = df['timestamp'].dt.floor('D')
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday

    # Aggregate to daily per-bin time series
    agg_funcs = {
        'latestFullness': 'mean',
        'reason_code': 'max',  # keep strongest code observed that day
        'longitude': 'first',
        'latitude': 'first',
        'fullnessThreshold': 'first',
        'year': 'first',
        'month': 'first',
        'day': 'first',
        'weekday': 'first'
    }

    grouped = df.groupby(['serialNumber', 'date']).agg(agg_funcs).rename_axis(['serialNumber','date']).reset_index()

    # Ensure continuous daily index per bin (reindex)
    out_frames = []
    for serial, g in grouped.groupby('serialNumber'):
        g = g.set_index('date').sort_index()
        # create full daily index
        full_idx = pd.date_range(start=g.index.min(), end=g.index.max(), freq=resample)
        g = g.reindex(full_idx)
        g['serialNumber'] = serial
        # Forward/backward fill location and threshold
        g['longitude'] = g['longitude'].ffill().bfill()
        g['latitude'] = g['latitude'].ffill().bfill()
        g['fullnessThreshold'] = g['fullnessThreshold'].ffill().bfill()
        # Fill latestFullness with interpolation then fill remaining with 0
        g['latestFullness'] = g['latestFullness'].interpolate().fillna(0)
        # reason_code: fill with previous
        g['reason_code'] = g['reason_code'].ffill().fillna(-1).astype(int)
        # add date features back
        g['year'] = g.index.year
        g['month'] = g.index.month
        g['day'] = g.index.day
        g['weekday'] = g.index.weekday
        g = g.reset_index().rename(columns={'index':'date'})
        out_frames.append(g)

    processed = pd.concat(out_frames, ignore_index=True)

    # Reorder columns
    cols = ['serialNumber','date','latestFullness','fullnessThreshold','reason_code','longitude','latitude','year','month','day','weekday']
    processed = processed[cols]

    # Save reason mapping for user if needed
    # mapping = dict(enumerate(df['reason'].astype('category').cat.categories))

    return processed
