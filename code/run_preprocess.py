from pathlib import Path
import json
import pandas as pd
from preprocess import preprocess_df


def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def main():
    base = Path(__file__).parent.parent
    data_path = base / 'data' / 'wyndham_smartbin_filllevel.json'
    out_csv = base / 'data' / 'processed_filllevel.csv'

    raw = load_json(data_path)
    df = pd.DataFrame([
        {
            'timestamp': feat['properties']['timestamp'],
            'serialNumber': feat['properties']['serialNumber'],
            'latestFullness': feat['properties'].get('latestFullness'),
            'reason': feat['properties'].get('reason'),
            'longitude': feat['geometry']['coordinates'][0],
            'latitude': feat['geometry']['coordinates'][1],
            'fullnessThreshold': feat['properties'].get('fullnessThreshold')
        }
        for feat in raw['features']
    ])

    processed = preprocess_df(df)
    processed.to_csv(out_csv, index=False)
    print(f"Saved processed data to: {out_csv}")
    print(processed.head())
    print(processed.shape)


if __name__ == '__main__':
    main()
