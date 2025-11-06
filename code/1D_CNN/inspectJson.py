import json
import sys


def inspect_json_structure(filepath):
    """
    Inspect the structure of the Wyndham waste data JSON file.
    """
    print("=" * 70)
    print("JSON DATA STRUCTURE INSPECTOR")
    print("=" * 70)

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        print(f"\n✓ Successfully loaded: {filepath}")
        print(f"\nTop-level type: {type(data)}")

        if isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())}")

            # Explore each key
            for key in data.keys():
                print(f"\n--- Key: '{key}' ---")
                print(f"Type: {type(data[key])}")

                if isinstance(data[key], list):
                    print(f"Length: {len(data[key])}")
                    if len(data[key]) > 0:
                        print(f"First item type: {type(data[key][0])}")
                        if isinstance(data[key][0], dict):
                            print(f"First item keys: {list(data[key][0].keys())}")

                            # Show sample values
                            print("\nSample values from first item:")
                            for k, v in list(data[key][0].items())[:5]:
                                print(f"  {k}: {v} (type: {type(v).__name__})")

                            # If it's a 'features' list, explore properties
                            if key == "features" and "properties" in data[key][0]:
                                props = data[key][0]["properties"]
                                print(f"\nProperties keys: {list(props.keys())}")
                                print("\nSample property values:")
                                for k, v in list(props.items())[:10]:
                                    print(f"  {k}: {v} (type: {type(v).__name__})")

                elif isinstance(data[key], dict):
                    print(f"Dictionary keys: {list(data[key].keys())}")

        elif isinstance(data, list):
            print(f"\nList length: {len(data)}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"First item keys: {list(data[0].keys())}")
                    print("\nSample values from first item:")
                    for k, v in list(data[0].items())[:10]:
                        print(f"  {k}: {v} (type: {type(v).__name__})")

        # Try to identify the data structure
        print("\n" + "=" * 70)
        print("STRUCTURE IDENTIFICATION")
        print("=" * 70)

        if isinstance(data, dict) and "features" in data:
            print("\n✓ Detected: GeoJSON format with 'features' array")
            print(f"  Number of features: {len(data['features'])}")
            if len(data["features"]) > 0 and "properties" in data["features"][0]:
                props = data["features"][0]["properties"]
                # Check for required fields
                required_fields = [
                    "Timestamp",
                    "timestamp",
                    "LatestFullness",
                    "latestFullness",
                ]
                found_fields = [f for f in required_fields if f in props]
                print(f"  Found fields: {found_fields}")

        elif isinstance(data, list):
            print("\n✓ Detected: Direct array format")
            print(f"  Number of items: {len(data)}")
            if len(data) > 0:
                required_fields = [
                    "Timestamp",
                    "timestamp",
                    "LatestFullness",
                    "latestFullness",
                ]
                found_fields = [f for f in required_fields if f in data[0]]
                print(f"  Found fields: {found_fields}")

        print("\n" + "=" * 70)

    except FileNotFoundError:
        print(f"\n✗ Error: File '{filepath}' not found!")
    except json.JSONDecodeError as e:
        print(f"\n✗ Error: Invalid JSON format - {str(e)}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "/data/wyndham_smartbin_filllevel.json"

    inspect_json_structure(filepath)
