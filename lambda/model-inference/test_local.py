"""
test_local.py — Test the inference pipeline locally without S3.

Creates mock data files in model-inference/data/ so you can verify
the full pipeline works before deploying to Lambda.

Usage:
    cd model-inference
    python test_local.py

Prerequisites:
    pip install -r requirements.txt
    
    Then place these REAL files in model-inference/data/:
      - avalanche_classifier_tuned.pkl  (from your training pipeline)
      - colorado_dem.tif                (from data/external/)
      - snotel_stations_const.csv       (from teammate's S3 or data/processed/)
      - daily_station_data.csv          (from teammate's S3 or run their script)

    OR: run with --mock to generate fake data for a quick smoke test.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Ensure we can import from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def create_mock_data():
    """Create minimal mock data files for smoke testing."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Mock snotel_stations_const.csv ────────────────────────────────────
    stations = pd.DataFrame([
        {"station_id": 335, "station_triplet": "335:CO:SNTL", "name": "Berthoud Summit",
         "latitude": 39.80, "longitude": -105.78},
        {"station_id": 412, "station_triplet": "412:CO:SNTL", "name": "Grizzly Peak",
         "latitude": 39.65, "longitude": -105.87},
        {"station_id": 505, "station_triplet": "505:CO:SNTL", "name": "Loveland Basin",
         "latitude": 39.68, "longitude": -105.90},
        {"station_id": 602, "station_triplet": "602:CO:SNTL", "name": "Niwot",
         "latitude": 40.03, "longitude": -105.55},
    ])
    stations_path = os.path.join(DATA_DIR, "snotel_stations_const.csv")
    stations.to_csv(stations_path, index=False)
    print(f"✓ Created {stations_path}")

    # ── Mock daily_station_data.csv ───────────────────────────────────────
    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    rows = []
    for date_str in [yesterday, today]:
        for _, s in stations.iterrows():
            rows.append({
                "date": date_str,
                "station_id": s["station_id"],
                "station_triplet": s["station_triplet"],
                "station_name": s["name"],
                "snow_depth": np.random.uniform(80, 200),
                "swe": np.random.uniform(20, 60),
                "temp": np.random.uniform(-15, 2),
                "new_snow_24hr": np.random.uniform(0, 30),
            })

    daily_path = os.path.join(DATA_DIR, "daily_station_data.csv")
    pd.DataFrame(rows).to_csv(daily_path, index=False)
    print(f"✓ Created {daily_path}")

    # ── Mock model ────────────────────────────────────────────────────────
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    feature_cols = ["elevation", "slope", "aspect_degrees", "snow_depth",
                    "new_snow_24h", "swe", "temp"]

    # Generate fake training data
    np.random.seed(42)
    n = 500
    X_train = pd.DataFrame({
        "elevation": np.random.uniform(2500, 4000, n),
        "slope": np.random.uniform(15, 50, n),
        "aspect_degrees": np.random.uniform(0, 360, n),
        "snow_depth": np.random.uniform(50, 250, n),
        "new_snow_24h": np.random.uniform(0, 50, n),
        "swe": np.random.uniform(10, 80, n),
        "temp": np.random.uniform(-20, 5, n),
    })
    y_train = (X_train["new_snow_24h"] > 20).astype(int) | (X_train["slope"] > 38).astype(int)

    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    model_package = {
        "model": model,
        "optimal_threshold": 0.42,
        "feature_cols": feature_cols,
    }

    model_path = os.path.join(DATA_DIR, "avalanche_classifier_tuned.pkl")
    joblib.dump(model_package, model_path)
    print(f"✓ Created {model_path}")

    print(f"\n✓ All mock data created in {DATA_DIR}/")
    print(f"  NOTE: DEM not mocked — terrain extraction will fail unless you")
    print(f"  copy colorado_dem.tif into {DATA_DIR}/")

    return True


def test_weather_only(lat, lon):
    """Test just the weather lookup (no DEM needed)."""
    from weather import get_weather_for_point

    print(f"\n{'='*50}")
    print(f"WEATHER TEST: ({lat}, {lon})")
    print(f"{'='*50}")

    weather, stations = get_weather_for_point(lat, lon)

    print(f"\nWeather (IDW from 3 nearest stations):")
    for k, v in weather.items():
        print(f"  {k}: {v}")

    print(f"\nStations used:")
    for s in stations:
        print(f"  {s['name']} ({s['station_triplet']}) — {s['distance_km']} km")

    return weather, stations


def test_terrain_only(lat, lon):
    """Test just the terrain extraction (needs DEM file)."""
    from terrain import extract_terrain

    print(f"\n{'='*50}")
    print(f"TERRAIN TEST: ({lat}, {lon})")
    print(f"{'='*50}")

    try:
        terrain = extract_terrain(lat, lon)
        print(f"\nTerrain:")
        for k, v in terrain.items():
            print(f"  {k}: {v}")
        return terrain
    except Exception as e:
        print(f"\n✗ Terrain extraction failed: {e}")
        print(f"  Make sure colorado_dem.tif is in {DATA_DIR}/")
        return None


def test_full_pipeline(lat, lon):
    """Test the full Lambda handler end-to-end."""
    from lambda_function import lambda_handler

    print(f"\n{'='*50}")
    print(f"FULL PIPELINE TEST: ({lat}, {lon})")
    print(f"{'='*50}")

    event = {"latitude": lat, "longitude": lon}
    result = lambda_handler(event, None)

    status = result["statusCode"]
    body = json.loads(result["body"])

    print(f"\nStatus: {status}")
    print(f"Response:\n{json.dumps(body, indent=2)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test avalanche inference locally")
    parser.add_argument("--mock", action="store_true",
                        help="Generate mock data files for smoke testing")
    parser.add_argument("--lat", type=float, default=39.6,
                        help="Test latitude (default: 39.6)")
    parser.add_argument("--lon", type=float, default=-105.8,
                        help="Test longitude (default: -105.8)")
    parser.add_argument("--weather-only", action="store_true",
                        help="Test only weather lookup (no DEM needed)")
    parser.add_argument("--terrain-only", action="store_true",
                        help="Test only terrain extraction (needs DEM)")
    args = parser.parse_args()

    if args.mock:
        create_mock_data()
        print()

    # Check data files exist
    required = ["snotel_stations_const.csv", "daily_station_data.csv",
                "avalanche_classifier_tuned.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]

    if missing:
        print(f"✗ Missing data files in {DATA_DIR}/:")
        for f in missing:
            print(f"    - {f}")
        print(f"\nRun with --mock to generate fake data, or copy real files there.")
        sys.exit(1)

    if args.weather_only:
        test_weather_only(args.lat, args.lon)
    elif args.terrain_only:
        test_terrain_only(args.lat, args.lon)
    else:
        # If no DEM, fall back to weather-only test
        dem_path = os.path.join(DATA_DIR, "colorado_dem.tif")
        if not os.path.exists(dem_path):
            print(f"⚠ No DEM file found at {dem_path}")
            print(f"  Running weather-only test. Copy colorado_dem.tif for full test.\n")
            test_weather_only(args.lat, args.lon)
        else:
            test_full_pipeline(args.lat, args.lon)


if __name__ == "__main__":
    main()
