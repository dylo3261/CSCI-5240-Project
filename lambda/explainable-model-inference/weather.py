"""
Weather data from SNOTEL stations via daily_station_data.csv.

Reads the pre-fetched daily station data that the cron job Lambda
already updates in S3. No live SNOTEL queries needed.

Flow:
  1. Load daily_station_data.csv (updated by teammate's cron job)
  2. Load snotel_stations_const.csv (station locations)
  3. Find 3 nearest stations to (lat, lon) using haversine
  4. Look up today's weather for those 3 stations
  5. IDW-interpolate → snow_depth, new_snow_24h, swe, temp
"""

import os
import logging
from math import radians, cos, sin, asin, sqrt
from datetime import datetime

import numpy as np
import pandas as pd

from utils.s3_helpers import (
    IS_LAMBDA, DATA_DIR,
    S3_DAILY_STATION_KEY, S3_STATIONS_KEY,
    download_data_file,
)

logger = logging.getLogger(__name__)

DAILY_STATION_FILE = os.path.join(DATA_DIR, "daily_station_data.csv")
STATIONS_FILE = os.path.join(DATA_DIR, "snotel_stations_const.csv")

# Cache across warm invocations
_stations_df = None
_daily_df = None
_daily_df_date = None  # Track when we last loaded daily data


def _load_stations() -> pd.DataFrame:
    """Load SNOTEL station locations (constant file)."""
    global _stations_df
    if _stations_df is not None:
        return _stations_df

    if IS_LAMBDA or not os.path.exists(STATIONS_FILE):
        download_data_file(S3_STATIONS_KEY, STATIONS_FILE)

    _stations_df = pd.read_csv(STATIONS_FILE)
    logger.info(f"Loaded {len(_stations_df)} SNOTEL stations")
    return _stations_df


def _load_daily_data() -> pd.DataFrame:
    """
    Load daily station weather data.
    Re-downloads from S3 at most once per Lambda invocation
    (or if the cached date is stale).
    """
    global _daily_df, _daily_df_date
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Use cache if we already loaded today's data in this invocation
    if _daily_df is not None and _daily_df_date == today_str:
        return _daily_df

    # Download fresh copy
    if IS_LAMBDA or not os.path.exists(DAILY_STATION_FILE):
        download_data_file(S3_DAILY_STATION_KEY, DAILY_STATION_FILE)

    _daily_df = pd.read_csv(DAILY_STATION_FILE)
    _daily_df["date"] = pd.to_datetime(_daily_df["date"]).dt.strftime("%Y-%m-%d")
    _daily_df_date = today_str
    logger.info(f"Loaded daily station data: {len(_daily_df)} rows, "
                f"dates: {_daily_df['date'].min()} to {_daily_df['date'].max()}")
    return _daily_df


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in km between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))


def find_nearest_stations(lat: float, lon: float, n: int = 3) -> list[dict]:
    """
    Find the n nearest SNOTEL stations to a (lat, lon) point.

    Returns list of dicts:
        [{"station_id": int, "station_triplet": str, "name": str, "distance_km": float}, ...]
    """
    stations_df = _load_stations()

    results = []
    for _, station in stations_df.iterrows():
        d = haversine_distance(lat, lon, station["latitude"], station["longitude"])
        results.append({
            "station_id": int(station["station_id"]),
            "station_triplet": station["station_triplet"],
            "name": station.get("name", ""),
            "distance_km": d,
        })

    results.sort(key=lambda x: x["distance_km"])
    return results[:n]


def inverse_distance_weighting(values: list, distances: list) -> float | None:
    """
    Inverse distance weighting interpolation.
    Matches the IDW logic from your weather_utils.py.
    """
    valid = [
        (v, d) for v, d in zip(values, distances)
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]

    if not valid:
        return None

    # If a station is essentially at the query point, use its value
    for v, d in valid:
        if d < 0.01:
            return float(v)

    weights = [1.0 / d for _, d in valid]
    total_weight = sum(weights)
    weighted_sum = sum(v * w for (v, _), w in zip(valid, weights))

    return round(float(weighted_sum / total_weight), 2)


def get_weather_for_point(lat: float, lon: float, date_str: str = None) -> tuple[dict, list]:
    """
    Get IDW-interpolated weather for a (lat, lon) point.

    Reads from daily_station_data.csv (updated by cron job).
    Uses the most recent available date if date_str is not specified.

    Args:
        lat: Latitude
        lon: Longitude
        date_str: Optional date string "YYYY-MM-DD". Defaults to most recent in data.

    Returns:
        (weather_dict, stations_used_list)

        weather_dict: {"snow_depth": float, "new_snow_24h": float, "swe": float, "temp": float}
        stations_used: [{"station_id": int, "station_triplet": str, "name": str, "distance_km": float}, ...]
    """
    daily_df = _load_daily_data()

    # Determine which date to use
    if date_str is None:
        # Use the most recent date available in the data
        date_str = daily_df["date"].max()
        logger.info(f"Using most recent date in data: {date_str}")

    # Filter to the target date
    day_data = daily_df[daily_df["date"] == date_str]
    if day_data.empty:
        available = sorted(daily_df["date"].unique())[-5:]
        raise ValueError(
            f"No weather data for date {date_str}. "
            f"Most recent dates available: {available}"
        )

    # Find 3 nearest stations
    nearest = find_nearest_stations(lat, lon, n=3)

    # Look up weather for each nearest station
    snow_depths = []
    new_snows = []
    swes = []
    temps = []
    distances = []

    for station_info in nearest:
        sid = station_info["station_id"]
        station_row = day_data[day_data["station_id"] == sid]

        if station_row.empty:
            logger.warning(f"No data for station {sid} on {date_str}")
            snow_depths.append(None)
            new_snows.append(None)
            swes.append(None)
            temps.append(None)
        else:
            row = station_row.iloc[0]
            snow_depths.append(row.get("snow_depth"))
            new_snows.append(row.get("new_snow_24hr"))  # Note: teammate uses "new_snow_24hr"
            swes.append(row.get("swe"))
            temps.append(row.get("temp"))

        distances.append(station_info["distance_km"])

    # IDW interpolation
    weather = {
        "snow_depth": inverse_distance_weighting(snow_depths, distances),
        "new_snow_24h": inverse_distance_weighting(new_snows, distances),
        "swe": inverse_distance_weighting(swes, distances),
        "temp": inverse_distance_weighting(temps, distances),
    }

    # Add metadata
    stations_used = [
        {
            "station_id": s["station_id"],
            "station_triplet": s["station_triplet"],
            "name": s["name"],
            "distance_km": round(s["distance_km"], 2),
        }
        for s in nearest
    ]

    return weather, stations_used
