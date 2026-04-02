"""
AWS Lambda handler for daily regional risk grid.

Generates a grid of avalanche risk predictions across Colorado's mountain terrain.
Triggered by EventBridge cron rule each morning.

Pipeline:
  1. Generate grid of (lat, lon) points at configurable resolution
  2. Load DEM, model, and weather data ONCE into memory
  3. Run batch predictions across all grid cells
  4. Aggregate into summary statistics
  5. Save results to S3 as JSON

Grid defaults:
  - North/South: Colorado borders (41.0 / 37.0)
  - West: Colorado border (-109.0)
  - East: Denver (-104.9)
  - Cell size: 2 miles (~0.029° lat × ~0.037° lon)
"""

import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import boto3
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import rowcol

from terrain import calculate_slope_aspect, _load_dem
from weather import (
    _load_stations, _load_daily_data,
    find_nearest_stations, inverse_distance_weighting,
    haversine_distance,
)
from predict import _load_model_package, _get_explainer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Grid Configuration (override via env vars or event payload) ──────────────
DEFAULT_GRID_CONFIG = {
    "north": 41.0,
    "south": 37.0,
    "west": -109.05,  # Extended west to match frontend grid
    "east": -105.314,  # Extended east to match frontend grid
    "cell_size_mi": 2,  # miles per grid cell (matches frontend 2mi × 2mi grid)
}

# Conversion: miles to degrees at Colorado's average latitude (~39°)
MI_TO_DEG_LAT = 1 / 69.0  # ~0.0145° per mile
MI_TO_DEG_LON = 1 / 54.6  # ~0.0183° per mile (at 39° latitude)

# Output bucket
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "explain-model-bucket-west2")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "daily-grid-scores/")


def lambda_handler(event, context):
    """
    Entry point for EventBridge cron or direct invocation.

    Optional event payload to override grid config:
        {
            "cell_size_mi": 5,
            "north": 41.0,
            "south": 37.0,
            "west": -109.0,
            "east": -104.9
        }
    """
    start_time = time.time()

    try:
        # ── Parse grid config ────────────────────────────────────────────
        config = {**DEFAULT_GRID_CONFIG}
        if isinstance(event, dict):
            for key in DEFAULT_GRID_CONFIG:
                if key in event:
                    config[key] = float(event[key])

        cell_size_mi = config["cell_size_mi"]
        
        # Use exact spacing from frontend grid (calibrated for 2mi cells)
        # These values ensure 139 lat × 100 lon cells matching the example file
        if cell_size_mi == 2:
            lat_step = 0.028986  # Exact spacing from example file
            lon_step = 0.037736  # Exact spacing from example file
        else:
            # For other cell sizes, use standard conversion
            lat_step = cell_size_mi * MI_TO_DEG_LAT
            lon_step = cell_size_mi * MI_TO_DEG_LON

        logger.info(f"Grid config: {config}")
        logger.info(f"Step sizes: lat={lat_step:.6f}°, lon={lon_step:.6f}°")

        # ── Generate grid points ─────────────────────────────────────────
        # Add small epsilon to ensure endpoints are included
        lats = np.arange(config["south"], config["north"] + lat_step/2, lat_step)
        lons = np.arange(config["west"], config["east"] + lon_step/2, lon_step)
        total_points = len(lats) * len(lons)
        logger.info(f"Grid: {len(lats)} rows × {len(lons)} cols = {total_points} points")

        # ── Load all data into memory ONCE ───────────────────────────────
        logger.info("Loading DEM...")
        dem_data = _load_dem()

        logger.info("Loading model...")
        pkg = _load_model_package()
        model = pkg["model"]
        optimal_threshold = pkg["optimal_threshold"]
        feature_cols = pkg["feature_cols"]

        logger.info("Loading weather data...")
        stations_df = _load_stations()
        daily_df = _load_daily_data()

        # Get the most recent date in the weather data
        date_str = daily_df["date"].max()
        day_data = daily_df[daily_df["date"] == date_str]
        logger.info(f"Using weather date: {date_str}")

        # Pre-compute station locations as arrays for fast distance calc
        station_lats = stations_df["latitude"].values
        station_lons = stations_df["longitude"].values
        station_ids = stations_df["station_id"].values

        # ── Open DEM once, read elevation array once ─────────────────────
        logger.info("Opening DEM for batch processing...")
        cells = []
        skipped = 0

        with MemoryFile(dem_data) as memfile:
            with memfile.open() as src:
                elevation_data = src.read(1)
                transform = src.transform
                nodata = src.nodata
                height = src.height
                width = src.width

                # ── Batch predict ────────────────────────────────────────
                for lat in lats:
                    for lon in lons:
                        result = _predict_single(
                            lat, lon,
                            elevation_data, transform, nodata, height, width,
                            station_lats, station_lons, station_ids,
                            stations_df, day_data,
                            model, optimal_threshold, feature_cols,
                        )
                        if result is not None:
                            cells.append(result)
                        else:
                            skipped += 1

        elapsed = time.time() - start_time
        logger.info(
            f"Completed: {len(cells)} predictions, {skipped} skipped, "
            f"{elapsed:.1f}s elapsed"
        )

        # ── Summary statistics ───────────────────────────────────────────
        if cells:
            predictions = [c["prediction"] for c in cells]
            summary = {
                "avg_risk": round(np.mean(predictions), 4),
                "max_risk": round(np.max(predictions), 4),
                "min_risk": round(np.min(predictions), 4),
                "median_risk": round(np.median(predictions), 4),
                "pct_high": round(
                    sum(1 for p in predictions if p >= optimal_threshold) / len(predictions), 4
                ),
                "pct_moderate": round(
                    sum(1 for p in predictions if (optimal_threshold - 0.1) <= p < optimal_threshold)
                    / len(predictions), 4
                ),
                "pct_low": round(
                    sum(1 for p in predictions if p < (optimal_threshold - 0.1)) / len(predictions), 4
                ),
                "total_cells": len(cells),
                "skipped_cells": skipped,
            }
        else:
            summary = {"error": "No valid predictions", "skipped_cells": skipped}

        # ── Save grid scores as CSV to S3 ─────────────────────────────
        s3 = boto3.client("s3")

        # Grid scores CSV (daily archive) - match frontend format
        cells_df = pd.DataFrame(cells)
        # Rename 'prediction' to 'value' and keep only lat, lon, value columns
        frontend_df = cells_df[['lat', 'lon', 'prediction']].copy()
        frontend_df.rename(columns={'prediction': 'value'}, inplace=True)
        csv_buffer = frontend_df.to_csv(index=False)

        csv_key = f"{OUTPUT_PREFIX}{date_str}/daily_map_grid_scores.csv"
        s3.put_object(
            Bucket=OUTPUT_BUCKET,
            Key=csv_key,
            Body=csv_buffer,
            ContentType="text/csv",
        )
        logger.info(f"Saved grid CSV to s3://{OUTPUT_BUCKET}/{csv_key}")

        # Also save as latest for easy frontend access
        s3.put_object(
            Bucket=OUTPUT_BUCKET,
            Key=f"{OUTPUT_PREFIX}latest/daily_map_grid_scores.csv",
            Body=csv_buffer,
            ContentType="text/csv",
        )

        # Summary JSON (metadata + stats, no cell data)
        summary_output = {
            "date": date_str,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "grid_config": {
                "cell_size_mi": cell_size_mi,
                "lat_step_deg": round(lat_step, 4),
                "lon_step_deg": round(lon_step, 4),
                "bounds": {
                    "north": config["north"],
                    "south": config["south"],
                    "west": config["west"],
                    "east": config["east"],
                },
            },
            "summary": summary,
            "elapsed_seconds": round(elapsed, 1),
        }

        summary_key = f"{OUTPUT_PREFIX}{date_str}/summary.json"
        s3.put_object(
            Bucket=OUTPUT_BUCKET,
            Key=summary_key,
            Body=json.dumps(summary_output),
            ContentType="application/json",
        )

        s3.put_object(
            Bucket=OUTPUT_BUCKET,
            Key=f"{OUTPUT_PREFIX}latest/summary.json",
            Body=json.dumps(summary_output),
            ContentType="application/json",
        )
        logger.info(f"Saved summary to s3://{OUTPUT_BUCKET}/{summary_key}")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"Generated {len(cells)} grid predictions for {date_str}",
                "csv_key": csv_key,
                "summary_key": summary_key,
                "summary": summary,
                "elapsed_seconds": round(elapsed, 1),
            }),
        }

    except Exception as e:
        logger.exception("Regional risk grid failed")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }


def _predict_single(
    lat, lon,
    elevation_data, transform, nodata, height, width,
    station_lats, station_lons, station_ids,
    stations_df, day_data,
    model, optimal_threshold, feature_cols,
):
    """
    Run a single prediction for one grid cell.
    Returns dict with lat, lon, prediction, risk_level — or None if cell is invalid.
    All data is pre-loaded; no S3 calls happen here.
    """
    try:
        # ── Terrain ──────────────────────────────────────────────────────
        py, px = rowcol(transform, lon, lat)

        if py < 1 or py >= height - 1 or px < 1 or px >= width - 1:
            return {
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "prediction": 0.0,
                "risk_level": "LOW",
                "elevation": 0.0,
            }

        elevation = float(elevation_data[py, px])
        if elevation == nodata or elevation < -999:
            return {
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "prediction": 0.0,
                "risk_level": "LOW",
                "elevation": 0.0,
            }

        slope, aspect_deg = calculate_slope_aspect(
            elevation_data, transform, py, px, lat
        )
        if slope is None:
            return {
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "prediction": 0.0,
                "risk_level": "LOW",
                "elevation": 0.0,
            }

        # ── Weather (fast: vectorized distance calc) ─────────────────────
        distances = np.array([
            haversine_distance(lat, lon, slat, slon)
            for slat, slon in zip(station_lats, station_lons)
        ])
        nearest_idx = np.argsort(distances)[:3]
        nearest_sids = station_ids[nearest_idx]
        nearest_dists = distances[nearest_idx]

        snow_depths, new_snows, swes, temps = [], [], [], []
        valid_dists = []

        for sid, dist in zip(nearest_sids, nearest_dists):
            station_row = day_data[day_data["station_id"] == int(sid)]
            if station_row.empty:
                continue
            row = station_row.iloc[0]
            sd = row.get("snow_depth")
            ns = row.get("new_snow_24hr")
            sw = row.get("swe")
            tp = row.get("temp")
            if pd.notna(sd) and pd.notna(sw) and pd.notna(tp):
                snow_depths.append(sd)
                new_snows.append(ns if pd.notna(ns) else 0.0)
                swes.append(sw)
                temps.append(tp)
                valid_dists.append(dist)

        if not valid_dists:
            return {
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "prediction": 0.0,
                "risk_level": "LOW",
                "elevation": round(elevation, 0),
            }

        weather = {
            "snow_depth": inverse_distance_weighting(snow_depths, valid_dists),
            "new_snow_24h": inverse_distance_weighting(new_snows, valid_dists),
            "swe": inverse_distance_weighting(swes, valid_dists),
            "temp": inverse_distance_weighting(temps, valid_dists),
        }

        if any(v is None for v in weather.values()):
            return {
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "prediction": 0.0,
                "risk_level": "LOW",
                "elevation": round(elevation, 0),
            }

        # ── Predict ──────────────────────────────────────────────────────
        features = {
            "elevation": round(elevation, 1),
            "slope": round(slope, 2),
            "aspect_degrees": round(aspect_deg, 2),
            "snow_depth": weather["snow_depth"],
            "new_snow_24h": weather["new_snow_24h"],
            "swe": weather["swe"],
            "temp": weather["temp"],
        }

        feature_df = pd.DataFrame([features])[feature_cols]
        prob = float(model.predict_proba(feature_df)[0, 1])

        if prob >= optimal_threshold:
            risk_level = "HIGH DANGER"
        elif prob >= (optimal_threshold - 0.1):
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return {
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "prediction": round(prob, 4),
            "risk_level": risk_level,
            "elevation": round(elevation, 0),
        }

    except Exception:
        return None