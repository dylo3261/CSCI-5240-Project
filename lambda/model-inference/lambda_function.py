"""
AWS Lambda handler for avalanche risk prediction.

Receives (lat, lon) via API Gateway, returns prediction + SHAP explanation.

Pipeline:
  1. Extract terrain from Colorado DEM in S3 → elevation, slope, aspect
  2. Find 3 nearest SNOTEL stations from snotel_stations_const.csv in S3
  3. Look up today's weather from daily_station_data.csv in S3 (cron job updates this)
  4. IDW-interpolate weather from 3 stations
  5. Run tuned RandomForest classifier + SHAP TreeExplainer
  6. Return prediction, risk label, SHAP explanation
"""

import json
import logging

from terrain import extract_terrain
from weather import get_weather_for_point
from predict import predict_and_explain

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Entry point for API Gateway → Lambda.

    Input (JSON body):
        {"latitude": 39.6, "longitude": -105.8}

    Optional:
        {"latitude": 39.6, "longitude": -105.8, "date": "2025-01-15"}

    Returns:
        {
            "prediction": 0.73,
            "risk_level": "HIGH DANGER",
            "optimal_threshold": 0.42,
            "shap_values": {...},
            "explanation": "...",
            "terrain": {"elevation": 3400, "slope": 38, "aspect_degrees": 315},
            "weather": {"snow_depth": 125, "new_snow_24h": 35, "swe": 32, "temp": -5},
            "stations_used": [...],
            "location": {"latitude": 39.6, "longitude": -105.8}
        }
    """
    try:
        # ── Parse input ──────────────────────────────────────────────────
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])  # API Gateway proxy format
        else:
            body = event  # Direct invocation / testing

        lat = float(body["latitude"])
        lon = float(body["longitude"])
        date_str = body.get("date")  # Optional: "YYYY-MM-DD"

        # Validate: rough Colorado bounding box
        if not (36.5 <= lat <= 41.5) or not (-109.5 <= lon <= -101.5):
            return _response(400, {
                "error": "Coordinates outside Colorado coverage area",
                "hint": "Latitude: 36.5–41.5, Longitude: -109.5 to -101.5"
            })

        logger.info(f"Predicting for ({lat}, {lon}), date={date_str}")

        # ── Step 1: Terrain from DEM ─────────────────────────────────────
        terrain = extract_terrain(lat, lon)
        logger.info(f"Terrain: {terrain}")

        # ── Step 2 & 3: Weather from daily station data (cron job CSV) ───
        weather, stations_used = get_weather_for_point(lat, lon, date_str)
        logger.info(f"Weather: {weather}")

        # ── Step 4: Predict + SHAP explain ───────────────────────────────
        result = predict_and_explain(terrain, weather)

        # ── Response ─────────────────────────────────────────────────────
        return _response(200, {
            "prediction": result["prediction"],
            "risk_level": result["risk_level"],
            "optimal_threshold": result["optimal_threshold"],
            "shap_values": result["shap_values"],
            "base_value": result["base_value"],
            "explanation": result["explanation"],
            "terrain": terrain,
            "weather": weather,
            "stations_used": stations_used,
            "location": {"latitude": lat, "longitude": lon},
        })

    except KeyError as e:
        return _response(400, {"error": f"Missing required field: {e}"})
    except ValueError as e:
        return _response(400, {"error": str(e)})
    except Exception as e:
        logger.exception("Prediction failed")
        return _response(500, {"error": str(e)})


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }
