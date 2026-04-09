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

        # ── Convert units for response (model runs on metric; output is US customary) ──
        terrain_us = {
            "elevation": round(terrain["elevation"] * 3.28084, 1),  # m → ft
            "slope": terrain["slope"],
            "aspect_degrees": terrain["aspect_degrees"],
        }
        weather_us = {
            "snow_depth": round(weather["snow_depth"] / 2.54, 2) if weather["snow_depth"] is not None else None,   # cm → in
            "new_snow_24h": round(weather["new_snow_24h"] / 2.54, 2) if weather["new_snow_24h"] is not None else None,  # cm → in
            "swe": round(weather["swe"] / 2.54, 2) if weather["swe"] is not None else None,                        # cm → in
            "temp": round(weather["temp"] * 9 / 5 + 32, 1) if weather["temp"] is not None else None,              # °C → °F
            "snow_ratio": result["features_used"]["snow_ratio"],
        }
        stations_us = [
            {**s, "distance_mi": round(s["distance_km"] * 0.621371, 2)}
            for s in stations_used
        ]
        explanation_us = _convert_explanation_units(result["explanation"])

        # ── Response ─────────────────────────────────────────────────────
        return _response(200, {
            "prediction": result["prediction"],
            "risk_level": result["risk_level"],
            "shap_values": result["shap_values"],
            "base_value": result["base_value"],
            "explanation": explanation_us,
            "terrain": terrain_us,
            "weather": weather_us,
            "stations_used": stations_us,
            "location": {"latitude": lat, "longitude": lon},
        })

    except KeyError as e:
        return _response(400, {"error": f"Missing required field: {e}"})
    except ValueError as e:
        return _response(400, {"error": str(e)})
    except Exception as e:
        logger.exception("Prediction failed")
        return _response(500, {"error": str(e)})


_FEATURE_UNIT_CONVERTERS = {
    "elevation": lambda v: round(v * 3.28084, 1),   # m → ft
    "snow_depth": lambda v: round(v / 2.54, 2),      # cm → in
    "new_snow_24h": lambda v: round(v / 2.54, 2),    # cm → in
    "temp": lambda v: round(v * 9 / 5 + 32, 1),      # °C → °F
}


def _convert_explanation_units(explanation: list) -> list:
    """Convert metric feature values in SHAP explanation entries to US customary."""
    result = []
    for item in explanation:
        item = dict(item)
        if item["feature"] in _FEATURE_UNIT_CONVERTERS:
            item["value"] = _FEATURE_UNIT_CONVERTERS[item["feature"]](item["value"])
        result.append(item)
    return result


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }
