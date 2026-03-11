"""SageMaker-compatible inference script for the avalanche logistic regression model.

SageMaker hosting calls four functions:
  model_fn   – load the .pkl from /opt/ml/model
  input_fn   – deserialise the incoming request (JSON)
  predict_fn – run the model
  output_fn  – serialise the response (JSON)

Expected JSON input (single or batch):
    {
        "elevation": 3200,
        "slope": 35,
        "aspect_degrees": 180,
        "snow_depth": 120,
        "new_snow_24h": 30,
        "temp": -5,
        "snow_ratio": 4.0
    }
  or a list of such objects.

Response:
    {
        "predictions": [
            {
                "probability": 0.73,
                "risk_class": "High"
            }
        ]
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODEL_FILENAME, SELECTED_FEATURES, pred_class


# ── SageMaker hosting hooks ─────────────────────────────────────────────────

def model_fn(model_dir: str) -> dict:
    """Load the serialised model artifact."""
    model_path = Path(model_dir) / MODEL_FILENAME
    model_data = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model_data


def input_fn(request_body: str, content_type: str = "application/json") -> pd.DataFrame:
    """Deserialise the request body into a DataFrame."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    payload = json.loads(request_body)

    # Accept a single dict or a list of dicts
    if isinstance(payload, dict):
        payload = [payload]

    df = pd.DataFrame(payload)

    # Validate that all required features are present
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df


def predict_fn(df: pd.DataFrame, model_data: dict) -> list[dict]:
    """Run inference and return probability + risk class for each row."""
    pipeline = model_data["pipeline"]
    numeric_features = model_data["numeric_features"]

    # Ensure column order matches training
    proba = pipeline.predict_proba(df)[:, 1]

    results = []
    for p in proba:
        results.append({
            "probability": round(float(p), 4),
            "risk_class": pred_class(float(p)),
        })
    return results


def output_fn(predictions: list[dict], accept: str = "application/json") -> str:
    """Serialise the predictions to JSON."""
    return json.dumps({"predictions": predictions})
