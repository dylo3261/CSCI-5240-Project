"""
Avalanche prediction + SHAP explanation.

Loads the tuned model package (joblib dict) from S3, runs prediction
with the optimal threshold, and generates SHAP TreeExplainer output.
"""

import os
import io
import logging

import numpy as np
import pandas as pd
import joblib
import shap

from utils.s3_helpers import (
    IS_LAMBDA, DATA_DIR, S3_MODEL_KEY,
    download_model_file,
)

logger = logging.getLogger(__name__)

MODEL_FILE = os.path.join(DATA_DIR, "avalanche_classifier_tuned.pkl")

# Cache across warm invocations
_model_package = None  # {"model": ..., "optimal_threshold": ..., "feature_cols": [...]}
_explainer = None


def _load_model_package() -> dict:
    """Load the tuned model package from S3 (joblib dict)."""
    global _model_package
    if _model_package is not None:
        return _model_package

    if IS_LAMBDA or not os.path.exists(MODEL_FILE):
        download_model_file(S3_MODEL_KEY, MODEL_FILE)

    _model_package = joblib.load(MODEL_FILE)
    logger.info(
        f"Model loaded — threshold: {_model_package['optimal_threshold']:.2f}, "
        f"features: {_model_package['feature_cols']}"
    )
    return _model_package


def _get_explainer():
    """Create and cache SHAP TreeExplainer (matches 07_explain_predictions.py)."""
    global _explainer
    if _explainer is not None:
        return _explainer

    pkg = _load_model_package()
    _explainer = shap.TreeExplainer(pkg["model"])
    logger.info("SHAP TreeExplainer created")
    return _explainer


def predict_and_explain(terrain: dict, weather: dict) -> dict:
    """
    Run prediction + SHAP explanation.

    Args:
        terrain: {"elevation": float, "slope": float, "aspect_degrees": float}
        weather: {"snow_depth": float, "new_snow_24h": float, "swe": float, "temp": float}

    Returns:
        {
            "prediction": float (probability),
            "risk_level": str,
            "optimal_threshold": float,
            "shap_values": {feature: shap_value, ...} sorted by |impact|,
            "base_value": float,
            "explanation": str (human-readable),
            "features_used": {feature: value, ...},
        }
    """
    pkg = _load_model_package()
    model = pkg["model"]
    optimal_threshold = pkg["optimal_threshold"]
    feature_cols = pkg["feature_cols"]

    # Build feature vector in exact order the model expects
    # feature_cols = ['elevation', 'slope', 'aspect_degrees', 'snow_depth', 'new_snow_24h', 'swe', 'temp']
    features = {
        "elevation": terrain["elevation"],
        "slope": terrain["slope"],
        "aspect_degrees": terrain["aspect_degrees"],
        "snow_depth": weather["snow_depth"],
        "new_snow_24h": weather["new_snow_24h"],
        "swe": weather["swe"],
        "temp": weather["temp"],
    }

    # Check for missing values
    missing = [k for k, v in features.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing feature values: {missing}. "
            f"Not enough SNOTEL data available for prediction."
        )

    # Create DataFrame with correct column order
    feature_df = pd.DataFrame([features])[feature_cols]

    # ── Predict ──────────────────────────────────────────────────────────
    prob = float(model.predict_proba(feature_df)[0, 1])

    # Risk label using tuned threshold (matches 07_explain_predictions.py)
    if prob >= optimal_threshold:
        risk_level = "HIGH DANGER"
    elif prob >= (optimal_threshold - 0.1):
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # ── SHAP ─────────────────────────────────────────────────────────────
    explainer = _get_explainer()
    shap_values = explainer.shap_values(feature_df)

    # Handle different SHAP return formats (from 07_explain_predictions.py)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # Binary: [class_0, class_1] → class_1, first sample
    else:
        sv = shap_values[0]

    if len(sv.shape) > 1:
        sv = sv.flatten()

    sv = sv[:len(feature_cols)]

    # SHAP dict sorted by absolute impact
    shap_dict = {col: round(float(val), 6) for col, val in zip(feature_cols, sv)}
    shap_sorted = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))

    # Base value
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])
    else:
        base_value = float(base_value)

    # ── Structured explanation ────────────────────────────────────────────
    explanation = _build_explanation(shap_sorted, features)

    return {
        "prediction": round(prob, 4),
        "risk_level": risk_level,
        "optimal_threshold": round(optimal_threshold, 4),
        "shap_values": shap_sorted,
        "base_value": round(base_value, 4),
        "explanation": explanation,
        "features_used": features,
    }


def _build_explanation(shap_sorted: dict, features: dict) -> list[dict]:
    """
    Build structured SHAP explanation as a list of dicts.

    Returns:
        [
            {
                "feature": "new_snow_24h",
                "value": 35.0,
                "shap_impact": 0.234,
                "direction": "increasing",
                "context": "heavy recent snowfall - CRITICAL"
            },
            ...
        ]

    Sorted by absolute SHAP impact. Top 3 increasing + top 3 decreasing.
    """
    explanation = []

    increasing = [(k, v) for k, v in shap_sorted.items() if v > 0][:3]
    for feat, val in increasing:
        context = _feature_context(feat, features[feat])
        explanation.append({
            "feature": feat,
            "value": round(features[feat], 1),
            "shap_impact": round(val, 4),
            "direction": "increasing",
            "context": context if context else None,
        })

    decreasing = [(k, v) for k, v in shap_sorted.items() if v < 0][:3]
    for feat, val in decreasing:
        context = _feature_context(feat, features[feat])
        explanation.append({
            "feature": feat,
            "value": round(features[feat], 1),
            "shap_impact": round(val, 4),
            "direction": "decreasing",
            "context": context if context else None,
        })

    return explanation


def _feature_context(feature: str, value: float) -> str:
    """Return short context string for a feature value."""
    if feature == "new_snow_24h":
        if value > 30: return "heavy recent snowfall - CRITICAL"
        if value > 15: return "significant new snow"
        if value < 5: return "minimal new snow"
    elif feature == "slope":
        if value > 40: return "very steep - HIGH IMPACT"
        if value > 35: return "steep avalanche terrain"
        if value < 25: return "lower angle terrain"
    elif feature == "temp":
        if value > 0: return "above freezing - unstable"
        if value < -15: return "very cold - generally stable"
    elif feature == "snow_depth":
        if value > 200: return "very deep snowpack"
        if value > 150: return "deep snowpack"
    elif feature == "elevation":
        if value > 3500: return "alpine zone"
        if value > 3000: return "high elevation"
    return ""
