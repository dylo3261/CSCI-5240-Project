"""
Avalanche prediction + SHAP explanation.

Loads the logistic regression pipeline (joblib dict) from S3 and runs
prediction with a 5-tier risk classification scheme, using SHAP
LinearExplainer for per-feature attribution.
"""

import os
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

MODEL_FILE = os.path.join(DATA_DIR, "logistic_avalanche.pkl")

# Cache across warm invocations
_model_package = None  # {"pipeline": ..., "numeric_features": [...], ...}
_explainer = None


def _pred_class(p: float) -> str:
    """Map probability to 5-tier avalanche risk label."""
    if p >= 0.80: return "Extreme"
    if p >= 0.60: return "High"
    if p >= 0.40: return "Considerable"
    if p >= 0.20: return "Moderate"
    return "Low"


def _load_model_package() -> dict:
    """Load the logistic regression pipeline from S3 (joblib dict)."""
    global _model_package
    if _model_package is not None:
        return _model_package

    if IS_LAMBDA or not os.path.exists(MODEL_FILE):
        download_model_file(S3_MODEL_KEY, MODEL_FILE)

    _model_package = joblib.load(MODEL_FILE)
    logger.info(f"Model loaded — features: {_model_package['numeric_features']}")
    return _model_package


def _get_explainer():
    """Create and cache SHAP LinearExplainer for the logistic regression pipeline."""
    global _explainer
    if _explainer is not None:
        return _explainer

    pkg = _load_model_package()
    pipeline = pkg["pipeline"]
    feature_cols = pkg["numeric_features"]

    # Representative Colorado mountain background for LinearExplainer.
    # Used as the reference point for SHAP deviation computation.
    background = pd.DataFrame([{
        "elevation": 3000,
        "slope": 25,
        "aspect_degrees": 180,
        "snow_depth": 80,
        "new_snow_24h": 10,
        "temp": -8,
        "snow_ratio": 3.0,
    }])[feature_cols]

    preprocessor = pipeline[:-1]   # SimpleImputer + RobustScaler
    X_background = preprocessor.transform(background)

    _explainer = shap.LinearExplainer(pipeline[-1], X_background)
    logger.info("SHAP LinearExplainer created")
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
            "risk_level": str (Low/Moderate/Considerable/High/Extreme),
            "shap_values": {feature: shap_value, ...} sorted by |impact|,
            "base_value": float,
            "explanation": list[dict],
            "features_used": {feature: value, ...},
        }
    """
    pkg = _load_model_package()
    pipeline = pkg["pipeline"]
    feature_cols = pkg["numeric_features"]

    # Compute snow_ratio from swe (swe still fetched from SNOTEL but not a direct feature)
    swe = weather.get("swe")
    snow_depth = weather["snow_depth"]
    snow_ratio = round(snow_depth / swe, 2) if swe and swe > 0 else 0.0

    features = {
        "elevation": terrain["elevation"],
        "slope": terrain["slope"],
        "aspect_degrees": terrain["aspect_degrees"],
        "snow_depth": snow_depth,
        "new_snow_24h": weather["new_snow_24h"],
        "temp": weather["temp"],
        "snow_ratio": snow_ratio,
    }

    missing = [k for k, v in features.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing feature values: {missing}. "
            "Not enough SNOTEL data available for prediction."
        )

    feature_df = pd.DataFrame([features])[feature_cols]

    # ── Predict ──────────────────────────────────────────────────────────
    prob = float(pipeline.predict_proba(feature_df)[0, 1]) * 0.35
    risk_level = _pred_class(prob)

    # ── SHAP ─────────────────────────────────────────────────────────────
    explainer = _get_explainer()
    preprocessor = pipeline[:-1]
    X_preprocessed = preprocessor.transform(feature_df)
    shap_values = explainer.shap_values(X_preprocessed)  # (n_samples, n_features)

    sv = np.array(shap_values[0]) if isinstance(shap_values, list) else np.array(shap_values[0])
    if sv.ndim > 1:
        sv = sv.flatten()
    sv = sv[:len(feature_cols)]

    shap_dict = {col: round(float(val), 6) for col, val in zip(feature_cols, sv)}
    shap_sorted = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))

    base_value = float(explainer.expected_value)

    # ── Structured explanation ────────────────────────────────────────────
    explanation = _build_explanation(shap_sorted, features)

    return {
        "prediction": round(prob, 4),
        "risk_level": risk_level,
        "shap_values": shap_sorted,
        "base_value": round(base_value, 4),
        "explanation": explanation,
        "features_used": features,
    }


def _build_explanation(shap_sorted: dict, features: dict) -> list[dict]:
    """Build structured SHAP explanation. Top 3 increasing + top 3 decreasing."""
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
    elif feature == "snow_ratio":
        if value > 5.0: return "very high ratio - instability risk"
        if value > 4.0: return "high snow ratio"
        if value < 2.0: return "low density snowpack"
    return ""
