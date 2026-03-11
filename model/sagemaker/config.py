"""Shared constants for training and inference."""

TARGET_COL = "avalanche_occurred"

# The 7 numeric features the model expects (after engineering)
SELECTED_FEATURES = [
    "elevation",
    "slope",
    "aspect_degrees",   # transformed: distance from south
    "snow_depth",
    "new_snow_24h",
    "temp",
    "snow_ratio",       # engineered: snow_depth / swe
]

RISK_THRESHOLDS = {
    "Low": 0.2,
    "Moderate": 0.4,
    "Considerable": 0.6,
    "High": 0.8,
}

MODEL_FILENAME = "logistic_avalanche.pkl"


def pred_class(prob: float) -> str:
    """Map avalanche probability to a risk category."""
    if prob < RISK_THRESHOLDS["Low"]:
        return "Low"
    if prob < RISK_THRESHOLDS["Moderate"]:
        return "Moderate"
    if prob < RISK_THRESHOLDS["Considerable"]:
        return "Considerable"
    if prob < RISK_THRESHOLDS["High"]:
        return "High"
    return "Extreme"
