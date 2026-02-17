"""Centralized configuration for avalanche prediction model."""

from dataclasses import dataclass, field
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # model/
DATA_DIR = PROJECT_ROOT / "src" / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "src" / "checkpoints"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_DATA_PATH = DATA_DIR / "caic_positive_2016_2026.csv"

# ── Feature Definitions ─────────────────────────────────────────────────────

TARGET_COL = "avalanche_size"

NUMERIC_FEATURES: list[str] = [
    "elevation",
    "slope",
    "aspect_degrees",
    "snow_depth",
    "new_snow_24h",
    "swe",
    "temp",
]

CATEGORICAL_FEATURES: list[str] = [
    "Area",
    # "Aspect",
    # "Type",
    # "Trigger",
]

# Columns intentionally excluded from modelling
DROP_COLS: list[str] = [
    "Observation ID",
    "Date",
    "Longitude",
    "latitude",
    "aspect_cardinal",   # redundant with aspect_degrees
    "avalanche_occurred", # redundant — implied by avalanche_size > 0
    "nearest_stations",
    "nearest_distances",
    "wind_speed",         # NaN data and wait for better measurements in future iterations
]


# ── Model Hyper-parameters ──────────────────────────────────────────────────

@dataclass
class XGBConfig:
    """XGBoost hyper-parameters with sensible defaults for regression."""

    # booster
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

    # objective / eval
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"
    early_stopping_rounds: int = 30

    # misc
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = 1


# ── Training Configuration ──────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """End-to-end training run configuration."""

    data_path: Path = DEFAULT_DATA_PATH
    test_size: float = 0.2
    val_size: float = 0.1          # fraction of *training* set used for validation
    random_state: int = 42
    xgb: XGBConfig = field(default_factory=XGBConfig)

    # artifact persistence
    save_model: bool = True
    model_name: str = "xgb_avalanche"
