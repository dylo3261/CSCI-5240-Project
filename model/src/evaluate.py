"""Model evaluation utilities for regression.

Computes regression metrics (RMSE, MAE, R²) and provides logging helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Return a dictionary of standard regression metrics.

    Parameters
    ----------
    y_true : ground-truth avalanche sizes
    y_pred : predicted avalanche sizes
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics: dict = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    return metrics


def log_regression_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str = "test",
) -> dict:
    """Compute, log, and return regression metrics for a split."""
    metrics = compute_metrics(y_true, y_pred)

    residuals = y_true - y_pred
    logger.info(
        "\n── %s regression summary ──\n"
        "  RMSE : %.4f\n"
        "  MAE  : %.4f\n"
        "  R²   : %.4f\n"
        "  Residual mean  : %.4f\n"
        "  Residual std   : %.4f\n"
        "  Residual range : [%.4f, %.4f]",
        split_name.upper(),
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
        float(np.mean(residuals)),
        float(np.std(residuals)),
        float(np.min(residuals)),
        float(np.max(residuals)),
    )
    return metrics


def save_metrics(metrics: dict, directory: Path | None = None) -> Path:
    """Persist evaluation metrics to a JSON file."""
    out_dir = directory or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", path)
    return path
