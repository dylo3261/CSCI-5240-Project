"""Training entry-point for the XGBoost avalanche size regression model.

Usage
-----
    cd model/
    uv run python train.py                      # default config
    uv run python train.py --data path/to.csv   # custom data path
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.config import ARTIFACTS_DIR, CHECKPOINT_DIR, TrainConfig, XGBConfig
from src.data.dataset import load_csv, make_splits
from src.evaluate import (
    compute_metrics,
    log_regression_summary,
    save_metrics,
)
from src.utils.preprocess import Preprocessor

logger = logging.getLogger(__name__)


# ── Core training logic ─────────────────────────────────────────────────────


def train(cfg: TrainConfig | None = None) -> xgb.XGBRegressor:
    """Run the full training pipeline and return the fitted model.

    Steps
    -----
    1. Load CSV
    2. Train / val / test split
    3. Fit preprocessor on training data, transform all splits
    4. Train XGBoost regressor with early stopping on the validation set
    5. Evaluate on the test set and persist artefacts
    """
    cfg = cfg or TrainConfig()

    # ── 1. Load ──────────────────────────────────────────────────────────
    df = load_csv(cfg.data_path)

    # ── 2. Split ─────────────────────────────────────────────────────────
    train_df, val_df, test_df = make_splits(df, cfg)

    # ── 3. Preprocess ────────────────────────────────────────────────────
    preprocessor = Preprocessor()
    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)

    X_train, y_train = Preprocessor.split_xy(train_df)
    X_val, y_val = Preprocessor.split_xy(val_df)
    X_test, y_test = Preprocessor.split_xy(test_df)

    # ── 4. Train ─────────────────────────────────────────────────────────
    xgb_params = cfg.xgb
    model = xgb.XGBRegressor(
        n_estimators=xgb_params.n_estimators,
        max_depth=xgb_params.max_depth,
        learning_rate=xgb_params.learning_rate,
        subsample=xgb_params.subsample,
        colsample_bytree=xgb_params.colsample_bytree,
        min_child_weight=xgb_params.min_child_weight,
        gamma=xgb_params.gamma,
        reg_alpha=xgb_params.reg_alpha,
        reg_lambda=xgb_params.reg_lambda,
        objective=xgb_params.objective,
        eval_metric=xgb_params.eval_metric,
        early_stopping_rounds=xgb_params.early_stopping_rounds,
        random_state=xgb_params.random_state,
        n_jobs=xgb_params.n_jobs,
        verbosity=xgb_params.verbosity,
    )

    logger.info("Starting XGBoost regression training …")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )
    logger.info("Training complete — best iteration: %s", model.best_iteration)

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    metrics = log_regression_summary(y_test.values, y_pred, split_name="test")

    logger.info("Test metrics: %s", metrics)

    # ── 6. Feature importance ────────────────────────────────────────────
    feature_names = Preprocessor.get_feature_columns()
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    logger.info("Feature importance (top 10):")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        logger.info("  %2d. %-20s  %.4f", rank, feature_names[idx], importances[idx])

    # ── 7. Persist artefacts ─────────────────────────────────────────────
    if cfg.save_model:
        _save_artifacts(model, preprocessor, metrics, cfg)

    return model


# ── Helpers ──────────────────────────────────────────────────────────────────


def _save_artifacts(
    model: xgb.XGBRegressor,
    preprocessor: Preprocessor,
    metrics: dict,
    cfg: TrainConfig,
) -> None:
    """Save model checkpoint, preprocessor state, and evaluation metrics."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = CHECKPOINT_DIR / f"{cfg.model_name}.json"
    model.save_model(model_path)
    logger.info("Model saved to %s", model_path)

    preprocessor.save(ARTIFACTS_DIR)
    save_metrics(metrics, ARTIFACTS_DIR)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for avalanche prediction.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for the test split (default: 0.2).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of training data for validation (default: 0.1).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Max boosting rounds (default: 300).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Max tree depth (default: 6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving model artefacts.",
    )
    args = parser.parse_args(argv)

    xgb_cfg = XGBConfig(
        learning_rate=args.lr,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
    )

    cfg = TrainConfig(
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        save_model=not args.no_save,
        xgb=xgb_cfg,
    )

    if args.data:
        cfg.data_path = args.data

    return cfg


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = parse_args(argv)
    logger.info("Training config: %s", cfg)
    train(cfg)


if __name__ == "__main__":
    main()
