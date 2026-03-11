"""Training entry-point for the Logistic Regression avalanche binary classifier.

Usage
-----
    cd model/
    python train_logistic.py                          # default config
    python train_logistic.py --data path/to/train.csv # custom data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent          # repo root
MODEL_DIR = Path(__file__).resolve().parent                    # model/
CHECKPOINT_DIR = MODEL_DIR / "src" / "checkpoints"
DATA_PROCESSED_DIR = MODEL_DIR / "src" /"data"

TARGET_COL = "avalanche_occurred"

# Numeric features used for modelling (after feature engineering)
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


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_training_data(data_path: Path | None = None) -> pd.DataFrame:
    """Locate and load the training CSV."""
    if data_path and data_path.exists():
        return pd.read_csv(data_path)

    csv_files = (
        list(DATA_PROCESSED_DIR.glob("*train*.csv"))
        + list(DATA_PROCESSED_DIR.glob("*training*.csv"))
    )
    if csv_files:
        return pd.read_csv(csv_files[0])

    csv_files = list(DATA_PROCESSED_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_PROCESSED_DIR}")
    return pd.read_csv(max(csv_files, key=lambda p: p.stat().st_mtime))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add snow_ratio and transform aspect_degrees (distance from south)."""
    df = df.copy()
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]

    snow_ratio = np.zeros(len(X))
    both_ok = ~(
        (X["snow_depth"] == 0) | X["snow_depth"].isnull()
        | (X["swe"] == 0) | X["swe"].isnull()
    )
    snow_ratio[both_ok] = X.loc[both_ok, "snow_depth"] / X.loc[both_ok, "swe"]
    df["snow_ratio"] = snow_ratio
    df["aspect_degrees"] = (180 - df["aspect_degrees"]).abs()
    return df


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


# ── Main training routine ───────────────────────────────────────────────────


def train(data_path: Path | None = None, test_size: float = 0.2) -> dict:
    """Train logistic regression, evaluate, and return results dict."""

    # 1. Load & engineer
    df = load_training_data(data_path)
    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    # 2. Select numeric features & build preprocessor
    numeric_features = [f for f in SELECTED_FEATURES if f in X.columns]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]), numeric_features),
    ])

    # 3. Split 80% train / 20% test (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    print(f"Dataset: {len(df)} total samples")
    print(f"  Train: {len(X_train)} ({100*(1-test_size):.0f}%)")
    print(f"  Test:  {len(X_test)} ({100*test_size:.0f}%)")
    print(f"  Train class dist: {dict(y_train.value_counts().sort_index())}")
    print(f"  Test  class dist: {dict(y_test.value_counts().sort_index())}")
    print(f"  Features used ({len(numeric_features)}): {numeric_features}")

    # 4. Grid search (recall-optimised)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            l1_ratio=0, solver="lbfgs", random_state=42, max_iter=1000,
        )),
    ])

    param_grid = {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "classifier__class_weight": ["balanced", {0: 1, 1: 3}],
    }

    grid = GridSearchCV(
        pipeline, param_grid,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        scoring=make_scorer(recall_score, pos_label=1),
        n_jobs=1, verbose=0,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    # 5. Evaluate on the 20% held-out test set
    print(f"\n{'='*50}")
    print(f"EVALUATION ON TEST SET ({len(X_test)} samples)")
    print(f"{'='*50}")
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    risk_classes = [pred_class(p) for p in y_proba]

    # 6. Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred),
        "false_negatives": int(fn),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
    }

    risk_dist = pd.Series(risk_classes).value_counts().sort_index().to_dict()

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV recall: {grid.best_score_:.4f}")
    print(f"\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nRisk class distribution:\n{risk_dist}")

    # 7. Save
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINT_DIR / "logistic_avalanche.pkl"
    joblib.dump({
        "pipeline": best,
        "feature_columns": feature_cols,
        "numeric_features": numeric_features,
        "metrics": metrics,
        "best_params": grid.best_params_,
    }, out_path)
    print(f"\nModel saved to {out_path}")

    return {
        "pipeline": best,
        "metrics": metrics,
        "risk_distribution": risk_dist,
        "predictions": y_pred,
        "probabilities": y_proba,
        "risk_classes": risk_classes,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression avalanche classifier")
    parser.add_argument("--data", type=Path, default=None, help="Path to training CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    train(data_path=args.data, test_size=args.test_size)
