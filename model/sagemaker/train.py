"""SageMaker-compatible training script for Logistic Regression avalanche classifier.

SageMaker invokes this file as:
    python train.py --train /opt/ml/input/data/train --model-dir /opt/ml/model

It reads CSV(s) from the train channel, trains, evaluates on an 80/20 split,
and writes the .pkl artifact to model-dir so SageMaker can package it.
"""

from __future__ import annotations

import argparse
import json
import os
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

from config import (
    MODEL_FILENAME,
    SELECTED_FEATURES,
    TARGET_COL,
    pred_class,
)


# ── Feature engineering ──────────────────────────────────────────────────────

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


def load_csvs(data_dir: str) -> pd.DataFrame:
    """Concatenate every CSV found under *data_dir*."""
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(csv_files)} file(s), {len(df)} rows from {data_dir}")
    return df


# ── Training ─────────────────────────────────────────────────────────────────

def train(train_dir: str, model_dir: str, test_size: float = 0.2) -> None:
    # 1. Load & engineer
    df = load_csvs(train_dir)
    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    # 2. Build preprocessor on the selected numeric features
    numeric_features = [f for f in SELECTED_FEATURES if f in X.columns]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]), numeric_features),
    ])

    # 3. Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    print(f"Dataset: {len(df)} total samples")
    print(f"  Train: {len(X_train)} ({100*(1-test_size):.0f}%)")
    print(f"  Test:  {len(X_test)} ({100*test_size:.0f}%)")
    print(f"  Train class dist: {dict(y_train.value_counts().sort_index())}")
    print(f"  Test  class dist: {dict(y_test.value_counts().sort_index())}")
    print(f"  Features ({len(numeric_features)}): {numeric_features}")

    # 4. Grid search — recall-optimised
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
        n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    # 5. Evaluate on 20% test set
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    risk_classes = [pred_class(p) for p in y_proba]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred)),
        "false_negatives": int(fn),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }

    risk_dist = pd.Series(risk_classes).value_counts().sort_index().to_dict()

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV recall: {grid.best_score_:.4f}")
    print(f"\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nRisk class distribution: {risk_dist}")

    # 6. Persist model artifact to /opt/ml/model (SageMaker convention)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "pipeline": best,
        "feature_columns": feature_cols,
        "numeric_features": numeric_features,
        "metrics": metrics,
        "best_params": grid.best_params_,
    }, model_path / MODEL_FILENAME)

    # Also write metrics as JSON for SageMaker Experiments / Pipelines
    with open(model_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {model_path / MODEL_FILENAME}")


# ── CLI (SageMaker passes these automatically) ──────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    train(args.train, args.model_dir, args.test_size)
