"""Preprocessing pipeline for avalanche prediction features.

Responsibilities:
- Imputation of missing values
- Categorical encoding (label-encoding for tree-based models)
- Optional standard scaling for numeric features
- Serialisation of fitted transformers for reproducible inference
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    DROP_COLS,
    NUMERIC_FEATURES,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


class Preprocessor:
    """Stateful preprocessor that can be fitted on training data and applied
    to validation / test / production data consistently."""

    def __init__(self) -> None:
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._numeric_medians: dict[str, float] = {}
        self._is_fitted: bool = False

    # ── public API ───────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Learn encoding / imputation statistics from training data."""
        df = self._drop_unused(df)

        # numeric medians for imputation
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                self._numeric_medians[col] = float(df[col].median())

        # label encoders for categoricals
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                le = LabelEncoder()
                # fit on known categories + a placeholder for unseen values
                vals = df[col].fillna("__MISSING__").astype(str)
                le.fit(vals)
                self._label_encoders[col] = le

        self._is_fitted = True
        logger.info("Preprocessor fitted on %d rows.", len(df))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformations.  Returns a copy."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call .fit() first.")

        df = self._drop_unused(df.copy())

        # impute numeric
        for col, median_val in self._numeric_medians.items():
            if col in df.columns:
                n_missing = df[col].isna().sum()
                if n_missing:
                    logger.debug("Imputing %d missing values in '%s'.", n_missing, col)
                df[col] = df[col].fillna(median_val).astype(np.float64)

        # encode categoricals
        for col, le in self._label_encoders.items():
            if col in df.columns:
                vals = df[col].fillna("__MISSING__").astype(str)
                # handle unseen categories gracefully
                known = set(le.classes_)
                vals = vals.map(lambda v: v if v in known else "__MISSING__")
                df[col] = le.transform(vals).astype(np.int32)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, directory: Path | None = None) -> Path:
        """Persist fitted encoder / imputation state as JSON."""
        out_dir = directory or ARTIFACTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "preprocessor_state.json"

        state = {
            "numeric_medians": self._numeric_medians,
            "label_encoders": {
                col: le.classes_.tolist() for col, le in self._label_encoders.items()
            },
        }
        path.write_text(json.dumps(state, indent=2))
        logger.info("Preprocessor state saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        """Reconstruct a fitted Preprocessor from saved JSON state."""
        state = json.loads(path.read_text())
        pp = cls()
        pp._numeric_medians = state["numeric_medians"]
        for col, classes in state["label_encoders"].items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            pp._label_encoders[col] = le
        pp._is_fitted = True
        logger.info("Preprocessor loaded from %s", path)
        return pp

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _drop_unused(df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        return df.drop(columns=cols_to_drop)

    @staticmethod
    def get_feature_columns() -> list[str]:
        """Return the ordered list of feature columns expected by the model."""
        return NUMERIC_FEATURES + CATEGORICAL_FEATURES

    @staticmethod
    def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split a preprocessed DataFrame into features (X) and target (y)."""
        feature_cols = Preprocessor.get_feature_columns()
        available = [c for c in feature_cols if c in df.columns]
        X = df[available]
        y = df[TARGET_COL] if TARGET_COL in df.columns else pd.Series(dtype=np.int32)
        return X, y
