"""Inference module — load a trained model + preprocessor and predict avalanche size."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import ARTIFACTS_DIR, CHECKPOINT_DIR
from src.utils.preprocess import Preprocessor

logger = logging.getLogger(__name__)


class AvalanchePredictor:
    """Encapsulates model + preprocessor for single-row or batch inference."""

    def __init__(
        self,
        model_path: Path | None = None,
        preprocessor_path: Path | None = None,
    ) -> None:
        model_path = model_path or CHECKPOINT_DIR / "xgb_avalanche.json"
        preprocessor_path = preprocessor_path or ARTIFACTS_DIR / "preprocessor_state.json"

        logger.info("Loading model from %s", model_path)
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        logger.info("Loading preprocessor from %s", preprocessor_path)
        self.preprocessor = Preprocessor.load(preprocessor_path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted avalanche sizes."""
        df_processed = self.preprocessor.transform(df)
        X, _ = Preprocessor.split_xy(df_processed)
        return self.model.predict(X)


def run_inference(input_path: str, output_path: str | None = None) -> pd.DataFrame:
    """Convenience function: load data, predict, optionally save results."""
    predictor = AvalanchePredictor()

    df = pd.read_csv(input_path)
    df["predicted_avalanche_size"] = predictor.predict(df)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Predictions saved to %s", out)

    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run avalanche size prediction inference.")
    parser.add_argument("input", help="Path to input CSV")
    parser.add_argument("-o", "--output", help="Path to save predictions CSV")
    args = parser.parse_args()

    run_inference(args.input, args.output)
