"""Data loading and train / validation / test splitting utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_DATA_PATH, TrainConfig

logger = logging.getLogger(__name__)


def load_csv(path: Path | None = None) -> pd.DataFrame:
    """Load the avalanche observations CSV into a DataFrame.

    Parameters
    ----------
    path : Path, optional
        Explicit path.  Falls back to the configured default.
    """
    path = path or DEFAULT_DATA_PATH
    logger.info("Loading data from %s …", path)
    df = pd.read_csv(path, parse_dates=["Date"])
    logger.info("Loaded %d rows × %d cols.", *df.shape)
    return df


def make_splits(
    df: pd.DataFrame,
    cfg: TrainConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train / val / test sets.

    Returns
    -------
    train, val, test : pd.DataFrame
    """
    cfg = cfg or TrainConfig()

    # first split: train+val vs. test
    train_val, test = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    # second split: train vs. val (fraction relative to train+val)
    val_frac = cfg.val_size / (1 - cfg.test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_frac,
        random_state=cfg.random_state,
    )

    logger.info(
        "Split sizes → train=%d  val=%d  test=%d", len(train), len(val), len(test)
    )
    return train, val, test
