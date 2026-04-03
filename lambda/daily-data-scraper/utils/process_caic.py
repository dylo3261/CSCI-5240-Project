import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_caic_data(source, natural_only=True):
    """
    Load and clean CAIC avalanche data.

    Parameters
    ----------
    source : str, Path, or pd.DataFrame
        Either a path to a CSV file or an already-loaded DataFrame.
    natural_only : bool
        If True, keep only natural (N), unknown (U), or missing triggers.

    Returns
    -------
    pd.DataFrame  —  cleaned, filtered avalanche observations.
    """
    # Accept DataFrame or filepath
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    # Strip whitespace from columns that actually contain strings
    # (some object-dtype columns, e.g. booleans, are not strings)
    for col in df.select_dtypes(include=["object"]).columns:
        first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if isinstance(first_valid, str):
            df[col] = df[col].str.strip()

    # Replace placeholder values with NaN
    df = df.replace(["-", ""], np.nan).infer_objects(copy=False)

    # Filter to valid coordinates
    df = df.dropna(subset=["Longitude", "latitude"])
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df = df.dropna(subset=["Longitude", "latitude"])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Extract avalanche size from Destructive Size (e.g. "D2" → 2.0)
    if "Destructive Size" in df.columns:
        df["avalanche_size"] = df["Destructive Size"].str.extract(r"D(\d+\.?\d*)")[0]
        df["avalanche_size"] = pd.to_numeric(df["avalanche_size"], errors="coerce")
    else:
        df["avalanche_size"] = np.nan

    # Keep useful columns
    keep_cols = [
        "Observation ID", "Date", "Longitude", "latitude",
        "Aspect", "Area", "avalanche_size", "Type", "Trigger",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_final = df[keep_cols].copy()

    # Add binary label
    df_final["avalanche_occurred"] = 1

    # Drop rows missing critical fields
    df_final = df_final.dropna(subset=["Date", "Longitude", "latitude", "avalanche_size"])

    # Filter by trigger type
    logger.info(f"Trigger distribution before filter:\n{df_final['Trigger'].value_counts(dropna=False)}")

    if natural_only:
        before_count = len(df_final)
        df_final = df_final[
            df_final["Trigger"].isin(["N", "U", "AS"]) | df_final["Trigger"].isna()
        ]
        after_count = len(df_final)
        logger.info(
            f"Filtered to natural/unknown/skier triggers: {before_count:,} → {after_count:,} "
            f"({before_count - after_count:,} removed)"
        )

    logger.info(f"Loaded {len(df_final):,} avalanche observations")
    if "Date" in df_final.columns and not df_final.empty:
        logger.info(f"Date range: {df_final['Date'].min()} to {df_final['Date'].max()}")

    return df_final


# ── Standalone CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(BASE_DIR, "data", "caic_observation_reports.csv")
    out_path = os.path.join(BASE_DIR, "data", "caic_clean.csv")

    # Enable console logging for standalone mode
    logging.basicConfig(level=logging.INFO)

    df = load_caic_data(raw_path, natural_only=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")