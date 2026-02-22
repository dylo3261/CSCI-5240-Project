import os, sys, json, logging
import pandas as pd
from datetime import datetime, timedelta
from utils.s3_helpers import (
    IS_LAMBDA, S3_DAILY_STATION_KEY, S3_STATIONS_KEY,
    download_from_s3, upload_to_s3,
)
from utils.weather_utils import get_station_weather

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/tmp/data" if IS_LAMBDA else os.path.join(BASE_DIR, "data")
DAILY_STATION_FILE = os.path.join(DATA_DIR, "daily_station_data.csv")
STATIONS_FILE = os.path.join(DATA_DIR, "snotel_stations_const.csv")


def fetch_station_day(station_triplet: str, date: datetime) -> dict:
    """Fetch snow_depth, swe, temp for a single station on a single date."""
    data = get_station_weather(station_triplet, date)
    return {
        "snow_depth": data.get("snow_depth"),
        "swe": data.get("swe"),
        "temp": data.get("temp"),
    }


def _build_rows_for_date(stations_df: pd.DataFrame, date: datetime,
                         yesterday_depths: dict = None) -> list[dict]:
    """
    Query every station for a given date and compute new_snow_24hr.

    Args:
        stations_df: SNOTEL stations DataFrame
        date: Date to fetch
        yesterday_depths: dict mapping station_id -> yesterday's snow_depth (cm).
                          If None, new_snow_24hr will be None for all stations.

    Returns:
        List of row dicts ready for DataFrame construction.
    """
    date_str = date.strftime("%Y-%m-%d")
    rows = []

    for idx, station in stations_df.iterrows():
        sid = station["station_id"]
        triplet = station["station_triplet"]
        name = station["name"]

        if idx % 20 == 0:
            logger.info(f"  Fetching station {idx + 1}/{len(stations_df)} ({name}) for {date_str} ...")

        data = fetch_station_day(triplet, date)

        # Calculate new_snow_24hr
        new_snow = None
        if yesterday_depths is not None and sid in yesterday_depths:
            prev = yesterday_depths[sid]
            curr = data["snow_depth"]
            if prev is not None and curr is not None:
                new_snow = curr - prev

        rows.append({
            "date": date_str,
            "station_id": sid,
            "station_triplet": triplet,
            "station_name": name,
            "snow_depth": data["snow_depth"],
            "swe": data["swe"],
            "temp": data["temp"],
            "new_snow_24hr": new_snow,
        })

    return rows


def _load_existing() -> pd.DataFrame:
    """Load the existing daily_station_data.csv or return empty DataFrame."""
    if os.path.exists(DAILY_STATION_FILE):
        df = pd.read_csv(DAILY_STATION_FILE)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    return pd.DataFrame()


def _get_yesterday_depths(existing_df: pd.DataFrame, yesterday_str: str) -> dict | None:
    """
    Extract a {station_id: snow_depth} mapping from existing data for a given date.
    Returns None if no data exists for that date.
    """
    if existing_df.empty:
        return None
    day_rows = existing_df[existing_df["date"] == yesterday_str]
    if day_rows.empty:
        return None
    return dict(zip(day_rows["station_id"], day_rows["snow_depth"]))


def _merge_and_save(existing_df: pd.DataFrame, new_rows: list[dict],
                    upload: bool = False) -> pd.DataFrame:
    """Append new rows, deduplicate (date + station_id), sort, and save."""
    new_df = pd.DataFrame(new_rows)
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.drop_duplicates(subset=["date", "station_id"], keep="last", inplace=True)
    combined.sort_values(["date", "station_id"], ascending=[False, True], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_csv(DAILY_STATION_FILE, index=False)
    logger.info(f"Saved {len(combined)} rows to {DAILY_STATION_FILE}")

    if upload:
        upload_to_s3(DAILY_STATION_FILE, S3_DAILY_STATION_KEY)

    return combined


# ── Lambda daily update ──────────────────────────────────────────────
def update_daily_station_data() -> int:
    """
    Daily update: fetch today's data for all stations.
    - First run (no existing CSV): fetch yesterday + today, compute new_snow_24hr.
    - Subsequent runs: fetch only today, use yesterday from existing CSV.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if IS_LAMBDA:
        download_from_s3(S3_DAILY_STATION_KEY, DAILY_STATION_FILE)
        download_from_s3(S3_STATIONS_KEY, STATIONS_FILE)

    stations_df = pd.read_csv(STATIONS_FILE)
    existing_df = _load_existing()

    today = datetime.now()
    yesterday = today - timedelta(days=1)
    today_str = today.strftime("%Y-%m-%d")
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    all_new_rows = []

    if existing_df.empty or yesterday_str not in existing_df["date"].values:
        # First run or missing yesterday — fetch yesterday first
        logger.info(f"Fetching yesterday ({yesterday_str}) for all stations ...")
        yesterday_rows = _build_rows_for_date(stations_df, yesterday)
        all_new_rows.extend(yesterday_rows)
        # Build yesterday depths from the fresh fetch
        yesterday_depths = {r["station_id"]: r["snow_depth"] for r in yesterday_rows}
    else:
        # Subsequent run — use existing yesterday data
        yesterday_depths = _get_yesterday_depths(existing_df, yesterday_str)

    logger.info(f"Fetching today ({today_str}) for all stations ...")
    today_rows = _build_rows_for_date(stations_df, today, yesterday_depths)
    all_new_rows.extend(today_rows)

    combined = _merge_and_save(existing_df, all_new_rows, upload=IS_LAMBDA)
    return len(combined)


# ── Full local rebuild ───────────────────────────────────────────────
def fetch_all_station_data() -> pd.DataFrame | None:
    """
    Local usage: same logic as daily update but never uploads to S3.
    """
    stations_df = pd.read_csv(STATIONS_FILE)
    existing_df = _load_existing()

    today = datetime.now()
    yesterday = today - timedelta(days=1)
    today_str = today.strftime("%Y-%m-%d")
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    # Check if today is already in the data
    if not existing_df.empty and today_str in existing_df["date"].values:
        logger.info("Daily station data is already up-to-date for today.")
        return existing_df

    all_new_rows = []

    if existing_df.empty or yesterday_str not in existing_df["date"].values:
        logger.info(f"Fetching yesterday ({yesterday_str}) for all stations ...")
        yesterday_rows = _build_rows_for_date(stations_df, yesterday)
        all_new_rows.extend(yesterday_rows)
        yesterday_depths = {r["station_id"]: r["snow_depth"] for r in yesterday_rows}
    else:
        yesterday_depths = _get_yesterday_depths(existing_df, yesterday_str)

    logger.info(f"Fetching today ({today_str}) for all stations ...")
    today_rows = _build_rows_for_date(stations_df, today, yesterday_depths)
    all_new_rows.extend(today_rows)

    combined = _merge_and_save(existing_df, all_new_rows, upload=False)
    return combined


def lambda_handler(event, context):
    logger.info("Lambda: starting daily station weather update ...")
    try:
        count = update_daily_station_data()
        return {"statusCode": 200, "body": json.dumps({"message": "OK", "rows": count})}
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--lambda":
        lambda_handler(None, None)
    else:
        fetch_all_station_data()