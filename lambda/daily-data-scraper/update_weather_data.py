import os, sys, json, logging
import pandas as pd
from datetime import datetime, timedelta
from utils.s3_helpers import (
    IS_LAMBDA, S3_DAILY_STATION_KEY, S3_STATIONS_KEY,
    download_from_s3, upload_to_s3, copy_s3_object, s3_archive_key,
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

# Archive filename (just the basename, used to build S3 keys)
DAILY_STATION_BASENAME = "daily_station_data.csv"


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
        new_snow = 0.0
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
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "snow_depth": data["snow_depth"],
            "swe": data["swe"],
            "temp": data["temp"],
            "new_snow_24hr": new_snow,
        })

    return rows


def _load_csv(path: str) -> pd.DataFrame:
    """Load a CSV or return empty DataFrame if not found."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    return pd.DataFrame()


def _get_yesterday_depths(df: pd.DataFrame, yesterday_str: str) -> dict | None:
    """
    Extract a {station_id: snow_depth} mapping from a DataFrame for a given date.
    Returns None if no data exists for that date.
    """
    if df.empty:
        return None
    day_rows = df[df["date"] == yesterday_str]
    if day_rows.empty:
        return None
    return dict(zip(day_rows["station_id"], day_rows["snow_depth"]))


def _save_single_day(rows: list[dict], upload: bool = False) -> pd.DataFrame:
    """Build a single-day DataFrame, clean it, save locally, and optionally upload."""
    df = pd.DataFrame(rows)
    df.dropna(subset=["snow_depth", "swe", "temp"], inplace=True)
    float_cols = df.select_dtypes(include="number").columns
    df[float_cols] = df[float_cols].round(3)
    df.sort_values("station_id", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DAILY_STATION_FILE, index=False)
    logger.info(f"Saved {len(df)} rows to {DAILY_STATION_FILE}")

    if upload:
        upload_to_s3(DAILY_STATION_FILE, S3_DAILY_STATION_KEY)

    return df


def _archive_latest() -> None:
    """
    Archive the current /latest/ file to a /YYYY-MM-DD/ folder in S3.
    Uses yesterday's date since the Lambda runs daily and /latest/ holds
    the previous day's data.
    """
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    archive_key = s3_archive_key(yesterday_str, DAILY_STATION_BASENAME)
    logger.info(f"Archiving latest weather data → {archive_key}")
    copy_s3_object(S3_DAILY_STATION_KEY, archive_key)


# ── Lambda daily update ──────────────────────────────────────────────
def update_daily_station_data() -> int:
    """
    Daily update (Lambda):
    1. Download current /latest/ file from S3
    2. Archive it to /YYYY-MM-DD/ folder
    3. Download yesterday's archive to compute new_snow_24hr
    4. Fetch today's data for all stations
    5. Save single-day CSV and upload to /latest/
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    download_from_s3(S3_STATIONS_KEY, STATIONS_FILE)
    stations_df = pd.read_csv(STATIONS_FILE)

    today = datetime.now()
    yesterday = today - timedelta(days=1)
    today_str = today.strftime("%Y-%m-%d")
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    # Step 1 & 2: Download current /latest/ and archive it
    has_latest = download_from_s3(S3_DAILY_STATION_KEY, DAILY_STATION_FILE)
    if has_latest:
        latest_df = _load_csv(DAILY_STATION_FILE)
        _archive_latest()
    else:
        latest_df = pd.DataFrame()

    # Step 3: Get yesterday's depths for new_snow_24hr calculation
    # Try the current latest first (it may be yesterday's data)
    yesterday_depths = _get_yesterday_depths(latest_df, yesterday_str)

    if yesterday_depths is None:
        # Try downloading from the archive
        yesterday_archive_key = s3_archive_key(yesterday_str, DAILY_STATION_BASENAME)
        yesterday_local = os.path.join(DATA_DIR, "yesterday_station_data.csv")
        if download_from_s3(yesterday_archive_key, yesterday_local):
            yesterday_df = _load_csv(yesterday_local)
            yesterday_depths = _get_yesterday_depths(yesterday_df, yesterday_str)

    if yesterday_depths is None:
        # No yesterday data available — fetch it, archive it, then use for depth calc
        logger.info(f"No yesterday data found. Fetching yesterday ({yesterday_str}) ...")
        yesterday_rows = _build_rows_for_date(stations_df, yesterday)
        yesterday_depths = {r["station_id"]: r["snow_depth"] for r in yesterday_rows}
        # Save and archive yesterday as its own single-day file
        yesterday_df = pd.DataFrame(yesterday_rows)
        yesterday_df.dropna(subset=["snow_depth", "swe", "temp"], inplace=True)
        yesterday_local = os.path.join(DATA_DIR, "yesterday_station_data.csv")
        yesterday_df.to_csv(yesterday_local, index=False)
        yesterday_archive_key = s3_archive_key(yesterday_str, DAILY_STATION_BASENAME)
        upload_to_s3(yesterday_local, yesterday_archive_key)

    # Step 4 & 5: Fetch today, save single-day CSV, upload to /latest/
    logger.info(f"Fetching today ({today_str}) for all stations ...")
    today_rows = _build_rows_for_date(stations_df, today, yesterday_depths)
    result_df = _save_single_day(today_rows, upload=IS_LAMBDA)
    return len(result_df)


# ── Full local rebuild ───────────────────────────────────────────────
def fetch_all_station_data() -> pd.DataFrame | None:
    """
    Local usage: fetch today's data (and yesterday if needed).
    Does not upload to S3. Saves multi-day appended CSV locally for convenience.
    """
    stations_df = pd.read_csv(STATIONS_FILE)
    existing_df = _load_csv(DAILY_STATION_FILE)

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

    # Local: append to existing multi-day CSV
    new_df = pd.DataFrame(all_new_rows)
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.drop_duplicates(subset=["date", "station_id"], keep="last", inplace=True)
    combined.dropna(subset=["snow_depth", "swe", "temp"], inplace=True)
    float_cols = combined.select_dtypes(include="number").columns
    combined[float_cols] = combined[float_cols].round(3)
    combined.sort_values(["date", "station_id"], ascending=[False, True], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_csv(DAILY_STATION_FILE, index=False)
    logger.info(f"Saved {len(combined)} rows to {DAILY_STATION_FILE}")

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