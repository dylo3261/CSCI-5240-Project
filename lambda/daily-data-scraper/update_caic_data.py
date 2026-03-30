import os, sys, time, json, logging, requests
import pandas as pd
from datetime import datetime, timedelta
from utils.s3_helpers import (
    IS_LAMBDA, S3_CAIC_CLEAN_KEY,
    download_from_s3, upload_to_s3, copy_s3_object, s3_archive_key,
)
from utils.process_caic import load_caic_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "/tmp/data" if IS_LAMBDA else os.path.join(BASE_DIR, "data")
CLEAN_FILE = os.path.join(OUTPUT_DIR, "daily_caic_data.csv")

# Archive filename (just the basename, used to build S3 keys)
CAIC_BASENAME = "daily_caic_data.csv"

API_BASE = "https://api.avalanche.state.co.us/api/v2/observation_reports"
PER_PAGE = 250
CHUNK_DAYS = 30
EARLIEST_DATE = "2016-01-01"
MAX_RETRIES = 3


def clean_text(text):
    if isinstance(text, str):
        return text.replace("\r", " ").replace("\n", " ").strip()
    return text


def flatten_report(report: dict) -> dict:
    zone_title = (report.get("backcountry_zone") or {}).get("title")
    if not zone_title:
        zone_title = (report.get("highway_zone") or {}).get("title")

    row = {
        "Observation ID": report.get("id"),
        "Date": report.get("observed_at"),
        "Date Known": report.get("date_known"),
        "Landmark": report.get("landmark"),
        "First Name": report.get("firstname"),
        "Longitude": report.get("longitude"),
        "latitude": report.get("latitude"),
        "Last Name": report.get("lastname"),
        "Area": report.get("area"),
        "Location": zone_title,
        "Description": clean_text(report.get("description")),
        "Comments": clean_text(report.get("comments_caic")),
        "Status": report.get("status"),
        "Locked": report.get("is_locked"),
    }

    aval_obs = report.get("avalanche_observations") or []
    if aval_obs:
        a = aval_obs[0]
        row.update({
            "#": a.get("number"), "Elevation": a.get("elevation"),
            "Aspect": a.get("aspect"), "Type": a.get("type_code"),
            "Trigger": a.get("primary_trigger"),
            "Secondary Trigger": a.get("secondary_trigger"),
            "Relative Size": a.get("relative_size"),
            "Destructive Size": a.get("destructive_size"),
            "Incident": a.get("is_incident"), "Sliding Sfc": a.get("surface"),
            "Weak Layer": a.get("weak_layer"),
            "Avg Width": a.get("width_average"), "Width Units": a.get("width_units"),
            "Avg Vertical": a.get("vertical_average"), "Vertical Units": a.get("vertical_units"),
            "Avg Crown": a.get("crown_average"), "Crown Units": a.get("crown_units"),
            "Terminus": a.get("terminus"),
        })
    else:
        for k in ["#", "Elevation", "Aspect", "Type", "Trigger", "Secondary Trigger",
                   "Relative Size", "Destructive Size", "Incident", "Sliding Sfc",
                   "Weak Layer", "Avg Width", "Width Units", "Avg Vertical",
                   "Vertical Units", "Avg Crown", "Crown Units", "Terminus"]:
            row[k] = None
    return row


def fetch_window(start_dt: datetime, end_dt: datetime) -> list[dict]:
    start_iso = start_dt.strftime("%Y-%m-%dT06:00:00.000Z")
    end_iso = end_dt.strftime("%Y-%m-%dT05:59:59.999Z")
    all_rows, page = [], 1
    while True:
        params = {
            "page": page, "per": PER_PAGE,
            "r[observed_at_gteq]": start_iso, "r[observed_at_lteq]": end_iso,
            "r[sorts][]": ["observed_at desc", "created_at desc"],
        }
        resp = requests.get(API_BASE, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(flatten_report(r) for r in data)
        if len(data) < PER_PAGE:
            break
        page += 1
        time.sleep(0.5)
    return all_rows


def _fetch_reports_for_window(start: datetime, end: datetime) -> list[dict]:
    """Fetch CAIC reports for a date window with retry logic."""
    logger.info(f"  {start.date()} → {end.date()}")
    retries = 0
    while retries < MAX_RETRIES:
        try:
            rows = fetch_window(start, end)
            logger.info(f"    {len(rows)} reports")
            return rows
        except requests.exceptions.RequestException as e:
            retries += 1
            if retries >= MAX_RETRIES:
                logger.error(f"  Failed after {MAX_RETRIES} retries for {start.date()}→{end.date()}: {e} — skipping window")
                return []
            logger.warning(f"  Error (attempt {retries}/{MAX_RETRIES}): {e} — retrying in 10s")
            time.sleep(10)
    return []


def _fetch_new_reports(start: datetime, end: datetime = None) -> list[dict]:
    """Fetch all new CAIC reports from start to end (default: today) in chunks."""
    if end is None:
        end = datetime.now()
    if start >= end:
        return []
    logger.info(f"Fetching reports from {start.date()} to {end.date()} ...")
    new_rows = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        rows = _fetch_reports_for_window(cur, chunk_end)
        new_rows.extend(rows)
        cur = chunk_end + timedelta(days=1)
        time.sleep(1)
    logger.info(f"Fetched {len(new_rows)} new reports total.")
    return new_rows


def _clean_and_save(rows: list[dict], upload_key: str = None) -> pd.DataFrame:
    """Clean raw rows, save to local CSV, and optionally upload."""
    if not rows:
        return pd.DataFrame()
    new_clean = load_caic_data(pd.DataFrame(rows), natural_only=True)

    if "Observation ID" in new_clean.columns:
        new_clean.drop_duplicates(subset=["Observation ID"], inplace=True)
    if "Date" in new_clean.columns:
        new_clean["Date"] = pd.to_datetime(new_clean["Date"], errors="coerce")
        new_clean.sort_values("Date", ascending=False, inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    new_clean.to_csv(CLEAN_FILE, index=False)
    logger.info(f"Saved {len(new_clean)} cleaned rows to {CLEAN_FILE}")

    if upload_key:
        upload_to_s3(CLEAN_FILE, upload_key)
        logger.info(f"Uploaded {CLEAN_FILE} to S3 key: {upload_key}")

    return new_clean


def _archive_latest() -> None:
    """
    Archive the current /latest/ file to a /YYYY-MM-DD/ folder in S3.
    Uses yesterday's date since the Lambda runs daily and /latest/ holds
    the previous day's data.
    """
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    archive_key = s3_archive_key(yesterday_str, CAIC_BASENAME)
    logger.info(f"Archiving latest CAIC data → {archive_key}")
    copy_s3_object(S3_CAIC_CLEAN_KEY, archive_key)


# ── Lambda daily update ──────────────────────────────────────────────
def update_data() -> int:
    """
    Daily update (Lambda):
    1. Download current /latest/ file from S3
    2. Archive it to /YYYY-MM-DD/ folder
    3. Fetch today's new reports
    4. Save single-day CSV and upload to /latest/
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1 & 2: Download current /latest/ and archive it
    has_latest = download_from_s3(S3_CAIC_CLEAN_KEY, CLEAN_FILE)
    if has_latest:
        _archive_latest()

    # Step 3: Determine fetch window — just today's reports
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    # Fetch from yesterday 6AM to today (the CAIC API uses 6AM boundaries)
    new_rows = _fetch_new_reports(yesterday, today)

    # Step 4: Save single-day CSV and upload to /latest/
    result_df = _clean_and_save(new_rows, upload_key=S3_CAIC_CLEAN_KEY)
    return len(result_df)


# ── Full local fetch ─────────────────────────────────────────────────
def fetch_all():
    """Local usage: fetches all historical data and appends to local CSV. No S3 upload."""
    existing = pd.read_csv(CLEAN_FILE) if os.path.exists(CLEAN_FILE) else pd.DataFrame()

    # Determine start date
    start = datetime.strptime(EARLIEST_DATE, "%Y-%m-%d")
    if not existing.empty and "Date" in existing.columns:
        existing["Date"] = pd.to_datetime(existing["Date"], errors="coerce", utc=True)
        last = existing["Date"].max()
        if pd.notna(last):
            start = last.to_pydatetime().replace(tzinfo=None) + timedelta(days=1)

    new_rows = _fetch_new_reports(start)
    if not new_rows and not existing.empty:
        print("Data is already up-to-date.")
        return len(existing)

    # Merge with existing for local multi-day file
    if new_rows:
        new_clean = load_caic_data(pd.DataFrame(new_rows), natural_only=True)
        combined = pd.concat([existing, new_clean], ignore_index=True) if not existing.empty else new_clean
    else:
        combined = existing

    if "Observation ID" in combined.columns:
        combined.drop_duplicates(subset=["Observation ID"], inplace=True)
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
        combined.sort_values("Date", ascending=False, inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined.to_csv(CLEAN_FILE, index=False)
    print(f"Saved {len(combined)} cleaned rows to {CLEAN_FILE}")
    return len(combined)


def lambda_handler(event, context):
    logger.info("Lambda: starting CAIC update...")
    try:
        count = update_data()
        return {"statusCode": 200, "body": json.dumps({"message": "OK", "rows": count})}
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--lambda":
        lambda_handler(None, None)
    else:
        fetch_all()
