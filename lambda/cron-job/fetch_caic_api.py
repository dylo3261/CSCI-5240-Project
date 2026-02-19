import os
import sys
import time
import json
import logging
import requests
import boto3
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

# Usage:
# Full historical fetch: python fetch_caic_api.py
# Resume from last saved progress: python fetch_caic_api.py --resume
# Update S3 csv with all new reports: python fetch_caic_api.py --lambda

# config

# Lambda environment detection & S3 config
IS_LAMBDA = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_KEY = os.environ.get("S3_KEY", "caic_observation_reports.csv")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if IS_LAMBDA:
    OUTPUT_DIR = "/tmp/data"
else:
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "api")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "caic_observation_reports.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, ".fetch_progress.json")

API_BASE = "https://api.avalanche.state.co.us/api/v2/observation_reports"
PER_PAGE = 250 # max per page the API allows
CHUNK_DAYS = 30 # size of each time window
SLEEP_BETWEEN_PAGES = 0.5 # seconds between page requests
SLEEP_BETWEEN_CHUNKS = 1 # seconds between time-window chunks
REQUEST_TIMEOUT = 60 # seconds

#EARLIEST_DATE = "1997-03-11"
EARLIEST_DATE = "2016-01-01" # Last 10 years

# S3 helpers (only used when running on Lambda)
def _download_from_s3(local_path: str):
    """Download the full CSV from S3 to a local path."""
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        s3.download_file(S3_BUCKET, S3_KEY, local_path)
        logger.info(f"Downloaded s3://{S3_BUCKET}/{S3_KEY} → {local_path}")
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.warning(f"s3://{S3_BUCKET}/{S3_KEY} not found — will create a new file.")
        else:
            raise

def _upload_to_s3(local_path: str):
    """Upload the full CSV from a local path to S3."""
    s3 = boto3.client("s3")
    s3.upload_file(local_path, S3_BUCKET, S3_KEY)
    logger.info(f"Uploaded {local_path} → s3://{S3_BUCKET}/{S3_KEY}")

# helper to clean text
def clean_text(text):
    if isinstance(text, str):
        return text.replace("\r", " ").replace("\n", " ").strip()
    return text

def flatten_report(report: dict) -> dict:
    """Extract the most useful fields from a single observation_report JSON."""

    # Extract zone title (Location)
    zone_title = (report.get("backcountry_zone") or {}).get("title")
    if not zone_title:
        zone_title = (report.get("highway_zone") or {}).get("title")

    row = {
        # ── Report-level ──
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

    # ── First avalanche observation (if any) ──
    aval_obs = report.get("avalanche_observations") or []
    if aval_obs:
        a = aval_obs[0]
        row.update({
            "#": a.get("number"),
            "Elevation": a.get("elevation"),
            "Aspect": a.get("aspect"),
            "Type": a.get("type_code"),
            "Trigger": a.get("primary_trigger"),
            "Secondary Trigger": a.get("secondary_trigger"),
            "Relative Size": a.get("relative_size"),
            "Destructive Size": a.get("destructive_size"),
            "Incident": a.get("is_incident"),
            "Sliding Sfc": a.get("surface"),
            "Weak Layer": a.get("weak_layer"),
            "Avg Width": a.get("width_average"),
            "Width Units": a.get("width_units"),
            "Avg Vertical": a.get("vertical_average"),
            "Vertical Units": a.get("vertical_units"),
            "Avg Crown": a.get("crown_average"),
            "Crown Units": a.get("crown_units"), # Might be null
            "Terminus": a.get("terminus"),
        })
    else:
        # Fill with None/empty if no avalanche observation so keys exist
        row.update({
            "#": None, "Elevation": None, "Aspect": None, "Type": None,
            "Trigger": None, "Secondary Trigger": None, "Relative Size": None,
            "Destructive Size": None, "Incident": None, "Sliding Sfc": None,
            "Weak Layer": None, "Avg Width": None, "Width Units": None,
            "Avg Vertical": None, "Vertical Units": None, "Avg Crown": None,
            "Crown Units": None, "Terminus": None
        })

    return row

def fetch_page(start_iso: str, end_iso: str, page: int) -> list[dict]:
    """Fetch a single page of observation_reports for a given time window."""
    params = {
        "page": page,
        "per": PER_PAGE,
        "r[observed_at_gteq]": start_iso,
        "r[observed_at_lteq]": end_iso,
        "r[sorts][]": ["observed_at desc", "created_at desc"],
    }
    resp = requests.get(API_BASE, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def fetch_window(start_dt: datetime, end_dt: datetime) -> list[dict]:
    """Paginate through all results for a single time window."""
    start_iso = start_dt.strftime("%Y-%m-%dT06:00:00.000Z")
    end_iso   = end_dt.strftime("%Y-%m-%dT05:59:59.999Z")
    all_rows = []
    page = 1

    while True:
        data = fetch_page(start_iso, end_iso, page)
        if not data:
            break
        for report in data:
            all_rows.append(flatten_report(report))
        if len(data) < PER_PAGE:
            break  # last page
        page += 1
        time.sleep(SLEEP_BETWEEN_PAGES)

    return all_rows

def save_progress(last_end: datetime, total_rows: int):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "last_window_end": last_end.isoformat(),
            "total_rows": total_rows,
            "updated_at": datetime.now().isoformat(),
        }, f)


def load_progress() -> datetime | None:
    if not os.path.exists(PROGRESS_FILE):
        return None
    with open(PROGRESS_FILE) as f:
        data = json.load(f)
    return datetime.fromisoformat(data["last_window_end"])


def fetch_all(resume: bool = False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    today = datetime.now()
    start = datetime.strptime(EARLIEST_DATE, "%Y-%m-%d")

    # Resume support
    if resume:
        last_end = load_progress()
        if last_end:
            start = last_end + timedelta(days=1)
            print(f"Resuming from {start.date()} (previous progress found)")
        else:
            print("No progress file found, starting from scratch")

    total_days = (today - start).days
    total_chunks = (total_days // CHUNK_DAYS) + (1 if total_days % CHUNK_DAYS else 0)
    chunk_num = 0
    all_rows = []

    # If resuming, load existing data
    if resume and os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        all_rows = existing_df.to_dict("records")
        print(f"Loaded {len(all_rows)} existing rows from {OUTPUT_FILE}")

    current_start = start
    while current_start < today:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), today)
        chunk_num += 1

        label = f"{current_start.date()} → {current_end.date()}"
        print(f"\n[{chunk_num}/{total_chunks}] Fetching {label} ...", end=" ", flush=True)

        try:
            rows = fetch_window(current_start, current_end)
            all_rows.extend(rows)
            print(f"got {len(rows)} reports  (total: {len(all_rows)})")
        except requests.exceptions.HTTPError as e:
            print(f"\n  HTTP error: {e}")
            print("  Saving progress and stopping.")
            save_progress(current_start - timedelta(days=1), len(all_rows))
            break
        except requests.exceptions.RequestException as e:
            print(f"\n  Request error: {e}")
            print("  Retrying in 10 seconds...")
            time.sleep(10)
            continue  # retry same chunk

        save_progress(current_end, len(all_rows))
        current_start = current_end + timedelta(days=1)
        time.sleep(SLEEP_BETWEEN_CHUNKS)

    # Build and save CSV
    if all_rows:
        df = pd.DataFrame(all_rows)

        # Drop duplicates by Observation ID
        if "Observation ID" in df.columns:
            before = len(df)
            df.drop_duplicates(subset=["Observation ID"], inplace=True)
            after = len(df)
            if before != after:
                print(f"\nDropped {before - after} duplicate report(s)")

        # Sort by Date descending
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.sort_values("Date", ascending=False, inplace=True)
            
        
        # Desired column order
        desired_columns = [
            "Observation ID", "Date", "Date Known", "Landmark", "First Name", 
            "Longitude", "latitude", "Last Name", "#", "Elevation", "Aspect", 
            "Type", "Trigger", "Secondary Trigger", "Relative Size", 
            "Destructive Size", "Incident", "Area", "Location", "Description", 
            "Comments", "Sliding Sfc", "Status", "Locked", "Weak Layer", 
            "Avg Width", "Width Units", "Avg Vertical", "Vertical Units", 
            "Avg Crown", "Crown Units", "Terminus"
        ]
        
        # Filter and reorder columns, keeping only what's available and requested
        final_cols = [c for c in desired_columns if c in df.columns]
        df = df[final_cols]

        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n{'='*60}")
        print(f"Saved {len(df)} observation reports to {OUTPUT_FILE}")
        if "Date" in df.columns:
            print(f"Date range: {df['Date'].min()} → {df['Date'].max()}")
        print(f"{'='*60}")
    else:
        print("\nNo data fetched.")

    return len(all_rows)

def update_data():
    """Download CSV from S3, fetch only new reports, merge, and re-upload."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _download_from_s3(OUTPUT_FILE)

    # Determine start date from existing CSV
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        existing_df["Date"] = pd.to_datetime(existing_df["Date"], errors="coerce")
        last_date = existing_df["Date"].max()
        if pd.notna(last_date):
            start = last_date.to_pydatetime() + timedelta(days=1)
            logger.info(f"Existing CSV has {len(existing_df)} rows, latest date: {last_date.date()}")
        else:
            start = datetime.strptime(EARLIEST_DATE, "%Y-%m-%d")
            logger.info("Existing CSV found but no valid dates — fetching from EARLIEST_DATE.")
    else:
        existing_df = pd.DataFrame()
        start = datetime.strptime(EARLIEST_DATE, "%Y-%m-%d")
        logger.info("No existing CSV found — fetching from EARLIEST_DATE.")

    today = datetime.now()
    if start >= today:
        logger.info("CSV is already up-to-date. Nothing to fetch.")
        _upload_to_s3(OUTPUT_FILE)
        return len(existing_df)

    # Fetch new data in chunks
    logger.info(f"Fetching new reports from {start.date()} to {today.date()} ...")
    new_rows = []
    current_start = start
    while current_start < today:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), today)
        label = f"{current_start.date()} → {current_end.date()}"
        logger.info(f"  Fetching {label} ...")
        try:
            rows = fetch_window(current_start, current_end)
            new_rows.extend(rows)
            logger.info(f"    got {len(rows)} reports")
        except requests.exceptions.RequestException as e:
            logger.error(f"  Request error on {label}: {e}")
            logger.info("  Retrying in 10 seconds...")
            time.sleep(10)
            continue  # retry same chunk
        current_start = current_end + timedelta(days=1)
        time.sleep(SLEEP_BETWEEN_CHUNKS)

    logger.info(f"Fetched {len(new_rows)} new reports total.")

    # Merge with existing data
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = existing_df

    # Deduplicate
    if "Observation ID" in combined_df.columns:
        before = len(combined_df)
        combined_df.drop_duplicates(subset=["Observation ID"], inplace=True)
        after = len(combined_df)
        if before != after:
            logger.info(f"Dropped {before - after} duplicate(s).")

    # Sort by Date descending
    if "Date" in combined_df.columns:
        combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")
        combined_df.sort_values("Date", ascending=False, inplace=True)

    # Reorder columns
    desired_columns = [
        "Observation ID", "Date", "Date Known", "Landmark", "First Name",
        "Longitude", "latitude", "Last Name", "#", "Elevation", "Aspect",
        "Type", "Trigger", "Secondary Trigger", "Relative Size",
        "Destructive Size", "Incident", "Area", "Location", "Description",
        "Comments", "Sliding Sfc", "Status", "Locked", "Weak Layer",
        "Avg Width", "Width Units", "Avg Vertical", "Vertical Units",
        "Avg Crown", "Crown Units", "Terminus"
    ]
    final_cols = [c for c in desired_columns if c in combined_df.columns]
    combined_df = combined_df[final_cols]

    # Save and upload
    combined_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved {len(combined_df)} total rows to {OUTPUT_FILE}")
    _upload_to_s3(OUTPUT_FILE)
    return len(combined_df)

def lambda_handler(event, context):
    """AWS Lambda entrypoint — triggered daily by EventBridge."""
    logger.info("Lambda invoked. Starting daily avalanche data update...")
    try:
        row_count = update_data()
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Update successful",
                "rows": row_count,
            }),
        }
    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--lambda":
        lambda_handler(None, None)
    elif arg == "--resume":
        fetch_all(resume=True)
    else:
        fetch_all()
