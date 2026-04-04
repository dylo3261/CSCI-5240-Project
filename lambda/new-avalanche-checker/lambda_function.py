import os
import sys
import json
import time
import uuid
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO

import boto3
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

# S3
S3_BUCKET = os.environ.get("AWS_BUCKET", "daily-weather-data-csv-bucket")
S3_CAIC_LATEST_KEY = "latest/daily_caic_data.csv"

# DynamoDB
REACTIONS_TABLE_NAME = os.environ.get("REACTIONS_TABLE", "UserReactionsTable")

# Lambda environment detection
IS_LAMBDA = os.environ.get("AWS_EXECUTION_ENV", "").startswith("AWS_Lambda")

# Local temp storage
OUTPUT_DIR = "/tmp/data" if IS_LAMBDA else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)
LOCAL_CSV = os.path.join(OUTPUT_DIR, "daily_caic_data.csv")

# CAIC API
API_BASE = "https://api.avalanche.state.co.us/api/v2/observation_reports"
PER_PAGE = 250
MAX_RETRIES = 3
LOOKBACK_HOURS = 2  # overlap window — much wider than 15 min for safety

def _clean_text(text):
    if isinstance(text, str):
        return text.replace("\r", " ").replace("\n", " ").strip()
    return text


def _flatten_report(report: dict) -> dict:
    """Flatten a single CAIC API report into a flat dict."""
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
        "Description": _clean_text(report.get("description")),
        "Comments": _clean_text(report.get("comments_caic")),
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


def _fetch_window(start_dt: datetime, end_dt: datetime) -> list[dict]:
    """Fetch all CAIC reports created in a time window (paginated)."""
    start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S.999Z")
    all_rows, page = [], 1

    while True:
        params = {
            "page": page, "per": PER_PAGE,
            "r[created_at_gteq]": start_iso,
            "r[created_at_lteq]": end_iso,
            "r[sorts][]": ["observed_at desc", "created_at desc"],
        }
        resp = requests.get(API_BASE, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(_flatten_report(r) for r in data)
        if len(data) < PER_PAGE:
            break
        page += 1
        time.sleep(0.5)

    return all_rows


def _fetch_recent_reports(lookback_hours: int = LOOKBACK_HOURS) -> list[dict]:
    """Fetch CAIC reports from the last N hours with retry logic."""
    now = datetime.utcnow()
    start = now - timedelta(hours=lookback_hours)

    logger.info(f"Fetching CAIC reports from {start.isoformat()} to {now.isoformat()} "
                f"({lookback_hours}h lookback)")

    retries = 0
    while retries < MAX_RETRIES:
        try:
            rows = _fetch_window(start, now)
            logger.info(f"Fetched {len(rows)} raw reports from CAIC API")
            return rows
        except requests.exceptions.RequestException as e:
            retries += 1
            if retries >= MAX_RETRIES:
                logger.error(f"CAIC API failed after {MAX_RETRIES} retries: {e}")
                raise
            logger.warning(f"CAIC API error (attempt {retries}/{MAX_RETRIES}): {e} — retrying in 5s")
            time.sleep(5)

    return []


def _clean_reports(rows: list[dict]) -> pd.DataFrame:
    """
    Clean raw CAIC rows — mirrors the logic from process_caic.load_caic_data
    with natural_only=True filtering.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if isinstance(first_valid, str):
            df[col] = df[col].str.strip()

    # Replace placeholder values
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

    # Natural / unknown / skier triggers only
    before = len(df_final)
    df_final = df_final[
        df_final["Trigger"].isin(["N", "U", "AS"]) | df_final["Trigger"].isna()
    ]
    logger.info(f"Trigger filter: {before} → {len(df_final)} rows "
                f"({before - len(df_final)} non-natural/unknown/skier removed)")

    return df_final

def _download_csv_from_s3() -> pd.DataFrame:
    """Download the latest daily CAIC CSV from S3 and return as DataFrame."""
    s3 = boto3.client("s3")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        s3.download_file(S3_BUCKET, S3_CAIC_LATEST_KEY, LOCAL_CSV)
        logger.info(f"Downloaded s3://{S3_BUCKET}/{S3_CAIC_LATEST_KEY}")
        return pd.read_csv(LOCAL_CSV)
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.warning(f"No existing CSV at s3://{S3_BUCKET}/{S3_CAIC_LATEST_KEY} "
                           "— starting with empty set")
            return pd.DataFrame()
        raise


def _upload_csv_to_s3(df: pd.DataFrame) -> None:
    """Upload an updated DataFrame back to S3 as the latest CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(LOCAL_CSV, index=False)

    s3 = boto3.client("s3")
    s3.upload_file(LOCAL_CSV, S3_BUCKET, S3_CAIC_LATEST_KEY)
    logger.info(f"Uploaded updated CSV ({len(df)} rows) → "
                f"s3://{S3_BUCKET}/{S3_CAIC_LATEST_KEY}")


def _insert_avalanche_pin(row: pd.Series) -> dict:
    """
    Insert a single avalanche reaction pin into UserReactionsTable.
    Schema matches websocket-handler's _send_reaction() format.
    """
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(REACTIONS_TABLE_NAME)

    reaction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # Build descriptive message
    area = row.get("Area", "Unknown Area")
    aval_size = row.get("avalanche_size", "?")
    aval_type = row.get("Type", "")
    message = f"CAIC Report: {area} - D{aval_size} {aval_type}".strip()

    item = {
        "reactionId":    reaction_id,
        "dataType":      "REACTION",
        "timestamp":     timestamp,
        "reactionType":  "avalanche",
        "message":       message,
        "latitude":      Decimal(str(row["latitude"])),
        "longitude":     Decimal(str(row["Longitude"])),
        "userId":        "CAIC_SYSTEM",
        "observationId": str(row["Observation ID"]),
    }

    table.put_item(Item=item)
    logger.info(f"  Inserted pin {reaction_id}: {message} "
                f"@ ({row['latitude']}, {row['Longitude']})")
    return item

def check_for_new_avalanches() -> dict:
    """
    Core logic:
      1. Download current CSV from S3 → known observation IDs
      2. Fetch last 2h of CAIC reports → clean → new observation IDs
      3. Diff: new_ids = fetched - known
      4. Insert avalanche pins for each new observation
      5. Append new rows to CSV and re-upload to S3
    """
    # Step 1: Get known observation IDs from S3
    existing_df = _download_csv_from_s3()

    if existing_df.empty or "Observation ID" not in existing_df.columns:
        known_ids = set()
    else:
        # Ensure IDs are comparable as strings
        known_ids = set(existing_df["Observation ID"].dropna().astype(str))

    logger.info(f"Known observation IDs in S3: {len(known_ids)}")

    # Step 2: Fetch recent reports from CAIC API
    raw_rows = _fetch_recent_reports()
    cleaned_df = _clean_reports(raw_rows)

    if cleaned_df.empty:
        logger.info("No cleaned reports from CAIC API — nothing to do.")
        return {"new_pins": 0, "total_known": len(known_ids)}

    # Step 3: Identify truly new observations
    cleaned_df["Observation ID"] = cleaned_df["Observation ID"].astype(str)
    new_df = cleaned_df[~cleaned_df["Observation ID"].isin(known_ids)]

    if new_df.empty:
        logger.info("All fetched observations already known — no new avalanches.")
        return {"new_pins": 0, "total_known": len(known_ids)}

    logger.info(f"Found {len(new_df)} NEW avalanche observation(s)!")

    # Step 4: Insert DynamoDB pins for each new observation
    pins_inserted = 0
    for _, row in new_df.iterrows():
        try:
            _insert_avalanche_pin(row)
            pins_inserted += 1
        except Exception as e:
            logger.error(f"Failed to insert pin for Obs ID {row['Observation ID']}: {e}")

    # Step 5: Append new rows to existing CSV and re-upload
    if not existing_df.empty:
        # Ensure matching dtypes for concat
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.drop_duplicates(subset=["Observation ID"], inplace=True)
    else:
        updated_df = new_df.copy()

    if "Date" in updated_df.columns:
        updated_df["Date"] = pd.to_datetime(updated_df["Date"], errors="coerce")
        updated_df.sort_values("Date", ascending=False, inplace=True)

    _upload_csv_to_s3(updated_df)

    result = {
        "new_pins": pins_inserted,
        "new_observations": len(new_df),
        "total_known": len(updated_df),
    }
    logger.info(f"Done: {result}")
    return result


# ── Lambda entry point ───────────────────────────────────────────────

def lambda_handler(event, context):
    """AWS Lambda handler — invoked every 15 minutes by EventBridge."""
    logger.info("new-avalanche-checker: starting...")
    try:
        result = check_for_new_avalanches()
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "OK", **result}),
        }
    except Exception as e:
        logger.error(f"new-avalanche-checker FAILED: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }


# ── Local CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(lambda_handler(None, None), indent=2))
