import os, sys, time, json, logging, requests
import pandas as pd
from datetime import datetime, timedelta
from utils.s3_helpers import IS_LAMBDA, S3_CAIC_CLEAN_KEY, download_from_s3, upload_to_s3
from utils.process_caic import load_caic_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "/tmp/data" if IS_LAMBDA else os.path.join(BASE_DIR, "data")
CLEAN_FILE = os.path.join(OUTPUT_DIR, "daily_caic_data.csv")

API_BASE = "https://api.avalanche.state.co.us/api/v2/observation_reports"
PER_PAGE = 250
CHUNK_DAYS = 30
EARLIEST_DATE = "2016-01-01"


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


def _get_start_date(df: pd.DataFrame) -> datetime:
    """Determine the fetch start date from an existing clean DataFrame."""
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        last = df["Date"].max()
        if pd.notna(last):
            return last.to_pydatetime().replace(tzinfo=None) + timedelta(days=1)
    return datetime.strptime(EARLIEST_DATE, "%Y-%m-%d")


def _fetch_new_reports(start: datetime) -> list[dict]:
    """Fetch all new CAIC reports from start to today in chunks."""
    today = datetime.now()
    if start >= today:
        return []
    logger.info(f"Fetching reports from {start.date()} to {today.date()} ...")
    new_rows = []
    cur = start
    while cur < today:
        end = min(cur + timedelta(days=CHUNK_DAYS - 1), today)
        logger.info(f"  {cur.date()} → {end.date()}")
        try:
            rows = fetch_window(cur, end)
            new_rows.extend(rows)
            logger.info(f"    {len(rows)} reports")
        except requests.exceptions.RequestException as e:
            logger.error(f"  Error: {e} — retrying in 10s")
            time.sleep(10)
            continue
        cur = end + timedelta(days=1)
        time.sleep(1)
    logger.info(f"Fetched {len(new_rows)} new reports total.")
    return new_rows


def _merge_and_save(existing_df: pd.DataFrame, new_rows: list[dict], upload: bool = True) -> pd.DataFrame:
    """Clean new data, merge with existing, deduplicate, sort, and save."""
    if new_rows:
        new_clean = load_caic_data(pd.DataFrame(new_rows), natural_only=True)
        combined = pd.concat([existing_df, new_clean], ignore_index=True) if not existing_df.empty else new_clean
    else:
        combined = existing_df

    if "Observation ID" in combined.columns:
        combined.drop_duplicates(subset=["Observation ID"], inplace=True)
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
        combined.sort_values("Date", ascending=False, inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined.to_csv(CLEAN_FILE, index=False)
    logger.info(f"Saved {len(combined)} cleaned rows to {CLEAN_FILE}")

    if upload:
        upload_to_s3(CLEAN_FILE, S3_CAIC_CLEAN_KEY)
        logger.info(f"Uploaded {CLEAN_FILE} to S3")

    return combined


# Lambda daily update
def update_data() -> int:
    if IS_LAMBDA:
        download_from_s3(S3_CAIC_CLEAN_KEY, CLEAN_FILE)
        upload = True
    else:
        upload = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = pd.read_csv(CLEAN_FILE) if os.path.exists(CLEAN_FILE) else pd.DataFrame()
    start = _get_start_date(existing)
    new_rows = _fetch_new_reports(start)

    combined = _merge_and_save(existing, new_rows, upload)
    return len(combined)


# Full local fetch
def fetch_all():
    existing = pd.read_csv(CLEAN_FILE) if os.path.exists(CLEAN_FILE) else pd.DataFrame()
    start = _get_start_date(existing)
    new_rows = _fetch_new_reports(start)
    if not new_rows and not existing.empty:
        print("Data is already up-to-date.")
        return len(existing)
    combined = _merge_and_save(existing, new_rows)
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
