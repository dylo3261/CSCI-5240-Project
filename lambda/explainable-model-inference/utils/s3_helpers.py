import os
import logging
import boto3
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── S3 Configuration ─────────────────────────────────────────────────────────
# Model bucket (where your trained model lives)
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "explain-model-bucket-west2")
S3_MODEL_KEY = os.environ.get("MODEL_KEY", "models/pkl/logistic_avalanche.pkl")
S3_DEM_KEY = os.environ.get("DEM_KEY", "data/colorado_dem.tif")

# Data bucket (your teammate's cron job bucket — already exists)
DATA_BUCKET = os.environ.get("DATA_BUCKET", "daily-weather-data-csv-bucket")
S3_DAILY_STATION_KEY = "latest/daily_station_data.csv"
S3_STATIONS_KEY = "constant/snotel_stations_const.csv"

# Lambda environment detection (same as teammate's pattern)
IS_LAMBDA = os.environ.get("AWS_EXECUTION_ENV", "").startswith("AWS_Lambda")

# Local paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of utils/)
DATA_DIR = "/tmp/data" if IS_LAMBDA else os.path.join(BASE_DIR, "data")


def download_from_s3(s3_key: str, local_path: str, bucket: str = DATA_BUCKET) -> bool:
    """
    Download a file from S3 to a local path.
    Returns True if downloaded, False if key doesn't exist.
    """
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    try:
        s3.download_file(bucket, s3_key, local_path)
        logger.info(f"Downloaded s3://{bucket}/{s3_key} → {local_path}")
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.warning(f"s3://{bucket}/{s3_key} not found")
            return False
        raise


def download_model_file(s3_key: str, local_path: str) -> bool:
    """Download from the model bucket."""
    return download_from_s3(s3_key, local_path, bucket=MODEL_BUCKET)


def download_data_file(s3_key: str, local_path: str) -> bool:
    """Download from the data bucket (teammate's cron job bucket)."""
    return download_from_s3(s3_key, local_path, bucket=DATA_BUCKET)
