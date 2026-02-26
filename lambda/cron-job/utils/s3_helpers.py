import os
import logging
import boto3
from dotenv import load_dotenv

# load .env file for local testing
load_dotenv()

logger = logging.getLogger(__name__)

# S3 configuration
S3_BUCKET = "daily-weather-data-csv-bucket"

# Cache files (updated daily by Lambda)
S3_CAIC_CLEAN_KEY = "daily_caic_data.csv"
S3_DAILY_STATION_KEY = "daily_station_data.csv"

# Constant files (not updated daily, but stored in S3)
S3_STATIONS_KEY = "snotel_stations_const.csv"
S3_TERRAIN_KEY = "terrain_const.csv"

# Lambda environment detection (AWS_EXECUTION_ENV is set by the Lambda runtime)
IS_LAMBDA = os.environ.get("AWS_EXECUTION_ENV", "").startswith("AWS_Lambda")

def download_from_s3(s3_key: str, local_path: str) -> bool:
    """
    Download a file from S3 to a local path.

    Returns True if the file was downloaded, False if the key doesn't exist.
    Raises on any other S3 error.
    """
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    try:
        s3.download_file(S3_BUCKET, s3_key, local_path)
        logger.info(f"Downloaded s3://{S3_BUCKET}/{s3_key} → {local_path}")
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.warning(
                f"s3://{S3_BUCKET}/{s3_key} not found — will create a new file."
            )
            return False
        raise


def upload_to_s3(local_path: str, s3_key: str) -> None:
    """Upload a local file to S3."""
    s3 = boto3.client("s3")
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    logger.info(f"Uploaded {local_path} → s3://{S3_BUCKET}/{s3_key}")
