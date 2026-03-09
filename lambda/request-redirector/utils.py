import os
import json
import time
import logging
import boto3
import urllib.request
import csv
from dotenv import load_dotenv # only load .env for local testing

logger = logging.getLogger(__name__)

# Lambda environment detection (AWS_EXECUTION_ENV is set by the Lambda runtime)
IS_LAMBDA = os.environ.get("AWS_EXECUTION_ENV", "").startswith("AWS_Lambda")

# only load .env for local testing
if not IS_LAMBDA:
    load_dotenv()

# Set names of S3 bucket and explainability model
S3_BUCKET = os.environ.get("S3_BUCKET", "location-data-app-bucket")
MODEL_NAME = os.environ.get("MODEL_NAME", "explainable-inference")

# Colorado bounds, may change is we make bounds smaller/more dynamic
COLORADO_MAX_LAT = 41.001
COLORADO_MIN_LAT = 36.999
COLORADO_MAX_LON = -102.001
COLORADO_MIN_LON = -109.001

# Today's date for S3 key
def _get_today():
    return time.strftime("%m_%d_%Y")

# Check if location is in Colorado / specified bounds
def is_valid_location(longitude, latitude):
    return COLORADO_MIN_LON <= longitude <= COLORADO_MAX_LON and COLORADO_MIN_LAT <= latitude <= COLORADO_MAX_LAT


# Fetch Colorado overview from S3
def fetch_colorado_overview():
    try:
        s3 = boto3.client("s3")
        key = f"colorado_overview_{_get_today()}.csv"
        response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        
        # Decode the file bytes and parse as CSV
        body_text = response["Body"].read().decode('utf-8').splitlines()
        reader = csv.DictReader(body_text)
        
        # Convert CSV rows to a list of dictionaries to return as JSON
        data = [row for row in reader]
        return data
    
    except Exception as e:
        logger.error(f"Failed to fetch Colorado overview: {e}")
        raise


# Invoke the explainability model
def fetch_location_explainability(longitude, latitude):
    try:
        lambda_client = boto3.client("lambda")

        response = lambda_client.invoke(
            FunctionName=MODEL_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps({
                "latitude": latitude,
                "longitude": longitude,
            }).encode("utf-8"),
        )

        payload = response["Payload"].read()

        if response.get("FunctionError"):
            raise RuntimeError(f"Lambda invocation error: {payload.decode()}")

        return json.loads(payload)

    except Exception as e:
        logger.error(f"Failed to fetch explainability: {e}")
        raise