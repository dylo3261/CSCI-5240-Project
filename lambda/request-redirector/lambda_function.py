import json
import logging
from utils import fetch_colorado_overview, fetch_location_explainability, is_valid_location

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    # parse body from API Gateway (arrives as a JSON string)
    raw_body = event.get("body")
    payload = None
    if raw_body and isinstance(raw_body, str):
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid JSON in request body"}),
            }
    elif isinstance(raw_body, dict):
        payload = raw_body

    # page load — no payload
    if payload is None:
        try:
            result = fetch_colorado_overview()
        except Exception as e:
            logger.error(f"Colorado overview fetch failed: {e}", exc_info=True)
            return {
                "statusCode": 500,
                "body": json.dumps({"error": f"{e}"}),
            }
    # user location
    elif "longitude" in payload and "latitude" in payload:
        longitude = payload.get("longitude")
        latitude = payload.get("latitude")
        if longitude is None or latitude is None:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing longitude or latitude"}),
            }
        try:
            if not is_valid_location(longitude, latitude):
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": "Location is outside Colorado"}),
                }
            result = fetch_location_explainability(longitude, latitude)
        except Exception as e:
            logger.error(f"Explainability fetch failed: {e}", exc_info=True)
            return {
                "statusCode": 500,
                "body": json.dumps({"error": f"{e}"}),
            }
    # invalid invocation
    else:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid invocation"}),
        }

    # return result
    return {
        "statusCode": 200,
        "body": json.dumps(result),
    }