import json, logging
from update_caic_data import update_data
from update_weather_data import update_daily_station_data

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """Single Lambda handler that runs both data pipelines."""
    results = {}
    
    # Step 1: CAIC data
    try:
        caic_count = update_data()
        results["caic"] = {"status": "OK", "rows": caic_count}
    except Exception as e:
        logger.error(f"CAIC update failed: {e}", exc_info=True)
        results["caic"] = {"status": "FAILED", "error": str(e)}
    
    # Step 2: Weather data
    try:
        weather_count = update_daily_station_data()
        results["weather"] = {"status": "OK", "rows": weather_count}
    except Exception as e:
        logger.error(f"Weather update failed: {e}", exc_info=True)
        results["weather"] = {"status": "FAILED", "error": str(e)}
    
    all_ok = all(r["status"] == "OK" for r in results.values())
    return {
        "statusCode": 200 if all_ok else 500,
        "body": json.dumps(results),
    }
