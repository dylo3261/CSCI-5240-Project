import json
import logging
import os
from datetime import datetime, timezone, timedelta

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REACTIONS_TABLE = os.environ["REACTIONS_TABLE"]

dynamodb = boto3.resource("dynamodb")
reactions_table = dynamodb.Table(REACTIONS_TABLE)

HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
}


def lambda_handler(event, context):
    since = (
        datetime.now(timezone.utc) - timedelta(hours=24)
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    items = []
    query_kwargs = {
        "IndexName": "TimeIndex",
        "KeyConditionExpression": (
            Key("dataType").eq("REACTION") & Key("timestamp").gte(since)
        ),
    }

    try:
        while True:
            response = reactions_table.query(**query_kwargs)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            query_kwargs["ExclusiveStartKey"] = last_key
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        logger.error("DynamoDB query failed — code=%s message=%s", code, msg)
        return {
            "statusCode": 500,
            "headers": HEADERS,
            "body": json.dumps({"error": code, "message": msg}),
        }

    def _convert(obj):
        """Recursively convert Decimal → int or float for JSON serialization."""
        from decimal import Decimal
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    # DynamoDB returns Decimal for numeric fields; convert all before JSON dump.
    # Skip items missing required coordinate fields.
    valid = []
    for item in items:
        try:
            _ = item["latitude"]
            _ = item["longitude"]
            valid.append(_convert(item))
        except KeyError as e:
            logger.warning("Skipping malformed item %s: missing %s", item.get("reactionId"), e)

    logger.info("Returning %d reactions (skipped %d malformed)", len(valid), len(items) - len(valid))
    return {
        "statusCode": 200,
        "headers": HEADERS,
        "body": json.dumps(valid),
    }
