import json
import os
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

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

    while True:
        response = reactions_table.query(**query_kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        query_kwargs["ExclusiveStartKey"] = last_key

    # DynamoDB returns Decimal for numeric fields; convert to float for JSON
    for item in items:
        item["latitude"] = float(item["latitude"])
        item["longitude"] = float(item["longitude"])

    return {
        "statusCode": 200,
        "headers": HEADERS,
        "body": json.dumps(items),
    }
