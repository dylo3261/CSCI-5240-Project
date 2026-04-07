import json
import os
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
    params = event.get("queryStringParameters") or {}
    user_id = params.get("userId")

    if not user_id:
        return {
            "statusCode": 400,
            "headers": HEADERS,
            "body": json.dumps({"error": "userId is required"}),
        }

    items = []
    query_kwargs = {
        "IndexName": "UserIndex",
        "KeyConditionExpression": Key("userId").eq(user_id),
    }

    while True:
        response = reactions_table.query(**query_kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        query_kwargs["ExclusiveStartKey"] = last_key

    # Sort ascending by timestamp so the frontend gets an ordered list
    items.sort(key=lambda x: x["timestamp"])

    # DynamoDB returns Decimal for numeric fields; convert to float for JSON.
    # Skip items missing required fields to prevent a single bad record from
    # crashing the entire Lambda.
    valid = []
    for item in items:
        try:
            item["latitude"] = float(item["latitude"])
            item["longitude"] = float(item["longitude"])
            valid.append(item)
        except (KeyError, TypeError, ValueError):
            pass
    items = valid

    return {
        "statusCode": 200,
        "headers": HEADERS,
        "body": json.dumps(items),
    }
