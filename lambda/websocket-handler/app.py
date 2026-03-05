import json
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

CONNECTIONS_TABLE = os.environ["CONNECTIONS_TABLE"]
REACTIONS_TABLE = os.environ["REACTIONS_TABLE"]
WEBSOCKET_ENDPOINT = os.environ["WEBSOCKET_ENDPOINT"]

dynamodb = boto3.resource("dynamodb")
connections_table = dynamodb.Table(CONNECTIONS_TABLE)
reactions_table = dynamodb.Table(REACTIONS_TABLE)


def lambda_handler(event, context):
    route_key = event["requestContext"]["routeKey"]
    connection_id = event["requestContext"]["connectionId"]

    if route_key == "$connect":
        return _connect(connection_id)
    elif route_key == "$disconnect":
        return _disconnect(connection_id)
    elif route_key == "sendReaction":
        return _send_reaction(event, connection_id)

    return {"statusCode": 400, "body": "Unknown route"}


def _connect(connection_id):
    connections_table.put_item(Item={"connectionId": connection_id})
    return {"statusCode": 200, "body": "Connected"}


def _disconnect(connection_id):
    connections_table.delete_item(Key={"connectionId": connection_id})
    return {"statusCode": 200, "body": "Disconnected"}


def _send_reaction(event, sender_connection_id):
    body = json.loads(event.get("body") or "{}")

    required = ("reactionType", "message", "latitude", "longitude", "userId")
    missing = [f for f in required if body.get(f) is None]
    if missing:
        return {"statusCode": 400, "body": f"Missing required fields: {', '.join(missing)}"}

    reaction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # DynamoDB rejects float — use Decimal(str(...)) to avoid precision artifacts
    item = {
        "reactionId":   reaction_id,
        "dataType":     "REACTION",
        "timestamp":    timestamp,
        "reactionType": body["reactionType"],
        "message":      body["message"],
        "latitude":     Decimal(str(body["latitude"])),
        "longitude":    Decimal(str(body["longitude"])),
        "userId":       body["userId"],
    }

    reactions_table.put_item(Item=item)

    # json.dumps cannot serialize Decimal, so broadcast uses original float values
    broadcast = {
        "reactionId":   reaction_id,
        "dataType":     "REACTION",
        "timestamp":    timestamp,
        "reactionType": body["reactionType"],
        "message":      body["message"],
        "latitude":     body["latitude"],
        "longitude":    body["longitude"],
        "userId":       body["userId"],
    }

    apigw = boto3.client("apigatewaymanagementapi", endpoint_url=WEBSOCKET_ENDPOINT)
    message = json.dumps(broadcast).encode("utf-8")

    # Scan with pagination to handle large connection counts
    connections = []
    scan_kwargs = {"ProjectionExpression": "connectionId"}
    while True:
        response = connections_table.scan(**scan_kwargs)
        connections.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    for conn in connections:
        cid = conn["connectionId"]
        try:
            apigw.post_to_connection(ConnectionId=cid, Data=message)
        except ClientError as e:
            if e.response["Error"]["Code"] == "GoneException":
                connections_table.delete_item(Key={"connectionId": cid})
            else:
                raise

    return {"statusCode": 200, "body": "Reaction sent"}
