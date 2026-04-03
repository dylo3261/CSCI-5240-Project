# New Avalanche Checker — 15-Minute Cron Lambda

Runs every 15 minutes via **EventBridge (cron) → Lambda** to detect newly reported CAIC avalanche observations and add them as map pins in DynamoDB.

## How It Works

```
Every 15 minutes:
  1. Download latest/daily_caic_data.csv from S3
     → Extract set of known Observation IDs
  2. Fetch CAIC API: last 2 hours of reports
     (2h window guards against API delays & late publishing)
  3. Clean fetched reports
  4. Diff: new_ids = fetched_ids − known_ids
     → If empty, exit early (no new avalanches)
  5. For each new observation:
     → Insert avalanche pin into UserReactionsTable (DynamoDB)
  6. Append new rows to CSV → re-upload to S3
```

### Why 2-Hour Lookback (not 15 minutes)?

The CAIC `observed_at` field reflects when an avalanche was **observed in the field**, but reports may be **published to the API** with delay (sometimes 30–60+ min). A 2-hour window ensures we catch late-published reports. Deduplication via `Observation ID` guarantees zero duplicate pins regardless of overlap.

## DynamoDB Pin Schema

Each new observation creates a row in `UserReactionsTable`:

| Field | Example Value | Notes |
|---|---|---|
| `reactionId` | `"a1b2c3d4-..."` | UUID, unique pin ID |
| `dataType` | `"REACTION"` | Matches GSI partition key |
| `timestamp` | `"2026-04-02T16:30:00.000Z"` | When pin was created |
| `reactionType` | `"avalanche"` | Valid type in websocket-handler |
| `message` | `"CAIC Report: Elk Mountains - D2 SS"` | Human-readable |
| `latitude` | `39.1234` | From CAIC observation |
| `longitude` | `-106.5678` | From CAIC observation |
| `userId` | `"CAIC_SYSTEM"` | Distinguishes automated pins |
| `observationId` | `"12345"` | CAIC observation ID |

## S3 Integration

- **Reads**: `latest/daily_caic_data.csv` (created by `daily-data-scraper`)
- **Writes**: Appends new rows to the same file so both this Lambda and the daily scraper stay in sync


## IAM Permissions Required

The Lambda execution role needs:
- `s3:GetObject`, `s3:PutObject` on `daily-weather-data-csv-bucket/*`
- `dynamodb:PutItem` on `UserReactionsTable`

## Local Usage

```bash
# Ensure .env file is configured (see .env in this directory)
python lambda_function.py
```

## Deployment

**STEP 1: LOGIN**
```bash
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 029953330549.dkr.ecr.us-west-2.amazonaws.com
```

**STEP 2: CREATE ECR REPO (first time only)**
```bash
aws ecr create-repository --repository-name new-avalanche-checker --region us-west-2
```

**STEP 3: BUILD & PUSH**
```bash
docker build --platform linux/amd64 -t new-avalanche-checker . && \
docker tag new-avalanche-checker:latest 029953330549.dkr.ecr.us-west-2.amazonaws.com/new-avalanche-checker:latest && \
docker push 029953330549.dkr.ecr.us-west-2.amazonaws.com/new-avalanche-checker:latest
```

**STEP 4: DEPLOY IN LAMBDA**
Lambda Console > new-avalanche-checker > Image tab > deploy new image > browse images > select latest > save

**STEP 5: SET UP EVENTBRIDGE RULE (first time only)**
EventBridge Console > Rules > Create rule > Schedule > rate(15 minutes) > Target: new-avalanche-checker Lambda
