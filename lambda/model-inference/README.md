# Model Inference — Avalanche Risk Prediction Lambda

Receives a (lat, lon) from the user, returns an avalanche risk prediction with SHAP explanation.

## Architecture

```
User: POST {"latitude": 39.6, "longitude": -105.8}
         │
         ▼
   API Gateway (HTTP API + JWT authorizer)
         │
         ▼
   Lambda: model-inference (container image, 2GB RAM, 120s timeout)
         │
         ├── terrain.py     → S3 [avalanche-model-bucket] colorado_dem.tif
         │                     Sample DEM → elevation, slope, aspect (Horn's method)
         │
         ├── weather.py     → S3 [daily-weather-data-csv-bucket] daily_station_data.csv
         │                   → S3 [daily-weather-data-csv-bucket] snotel_stations_const.csv
         │                     Find 3 nearest stations → IDW interpolate weather
         │                     (CSV updated daily by teammate's cron job Lambda)
         │
         └── predict.py     → S3 [avalanche-model-bucket] avalanche_classifier_tuned.pkl
                               model.predict_proba() + SHAP TreeExplainer
         │
         ▼
   Response: {prediction, risk_level, shap_values, explanation, terrain, weather, stations_used}
```

## S3 Layout

### avalanche-model-bucket (your bucket — model artifacts)

| S3 Key | Description |
|--------|-------------|
| `models/avalanche_classifier_tuned.pkl` | Tuned RF classifier (joblib dict: model, optimal_threshold, feature_cols) |
| `data/colorado_dem.tif` | SRTM elevation raster for Colorado |

### daily-weather-data-csv-bucket (teammate's bucket — already exists)

| S3 Key | Description |
|--------|-------------|
| `daily_station_data.csv` | Daily SNOTEL weather (snow_depth, swe, temp, new_snow_24hr) — updated by cron job |
| `snotel_stations_const.csv` | SNOTEL station metadata (station_id, station_triplet, lat, lon) |

## Directory Structure

```
model-inference/
├── lambda_function.py      ← Entry point: parse input, orchestrate pipeline, return response
├── terrain.py              ← DEM sampling + Horn's method slope/aspect
├── weather.py              ← Find 3 nearest stations + IDW from daily CSV
├── predict.py              ← Load model, predict, SHAP explain
├── utils/
│   ├── __init__.py
│   └── s3_helpers.py       ← S3 download helpers (follows teammate's pattern)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BUCKET` | `avalanche-model-bucket` | S3 bucket for model + DEM |
| `MODEL_KEY` | `models/avalanche_classifier_tuned.pkl` | S3 key for tuned model |
| `DEM_KEY` | `data/colorado_dem.tif` | S3 key for Colorado DEM |
| `DATA_BUCKET` | `daily-weather-data-csv-bucket` | S3 bucket for weather data (teammate's) |

## Deployment

This Lambda uses a **container image** (not a zip) because rasterio/GDAL are too large.

```bash
# 1. Upload model artifacts to your model bucket
aws s3 cp data/models/avalanche_classifier_tuned.pkl s3://avalanche-model-bucket/models/
aws s3 cp data/external/colorado_dem.tif s3://avalanche-model-bucket/data/

# 2. Build and push container image
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="us-east-1"

aws ecr create-repository --repository-name avalanche-inference 2>/dev/null

docker build -t avalanche-inference .

aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker tag avalanche-inference:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/avalanche-inference:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/avalanche-inference:latest

# 3. Create Lambda function
aws lambda create-function \
    --function-name avalanche-inference \
    --package-type Image \
    --code ImageUri=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/avalanche-inference:latest \
    --role arn:aws:iam::$AWS_ACCOUNT_ID:role/avalanche-lambda-role \
    --timeout 120 \
    --memory-size 2048 \
    --environment "Variables={MODEL_BUCKET=avalanche-model-bucket,DATA_BUCKET=daily-weather-data-csv-bucket}"

# 4. Create HTTP API Gateway
API_ID=$(aws apigatewayv2 create-api \
    --name avalanche-inference-api \
    --protocol-type HTTP \
    --target arn:aws:lambda:$AWS_REGION:$AWS_ACCOUNT_ID:function:avalanche-inference \
    --query ApiId --output text)

aws lambda add-permission \
    --function-name avalanche-inference \
    --statement-id apigateway-invoke \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:$AWS_REGION:$AWS_ACCOUNT_ID:$API_ID/*"

echo "API URL: https://$API_ID.execute-api.$AWS_REGION.amazonaws.com"
```

## Testing

```bash
# Direct Lambda invocation
aws lambda invoke \
    --function-name avalanche-inference \
    --payload '{"latitude": 39.6, "longitude": -105.8}' \
    response.json && cat response.json | python -m json.tool

# Via API Gateway
curl -X POST 'https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com' \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer YOUR_JWT_TOKEN' \
    -d '{"latitude": 39.6, "longitude": -105.8}'
```

## IAM Role Permissions

The Lambda role needs:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": [
        "arn:aws:s3:::avalanche-model-bucket/*",
        "arn:aws:s3:::daily-weather-data-csv-bucket/*"
      ]
    }
  ]
}
```

## Performance

| Phase | Cold Start | Warm |
|-------|-----------|------|
| DEM load from S3 | ~5-10s | cached in memory |
| Model load from S3 | ~2-3s | cached in memory |
| Station CSV load | ~1s | cached in memory |
| Daily weather CSV load | ~1s | refreshed daily |
| Terrain extraction | ~100ms | ~100ms |
| Nearest station lookup | ~10ms | ~10ms |
| Model prediction + SHAP | ~200ms | ~200ms |
| **Total** | **~10-15s** | **~500ms** |
