# Model Inference — Avalanche Risk Prediction Lambda

Receives a (lat, lon) coordinate in Colorado, returns an avalanche risk prediction with SHAP explanations using real-time SNOTEL weather data and precise DEM terrain sampling.

## Architecture

```
User → API Gateway (HTTP API) ──┐
                                ├──▶ Lambda: explainable-inference (container image, arm64, 2GB RAM, 120s timeout)
SQS: explainable-inference-queue┘         │
                                          ├── terrain.py     → S3 [explain-model-bucket-west2] colorado_dem.tif
                                          │                     Sample DEM → elevation, slope, aspect (Horn's method)
                                          │
                                          ├── weather.py     → S3 [daily-weather-data-csv-bucket] daily_station_data.csv
                                          │                   → S3 [daily-weather-data-csv-bucket] snotel_stations_const.csv
                                          │                     Find 3 nearest SNOTEL stations → IDW interpolate weather
                                          │                     (CSV updated daily by teammate's cron job Lambda)
                                          │
                                          └── predict.py     → S3 [explain-model-bucket-west2] avalanche_classifier_tuned.pkl
                                                                model.predict_proba() + SHAP TreeExplainer
                                          │
                                          ▼
                                    Response JSON:
                                    {prediction, risk_level, shap_values, explanation,
                                     terrain, weather, stations_used, location}
```

## AWS Resources

| Resource | Name / ID | Region |
|----------|-----------|--------|
| Lambda Function | `explainable-inference` | us-west-2 |
| ECR Repository | `explainable-inference` | us-west-2 |
| IAM Role | `explainable-inference-role` | global |
| API Gateway | `explainable-inference-api` | us-west-2 |
| SQS Queue | `explainable-inference-queue` | us-west-2 |
| S3 Bucket (model) | `explain-model-bucket-west2` | us-west-2 |
| S3 Bucket (weather) | `daily-weather-data-csv-bucket` | us-west-2 |

## S3 Layout

### explain-model-bucket-west2 (my bucket)

| S3 Key | Description |
|--------|-------------|
| `models/avalanche_classifier_tuned.pkl` | Tuned RF classifier (joblib dict: model, optimal_threshold, feature_cols) |
| `data/colorado_dem.tif` | SRTM elevation raster for Colorado |

### daily-weather-data-csv-bucket (teammate's bucket)

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
├── predict.py              ← Load model, predict, SHAP explain (structured list output)
├── utils/
│   ├── __init__.py
│   └── s3_helpers.py       ← S3 download helpers for both buckets
├── test_local.py           ← Local testing script (--mock flag for fake data)
├── deploy.sh               ← One-command build, push, update Lambda, and test
├── requirements.txt
├── Dockerfile              ← Single-stage build with SQLite, PROJ, GDAL compiled from source
└── README.md
```

## Response Format

```json
{
  "prediction": 0.12,
  "risk_level": "LOW",
  "optimal_threshold": 0.42,
  "shap_values": {
    "aspect_degrees": 0.1789,
    "snow_depth": -0.1789,
    "new_snow_24h": -0.0295,
    "swe": 0.0295,
    "temp": -0.0108,
    "elevation": -0.0085,
    "slope": 0.0085
  },
  "base_value": 0.7392,
  "explanation": [
    {
      "feature": "aspect_degrees",
      "value": 319.5,
      "shap_impact": 0.1789,
      "direction": "increasing",
      "context": null
    }
  ],
  "terrain": {"elevation": 3575.0, "slope": 18.39, "aspect_degrees": 319.46},
  "weather": {"snow_depth": 93.98, "new_snow_24h": 7.62, "swe": 17.16, "temp": -4.93},
  "stations_used": [
    {"station_id": 935, "station_triplet": "935:CO:SNTL", "name": "Jackwhacker Gulch", "distance_km": 3.24}
  ],
  "location": {"latitude": 39.6, "longitude": -105.8}
}
```

### Risk Levels

The model is a binary classifier (avalanche vs. no avalanche). Risk levels are binned from the probability:

| Risk Level | Condition |
|-----------|-----------|
| **HIGH DANGER** | `prob >= 0.42` (at or above optimal threshold) |
| **MODERATE** | `prob >= 0.32` (within 0.1 below threshold) |
| **LOW** | `prob < 0.32` |

### SHAP Explanation

Each item in the `explanation` array represents a feature's contribution to the prediction. Sorted by absolute impact (most important first). The `direction` field indicates whether the feature pushes risk up ("increasing") or down ("decreasing"). The `context` field provides human-readable interpretation when applicable (e.g., "heavy recent snowfall - CRITICAL").

## Input Validation

Two layers of validation protect against invalid coordinates:

1. **Bounding box check** — Rejects coordinates outside Colorado (lat 36.5–41.5, lon -109.5 to -101.5) with a 400 error and valid ranges hint.
2. **DEM edge check** — Catches points inside the bounding box but at DEM edges where slope/aspect can't be calculated, with a descriptive error suggesting the user try a more central location.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BUCKET` | `explain-model-bucket-west2` | S3 bucket for model + DEM |
| `MODEL_KEY` | `models/avalanche_classifier_tuned.pkl` | S3 key for tuned model |
| `DEM_KEY` | `data/colorado_dem.tif` | S3 key for Colorado DEM |
| `DATA_BUCKET` | `daily-weather-data-csv-bucket` | S3 bucket for weather data (teammate's) |

## Deployment

### Quick Deploy (code changes only)

```bash
./deploy.sh
```

This builds the Docker image, pushes to ECR, updates the Lambda, waits for it to be ready, and runs a smoke test. Code-only changes rebuild in seconds because all heavy Docker layers (GDAL, numpy, etc.) are cached.

### First-Time Setup

Already completed. For reference, the initial setup involved:

1. **Created ECR repo**: `aws ecr create-repository --repository-name explainable-inference --region us-west-2`
2. **Built Docker image**: `docker build -t explainable-inference .` (single-stage build compiling SQLite 3.44, PROJ 7.2.1, GDAL 3.4.3 from source inside the Lambda base image)
3. **Pushed to ECR**: Tagged and pushed to `029953330549.dkr.ecr.us-west-2.amazonaws.com/explainable-inference:latest`
4. **Created IAM role** (`explainable-inference-role`) with S3 read access to both buckets, SQS permissions, and CloudWatch logging
5. **Created Lambda function** with arm64 architecture, 2GB memory, 120s timeout
6. **Created HTTP API Gateway** (`explainable-inference-api`)
7. **Created SQS queue** (`explainable-inference-queue`) with 180s visibility timeout and event source mapping (batch size 1)

### Updating the Model

To deploy a new model without changing code:

```bash
aws s3 cp path/to/new_model.pkl s3://explain-model-bucket-west2/models/avalanche_classifier_tuned.pkl

# Force Lambda cold start to re-download model
aws lambda update-function-configuration \
    --function-name explainable-inference \
    --environment "Variables={MODEL_BUCKET=explain-model-bucket-west2,DATA_BUCKET=daily-weather-data-csv-bucket,MODEL_VERSION=2}" \
    --region us-west-2
```

Changing any env var forces a cold start, which re-downloads the model from S3.

**Model requirements:**
```python
{
    "model": "<sklearn classifier with .predict_proba()>",
    "optimal_threshold": float,
    "feature_cols": ["elevation", "slope", "aspect_degrees", "snow_depth", "new_snow_24h", "swe", "temp"]
}
```

Any binary classifier with `predict_proba` works (RandomForest, XGBoost, etc.). Feature set must match these 7 features.

## Docker Build Details

The Dockerfile compiles native dependencies from source inside the Lambda base image (`public.ecr.aws/lambda/python:3.11`) to avoid glibc compatibility issues. This was necessary because pre-built wheels for rasterio/GDAL don't exist for Amazon Linux 2 aarch64.

**Build chain:**
1. GCC 10 (from yum) — needed to compile C extensions
2. SQLite 3.44 (from source) — Amazon Linux 2 ships 3.7, PROJ requires >= 3.11
3. PROJ 7.2.1 (from source) — coordinate transformation library for GDAL
4. GDAL 3.4.3 (from source) — geospatial data abstraction library for rasterio
5. rasterio 1.3.x (pip, compiled against our GDAL) — DEM reading
6. numpy 1.26.4 (pip, manylinux2014 wheel) — pinned to avoid glibc 2.27 requirement of numpy 2.x
7. scipy < 1.14, llvmlite 0.43.0, numba 0.60.0 (pip, pre-built wheels) — pinned to avoid source compilation
8. scikit-learn < 1.6, shap < 0.50, pandas < 2.3 (pip) — ML libraries

**First build: ~20-30 minutes** (compiling SQLite, PROJ, GDAL). **Subsequent code-only rebuilds: seconds** (Docker caches all compilation layers).

## Testing

```bash
# Direct Lambda invocation (pretty-printed)
aws lambda invoke \
    --function-name explainable-inference \
    --payload '{"latitude": 39.6, "longitude": -105.8}' \
    --cli-binary-format raw-in-base64-out \
    --region us-west-2 \
    response.json \
  && cat response.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(json.loads(d['body']), indent=2))"

# Via API Gateway
curl -X POST "https://YOUR_API_ID.execute-api.us-west-2.amazonaws.com" \
    -H "Content-Type: application/json" \
    -d '{"latitude": 39.6, "longitude": -105.8}'

# Via SQS (async — check CloudWatch for result)
aws sqs send-message \
    --queue-url https://sqs.us-west-2.amazonaws.com/029953330549/explainable-inference-queue \
    --message-body '{"latitude": 39.6, "longitude": -105.8}' \
    --region us-west-2

# View logs
aws logs tail /aws/lambda/explainable-inference --region us-west-2 --follow

# Local Docker test (no S3 access)
docker run --entrypoint "" explainable-inference python -c "from lambda_function import lambda_handler; print('OK')"
```

## IAM Role Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": [
        "arn:aws:s3:::explain-model-bucket-west2/*",
        "arn:aws:s3:::daily-weather-data-csv-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"],
      "Resource": "arn:aws:sqs:us-west-2:029953330549:explainable-inference-queue"
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

## Versioning & Future Models

Use Lambda versions and aliases for iterating on the same model:

```bash
# Publish a version after a stable deploy
aws lambda publish-version \
    --function-name explainable-inference \
    --description "v1 - Random Forest with 7 features" \
    --region us-west-2

# Create aliases for environments
aws lambda create-alias --function-name explainable-inference --name prod --function-version 1 --region us-west-2
aws lambda create-alias --function-name explainable-inference --name staging --function-version '$LATEST' --region us-west-2

# Promote staging to prod
aws lambda update-alias --function-name explainable-inference --name prod --function-version 2 --region us-west-2
```

For fundamentally different model architectures (different features, different classifiers), create separate Lambda functions that share the same weather/terrain code.

## TODO

- [ ] Add JWT authorizer to API Gateway
- [ ] Set up SQS result destination (DynamoDB or SNS) for async responses
- [ ] Consider provisioned concurrency if cold starts are an issue (~$108/month for 5 warm containers)
