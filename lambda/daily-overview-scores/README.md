# Grid Scores Lambda — Daily Regional Avalanche Risk Map

Generates a grid of avalanche risk predictions across Colorado's mountain terrain every morning. Outputs a CSV of scores and a summary JSON to S3.

## Architecture

```
EventBridge (cron: 7am MST daily)
    │
    ▼
Lambda: regional-risk-grid (arm64, 4GB RAM, 15min timeout)
    │
    ├── Load DEM, model, weather data ONCE into memory
    ├── Generate lat/lon grid (configurable cell size, default 5mi)
    ├── Loop ~2,500 points × ~50ms each = ~2 minutes
    ├── Aggregate summary statistics
    │
    ▼
S3: explain-model-bucket-west2/daily-grid-scores/
    ├── 2026-03-10/
    │   ├── daily_map_grid_scores.csv
    │   └── summary.json
    └── latest/
        ├── daily_map_grid_scores.csv    ← always current
        └── summary.json
```

## Grid Coverage

- **North:** 41.0° (Colorado border)
- **South:** 37.0° (Colorado border)
- **West:** -109.05° (extended west of Colorado border for complete coverage)
- **East:** -105.314° (covers Denver metro area)
- **Default cell size:** 2 miles (~0.029° lat × ~0.037° lon)
- **Total cells:** ~13,900 at 2mi, ~2,100 at 5mi, ~550 at 10mi

All parameters are configurable via the event payload.

## Output

### daily_map_grid_scores.csv

**Frontend-compatible format** (matches existing grid structure):

| Column | Type | Description |
|--------|------|-------------|
| lat | float | Grid cell latitude (SW corner) |
| lon | float | Grid cell longitude (SW corner) |
| value | float | Avalanche probability (0-1) |

Grid iterates southwest → northeast (lat increases row-by-row, lon increases within each row).

### summary.json

```json
{
  "date": "2026-02-27",
  "generated_at": "2026-02-27T14:00:05Z",
  "grid_config": {
    "cell_size_mi": 2,
    "bounds": {"north": 41.0, "south": 37.0, "west": -109.05, "east": -105.314}
  },
  "summary": {
    "avg_risk": 0.23,
    "max_risk": 0.78,
    "min_risk": 0.02,
    "median_risk": 0.18,
    "pct_high": 0.05,
    "pct_moderate": 0.18,
    "pct_low": 0.77,
    "total_cells": 2487,
    "skipped_cells": 213
  }
}
```

## Deployment

```bash
# 1. Update IAM (adds S3 write permissions — run once)
chmod +x update_iam.sh && ./update_iam.sh

# 2. Build and deploy Lambda
chmod +x deploy.sh && ./deploy.sh

# 3. Set up daily cron trigger
chmod +x setup_cron.sh && ./setup_cron.sh
```

First build takes ~20-30 min (compiling GDAL from source). Subsequent code changes rebuild in seconds.

## Manual Testing

```bash
# Full production run (2mi grid, ~13,900 points, ~12 minutes)
aws lambda invoke \
    --function-name regional-risk-grid \
    --payload '{"cell_size_mi": 2}' \
    --cli-binary-format raw-in-base64-out \
    --region us-west-2 \
    response.json && cat response.json | python3 -m json.tool

# Quick test (10mi grid, ~550 points, ~30 seconds)
aws lambda invoke \
    --function-name regional-risk-grid \
    --payload '{"cell_size_mi": 10}' \
    --cli-binary-format raw-in-base64-out \
    --region us-west-2 \
    response.json && cat response.json | python3 -m json.tool

# Medium resolution (5mi grid, ~2,100 points, ~2 minutes)
aws lambda invoke \
    --function-name regional-risk-grid \
    --payload '{"cell_size_mi": 5}' \
    --cli-binary-format raw-in-base64-out \
    --region us-west-2 \
    response.json && cat response.json | python3 -m json.tool

# Custom grid bounds
aws lambda invoke \
    --function-name regional-risk-grid \
    --payload '{"cell_size_mi": 5, "north": 40.0, "south": 39.0, "west": -106.5, "east": -105.0}' \
    --cli-binary-format raw-in-base64-out \
    --region us-west-2 \
    response.json && cat response.json | python3 -m json.tool
```

## Changing Grid Resolution

Pass `cell_size_mi` in the event payload:

| cell_size_mi | Approx cells | Approx time |
|-------------|-------------|-------------|
| 2 | ~13,900 | ~12 min |
| 5 | ~2,100 | ~2 min |
| 10 | ~550 | ~30 sec |

To change the default for the cron job, update the EventBridge target:

```bash
aws events put-targets \
    --rule daily-risk-grid \
    --targets 'Id=1,Arn=arn:aws:lambda:us-west-2:029953330549:function:regional-risk-grid,Input={"cell_size_mi":2}' \
    --region us-west-2
```

## Files

```
grid-scores-lambda/
├── grid_lambda.py          ← Lambda handler (grid generation + batch predict)
├── terrain.py              ← DEM sampling (shared with inference-lambda)
├── weather.py              ← SNOTEL interpolation (shared with inference-lambda)
├── predict.py              ← Model loading (shared with inference-lambda)
├── utils/
│   ├── __init__.py
│   └── s3_helpers.py       ← S3 helpers (shared with inference-lambda)
├── Dockerfile              ← Container image build
├── deploy.sh               ← Build, push, deploy Lambda
├── setup_cron.sh           ← Create EventBridge daily trigger
├── update_iam.sh           ← Add S3 write permissions
└── README.md
```

## AWS Resources

| Resource | Name | Details |
|----------|------|---------|
| Lambda | regional-risk-grid | arm64, 4GB, 15min timeout |
| ECR | regional-risk-grid | Container image |
| EventBridge | daily-risk-grid | cron(0 14 * * ? *) — 7am MST |
| IAM | explainable-inference-role | Shared with inference Lambda |
| S3 Output | explain-model-bucket-west2/daily-grid-scores/ | CSV + JSON |