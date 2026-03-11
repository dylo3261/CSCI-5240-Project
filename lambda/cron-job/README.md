# Cron Job — Daily Avalanche & Weather Data Pipeline

Runs daily via **EventBridge → Lambda**.

## Pipeline Flow

1. **EventBridge** triggers the Lambda daily
2. **CAIC update** (`python update_caic_data.py --lambda`)
   - Downloads `daily_caic_data.csv` from S3
   - Determines the latest date in the existing clean data
   - Fetches new observation reports from the CAIC API since that date
   - Cleans new data via `utils/process_caic.load_caic_data()`
   - Merges with existing clean data, deduplicates, and sorts
   - Uploads updated `daily_caic_data.csv` to S3
3. **Weather update** (`python update_weather_data.py --lambda`)
   - Downloads `daily_station_data.csv` and `snotel_stations_const.csv` from S3
   - For each station in `snotel_stations_const.csv`, fetches `snow_depth`, `swe`, and `temp`
   - Computes `new_snow_24hr` as today's `snow_depth` minus yesterday's stored value
   - On first run (no existing data), fetches both today and yesterday to bootstrap
   - On subsequent runs, only fetches today and uses yesterday's data from the CSV
   - Merges, deduplicates, and uploads `daily_station_data.csv` to S3

## S3 Bucket: `daily-weather-data-csv-bucket`

| S3 Key | Type | Description |
|--------|------|-------------|
| `daily_caic_data.csv` | Cache (daily) | Cleaned & filtered CAIC avalanche observations starting in 2016 (updated daily by Lambda) |
| `daily_weather_data.csv` | Cache (daily) | Daily SNOTEL station weather (snow_depth, swe, temp, new_snow_24hr) (updated daily by Lambda) |
| `snotel_stations_const.csv` | Constant | Colorado SNOTEL station metadata (not updated daily, but stored in S3) |
| `terrain_const.csv` | Constant | Terrain features for avalanche locations (not updated daily, but stored in S3) |

## Local Usage

```bash
# Full historical CAIC fetch + clean (resumes from existing clean file)
python update_caic_data.py

# Local weather station fetch (fetches today's data, bootstraps if first run)
python update_weather_data.py

# Simulate Lambda daily update locally (requires AWS credentials in .env file)
python update_caic_data.py --lambda
python update_weather_data.py --lambda
```

Ensure set up .env file with AWS credentials in `/cron-job` directory
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_BUCKET=...
AWS_DEFAULT_REGION=...
AWS_EXECUTION_ENV=...
```

## Deployment
Amazon ECR container using docker to deploy the function

**STEP 1: LOGIN**
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 029953330549.dkr.ecr.us-west-2.amazonaws.com

**STEP 2: PUSH CHANGES TO AMAZON ECR**
docker build --platform linux/amd64 -t cron-lambda . && \
docker tag cron-lambda:latest 029953330549.dkr.ecr.us-west-2.amazonaws.com/cron-lambda:latest && \
docker push 029953330549.dkr.ecr.us-west-2.amazonaws.com/cron-lambda:latest

**STEP 3: REDEPLOY IN LAMBDA**
Lambda Console > daily-data-scraper > Image tab > deploy new image> browse images > select latest > save