# Cron Job — Daily Avalanche & Weather Data Pipeline

Runs daily via **EventBridge → Lambda**.

## Pipeline Flow

1. **EventBridge** triggers the Lambda daily
2. **CAIC update** (`python update_caic_data.py --lambda`)
   - Downloads current `latest/daily_caic_data.csv` from S3
   - Archives it to `YYYY-MM-DD/daily_caic_data.csv` (date from the data)
   - Fetches today's new observation reports from the CAIC API
   - Cleans new data via `utils/process_caic.load_caic_data()`
   - Saves single-day CSV and uploads to `latest/daily_caic_data.csv`
3. **Weather update** (`python update_weather_data.py --lambda`)
   - Downloads current `latest/daily_station_data.csv` from S3
   - Archives it to `YYYY-MM-DD/daily_station_data.csv` (date from the data)
   - Downloads `constant/snotel_stations_const.csv` for station list
   - Looks up yesterday's snow depths from archive for `new_snow_24hr` calculation
   - For each station, fetches today's `snow_depth`, `swe`, and `temp`
   - Saves single-day CSV and uploads to `latest/daily_station_data.csv`

## S3 Bucket: `daily-weather-data-csv-bucket`

| S3 Key Pattern | Type | Description |
|---|---|---|
| `constant/snotel_stations_const.csv` | Constant | Colorado SNOTEL station metadata (not updated daily) |
| `constant/terrain_const.csv` | Constant | Terrain features for avalanche locations (not updated daily) |
| `latest/daily_caic_data.csv` | Latest (daily) | Today's cleaned & filtered CAIC avalanche observations |
| `latest/daily_station_data.csv` | Latest (daily) | Today's SNOTEL station weather (snow_depth, swe, temp, new_snow_24hr) |
| `YYYY-MM-DD/daily_caic_data.csv` | Archive | Single-day CAIC data for that date |
| `YYYY-MM-DD/daily_station_data.csv` | Archive | Single-day weather data for that date |

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