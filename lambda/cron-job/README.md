# Cron Job — Daily Avalanche & Weather Data Pipeline

Runs daily via **EventBridge → Lambda**.

## Pipeline Flow

1. **EventBridge** triggers the Lambda daily
2. **CAIC update** (`python update_caic_data.py --lambda`)
   - Downloads `caic_clean_cache.csv` from S3
   - Determines the latest date in the existing clean data
   - Fetches new observation reports from the CAIC API since that date
   - Cleans new data via `utils/process_caic.load_caic_data()`
   - Merges with existing clean data, deduplicates, and sorts
   - Uploads updated `caic_clean_cache.csv` to S3
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
| `caic_clean_cache.csv` | Cache (daily) | Cleaned & filtered CAIC avalanche observations starting in 2016 (updated daily by Lambda) |
| `daily_station_data.csv` | Cache (daily) | Daily SNOTEL station weather (snow_depth, swe, temp, new_snow_24hr) (updated daily by Lambda) |
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

## Deployment

> **Important:** Dependencies must be installed for **Linux x86_64** (the Lambda
> execution environment). The `--platform` flag tells pip to download
> Linux wheels. The Lambda runtime must be set to **Python 3.13** in the AWS console.

```bash
cd /Users/eddiekiernan/Desktop/CSCI-5240-Project/lambda/cron-job

# 1. Clean previous build
rm -rf build && mkdir build

# 2. Install Linux x86_64 dependencies (runs on macOS, downloads Linux wheels)
pip install \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 3.13 \
  --only-binary=:all: \
  -r requirements.txt \
  -t build/

# 3. Copy source code into build
cp lambda_function.py update_caic_data.py update_weather_data.py build/
cp -r utils build/

# 4. Trim the build
cd build

# Remove packages already provided by Lambda runtime
rm -rf boto3* botocore* s3transfer* jmespath*

# Strip test dirs, metadata, and docs
find . -type d -name "tests" -exec rm -rf {} + 2>/dev/null
find . -type d -name "test" -exec rm -rf {} + 2>/dev/null
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
find . -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
find . -name "*.md" -delete 2>/dev/null
find . -name "*.txt" -delete 2>/dev/null
find . -name "*.rst" -delete 2>/dev/null

# Check size of /build
du -sh .

# 5. Create deployment zip
zip -r9 ../deployment.zip .
cd ..
ls -lh deployment.zip
```