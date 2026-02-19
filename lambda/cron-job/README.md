# Scrape CAIC (fetch_caic_api.py)
Uses the CAIC API to get, process, and update S3 CSV. Ideally we use this as the primary method for scraping and updating data, but since this leverages an undocumented API, it may be unstable.

Reference to CAIC API only occurs in 1.1.0 release notes: https://avalanche.state.co.us/release-notes.

Ideal usage:
Run script daily (via cron job) to update processed data (which already exists as CSV in S3) from yesterdays reports:
```bash
python fetch_caic_api.py --lambda
```

1. Cron job 6AM w/ Event Bridge trigger calls Lambda
2. Lambda calls CAIC API and gets, processes, and updates S3 CSV
3. Same Lambda gets weather data from API and updates S3 CSV

# 