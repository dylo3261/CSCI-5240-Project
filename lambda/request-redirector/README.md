# Request Redirector Lambda

## Flow
Invoked by API Gateway with 2 types of requests:
- **Page load:** No payload. Called from frontend on render. Fetches and returns `colorado_overview_{date}.json` from S3 (shaded square data updated daily).
- **User location:** Payload contains `longitude` and `latitude`. Makes a request to the explainability model with the coordinates. Returns the model's JSON response to the frontend.

## Function Invocations
**Colorado Overview**
```json
{
  "body": null
}
```

**User Location**
```json
{
  "body": {
    "longitude": -105.5,
    "latitude": 39.7
  }
}
```

## Local Testing
Set the following in a `.env` file and run `python test_local.py`:
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
MODEL_NAME
S3_BUCKET
```

## Deployment

