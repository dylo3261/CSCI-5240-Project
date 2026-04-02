# Request Redirector Lambda

## Flow
Invoked by API Gateway with 2 types of requests:
- **Page load:** No payload. Called from frontend on render. Fetches and returns `daily_map_grid_scores.csv` from S3 (shaded square data updated daily).
- **User location:** Payload contains `longitude` and `latitude`. Makes a request to the explainability model with the coordinates. Returns the model's JSON response to the frontend.

## Function Invocations
**Colorado Overview**
```json
{
  "body": null
}
```

```bash
curl -s -X POST https://mera3wkzuj.execute-api.us-west-2.amazonaws.com/request-redirector
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

```bash
curl -s -X POST https://mera3wkzuj.execute-api.us-west-2.amazonaws.com/request-redirector \
  -H "Content-Type: application/json" \
  -d '{"longitude": -104.99, "latitude": 39.73}'
```

## Local Testing
Set the following in a `.env` file and run `python3 test_local.py`:
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
MODEL_NAME
S3_BUCKET
```

## Deployment

Because `boto3` is included natively in the AWS Lambda Python runtime, and `python-dotenv` is only used for local testing, **you do not need to package any external dependencies**.

### 1. Build the zip file

```bash
zip -j deployment.zip lambda_function.py utils.py
```
*(The `-j` flag ensures the files are added to the root of the zip, without their parent folder structure.)*

### 2. Upload to Lambda
You can upload `deployment.zip` directly through the AWS Console, or deploy it instantly using the AWS CLI:

```bash
aws lambda update-function-code \
    --function-name request-redirector \
    --zip-file fileb://deployment.zip
```
