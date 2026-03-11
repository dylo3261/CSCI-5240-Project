# Avalanche Logistic Classifier — SageMaker Deployment

This folder contains everything needed to train and serve the avalanche binary classifier on AWS SageMaker using the **SageMaker Python SDK v2** (`sagemaker>=2,<3`).

---

## Folder Structure

```
sagemaker/
├── config.py          # Shared constants: features, risk thresholds, pred_class()
├── train.py           # SageMaker Training Job entry point
├── inference.py       # SageMaker Endpoint serving hooks
├── deploy.py          # CLI helper: package → train → deploy → invoke
├── requirements.txt   # Python dependencies installed inside the container
└── README.md          # This file
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| AWS credentials | Run `aws configure` or attach an IAM role |
| S3 bucket | Stores training data, source code, and model artifacts |
| SageMaker execution role | IAM role with `AmazonSageMakerFullAccess` + S3 access |
| `sagemaker>=2,<3` | Install locally: `pip install 'sagemaker>=2,<3'` |

### Find your account ID
```bash
aws sts get-caller-identity --query Account --output text
```

### Find or create a SageMaker execution role
```bash
# List existing SageMaker roles
aws iam list-roles --query "Roles[?contains(RoleName, 'SageMaker')].Arn" --output table
```
If none exist, create one in **IAM → Roles → Create role → AWS service → SageMaker** and attach `AmazonSageMakerFullAccess`.

The role ARN format is:
```
arn:aws:iam::<account-id>:role/<role-name>
```

---

## Model Overview

### Task
Binary classification: predict whether an avalanche occurred (`avalanche_occurred = 1`).

### Features (7 inputs)

| Feature | Description |
|---|---|
| `elevation` | Terrain elevation (meters) |
| `slope` | Slope angle (degrees) |
| `aspect_degrees` | Distance from south (0 = south-facing, 180 = north-facing) |
| `snow_depth` | Total snow depth (cm) |
| `new_snow_24h` | New snow in last 24 hours (cm) |
| `temp` | Temperature (°C) |
| `snow_ratio` | Engineered: `snow_depth / swe` (0 if either is zero) |

### Output

| Field | Description |
|---|---|
| `probability` | Float 0–1, probability of avalanche |
| `risk_class` | One of `Low`, `Moderate`, `Considerable`, `High`, `Extreme` |

### Risk Class Thresholds

| Class | Probability Range |
|---|---|
| Low | < 0.2 |
| Moderate | 0.2 – 0.4 |
| Considerable | 0.4 – 0.6 |
| High | 0.6 – 0.8 |
| Extreme | ≥ 0.8 |

### Training Details
- Algorithm: Logistic Regression (scikit-learn)
- Regularization: L2 (`l1_ratio=0`, `solver=lbfgs`)
- Hyperparameter tuning: Grid search over `C` and `class_weight`, optimised for **recall** (minimise false negatives)
- Cross-validation: Stratified 10-fold
- Train/test split: 80% / 20% stratified

---

## Usage

All commands are run from this folder via `deploy.py`. Pass `--region` to set the AWS region (defaults to `AWS_DEFAULT_REGION` env var, then `us-east-1`).

```bash
cd model/sagemaker/
```

### Step 1 — Run a Training Job

Kicks off a SageMaker Training Job. The SDK automatically packages the source code and uploads it to S3. New data from the scraping job should be uploaded to S3 before re-running this command.

```bash
python deploy.py --region us-east-1 train \
    --role arn:aws:iam::123456789012:role/SageMakerRole \
    --bucket my-bucket \
    --data-s3 s3://my-bucket/data/
```

- Polls every 30 seconds and prints status until complete.
- Saves the trained `.pkl` model artifact to `s3://my-bucket/models/`.

### Step 2 — Deploy Inference Endpoint

Creates (or updates) a real-time SageMaker endpoint from a completed training job.

```bash
python deploy.py --region us-east-1 deploy \
    --role arn:aws:iam::123456789012:role/SageMakerRole \
    --training-job avalanche-logistic-2026-03-11-120000
```

- Endpoint name: `avalanche-logistic` (configurable via `ENDPOINT_NAME` in `deploy.py`)
- Waits until the endpoint status is `InService`.
- Re-running `deploy` on an existing endpoint performs an **in-place update** with zero downtime.

### Step 3 — Invoke Endpoint

Send a single observation for prediction:

```bash
python deploy.py --region us-east-1 invoke \
    --payload '{"elevation":3200,"slope":35,"aspect_degrees":120,"snow_depth":120,"new_snow_24h":30,"temp":-5,"snow_ratio":4.0}'
```

Example response:
```json
{
  "predictions": [
    {
      "probability": 0.73,
      "risk_class": "High"
    }
  ]
}
```

Batch request (list of observations):
```bash
python deploy.py invoke \
    --payload '[{"elevation":3200,"slope":35,"aspect_degrees":120,"snow_depth":120,"new_snow_24h":30,"temp":-5,"snow_ratio":4.0},
                {"elevation":2800,"slope":20,"aspect_degrees":45,"snow_depth":60,"new_snow_24h":5,"temp":2,"snow_ratio":2.1}]'
```

---

## Retraining Workflow (New Data)

When the data scraping cron job produces new data:

1. Upload the new CSV to S3:
   ```bash
   aws s3 cp /path/to/new_data.csv s3://my-bucket/data/
   ```

2. Run a new training job:
   ```bash
   python deploy.py train --role <ROLE> --bucket my-bucket --data-s3 s3://my-bucket/data/
   ```

3. Update the endpoint:
   ```bash
   python deploy.py deploy --role <ROLE> --training-job <new-job-name>
   ```

---

## Instance Types

| Stage | Default | Notes |
|---|---|---|
| Training | `ml.m5.large` | Change `INSTANCE_TYPE_TRAIN` in `deploy.py` |
| Inference | `ml.t2.medium` | Change `INSTANCE_TYPE_DEPLOY` in `deploy.py` |

---

## Container Image

The pre-built SageMaker scikit-learn container (`1.2-1`) is resolved automatically by the SDK via `sagemaker.sklearn.SKLearn`. No manual ECR image URI management is needed.
