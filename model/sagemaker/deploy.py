"""Helper to kick off SageMaker Training Jobs and deploy an Inference Endpoint.

Uses the SageMaker Python SDK v2 (``sagemaker.sklearn.SKLearn`` estimator)
to train, deploy, and invoke without manually managing container URIs or
low-level boto3 API calls.

Prerequisites
-------------
- AWS credentials configured (e.g. via ``aws configure`` or an IAM role)
- An S3 bucket for training data and model artifacts (default: explain-model-bucket-west2)
- A SageMaker execution role ARN
- ``sagemaker>=2,<3`` installed locally

Usage
-----
    # Train (bucket defaults to explain-model-bucket-west2)
    python deploy.py train \\
        --role arn:aws:iam::029953330549:role/service-role/AmazonSageMakerAdminIAMExecutionRole\\
        --data-s3 s3://explain-model-bucket-west2/caic_2016_2026/train.csv

    # Train with a custom bucket override
    python deploy.py train \\
        --role arn:aws:iam::123456789012:role/SageMakerRole \\
        --bucket other-bucket \\
        --data-s3 s3://other-bucket/caic_2016_2026/train.csv

    # Deploy endpoint from a completed training job
    python deploy.py deploy \\
        --role arn:aws:iam::123456789012:role/SageMakerRole \\
        --training-job avalanche-logistic-2026-03-11

    # Invoke endpoint
    python deploy.py invoke \\
        --payload '{"elevation":3200,"slope":35,"aspect_degrees":120,
                    "snow_depth":120,"new_snow_24h":30,"temp":-5,"snow_ratio":4.0}'
    python test_inference.py \
    --bucket explain-model-bucket-west2 \
    --key    models/pkl/avalanche-logistic-2026-03-31-19-51-23-039/logistic_avalanche.pkl \
    --payload '{"elevation":3200,"slope":35,"aspect_degrees":180,"snow_depth":120,"new_snow_24h":30,"temp":-5,"snow_ratio":4.0}'

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import boto3
import sagemaker
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.sklearn import SKLearn


# ── Constants ────────────────────────────────────────────────────────────────

# scikit-learn framework version for the SageMaker managed container
SKLEARN_VERSION = "1.4-2"
INSTANCE_TYPE_TRAIN = "ml.m5.large"
INSTANCE_TYPE_DEPLOY = "ml.t2.medium"
ENDPOINT_NAME = "avalanche-logistic"
SOURCE_DIR = str(Path(__file__).resolve().parent)

DEFAULT_BUCKET = "explain-model-bucket-west2"

# Fallback region — override with --region or the AWS_DEFAULT_REGION env var
DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")


def _session(region: str) -> sagemaker.Session:
    """Create a SageMaker Session with an explicit region."""
    boto_session = boto3.Session(region_name=region)
    return sagemaker.Session(boto_session=boto_session)


def _endpoint_exists(session: sagemaker.Session, endpoint_name: str) -> bool:
    """Return True if a SageMaker endpoint with the given name already exists."""
    try:
        session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except session.sagemaker_client.exceptions.ClientError:
        return False


# ── Training ─────────────────────────────────────────────────────────────────

def run_training_job(
    role: str, data_s3: str, bucket: str = DEFAULT_BUCKET, region: str = DEFAULT_REGION,
) -> str:
    """Submit a SageMaker Training Job using the SKLearn estimator."""
    session = _session(region)

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=SOURCE_DIR,
        framework_version=SKLEARN_VERSION,
        role=role,
        instance_type=INSTANCE_TYPE_TRAIN,
        instance_count=1,
        sagemaker_session=session,
        output_path=f"s3://{bucket}/models/",
        base_job_name="avalanche-logistic",
        hyperparameters={
            "test-size": "0.2",
            "bucket": bucket,
        },
    )

    estimator.fit({"train": data_s3}, wait=True)

    job_name = estimator.latest_training_job.name
    print(f"\nTraining job completed: {job_name}")
    print(f"Model artifact: {estimator.model_data}")
    return job_name


# ── Deployment ───────────────────────────────────────────────────────────────

def deploy_endpoint(
    role: str, training_job: str, region: str = DEFAULT_REGION,
) -> str:
    """Create or update a SageMaker real-time endpoint from a completed training job."""
    session = _session(region)

    # Attach to a completed training job to recover model artifact location
    estimator = SKLearn.attach(training_job, sagemaker_session=session)

    # Build a Model object with inference.py as the serving entry point
    model = estimator.create_model(
        entry_point="inference.py",
        source_dir=SOURCE_DIR,
        role=role,
    )

    update = _endpoint_exists(session, ENDPOINT_NAME)
    if update:
        print(f"Endpoint {ENDPOINT_NAME} exists — updating in-place")

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE_DEPLOY,
        endpoint_name=ENDPOINT_NAME,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        update_endpoint=update,
    )

    print(f"Endpoint live: {predictor.endpoint_name}")
    return predictor.endpoint_name


# ── Invoke ───────────────────────────────────────────────────────────────────

def invoke_endpoint(
    endpoint_name: str, payload: str, region: str = DEFAULT_REGION,
) -> dict:
    """Send a JSON payload to the endpoint and print the response."""
    session = _session(region)

    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    data = json.loads(payload)
    result = predictor.predict(data)
    print(json.dumps(result, indent=2))
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SageMaker train / deploy / invoke helper")
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION}, or set AWS_DEFAULT_REGION env var)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    t = sub.add_parser("train")
    t.add_argument("--role", required=True, help="SageMaker execution role ARN")
    t.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"S3 bucket for artifacts (default: {DEFAULT_BUCKET})")
    t.add_argument("--data-s3", required=True, help="S3 URI to training CSV(s)")

    # deploy
    d = sub.add_parser("deploy")
    d.add_argument("--role", required=True)
    d.add_argument("--training-job", required=True, help="Name of completed training job")

    # invoke
    i = sub.add_parser("invoke")
    i.add_argument("--endpoint", default=ENDPOINT_NAME)
    i.add_argument("--payload", required=True, help="JSON string with feature values")

    args = parser.parse_args()
    region = args.region

    if args.command == "train":
        run_training_job(args.role, args.data_s3, args.bucket, region)
    elif args.command == "deploy":
        deploy_endpoint(args.role, args.training_job, region)
    elif args.command == "invoke":
        invoke_endpoint(args.endpoint, args.payload, region)


if __name__ == "__main__":
    main()
