"""End-to-end inference test — loads the trained .pkl from S3 and runs predictions.

This script is the reference implementation for teammates writing AWS Lambda or
any other downstream inference service.  It mirrors the four-step pattern used
by the SageMaker inference.py (load → input → predict → output) but operates
completely outside of SageMaker, pulling the raw .pkl artifact directly from S3.

──────────────────────────────────────────────────────────────────────────────
HOW THE MODEL ARTIFACT GETS INTO S3
──────────────────────────────────────────────────────────────────────────────
When a SageMaker training job finishes, train.py uploads the raw .pkl to:

    s3://<bucket>/models/pkl/<training-job-name>/logistic_avalanche.pkl

You can find the exact S3 URI in the training job output, or list all pkl
artifacts with:

    aws s3 ls s3://explain-model-bucket-west2/models/pkl/ --recursive

──────────────────────────────────────────────────────────────────────────────
LAMBDA ADAPTATION GUIDE  (read this before writing the Lambda function)
──────────────────────────────────────────────────────────────────────────────
The function ``lambda_handler`` at the bottom of this file is a *drop-in
skeleton* for AWS Lambda.  Key points for your Lambda implementation:

1. COLD-START CACHING
   Download / load the model once at module level (outside the handler) so it
   is reused across warm invocations.  See ``_load_model_from_s3`` below and
   the module-level ``_CACHED_MODEL`` pattern in the Lambda skeleton.

2. ENVIRONMENT VARIABLES
   Store the bucket name and S3 key in Lambda environment variables rather
   than hard-coding them:
       MODEL_BUCKET  = os.environ["MODEL_BUCKET"]   # explain-model-bucket-west2
       MODEL_S3_KEY  = os.environ["MODEL_S3_KEY"]   # models/pkl/<job>/logistic_avalanche.pkl

3. IAM PERMISSIONS
   The Lambda execution role needs:
       s3:GetObject  on  arn:aws:s3:::explain-model-bucket-west2/models/pkl/*

4. DEPENDENCIES
   Lambda layers (or a container image) must include:
       scikit-learn, pandas, numpy, joblib, boto3
   A ready-made requirements file is at model/sagemaker/requirements.txt.

5. INPUT / OUTPUT CONTRACT
   Input  – API Gateway passes ``event["body"]`` as a JSON string.
   Output – return a dict with ``statusCode`` and a JSON ``body``.
   The exact shapes are documented on ``input_fn`` and ``output_fn`` below.

──────────────────────────────────────────────────────────────────────────────
QUICK START
──────────────────────────────────────────────────────────────────────────────
    # Run the built-in test suite against a real S3 artifact
    python test_inference.py \\
        --bucket explain-model-bucket-west2 \\
        --key    models/pkl/<training-job-name>/logistic_avalanche.pkl

    # Predict from a custom JSON payload
    python test_inference.py \\
        --bucket explain-model-bucket-west2 \\
        --key    models/pkl/<training-job-name>/logistic_avalanche.pkl \\
        --payload '[{"elevation":3200,"slope":35,"aspect_degrees":180,
                     "snow_depth":120,"new_snow_24h":30,"temp":-5,"snow_ratio":4.0}]'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import boto3
import joblib
import pandas as pd

# Allow running from the sagemaker/ directory without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_FILENAME, SELECTED_FEATURES, pred_class  # noqa: E402


# ── Step 1: Load ─────────────────────────────────────────────────────────────

def load_model_from_s3(bucket: str, key: str) -> dict:
    """Download the .pkl artifact from S3 and deserialise it.

    Parameters
    ----------
    bucket:
        S3 bucket name, e.g. ``"explain-model-bucket-west2"``.
    key:
        S3 object key, e.g.
        ``"models/pkl/avalanche-logistic-2026-03-30/logistic_avalanche.pkl"``.

    Returns
    -------
    dict with keys:
        ``pipeline``        – fitted sklearn Pipeline (preprocessor + classifier)
        ``feature_columns`` – all column names present at training time
        ``numeric_features``– the 7 features the pipeline actually uses
        ``metrics``         – evaluation metrics recorded during training
        ``best_params``     – best hyperparameters found by GridSearchCV

    Raises
    ------
    botocore.exceptions.ClientError
        If the object does not exist or the caller lacks s3:GetObject permission.
    """
    local_path = Path("/tmp") / Path(key).name
    s3 = boto3.client("s3")
    print(f"Downloading s3://{bucket}/{key} -> {local_path} ...")
    s3.download_file(bucket, key, str(local_path))
    model_data = joblib.load(local_path)
    print("Model loaded successfully.")
    print(f"  Training metrics: {model_data.get('metrics', {})}")
    return model_data


# ── Step 2: Parse input ───────────────────────────────────────────────────────

def input_fn(payload: str | list | dict) -> pd.DataFrame:
    """Parse and validate the inference payload into a DataFrame.

    Accepts three input forms so the function is easy to call from both
    a test harness and a Lambda handler:

    - JSON string:  ``'[{"elevation": 3200, ...}]'``
    - Python dict:  ``{"elevation": 3200, ...}``   (single observation)
    - Python list:  ``[{"elevation": 3200, ...}, ...]``

    Required features (all numeric):
        elevation        – metres above sea level
        slope            – slope angle in degrees
        aspect_degrees   – compass bearing of the slope face (0–360)
        snow_depth       – total snowpack depth in cm
        new_snow_24h     – fresh snow in the last 24 h, in cm
        temp             – air temperature in °C
        snow_ratio       – snow_depth / SWE (snow water equivalent)

    Returns
    -------
    pd.DataFrame with one row per observation.

    Raises
    ------
    ValueError
        If any required feature is missing from the payload.
    """
    if isinstance(payload, str):
        payload = json.loads(payload)

    if isinstance(payload, dict):
        payload = [payload]

    df = pd.DataFrame(payload)

    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Payload is missing required features: {missing}\n"
            f"Required: {SELECTED_FEATURES}"
        )

    return df


# ── Step 3: Predict ───────────────────────────────────────────────────────────

def predict_fn(df: pd.DataFrame, model_data: dict) -> list[dict]:
    """Run the sklearn pipeline and return probability + risk class per row.

    The pipeline internally handles:
      - Median imputation for any NaN values
      - RobustScaler normalisation
      - Logistic Regression with recall-optimised hyperparameters

    Parameters
    ----------
    df:
        DataFrame produced by ``input_fn``.
    model_data:
        Dict returned by ``load_model_from_s3``.

    Returns
    -------
    List of dicts, one per input row::

        [
            {"probability": 0.73, "risk_class": "High"},
            ...
        ]

    Risk classes and their probability thresholds:
        Low          – p < 0.20
        Moderate     – 0.20 ≤ p < 0.40
        Considerable – 0.40 ≤ p < 0.60
        High         – 0.60 ≤ p < 0.80
        Extreme      – p ≥ 0.80
    """
    pipeline = model_data["pipeline"]
    proba = pipeline.predict_proba(df)[:, 1]

    return [
        {"probability": round(float(p), 4), "risk_class": pred_class(float(p))}
        for p in proba
    ]


# ── Step 4: Format output ─────────────────────────────────────────────────────

def output_fn(predictions: list[dict]) -> str:
    """Serialise predictions to a JSON string.

    Output shape::

        {
            "predictions": [
                {"probability": 0.73, "risk_class": "High"},
                ...
            ]
        }

    In a Lambda function this string becomes the ``body`` of the HTTP response.
    """
    return json.dumps({"predictions": predictions}, indent=2)


# ── Convenience: full pipeline in one call ────────────────────────────────────

def run_inference(model_data: dict, payload: str | list | dict) -> str:
    """Run the full input → predict → output pipeline.

    This is the single function to copy into Lambda (after replacing
    ``load_model_from_s3`` with a cached version — see the Lambda skeleton
    below).

    Parameters
    ----------
    model_data:
        Loaded model dict from ``load_model_from_s3``.
    payload:
        Raw request body (JSON string, dict, or list of dicts).

    Returns
    -------
    JSON string ready to send back as an HTTP response body.
    """
    df = input_fn(payload)
    predictions = predict_fn(df, model_data)
    return output_fn(predictions)


# ── Lambda skeleton ───────────────────────────────────────────────────────────
#
# Copy the block below into your Lambda function.  The module-level load means
# the model is fetched from S3 only on cold starts; warm invocations reuse the
# cached object.
#
# Required Lambda environment variables:
#   MODEL_BUCKET  – e.g. "explain-model-bucket-west2"
#   MODEL_S3_KEY  – e.g. "models/pkl/avalanche-logistic-2026-03-30/logistic_avalanche.pkl"
#
# ─────────────────────────────────────────────────────────────────────────────
#
#   import os, json
#   from test_inference import load_model_from_s3, run_inference
#
#   # Loaded once per container — reused on warm invocations
#   _MODEL = load_model_from_s3(
#       bucket=os.environ["MODEL_BUCKET"],
#       key=os.environ["MODEL_S3_KEY"],
#   )
#
#   def lambda_handler(event: dict, context) -> dict:
#       """AWS Lambda entry point.
#
#       Triggered by API Gateway (proxy integration).
#       event["body"] must be a JSON string matching the input contract
#       documented on input_fn() above.
#       """
#       try:
#           body = event.get("body") or json.dumps(event)  # support direct test invocations
#           result = run_inference(_MODEL, body)
#           return {"statusCode": 200, "body": result}
#       except ValueError as exc:
#           return {"statusCode": 400, "body": json.dumps({"error": str(exc)})}
#       except Exception as exc:
#           return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}
#
# ─────────────────────────────────────────────────────────────────────────────


# ── Built-in test suite ───────────────────────────────────────────────────────

# Representative samples — one per expected risk class.
# Add more rows here to stress-test edge cases (missing features, extreme values, etc.)
_TEST_CASES: list[dict[str, Any]] = [
    {
        "label": "High-risk: steep south-facing slope, heavy fresh snow",
        "payload": {
            "elevation": 3500,
            "slope": 42,
            "aspect_degrees": 180,
            "snow_depth": 200,
            "new_snow_24h": 60,
            "temp": -3,
            "snow_ratio": 6.0,
        },
    },
    {
        "label": "Low-risk: gentle north-facing slope, shallow snowpack",
        "payload": {
            "elevation": 2000,
            "slope": 15,
            "aspect_degrees": 0,
            "snow_depth": 30,
            "new_snow_24h": 2,
            "temp": -15,
            "snow_ratio": 1.2,
        },
    },
    {
        "label": "Moderate-risk: mid-elevation, moderate fresh snow",
        "payload": {
            "elevation": 2800,
            "slope": 28,
            "aspect_degrees": 135,
            "snow_depth": 90,
            "new_snow_24h": 20,
            "temp": -7,
            "snow_ratio": 3.0,
        },
    },
    {
        "label": "Batch: two observations in one request",
        "payload": [
            {
                "elevation": 3200,
                "slope": 35,
                "aspect_degrees": 200,
                "snow_depth": 150,
                "new_snow_24h": 45,
                "temp": -2,
                "snow_ratio": 5.0,
            },
            {
                "elevation": 1800,
                "slope": 10,
                "aspect_degrees": 45,
                "snow_depth": 20,
                "new_snow_24h": 0,
                "temp": -20,
                "snow_ratio": 0.8,
            },
        ],
    },
]


def run_test_suite(model_data: dict) -> None:
    """Execute all built-in test cases and print results."""
    print("\n" + "=" * 60)
    print("INFERENCE TEST SUITE")
    print("=" * 60)

    for i, case in enumerate(_TEST_CASES, 1):
        print(f"\n[Test {i}] {case['label']}")
        print("-" * 50)
        result = run_inference(model_data, case["payload"])
        parsed = json.loads(result)
        for pred in parsed["predictions"]:
            print(f"  probability : {pred['probability']:.4f}")
            print(f"  risk_class  : {pred['risk_class']}")

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a .pkl model from S3 and run inference tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run the built-in test suite\n"
            "  python test_inference.py \\\n"
            "      --bucket explain-model-bucket-west2 \\\n"
            "      --key    models/pkl/<job-name>/logistic_avalanche.pkl\n\n"
            "  # Predict from a custom payload\n"
            "  python test_inference.py \\\n"
            "      --bucket explain-model-bucket-west2 \\\n"
            "      --key    models/pkl/<job-name>/logistic_avalanche.pkl \\\n"
            '      --payload \'{"elevation":3200,"slope":35,"aspect_degrees":180,\n'
            '                  "snow_depth":120,"new_snow_24h":30,"temp":-5,"snow_ratio":4.0}\''
        ),
    )
    parser.add_argument(
        "--bucket",
        default="explain-model-bucket-west2",
        help="S3 bucket containing the .pkl artifact (default: explain-model-bucket-west2)",
    )
    parser.add_argument(
        "--key",
        required=True,
        help="S3 key to the .pkl file, e.g. models/pkl/<job-name>/logistic_avalanche.pkl",
    )
    parser.add_argument(
        "--payload",
        default=None,
        help=(
            "Optional JSON string to predict on instead of the built-in test suite. "
            "Accepts a single object or a JSON array of objects."
        ),
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
        help="AWS region (default: us-east-1 or AWS_DEFAULT_REGION env var)",
    )
    args = parser.parse_args()

    # Ensure boto3 uses the right region
    boto3.setup_default_session(region_name=args.region)

    model_data = load_model_from_s3(args.bucket, args.key)

    if args.payload:
        print("\nCustom payload result:")
        print(run_inference(model_data, args.payload))
    else:
        run_test_suite(model_data)


if __name__ == "__main__":
    main()
