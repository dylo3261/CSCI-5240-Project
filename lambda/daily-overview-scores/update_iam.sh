#!/bin/bash
# update_iam.sh — Add S3 write permissions for grid output
# Run this once before deploying the grid Lambda

set -e

aws iam put-role-policy \
    --role-name explainable-inference-role \
    --policy-name LambdaAccessPolicy \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": [
                    "arn:aws:s3:::explain-model-bucket-west2/*",
                    "arn:aws:s3:::daily-weather-data-csv-bucket/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": ["s3:PutObject"],
                "Resource": [
                    "arn:aws:s3:::location-data-app-bucket/daily-grid-scores/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:GetQueueAttributes"
                ],
                "Resource": "arn:aws:sqs:us-west-2:029953330549:explainable-inference-queue"
            }
        ]
    }'

echo "✅ IAM policy updated with S3 write access for location-data-app-bucket/daily-grid-scores/"