#!/bin/bash
# deploy.sh — Build, push, and update Lambda in one command
# Usage: ./deploy.sh

set -e

AWS_ACCOUNT_ID="029953330549"
AWS_REGION="us-west-2"
ECR_REPO="explainable-inference"
FUNCTION_NAME="explainable-inference"
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest"

echo "🔨 Building Docker image..."
docker build -t $ECR_REPO .

echo "🔑 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "🏷️  Tagging image..."
docker tag $ECR_REPO:latest $IMAGE_URI

echo "📤 Pushing to ECR..."
docker push $IMAGE_URI

echo "🔄 Updating Lambda..."
aws lambda update-function-code \
    --function-name $FUNCTION_NAME \
    --image-uri $IMAGE_URI \
    --region $AWS_REGION > /dev/null

echo "⏳ Waiting for update to complete..."
aws lambda wait function-updated \
    --function-name $FUNCTION_NAME \
    --region $AWS_REGION 2>/dev/null || sleep 15

echo "🧪 Testing..."
aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload '{"latitude": 39.6, "longitude": -105.8}' \
    --cli-binary-format raw-in-base64-out \
    --region $AWS_REGION \
    /tmp/response.json > /dev/null

STATUS=$(cat /tmp/response.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('statusCode', 'error'))")

if [ "$STATUS" = "200" ]; then
    echo "✅ Deploy successful! Response:"
    cat /tmp/response.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(json.loads(d['body']), indent=2))"
else
    echo "❌ Deploy completed but test failed:"
    cat /tmp/response.json | python3 -m json.tool
fi