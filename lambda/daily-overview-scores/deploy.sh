#!/bin/bash
# deploy.sh — Build, push, and deploy the regional risk grid Lambda
# Usage: ./deploy.sh

set -e

AWS_ACCOUNT_ID="029953330549"
AWS_REGION="us-west-2"
ECR_REPO="regional-risk-grid"
FUNCTION_NAME="regional-risk-grid"
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest"
ROLE_ARN="arn:aws:iam::$AWS_ACCOUNT_ID:role/explainable-inference-role"

echo "🔨 Building Docker image..."
echo "   Note: First build takes 20-30 minutes (compiling GDAL from source)"
echo "   Subsequent builds are much faster (seconds)"
docker build -t $ECR_REPO .

echo ""
echo "🔑 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo ""
echo "📦 Creating ECR repo (if needed)..."
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION 2>/dev/null || \
    echo "   Repository already exists, continuing..."

echo ""
echo "🏷️  Tagging image..."
docker tag $ECR_REPO:latest $IMAGE_URI

echo ""
echo "📤 Pushing to ECR..."
docker push $IMAGE_URI

echo ""
echo "🔄 Creating/updating Lambda..."
if aws lambda get-function --function-name $FUNCTION_NAME --region $AWS_REGION > /dev/null 2>&1; then
    echo "   Updating existing function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --image-uri $IMAGE_URI \
        --region $AWS_REGION > /dev/null
    
    echo "   ⏳ Waiting for update to complete..."
    aws lambda wait function-updated-v2 --function-name $FUNCTION_NAME --region $AWS_REGION
    
    echo "   Updating function configuration..."
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --timeout 900 \
        --memory-size 2048 \
        --environment "Variables={MODEL_BUCKET=explain-model-bucket-west2,DATA_BUCKET=daily-weather-data-csv-bucket,OUTPUT_BUCKET=location-data-app-bucket,OUTPUT_PREFIX=daily-grid-scores/}" \
        --region $AWS_REGION > /dev/null
else
    echo "   Creating new function..."
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --package-type Image \
        --code ImageUri=$IMAGE_URI \
        --role $ROLE_ARN \
        --timeout 900 \
        --memory-size 2048 \
        --architectures arm64 \
        --environment "Variables={MODEL_BUCKET=explain-model-bucket-west2,DATA_BUCKET=daily-weather-data-csv-bucket,OUTPUT_BUCKET=location-data-app-bucket,OUTPUT_PREFIX=daily-grid-scores/}" \
        --region $AWS_REGION > /dev/null
fi

echo ""
echo "⏳ Waiting for Lambda to become active..."
aws lambda wait function-active-v2 --function-name $FUNCTION_NAME --region $AWS_REGION

echo ""
echo "🧪 Testing with small grid (10mi cells)..."
echo "   Expected runtime: ~30-60 seconds"
echo ""

aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload '{"cell_size_mi": 10}' \
    --cli-binary-format raw-in-base64-out \
    --region $AWS_REGION \
    /tmp/grid_response.json > /dev/null

STATUS=$(cat /tmp/grid_response.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('statusCode', 'error'))" 2>/dev/null || echo "error")

echo ""
if [ "$STATUS" = "200" ]; then
    echo "✅ Deploy successful!"
    echo ""
    echo "Response:"
    cat /tmp/grid_response.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(json.loads(d['body']), indent=2))"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Next steps:"
    echo "  1. Check S3 output: aws s3 ls s3://explain-model-bucket-west2/daily-grid-scores/latest/"
    echo "  2. Set up cron: ./setup_cron.sh"
    echo "  3. Test full grid: aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"cell_size_mi\": 5}' --cli-binary-format raw-in-base64-out --region $AWS_REGION /tmp/full_grid.json"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "❌ Deploy completed but test invocation failed"
    echo ""
    echo "Error response:"
    cat /tmp/grid_response.json | python3 -m json.tool
    echo ""
    echo "Check CloudWatch Logs:"
    echo "  aws logs tail /aws/lambda/$FUNCTION_NAME --follow --region $AWS_REGION"
    exit 1
fi