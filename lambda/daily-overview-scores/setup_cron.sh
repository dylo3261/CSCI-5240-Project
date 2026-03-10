#!/bin/bash
# setup_cron.sh — Create EventBridge rule to run grid Lambda every morning at 7am MST (2pm UTC)
# Usage: ./setup_cron.sh

set -e

AWS_REGION="us-west-2"
AWS_ACCOUNT_ID="029953330549"
FUNCTION_NAME="regional-risk-grid"
RULE_NAME="daily-risk-grid"

echo "📅 Creating EventBridge cron rule (7am MST / 2pm UTC daily)..."
aws events put-rule \
    --name $RULE_NAME \
    --schedule-expression "cron(0 14 * * ? *)" \
    --state ENABLED \
    --description "Trigger regional risk grid Lambda every morning at 7am MST" \
    --region $AWS_REGION

echo "🔗 Adding Lambda as target..."
aws events put-targets \
    --rule $RULE_NAME \
    --targets "Id=1,Arn=arn:aws:lambda:$AWS_REGION:$AWS_ACCOUNT_ID:function:$FUNCTION_NAME,Input='{\"cell_size_mi\":2}'" \
    --region $AWS_REGION

echo "🔓 Granting EventBridge permission to invoke Lambda..."
aws lambda add-permission \
    --function-name $FUNCTION_NAME \
    --statement-id eventbridge-daily-grid \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn arn:aws:events:$AWS_REGION:$AWS_ACCOUNT_ID:rule/$RULE_NAME \
    --region $AWS_REGION 2>/dev/null || echo "   (Permission already exists - skipping)"

echo "✅ Cron rule created: $RULE_NAME"
echo "   Schedule: 7:00 AM MST daily (2:00 PM UTC)"
echo "   Target: $FUNCTION_NAME with 2mi grid (~13,900 cells)"
echo ""
echo "   To test manually:"
echo "   aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"cell_size_mi\": 10}' --cli-binary-format raw-in-base64-out --region $AWS_REGION /tmp/grid_test.json && cat /tmp/grid_test.json | python3 -m json.tool"