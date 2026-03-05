# Deploying the Real-Time Reactions Feature

This guide covers deploying the WebSocket backend via AWS SAM and wiring the generated endpoint into the frontend.

---

## Prerequisites

Verify these are installed and configured before starting.

```bash
aws --version          # AWS CLI v2
sam --version          # SAM CLI >= 1.100.0
node --version         # Node.js >= 18
python3.11 --version   # Must be on PATH for sam build
```

Configure the AWS CLI with credentials that have permission to create Lambda, API Gateway, DynamoDB, IAM roles, and CloudFormation stacks:

```bash
aws configure
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region name:   us-west-2
# Default output format: json
```

Confirm the correct identity is active:

```bash
aws sts get-caller-identity
```

---

## Step 1 — Build and Deploy the SAM Stack

Run both commands from the **repository root** (where `template.yaml` lives).

```bash
sam build
```

SAM reads `template.yaml`, installs `lambda/websocket-handler/requirements.txt` using Python 3.11, and writes the packaged output to `.aws-sam/build/`.

Then deploy:

```bash
sam deploy --guided
```

Answer each prompt as follows:

```
Stack Name [sam-app]:                        avalanche-reactions
AWS Region [us-east-1]:                      us-west-2
Confirm changes before deploy [y/N]:         y
Allow SAM CLI IAM role creation [Y/n]:       Y
Disable rollback [y/N]:                      N
WebSocketHandlerFunction ... may not have authorization defined, Is this okay? [y/N]:  y
  (SAM asks this once per route — answer y for all three)
Save arguments to configuration file [Y/n]:  Y
SAM configuration file [samconfig.toml]:     samconfig.toml
SAM configuration environment [default]:     default
```

SAM will print a changeset and ask:

```
Deploy this changeset? [y/N]: y
```

Wait for `Successfully created/updated stack - avalanche-reactions in us-west-2`.

> **Subsequent deploys** (after code changes): `sam build && sam deploy` — no `--guided` needed, `samconfig.toml` stores the settings.

---

## Step 2 — Capture the Stack Outputs

Once the deploy finishes, retrieve the WebSocket endpoint:

```bash
aws cloudformation describe-stacks \
  --stack-name avalanche-reactions \
  --query "Stacks[0].Outputs" \
  --output table \
  --region us-west-2
```

You will see output like:

```
-------------------------------------------------------------
|                      DescribeStacks                       |
+-----------------------------+-----------------------------+
|          OutputKey          |         OutputValue         |
+-----------------------------+-----------------------------+
| WebSocketApiEndpoint        | wss://abc123xyz.execute-api.us-west-2.amazonaws.com/prod |
| WebSocketApiId              | abc123xyz                   |
| ActiveConnectionsTableName  | ActiveConnectionsTable      |
| ActiveConnectionsTableArn   | arn:aws:dynamodb:...        |
| UserReactionsTableName      | UserReactionsTable          |
| UserReactionsTableArn       | arn:aws:dynamodb:...        |
+-----------------------------+-----------------------------+
```

Save the `WebSocketApiEndpoint` value — you need it in the next step.

---

## Step 3 — Wire the WebSocket URL into the Frontend

Open `frontend/src/pages/Map.tsx` and replace line 8:

```ts
// Before
const WS_URL = "wss://YOUR_WEBSOCKET_API_ID.execute-api.us-west-2.amazonaws.com/prod";

// After — paste your actual endpoint from the stack output
const WS_URL = "wss://abc123xyz.execute-api.us-west-2.amazonaws.com/prod";
```

Or run this one-liner from the repo root (replace the URL with your real endpoint):

```bash
sed -i 's|wss://YOUR_WEBSOCKET_API_ID.execute-api.us-west-2.amazonaws.com/prod|wss://abc123xyz.execute-api.us-west-2.amazonaws.com/prod|' \
  frontend/src/pages/Map.tsx
```

Verify the change took effect:

```bash
grep "WS_URL" frontend/src/pages/Map.tsx
```

---

## Step 4 — Run the Frontend

### Local development

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Navigate to `/map` and log in via Cognito. Once authenticated, the browser opens a WebSocket connection automatically. Enter coordinates, then click a reaction button (❄️ / ✨ / ⚠️) to broadcast to all connected users.

### Production build

```bash
cd frontend
npm run build   # outputs to frontend/dist/
```

Deploy `frontend/dist/` to your static host (S3 + CloudFront, Netlify, etc.).

---

## Verification

### Check the WebSocket connection in the browser

Open DevTools → Network → filter by **WS**. You should see the `prod` WebSocket connection with status `101 Switching Protocols` after logging in.

### Manually test the backend with wscat

```bash
npm install -g wscat

# Connect
wscat -c wss://abc123xyz.execute-api.us-west-2.amazonaws.com/prod

# Send a reaction (paste after the > prompt)
{"action":"sendReaction","type":"icy","lat":39.7392,"lng":-104.9903}
```

The message should be broadcast back to all open connections, and a new item should appear in DynamoDB.

### Inspect DynamoDB

```bash
# See active connections
aws dynamodb scan --table-name ActiveConnectionsTable --region us-west-2

# See stored reactions
aws dynamodb scan --table-name UserReactionsTable --region us-west-2
```

---

## Teardown

To delete all provisioned resources:

```bash
sam delete --stack-name avalanche-reactions --region us-west-2
```

This removes the Lambda, API Gateway WebSocket API, both DynamoDB tables, and all associated IAM roles.
