# Deployment Guide — SafeSight AI

## Overview

This guide covers deploying SafeSight AI in various environments: local development, Docker, AWS ECS Fargate, and AWS Lambda.

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for containerized deployments)
- AWS CLI configured (for AWS deployments)
- YOLO11n ONNX model at `models/yolo11n.onnx`

## Local Development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set optional AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
export AWS_S3_BUCKET=your-evidence-bucket
export AWS_CLOUDWATCH_LOG_GROUP=/aws/industrial-safety-inspector

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens http://localhost:5173
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | No | — | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | No | — | AWS secret key |
| `AWS_REGION` | No | `us-east-1` | AWS region |
| `AWS_S3_BUCKET` | No | — | S3 bucket for evidence |
| `AWS_CLOUDWATCH_LOG_GROUP` | No | `/aws/industrial-safety-inspector` | CW log group |
| `MODEL_PATH` | No | `models/yolo11n.onnx` | Path to ONNX model |

## Docker Deployment

### Build Image

```bash
docker build -t safesight-ai .
```

### Run Container

```bash
docker run -d \
  --name safesight-backend \
  -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_REGION \
  -e AWS_S3_BUCKET \
  -e AWS_CLOUDWATCH_LOG_GROUP \
  safesight-ai
```

### Docker Compose (Full Stack)

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_REGION=us-east-1
      - AWS_S3_BUCKET
      - AWS_CLOUDWATCH_LOG_GROUP=/aws/industrial-safety-inspector
    volumes:
      - ./incidents:/app/incidents
      - ./temp:/app/temp
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "5173:5173"
    environment:
      - VITE_API_URL=http://localhost:8000
    depends_on:
      - backend
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## AWS ECS Fargate (Recommended for Production)

### 1. Push Image to ECR

```bash
aws ecr create-repository --repository-name safesight-ai --region us-east-1

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker tag safesight-ai:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/safesight-ai:latest

docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/safesight-ai:latest
```

### 2. Create Task Definition

```json
{
  "family": "safesight-ai",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::<account-id>:role/safesightTaskRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/safesight-ai:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [
        {"name": "AWS_REGION", "value": "us-east-1"},
        {"name": "AWS_S3_BUCKET", "value": "safesight-evidence"},
        {"name": "AWS_CLOUDWATCH_LOG_GROUP", "value": "/aws/industrial-safety-inspector"}
      ],
      "secrets": [
        {"name": "AWS_ACCESS_KEY_ID", "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:safesight-aws-credentials:access_key::"},
        {"name": "AWS_SECRET_ACCESS_KEY", "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:safesight-aws-credentials:secret_key::"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/safesight-ai",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "backend"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### 3. Create IAM Roles

**Task Execution Role** (`ecsTaskExecutionRole`):
- `AmazonECSTaskExecutionRolePolicy`
- Secrets Manager read access for credentials

**Task Role** (`safesightTaskRole`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::safesight-evidence/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:<account-id>:log-group:/aws/industrial-safety-inspector:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "cloudwatch:namespace": "SafeSightAI"
        }
      }
    }
  ]
}
```

### 4. Create Service

```bash
aws ecs create-service \
  --cluster safesight-cluster \
  --service-name safesight-backend \
  --task-definition safesight-ai \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:us-east-1:<account-id>:targetgroup/safesight/xxx,containerName=backend,containerPort=8000"
```

### 5. Configure ALB

- Create Application Load Balancer
- Target group: HTTP port 8000, health check `/health`
- Listener: HTTPS (port 443) → target group
- ACM certificate for custom domain

## AWS Lambda (For Short Videos Only)

**Note**: Lambda has 15-minute timeout and 10GB memory limit. Only suitable for videos < 2 minutes.

### SAM Template

```yaml
# deploy/lambda-config.yml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: SafeSight AI Lambda Function

Parameters:
  ModelS3Bucket:
    Type: String
  ModelS3Key:
    Type: String

Resources:
  SafesightFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: safesight-analyze
      Runtime: python3.11
      Handler: lambda_handler.handler
      CodeUri: backend/
      Timeout: 900
      MemorySize: 3008
      EphemeralStorage:
        Size: 2048
      Environment:
        Variables:
          AWS_S3_BUCKET: !Ref EvidenceBucket
          AWS_CLOUDWATCH_LOG_GROUP: !Ref LogGroup
      Policies:
        - S3WritePolicy:
            BucketName: !Ref EvidenceBucket
        - CloudWatchPutMetricPolicy: {}
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /analyze
            Method: post
            RestApiId: !Ref ApiGateway

  EvidenceBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: safesight-evidence-prod
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256

  LogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: /aws/lambda/safesight-analyze
      RetentionInDays: 30

  ApiGateway:
    Type: AWS::Serverless::Api
    Properties:
      StageName: prod
      Cors:
        AllowMethods: "'*'"
        AllowHeaders: "'*'"
        AllowOrigin: "'*'"
```

### Lambda Handler

```python
# backend/lambda_handler.py
import json
import base64
import tempfile
import uuid
from main import app
from mangum import Mangum

handler = Mangum(app, lifespan="off")

def lambda_handler(event, context):
    # Handle base64 encoded video from API Gateway
    if event.get("isBase64Encoded"):
        body = base64.b64decode(event["body"])
    else:
        body = event["body"].encode() if isinstance(event["body"], str) else event["body"]

    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(body)
        temp_path = f.name

    # Create modified event for FastAPI
    # (simplified - use mangum for full ASGI support)
    return handler(event, context)
```

### Deploy

```bash
sam build --template deploy/lambda-config.yml
sam deploy --guided \
  --stack-name safesight-ai \
  --parameter-overrides ModelS3Bucket=my-models ModelS3Key=yolo11n.onnx \
  --capabilities CAPABILITY_IAM
```

## Frontend Deployment (Static Hosting)

### Build

```bash
cd frontend
npm run build
# Output in dist/
```

### Deploy to S3 + CloudFront

```bash
aws s3 sync dist/ s3://safesight-frontend --delete
aws cloudfront create-invalidation --distribution-id EXXX --paths "/*"
```

### Or Deploy to Vercel/Netlify

```bash
# Vercel
npx vercel --prod

# Netlify
npx netlify deploy --prod --dir=dist
```

## Configuration by Environment

| Setting | Development | Staging | Production |
|---------|-------------|---------|------------|
| Replicas | 1 | 2 | 3+ |
| CPU | 0.5 vCPU | 1 vCPU | 2 vCPU |
| Memory | 1 GB | 2 GB | 4 GB |
| Log Level | DEBUG | INFO | WARN |
| AWS Credentials | Local env | Parameter Store | Secrets Manager |
| Model | Local file | EFS mount | EFS mount |

## Monitoring & Observability

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Expected: {"status": "ok", "timestamp": 1234567890.123}
```

### Key Metrics (CloudWatch)

- `SafeSightAI/ProcessingLatencyMs` — P50, P95, P99
- `SafeSightAI/FrameProcessingTimeMs` — Per-frame latency
- `SafeSightAI/DetectionsPerVideo` — Detection count
- `SafeSightAI/IncidentsCreated` — Incident count
- `SafeSightAI/AWSErrors` — S3/CW failures

### Alerts

```bash
# High latency alert
aws cloudwatch put-metric-alarm \
  --alarm-name "SafeSight-High-Latency" \
  --metric-name ProcessingLatencyMs \
  --namespace SafeSightAI \
  --statistic Average \
  --period 300 \
  --threshold 5000 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2

# Error rate alert
aws cloudwatch put-metric-alarm \
  --alarm-name "SafeSight-AWS-Errors" \
  --metric-name AWSErrors \
  --namespace SafeSightAI \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Ensure `models/yolo11n.onnx` exists in container |
| CORS errors | Add frontend origin to `allow_origins` in main.py |
| S3 upload fails | Check IAM permissions, bucket exists, region matches |
| CloudWatch metrics missing | Verify CW log group exists, IAM has `PutMetricData` |
| Video timeout | Increase timeout, reduce video length, add GPU |
| OOM errors | Increase memory, reduce batch size, use smaller model |

## Scaling Considerations

- **Horizontal**: ECS service auto-scaling on CPU > 70% or request count
- **Video processing**: Each video is independent — stateless, easy to parallelize
- **GPU**: For higher FPS, use `gpu` instance types with CUDA-enabled OpenCV
- **Queue**: For high throughput, add SQS queue + worker fleet instead of direct API

## Security Checklist

- [ ] No credentials in code or Docker images
- [ ] Secrets in Secrets Manager / Parameter Store
- [ ] S3 bucket encryption enabled (SSE-S3 or SSE-KMS)
- [ ] CloudWatch log retention configured
- [ ] VPC endpoints for S3/CloudWatch (no NAT gateway)
- [ ] ALB with WAF rules for API protection
- [ ] Regular image vulnerability scans
- [ ] Incident data access logged and auditable