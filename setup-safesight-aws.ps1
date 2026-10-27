# SafeSight AI - Complete AWS Setup Script
# Run in PowerShell as administrator (or with appropriate IAM permissions)

$InstanceArn = "arn:aws:sso:::instance/ssoins-6595a16c1203593f"
$IdentityStoreId = "d-9f675b8002"
$AccountId = "513483285726"
$Region = "ap-south-1"
$BucketName = "safesight-ai-evidence"
$LogGroupName = "/aws/industrial-safety-inspector"
$ProfileName = "safesight"
$PermissionSetName = "SafeSightAI-Access"

Write-Host "=== Step 1: Create Permission Set ===" -ForegroundColor Cyan
$ps = aws sso-admin create-permission-set `
  --instance-arn $InstanceArn `
  --name $PermissionSetName `
  --description "Least-privilege access for SafeSight AI application" `
  --session-duration PT1H `
  --output json | ConvertFrom-Json
$PermissionSetArn = $ps.PermissionSet.PermissionSetArn
Write-Host "Created: $PermissionSetArn" -ForegroundColor Green

Write-Host "=== Step 2: Create Inline Policy ===" -ForegroundColor Cyan
$policyJson = @'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3EvidenceAccess",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::safesight-ai-evidence",
        "arn:aws:s3:::safesight-ai-evidence/*"
      ]
    },
    {
      "Sid": "CloudWatchLogsAccess",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
        "logs:DescribeLogGroups", "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:ap-south-1:513483285726:log-group:/aws/industrial-safety-inspector:*"
    },
    {
      "Sid": "CloudWatchMetricsAccess",
      "Effect": "Allow",
      "Action": ["cloudwatch:PutMetricData", "cloudwatch:ListMetrics"],
      "Resource": "*",
      "Condition": { "StringEquals": { "cloudwatch:namespace": "SafeSightAI" } }
    }
  ]
}
'@
$policyJson | Set-Content -Encoding UTF8 safesight-policy.json

aws sso-admin put-inline-policy-to-permission-set `
  --instance-arn $InstanceArn `
  --permission-set-arn $PermissionSetArn `
  --inline-policy file://safesight-policy.json
Write-Host "Policy attached" -ForegroundColor Green

Write-Host "=== Step 3: Provision Permission Set ===" -ForegroundColor Cyan
aws sso-admin provision-permission-set `
  --instance-arn $InstanceArn `
  --permission-set-arn $PermissionSetArn `
  --target-type ALL_PROVISIONED_ACCOUNTS
Write-Host "Provisioned" -ForegroundColor Green

Write-Host "=== Step 4: Assign to User ===" -ForegroundColor Cyan
$UserId = aws identitystore list-users `
  --identity-store-id $IdentityStoreId `
  --query "Users[?UserName=='safesight'].UserId" `
  --output text

if (-not $UserId) {
    Write-Error "User 'safesight' not found in Identity Store"
    exit 1
}
Write-Host "User ID: $UserId"

aws sso-admin create-account-assignment `
  --instance-arn $InstanceArn `
  --target-id $AccountId `
  --target-type AWS_ACCOUNT `
  --permission-set-arn $PermissionSetArn `
  --principal-type USER `
  --principal-id $UserId
Write-Host "Assigned to user" -ForegroundColor Green

Write-Host "=== Step 5: Re-authenticate ===" -ForegroundColor Cyan
aws sso login --profile $ProfileName
Write-Host "Re-authenticated" -ForegroundColor Green

Write-Host "=== Step 6: Verify Identity ===" -ForegroundColor Cyan
aws sts get-caller-identity --profile $ProfileName

Write-Host "=== Step 7: Create S3 Bucket ===" -ForegroundColor Cyan
aws s3api create-bucket `
  --bucket $BucketName `
  --region $Region `
  --create-bucket-configuration LocationConstraint=$Region `
  --profile $ProfileName
Write-Host "Bucket created" -ForegroundColor Green

Write-Host "=== Step 8: Run Integration Tests ===" -ForegroundColor Cyan
$env:AWS_PROFILE = $ProfileName
$env:AWS_REGION = $Region
$env:AWS_S3_BUCKET = $BucketName
$env:AWS_CLOUDWATCH_LOG_GROUP = $LogGroupName

cd E:\opencv5-pro\opencv-agentic-vision
.venv\Scripts\python.exe -m pytest tests\test_aws.py -m integration -v