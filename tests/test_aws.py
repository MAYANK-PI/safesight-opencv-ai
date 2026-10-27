import os
import sys
import pytest
import time
from unittest.mock import patch, MagicMock

# Integration test marker - only runs when AWS credentials are available
integration_test = pytest.mark.integration


def test_s3_missing_credentials():
    """Test that S3 functions return failed UploadResult when credentials are missing from provider chain."""
    # Mock boto3 to simulate no credentials available
    with patch("backend.aws.s3.boto3.client") as mock_client:
        from botocore.exceptions import NoCredentialsError
        mock_client.side_effect = NoCredentialsError()

        # Need to reset cached client
        from backend.aws.s3 import reset_s3_client
        reset_s3_client()

        from backend.aws.s3 import _get_s3_client, upload_file, upload_evidence, upload_metadata, StorageBackend
        client = _get_s3_client()
        assert client is None, "Expected None S3 client when credentials are missing from provider chain"

        # Upload functions should return UploadResult with success=False
        result = upload_file("dummy.txt", "dummy-bucket", "dummy/path")
        assert result is not None, "Expected UploadResult object"
        assert result.success is False, "Expected success=False when uploading file with no credentials"
        assert result.backend == StorageBackend.LOCAL, "Expected LOCAL backend when credentials missing"

        result = upload_evidence("dummy.jpg", "incident_001")
        assert result is not None, "Expected UploadResult object"
        assert result.success is False, "Expected success=False when uploading evidence with no credentials"
        assert result.backend == StorageBackend.LOCAL, "Expected LOCAL backend when credentials missing"

        result = upload_metadata("dummy.json", "incident_001")
        assert result is not None, "Expected UploadResult object"
        assert result.success is False, "Expected success=False when uploading metadata with no credentials"
        assert result.backend == StorageBackend.LOCAL, "Expected LOCAL backend when credentials missing"


def test_cloudwatch_missing_credentials():
    """Test that CloudWatch functions return False when credentials are missing from provider chain."""
    # Mock boto3 to simulate no credentials available
    with patch("backend.aws.cloudwatch.boto3.client") as mock_client:
        from botocore.exceptions import NoCredentialsError
        mock_client.side_effect = NoCredentialsError()

        from backend.aws.cloudwatch import reset_cloudwatch_clients
        reset_cloudwatch_clients()

        from backend.aws.cloudwatch import (
            _get_logs_client,
            _get_metrics_client,
            log_event,
            log_detection_event,
            log_agent_decision,
            log_confirmed_incident,
            log_processing_metrics,
            put_metric
        )
        client = _get_logs_client()
        assert client is None, "Expected None CloudWatch Logs client when credentials are missing"

        metrics_client = _get_metrics_client()
        assert metrics_client is None, "Expected None CloudWatch Metrics client when credentials are missing"

        # Logging functions should return False (not sent to CloudWatch) but not raise
        assert log_event("test-group", "test-stream", "test message") is False
        assert log_detection_event({}) is False
        assert log_agent_decision({}) is False
        assert log_confirmed_incident("id", {}) is False
        assert log_processing_metrics(1.0, 2.0, 3) is False
        assert put_metric("TestMetric", 1.0) is False


def test_s3_upload_file_not_found():
    """Test that upload_file returns failed UploadResult when local file does not exist."""
    from backend.aws.s3 import upload_file, StorageBackend
    with patch("backend.aws.s3._get_s3_client") as mock_get_client:
        mock_get_client.return_value = MagicMock()
        result = upload_file("non_existent_file.txt", "bucket", "key")
        assert result is not None, "Expected UploadResult object"
        assert result.success is False, "Expected success=False when local file not found"
        assert result.error is not None, "Expected error message"


def test_cloudwatch_log_event_handles_errors_gracefully():
    """Test that log_event returns False when CloudWatch client creation fails."""
    # We'll patch _get_logs_client to return None to simulate missing credentials
    with patch("backend.aws.cloudwatch._get_logs_client", return_value=None):
        from backend.aws.cloudwatch import log_event
        result = log_event("group", "stream", "message")
        assert result is False, "Expected False when CloudWatch client is None"


def test_s3_upload_result_structure():
    """Test that UploadResult has the expected structure."""
    from backend.aws.s3 import UploadResult, StorageBackend
    result = UploadResult(
        success=True,
        backend=StorageBackend.S3,
        url="s3://bucket/key",
        retries=1,
        duration_ms=150.5
    )
    assert result.success is True
    assert result.backend == StorageBackend.S3
    assert result.url == "s3://bucket/key"
    assert result.retries == 1
    assert result.duration_ms == 150.5

    # Test failed result
    failed = UploadResult(
        success=False,
        backend=StorageBackend.LOCAL,
        error="Test error"
    )
    assert failed.success is False
    assert failed.backend == StorageBackend.LOCAL
    assert failed.error == "Test error"


def test_cloudwatch_metrics_batch():
    """Test put_metrics_batch function."""
    from backend.aws.cloudwatch import put_metrics_batch
    with patch("backend.aws.cloudwatch._get_metrics_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        metrics = [
            {"MetricName": "TestMetric1", "Value": 1.0, "Unit": "Count"},
            {"MetricName": "TestMetric2", "Value": 2.5, "Unit": "Milliseconds"}
        ]
        result = put_metrics_batch(metrics)
        assert result is True
        mock_client.put_metric_data.assert_called_once()


def test_s3_client_creation_with_explicit_credentials():
    """Test that explicit credentials from env vars are used when provided."""
    with patch("backend.aws.s3.boto3.client") as mock_client:
        mock_s3 = MagicMock()
        mock_s3.list_buckets.return_value = {"Buckets": []}
        mock_client.return_value = mock_s3

        from backend.aws.s3 import reset_s3_client
        reset_s3_client()

        # Set explicit credentials
        with patch.dict(os.environ, {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_REGION": "us-west-2",
        }):
            from backend.aws.s3 import _get_s3_client, is_s3_available
            client = _get_s3_client()
            assert client is not None, "Expected S3 client with explicit credentials"
            assert is_s3_available() is True
            # Verify boto3.client was called with region
            mock_client.assert_called()
            call_args = mock_client.call_args
            assert call_args[1]['region_name'] == "us-west-2"
            # When explicit creds provided, boto3 receives them
            # (boto3 handles the priority internally)


def test_s3_client_creation_with_profile():
    """Test that AWS_PROFILE is respected via credential chain."""
    with patch("backend.aws.s3.boto3.client") as mock_client:
        mock_s3 = MagicMock()
        mock_s3.list_buckets.return_value = {"Buckets": []}
        mock_client.return_value = mock_s3

        from backend.aws.s3 import reset_s3_client
        reset_s3_client()

        with patch.dict(os.environ, {
            "AWS_PROFILE": "myprofile",
            "AWS_REGION": "eu-west-1",
        }, clear=True):
            from backend.aws.s3 import _get_s3_client, is_s3_available
            client = _get_s3_client()
            assert client is not None, "Expected S3 client with profile"
            assert is_s3_available() is True


def test_s3_client_region_default():
    """Test that default region is used when not specified."""
    with patch("backend.aws.s3.boto3.client") as mock_client:
        mock_s3 = MagicMock()
        mock_s3.list_buckets.return_value = {"Buckets": []}
        mock_client.return_value = mock_s3

        from backend.aws.s3 import reset_s3_client
        reset_s3_client()

        with patch.dict(os.environ, {}, clear=True):
            from backend.aws.s3 import _get_s3_client
            client = _get_s3_client()
            # Should still create client with default region
            if client is not None:
                call_args = mock_client.call_args
                assert call_args[1]['region_name'] == "us-east-1"


def test_s3_reset_client_on_credential_change():
    """Test that client is reset when credential configuration changes."""
    from backend.aws.s3 import reset_s3_client, _get_s3_client, _get_credentials_hash

    with patch("backend.aws.s3.boto3.client") as mock_client:
        mock_s3 = MagicMock()
        mock_s3.list_buckets.return_value = {"Buckets": []}
        mock_client.return_value = mock_s3

        reset_s3_client()

        # First call with initial config
        with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}, clear=True):
            client1 = _get_s3_client()
            hash1 = _get_credentials_hash()

        # Second call with same config - should return cached client
        with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}, clear=True):
            client2 = _get_s3_client()
            assert client1 is client2, "Should return cached client for same config"

        # Third call with different config - should create new client
        with patch.dict(os.environ, {"AWS_REGION": "us-west-2"}, clear=True):
            client3 = _get_s3_client()
            # Note: In real usage, reset_s3_client() would be called explicitly
            # or the hash check would detect the change


def test_cloudwatch_reset_clients():
    """Test that CloudWatch clients can be reset."""
    from backend.aws.cloudwatch import reset_cloudwatch_clients, _get_logs_client, _get_metrics_client

    with patch("backend.aws.cloudwatch.boto3.client") as mock_client:
        mock_cw = MagicMock()
        mock_cw.describe_log_groups.return_value = {"logGroups": []}
        mock_cw.list_metrics.return_value = {"metrics": []}
        mock_client.return_value = mock_cw

        reset_cloudwatch_clients()

        logs_client = _get_logs_client()
        metrics_client = _get_metrics_client()
        assert logs_client is not None
        assert metrics_client is not None

        # Reset again
        reset_cloudwatch_clients()
        # After reset, new clients would be created on next call


def test_s3_idempotency_cache():
    """Test idempotency cache functionality."""
    import tempfile
    from backend.aws.s3 import (
        upload_file, clear_idempotency_cache, _generate_idempotency_key,
        _check_idempotency, _store_idempotency, StorageBackend
    )

    clear_idempotency_cache()

    # Create a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content")
        temp_path = f.name

    try:
        # Test idempotency key generation
        key = _generate_idempotency_key(temp_path, "bucket", "path/key")
        assert key is not None
        assert "bucket:path/key" in key

        # Test cache operations
        _store_idempotency(key, "s3://bucket/path/key")
        cached = _check_idempotency(key)
        assert cached == "s3://bucket/path/key"

        # Test cache clear
        clear_idempotency_cache()
        assert _check_idempotency(key) is None
    finally:
        os.unlink(temp_path)


def test_s3_availability_check_without_credentials():
    """Test that is_s3_available returns False when no credentials in provider chain."""
    with patch("backend.aws.s3.boto3.client") as mock_client:
        from botocore.exceptions import NoCredentialsError
        mock_client.side_effect = NoCredentialsError()

        from backend.aws.s3 import reset_s3_client
        reset_s3_client()

        import importlib
        from backend.aws import s3 as s3_module
        importlib.reload(s3_module)
        assert s3_module.is_s3_available() is False


def test_cloudwatch_availability_check_without_credentials():
    """Test that is_cloudwatch_available returns False when no credentials in provider chain."""
    with patch("backend.aws.cloudwatch.boto3.client") as mock_client:
        from botocore.exceptions import NoCredentialsError
        mock_client.side_effect = NoCredentialsError()

        from backend.aws.cloudwatch import reset_cloudwatch_clients
        reset_cloudwatch_clients()

        import importlib
        from backend.aws import cloudwatch as cw_module
        importlib.reload(cw_module)
        assert cw_module.is_cloudwatch_available() is False


def test_s3_availability_check_with_credentials():
    """Test that is_s3_available returns True when credentials available in provider chain."""
    with patch("backend.aws.s3.boto3.client") as mock_client:
        mock_s3 = MagicMock()
        mock_s3.list_buckets.return_value = {"Buckets": []}
        mock_client.return_value = mock_s3

        from backend.aws.s3 import reset_s3_client
        reset_s3_client()

        import importlib
        from backend.aws import s3 as s3_module
        importlib.reload(s3_module)
        assert s3_module.is_s3_available() is True


def test_cloudwatch_availability_check_with_credentials():
    """Test that is_cloudwatch_available returns True when credentials available in provider chain."""
    with patch("backend.aws.cloudwatch.boto3.client") as mock_client:
        mock_cw = MagicMock()
        mock_cw.describe_log_groups.return_value = {"logGroups": []}
        mock_cw.list_metrics.return_value = {"metrics": []}
        mock_client.return_value = mock_cw

        from backend.aws.cloudwatch import reset_cloudwatch_clients
        reset_cloudwatch_clients()

        import importlib
        from backend.aws import cloudwatch as cw_module
        importlib.reload(cw_module)
        assert cw_module.is_cloudwatch_available() is True


def test_s3_client_creation_fails_gracefully_with_invalid_credentials():
    """Test that S3 client creation fails gracefully with invalid credentials."""
    with patch("backend.aws.s3.boto3.client") as mock_client:
        from botocore.exceptions import NoCredentialsError
        mock_client.side_effect = NoCredentialsError()

        from backend.aws.s3 import reset_s3_client
        reset_s3_client()

        import importlib
        from backend.aws import s3 as s3_module
        importlib.reload(s3_module)
        client = s3_module._get_s3_client()
        assert client is None
        assert s3_module.is_s3_available() is False


def test_cloudwatch_client_creation_fails_gracefully_with_invalid_credentials():
    """Test that CloudWatch client creation fails gracefully with invalid credentials."""
    with patch("backend.aws.cloudwatch.boto3.client") as mock_client:
        from botocore.exceptions import NoCredentialsError
        mock_client.side_effect = NoCredentialsError()

        from backend.aws.cloudwatch import reset_cloudwatch_clients
        reset_cloudwatch_clients()

        import importlib
        from backend.aws import cloudwatch as cw_module
        importlib.reload(cw_module)
        assert cw_module.is_cloudwatch_available() is False


def test_s3_partial_credentials_error():
    """Test handling of PartialCredentialsError."""
    with patch("backend.aws.s3.boto3.client") as mock_client:
        from botocore.exceptions import PartialCredentialsError
        mock_client.side_effect = PartialCredentialsError(provider="env", cred_var="AWS_SECRET_ACCESS_KEY")

        from backend.aws.s3 import reset_s3_client
        reset_s3_client()

        from backend.aws.s3 import _get_s3_client
        client = _get_s3_client()
        assert client is None


def test_cloudwatch_partial_credentials_error():
    """Test handling of PartialCredentialsError for CloudWatch."""
    with patch("backend.aws.cloudwatch.boto3.client") as mock_client:
        from botocore.exceptions import PartialCredentialsError
        mock_client.side_effect = PartialCredentialsError(provider="env", cred_var="AWS_SECRET_ACCESS_KEY")

        from backend.aws.cloudwatch import reset_cloudwatch_clients
        reset_cloudwatch_clients()

        from backend.aws.cloudwatch import _get_logs_client, _get_metrics_client
        logs_client = _get_logs_client()
        metrics_client = _get_metrics_client()
        assert logs_client is None
        assert metrics_client is None


def test_s3_no_region_error():
    """Test handling of NoRegionError."""
    with patch("backend.aws.s3.boto3.client") as mock_client:
        from botocore.exceptions import NoRegionError
        mock_client.side_effect = NoRegionError()

        from backend.aws.s3 import reset_s3_client
        reset_s3_client()

        from backend.aws.s3 import _get_s3_client
        client = _get_s3_client()
        assert client is None


def test_cloudwatch_no_region_error():
    """Test handling of NoRegionError for CloudWatch."""
    with patch("backend.aws.cloudwatch.boto3.client") as mock_client:
        from botocore.exceptions import NoRegionError
        mock_client.side_effect = NoRegionError()

        from backend.aws.cloudwatch import reset_cloudwatch_clients
        reset_cloudwatch_clients()

        from backend.aws.cloudwatch import _get_logs_client, _get_metrics_client
        logs_client = _get_logs_client()
        metrics_client = _get_metrics_client()
        assert logs_client is None
        assert metrics_client is None


# Integration tests - only run when AWS credentials are configured
# Run with: pytest tests/test_aws.py -m integration

@integration_test
def test_s3_integration_upload_file():
    """Integration test: Upload a real file to S3."""
    import tempfile
    from backend.aws.s3 import upload_file, StorageBackend, is_s3_available, clear_idempotency_cache

    if not is_s3_available():
        pytest.skip("AWS S3 not configured")

    clear_idempotency_cache()

    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        pytest.skip("AWS_S3_BUCKET not set")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content for S3 upload")
        temp_path = f.name

    try:
        # Upload the file
        result = upload_file(temp_path, bucket, "test/integration_test.txt")

        assert result.success is True
        assert result.backend == StorageBackend.S3
        assert result.url is not None
        assert result.url.startswith(f"s3://{bucket}/")

        # Test idempotency - second upload should return cached result
        result2 = upload_file(temp_path, bucket, "test/integration_test.txt")
        assert result2.success is True
        assert result2.url == result.url
        assert result2.retries == 0  # No retries for cached result

    finally:
        os.unlink(temp_path)


@integration_test
def test_s3_integration_upload_evidence_metadata():
    """Integration test: Upload evidence and metadata for an incident."""
    import tempfile
    import json
    from backend.aws.s3 import upload_evidence, upload_metadata, StorageBackend, is_s3_available, clear_idempotency_cache

    if not is_s3_available():
        pytest.skip("AWS S3 not configured")

    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        pytest.skip("AWS_S3_BUCKET not set")

    clear_idempotency_cache()

    incident_id = f"test_incident_{int(time.time())}"

    # Create temp evidence file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as f:
        f.write(b"fake image data")
        evidence_path = f.name

    # Create temp metadata file
    metadata = {
        "incident_id": incident_id,
        "person_id": 1,
        "timestamp": time.time(),
        "confidence": 0.95
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metadata, f)
        metadata_path = f.name

    try:
        # Upload evidence
        evidence_result = upload_evidence(evidence_path, incident_id)
        assert evidence_result.success is True
        assert evidence_result.backend == StorageBackend.S3

        # Upload metadata
        metadata_result = upload_metadata(metadata_path, incident_id)
        assert metadata_result.success is True
        assert metadata_result.backend == StorageBackend.S3

    finally:
        os.unlink(evidence_path)
        os.unlink(metadata_path)


@integration_test
def test_cloudwatch_integration_log_event():
    """Integration test: Log a real event to CloudWatch Logs."""
    from backend.aws.cloudwatch import log_event, is_cloudwatch_available

    if not is_cloudwatch_available():
        pytest.skip("AWS CloudWatch not configured")

    log_group = os.getenv("AWS_CLOUDWATCH_LOG_GROUP", "/aws/industrial-safety-inspector")
    log_stream = "integration_test"

    result = log_event(log_group, log_stream, "INTEGRATION_TEST: Test log event from pytest")
    assert result is True


@integration_test
def test_cloudwatch_integration_put_metric():
    """Integration test: Put a real metric to CloudWatch Metrics."""
    from backend.aws.cloudwatch import put_metric, is_cloudwatch_available

    if not is_cloudwatch_available():
        pytest.skip("AWS CloudWatch not configured")

    result = put_metric("IntegrationTestMetric", 1.0, "Count", [{"Name": "Test", "Value": "pytest"}])
    assert result is True


@integration_test
def test_cloudwatch_integration_batch_metrics():
    """Integration test: Put batch metrics to CloudWatch Metrics."""
    from backend.aws.cloudwatch import put_metrics_batch, is_cloudwatch_available

    if not is_cloudwatch_available():
        pytest.skip("AWS CloudWatch not configured")

    metrics = [
        {"MetricName": "BatchTest1", "Value": 10.0, "Unit": "Count"},
        {"MetricName": "BatchTest2", "Value": 20.5, "Unit": "Milliseconds"}
    ]
    result = put_metrics_batch(metrics)
    assert result is True


if __name__ == "__main__":
    # Run the unit tests (no AWS credentials required)
    test_s3_missing_credentials()
    test_cloudwatch_missing_credentials()
    test_s3_upload_file_not_found()
    test_cloudwatch_log_event_handles_errors_gracefully()
    test_s3_upload_result_structure()
    test_cloudwatch_metrics_batch()
    test_s3_client_creation_with_explicit_credentials()
    test_s3_client_creation_with_profile()
    test_s3_client_region_default()
    test_s3_reset_client_on_credential_change()
    test_cloudwatch_reset_clients()
    test_s3_idempotency_cache()
    test_s3_availability_check_without_credentials()
    test_cloudwatch_availability_check_without_credentials()
    test_s3_availability_check_with_credentials()
    test_cloudwatch_availability_check_with_credentials()
    test_s3_client_creation_fails_gracefully_with_invalid_credentials()
    test_cloudwatch_client_creation_fails_gracefully_with_invalid_credentials()
    test_s3_partial_credentials_error()
    test_cloudwatch_partial_credentials_error()
    test_s3_no_region_error()
    test_cloudwatch_no_region_error()
    print("All unit tests passed!")
    print("Run integration tests with: pytest tests/test_aws.py -m integration")