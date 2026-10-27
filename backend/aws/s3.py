"""
Enhanced S3 module for SafeSight AI.
Uploads incident evidence and metadata to S3 with retry logic and reporting.
Supports idempotency keys to prevent duplicate uploads.
"""
import os
import logging
import time
import hashlib
import json
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError, NoRegionError
from botocore.config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Idempotency cache (in-memory, per-process)
_idempotency_cache: dict = {}


class StorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"


@dataclass
class UploadResult:
    """Result of an S3 upload operation."""
    success: bool
    backend: StorageBackend
    url: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0
    duration_ms: float = 0.0


# S3 client configuration with retries
S3_CONFIG = Config(
    retries={
        'max_attempts': 3,
        'mode': 'adaptive'
    },
    connect_timeout=5,
    read_timeout=30,
    max_pool_connections=10
)

# Global S3 client (lazy initialization)
_s3_client = None
_client_init_error = None
_client_credentials_hash = None


def _generate_idempotency_key(local_file_path: str, bucket: str, s3_file_path: str) -> str:
    """Generate a deterministic idempotency key based on file content and destination."""
    try:
        with open(local_file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:16]
    except Exception:
        file_hash = "unknown"
    return f"{file_hash}:{bucket}:{s3_file_path}"


def _check_idempotency(key: str) -> Optional[str]:
    """Check if an idempotency key has been used. Returns the cached S3 URL if found."""
    return _idempotency_cache.get(key)


def _store_idempotency(key: str, s3_url: str):
    """Store an idempotency key with its S3 URL."""
    _idempotency_cache[key] = s3_url


def clear_idempotency_cache():
    """Clear the idempotency cache (useful for testing)."""
    global _idempotency_cache
    _idempotency_cache.clear()


def _get_credentials_hash() -> str:
    """Generate a hash of the current credential configuration to detect changes."""
    # Hash relevant env vars that affect credential resolution
    # Note: We don't log actual credentials, just a hash for change detection
    import hashlib
    parts = [
        os.getenv("AWS_ACCESS_KEY_ID", ""),
        os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        os.getenv("AWS_SESSION_TOKEN", ""),
        os.getenv("AWS_PROFILE", ""),
        os.getenv("AWS_REGION", ""),
        os.getenv("AWS_DEFAULT_REGION", ""),
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def reset_s3_client():
    """Reset the S3 client cache. Call when configuration may have changed."""
    global _s3_client, _client_init_error, _client_credentials_hash
    _s3_client = None
    _client_init_error = None
    _client_credentials_hash = None


def _get_s3_client() -> Optional[boto3.client]:
    """
    Create and return an S3 client with retry config using boto3's default credential chain.
    Returns None if credentials cannot be resolved or region is missing.
    """
    global _s3_client, _client_init_error, _client_credentials_hash

    # Check if credentials configuration has changed
    current_hash = _get_credentials_hash()
    if _client_credentials_hash is not None and _client_credentials_hash != current_hash:
        logger.info("AWS credential configuration changed, resetting S3 client")
        reset_s3_client()

    if _s3_client is not None:
        return _s3_client

    if _client_init_error is not None:
        return None

    try:
        # Determine region - use AWS_REGION, then AWS_DEFAULT_REGION, then default
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

        # Create client using boto3's default credential provider chain
        # This supports: env vars, ~/.aws/credentials, IAM roles, AWS CLI SSO, etc.
        # If explicit credentials are provided in env vars, boto3 will use them
        _s3_client = boto3.client(
            's3',
            region_name=region,
            config=S3_CONFIG
        )

        # Verify credentials work by making a lightweight call
        # This will fail fast if credentials are invalid
        try:
            _s3_client.list_buckets()
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.warning(f"AWS credentials not available via provider chain: {e}")
            _client_init_error = str(e)
            _s3_client = None
            return None
        except NoRegionError:
            logger.warning("AWS region not configured")
            _client_init_error = "Missing region"
            _s3_client = None
            return None
        except ClientError as e:
            # Other client errors (like AccessDenied) mean credentials work but permissions may be limited
            # We still allow the client to be used - operations will fail at runtime if permissions insufficient
            logger.info(f"S3 client created (note: {e.response['Error']['Code']})")

        _client_credentials_hash = current_hash
        logger.info("S3 client initialized successfully using default credential chain")
        return _s3_client
    except Exception as e:
        logger.warning(f"Failed to create S3 client: {e}")
        _client_init_error = str(e)
        return None


def _upload_with_retry(
    client: boto3.client,
    local_file_path: str,
    bucket: str,
    s3_file_path: str,
    max_retries: int = 3
) -> Tuple[bool, int]:
    """
    Upload file with exponential backoff retry.
    Returns (success, retry_count).
    """
    for attempt in range(max_retries):
        try:
            client.upload_file(local_file_path, bucket, s3_file_path)
            return True, attempt
        except ClientError as e:
            error_code = e.response['Error']['Code']
            # Don't retry on auth errors
            if error_code in ['NoSuchBucket', 'InvalidAccessKeyId', 'SignatureDoesNotMatch']:
                raise
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    return False, max_retries


def upload_file(
    local_file_path: str,
    bucket: str,
    s3_file_path: str,
    idempotency_key: Optional[str] = None
) -> UploadResult:
    """
    Upload a file to an S3 bucket with retry logic and detailed result.
    Supports idempotency keys to prevent duplicate uploads.

    Args:
        local_file_path: Path to the local file to upload.
        bucket: Name of the S3 bucket.
        s3_file_path: Destination path (key) in the S3 bucket.
        idempotency_key: Optional key to ensure idempotent uploads.
                         If provided and a previous upload with this key succeeded,
                         returns the cached result without re-uploading.

    Returns:
        UploadResult with success status, URL, backend used, and timing.
    """
    start_time = time.time()

    if not os.path.isfile(local_file_path):
        return UploadResult(
            success=False,
            backend=StorageBackend.LOCAL,
            error=f"Local file not found: {local_file_path}",
            duration_ms=(time.time() - start_time) * 1000
        )

    # Generate idempotency key if not provided
    if idempotency_key is None:
        idempotency_key = _generate_idempotency_key(local_file_path, bucket, s3_file_path)

    # Check idempotency cache
    cached_url = _check_idempotency(idempotency_key)
    if cached_url:
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Idempotent upload hit for {s3_file_path}: {cached_url}")
        return UploadResult(
            success=True,
            backend=StorageBackend.S3,
            url=cached_url,
            retries=0,
            duration_ms=duration_ms
        )

    s3_client = _get_s3_client()
    if s3_client is None:
        return UploadResult(
            success=False,
            backend=StorageBackend.LOCAL,
            error="S3 client not available - using local storage only",
            duration_ms=(time.time() - start_time) * 1000
        )

    try:
        success, retries = _upload_with_retry(s3_client, local_file_path, bucket, s3_file_path)
        duration_ms = (time.time() - start_time) * 1000

        if success:
            s3_url = f"s3://{bucket}/{s3_file_path}"
            # Store in idempotency cache
            _store_idempotency(idempotency_key, s3_url)
            logger.info(f"Uploaded {local_file_path} to {s3_url} (retries: {retries}, {duration_ms:.1f}ms)")
            return UploadResult(
                success=True,
                backend=StorageBackend.S3,
                url=s3_url,
                retries=retries,
                duration_ms=duration_ms
            )
        else:
            return UploadResult(
                success=False,
                backend=StorageBackend.S3,
                error="Upload failed after retries",
                retries=retries,
                duration_ms=duration_ms
            )
    except FileNotFoundError:
        return UploadResult(
            success=False,
            backend=StorageBackend.S3,
            error=f"The file {local_file_path} was not found",
            duration_ms=(time.time() - start_time) * 1000
        )
    except NoCredentialsError:
        return UploadResult(
            success=False,
            backend=StorageBackend.S3,
            error="AWS credentials not available",
            duration_ms=(time.time() - start_time) * 1000
        )
    except PartialCredentialsError:
        return UploadResult(
            success=False,
            backend=StorageBackend.S3,
            error="Incomplete AWS credentials provided",
            duration_ms=(time.time() - start_time) * 1000
        )
    except ClientError as e:
        return UploadResult(
            success=False,
            backend=StorageBackend.S3,
            error=f"AWS client error: {e}",
            duration_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        return UploadResult(
            success=False,
            backend=StorageBackend.S3,
            error=f"Unexpected error uploading to S3: {e}",
            duration_ms=(time.time() - start_time) * 1000
        )


def upload_evidence(local_image_path: str, incident_id: str, idempotency_key: Optional[str] = None) -> UploadResult:
    """
    Upload an evidence image to S3.
    Stored under: incidents/<incident_id>/evidence.jpg
    """
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        return UploadResult(
            success=False,
            backend=StorageBackend.LOCAL,
            error="AWS_S3_BUCKET not set - using local storage only"
        )
    s3_key = f"incidents/{incident_id}/evidence.jpg"
    return upload_file(local_image_path, bucket, s3_key, idempotency_key)


def upload_metadata(local_json_path: str, incident_id: str, idempotency_key: Optional[str] = None) -> UploadResult:
    """
    Upload incident metadata to S3.
    Stored under: incidents/<incident_id>/metadata.json
    """
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        return UploadResult(
            success=False,
            backend=StorageBackend.LOCAL,
            error="AWS_S3_BUCKET not set - using local storage only"
        )
    s3_key = f"incidents/{incident_id}/metadata.json"
    return upload_file(local_json_path, bucket, s3_key, idempotency_key)


def upload_incident_bundle(incident_dir: str, incident_id: str) -> dict:
    """
    Upload complete incident bundle (evidence + metadata) to S3.
    Uses idempotency keys based on incident_id to prevent duplicate uploads.
    Returns dict with results for each file.
    """
    results = {}

    # Generate idempotency keys for this incident
    evidence_key = f"incident:{incident_id}:evidence"
    metadata_key = f"incident:{incident_id}:metadata"

    # Upload evidence
    evidence_path = os.path.join(incident_dir, "evidence.jpg")
    if os.path.isfile(evidence_path):
        results["evidence"] = upload_evidence(evidence_path, incident_id, evidence_key)
    else:
        results["evidence"] = UploadResult(
            success=False,
            backend=StorageBackend.LOCAL,
            error="Evidence file not found"
        )

    # Upload metadata
    metadata_path = os.path.join(incident_dir, "metadata.json")
    if os.path.isfile(metadata_path):
        results["metadata"] = upload_metadata(metadata_path, incident_id, metadata_key)
    else:
        results["metadata"] = UploadResult(
            success=False,
            backend=StorageBackend.LOCAL,
            error="Metadata file not found"
        )

    return results


def is_s3_available() -> bool:
    """Check if S3 is configured and available."""
    return _get_s3_client() is not None


def get_storage_backend() -> StorageBackend:
    """Get the active storage backend."""
    return StorageBackend.S3 if is_s3_available() else StorageBackend.LOCAL