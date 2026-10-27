"""
Enhanced CloudWatch module for SafeSight AI.
Provides structured logging and metrics with retry logic.
"""
import json
import logging
import os
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError, NoRegionError
from botocore.config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LogLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# CloudWatch client configuration with retries
LOGS_CONFIG = Config(
    retries={
        'max_attempts': 3,
        'mode': 'adaptive'
    },
    connect_timeout=5,
    read_timeout=10,
    max_pool_connections=10
)

METRICS_CONFIG = Config(
    retries={
        'max_attempts': 3,
        'mode': 'adaptive'
    },
    connect_timeout=5,
    read_timeout=10,
    max_pool_connections=10
)

# Global clients (lazy initialization)
_logs_client = None
_metrics_client = None
_logs_client_error = None
_metrics_client_error = None
_logs_client_credentials_hash = None
_metrics_client_credentials_hash = None

# Default log group/stream
DEFAULT_LOG_GROUP = "/aws/industrial-safety-inspector"
DEFAULT_NAMESPACE = "SafeSightAI"


def _get_credentials_hash() -> str:
    """Generate a hash of the current credential configuration to detect changes."""
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


def reset_cloudwatch_clients():
    """Reset the CloudWatch clients cache. Call when configuration may have changed."""
    global _logs_client, _metrics_client, _logs_client_error, _metrics_client_error
    global _logs_client_credentials_hash, _metrics_client_credentials_hash
    _logs_client = None
    _metrics_client = None
    _logs_client_error = None
    _metrics_client_error = None
    _logs_client_credentials_hash = None
    _metrics_client_credentials_hash = None


def _get_logs_client() -> Optional[boto3.client]:
    """Get or create CloudWatch Logs client using boto3's default credential chain."""
    global _logs_client, _logs_client_error, _logs_client_credentials_hash

    # Check if credentials configuration has changed
    current_hash = _get_credentials_hash()
    if _logs_client_credentials_hash is not None and _logs_client_credentials_hash != current_hash:
        logger.info("AWS credential configuration changed, resetting CloudWatch Logs client")
        _logs_client = None
        _logs_client_error = None
        _logs_client_credentials_hash = None

    if _logs_client is not None:
        return _logs_client

    if _logs_client_error is not None:
        return None

    try:
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

        _logs_client = boto3.client(
            'logs',
            region_name=region,
            config=LOGS_CONFIG
        )

        # Verify credentials work with a lightweight call
        try:
            _logs_client.describe_log_groups(limit=1)
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.warning(f"AWS credentials not available via provider chain for CloudWatch Logs: {e}")
            _logs_client_error = str(e)
            _logs_client = None
            return None
        except NoRegionError:
            logger.warning("AWS region not configured for CloudWatch Logs")
            _logs_client_error = "Missing region"
            _logs_client = None
            return None
        except ClientError as e:
            # Other client errors mean credentials work but permissions may be limited
            logger.info(f"CloudWatch Logs client created (note: {e.response['Error']['Code']})")

        _logs_client_credentials_hash = current_hash
        logger.info("CloudWatch Logs client initialized successfully using default credential chain")
        return _logs_client
    except Exception as e:
        logger.warning(f"Failed to create CloudWatch Logs client: {e}")
        _logs_client_error = str(e)
        return None


def _get_metrics_client() -> Optional[boto3.client]:
    """Get or create CloudWatch Metrics client using boto3's default credential chain."""
    global _metrics_client, _metrics_client_error, _metrics_client_credentials_hash

    # Check if credentials configuration has changed
    current_hash = _get_credentials_hash()
    if _metrics_client_credentials_hash is not None and _metrics_client_credentials_hash != current_hash:
        logger.info("AWS credential configuration changed, resetting CloudWatch Metrics client")
        _metrics_client = None
        _metrics_client_error = None
        _metrics_client_credentials_hash = None

    if _metrics_client is not None:
        return _metrics_client

    if _metrics_client_error is not None:
        return None

    try:
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

        _metrics_client = boto3.client(
            'cloudwatch',
            region_name=region,
            config=METRICS_CONFIG
        )

        # Verify credentials work with a lightweight call
        try:
            _metrics_client.list_metrics()
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.warning(f"AWS credentials not available via provider chain for CloudWatch Metrics: {e}")
            _metrics_client_error = str(e)
            _metrics_client = None
            return None
        except NoRegionError:
            logger.warning("AWS region not configured for CloudWatch Metrics")
            _metrics_client_error = "Missing region"
            _metrics_client = None
            return None
        except ClientError as e:
            logger.info(f"CloudWatch Metrics client created (note: {e.response['Error']['Code']})")

        _metrics_client_credentials_hash = current_hash
        logger.info("CloudWatch Metrics client initialized successfully using default credential chain")
        return _metrics_client
    except Exception as e:
        logger.warning(f"Failed to create CloudWatch Metrics client: {e}")
        _metrics_client_error = str(e)
        return None


def _ensure_log_group(log_group: str) -> bool:
    """Ensure the log group exists; create it if it doesn't."""
    client = _get_logs_client()
    if client is None:
        return False
    try:
        # logGroupNamePrefix should include the full path (with leading slash) for correct matching
        client.describe_log_groups(logGroupNamePrefix=log_group, limit=1)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            try:
                client.create_log_group(logGroupName=log_group)
                logger.info(f"Created CloudWatch log group: {log_group}")
                return True
            except ClientError as create_err:
                logger.error(f"Failed to create log group {log_group}: {create_err}")
                return False
        else:
            logger.error(f"Error describing log groups: {e}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error ensuring log group: {e}")
        return False


def _ensure_log_stream(log_group: str, log_stream: str) -> Optional[str]:
    """Ensure the log stream exists; create it if it doesn't. Returns sequence token."""
    client = _get_logs_client()
    if client is None:
        return None

    try:
        response = client.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=log_stream,
            limit=1
        )
        streams = response.get('logStreams', [])
        if streams:
            log_stream_name = streams[0]['logStreamName']
            sequence_token = streams[0].get('uploadSequenceToken')
            return sequence_token
        else:
            client.create_log_stream(logGroupName=log_group, logStreamName=log_stream)
            return None
    except ClientError as e:
        logger.error(f"Error describing/creating log streams: {e}")
        return None


def log_event(
    log_group: str,
    log_stream: str,
    message: str,
    level: LogLevel = LogLevel.INFO
) -> bool:
    """
    Log a single event to CloudWatch Logs with retry logic.

    Args:
        log_group: The log group name (e.g., /aws/industrial-safety-inspector).
        log_stream: The log stream name (e.g., detection_events).
        message: The log message to record.
        level: Log level (info, warning, error).

    Returns:
        True if the log was sent successfully, False otherwise.
    """
    client = _get_logs_client()
    if client is None:
        logger.log(
            logging.INFO if level == LogLevel.INFO else logging.WARNING if level == LogLevel.WARNING else logging.ERROR,
            f"[{log_stream}] {message}"
        )
        return False

    try:
        if not _ensure_log_group(log_group):
            return False

        sequence_token = _ensure_log_stream(log_group, log_stream)
        # If sequence_token is None, stream may be newly created (no uploadSequenceToken yet).
        # Try once more; if still None, proceed anyway — put_log_events without token works for first event.
        if sequence_token is None:
            sequence_token = _ensure_log_stream(log_group, log_stream)

        timestamp = int(time.time() * 1000)
        log_event_data = {
            'timestamp': timestamp,
            'message': message
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if sequence_token:
                    response = client.put_log_events(
                        logGroupName=log_group,
                        logStreamName=log_stream,
                        logEvents=[log_event_data],
                        sequenceToken=sequence_token
                    )
                else:
                    response = client.put_log_events(
                        logGroupName=log_group,
                        logStreamName=log_stream,
                        logEvents=[log_event_data]
                    )
                sequence_token = response.get('nextSequenceToken')
                logger.debug(f"Logged to CloudWatch: {message}")
                return True
            except ClientError as e:
                if e.response['Error']['Code'] == 'InvalidSequenceTokenException' and attempt < max_retries - 1:
                    # Get new sequence token and retry
                    sequence_token = _ensure_log_stream(log_group, log_stream)
                    time.sleep(0.1)
                    continue
                raise

        return False

    except ClientError as e:
        logger.error(f"AWS client error logging to CloudWatch: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error logging to CloudWatch: {e}")
        return False


def put_metric(
    metric_name: str,
    value: float,
    unit: str = "Count",
    dimensions: Optional[List[Dict[str, str]]] = None,
    namespace: str = DEFAULT_NAMESPACE
) -> bool:
    """
    Put a single metric data point to CloudWatch Metrics.

    Args:
        metric_name: Name of the metric (e.g., "ProcessingLatency", "IncidentsCreated").
        value: Metric value.
        unit: CloudWatch unit (Count, Seconds, Milliseconds, Bytes, Percent, etc.).
        dimensions: List of dimension dicts, e.g., [{"Name": "Zone", "Value": "restricted_zone_1"}].
        namespace: CloudWatch namespace.

    Returns:
        True if successful, False otherwise.
    """
    client = _get_metrics_client()
    if client is None:
        logger.debug(f"Metric (local only): {metric_name}={value} {unit}")
        return False

    try:
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': time.time()
        }
        if dimensions:
            metric_data['Dimensions'] = dimensions

        client.put_metric_data(
            Namespace=namespace,
            MetricData=[metric_data]
        )
        logger.debug(f"Put metric: {metric_name}={value} {unit}")
        return True
    except ClientError as e:
        logger.error(f"AWS client error putting metric: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error putting metric: {e}")
        return False


def put_metrics_batch(
    metrics: List[Dict[str, Any]],
    namespace: str = DEFAULT_NAMESPACE
) -> bool:
    """
    Put multiple metrics in a single API call.

    Args:
        metrics: List of metric dicts with keys: MetricName, Value, Unit, Dimensions (optional).
        namespace: CloudWatch namespace.

    Returns:
        True if successful, False otherwise.
    """
    client = _get_metrics_client()
    if client is None:
        for m in metrics:
            logger.debug(f"Metric (local only): {m['MetricName']}={m['Value']} {m.get('Unit', 'Count')}")
        return False

    try:
        metric_data = []
        for m in metrics:
            data = {
                'MetricName': m['MetricName'],
                'Value': m['Value'],
                'Unit': m.get('Unit', 'Count'),
                'Timestamp': time.time()
            }
            if 'Dimensions' in m:
                data['Dimensions'] = m['Dimensions']
            metric_data.append(data)

        client.put_metric_data(
            Namespace=namespace,
            MetricData=metric_data
        )
        logger.debug(f"Put {len(metrics)} metrics batch")
        return True
    except ClientError as e:
        logger.error(f"AWS client error putting metrics batch: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error putting metrics batch: {e}")
        return False


# Convenience functions for structured logging
def log_detection_event(event: dict) -> bool:
    """Log a detection event from the analyzer."""
    log_group = os.getenv("AWS_CLOUDWATCH_LOG_GROUP", DEFAULT_LOG_GROUP)
    log_stream = "detection_events"
    message = f"DETECTION_EVENT: {json.dumps(event)}"
    return log_event(log_group, log_stream, message)


def log_agent_decision(decision: dict) -> bool:
    """Log an agent decision."""
    log_group = os.getenv("AWS_CLOUDWATCH_LOG_GROUP", DEFAULT_LOG_GROUP)
    log_stream = "agent_decisions"
    message = f"AGENT_DECISION: {json.dumps(decision)}"
    return log_event(log_group, log_stream, message)


def log_confirmed_incident(incident_id: str, metadata: dict) -> bool:
    """Log a confirmed incident."""
    log_group = os.getenv("AWS_CLOUDWATCH_LOG_GROUP", DEFAULT_LOG_GROUP)
    log_stream = "confirmed_incidents"
    message = f"CONFIRMED_INCIDENT: incident_id={incident_id}, metadata={json.dumps(metadata)}"
    return log_event(log_group, log_stream, message)


def log_processing_metrics(
    fps: float,
    latency_ms: float,
    frame_count: int,
    detections: int = 0,
    events: int = 0
) -> bool:
    """Log processing performance metrics to both Logs and Metrics."""
    log_group = os.getenv("AWS_CLOUDWATCH_LOG_GROUP", DEFAULT_LOG_GROUP)
    log_stream = "processing_metrics"
    message = f"METRICS: fps={fps:.2f}, latency={latency_ms:.2f}ms, frames={frame_count}, detections={detections}, events={events}"
    log_sent = log_event(log_group, log_stream, message)

    # Also send to CloudWatch Metrics
    dimensions = []  # Could add zone, camera_id, etc.
    metric1 = put_metric("FramesProcessed", frame_count, "Count", dimensions)
    metric2 = put_metric("ProcessingFPS", fps, "Count/Second", dimensions)
    metric3 = put_metric("ProcessingLatencyMs", latency_ms, "Milliseconds", dimensions)
    metric4 = put_metric("DetectionsCount", detections, "Count", dimensions)
    metric5 = put_metric("EventsGenerated", events, "Count", dimensions)

    # Return True only if at least one CloudWatch call succeeded
    return log_sent or metric1 or metric2 or metric3 or metric4 or metric5


def log_incident_metrics(
    incident_id: str,
    severity: str,
    confidence: float,
    duration: float,
    zone: str
) -> bool:
    """Log incident-specific metrics."""
    dimensions = [
        {"Name": "Severity", "Value": severity},
        {"Name": "Zone", "Value": zone}
    ]
    put_metric("IncidentsCreated", 1, "Count", dimensions)
    put_metric("IncidentConfidence", confidence, "Percent", dimensions)
    put_metric("IncidentDurationSeconds", duration, "Seconds", dimensions)
    return True


def log_error_metric(error_type: str, zone: str = "unknown") -> bool:
    """Log error occurrence metric."""
    dimensions = [
        {"Name": "ErrorType", "Value": error_type},
        {"Name": "Zone", "Value": zone}
    ]
    put_metric("Errors", 1, "Count", dimensions)
    return True


def is_cloudwatch_available() -> bool:
    """Check if CloudWatch is configured and available."""
    return _get_logs_client() is not None or _get_metrics_client() is not None