"""
Enhanced Actions module for SafeSight AI agentic vision system.
Implements incident intelligence: deduplication, lifecycle, severity, persistent storage.
"""
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field, asdict

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# AWS integration (optional - graceful degradation)
try:
    from backend.aws.s3 import upload_incident_bundle, get_storage_backend, UploadResult
    from backend.aws.cloudwatch import log_confirmed_incident, log_incident_metrics, is_cloudwatch_available
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    upload_incident_bundle = None
    get_storage_backend = None
    UploadResult = None
    log_confirmed_incident = None
    log_incident_metrics = None
    is_cloudwatch_available = None


# Project root: <project>/backend/agent/actions.py -> <project>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INCIDENTS_DIR = PROJECT_ROOT / "incidents"


class IncidentStatus(str, Enum):
    """Incident lifecycle states."""
    CANDIDATE = "candidate"          # Potential violation detected
    CONFIRMED = "confirmed"          # Violation confirmed by agent
    UNDER_REVIEW = "under_review"    # Human review requested
    APPROVED = "approved"            # Human approved
    REJECTED = "rejected"            # Human rejected (false alarm)
    RESOLVED = "resolved"            # Person exited, incident closed


class SeverityLevel(str, Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IncidentTimelineEvent:
    """Single event in incident timeline."""
    timestamp: float
    event_type: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


def _json_default(value):
    """Convert common NumPy values to JSON-compatible data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return str(value)


def _calculate_severity(confidence: float, duration: float, risk_level: str) -> SeverityLevel:
    """Calculate severity based on confidence, duration, and risk level."""
    if risk_level == "critical" or (confidence >= 0.8 and duration >= 10.0):
        return SeverityLevel.CRITICAL
    elif risk_level == "high" or (confidence >= 0.7 and duration >= 5.0):
        return SeverityLevel.HIGH
    elif risk_level == "medium" or (confidence >= 0.5 and duration >= 3.0):
        return SeverityLevel.MEDIUM
    return SeverityLevel.LOW


def _find_existing_incident(person_id: int, zone: str) -> Optional[str]:
    """Find existing open incident for same person/zone to prevent duplicates."""
    if not INCIDENTS_DIR.is_dir():
        return None

    for incident_dir in INCIDENTS_DIR.iterdir():
        if not incident_dir.is_dir():
            continue
        metadata_path = incident_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            # Check if same person, same zone, and not resolved/rejected
            if (metadata.get("person_id") == person_id and
                metadata.get("zone") == zone and
                metadata.get("status") in [IncidentStatus.CANDIDATE.value,
                                           IncidentStatus.CONFIRMED.value,
                                           IncidentStatus.UNDER_REVIEW.value]):
                return incident_dir.name
        except (OSError, json.JSONDecodeError):
            continue
    return None


def continue_tracking(person_id: int, parameters: Dict[str, Any]) -> bool:
    duration = parameters.get("duration_seconds", 3)
    print(f"ACTION: Continue tracking person {person_id} for {duration} seconds")
    return True


def create_incident(
    person_id: int,
    parameters: Dict[str, Any],
    frame: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create an incident with deduplication, severity assessment, and full timeline.
    Returns the incident ID.
    """
    incident_id = str(uuid.uuid4())
    timestamp = time.time()

    # Extract metadata
    event_type = metadata.get("event_type", "unknown")
    confidence = metadata.get("confidence", 0.0)
    zone = metadata.get("zone", "unknown")
    duration = metadata.get("duration", 0.0)
    risk_level = metadata.get("risk_level", "low")
    reason_codes = metadata.get("reason_codes", [])
    evidence_refs = metadata.get("evidence_refs", [])
    decision_id = metadata.get("decision_id")
    human_approval_required = metadata.get("human_approval_required", False)

    # Calculate severity
    severity = _calculate_severity(confidence, duration, risk_level)

    # DEDUPLICATION: Check for existing open incident for same person/zone
    existing_id = _find_existing_incident(person_id, zone)
    if existing_id:
        print(f"ACTION: Duplicate prevented - updating existing incident {existing_id} for person {person_id} in zone {zone}")
        # Update existing incident with new evidence
        return _update_existing_incident(existing_id, frame, metadata, severity)

    # Determine initial status
    initial_status = IncidentStatus.UNDER_REVIEW if human_approval_required else IncidentStatus.CONFIRMED

    incident_dir = INCIDENTS_DIR / incident_id
    incident_dir.mkdir(parents=True, exist_ok=False)

    evidence_path = None
    if frame is not None:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Evidence frame must be a non-empty NumPy array.")

        evidence_file = incident_dir / "evidence.jpg"
        if not cv2.imwrite(str(evidence_file), frame):
            raise RuntimeError(f"Failed to save evidence image: {evidence_file}")

        evidence_path = str(
            Path("incidents") / incident_id / "evidence.jpg"
        ).replace("\\", "/")

    # Build comprehensive incident record
    record = {
        "incident_id": incident_id,
        "person_id": person_id,
        "timestamp": timestamp,
        "status": initial_status.value,
        "severity": severity.value,
        "zone": zone,
        "event_type": event_type,
        "confidence": confidence,
        "duration": duration,
        "risk_level": risk_level,
        "reason_codes": reason_codes,
        "evidence_refs": evidence_refs,
        "decision_id": decision_id,
        "human_approval_required": human_approval_required,
        "parameters": parameters or {},
        "evidence_path": evidence_path,
        "timeline": [
            {
                "timestamp": timestamp,
                "event_type": "incident_created",
                "description": f"Incident created from {event_type}",
                "details": {"confidence": confidence, "severity": severity.value}
            }
        ],
        "created_at": timestamp,
        "updated_at": timestamp,
    }

    metadata_file = incident_dir / "metadata.json"
    with metadata_file.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False, default=_json_default)

    # Add storage backend info to record
    storage_backend = get_storage_backend().value if AWS_AVAILABLE and get_storage_backend else "local"
    record["storage_backend"] = storage_backend
    record["aws_uploaded"] = False

    print(f"ACTION: Created incident {incident_id} for person {person_id} (severity: {severity.value})")
    print(f"  - Metadata saved: {metadata_file}")
    if evidence_path:
        print(f"  - Evidence saved: {incident_dir / 'evidence.jpg'}")

    # AWS Integration: Upload to S3 and log to CloudWatch (async in production, sync here for simplicity)
    if AWS_AVAILABLE:
        try:
            # Upload incident bundle to S3
            s3_results = upload_incident_bundle(str(incident_dir), incident_id)
            record["s3_upload"] = {
                "evidence": {
                    "success": s3_results.get("evidence", {}).success if s3_results.get("evidence") else False,
                    "backend": s3_results.get("evidence", {}).backend.value if s3_results.get("evidence") else None,
                    "url": s3_results.get("evidence", {}).url if s3_results.get("evidence") else None,
                    "error": s3_results.get("evidence", {}).error if s3_results.get("evidence") else None
                },
                "metadata": {
                    "success": s3_results.get("metadata", {}).success if s3_results.get("metadata") else False,
                    "backend": s3_results.get("metadata", {}).backend.value if s3_results.get("metadata") else None,
                    "url": s3_results.get("metadata", {}).url if s3_results.get("metadata") else None,
                    "error": s3_results.get("metadata", {}).error if s3_results.get("metadata") else None
                }
            }
            # Update record with S3 URLs if successful
            if s3_results.get("evidence", {}).success and s3_results.get("evidence", {}).url:
                record["evidence_path_s3"] = s3_results["evidence"].url
                record["aws_uploaded"] = True
            if s3_results.get("metadata", {}).success and s3_results.get("metadata", {}).url:
                record["metadata_path_s3"] = s3_results["metadata"].url

            # Log to CloudWatch Logs
            log_confirmed_incident(incident_id, record)

            # Log metrics to CloudWatch Metrics
            log_incident_metrics(
                incident_id=incident_id,
                severity=severity.value,
                confidence=confidence,
                duration=duration,
                zone=zone
            )

        except Exception as e:
            logger.warning(f"AWS integration failed for incident {incident_id}: {e}")
            record["aws_integration_error"] = str(e)

    # Re-save metadata with AWS info
    with metadata_file.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False, default=_json_default)

    return incident_id


def _update_existing_incident(
    incident_id: str,
    frame: Optional[np.ndarray],
    metadata: Dict[str, Any],
    severity: SeverityLevel
) -> str:
    """Update existing incident with new evidence/observation."""
    incident_dir = INCIDENTS_DIR / incident_id
    metadata_path = incident_dir / "metadata.json"

    if not metadata_path.is_file():
        return create_incident(metadata.get("person_id", -1), {}, frame, metadata)

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            record = json.load(f)

        # Update with new evidence if higher confidence
        new_confidence = metadata.get("confidence", 0.0)
        if new_confidence > record.get("confidence", 0.0):
            record["confidence"] = new_confidence
            record["severity"] = severity.value
            record["risk_level"] = metadata.get("risk_level", record.get("risk_level"))
            record["reason_codes"] = list(set(record.get("reason_codes", []) + metadata.get("reason_codes", [])))
            record["evidence_refs"] = list(set(record.get("evidence_refs", []) + metadata.get("evidence_refs", [])))

        # Update duration if longer
        new_duration = metadata.get("duration", 0.0)
        if new_duration > record.get("duration", 0.0):
            record["duration"] = new_duration

        # Add timeline event
        record["timeline"].append({
            "timestamp": time.time(),
            "event_type": "evidence_updated",
            "description": f"Additional evidence from {metadata.get('event_type', 'unknown')}",
            "details": {"confidence": new_confidence, "duration": new_duration}
        })

        record["updated_at"] = time.time()

        # Save new evidence frame if provided and better quality
        if frame is not None:
            evidence_file = incident_dir / "evidence.jpg"
            cv2.imwrite(str(evidence_file), frame)
            record["evidence_path"] = str(
                Path("incidents") / incident_id / "evidence.jpg"
            ).replace("\\", "/")

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=_json_default)

        print(f"ACTION: Updated incident {incident_id} with new evidence")
        return incident_id

    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: Failed to update incident {incident_id}: {e}")
        return create_incident(metadata.get("person_id", -1), {}, frame, metadata)


def resolve_incident(person_id: int, parameters: Dict[str, Any]) -> bool:
    """Resolve incident when person exits zone."""
    # Find and update incident to RESOLVED
    incident_id = _find_incident_by_person(person_id)
    if incident_id:
        _update_incident_status(incident_id, IncidentStatus.RESOLVED, "Person exited restricted zone")
    print(f"ACTION: Resolve incident for person {person_id}")
    return True


def request_human_approval(incident_id: str, parameters: Dict[str, Any]) -> bool:
    """Request human approval for an incident."""
    # This is called with incident_id as person_id string in some cases
    # Find the actual incident
    actual_id = _find_incident_by_candidate_id(parameters.get("candidate_id", ""))
    if actual_id:
        _update_incident_status(actual_id, IncidentStatus.UNDER_REVIEW, "Human approval requested")
    print(f"ACTION: Request human approval for incident {incident_id}")
    return True


def no_action(person_id: int, parameters: Dict[str, Any]) -> bool:
    print(f"ACTION: No action for person {person_id}")
    return True


def _find_incident_by_person(person_id: int) -> Optional[str]:
    """Find incident by person_id with open status."""
    if not INCIDENTS_DIR.is_dir():
        return None
    for incident_dir in INCIDENTS_DIR.iterdir():
        if not incident_dir.is_dir():
            continue
        metadata_path = incident_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            if (metadata.get("person_id") == person_id and
                metadata.get("status") in [IncidentStatus.CANDIDATE.value,
                                           IncidentStatus.CONFIRMED.value,
                                           IncidentStatus.UNDER_REVIEW.value]):
                return incident_dir.name
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _find_incident_by_candidate_id(candidate_id: str) -> Optional[str]:
    """Find incident by candidate ID."""
    if not candidate_id:
        return None
    if not INCIDENTS_DIR.is_dir():
        return None
    for incident_dir in INCIDENTS_DIR.iterdir():
        if not incident_dir.is_dir():
            continue
        metadata_path = incident_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            # Check if candidate_id matches (stored in metadata or timeline)
            if (metadata.get("decision_id") == candidate_id or
                any(candidate_id in str(event) for event in metadata.get("timeline", []))):
                return incident_dir.name
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _update_incident_status(incident_id: str, status: IncidentStatus, description: str) -> bool:
    """Update incident status and add timeline entry."""
    incident_dir = INCIDENTS_DIR / incident_id
    metadata_path = incident_dir / "metadata.json"

    if not metadata_path.is_file():
        return False

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        metadata["status"] = status.value
        metadata["updated_at"] = time.time()
        metadata["timeline"].append({
            "timestamp": time.time(),
            "event_type": f"status_{status.value}",
            "description": description,
            "details": {}
        })

        if status == IncidentStatus.APPROVED:
            metadata["approved_at"] = time.time()
        elif status == IncidentStatus.REJECTED:
            metadata["rejected_at"] = time.time()
        elif status == IncidentStatus.RESOLVED:
            metadata["resolved_at"] = time.time()

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=_json_default)

        print(f"ACTION: Updated incident {incident_id} status to {status.value}")
        return True
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: Failed to update incident status: {e}")
        return False


def approve_incident(incident_id: str) -> Dict[str, Any]:
    """Approve an incident (human review)."""
    result = _update_incident_status(incident_id, IncidentStatus.APPROVED, "Incident approved by human reviewer")
    return {"success": result, "incident_id": incident_id, "status": IncidentStatus.APPROVED.value}


def reject_incident(incident_id: str) -> Dict[str, Any]:
    """Reject an incident (human review)."""
    result = _update_incident_status(incident_id, IncidentStatus.REJECTED, "Incident rejected by human reviewer")
    return {"success": result, "incident_id": incident_id, "status": IncidentStatus.REJECTED.value}


def get_incident_timeline(incident_id: str) -> List[Dict[str, Any]]:
    """Get incident timeline."""
    incident_dir = INCIDENTS_DIR / incident_id
    metadata_path = incident_dir / "metadata.json"

    if not metadata_path.is_file():
        return []

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata.get("timeline", [])
    except (OSError, json.JSONDecodeError):
        return []


def list_incidents(status: Optional[str] = None, severity: Optional[str] = None) -> List[Dict[str, Any]]:
    """List incidents with optional filtering."""
    if not INCIDENTS_DIR.is_dir():
        return []

    incidents = []
    for incident_dir in INCIDENTS_DIR.iterdir():
        if not incident_dir.is_dir():
            continue
        metadata_path = incident_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Apply filters
            if status and metadata.get("status") != status:
                continue
            if severity and metadata.get("severity") != severity:
                continue

            incidents.append({
                "incident_id": metadata.get("incident_id", incident_dir.name),
                "event_type": metadata.get("event_type"),
                "timestamp": metadata.get("timestamp"),
                "person_id": metadata.get("person_id"),
                "status": metadata.get("status"),
                "severity": metadata.get("severity"),
                "zone": metadata.get("zone"),
                "confidence": metadata.get("confidence"),
                "duration": metadata.get("duration"),
            })
        except (OSError, json.JSONDecodeError):
            continue

    # Sort by timestamp descending (newest first)
    incidents.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return incidents


ACTION_FUNCTIONS = {
    "continue_tracking": continue_tracking,
    "create_incident": create_incident,
    "resolve_incident": resolve_incident,
    "request_human_approval": request_human_approval,
    "none": no_action,
}


def execute_action(
    action_type: str,
    person_id: int,
    parameters: Dict[str, Any],
    frame: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    func = ACTION_FUNCTIONS.get(action_type, no_action)

    if func == create_incident:
        return func(person_id, parameters, frame, metadata)

    if func == request_human_approval:
        return func(str(person_id), parameters)

    return func(person_id, parameters)