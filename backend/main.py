import cv2
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

# Load environment variables from .env file at the very start
# This must happen before any modules that use os.getenv() are imported
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from backend.vision.analyzer import SafetyAnalyzer
from backend.agent.agent import SafetyAgent
from backend.agent.actions import (
    continue_tracking,
    create_incident,
    resolve_incident,
    request_human_approval,
    no_action,
    list_incidents as list_incidents_filtered,
)

# AWS CloudWatch integration (optional)
try:
    from backend.aws.cloudwatch import log_processing_metrics, is_cloudwatch_available
    CLOUDWATCH_AVAILABLE = True
except ImportError:
    CLOUDWATCH_AVAILABLE = False
    log_processing_metrics = None
    is_cloudwatch_available = None

# ------------------ Configuration ------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCIDENTS_DIR = PROJECT_ROOT / "incidents"
TEMP_DIR = PROJECT_ROOT / "temp"

MODEL_PATH = str(PROJECT_ROOT / "models" / "yolo11n.onnx")

# TEST ROI ONLY: calibrate this polygon for your actual camera.
ROI_POLYGON = [(0, 350), (800, 350), (800, 800), (0, 800)]

CONFIRMATION_DURATION = 3.0
REQUEST_DURATION = 3.0

# ------------------ Global instances ------------------

if not os.path.isfile(MODEL_PATH):
    print(f"Warning: Model file not found: {MODEL_PATH}")

# Analyzer is global but stateless after reset()
analyzer = SafetyAnalyzer(
    model_path=MODEL_PATH,
    roi_polygon=ROI_POLYGON,
    confirmation_duration=CONFIRMATION_DURATION,
    confidence_threshold=0.5,
    nms_threshold=0.4,
)

# Agent factory - create fresh instance per request to avoid cross-video state leakage
def create_agent() -> SafetyAgent:
    return SafetyAgent(request_duration=REQUEST_DURATION)

# ------------------ FastAPI app ------------------

app = FastAPI(
    title="AI Industrial Safety Inspector",
    version="0.2.0",
)

# Allow the local Vite frontend to call this API.
# CORS origins can be extended via CORS_ORIGINS environment variable (comma-separated)
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:5176",
    "http://127.0.0.1:5176",
    "http://localhost:5177",
    "http://127.0.0.1:5177",
    "http://localhost:5178",
    "http://127.0.0.1:5178",
    "http://localhost:5179",
    "http://127.0.0.1:5179",
]

# Parse additional origins from environment variable
extra_origins = os.getenv("CORS_ORIGINS", "").split(",")
extra_origins = [o.strip() for o in extra_origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=default_origins + extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Helper functions ------------------

def validate_video_file(video_path: Path) -> Dict[str, Any]:
    """
    Validate a video file and return metadata.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with video metadata: width, height, fps, frame_count, duration.

    Raises:
        HTTPException: If video is invalid, corrupt, or empty.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail="Could not open video file. File may be corrupt or unsupported format.",
        )

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid video dimensions.",
        )

    if fps <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid video FPS.",
        )

    if frame_count <= 0:
        raise HTTPException(
            status_code=400,
            detail="Video contains no frames.",
        )

    duration = frame_count / fps

    # Warn if duration is very long (potential timeout issues)
    if duration > 300:  # 5 minutes
        print(f"Warning: Video duration is {duration:.1f}s, processing may take a while.")

    return {
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "frame_count": int(frame_count),
        "duration": duration,
    }


def validate_incident_id(incident_id: str) -> str:
    """Validate and normalize a UUID incident ID."""
    try:
        return str(uuid.UUID(incident_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid incident ID.",
        )


def get_incident_dirs() -> List[Path]:
    """Return incident directories, newest first."""
    if not INCIDENTS_DIR.is_dir():
        return []

    directories = [
        path for path in INCIDENTS_DIR.iterdir()
        if path.is_dir()
    ]
    directories.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return directories


def load_incident_metadata(
    incident_id: str,
) -> Optional[Dict[str, Any]]:
    """Load metadata for a validated incident."""
    incident_id = validate_incident_id(incident_id)
    metadata_path = INCIDENTS_DIR / incident_id / "metadata.json"

    if not metadata_path.is_file():
        return None

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail="Could not read incident metadata.",
        )


def save_incident_status(
    incident_id: str,
    expected_status: str,
) -> Dict[str, str]:
    """Update incident status and its corresponding timestamp."""
    incident_id = validate_incident_id(incident_id)
    metadata_path = INCIDENTS_DIR / incident_id / "metadata.json"

    if not metadata_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Incident not found.",
        )

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        metadata["status"] = expected_status
        metadata[f"{expected_status}_at"] = time.time()

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail="Could not update incident status.",
        )

    return {
        "message": f"Incident {incident_id} {expected_status}.",
        "incident_id": incident_id,
        "status": expected_status,
    }


# ------------------ Pydantic models ------------------

class AnalyzeRequest(BaseModel):
    roi_polygon: Optional[List[List[int]]] = None


class AnalyzeResponse(BaseModel):
    message: str
    events: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]] = []  # Tool calls from agent feedback loop
    incident_id: Optional[str] = None
    video_metadata: Optional[Dict[str, Any]] = None
    inference_metrics: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(serialize_by_default=True)


class IncidentInfo(BaseModel):
    incident_id: str
    event_type: str
    timestamp: float
    person_id: int
    status: str
    severity: Optional[str] = None
    zone: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None


class IncidentFilter(BaseModel):
    status: Optional[str] = None
    severity: Optional[str] = None


class StatusUpdate(BaseModel):
    status: str


# ------------------ Routes ------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "timestamp": time.time(),
    }


def _process_video(
    temp_file: Path, analyzer: SafetyAnalyzer, roi_polygon: Optional[List[List[int]]] = None
) -> AnalyzeResponse:
    """Process video synchronously in a thread pool. This avoids blocking the event loop."""
    # Create fresh agent instance per request to avoid cross-video state leakage
    agent = create_agent()

    # Explicitly reset video session for vision API budget tracking
    agent.reset_video_session()

    # Set custom ROI if provided
    if roi_polygon is not None:
        # Convert List[List[int]] to List[Tuple[int, int]]
        analyzer.set_roi([tuple(point) for point in roi_polygon])

    cap = None
    events_all = []
    decisions_all = []
    incident_id = None

    try:
        # Reset analyzer state for fresh video analysis
        analyzer.reset()

        cap = cv2.VideoCapture(str(temp_file))
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open video file.",
            )

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1

            annotated_frame, events = analyzer.update(frame)

            if not events:
                continue

            events_all.extend(events)

            # Pass frame and frame_number to agent for tool calls and tracking
            decisions = agent.update(events, frame=annotated_frame, frame_number=frame_number)
            decisions_all.extend(decisions)

            # Match decisions to events (agent processes events sequentially, so order matches)
            for event, decision in zip(events, decisions):
                action_type = decision.get("next_action")
                person_id = decision.get("person_id")
                parameters = decision.get("parameters") or {}
                decision_incident_id = decision.get("incident_id")

                action_metadata = {
                    "event_type": event.get("event_type"),
                    "confidence": event.get("confidence"),
                    "zone": event.get("zone"),
                    "timestamp": event.get("timestamp"),
                    "duration": event.get("duration"),
                    "decision_id": decision.get("decision_id"),
                    "risk_level": decision.get("risk_level"),
                    "reason_codes": decision.get("reason_codes"),
                    "evidence_refs": decision.get("evidence_refs"),
                    "human_approval_required": decision.get("human_approval_required"),
                }

                if action_type == "continue_tracking":
                    continue_tracking(person_id, parameters)

                elif action_type == "create_incident":
                    incident_id = create_incident(
                        person_id,
                        parameters,
                        annotated_frame,
                        action_metadata,
                    )

                elif action_type == "resolve_incident":
                    resolve_incident(person_id, parameters)

                elif action_type == "request_human_approval":
                    # Use incident_id from agent decision if available (created internally for critical violations)
                    if decision_incident_id:
                        incident_id = decision_incident_id
                    request_human_approval(
                        str(person_id),
                        parameters,
                    )

                else:
                    no_action(person_id, parameters)

        # Capture inference metrics before returning
        metrics = analyzer.get_metrics()

        # Collect tool calls from agent traces
        tool_calls_all = []
        for trace in agent.get_all_traces().values():
            tool_calls_all.extend(trace.tool_calls)

        # Log processing metrics to CloudWatch (if available)
        if CLOUDWATCH_AVAILABLE and log_processing_metrics:
            try:
                log_processing_metrics(
                    fps=metrics.get("avg_fps", 0.0),
                    latency_ms=metrics.get("avg_frame_time", 0.0) * 1000,
                    frame_count=metrics.get("total_frames", 0),
                    detections=metrics.get("total_detections", 0),
                    events=metrics.get("total_events", 0)
                )
            except Exception as e:
                print(f"Warning: CloudWatch metrics logging failed: {e}")

        return AnalyzeResponse(
            message="Video processing completed.",
            events=events_all,
            decisions=decisions_all,
            tool_calls=tool_calls_all,
            incident_id=incident_id,
            inference_metrics=metrics,
        )

    finally:
        if cap is not None:
            cap.release()

        try:
            temp_file.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Warning: Could not remove temporary video: {exc}")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(file: UploadFile = File(...), roi_polygon: Optional[str] = None):
    """Analyze an uploaded video and create safety incidents.

    Args:
        file: Video file to analyze
        roi_polygon: Optional JSON string of ROI polygon points, e.g. '[[0, 350], [800, 350], [800, 800], [0, 800]]'
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Parse optional ROI polygon from form field
    parsed_roi = None
    if roi_polygon:
        try:
            parsed_roi = json.loads(roi_polygon)
            # Validate it's a list of points
            if not isinstance(parsed_roi, list) or not all(
                isinstance(p, list) and len(p) == 2 and all(isinstance(c, (int, float)) for c in p)
                for p in parsed_roi
            ):
                raise ValueError("Invalid ROI format")
        except (json.JSONDecodeError, ValueError) as e:
            await file.close()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid roi_polygon format. Expected JSON array of [x, y] points. Error: {e}",
            )

    # Do not use the client-provided filename as a filesystem path.
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Unsupported video extension.",
        )

    temp_file = TEMP_DIR / f"{uuid.uuid4()}{suffix}"

    try:
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate video file before processing
        video_meta = validate_video_file(temp_file)

        # Offload CPU-bound video processing to a thread pool to avoid blocking the event loop
        response = await run_in_threadpool(
            _process_video,
            temp_file,
            analyzer,
            parsed_roi,
        )

        # Add video metadata to response
        response_dict = response.model_dump(exclude_none=False)
        response_dict["video_metadata"] = video_meta
        return AnalyzeResponse(**response_dict)

    finally:
        await file.close()


def _get_or_default(data: Dict, key: str, default: Any) -> Any:
    """Get value from dict, returning default if key missing or value is None."""
    val = data.get(key)
    return default if val is None else val


@app.get("/incidents", response_model=List[IncidentInfo])
def list_incidents(status: Optional[str] = None, severity: Optional[str] = None):
    """List saved incidents with optional filtering."""
    raw_incidents = list_incidents_filtered(status=status, severity=severity)

    incidents = []
    for inc in raw_incidents:
        incidents.append(
            IncidentInfo(
                incident_id=_get_or_default(inc, "incident_id", str(uuid.uuid4())),
                event_type=_get_or_default(inc, "event_type", "unknown"),
                timestamp=_get_or_default(inc, "timestamp", 0.0),
                person_id=_get_or_default(inc, "person_id", 0),
                status=_get_or_default(inc, "status", "unknown"),
                severity=inc.get("severity"),
                zone=inc.get("zone"),
                confidence=inc.get("confidence"),
                duration=inc.get("duration"),
            )
        )

    return incidents


@app.get("/incidents/{incident_id}/evidence.jpg")
def get_incident_evidence(incident_id: str):
    """Serve the JPEG evidence image for an incident."""
    incident_id = validate_incident_id(incident_id)
    evidence_path = INCIDENTS_DIR / incident_id / "evidence.jpg"

    if not evidence_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Evidence image not found.",
        )

    return FileResponse(
        path=str(evidence_path),
        media_type="image/jpeg",
        filename="evidence.jpg",
    )


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str):
    """Get full incident metadata."""
    metadata = load_incident_metadata(incident_id)

    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found.",
        )

    return metadata


@app.get("/incidents/{incident_id}/timeline")
def get_incident_timeline(incident_id: str):
    """Get incident timeline."""
    incident_id = validate_incident_id(incident_id)
    metadata = load_incident_metadata(incident_id)

    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found.",
        )

    return metadata.get("timeline", [])


@app.post("/incidents/{incident_id}/approve")
def approve_incident(
    incident_id: str,
    status_update: StatusUpdate,
):
    """Approve an incident."""
    if status_update.status != "approved":
        raise HTTPException(
            status_code=400,
            detail="Status must be 'approved'.",
        )

    return save_incident_status(incident_id, "approved")


@app.post("/incidents/{incident_id}/reject")
def reject_incident(
    incident_id: str,
    status_update: StatusUpdate,
):
    """Reject an incident."""
    if status_update.status != "rejected":
        raise HTTPException(
            status_code=400,
            detail="Status must be 'rejected'.",
        )

    return save_incident_status(incident_id, "rejected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
