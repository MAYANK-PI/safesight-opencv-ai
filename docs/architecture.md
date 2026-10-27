# System Architecture — SafeSight AI

## Overview

SafeSight AI follows a modular, event-driven architecture with clear separation between perception, decision, and action layers. The system processes video files through an OpenCV 5 pipeline, feeds structured observations into an agentic decision engine, and persists auditable incidents with optional AWS cloud integration.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLIENT LAYER                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   React     │  │   Upload    │  │  Incident   │  │  Dashboard  │       │
│  │  Frontend   │  │  Component  │  │   Browser   │  │  (Metrics)  │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
└─────────┼────────────────┼────────────────┼────────────────┼──────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY (FastAPI)                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  POST /analyze    GET /incidents    GET /incidents/{id}             │   │
│  │  POST /incidents/{id}/approve  POST /incidents/{id}/reject          │   │
│  │  GET /incidents/{id}/timeline   GET /incidents/{id}/evidence.jpg    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  PERCEPTION      │  │  DECISION        │  │  ACTION          │
│  (Vision)        │  │  (Agent)         │  │  (Tools/Storage) │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ • PersonDetector │  │ • SafetyAgent    │  │ • create_incident│
│ • CentroidTracker│  │   - State Machine│  │ • resolve_incident│
│ • RegionOfInterest│ │   - Risk Assessment│ │ • request_approval│
│ • MotionAnalyzer │  │   - Tool Registry│  │ • continue_track │
│ • SafetyAnalyzer │  │   - Audit Trail  │  │ • S3 Upload      │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   PERSISTENCE        │
                    │  • Local JSON + JPG  │
                    │  • AWS S3 (optional) │
                    │  • CloudWatch (opt)  │
                    └──────────────────────┘
```

## Data Flow

### 1. Video Analysis Pipeline

```
Video File
    │
    ▼
[Validate Video] ──▶ [Extract Metadata] (width, height, fps, duration)
    │
    ▼
[Frame Loop] ──▶ [PersonDetector.detect(frame)] ──▶ YOLO11 ONNX inference
    │                     │
    │                     ▼
    │              [CentroidTracker.update(detections)]
    │                     │
    │                     ▼
    │              [MotionAnalyzer.update(tracks)]
    │                     │
    │                     ▼
    │              [RegionOfInterest.is_inside(centroid)]
    │                     │
    │                     ▼
    │              [State Machine: OUTSIDE → POTENTIAL → CONFIRMED → EXITED]
    │                     │
    │                     ▼
    └──────────▶ [Events: entry, confirmation, exit]
```

### 2. Agentic Decision Loop

```
Events from Analyzer
    │
    ▼
[Perceive] ──▶ Create Observation (structured, timestamped, evidence-ref)
    │
    ▼
[Interpret] ──▶ Assess Risk Level (low/medium/high/critical)
    │                   │
    │                   ├── confidence + duration + status
    │                   └── reason_codes (zone_entry, temporal_confirmation, etc.)
    ▼
[Assess] ──▶ Determine if violation supported by evidence
    │
    ▼
[Plan] ──▶ Select Action (CONTINUE_TRACKING, CREATE_INCIDENT, REQUEST_HUMAN_APPROVAL, etc.)
    │
    ▼
[Act] ──▶ Execute Tool (inspect_crop, check_history, verify_ppe)
    │         │
    │         ├── Tool result may boost confidence
    │         └── Verification recorded in audit trail
    ▼
[Verify] ──▶ Evaluate tool result, update risk if needed
    │
    ▼
[Escalate] ──▶ If critical risk or low confidence → REQUEST_HUMAN_APPROVAL
    │
    ▼
[Record] ──▶ Append to AgentTrace (per person_id)
    │         ├── observations[]
    │         ├── decisions[]
    │         ├── tool_calls[]
    │         └── timestamps
```

### 3. Incident Lifecycle

```
CREATE_INCIDENT action
    │
    ▼
[Deduplication Check] ──▶ Existing open incident for same person+zone?
    │                         │
    │                         ├── YES → Update existing (new evidence, extended duration)
    │                         └── NO  → Create new incident
    ▼
[Severity Assessment] ──▶ CRITICAL / HIGH / MEDIUM / LOW
    │                         │
    │                         ├── confidence ≥ 0.8 + duration ≥ 10s → CRITICAL
    │                         ├── confidence ≥ 0.7 + duration ≥ 5s  → HIGH
    │                         ├── confidence ≥ 0.5 + duration ≥ 3s  → MEDIUM
    │                         └── else → LOW
    ▼
[Initial Status] ──▶ CONFIRMED (default) or UNDER_REVIEW (if human_approval_required)
    │
    ▼
[Local Persistence] ──▶ incidents/{uuid}/
    │                         ├── metadata.json (full record + timeline)
    │                         └── evidence.jpg (annotated frame)
    │
    ▼
[AWS Sync] (async, best-effort)
    │                         ├── S3: upload evidence + metadata
    │                         ├── CloudWatch Logs: structured event log
    │                         └── CloudWatch Metrics: latency, counts, errors
    │
    ▼
[Human Review] (if UNDER_REVIEW)
    │                         ├── APPROVE → status=approved, approved_at
    │                         └── REJECT  → status=rejected, rejected_at
    │
    ▼
[Resolution] ──▶ On exit event → status=resolved, resolved_at
```

## Module Responsibilities

### Vision Modules (`backend/vision/`)

| Module | Responsibility | Key Classes/Functions |
|--------|---------------|----------------------|
| `detector.py` | YOLO11 ONNX person detection | `PersonDetector.detect(frame)` → List[bbox, conf] |
| `tracker.py` | Centroid-based multi-object tracking | `CentroidTracker.update(detections)` → Dict[id→track] |
| `roi.py` | Polygon restricted zone geometry | `RegionOfInterest.is_inside(point)` → bool |
| `motion.py` | Velocity/smoothing for tracks | `MotionAnalyzer.update(tracks)` → tracks with motion |
| `analyzer.py` | Orchestrates perception pipeline | `SafetyAnalyzer.update(frame)` → (annotated_frame, events) |

### Agent Modules (`backend/agent/`)

| Module | Responsibility | Key Classes/Functions |
|--------|---------------|----------------------|
| `agent.py` | Perception→Decision→Action loop | `SafetyAgent.update(events, frame)` → decisions |
| `actions.py` | Tool execution + incident persistence | `create_incident()`, `resolve_incident()`, etc. |

### API Layer (`backend/main.py`)

| Route | Purpose |
|-------|---------|
| `POST /analyze` | Upload video, run full pipeline, return results |
| `GET /health` | Health check |
| `GET /incidents` | List incidents with optional status/severity filters |
| `GET /incidents/{id}` | Full incident metadata |
| `GET /incidents/{id}/timeline` | Event timeline |
| `GET /incidents/{id}/evidence.jpg` | Evidence image |
| `POST /incidents/{id}/approve` | Human approval |
| `POST /incidents/{id}/reject` | Human rejection |

## Concurrency Model

- **FastAPI async** for I/O (file upload, HTTP responses)
- **Thread pool** (`run_in_threadpool`) for CPU-bound video processing
- **Per-request analyzer reset** — no cross-video state leakage
- **Per-request agent instance** — fresh decision state per video

## Configuration

| Setting | Source | Default |
|---------|--------|---------|
| Model path | `MODEL_PATH` constant | `models/yolo11n.onnx` |
| Default ROI | `ROI_POLYGON` constant | `[(0,350), (800,350), (800,800), (0,800)]` |
| Confirmation duration | `CONFIRMATION_DURATION` | `3.0` seconds |
| Request duration (agent) | `REQUEST_DURATION` | `3.0` seconds |
| Detection confidence | `confidence_threshold` | `0.5` |
| NMS threshold | `nms_threshold` | `0.4` |
| AWS region | `AWS_REGION` env | — |
| S3 bucket | `AWS_S3_BUCKET` env | — |
| CloudWatch log group | `AWS_CLOUDWATCH_LOG_GROUP` | `/aws/industrial-safety-inspector` |

## Security Considerations

- No credentials in code — all via environment variables
- Input validation on video files (format, dimensions, duration limits)
- CORS restricted to known frontend origins
- UUID-based incident IDs (no sequential enumeration)
- Evidence served via authenticated API (not direct filesystem)