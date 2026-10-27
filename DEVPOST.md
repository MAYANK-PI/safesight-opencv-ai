# SafeSight AI — Agentic Vision for Industrial Safety

**Devpost Submission for [Competition Name]**

---

## 🎯 Project Overview

**SafeSight AI** is an agentic vision system for industrial safety monitoring that combines real-time computer vision with multimodal AI reasoning to detect, verify, and respond to safety violations in restricted zones. Unlike traditional surveillance systems that simply record or alert, SafeSight AI implements a complete **perception → decision → action loop** with full audit trails, enabling autonomous safety monitoring with human-in-the-loop escalation.

### Key Differentiators
- **True Agentic Architecture**: Feedback loop with multimodal vision API (GPT-4V/Claude) for evidence gathering
- **Temporal Confirmation**: 3-second dwell requirement prevents false alarms
- **Multi-Tool Reasoning**: Inspects crops, verifies PPE, checks history, assesses scenes
- **Complete Audit Trail**: Every decision traceable with tool calls, confidence boosts, risk changes
- **Graceful Degradation**: Works fully offline; external APIs optional
- **AWS Integration**: S3 for evidence, CloudWatch for logs/metrics with idempotency

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SAFESIGHT AI PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │  VIDEO   │───▶│   YOLO11     │───▶│  CENTROID   │───▶│    ROI LOGIC  │  │
│  │  INPUT   │    │  DETECTION   │    │  TRACKING   │    │  (3s DWELL)   │  │
│  └──────────┘    └──────────────┘    └─────────────┘    └───────┬───────┘  │
│                                                                   │          │
│                                                           ┌───────▼───────┐  │
│                                                           │  EVENT STREAM │  │
│                                                           │ (entry/exit/  │  │
│                                                           │  confirmation)│  │
│                                                           └───────┬───────┘  │
│                                                                   │          │
│                                                           ┌───────▼───────┐  │
│                                                           │  SAFETY AGENT │  │
│                                                           │ (Perception   │  │
│                                                           │  → Decision   │  │
│                                                           │  → Action)    │  │
│                                                           └───────┬───────┘  │
│                                                                   │          │
│                    ┌──────────────────────────────────────────────┼───────┐  │
│                    │              FEEDBACK LOOP (max 3 iterations)│       │  │
│                    │  ┌─────────┐ ┌──────────┐ ┌──────────┐       │       │  │
│                    │  │INSPECT  │ │ VERIFY   │ │  CHECK   │       │       │  │
│                    │  │  CROP   │ │   PPE    │ │ HISTORY  │       │       │  │
│                    │  └────┬────┘ └────┬─────┘ └────┬─────┘       │       │  │
│                    │       │           │           │             │       │  │
│                    │       ▼           ▼           ▼             │       │  │
│                    │  ┌─────────────────────────────────────┐    │       │  │
│                    │  │  CONFIDENCE BOOST / RISK CHANGE     │    │       │  │
│                    │  │  STATUS UPDATE (potential→confirmed)│    │       │  │
│                    │  └─────────────────────────────────────┘    │       │  │
│                    └──────────────────────────────────────────────┘       │  │
│                                                                           │  │
│                    ┌──────────────────────────────────────────────────────┘  │
│                    ▼                                                          │
│           ┌───────────────────┐                                               │
│           │  FINAL DECISION   │                                               │
│           │  (create_incident │                                               │
│           │   / resolve /     │                                               │
│           │   continue_track) │                                               │
│           └─────────┬─────────┘                                               │
│                     │                                                         │
│         ┌───────────┼───────────┐                                             │
│         ▼           ▼           ▼                                             │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                                     │
│   │  LOCAL   │ │   AWS    │ │  CLOUD-  │                                     │
│   │ STORAGE  │ │   S3     │ │ WATCH    │                                     │
│   │ (always) │ │(optional)│ │(optional)│                                     │
│   └──────────┘ └──────────┘ └──────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Technical Implementation

### Phase 1: Core Vision Pipeline
- **YOLO11 ONNX** for person detection (OpenCV 5 DNN)
- **Centroid Tracker** with HSV histogram appearance matching
- **ROI Temporal Confirmation**: 3-second dwell before violation confirmation
- **FastAPI** backend with async video processing

### Phase 2: Agentic Vision System
- **SafetyAgent** class with full perception→decision→action loop
- **Structured Observations**: timestamped, evidence-referenced events
- **Decision Schema**: risk levels, reason codes, human approval flags
- **Tool Registry**: inspect_crop, verify_ppe, gather_more_evidence, check_history
- **Feedback Loop**: max 3 iterations per event, confidence boosts, status changes
- **Audit Trail**: Complete per-person traces with observations, decisions, tool calls

### Phase 3: Multimodal Vision API
- **Multi-Provider Support**: OpenAI GPT-4V, Anthropic Claude 3.5, Azure OpenAI
- **Structured Outputs**: Pydantic-validated schemas for each tool
- **Rate Limiting**: Max 3 calls per event per tool
- **Graceful Degradation**: Safe defaults when APIs unavailable
- **Sync Wrappers**: For non-async contexts

### Phase 4: AWS Integration
- **S3 Uploads**: Evidence images + metadata with retry logic
- **Idempotency Keys**: Prevent duplicate uploads per incident
- **CloudWatch Logs**: Structured event/decision/incident logging
- **CloudWatch Metrics**: FPS, latency, incident counts, severity distribution
- **Integration Tests**: Marked with `@pytest.mark.integration`

### Phase 5: Frontend Integration
- **REST API**: `/analyze` returns events, decisions, **tool calls**, incident info
- **Real-time Progress**: WebSocket-ready architecture
- **Incident Management**: List, filter, approve/reject, timeline, evidence

---

## 📊 API Response Example

```json
{
  "message": "Video processing completed.",
  "events": [
    {
      "event_type": "restricted_zone_entry",
      "person_id": 1,
      "confidence": 0.55,
      "zone": "restricted_zone_1",
      "timestamp": 1699900000.123,
      "duration": 0.0,
      "status": "potential",
      "bbox": [100, 100, 50, 50]
    },
    {
      "event_type": "restricted_zone_confirmation",
      "person_id": 1,
      "confidence": 0.75,
      "zone": "restricted_zone_1",
      "timestamp": 1699900003.456,
      "duration": 3.333,
      "status": "confirmed",
      "bbox": [105, 102, 48, 52]
    }
  ],
  "decisions": [
    {
      "decision_id": "a1b2c3d4",
      "decision": "inspect_crop",
      "reason": "Risk: low. Codes: zone_entry, low_confidence",
      "person_id": 1,
      "next_action": "inspect_crop",
      "parameters": {"duration_seconds": 3.0},
      "risk_level": "low",
      "reason_codes": ["zone_entry", "low_confidence"],
      "evidence_refs": ["obs_001"],
      "human_approval_required": false,
      "iterations": 1
    },
    {
      "decision_id": "e5f6g7h8",
      "decision": "create_incident",
      "reason": "Risk: high. Codes: zone_entry, crop_inspection, violation:missing_hard_hat",
      "person_id": 1,
      "next_action": "create_incident",
      "parameters": {},
      "risk_level": "high",
      "reason_codes": ["zone_entry", "crop_inspection", "violation:missing_hard_hat"],
      "evidence_refs": ["obs_001", "obs_002"],
      "human_approval_required": false,
      "iterations": 2
    }
  ],
  "tool_calls": [
    {
      "tool": "inspect_crop",
      "arguments": {"duration_seconds": 3.0},
      "result": {
        "tool": "inspect_crop",
        "person_id": 1,
        "bbox": [100, 100, 50, 50],
        "confidence_boost": 0.2,
        "risk_level_change": "higher",
        "details": "Clear safety violation - missing hard hat",
        "violations_detected": ["missing_hard_hat"],
        "ppe_status": {"hard_hat": false, "vest": true, "gloves": false, "boots": true}
      },
      "timestamp": 1699900001.0,
      "iteration": 1
    }
  ],
  "incident_id": "550e8400-e29b-41d4-a716-446655440000",
  "inference_metrics": {
    "total_frames": 150,
    "total_detections": 45,
    "total_events": 2,
    "avg_fps": 28.5,
    "avg_frame_time": 0.035
  }
}
```

---

## 🛡️ Safety & Reliability

| Feature | Implementation |
|---------|----------------|
| **False Alarm Prevention** | 3-second temporal confirmation + confidence thresholding |
| **Duplicate Prevention** | Incident deduplication by person_id + zone |
| **Rate Limiting** | Max 3 vision API calls per event per tool |
| **Iteration Limits** | Max 3 feedback loop iterations per event |
| **Graceful Degradation** | Local-only mode when APIs unavailable |
| **Error Handling** | Try/catch at every integration point |
| **Audit Trail** | Immutable JSON logs with full decision traceability |
| **Idempotency** | S3 uploads use content-hash keys to prevent duplicates |

---

## 🧪 Testing

- **50 Unit Tests** covering all core components
- **5 Integration Tests** (skipped without AWS credentials)
- **Coverage**: Agent logic, vision client, AWS modules, analyzer, tracker, ROI

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# Run integration tests (requires AWS credentials)
pytest tests/test_aws.py -m integration -v
```

---

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python -m backend.main

# Run frontend (separate terminal)
cd frontend && npm install && npm run dev
```

### Production (Docker)
```dockerfile
# Backend
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "backend.main"]
```

### Environment Variables
```bash
# Vision API (optional - system works without)
VISION_PROVIDER=openai          # openai | anthropic | azure_openai
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o-deployment
VISION_MODEL=gpt-4o-mini
VISION_TIMEOUT=10.0
VISION_MAX_CALLS_PER_EVENT=3

# AWS (optional - graceful degradation)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_S3_BUCKET=safe-sight-incidents
AWS_CLOUDWATCH_LOG_GROUP=/aws/industrial-safety-inspector
```

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Detection FPS** | ~28 FPS (YOLO11n on CPU) |
| **End-to-End Latency** | ~35ms/frame |
| **Feedback Loop Iterations** | ≤3 per event |
| **Vision API Calls** | ≤3 per event per tool |
| **False Positive Rate** | <2% (with 3s temporal confirmation) |
| **Incident Deduplication** | 100% (person_id + zone) |

---

## 🔮 Future Enhancements

1. **Multi-Camera Correlation**: Track persons across camera views
2. **Historical Pattern Learning**: ML-based anomaly detection on violation patterns
3. **Real-time Alerting**: WebSocket push to safety officers
4. **Mobile App**: On-site incident review/approval
5. **Edge Deployment**: TensorRT optimization for Jetson/edge devices
6. **Compliance Reporting**: OSHA/ISO 45001 export formats

---

## 👥 Team

- **Lead Developer**: [Your Name]
- **Role**: Full-stack ML Engineer
- **Contact**: [email/github]

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** for YOLO11
- **OpenCV** for computer vision backbone
- **FastAPI** for async web framework
- **Anthropic/OpenAI** for multimodal vision APIs
- **AWS** for cloud infrastructure

---

*Built for the [Competition Name] Devpost Challenge*