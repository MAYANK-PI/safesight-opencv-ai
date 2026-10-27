# Demo Script — SafeSight AI

## 5-Minute Demonstration Walkthrough

### Target Audience
- OpenCV + AWS Competition Judges
- Technical evaluators
- Safety engineering stakeholders

### Demo Objectives
1. Show end-to-end video analysis pipeline
2. Demonstrate agentic decision loop (evidence → action)
3. Highlight human-in-the-loop approval workflow
4. Show AWS integration (S3 + CloudWatch)
5. Display evaluation metrics dashboard

---

## Pre-Demo Setup (5 minutes before)

### Environment Checklist
- [ ] Backend running: `uvicorn main:app --host 0.0.0.0 --port 8000`
- [ ] Frontend running: `npm run dev` (opens http://localhost:5173)
- [ ] Test videos prepared in `test_videos/` folder
- [ ] AWS credentials configured (optional but recommended)
- [ ] Incidents folder empty for clean demo
- [ ] Screen recording started (if recording)

### Test Videos to Have Ready
| Video | Duration | Expected Outcome |
|-------|----------|------------------|
| `person_enters_zone.mp4` | 10s | Creates CRITICAL incident |
| `person_brief_entry.mp4` | 5s | No incident (exits before 3s) |
| `multiple_people.mp4` | 15s | Multiple tracks, deduplication |
| `no_people.mp4` | 5s | Empty events, clean response |

---

## Demo Flow (5 minutes)

### 0:00–0:30 | Introduction & Architecture Overview

**Script:**
> "SafeSight AI is an agentic vision system for industrial safety inspection. It detects restricted zone violations using OpenCV 5 with YOLO11, tracks people with centroid tracking, and implements a full perception→decision→action loop with human-in-the-loop approval."

**Show:**
- Architecture diagram (from README or architecture.md)
- Key components: PersonDetector → CentroidTracker → SafetyAnalyzer → SafetyAgent → Incident Store

**Key Points:**
- OpenCV 5 DNN for ONNX inference (no PyTorch/TensorFlow runtime)
- Deterministic safety logic (no ML overrides)
- AWS integration with graceful local fallback

---

### 0:30–1:30 | Live Video Analysis

**Actions:**
1. Open frontend at http://localhost:5173
2. Click "Upload Video" 
3. Select `person_enters_zone.mp4`
4. Show upload progress bar
5. Wait for processing (3–5 seconds)

**Script during processing:**
> "The video is being processed frame-by-frame. YOLO11n detects persons, the centroid tracker maintains IDs across frames, and the region-of-interest check determines zone entry. The state machine requires 3 seconds of confirmed presence before flagging a violation."

**Show results:**
- Inference metrics dashboard (FPS, latency, detections)
- Events timeline (entry → confirmation)
- Decision log (risk assessment → action selection)

**Key metrics to highlight:**
- FPS: ~10–12 (CPU only)
- Per-frame latency: ~90ms
- Detection confidence shown

---

### 1:30–2:30 | Agentic Decision Loop Deep Dive

**Actions:**
1. Click on the created incident in the list
2. Open Incident Detail view
3. Show timeline: `restricted_zone_entry` → `restricted_zone_confirmation` → `incident_created`
4. Show decision metadata:
   - Risk level: CRITICAL
   - Reason codes: `zone_entry`, `temporal_confirmation`
   - Evidence refs linking to observation IDs
   - Human approval required: true

**Script:**
> "This is the agentic workflow in action. The agent PERCEIVED the entry event, INTERPRETED it with tracker history, ASSESSED risk as CRITICAL (high confidence + confirmed duration), PLANNED the CREATE_INCIDENT action, ACTED by creating the incident record, and ESCALATED to human review because confidence exceeded 0.8."

**Show:**
- Agent trace (audit trail) — every observation, decision, tool call
- Evidence image (annotated frame at confirmation moment)
- Severity badge: CRITICAL (red)

---

### 2:30–3:30 | Human-in-the-Loop Approval

**Actions:**
1. In Incident Detail, click "Approve" button
2. Show status change: `under_review` → `approved`
3. Show timestamp recorded: `approved_at`
4. Click "Reject" on another incident (if available) to show rejection flow

**Script:**
> "No automated system should take punitive action without human oversight. The agent creates a CANDIDATE incident and requests approval. The safety reviewer sees the evidence frame, confidence score, duration, and risk rationale — then explicitly approves or rejects."

**Show:**
- Approval/rejection API calls in network tab
- Timeline updated with `status_approved` event

---

### 3:30–4:00 | AWS Integration (S3 + CloudWatch)

**Actions:**
1. In Incident Detail, show S3 URLs for evidence and metadata
2. Open CloudWatch console (if AWS configured):
   - Log group: `/aws/industrial-safety-inspector`
   - Metrics: `SafeSightAI` namespace
3. Show structured log entry for the incident
4. Show metric: `ProcessingLatencyMs`, `IncidentsCreated`

**Script:**
> "Production deployments get full observability. Evidence images and metadata are uploaded to S3 with encryption. CloudWatch Logs capture structured incident events. CloudWatch Metrics track latency, throughput, and error rates — all with zero code changes when AWS credentials are present."

**If no AWS configured:**
> "Notice the system works fully locally — AWS is optional. The 'Local' storage backend indicator shows graceful degradation."

---

### 4:00–4:30 | Edge Cases & Robustness

**Actions:**
1. Upload `person_brief_entry.mp4` (person enters 2s, exits)
2. Show: No incident created (temporal confirmation prevents false alarms)
3. Upload `multiple_people.mp4`
4. Show: Multiple tracks, each evaluated independently
5. Show deduplication: Re-upload same video → no duplicate incident

**Script:**
> "Three key robustness features: temporal confirmation (3s dwell) eliminates brief-passage false alarms; centroid tracking handles occlusion and ID persistence; deduplication prevents multiple incidents for the same person-zone event."

---

### 4:30–5:00 | Evaluation & Competition Highlights

**Actions:**
1. Open evaluation results (docs/evaluation.md or run `python evaluation/evaluate.py`)
2. Show metrics table:
   - F1: 0.87
   - False incidents/min: 0.12
   - Duplicate rate: 0%
   - Avg latency: 92ms/frame
   - Cost/video: $0.00034

**Closing Script:**
> "SafeSight AI demonstrates production-ready agentic vision: OpenCV 5 for high-performance inference, deterministic safety logic for trustworthiness, human-in-the-loop for accountability, and cloud-native observability. The system is competition-ready with comprehensive tests, documentation, and deployment configurations."

---

## Shot List (for video recording)

| Time | Shot | Duration | Notes |
|------|------|----------|-------|
| 0:00 | Architecture diagram | 15s | Pan/zoom on components |
| 0:15 | Frontend home page | 10s | Show upload area |
| 0:25 | File selection dialog | 5s | Select test video |
| 0:30 | Upload progress bar | 5s | Animated |
| 0:35 | Results dashboard | 15s | Metrics, events, decisions |
| 0:50 | Incident list | 10s | Show new incident |
| 1:00 | Incident detail view | 20s | Timeline, evidence, metadata |
| 1:20 | Agent trace (JSON) | 10s | Scroll through audit trail |
| 1:30 | Evidence image | 10s | Annotated frame with bbox |
| 1:40 | Approve button click | 10s | Status change animation |
| 1:50 | CloudWatch console | 15s | Logs + Metrics |
| 2:05 | Brief entry test | 15s | No incident result |
| 2:20 | Multiple people test | 15s | Multiple tracks |
| 2:35 | Deduplication test | 10s | Re-upload same video |
| 2:45 | Evaluation metrics | 15s | Table from docs |
| 3:00 | Code snippets (optional) | 15s | agent.py decision loop |

---

## Talking Points (Backup)

### Technical Differentiators
1. **Agentic, not just analytic** — Evidence changes behavior (tools, escalation, retry)
2. **Deterministic safety** — Geometry + time, no ML hallucination risk
3. **Audit-first design** — Every decision traceable to pixels
4. **Cloud-agnostic** — Runs locally, adds AWS when available
5. **Competition-aligned** — OpenCV 5, ONNX, AWS services

### Common Questions & Answers

| Question | Answer |
|----------|--------|
| Why not YOLOv8/v10? | YOLO11n is latest Ultralytics, better accuracy/speed tradeoff |
| Why centroid tracking not DeepSORT? | Lightweight, no appearance features needed for zone monitoring |
| Can it run on Jetson/Edge? | Yes — ONNX + OpenCV DNN runs on ARM, ~15 FPS on Jetson Orin |
| How to add PPE detection? | Implement `VERIFY_PPE` tool with separate PPE model (interface ready) |
| Multi-camera support? | Architecture supports it — add camera registry, fuse tracks by ReID |
| Real-time streaming? | Current: batch upload. Streaming: replace file loop with RTSP/WS frame source |

---

## Demo Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| Backend not responding | Check `uvicorn` process, port 8000 free |
| CORS error | Frontend port in `allow_origins` (5173, 5174) |
| Model not found | Verify `models/yolo11n.onnx` exists |
| No detections | Check ROI polygon matches video resolution |
| AWS errors | Verify credentials, bucket exists, region matches |
| Frontend blank | `npm run dev`, check console for errors |

---

## Post-Demo

1. Share GitHub repo link
2. Provide `docs/` folder for deep dive
3. Offer to run evaluation on custom video
4. Discuss production hardening needs (auth, rate limits, multi-tenancy)

---

## Appendix: One-Command Demo Setup

```bash
# Clone and run everything
git clone <repo>
cd opencv-agentic-vision

# Backend
cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cd .. && uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Frontend
cd frontend && npm install && npm run dev &

# Open http://localhost:5173
```