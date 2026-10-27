<<<<<<< HEAD
# SafeSight AI — Agentic Vision for Industrial Safety Inspection

[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-blue)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19+-blue)](https://react.dev/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **OpenCV + AWS Competition Entry** — An agentic vision system that detects restricted zone violations in industrial environments using OpenCV 5, YOLO11, and a full perception→decision→action loop with human-in-the-loop approval.

## 🎯 Problem Statement

Industrial facilities face critical safety risks when personnel enter restricted zones (robot cells, high-voltage areas, chemical storage). Traditional monitoring relies on human operators watching camera feeds — prone to fatigue, distraction, and slow response times.

**SafeSight AI** automates this with:
- **Real-time person detection** using YOLO11 ONNX on OpenCV 5
- **Multi-object tracking** with centroid-based association
- **Temporal confirmation** (3-second dwell) to reduce false alarms
- **Agentic decision loop** that requests evidence, escalates to humans, and creates auditable incidents
- **AWS integration** for evidence storage (S3) and observability (CloudWatch)

## ✨ Features

| Category | Features |
|----------|----------|
| **Computer Vision** | YOLO11 person detection, centroid tracking, configurable ROI polygons, temporal confirmation (3s) |
| **Agentic Workflow** | Perception→Decision→Action loop, tool calling (crop inspection, history check), human approval escalation |
| **Incident Management** | Deduplication, lifecycle (candidate→confirmed→review→approved/rejected→resolved), severity levels, timeline |
| **AWS Integration** | S3 evidence upload with retry, CloudWatch Logs + Metrics, graceful local-only fallback |
| **Frontend** | Upload progress, inference metrics dashboard, severity badges, incident search/filter, approval UI |
| **Evaluation** | Precision/Recall/F1 scripts, latency tracking, cost estimation |

## 🏗 Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Video     │────▶│  OpenCV 5 / YOLO │────▶│  SafetyAnalyzer │
│   Upload    │     │  Person Detection│     │  (Perception)   │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Frontend   │◀───│   FastAPI REST   │◀───│  SafetyAgent    │
│  (React)    │     │   + WebSocket    │     │  (Decision)     │
└─────────────┘     └────────┬─────────┘     └────────┬────────┘
                              │                      │
                              ▼                      ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │  Incident Store │     │   Tool Calls    │
                    │  (Local/JSON)   │     │  (Inspect/Check)│
                    └────────┬────────┘     └────────┬────────┘
                             │                       │
                             ▼                       ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │      AWS        │     │  Human Review   │
                    │  S3 + CloudWatch│     │  (Approval UI)  │
                    └─────────────────┘     └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- OpenCV 5 compatible system
- (Optional) AWS credentials for cloud features

### Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download YOLO11n ONNX model (place in models/)
# wget -O models/yolo11n.onnx https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx

uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### AWS Configuration (Optional)
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
export AWS_S3_BUCKET=your-evidence-bucket
export AWS_CLOUDWATCH_LOG_GROUP=/aws/industrial-safety-inspector
```

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System architecture, data flows, component details |
| [Agentic Workflow](docs/agentic-workflow.md) | Perception→Decision→Action loop, tool calling, escalation logic |
| [Evaluation](docs/evaluation.md) | Metrics methodology, test procedures, benchmark results |
| [Responsible Use](docs/responsible-use.md) | Privacy, limitations, failure modes, safety constraints |
| [Deployment](docs/deployment.md) | Docker, ECS, Lambda, local development |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run evaluation on test video
python evaluation/evaluate.py --video test_video.mp4
```

## 📊 Evaluation Results

| Metric | Value | Notes |
|--------|-------|-------|
| Detection Precision | 0.89 | YOLO11n @ 0.5 conf |
| Detection Recall | 0.85 | Test set: 50 videos |
| F1 Score | 0.87 | Balanced |
| False Incidents/min | 0.12 | Low false alarm rate |
| Duplicate Rate | 0% | Deduplication active |
| Avg Processing Latency | 92ms/frame | CPU only (no GPU) |
| Avg FPS | 10.8 | YOLO11n on i7-12700 |
| Est. Cost/video | $0.0003 | AWS Fargate + S3 + CW |

*Detailed results in [evaluation.md](docs/evaluation.md)*

## 🐳 Docker Deployment

```bash
# Build
docker build -t safesight-ai .

# Run
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_REGION \
  -e AWS_S3_BUCKET \
  safesight-ai
```

## ☁️ AWS Deployment

### ECS Fargate (Recommended for Production)
```bash
# Deploy using the ECS task definition
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.json
aws ecs create-service --cluster safesight --service-name safesight-backend --task-definition safesight-ai
```

### Lambda (For Short Videos Only)
```bash
# Deploy via SAM
sam build --template deploy/lambda-config.yml
sam deploy --guided
```

## ⚠️ Limitations & Failure Modes

| Limitation | Mitigation |
|------------|------------|
| No PPE detection in base model | Optional PPE detector interface; document how to add |
| Occlusion handling limited | Tracker max_disappeared=30; document tuning |
| Lighting sensitivity | Recommend IR/thermal cameras for 24/7 |
| Single-camera only | Multi-camera fusion not implemented |
| No real-time streaming | Batch video processing only |

See [Responsible Use](docs/responsible-use.md) for complete analysis.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/ -v`
4. Submit a PR with description of changes

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- OpenCV team for OpenCV 5
- Ultralytics for YOLO11
- FastAPI and React communities
- AWS for cloud credits (competition)

---

**Competition Entry**: OpenCV + AWS Competition 2024
**Team**: SafeSight AI
**Demo**: [5-minute walkthrough](docs/demo-script.md)
=======
# safesight-opencv-ai
its an opencv-ai vision project
>>>>>>> 8a8604e7042b1470c91a15574106272036560ca8
