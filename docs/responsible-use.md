# Responsible Use, Limitations & Privacy — SafeSight AI

## Purpose Statement

SafeSight AI is designed **exclusively** for industrial safety monitoring in controlled environments (manufacturing floors, warehouses, robotics cells) where:
- Personnel are employees/contractors with safety training
- Restricted zones are clearly marked and documented
- Monitoring is disclosed per occupational safety regulations
- Human review is required before any consequential action

**This system is NOT designed for**: public surveillance, law enforcement, behavioral profiling, or any covert monitoring.

## Privacy Safeguards

| Safeguard | Implementation |
|-----------|----------------|
| **No facial recognition** | YOLO11 detects "person" class only; no identity features extracted |
| **No biometric data** | No gait analysis, face embeddings, or soft biometrics |
| **Minimal retention** | Evidence images: 90 days (configurable); Metadata: 1 year |
| **Access control** | API requires authentication; incident browser requires login |
| **Audit logging** | Every access to incident data logged with user, timestamp, action |
| **Data minimization** | Only annotated frame at confirmation moment stored; no continuous recording |

### Data Flow Privacy

```
Video Upload (user-initiated)
    │
    ▼
[Process in Memory] ──▶ No raw video persisted
    │
    ▼
[Single Annotated Frame] ──▶ Only confirmation moment saved
    │
    ▼
[Local Storage] ──▶ Optional S3 (encrypted, access-controlled)
    │
    ▼
[Auto-Expiry] ──▶ Configurable TTL (default 90 days)
```

## Limitations & Failure Modes

### Detection Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Occlusion** | Person behind equipment → missed detection | Tracker `max_disappeared=30` holds track briefly |
| **Lighting variation** | Low light → false negatives | Recommend IR/thermal cameras for 24/7 |
| **Camera angle** | Top-down vs side-view changes bbox | Calibrate ROI per camera; document limitations |
| **Crowding** | Overlapping persons → merged detections | `max_distance=50` limits association; tune per site |
| **Fast motion** | Motion blur → detection failure | Higher FPS camera; lower confirmation duration |

### Tracking Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **ID Switch** | Two people cross paths → swapped IDs | Centroid distance + motion model; log switches |
| **Re-entry = New ID** | Person leaves >30 frames → new track | Acceptable for incident logic (dedup on zone+time) |
| **Stationary objects** | Equipment detected as person | Confidence threshold 0.5; manual ROI exclusion |

### Agent Logic Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **No PPE detection** | Cannot verify hard hat/vest compliance | Interface defined; document how to add PPE model |
| **Single camera** | No multi-view fusion | Architecture supports future multi-camera |
| **Batch only** | No live stream processing | Design for video upload; streaming requires redesign |
| **Fixed ROI** | Per-request ROI only, not per-camera persistent | Add camera registry in Phase 2 |

### Adversarial / Edge Cases

| Scenario | System Behavior |
|----------|----------------|
| Video with no persons | Returns empty events, no incidents |
| Corrupt video | Validation rejects with 400 error |
| Extremely long video (>5 min) | Warning logged; processes but may timeout |
| Adversarial patch on clothing | YOLO11 robust to moderate; no certified defense |
| Camera tampering (spray paint) | Frame validation fails; health check alerts |

## Safety Constraints

### Hard Constraints (Never Violated)

1. **No autonomous machinery control** — System only creates incidents, requests human review
2. **No identity claims** — Never outputs "Person X is John Doe"
3. **No punitive automation** — Incidents require human approval before any HR/safety action
4. **Bounded loops** — Max 10 tool calls, 3 retries, 3s request window per track
5. **Fail-safe defaults** — Missing AWS → local only; missing model → clear error

### Human-in-the-Loop Requirements

| Trigger | Required Human Action |
|---------|----------------------|
| `risk_level == CRITICAL` | Review evidence, approve/reject incident |
| `confidence < 0.6` on confirmation | Verify before incident creation |
| `human_approval_required == true` | Explicit approve/reject in UI |
| Multiple incidents same person/day | Safety supervisor review |

## Regulatory Alignment

| Regulation | Alignment |
|------------|-----------|
| **OSHA 29 CFR 1910** | Supports general duty clause; documents hazard identification |
| **ISO 45001** | Incident documentation, corrective action tracking |
| **GDPR (if EU)** | No personal data processed; legitimate interest for worker safety |
| **CCPA (if CA)** | No consumer data; employee safety exception applies |
| **AI Act (EU)** | Not high-risk AI (no biometrics, no autonomous decisions) |

## Ethical Considerations

### Worker Trust
- **Transparency**: Cameras marked, purpose communicated, dashboard accessible to safety reps
- **Proportionality**: Only restricted zones monitored; break rooms, restrooms excluded
- **Accountability**: Every incident traceable to frame, detection, decision, reviewer

### Bias Mitigation
- YOLO11 trained on COCO (diverse demographics)
- No demographic classification attempted
- Confidence thresholds uniform across all detections
- Regular auditing: false positive rate by camera/zone (not by person)

### Misuse Prevention
- No API for "track person across facility"
- No facial recognition endpoint
- No behavioral analysis (loitering, pace, posture)
- Incident export requires admin role

## Deployment Checklist

Before production deployment:

- [ ] Privacy impact assessment completed
- [ ] Worker notification posted at all monitored zones
- [ ] Safety committee reviewed incident workflow
- [ ] AWS credentials rotated, least-privilege IAM roles
- [ ] S3 bucket encryption enabled (SSE-S3 or SSE-KMS)
- [ ] CloudWatch log retention set (30-90 days)
- [ ] Incident retention policy documented (legal hold procedure)
- [ ] Emergency stop: disable `/analyze` endpoint via feature flag
- [ ] Runbook for false alarm cascade documented

## Incident Response Playbook

| Event | Response |
|-------|----------|
| False positive spike (>5/hr) | Temporarily raise confidence threshold; investigate camera |
| Missed violation reported | Review ground truth; adjust ROI/confirmation duration |
| AWS unavailable | System degrades to local-only; alert ops |
| Data breach suspected | Rotate keys, audit S3 access logs, notify per policy |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01 | Initial responsible use documentation |

## Contact

For safety concerns, vulnerability reports, or ethical questions:
- **Security**: security@safesight.ai (PGP key in repo)
- **Safety**: safety@safesight.ai
- **Privacy**: privacy@safesight.ai