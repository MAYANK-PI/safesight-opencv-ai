# Agentic Vision Workflow — SafeSight AI

## Overview

The agentic workflow is the core innovation of SafeSight AI. Unlike traditional pipelines that merely detect and report, this system implements a full **perception → decision → action** loop where visual evidence directly changes subsequent behavior.

## Core Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGENTIC DECISION LOOP                                │
│                                                                             │
│   PERCEIVE ──▶ INTERPRET ──▶ ASSESS ──▶ PLAN ──▶ ACT ──▶ VERIFY ──▶ ESCALATE ──▶ RECORD │
│       │           │           │         │         │        │         │            │
│       ▼           ▼           ▼         ▼         ▼        ▼         ▼            ▼
│   Raw events   Structured  Risk      Action    Tool     Result   Human      Audit
│   from CV      observations level     selection  calls   check    approval   trail
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stage Details

### 1. PERCEIVE — Structured Observations

Raw events from the vision analyzer are converted into structured `Observation` objects:

```python
@dataclass
class Observation:
    observation_id: str      # Unique ID for traceability
    timestamp: float         # Unix timestamp
    event_type: str          # restricted_zone_entry, _confirmation, _exit
    person_id: int           # Tracker-assigned ID
    confidence: float        # Detection confidence (0-1)
    zone: str                # Zone identifier
    duration: float          # Time in zone (seconds)
    status: str              # potential, confirmed, resolved
    bbox: List[int]          # Bounding box [x, y, w, h]
    frame_number: int        # Video frame index
```

**Key principle**: Every observation carries evidence references (`observation_id`) that link back to the specific frame and detection.

### 2. INTERPRET — Context Enrichment

Observations are enriched with context from the agent's memory:

- **Track history**: Previous observations for this `person_id`
- **Zone geometry**: ROI polygon, entry/exit boundaries
- **Temporal state**: How long in each state (OUTSIDE, POTENTIAL, CONFIRMED)
- **Tool availability**: Which inspection tools are registered

### 3. ASSESS — Risk Evaluation

Risk level is computed deterministically from evidence:

| Factor | Weight | Thresholds |
|--------|--------|------------|
| Confidence | 40% | ≥0.8 critical, ≥0.7 high, ≥0.5 medium |
| Duration in zone | 30% | ≥10s critical, ≥5s high, ≥3s medium |
| Event status | 30% | confirmed=critical, potential=low |

**Risk Levels**:
- **CRITICAL**: Confirmed violation, high confidence, long duration
- **HIGH**: Confirmed or near-confirmation with strong evidence
- **MEDIUM**: Potential violation with moderate evidence
- **LOW**: Early potential, brief entry

### 4. PLAN — Action Selection

The agent selects from an allowlisted set of actions:

| Action | Trigger | Parameters |
|--------|---------|------------|
| `CONTINUE_TRACKING` | New potential entry | `duration_seconds` |
| `CREATE_INCIDENT` | Confirmed violation (risk ≥ HIGH) | — |
| `RESOLVE_INCIDENT` | Exit after confirmed | — |
| `REQUEST_HUMAN_APPROVAL` | Critical risk OR low confidence | `candidate_id` |
| `INSPECT_CROP` | Uncertain detection, need detail | `frame`, `bbox` |
| `CHECK_HISTORY` | Repeat offender suspected | `person_id` |
| `VERIFY_PPE` | PPE compliance check needed | `frame`, `bbox` |
| `NO_ACTION` | False alarm, exit without confirmation | — |

**Decision Schema** (every decision is auditable):
```python
@dataclass
class Decision:
    decision_id: str
    observation_id: str          # Links back to evidence
    incident_candidate_id: str   # For tracking deduplication
    event_type: str
    person_id: int
    confidence: float
    risk_level: str              # low/medium/high/critical
    evidence_refs: List[str]     # Observation IDs
    reason_codes: List[str]      # zone_entry, temporal_confirmation, etc.
    selected_action: ActionType
    tool_arguments: Dict
    verification_result: str     # Tool output summary
    human_approval_required: bool
    final_status: DecisionOutcome
    retry_count: int
    max_retries: int = 3
```

### 5. ACT — Tool Execution

Tools are registered in a `ToolRegistry` with:
- **Bounded calls**: Max 10 per person track
- **Timeout**: 5 seconds per call
- **Retry logic**: Exponential backoff (3 retries max)

**Available Tools**:
1. **INSPECT_CROP** — Runs secondary analysis on cropped region (simulated confidence boost)
2. **CHECK_HISTORY** — Returns track history: entry count, previous decisions, risk pattern
3. **VERIFY_PPE** — Placeholder for future PPE model integration

Tool results can **modify the action** (e.g., crop inspection boosts confidence → risk escalates → incident created).

### 6. VERIFY — Result Evaluation

After tool execution:
- Success → `verification_result = "completed"`, `final_status = SUCCESS`
- Failure → `verification_result = "failed: <error>"`, `final_status = FAILED`, retry if under `max_retries`
- Confidence change → Re-assess risk level, potentially change action

### 7. ESCALATE — Human Approval

Escalation triggers:
- `risk_level == "critical"` AND `action == CREATE_INCIDENT`
- `confidence >= 0.8` (escalation threshold) AND timeout waiting for confirmation
- Explicit `human_approval_required` from decision logic

Escalated incidents get status `UNDER_REVIEW` and appear in the frontend approval queue.

### 8. RECORD — Audit Trail

Every person track maintains an `AgentTrace`:
```python
@dataclass
class AgentTrace:
    person_id: int
    observations: List[Observation]
    decisions: List[Decision]
    tool_calls: List[Dict]       # tool, arguments, result, timestamp
    created_at: float
    updated_at: float
```

This provides **full reconstructability** — auditors can trace from final incident back to raw pixels.

## Deterministic Safety Logic

All safety-critical decisions use **deterministic rules**, not ML:

- Entry/exit geometry: Point-in-polygon test (exact)
- Temporal confirmation: Wall-clock duration ≥ 3.0s (configurable)
- Deduplication: Same `person_id` + `zone` + open status = update, not create
- State machine: Explicit transitions, no hidden state

LLMs (if used) only assist with:
- Prioritizing which track to inspect next
- Summarizing evidence for human reviewers
- **Never** overriding detection or creating incidents

## Bounded Execution Guarantees

| Limit | Value | Purpose |
|-------|-------|---------|
| Max tool calls/track | 10 | Prevent runaway inspection |
| Tool timeout | 5s | Prevent hung tools |
| Max retries | 3 | Handle transient failures |
| Request duration | 3s | Bound evidence-gathering window |
| Max incident candidates | 1/track | Prevent duplicate escalation |

## Evidence Changes Behavior — Concrete Examples

| Visual Evidence | Agent Response |
|-----------------|----------------|
| Person enters zone → 2s → exits | `POTENTIAL` → timeout → `NO_ACTION` (no incident) |
| Person enters zone → 4s → confirmed | `POTENTIAL` → `REQUESTING` → `CONFIRMED` → `CREATE_INCIDENT` |
| Low confidence (0.45) detection | `LOW` risk → `CHECK_HISTORY` → if repeat → `INSPECT_CROP` → boost → incident |
| Person confirmed, high confidence (0.95) | `CRITICAL` risk → `REQUEST_HUMAN_APPROVAL` (not auto-create) |
| Person re-enters after exit | New `CANDIDATE` incident (not duplicate of resolved) |

## Integration with Incident System

When `CREATE_INCIDENT` action executes:
1. Deduplication check (same person+zone+open status)
2. Severity computed from risk level + confidence + duration
3. Local persistence: `incidents/{uuid}/{metadata.json, evidence.jpg}`
4. AWS sync (best-effort): S3 upload + CloudWatch log + metric
5. Timeline event recorded: `incident_created`

## Testing the Workflow

```python
# Test: Evidence changes action
agent = SafetyAgent(request_duration=0.1)

# Low confidence → continue tracking
event = {"event_type": "restricted_zone_entry", "person_id": 1, "confidence": 0.45, ...}
decision = agent.update([event])
assert decision[0]["next_action"] == "continue_tracking"

# After tool inspection boosts confidence → create incident
# (requires mock tool or integration test)
```

## Extending the Workflow

To add new tools:
1. Register in `ToolRegistry.register(name, func, description)`
2. Add `ActionType` enum value
3. Handle in `_select_action()` logic
4. Add verification handling in `update()`

To add new risk factors:
1. Modify `_assess_risk()` 
2. Update `_determine_reason_codes()`
3. Adjust thresholds in `_calculate_severity()`