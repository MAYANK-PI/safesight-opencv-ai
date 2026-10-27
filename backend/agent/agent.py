"""
Enhanced Agentic Vision Agent for SafeSight AI.
Implements a full perception → decision → action loop with audit trail.
"""
import time
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# Vision client (optional multimodal API)
try:
    from backend.agent.vision_client import (
        get_vision_client,
        inspect_crop_sync,
        verify_ppe_sync,
        assess_scene_sync,
        VisionAPIError,
        InspectCropResult,
        VerifyPPEResult,
        VisionClient,
    )
    VISION_CLIENT_AVAILABLE = True
except ImportError:
    VISION_CLIENT_AVAILABLE = False
    get_vision_client = None
    inspect_crop_sync = None
    verify_ppe_sync = None
    assess_scene_sync = None
    VisionAPIError = Exception
    InspectCropResult = None
    VerifyPPEResult = None
    VisionClient = None


def _crop_signature(bbox: List[int], frame_shape: Tuple[int, int]) -> str:
    """Generate a signature for a crop to detect duplicates."""
    x, y, w, h = bbox
    h_frame, w_frame = frame_shape
    # Normalize to handle slight variations
    norm_x = round(x / w_frame, 3)
    norm_y = round(y / h_frame, 3)
    norm_w = round(w / w_frame, 3)
    norm_h = round(h / h_frame, 3)
    return f"{norm_x}:{norm_y}:{norm_w}:{norm_h}"


class AgentState(str, Enum):
    OBSERVE = "observe"
    POTENTIAL = "potential"
    REQUESTING = "requesting"
    CONFIRMED = "confirmed"
    ESCALATED = "escalated"
    EXITED = "exited"


class ActionType(str, Enum):
    CONTINUE_TRACKING = "continue_tracking"
    CREATE_INCIDENT = "create_incident"
    RESOLVE_INCIDENT = "resolve_incident"
    REQUEST_HUMAN_APPROVAL = "request_human_approval"
    INSPECT_CROP = "inspect_crop"
    CHECK_HISTORY = "check_history"
    VERIFY_PPE = "verify_ppe"
    GATHER_MORE_EVIDENCE = "gather_more_evidence"
    NO_ACTION = "none"


class DecisionOutcome(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"


class VerificationStatus(str, Enum):
    """Verification status for agent decisions."""
    NOT_REQUIRED = "not_required"
    UNVERIFIED = "unverified"
    TOOL_COMPLETED = "tool_completed"  # Tool executed successfully (API call returned valid response)
    VERIFICATION_FAILED = "verification_failed"


@dataclass
class Observation:
    """Structured observation from perception."""
    observation_id: str
    timestamp: float
    event_type: str
    person_id: int
    confidence: float
    zone: str
    duration: float
    status: str  # potential, confirmed, resolved
    bbox: Optional[List[int]] = None
    frame_number: Optional[int] = None


@dataclass
class Decision:
    """Structured decision with full traceability."""
    decision_id: str
    observation_id: str
    incident_candidate_id: Optional[str]
    event_type: str
    person_id: int
    confidence: float
    risk_level: str  # low, medium, high, critical
    evidence_refs: List[str]
    reason_codes: List[str]
    selected_action: ActionType
    tool_arguments: Dict[str, Any]
    verification_result: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.NOT_REQUIRED
    human_approval_required: bool = False
    final_status: DecisionOutcome = DecisionOutcome.PENDING
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AgentTrace:
    """Complete audit trail for a person track."""
    person_id: int
    observations: List[Observation] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ToolRegistry:
    """Registry of allowed tools the agent can call."""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.max_calls_per_event = 5  # Max calls per tool per event
        self.max_iterations_per_event = 3  # Max feedback loop iterations
        self.call_timeout = 5.0  # seconds
        # Track calls per event: event_key -> {tool_name -> count}
        self._event_call_counts: Dict[str, Dict[str, int]] = {}

    def register(self, name: str, func: Callable, description: str = ""):
        self.tools[name] = {
            "func": func,
            "description": description,
        }

    def call(self, event_key: str, name: str, *args, **kwargs) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not registered")

        # Initialize event call tracking
        if event_key not in self._event_call_counts:
            self._event_call_counts[event_key] = {}

        tool_count = self._event_call_counts[event_key].get(name, 0)
        if tool_count >= self.max_calls_per_event:
            raise RuntimeError(f"Max tool calls ({self.max_calls_per_event}) exceeded for '{name}' on event {event_key}")

        self._event_call_counts[event_key][name] = tool_count + 1
        return self.tools[name]["func"](*args, **kwargs)

    def reset_event(self, event_key: str):
        """Reset call counts for a specific event."""
        if event_key in self._event_call_counts:
            del self._event_call_counts[event_key]

    def get_available_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": name, "description": info["description"]}
            for name, info in self.tools.items()
        ]


class SafetyAgent:
    """
    Enhanced agentic vision agent with full perception-decision-action loop.

    Key features:
    - Structured observations with evidence references
    - Explicit decision schema with risk levels and reason codes
    - Tool calling with bounded retries and timeouts
    - Complete audit trail per person track
    - Human approval escalation for high-risk decisions
    - Per-video API call budget enforcement
    - Duplicate crop inspection prevention
    """

    def __init__(
        self,
        request_duration: float = 3.0,
        max_tool_calls: int = 10,
        tool_timeout: float = 5.0,
        escalation_threshold: float = 0.8,
        tool_registry: Optional[ToolRegistry] = None,
        vision_max_calls_per_video: int = 10,
    ):
        self.request_duration = request_duration
        self.escalation_threshold = escalation_threshold
        self.vision_max_calls_per_video = vision_max_calls_per_video

        # Track state per person_id
        self.person_state: Dict[int, Dict[str, Any]] = {}
        self.person_traces: Dict[int, AgentTrace] = {}

        # Tool registry
        self.tool_registry = tool_registry or ToolRegistry()
        self.tool_registry.max_calls_per_track = max_tool_calls
        self.tool_registry.call_timeout = tool_timeout

        # Register default tools
        self._register_default_tools()

        # Iteration tracking per person (for feedback loop limits)
        self.person_iterations: Dict[int, int] = {}

        # Crop inspection cache to prevent duplicate API calls
        # Key: (person_id, crop_signature) -> result
        self.crop_inspection_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}

        # Vision call counter for this video session
        self.vision_call_count: int = 0

        # Statistics
        self.stats = {
            "total_observations": 0,
            "total_decisions": 0,
            "total_tool_calls": 0,
            "escalations": 0,
            "incidents_created": 0,
            "vision_calls_made": 0,
            "vision_calls_skipped_cache": 0,
            "vision_calls_skipped_budget": 0,
        }

    def reset_video_session(self):
        """Reset per-video state (call at start of new video)."""
        self.vision_call_count = 0
        self.crop_inspection_cache.clear()
        if VISION_CLIENT_AVAILABLE and get_vision_client:
            client = get_vision_client()
            if hasattr(client, 'reset_video_counter'):
                client.reset_video_counter()

    def _register_default_tools(self):
        """Register default inspection tools."""
        # Tool: Inspect crop using vision API
        self.tool_registry.register(
            "inspect_crop",
            self._tool_inspect_crop,
            "Inspect a cropped region of interest using multimodal vision API"
        )
        # Tool: Check tracking history for patterns
        self.tool_registry.register(
            "check_history",
            self._tool_check_history,
            "Check tracking history for a person ID"
        )
        # Tool: Verify PPE compliance using vision API
        self.tool_registry.register(
            "verify_ppe",
            self._tool_verify_ppe,
            "Verify PPE compliance in a region using multimodal vision API"
        )
        # Tool: Gather more evidence by re-analyzing frames around event
        self.tool_registry.register(
            "gather_more_evidence",
            self._tool_gather_more_evidence,
            "Re-analyze video frames around event timestamp for additional evidence"
        )

    def _tool_inspect_crop(self, frame: Any, bbox: List[int], person_id: int, event_id: str = "") -> Dict[str, Any]:
        """Tool: Inspect a cropped region using multimodal vision API.

        Enforces per-video budget and prevents duplicate inspections of identical crops.
        """
        # Check per-video budget
        if self.vision_call_count >= self.vision_max_calls_per_video:
            self.stats["vision_calls_skipped_budget"] += 1
            return {
                "tool": "inspect_crop",
                "person_id": person_id,
                "bbox": bbox,
                "result": "budget_exhausted",
                "confidence_boost": 0.0,
                "risk_level_change": "same",
                "details": f"Vision API call budget exhausted ({self.vision_max_calls_per_video} calls per video)",
                "violations_detected": [],
                "inspection_error": {"error_type": "budget_exhausted", "retryable": False},
            }

        if not VISION_CLIENT_AVAILABLE or not get_vision_client or not get_vision_client().is_available():
            return {
                "tool": "inspect_crop",
                "person_id": person_id,
                "bbox": bbox,
                "result": "vision_api_unavailable",
                "confidence_boost": 0.0,
                "risk_level_change": "same",
                "details": "Multimodal vision API not configured",
                "violations_detected": [],
                "inspection_error": {"error_type": "not_configured", "retryable": False},
            }

        # Check cache for duplicate crop
        if frame is not None and hasattr(frame, 'shape'):
            crop_sig = _crop_signature(bbox, frame.shape[:2])
            cache_key = (person_id, crop_sig)
            if cache_key in self.crop_inspection_cache:
                self.stats["vision_calls_skipped_cache"] += 1
                cached_result = self.crop_inspection_cache[cache_key].copy()
                cached_result["cached"] = True
                logger.debug(f"Returning cached inspection for person {person_id}, crop {crop_sig}")
                return cached_result

        try:
            # Crop the frame to the bounding box
            if frame is not None and hasattr(frame, 'shape'):
                x, y, w, h = bbox
                h_frame, w_frame = frame.shape[:2]
                x = max(0, min(x, w_frame - 1))
                y = max(0, min(y, h_frame - 1))
                w = max(1, min(w, w_frame - x))
                h = max(1, min(h, h_frame - y))
                crop = frame[y:y+h, x:x+w]
                if crop.size == 0:
                    raise ValueError("Empty crop")
                # Encode crop to bytes
                import cv2
                _, buffer = cv2.imencode('.jpg', crop)
                crop_bytes = buffer.tobytes()
            else:
                raise ValueError("Invalid frame")

            # Call vision API synchronously
            result: InspectCropResult = inspect_crop_sync(crop_bytes, bbox, person_id, event_id)
            result_dict = result.model_dump()

            # Track vision call
            self.vision_call_count += 1
            self.stats["vision_calls_made"] += 1

            # Cache the result
            if frame is not None and hasattr(frame, 'shape'):
                crop_sig = _crop_signature(bbox, frame.shape[:2])
                cache_key = (person_id, crop_sig)
                self.crop_inspection_cache[cache_key] = result_dict.copy()

            return result_dict

        except Exception as e:
            logger.warning(f"inspect_crop tool failed: {e}")
            return {
                "tool": "inspect_crop",
                "person_id": person_id,
                "bbox": bbox,
                "result": "failed",
                "confidence_boost": 0.0,
                "risk_level_change": "same",
                "details": f"Tool error: {str(e)}",
                "violations_detected": [],
                "inspection_error": {"error_type": "tool_error", "message": str(e), "retryable": False},
            }

    def _tool_check_history(self, person_id: int) -> Dict[str, Any]:
        """Tool: Check tracking history for patterns."""
        trace = self.person_traces.get(person_id)
        if not trace:
            return {"tool": "check_history", "person_id": person_id, "history": "no_history"}

        entry_count = sum(1 for o in trace.observations if o.event_type == "restricted_zone_entry")
        return {
            "tool": "check_history",
            "person_id": person_id,
            "total_observations": len(trace.observations),
            "entry_count": entry_count,
            "previous_decisions": [d.selected_action.value for d in trace.decisions],
            "risk_pattern": "repeated" if entry_count > 2 else "first"
        }

    def _tool_verify_ppe(self, frame: Any, bbox: List[int], event_id: str = "") -> Dict[str, Any]:
        """Tool: Verify PPE compliance using multimodal vision API."""
        # Check per-video budget
        if self.vision_call_count >= self.vision_max_calls_per_video:
            self.stats["vision_calls_skipped_budget"] += 1
            return {
                "tool": "verify_ppe",
                "bbox": bbox,
                "ppe_detected": {"hard_hat": None, "vest": None, "gloves": None, "boots": None},
                "compliance": "uncertain",
                "confidence": 0.0,
                "details": f"Vision API call budget exhausted ({self.vision_max_calls_per_video} calls per video)",
                "missing_items": ["hard_hat", "safety_vest", "gloves", "boots"],
                "inspection_error": {"error_type": "budget_exhausted", "retryable": False},
            }

        if not VISION_CLIENT_AVAILABLE or not get_vision_client or not get_vision_client().is_available():
            return {
                "tool": "verify_ppe",
                "bbox": bbox,
                "ppe_detected": {"hard_hat": None, "vest": None, "gloves": None, "boots": None},
                "compliance": "uncertain",
                "confidence": 0.0,
                "details": "Multimodal vision API not configured",
                "missing_items": ["hard_hat", "safety_vest", "gloves", "boots"],
                "inspection_error": {"error_type": "not_configured", "retryable": False},
            }

        try:
            # Crop the frame to the bounding box
            if frame is not None and hasattr(frame, 'shape'):
                x, y, w, h = bbox
                h_frame, w_frame = frame.shape[:2]
                x = max(0, min(x, w_frame - 1))
                y = max(0, min(y, h_frame - 1))
                w = max(1, min(w, w_frame - x))
                h = max(1, min(h, h_frame - y))
                crop = frame[y:y+h, x:x+w]
                if crop.size == 0:
                    raise ValueError("Empty crop")
                import cv2
                _, buffer = cv2.imencode('.jpg', crop)
                crop_bytes = buffer.tobytes()
            else:
                raise ValueError("Invalid frame")

            # Call vision API synchronously
            result: VerifyPPEResult = verify_ppe_sync(crop_bytes, bbox, event_id)
            result_dict = result.model_dump()

            # Track vision call
            self.vision_call_count += 1
            self.stats["vision_calls_made"] += 1

            return result_dict

        except Exception as e:
            logger.warning(f"verify_ppe tool failed: {e}")
            return {
                "tool": "verify_ppe",
                "bbox": bbox,
                "ppe_detected": {"hard_hat": None, "vest": None, "gloves": None, "boots": None},
                "compliance": "uncertain",
                "confidence": 0.0,
                "details": f"Tool error: {str(e)}",
                "missing_items": ["hard_hat", "safety_vest", "gloves", "boots"],
                "inspection_error": {"error_type": "tool_error", "message": str(e), "retryable": False},
            }

    def _tool_gather_more_evidence(self, frame: Any, bbox: List[int], person_id: int,
                                    frame_number: int, event_context: Dict[str, Any],
                                    event_id: str = "") -> Dict[str, Any]:
        """Tool: Gather more evidence by assessing the broader scene context."""
        # Check per-video budget
        if self.vision_call_count >= self.vision_max_calls_per_video:
            self.stats["vision_calls_skipped_budget"] += 1
            return {
                "tool": "gather_more_evidence",
                "person_id": person_id,
                "result": "budget_exhausted",
                "risk_level": "low",
                "confidence": 0.0,
                "reason_codes": ["vision_budget_exhausted"],
                "recommended_action": "continue_tracking",
                "details": f"Vision API call budget exhausted ({self.vision_max_calls_per_video} calls per video)",
                "inspection_error": {"error_type": "budget_exhausted", "retryable": False},
            }

        if not VISION_CLIENT_AVAILABLE or not get_vision_client or not get_vision_client().is_available():
            return {
                "tool": "gather_more_evidence",
                "person_id": person_id,
                "result": "vision_api_unavailable",
                "risk_level": "low",
                "confidence": 0.0,
                "reason_codes": ["vision_api_unavailable"],
                "recommended_action": "continue_tracking",
                "details": "Multimodal vision API not configured",
                "inspection_error": {"error_type": "not_configured", "retryable": False},
            }

        try:
            # Use the full frame for scene assessment
            if frame is not None and hasattr(frame, 'shape'):
                import cv2
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
            else:
                raise ValueError("Invalid frame")

            # Call vision API for scene assessment
            result: AssessSceneResult = assess_scene_sync(frame_bytes, event_id, event_context)
            result_dict = result.model_dump()

            # Track vision call
            self.vision_call_count += 1
            self.stats["vision_calls_made"] += 1

            return result_dict

        except Exception as e:
            logger.warning(f"gather_more_evidence tool failed: {e}")
            return {
                "tool": "gather_more_evidence",
                "person_id": person_id,
                "result": "failed",
                "risk_level": "low",
                "confidence": 0.0,
                "reason_codes": ["tool_error"],
                "recommended_action": "continue_tracking",
                "details": f"Tool error: {str(e)}",
                "inspection_error": {"error_type": "tool_error", "message": str(e), "retryable": False},
            }

    def _get_person_state(self, person_id: int) -> Dict[str, Any]:
        """Get or initialize state for a person."""
        if person_id not in self.person_state:
            self.person_state[person_id] = {
                'state': AgentState.OBSERVE,
                'potential_start': None,
                'last_request': None,
                'confirmed': False,
                'incident_candidate_id': None,
            }
            self.person_traces[person_id] = AgentTrace(person_id=person_id)
        return self.person_state[person_id]

    def _create_observation(self, event: Dict[str, Any], frame_number: Optional[int] = None) -> Observation:
        """Create structured observation from event."""
        return Observation(
            observation_id=str(uuid.uuid4())[:8],
            timestamp=event.get("timestamp", time.time()),
            event_type=event.get("event_type", "unknown"),
            person_id=event.get("person_id", -1),
            confidence=event.get("confidence", 0.0),
            zone=event.get("zone", "unknown"),
            duration=event.get("duration", 0.0),
            status=event.get("status", "unknown"),
            bbox=event.get("bbox"),
            frame_number=frame_number
        )

    def _assess_risk(self, event: Dict[str, Any], pstate: Dict[str, Any]) -> str:
        """Assess risk level based on evidence."""
        confidence = event.get("confidence", 0.0)
        status = event.get("status", "unknown")
        duration = event.get("duration", 0.0)

        if status == "confirmed" and confidence >= 0.8:
            return "critical"
        elif status == "confirmed" and confidence >= 0.6:
            return "high"
        elif status == "potential" and duration > self.request_duration:
            return "medium"
        elif status == "potential":
            return "low"
        return "low"

    def _determine_reason_codes(self, event: Dict[str, Any], pstate: Dict[str, Any]) -> List[str]:
        """Determine reason codes for the decision."""
        codes = []
        event_type = event.get("event_type", "")
        status = event.get("status", "")

        if event_type == "restricted_zone_entry":
            codes.append("zone_entry")
        if event_type == "restricted_zone_confirmation":
            codes.append("temporal_confirmation")
        if status == "confirmed":
            codes.append("confirmed_violation")
        if pstate.get("confirmed"):
            codes.append("repeat_offender")
        if event.get("confidence", 0) < 0.6:
            codes.append("low_confidence")

        return codes or ["unknown"]

    def _select_action(
        self,
        event: Dict[str, Any],
        pstate: Dict[str, Any],
        risk_level: str,
        reason_codes: List[str],
        frame: Optional[Any] = None
    ) -> tuple[ActionType, Dict[str, Any], bool]:
        """Select the next action based on evidence and state."""
        event_type = event.get("event_type", "")
        status = event.get("status", "")
        confidence = event.get("confidence", 0.0)
        current_time = time.time()
        person_id = event.get("person_id", -1)

        # If PPE issues detected from previous inspect_crop, verify PPE first
        ppe_status = event.get("ppe_status")
        if ppe_status and isinstance(ppe_status, dict):
            missing_ppe = [item for item, detected in ppe_status.items() if detected is False]
            if missing_ppe and frame is not None:
                return ActionType.VERIFY_PPE, {"duration_seconds": self.request_duration}, False

        # Confirmation -> first verify with vision if not already verified, then create incident
        # Per spec: call INSPECT_CROP for confirmed events when frame is available (regardless of confidence)
        if event_type == "restricted_zone_confirmation" and status == "confirmed":
            print(f"DEBUG: Confirmation event for person {person_id}, pstate={pstate['state']}, frame={frame is not None}, vision_verified={event.get('ppe_status') is not None or event.get('violations_detected') is not None}")
            if pstate['state'] in [AgentState.POTENTIAL, AgentState.REQUESTING]:
                pstate['state'] = AgentState.CONFIRMED
                pstate['confirmed'] = True

                # Check if vision verification has already been done for this event
                # Vision verification is indicated by presence of ppe_status or violations_detected from vision tools
                vision_verified = event.get("ppe_status") is not None or event.get("violations_detected") is not None

                # If frame is available and we haven't done vision verification yet, inspect the crop first
                if frame is not None and not vision_verified:
                    print(f"DEBUG: Calling INSPECT_CROP for person {person_id}")
                    pstate['state'] = AgentState.REQUESTING
                    return ActionType.INSPECT_CROP, {"duration_seconds": self.request_duration}, False
                else:
                    print(f"DEBUG: Skipping INSPECT_CROP for person {person_id} (frame={frame is not None}, vision_verified={vision_verified})")

                # Vision already done (or no frame available) - check for violations and escalate appropriately
                if vision_verified:
                    # Vision was performed - check for violations
                    violations = event.get("violations_detected", [])
                    if violations:
                        # Supported violations detected - create incident, require human approval for high/critical risk
                        return ActionType.CREATE_INCIDENT, {}, risk_level in ["high", "critical"]
                    else:
                        # Vision completed but no violations found - no incident needed
                        return ActionType.NO_ACTION, {}, False
                else:
                    # No frame available for vision - fall back to risk-based incident creation
                    return ActionType.CREATE_INCIDENT, {}, risk_level == "critical"

        # High confidence confirmed violation -> create incident (for non-confirmation events or already vision-verified)
        if status == "confirmed" and risk_level in ["high", "critical"]:
            if not pstate.get("confirmed"):
                return ActionType.CREATE_INCIDENT, {}, risk_level == "critical"

        # Potential entry -> request more evidence (continue tracking)
        if event_type == "restricted_zone_entry" and status == "potential":
            if pstate['state'] == AgentState.OBSERVE:
                pstate['state'] = AgentState.POTENTIAL
                pstate['potential_start'] = current_time
                pstate['last_request'] = current_time
                pstate['incident_candidate_id'] = str(uuid.uuid4())[:8]
                # For low confidence potential events, first inspect crop to gather evidence
                if confidence < 0.7 and frame is not None:
                    pstate['state'] = AgentState.REQUESTING
                    return ActionType.INSPECT_CROP, {"duration_seconds": self.request_duration}, False
                return ActionType.CONTINUE_TRACKING, {"duration_seconds": self.request_duration}, False

            elif pstate['state'] in [AgentState.POTENTIAL, AgentState.REQUESTING]:
                if pstate['last_request']:
                    elapsed = current_time - pstate['last_request']
                    if elapsed < self.request_duration:
                        # If we haven't inspected yet and confidence is low, do it now
                        if confidence < 0.7 and pstate['state'] == AgentState.POTENTIAL and frame is not None:
                            pstate['state'] = AgentState.REQUESTING
                            return ActionType.INSPECT_CROP, {"duration_seconds": self.request_duration}, False
                        return ActionType.CONTINUE_TRACKING, {"duration_seconds": max(0, self.request_duration - elapsed)}, False
                    else:
                        # Timeout - escalate if confidence is high
                        if confidence >= self.escalation_threshold:
                            return ActionType.REQUEST_HUMAN_APPROVAL, {"candidate_id": pstate['incident_candidate_id']}, True
                        pstate['state'] = AgentState.OBSERVE
                        pstate['potential_start'] = None
                        pstate['incident_candidate_id'] = None
                        return ActionType.NO_ACTION, {}, False

        # Exit -> resolve
        if event_type == "restricted_zone_exit" and status == "resolved":
            if pstate['state'] == AgentState.CONFIRMED:
                pstate['state'] = AgentState.EXITED
                return ActionType.RESOLVE_INCIDENT, {}, False
            else:
                pstate['state'] = AgentState.OBSERVE
                pstate['potential_start'] = None
                pstate['last_request'] = None
                pstate['confirmed'] = False
                pstate['incident_candidate_id'] = None
                return ActionType.NO_ACTION, {}, False

        return ActionType.NO_ACTION, {}, False

    def update(
        self,
        events: List[Dict[str, Any]],
        frame: Optional[Any] = None,
        frame_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process events through the full perception-decision-action loop with feedback.

        Args:
            events: List of event dictionaries from the analyzer
            frame: Optional current frame for tool calls
            frame_number: Optional frame number for tracking

        Returns:
            List of decision dictionaries with full traceability
        """
        # Reset video session on first frame
        if frame_number is not None and frame_number <= 1:
            self.reset_video_session()

        decisions_output = []
        current_time = time.time()

        for event in events:
            person_id = event.get('person_id')
            if person_id is None:
                continue

            print(f"DEBUG UPDATE: Processing event: type={event.get('event_type')}, person={person_id}, status={event.get('status')}, confidence={event.get('confidence')}")

            pstate = self._get_person_state(person_id)
            trace = self.person_traces[person_id]

            # PERCEIVE: Create structured observation
            observation = self._create_observation(event, frame_number)
            trace.observations.append(observation)
            self.stats["total_observations"] += 1

            # Initialize iteration tracking for this event
            event_key = f"{person_id}:{observation.observation_id}"
            self.person_iterations[event_key] = 0
            max_iterations = self.tool_registry.max_iterations_per_event

            # Current event data (mutable for feedback loop)
            current_event = event.copy()
            current_confidence = event.get("confidence", 0.0)
            current_risk_level = "low"
            current_reason_codes = []
            current_evidence_refs = [observation.observation_id]

            # FEEDBACK LOOP: perception -> decision -> action -> reassess
            while self.person_iterations[event_key] < max_iterations:
                self.person_iterations[event_key] += 1
                iteration = self.person_iterations[event_key]

                # INTERPRET & ASSESS
                current_risk_level = self._assess_risk(current_event, pstate)
                current_reason_codes = self._determine_reason_codes(current_event, pstate)

                # PLAN: Select action based on evidence
                action_type, tool_args, requires_approval = self._select_action(
                    current_event, pstate, current_risk_level, current_reason_codes, frame
                )

                # ACT: Execute tool if needed
                tool_result = None
                verification_result = None
                tool_called = False

                # Tools that can be called in the feedback loop
                actionable_tools = [
                    ActionType.INSPECT_CROP,
                    ActionType.CHECK_HISTORY,
                    ActionType.VERIFY_PPE,
                    ActionType.GATHER_MORE_EVIDENCE,
                ]

                if action_type in actionable_tools:
                    try:
                        # Pass additional context for tools that need it
                        if action_type == ActionType.INSPECT_CROP:
                            tool_result = self.tool_registry.call(
                                event_key, action_type.value, frame, event.get("bbox"), person_id,
                                observation.observation_id
                            )
                            print(f"DEBUG INSPECT_CROP result: {tool_result}")
                        elif action_type == ActionType.VERIFY_PPE:
                            tool_result = self.tool_registry.call(
                                event_key, action_type.value, frame, event.get("bbox"),
                                observation.observation_id
                            )
                        elif action_type == ActionType.GATHER_MORE_EVIDENCE:
                            # Build event context for scene assessment
                            event_context = {
                                "event_type": current_event.get("event_type"),
                                "person_id": person_id,
                                "zone": current_event.get("zone"),
                                "duration": current_event.get("duration"),
                                "status": current_event.get("status"),
                                "confidence": current_confidence,
                                "risk_level": current_risk_level,
                                "frame_number": frame_number,
                            }
                            tool_result = self.tool_registry.call(
                                event_key, action_type.value, frame, event.get("bbox"), person_id,
                                frame_number, event_context, observation.observation_id
                            )
                        else:
                            tool_result = self.tool_registry.call(
                                event_key, action_type.value, person_id
                            )

                        trace.tool_calls.append({
                            "tool": action_type.value,
                            "arguments": tool_args,
                            "result": tool_result,
                            "timestamp": time.time(),
                            "iteration": iteration,
                        })
                        self.stats["total_tool_calls"] += 1
                        tool_called = True
                        verification_result = tool_result.get("result", "completed")

                        # FEEDBACK: Tool result may change confidence, risk, or action
                        # Update event data based on tool result for next iteration
                        if action_type == ActionType.INSPECT_CROP:
                            boost = tool_result.get("confidence_boost", 0.0)
                            if boost != 0:
                                current_confidence = min(1.0, max(0.0, current_confidence + boost))
                                current_event = {**current_event, "confidence": current_confidence}
                                current_reason_codes.append("crop_inspection")

                            # Check if violations were detected
                            violations = tool_result.get("violations_detected", [])
                            if violations:
                                current_reason_codes.extend([f"violation:{v}" for v in violations])
                                current_event = {**current_event, "status": "confirmed", "violations_detected": violations}

                            # PPE status from crop inspection
                            ppe_status = tool_result.get("ppe_status")
                            if ppe_status is not None:
                                current_event["ppe_status"] = ppe_status

                            # Reconcile ppe_status with violations_detected for consistency
                            self._reconcile_ppe_status(current_event, tool_result)

                        elif action_type == ActionType.VERIFY_PPE:
                            compliance = tool_result.get("compliance", "uncertain")
                            ppe_confidence = tool_result.get("confidence", 0.0)
                            missing = tool_result.get("missing_items", [])

                            if compliance == "non_compliant" and ppe_confidence > 0.6:
                                current_confidence = min(1.0, current_confidence + 0.15)
                                current_reason_codes.append("ppe_non_compliant")
                                current_event = {**current_event, "confidence": current_confidence, "status": "confirmed"}
                            elif compliance == "compliant" and ppe_confidence > 0.7:
                                current_confidence = max(0.0, current_confidence - 0.1)
                                current_reason_codes.append("ppe_compliant")
                                current_event = {**current_event, "confidence": current_confidence}

                            # Reconcile ppe_detected with missing_items for consistency
                            self._reconcile_ppe_status(current_event, tool_result)

                        elif action_type == ActionType.GATHER_MORE_EVIDENCE:
                            # Scene assessment can recommend action directly
                            rec_action = tool_result.get("recommended_action")
                            scene_risk = tool_result.get("risk_level", "low")
                            scene_confidence = tool_result.get("confidence", 0.0)
                            scene_reasons = tool_result.get("reason_codes", [])

                            if scene_confidence > 0.6:
                                # Weight the scene assessment into our decision
                                current_risk_level = scene_risk
                                current_reason_codes.extend(scene_reasons)
                                # If scene assessment strongly recommends an action, consider it
                                if rec_action in ["create_incident", "request_human_approval"] and scene_confidence > 0.7:
                                    # Let the next iteration pick this up via _select_action
                                    current_event = {**current_event, "status": "confirmed" if rec_action == "create_incident" else "potential"}

                        elif action_type == ActionType.CHECK_HISTORY:
                            risk_pattern = tool_result.get("risk_pattern", "first")
                            if risk_pattern == "repeated":
                                current_reason_codes.append("repeat_offender")
                                current_event = {**current_event, "status": "confirmed"}

                    except Exception as e:
                        logger.warning(f"Tool {action_type.value} failed on iteration {iteration}: {e}")
                        verification_result = f"failed: {str(e)}"

                # If no tool was called, or tool didn't change anything significant, break
                if not tool_called:
                    break

                # Check if we should continue iterating
                # Continue if: tool boosted confidence significantly, or changed status, or found violations
                significant_change = (
                    tool_result.get("confidence_boost", 0) > 0.05 or
                    tool_result.get("violations_detected") or
                    tool_result.get("compliance") == "non_compliant" or
                    tool_result.get("risk_pattern") == "repeated" or
                    tool_result.get("recommended_action") in ["create_incident", "request_human_approval"]
                )

                if not significant_change:
                    break

            # Track if original action was CREATE_INCIDENT (before potential escalation)
            original_action_was_create_incident = (action_type == ActionType.CREATE_INCIDENT)

            # ESCALATE: Human approval for high-risk
            if requires_approval or (current_risk_level == "critical" and action_type == ActionType.CREATE_INCIDENT):
                action_type = ActionType.REQUEST_HUMAN_APPROVAL
                tool_args = {"candidate_id": pstate.get('incident_candidate_id', str(uuid.uuid4())[:8])}
                self.stats["escalations"] += 1

            # Determine verification status
            verification_status = self._determine_verification_status(
                verification_result, action_type, current_event
            )

            # Create incident if the original action was CREATE_INCIDENT (even if escalated to human approval)
            incident_id = None
            if original_action_was_create_incident:
                incident_id = f"inc_{uuid.uuid4().hex[:12]}"
                self.stats["incidents_created"] += 1
                # Store incident ID in person state for reference
                pstate['last_incident_id'] = incident_id

            # CREATE DECISION
            decision = Decision(
                decision_id=str(uuid.uuid4())[:8],
                observation_id=observation.observation_id,
                incident_candidate_id=pstate.get('incident_candidate_id'),
                event_type=event.get("event_type", "unknown"),
                person_id=person_id,
                confidence=current_confidence,
                risk_level=current_risk_level,
                evidence_refs=current_evidence_refs,
                reason_codes=current_reason_codes,
                selected_action=action_type,
                tool_arguments=tool_args,
                verification_result=verification_result,
                verification_status=verification_status,
                human_approval_required=requires_approval,
                final_status=DecisionOutcome.SUCCESS if verification_status == VerificationStatus.TOOL_COMPLETED else DecisionOutcome.FAILED,
                retry_count=self.person_iterations[event_key] - 1,
                max_retries=max_iterations,
            )

            trace.decisions.append(decision)
            trace.updated_at = current_time
            self.stats["total_decisions"] += 1

            # Clean up iteration tracking
            del self.person_iterations[event_key]
            self.tool_registry.reset_event(event_key)

            # Build output decision
            decisions_output.append({
                "decision_id": decision.decision_id,
                "decision": decision.selected_action.value,
                "reason": f"Risk: {current_risk_level}. Codes: {', '.join(current_reason_codes)}",
                "person_id": person_id,
                "next_action": action_type.value,
                "parameters": tool_args,
                "risk_level": current_risk_level,
                "reason_codes": current_reason_codes,
                "evidence_refs": current_evidence_refs,
                "human_approval_required": decision.human_approval_required,
                "verification_result": verification_result,
                "verification_status": decision.verification_status.value,
                "iterations": decision.retry_count + 1,
                "incident_id": incident_id,
                # Structured assessment fields
                "event_id": observation.observation_id,
                "event_type": event.get("event_type", "unknown"),
                "severity": self._map_risk_to_severity(current_risk_level, current_confidence),
                "confidence": round(current_confidence, 3),
                "evidence": {
                    "timestamps": [event.get("timestamp", time.time())],
                    "crop_refs": current_evidence_refs,
                    "supporting_observations": current_reason_codes,
                },
                "rationale": self._build_rationale(current_event, current_reason_codes, verification_status),
                "recommended_action": self._map_action_to_recommendation(action_type, verification_status),
                "human_review_required": decision.human_approval_required or verification_status == VerificationStatus.UNVERIFIED,
                "model_used": self._get_model_used_from_tools(trace.tool_calls),
                "inspection_error": self._get_inspection_error_from_tools(trace.tool_calls),
            })

        return decisions_output

    def _determine_verification_status(
        self,
        verification_result: Optional[str],
        action_type: ActionType,
        current_event: Dict[str, Any]
    ) -> VerificationStatus:
        """Determine verification status based on tool results and action."""
        if verification_result is None:
            # No tool was called - verification not required for this action
            if action_type in [ActionType.CONTINUE_TRACKING, ActionType.NO_ACTION]:
                return VerificationStatus.NOT_REQUIRED
            return VerificationStatus.UNVERIFIED

        # Check for specific error conditions
        if "vision_api_unavailable" in str(verification_result):
            return VerificationStatus.UNVERIFIED

        if "budget_exhausted" in str(verification_result):
            return VerificationStatus.UNVERIFIED

        if "failed" in str(verification_result).lower():
            return VerificationStatus.VERIFICATION_FAILED

        # Tool succeeded - API call completed and returned valid response
        # This does NOT mean safety violations were confirmed, only that the tool executed
        if action_type in [ActionType.INSPECT_CROP, ActionType.VERIFY_PPE, ActionType.GATHER_MORE_EVIDENCE]:
            return VerificationStatus.TOOL_COMPLETED

        return VerificationStatus.UNVERIFIED

    def _reconcile_ppe_status(self, current_event: Dict[str, Any], tool_result: Dict[str, Any]) -> None:
        """
        Reconcile ppe_status/ppe_detected with violations_detected/missing_items for consistency.

        If violations_detected or missing_items indicate a PPE item is missing, but ppe_status/ppe_detected
        has null for that item, update it to false. This fixes model output inconsistencies.
        """
        # Map violation names to PPE keys
        violation_to_ppe = {
            "missing_hard_hat": "hard_hat",
            "missing_safety_vest": "safety_vest",
            "missing_vest": "safety_vest",
            "missing_gloves": "gloves",
            "missing_boots": "boots",
            "missing_ppe": None,  # generic - handled via details text
        }

        # Get PPE status from event (set by INSPECT_CROP)
        ppe_status = current_event.get("ppe_status")
        if ppe_status is None:
            return

        violations = tool_result.get("violations_detected", [])
        missing_items = tool_result.get("missing_items", [])
        details = tool_result.get("details", "").lower()

        # Check violations_detected for specific missing items
        for violation in violations:
            ppe_key = violation_to_ppe.get(violation)
            if ppe_key and ppe_key in ppe_status and ppe_status[ppe_key] is None:
                # Model said item is missing but ppe_status is null - fix to false
                ppe_status[ppe_key] = False

        # Check missing_items from VERIFY_PPE
        for item in missing_items:
            if item in ppe_status and ppe_status[item] is None:
                ppe_status[item] = False

        # Check details text for explicit mentions of missing items
        detail_mappings = {
            "hard hat": "hard_hat",
            "safety vest": "safety_vest",
            "vest": "safety_vest",
            "gloves": "gloves",
            "boots": "boots",
        }
        for text, ppe_key in detail_mappings.items():
            # Look for phrases like "not wearing", "no ... visible", "missing", "absent"
            if (f"not wearing {text}" in details or
                f"no {text} visible" in details or
                f"no {text} are visible" in details or
                f"{text} is missing" in details or
                f"{text} are missing" in details or
                f"missing {text}" in details):
                if ppe_key in ppe_status and ppe_status[ppe_key] is None:
                    ppe_status[ppe_key] = False

    def _map_risk_to_severity(self, risk_level: str, confidence: float) -> str:
        """Map risk level and confidence to severity."""
        if risk_level == "critical":
            return "critical"
        elif risk_level == "high":
            return "high"
        elif risk_level == "medium":
            return "medium"
        return "low"

    def _build_rationale(self, event: Dict[str, Any], reason_codes: List[str], verification_status: VerificationStatus) -> str:
        """Build a concise, evidence-based rationale."""
        event_type = event.get("event_type", "unknown")
        status = event.get("status", "unknown")
        confidence = event.get("confidence", 0.0)

        parts = []
        if event_type == "restricted_zone_entry":
            parts.append("Person detected entering restricted zone")
        elif event_type == "restricted_zone_confirmation":
            parts.append("Person confirmed in restricted zone")
        elif event_type == "restricted_zone_exit":
            parts.append("Person exited restricted zone")
        else:
            parts.append(f"Event: {event_type}")

        if "confirmed_violation" in reason_codes:
            parts.append("violation confirmed")
        if "crop_inspection" in reason_codes:
            parts.append("validated by vision inspection")
        if "ppe_non_compliant" in reason_codes:
            parts.append("PPE non-compliance detected")
        if "repeat_offender" in reason_codes:
            parts.append("repeat entry pattern")

        if verification_status == VerificationStatus.TOOL_COMPLETED:
            parts.append("AI tool executed")
        elif verification_status == VerificationStatus.UNVERIFIED:
            parts.append("not AI-verified")
        elif verification_status == VerificationStatus.VERIFICATION_FAILED:
            parts.append("verification failed")

        rationale = ". ".join(parts) + f" (confidence: {confidence:.2f})"
        return rationale

    def _map_action_to_recommendation(self, action_type: ActionType, verification_status: VerificationStatus) -> str:
        """Map agent action to recommended action string."""
        if action_type == ActionType.CREATE_INCIDENT:
            return "create_incident"
        elif action_type == ActionType.REQUEST_HUMAN_APPROVAL:
            return "request_human_review"
        elif action_type == ActionType.RESOLVE_INCIDENT:
            return "resolve_incident"
        elif action_type == ActionType.VERIFY_PPE:
            return "verify_ppe"
        elif action_type == ActionType.INSPECT_CROP:
            return "inspect_evidence"
        elif action_type == ActionType.GATHER_MORE_EVIDENCE:
            return "gather_evidence"
        elif action_type == ActionType.CONTINUE_TRACKING:
            return "continue_monitoring"
        return "no_action"

    def _get_model_used_from_tools(self, tool_calls: List[Dict[str, Any]]) -> Optional[str]:
        """Extract model used from tool call results."""
        for tc in reversed(tool_calls):
            result = tc.get("result", {})
            if isinstance(result, dict) and result.get("model_used"):
                return result["model_used"]
        return None

    def _get_inspection_error_from_tools(self, tool_calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract inspection error from tool call results."""
        for tc in reversed(tool_calls):
            result = tc.get("result", {})
            if isinstance(result, dict) and result.get("inspection_error"):
                return result["inspection_error"]
        return None

    def get_trace(self, person_id: int) -> Optional[AgentTrace]:
        """Get complete audit trail for a person."""
        return self.person_traces.get(person_id)

    def get_all_traces(self) -> Dict[int, AgentTrace]:
        """Get all audit trails."""
        return self.person_traces

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return self.stats.copy()

    def reset_person(self, person_id: int):
        """Reset state for a specific person."""
        if person_id in self.person_state:
            del self.person_state[person_id]
        if person_id in self.person_traces:
            del self.person_traces[person_id]

    def reset_all(self):
        """Reset all agent state."""
        self.person_state.clear()
        self.person_traces.clear()
        self.stats = {
            "total_observations": 0,
            "total_decisions": 0,
            "total_tool_calls": 0,
            "escalations": 0,
            "incidents_created": 0
        }


def create_agent() -> SafetyAgent:
    """Factory function for creating agent instances."""
    import os
    vision_max_calls = int(os.getenv("VISION_MAX_CALLS_PER_VIDEO", "10"))
    return SafetyAgent(request_duration=3.0, vision_max_calls_per_video=vision_max_calls)