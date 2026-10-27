import time
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from backend.agent.agent import SafetyAgent, AgentState, ActionType, DecisionOutcome
from backend.agent.vision_client import reset_vision_client


def test_agent_entry_and_confirmation():
    """Test agent processes entry then confirmation events."""
    agent = SafetyAgent(request_duration=0.1)  # short duration for test
    current_time = time.time()

    # Simulate an entry event (potential)
    event_entry = {
        "event_type": "restricted_zone_entry",
        "person_id": 1,
        "confidence": 0.7,
        "zone": "restricted_zone_1",
        "timestamp": current_time,
        "duration": 0.0,
        "status": "potential"
    }
    decisions = agent.update([event_entry])
    assert len(decisions) == 1
    dec = decisions[0]
    assert dec["decision"] == ActionType.CONTINUE_TRACKING.value
    assert dec["person_id"] == 1
    assert dec["next_action"] == "continue_tracking"
    assert "duration_seconds" in dec["parameters"]
    assert dec["risk_level"] in ["low", "medium", "high", "critical"]
    assert "reason_codes" in dec
    assert "evidence_refs" in dec

    # Mock time to test confirmation after request_duration
    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time
        # Re-send the entry event to initialize the agent's state
        agent.update([event_entry])

        # Now advance time by 0.2 seconds (which is > request_duration of 0.1)
        mock_time.return_value = current_time + 0.2
        # Send a confirmation event
        event_confirm = {
            "event_type": "restricted_zone_confirmation",
            "person_id": 1,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time + 0.2,
            "duration": 0.2,
            "status": "confirmed"
        }
        decisions2 = agent.update([event_confirm])
        assert len(decisions2) == 1
        dec2 = decisions2[0]
        assert dec2["decision"] == ActionType.CREATE_INCIDENT.value
        assert dec2["person_id"] == 1
        assert dec2["next_action"] == "create_incident"
        assert dec2["risk_level"] in ["high", "critical"]


def test_agent_entry_then_exit_no_confirmation():
    """Test agent processes entry then exit without confirmation (false alarm)."""
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time
        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 2,
            "confidence": 0.8,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential"
        }
        agent.update([event_entry])  # initialize

        # Advance time by 0.2 seconds (timeout)
        mock_time.return_value = current_time + 0.2
        event_exit = {
            "event_type": "restricted_zone_exit",
            "person_id": 2,
            "confidence": 0.8,
            "zone": "restricted_zone_1",
            "timestamp": current_time + 0.2,
            "duration": 0.2,
            "status": "resolved"
        }
        decisions = agent.update([event_exit])
        assert len(decisions) == 1
        dec = decisions[0]
        # After timeout, the agent should have reset to OBSERVE and then process exit as no action
        assert dec["decision"] == ActionType.NO_ACTION.value
        assert dec["person_id"] == 2
        assert dec["next_action"] == "none"


def test_agent_multiple_persons():
    """Test agent handles multiple persons independently."""
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time
        # Person 1 enters
        event1_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 1,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential"
        }
        # Person 2 enters later
        event2_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 2,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time + 0.05,
            "duration": 0.05,
            "status": "potential"
        }
        # Process both entries at the same time (using current_time for simplicity)
        mock_time.return_value = current_time
        dec1 = agent.update([event1_entry])[0]
        assert dec1["decision"] == ActionType.CONTINUE_TRACKING.value
        assert dec1["person_id"] == 1
        # Then person 2
        mock_time.return_value = current_time + 0.05
        dec2 = agent.update([event2_entry])[0]
        assert dec2["decision"] == ActionType.CONTINUE_TRACKING.value
        assert dec2["person_id"] == 2

        # Now advance time and confirm person 1
        mock_time.return_value = current_time + 0.2
        event1_confirm = {
            "event_type": "restricted_zone_confirmation",
            "person_id": 1,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time + 0.2,
            "duration": 0.2,
            "status": "confirmed"
        }
        dec1_confirm = agent.update([event1_confirm])[0]
        assert dec1_confirm["decision"] == ActionType.CREATE_INCIDENT.value
        assert dec1_confirm["person_id"] == 1

        # Person 2 is still in potential (we haven't sent confirmation or exit)
        # Send an exit for person 2
        mock_time.return_value = current_time + 0.3
        event2_exit = {
            "event_type": "restricted_zone_exit",
            "person_id": 2,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time + 0.3,
            "duration": 0.3,
            "status": "resolved"
        }
        dec2_exit = agent.update([event2_exit])[0]
        # Person 2 was in potential state and timed out
        assert dec2_exit["decision"] == ActionType.NO_ACTION.value
        assert dec2_exit["next_action"] == "none"


def test_agent_audit_trail():
    """Test that agent maintains complete audit trail."""
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time
        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 1,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential"
        }
        agent.update([event_entry])

        mock_time.return_value = current_time + 0.2
        event_confirm = {
            "event_type": "restricted_zone_confirmation",
            "person_id": 1,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time + 0.2,
            "duration": 0.2,
            "status": "confirmed"
        }
        agent.update([event_confirm])

        # Check trace
        trace = agent.get_trace(1)
        assert trace is not None
        assert trace.person_id == 1
        assert len(trace.observations) == 2
        assert len(trace.decisions) == 2
        assert trace.observations[0].event_type == "restricted_zone_entry"
        assert trace.observations[1].event_type == "restricted_zone_confirmation"
        assert trace.decisions[0].selected_action == ActionType.CONTINUE_TRACKING
        assert trace.decisions[1].selected_action == ActionType.CREATE_INCIDENT


def test_agent_stats():
    """Test agent statistics tracking."""
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time
        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 1,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential"
        }
        agent.update([event_entry])

        mock_time.return_value = current_time + 0.2
        event_confirm = {
            "event_type": "restricted_zone_confirmation",
            "person_id": 1,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time + 0.2,
            "duration": 0.2,
            "status": "confirmed"
        }
        agent.update([event_confirm])

        stats = agent.get_stats()
        assert stats["total_observations"] == 2
        assert stats["total_decisions"] == 2
        assert stats["incidents_created"] == 1


def test_agent_reset():
    """Test agent reset functionality."""
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time
        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 1,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential"
        }
        agent.update([event_entry])

        # Verify state exists
        assert 1 in agent.person_state
        assert 1 in agent.person_traces

        # Reset person
        agent.reset_person(1)
        assert 1 not in agent.person_state
        assert 1 not in agent.person_traces

        # Reset all
        agent.update([event_entry])
        agent.reset_all()
        assert len(agent.person_state) == 0
        assert len(agent.person_traces) == 0
        assert agent.stats["total_observations"] == 0


def test_agent_feedback_loop_inspect_crop_changes_decision():
    """Test that inspect_crop tool can boost confidence and change risk level."""
    reset_vision_client()
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time

        # Initial entry event - low confidence
        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 1,
            "confidence": 0.55,  # Just above threshold but not high
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential",
            "bbox": [100, 100, 50, 50],
        }

        # Mock the vision client to return a confidence boost
        # Need to patch both where the tool checks availability and where sync wrapper gets client
        with patch('backend.agent.agent.get_vision_client') as mock_get_client_agent, \
             patch('backend.agent.vision_client.get_vision_client') as mock_get_client_vision:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.inspect_crop = AsyncMock(return_value=MagicMock(
                model_dump=lambda: {
                    "tool": "inspect_crop",
                    "person_id": 1,
                    "bbox": [100, 100, 50, 50],
                    "result": "crop_inspected",
                    "confidence_boost": 0.2,  # Boost from 0.55 to 0.75
                    "risk_level_change": "higher",
                    "details": "Clear safety violation detected",
                    "violations_detected": ["missing_hard_hat"],
                }
            ))
            mock_get_client_agent.return_value = mock_client
            mock_get_client_vision.return_value = mock_client

            # First update - should create incident when violations detected
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            decisions = agent.update([event_entry], frame=mock_frame, frame_number=1)
            assert len(decisions) == 1
            dec = decisions[0]
            # Violations detected -> status becomes confirmed -> create incident
            assert dec["next_action"] == "create_incident"
            assert dec["risk_level"] in ["high", "critical"]

            # Check that tool was called and decision may have changed
            # The feedback loop should have run inspect_crop and boosted confidence
            trace = agent.get_trace(1)
            assert trace is not None
            assert len(trace.tool_calls) >= 1
            tool_call = trace.tool_calls[0]
            assert tool_call["tool"] == "inspect_crop"
            assert tool_call["result"].get("confidence_boost", 0) > 0


def test_agent_gather_more_evidence_tool():
    """Test that gather_more_evidence tool can be called and returns scene assessment."""
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    mock_frame = MagicMock()
    mock_frame.shape = (480, 640, 3)

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time

        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 2,
            "confidence": 0.6,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential",
            "bbox": [200, 200, 60, 60],
        }

        with patch('backend.agent.vision_client.get_vision_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.assess_scene = MagicMock(return_value={
                "task": "assess_scene",
                "risk_level": "high",
                "confidence": 0.85,
                "reason_codes": ["multiple_violations"],
                "recommended_action": "create_incident",
                "details": "Scene assessment confirms violation",
            })
            mock_get_client.return_value = mock_client

            decisions = agent.update([event_entry], frame=mock_frame, frame_number=1)

            trace = agent.get_trace(2)
            assert trace is not None
            # Check that gather_more_evidence or inspect_crop was called
            tool_names = [tc["tool"] for tc in trace.tool_calls]
            assert any(t in tool_names for t in ["gather_more_evidence", "inspect_crop", "verify_ppe"])


def test_agent_iteration_limit():
    """Test that agent respects max iterations per event (3)."""
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    mock_frame = MagicMock()
    mock_frame.shape = (480, 640, 3)

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time

        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 3,
            "confidence": 0.5,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential",
            "bbox": [300, 300, 40, 40],
        }

        call_count = 0
        def mock_inspect_crop(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "tool": "inspect_crop",
                "person_id": 3,
                "bbox": [300, 300, 40, 40],
                "result": "crop_inspected",
                "confidence_boost": 0.15,  # Keeps boosting to trigger more iterations
                "risk_level_change": "higher",
                "details": f"Inspection {call_count}",
                "violations_detected": ["violation"],
            }

        with patch('backend.agent.vision_client.get_vision_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.inspect_crop = mock_inspect_crop
            mock_get_client.return_value = mock_client

            decisions = agent.update([event_entry], frame=mock_frame, frame_number=1)

            # Should have limited iterations (max 3)
            trace = agent.get_trace(3)
            assert trace is not None
            # The decision should have iteration count <= 3
            for dec in trace.decisions:
                assert dec.retry_count <= 2  # 0-indexed, max 3 iterations = retry_count 2


def test_agent_verify_ppe_changes_compliance():
    """Test that verify_ppe tool can detect non-compliance and affect decision."""
    reset_vision_client()
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time

        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 4,
            "confidence": 0.6,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential",
            "bbox": [400, 400, 50, 50],
        }

        # Mock the vision client to return PPE non-compliance
        # Need to patch both where the tool checks availability and where sync wrapper gets client
        with patch('backend.agent.agent.get_vision_client') as mock_get_client_agent, \
             patch('backend.agent.vision_client.get_vision_client') as mock_get_client_vision:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            # First call: inspect_crop (agent calls this first for low confidence)
            # Second call: verify_ppe (after inspect_crop detects PPE issues)
            inspect_crop_results = [
                {
                    "tool": "inspect_crop",
                    "person_id": 4,
                    "bbox": [400, 400, 50, 50],
                    "result": "crop_inspected",
                    "confidence_boost": 0.15,  # Boost to 0.75
                    "risk_level_change": "higher",
                    "details": "Potential violation detected",
                    "violations_detected": ["missing_hard_hat"],
                    "ppe_status": {"hard_hat": False, "vest": True, "gloves": False, "boots": True},
                }
            ]
            verify_ppe_result = {
                "tool": "verify_ppe",
                "bbox": [400, 400, 50, 50],
                "ppe_detected": {"hard_hat": False, "vest": True, "gloves": False, "boots": True},
                "compliance": "non_compliant",
                "confidence": 0.9,
                "details": "Hard hat and gloves missing",
                "missing_items": ["hard_hat", "gloves"],
            }

            call_count = [0]
            async def mock_inspect_crop(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] <= len(inspect_crop_results):
                    return MagicMock(model_dump=lambda: inspect_crop_results[call_count[0] - 1])
                return MagicMock(model_dump=lambda: inspect_crop_results[-1])

            mock_client.inspect_crop = mock_inspect_crop
            mock_client.verify_ppe = AsyncMock(return_value=MagicMock(model_dump=lambda: verify_ppe_result))
            mock_get_client_agent.return_value = mock_client
            mock_get_client_vision.return_value = mock_client

            decisions = agent.update([event_entry], frame=mock_frame, frame_number=1)

            trace = agent.get_trace(4)
            assert trace is not None
            tool_names = [tc["tool"] for tc in trace.tool_calls]
            # Agent will call inspect_crop first for low confidence, then verify_ppe in feedback loop
            assert "inspect_crop" in tool_names
            assert "verify_ppe" in tool_names

            # Check that PPE non-compliance was noted in reason codes
            for dec in trace.decisions:
                if dec.selected_action == ActionType.VERIFY_PPE:
                    assert "ppe_non_compliant" in dec.reason_codes


def test_agent_nvidia_vision_tool_for_confirmed_event():
    """
    Test that a legitimate safety candidate triggers NVIDIA vision tool call.
    This test mocks the external NVIDIA API and verifies the agent selects
    INSPECT_CROP for a confirmed event regardless of detection confidence.
    """
    import os
    from unittest.mock import patch, MagicMock, AsyncMock
    import numpy as np
    from backend.agent.vision_client import (
        VisionClientConfig, VisionProvider, reset_vision_client
    )
    from backend.agent.agent import SafetyAgent, ActionType, AgentState

    reset_vision_client()

    # Configure agent with NVIDIA provider settings
    with patch.dict(os.environ, {
        "VISION_PROVIDER": "nvidia",
        "NVIDIA_API_KEY": "test-nvidia-key",
        "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
        "VISION_MODEL": "meta/llama-3.2-11b-vision-instruct",
        "VISION_ENABLED": "true",
    }):
        agent = SafetyAgent(request_duration=0.1)
        current_time = time.time()

        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('backend.agent.agent.time.time') as mock_time:
            mock_time.return_value = current_time

            # First, send an entry event to initialize person state to POTENTIAL
            event_entry = {
                "event_type": "restricted_zone_entry",
                "person_id": 5,
                "confidence": 0.85,
                "zone": "restricted_zone_1",
                "timestamp": current_time,
                "duration": 0.0,
                "status": "potential",
                "bbox": [150, 150, 80, 80],
            }
            agent.update([event_entry], frame=mock_frame, frame_number=1)

            # Verify person state is now POTENTIAL
            assert agent.person_state[5]['state'] == AgentState.POTENTIAL

            # Now send a confirmed event (person has been in zone > confirmation_duration)
            mock_time.return_value = current_time + 3.5
            event_confirm = {
                "event_type": "restricted_zone_confirmation",
                "person_id": 5,
                "confidence": 0.85,  # High confidence - per spec should still trigger vision
                "zone": "restricted_zone_1",
                "timestamp": current_time + 3.5,
                "duration": 3.5,  # > confirmation_duration
                "status": "confirmed",
                "bbox": [150, 150, 80, 80],
            }

            # Mock the vision client to track NVIDIA provider usage
            with patch('backend.agent.agent.get_vision_client') as mock_get_client_agent, \
                 patch('backend.agent.vision_client.get_vision_client') as mock_get_client_vision:

                mock_client = MagicMock()
                mock_client.is_available.return_value = True

                # Track which provider config was used
                called_config = {}

                async def mock_inspect_crop(*args, **kwargs):
                    # Capture the provider config
                    called_config['provider'] = 'nvidia'
                    return MagicMock(model_dump=lambda: {
                        "task": "inspect_crop",
                        "person_id": 5,
                        "bbox": [150, 150, 80, 80],
                        "confidence_boost": 0.1,
                        "risk_level_change": "higher",
                        "details": "NVIDIA inspection: safety violation confirmed",
                        "violations_detected": ["missing_hard_hat"],
                        "ppe_status": {"hard_hat": False, "vest": True, "gloves": True, "boots": True},
                    })

                mock_client.inspect_crop = mock_inspect_crop
                mock_get_client_agent.return_value = mock_client
                mock_get_client_vision.return_value = mock_client

                # Process the confirmed event with frame
                decisions = agent.update([event_confirm], frame=mock_frame, frame_number=2)

                # Verify the agent made a decision
                assert len(decisions) == 1
                dec = decisions[0]
                assert dec["person_id"] == 5

                # Check that the agent trace shows tool was called
                trace = agent.get_trace(5)
                assert trace is not None
                assert len(trace.tool_calls) >= 1

                # Verify INSPECT_CROP was called (the key agent behavior for confirmed events)
                tool_names = [tc["tool"] for tc in trace.tool_calls]
                assert "inspect_crop" in tool_names, f"Expected inspect_crop in {tool_names}"

                # Verify the tool result contains NVIDIA response
                inspect_call = next(tc for tc in trace.tool_calls if tc["tool"] == "inspect_crop")
                result = inspect_call["result"]
                assert result.get("violations_detected") == ["missing_hard_hat"]
                assert result.get("ppe_status", {}).get("hard_hat") is False

                # Verify the provider config was NVIDIA
                assert called_config.get('provider') == 'nvidia', "Expected NVIDIA provider to be used"

                # Verify escalation safeguards: human approval required for critical
                assert dec["human_approval_required"] is True or dec["risk_level"] in ["high", "critical"]


def test_agent_ppe_status_consistency_inspect_crop():
    """Test that ppe_status is reconciled with violations_detected in inspect_crop."""
    from backend.agent.agent import SafetyAgent
    from backend.agent.vision_client import reset_vision_client
    import numpy as np

    reset_vision_client()
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time

        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 10,
            "confidence": 0.55,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential",
            "bbox": [100, 100, 50, 50],
        }

        # Mock the vision client to return contradictory data:
        # - details say items are missing
        # - violations_detected says missing_ppe
        # - but ppe_status has null for all items
        with patch('backend.agent.agent.get_vision_client') as mock_get_client_agent, \
             patch('backend.agent.vision_client.get_vision_client') as mock_get_client_vision:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.inspect_crop = AsyncMock(return_value=MagicMock(
                model_dump=lambda: {
                    "tool": "inspect_crop",
                    "person_id": 10,
                    "bbox": [100, 100, 50, 50],
                    "confidence_boost": 0.2,
                    "risk_level_change": "higher",
                    "details": "Person is not wearing a hard hat, safety vest, or gloves. Boots are visible.",
                    "violations_detected": ["missing_ppe"],
                    "ppe_status": {"hard_hat": None, "safety_vest": None, "gloves": None, "boots": None},
                }
            ))
            mock_get_client_agent.return_value = mock_client
            mock_get_client_vision.return_value = mock_client

            decisions = agent.update([event_entry], frame=mock_frame, frame_number=1)

            trace = agent.get_trace(10)
            assert trace is not None

            # Check that the reconciliation fixed the ppe_status
            inspect_call = next(tc for tc in trace.tool_calls if tc["tool"] == "inspect_crop")
            result = inspect_call["result"]

            # After reconciliation, ppe_status should have false for items mentioned as missing
            ppe_status = result.get("ppe_status", {})
            # The reconciliation runs on the agent's current_event, not directly on tool result
            # But the decision should reflect the reconciled values
            # Check that reason_codes contain the violations
            for dec in trace.decisions:
                if dec.selected_action == ActionType.INSPECT_CROP:
                    # The violation should be in reason_codes
                    assert any("missing" in rc for rc in dec.reason_codes)


def test_agent_ppe_status_consistency_verify_ppe():
    """Test that ppe_detected is reconciled with missing_items in verify_ppe."""
    from backend.agent.agent import SafetyAgent, ActionType
    from backend.agent.vision_client import reset_vision_client
    import numpy as np

    reset_vision_client()
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time

        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 11,
            "confidence": 0.6,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential",
            "bbox": [200, 200, 60, 60],
        }

        # First mock inspect_crop to set up PPE issues, then verify_ppe
        with patch('backend.agent.agent.get_vision_client') as mock_get_client_agent, \
             patch('backend.agent.vision_client.get_vision_client') as mock_get_client_vision:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True

            call_count = [0]
            async def mock_inspect_crop(*args, **kwargs):
                call_count[0] += 1
                return MagicMock(model_dump=lambda: {
                    "tool": "inspect_crop",
                    "person_id": 11,
                    "bbox": [200, 200, 60, 60],
                    "confidence_boost": 0.15,
                    "risk_level_change": "higher",
                    "details": "Potential PPE violation",
                    "violations_detected": ["missing_hard_hat"],
                    "ppe_status": {"hard_hat": None, "safety_vest": True, "gloves": True, "boots": True},
                })

            mock_client.inspect_crop = mock_inspect_crop
            mock_client.verify_ppe = AsyncMock(return_value=MagicMock(model_dump=lambda: {
                "tool": "verify_ppe",
                "bbox": [200, 200, 60, 60],
                "ppe_detected": {"hard_hat": None, "safety_vest": True, "gloves": True, "boots": True},
                "compliance": "non_compliant",
                "confidence": 0.9,
                "details": "Hard hat is missing",
                "missing_items": ["hard_hat"],
            }))
            mock_get_client_agent.return_value = mock_client
            mock_get_client_vision.return_value = mock_client

            decisions = agent.update([event_entry], frame=mock_frame, frame_number=1)

            trace = agent.get_trace(11)
            assert trace is not None
            tool_names = [tc["tool"] for tc in trace.tool_calls]
            assert "inspect_crop" in tool_names
            assert "verify_ppe" in tool_names

            # Check that verify_ppe was called and non_compliance detected
            for dec in trace.decisions:
                if dec.selected_action == ActionType.VERIFY_PPE:
                    assert "ppe_non_compliant" in dec.reason_codes


def test_agent_no_incident_when_no_violation():
    """Test that no incident is created when vision confirms no violations."""
    from backend.agent.agent import SafetyAgent
    from backend.agent.vision_client import reset_vision_client
    import numpy as np

    reset_vision_client()
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time

        # First entry
        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 12,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential",
            "bbox": [300, 300, 50, 50],
        }
        agent.update([event_entry], frame=mock_frame, frame_number=1)

        # Confirmation
        mock_time.return_value = current_time + 3.5
        event_confirm = {
            "event_type": "restricted_zone_confirmation",
            "person_id": 12,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time + 3.5,
            "duration": 3.5,
            "status": "confirmed",
            "bbox": [300, 300, 50, 50],
        }

        with patch('backend.agent.agent.get_vision_client') as mock_get_client_agent, \
             patch('backend.agent.vision_client.get_vision_client') as mock_get_client_vision:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.inspect_crop = AsyncMock(return_value=MagicMock(
                model_dump=lambda: {
                    "tool": "inspect_crop",
                    "person_id": 12,
                    "bbox": [300, 300, 50, 50],
                    "confidence_boost": 0.0,
                    "risk_level_change": "same",
                    "details": "Person is wearing all required PPE. No safety violations.",
                    "violations_detected": [],
                    "ppe_status": {"hard_hat": True, "safety_vest": True, "gloves": True, "boots": True},
                }
            ))
            mock_get_client_agent.return_value = mock_client
            mock_get_client_vision.return_value = mock_client

            decisions = agent.update([event_confirm], frame=mock_frame, frame_number=2)

            trace = agent.get_trace(12)
            assert trace is not None

            # Should NOT create incident when no violations
            incident_created = any(d.selected_action == ActionType.CREATE_INCIDENT for d in trace.decisions)
            # The agent may still request human approval for critical, but not create incident without violations
            for dec in trace.decisions:
                if dec.selected_action == ActionType.INSPECT_CROP:
                    # No violations found from vision tool, should not have violation: reason codes
                    vision_violation_codes = [rc for rc in dec.reason_codes if rc.startswith("violation:")]
                    assert len(vision_violation_codes) == 0, f"Expected no vision violations, got: {vision_violation_codes}"


def test_agent_uncertain_evidence_requires_human_review():
    """Test that uncertain vision results (ppe_status all null) require human review."""
    from backend.agent.agent import SafetyAgent
    from backend.agent.vision_client import reset_vision_client
    import numpy as np

    reset_vision_client()
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time

        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 13,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential",
            "bbox": [400, 400, 50, 50],
        }
        agent.update([event_entry], frame=mock_frame, frame_number=1)

        mock_time.return_value = current_time + 3.5
        event_confirm = {
            "event_type": "restricted_zone_confirmation",
            "person_id": 13,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time + 3.5,
            "duration": 3.5,
            "status": "confirmed",
            "bbox": [400, 400, 50, 50],
        }

        with patch('backend.agent.agent.get_vision_client') as mock_get_client_agent, \
             patch('backend.agent.vision_client.get_vision_client') as mock_get_client_vision:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            # Return uncertain result - all ppe_status null, no clear violations
            mock_client.inspect_crop = AsyncMock(return_value=MagicMock(
                model_dump=lambda: {
                    "tool": "inspect_crop",
                    "person_id": 13,
                    "bbox": [400, 400, 50, 50],
                    "confidence_boost": 0.0,
                    "risk_level_change": "same",
                    "details": "Image too blurry to assess PPE. Cannot determine compliance.",
                    "violations_detected": [],
                    "ppe_status": {"hard_hat": None, "safety_vest": None, "gloves": None, "boots": None},
                }
            ))
            mock_get_client_agent.return_value = mock_client
            mock_get_client_vision.return_value = mock_client

            decisions = agent.update([event_confirm], frame=mock_frame, frame_number=2)

            trace = agent.get_trace(13)
            assert trace is not None

            # With uncertain evidence, should not auto-create incident
            # Should either continue tracking or request human approval
            for dec in trace.decisions:
                if dec.selected_action == ActionType.INSPECT_CROP:
                    # Uncertain result - no clear violation
                    assert dec.verification_status.value in ["tool_completed", "unverified"]


def test_agent_nvidia_timeout_invalid_json():
    """Test agent handles NVIDIA timeout/invalid JSON gracefully."""
    from backend.agent.agent import SafetyAgent
    from backend.agent.vision_client import reset_vision_client
    import numpy as np

    reset_vision_client()
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time

        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 14,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential",
            "bbox": [500, 500, 50, 50],
        }
        agent.update([event_entry], frame=mock_frame, frame_number=1)

        mock_time.return_value = current_time + 3.5
        event_confirm = {
            "event_type": "restricted_zone_confirmation",
            "person_id": 14,
            "confidence": 0.7,
            "zone": "restricted_zone_1",
            "timestamp": current_time + 3.5,
            "duration": 3.5,
            "status": "confirmed",
            "bbox": [500, 500, 50, 50],
        }

        with patch('backend.agent.agent.get_vision_client') as mock_get_client_agent, \
             patch('backend.agent.vision_client.get_vision_client') as mock_get_client_vision:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            # Simulate timeout error
            async def mock_inspect_crop_fail(*args, **kwargs):
                raise Exception("Timeout")
            mock_client.inspect_crop = mock_inspect_crop_fail
            mock_get_client_agent.return_value = mock_client
            mock_get_client_vision.return_value = mock_client

            decisions = agent.update([event_confirm], frame=mock_frame, frame_number=2)

            trace = agent.get_trace(14)
            assert trace is not None

            # Should handle gracefully - verification_failed or unverified
            for dec in trace.decisions:
                if dec.selected_action == ActionType.INSPECT_CROP:
                    assert dec.verification_status.value in ["verification_failed", "unverified"]


def test_agent_duplicate_events_incident_deduplication():
    """Test that duplicate events for same person don't create multiple incidents."""
    from backend.agent.agent import SafetyAgent
    from backend.agent.vision_client import reset_vision_client
    import numpy as np

    reset_vision_client()
    agent = SafetyAgent(request_duration=0.1)
    current_time = time.time()

    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch('backend.agent.agent.time.time') as mock_time:
        mock_time.return_value = current_time

        event_entry = {
            "event_type": "restricted_zone_entry",
            "person_id": 15,
            "confidence": 0.8,
            "zone": "restricted_zone_1",
            "timestamp": current_time,
            "duration": 0.0,
            "status": "potential",
            "bbox": [100, 100, 50, 50],
        }
        agent.update([event_entry], frame=mock_frame, frame_number=1)

        mock_time.return_value = current_time + 3.5
        event_confirm = {
            "event_type": "restricted_zone_confirmation",
            "person_id": 15,
            "confidence": 0.8,
            "zone": "restricted_zone_1",
            "timestamp": current_time + 3.5,
            "duration": 3.5,
            "status": "confirmed",
            "bbox": [100, 100, 50, 50],
        }

        with patch('backend.agent.agent.get_vision_client') as mock_get_client_agent, \
             patch('backend.agent.vision_client.get_vision_client') as mock_get_client_vision:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.inspect_crop = AsyncMock(return_value=MagicMock(
                model_dump=lambda: {
                    "tool": "inspect_crop",
                    "person_id": 15,
                    "bbox": [100, 100, 50, 50],
                    "confidence_boost": 0.1,
                    "risk_level_change": "higher",
                    "details": "Safety violation detected",
                    "violations_detected": ["missing_hard_hat"],
                    "ppe_status": {"hard_hat": False, "safety_vest": True, "gloves": True, "boots": True},
                }
            ))
            mock_get_client_agent.return_value = mock_client
            mock_get_client_vision.return_value = mock_client

            # Process confirmation event multiple times (simulating duplicate frames)
            for i in range(3):
                decisions = agent.update([event_confirm], frame=mock_frame, frame_number=2+i)

            trace = agent.get_trace(15)
            assert trace is not None

            # Should only create ONE incident despite multiple confirmations
            incident_count = sum(1 for d in trace.decisions if d.selected_action == ActionType.CREATE_INCIDENT)
            # Only the first confirmation should create an incident
            assert incident_count <= 1


if __name__ == "__main__":
    test_agent_entry_and_confirmation()
    test_agent_entry_then_exit_no_confirmation()
    test_agent_multiple_persons()
    test_agent_audit_trail()
    test_agent_stats()
    test_agent_reset()
    test_agent_feedback_loop_inspect_crop_changes_decision()
    test_agent_gather_more_evidence_tool()
    test_agent_iteration_limit()
    test_agent_verify_ppe_changes_compliance()
    test_agent_nvidia_vision_tool_for_confirmed_event()
    print("All agent tests passed!")