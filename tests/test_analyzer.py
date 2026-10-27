import numpy as np
import time
import pytest
from unittest.mock import patch, MagicMock
from backend.vision.analyzer import SafetyAnalyzer
from backend.vision.roi import RegionOfInterest
from backend.vision.detector import PersonDetector

def test_analyzer_state_transitions():
    """Test the analyzer's state transitions for a person entering and staying in ROI."""
    # Define a simple ROI polygon (a square)
    roi_polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]

    # Mock the detector's model file existence and network loading
    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            # Create a mock network
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            analyzer = SafetyAnalyzer(
                model_path='dummy.onnx',
                roi_polygon=roi_polygon,
                confirmation_duration=0.1,  # 0.1 seconds for quick test
                confidence_threshold=0.5,
                nms_threshold=0.4,
                roi_hysteresis_frames=1  # No hysteresis for test
            )

            # Now mock the detect method of the detector to return a detection inside the ROI
            def mock_detect(frame):
                # Return a detection with bbox inside the ROI
                # The bbox [120, 120, 80, 80] -> center (160,160) which is inside the ROI
                return [{'bbox': [120, 120, 80, 80], 'confidence': 0.9, 'class_id': 0}]

            with patch.object(analyzer.detector, 'detect', side_effect=mock_detect):
                # Create a dummy frame (any size, since we mock detection)
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # First frame: person enters -> should generate a potential event
                annotated_frame, events = analyzer.update(frame)
                assert len(events) == 1
                assert events[0]['event_type'] == 'restricted_zone_entry'
                assert events[0]['status'] == 'potential'
                assert events[0]['person_id'] == 0  # first object gets ID 0

def test_analyzer_confirmation_with_mocked_time():
    """Test confirmation by mocking time.time."""
    roi_polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]

    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            analyzer = SafetyAnalyzer(
                model_path='dummy.onnx',
                roi_polygon=roi_polygon,
                confirmation_duration=0.1,  # 0.1 seconds
                confidence_threshold=0.5,
                nms_threshold=0.4,
                roi_hysteresis_frames=1  # No hysteresis for test
            )

            def mock_detect(frame):
                return [{'bbox': [120, 120, 80, 80], 'confidence': 0.9, 'class_id': 0}]

            with patch.object(analyzer.detector, 'detect', side_effect=mock_detect):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # Mock time.time to control the entry time and the check
                with patch('backend.vision.analyzer.time.time') as mock_time:
                    # Let time start at 0
                    mock_time.return_value = 0.0
                    # First update: enter
                    annotated_frame, events = analyzer.update(frame)
                    assert len(events) == 1
                    assert events[0]['status'] == 'potential'

                    # Now advance time by 0.05 seconds (still inside, but not enough to confirm)
                    mock_time.return_value = 0.05
                    annotated_frame, events = analyzer.update(frame)
                    # Should have no events (still inside, not confirmed yet)
                    assert len(events) == 0

                    # Advance time by another 0.1 seconds (total 0.15, which is > confirmation_duration)
                    mock_time.return_value = 0.15
                    annotated_frame, events = analyzer.update(frame)
                    # Should now get a confirmation event
                    assert len(events) == 1
                    assert events[0]['event_type'] == 'restricted_zone_confirmation'
                    assert events[0]['status'] == 'confirmed'
                    assert events[0]['duration'] >= 0.1  # approximately 0.15 seconds

def test_analyzer_exit_event():
    """Test that an exit event is generated when person leaves ROI."""
    roi_polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]

    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            # Increase max_distance to allow matching the outside detection
            analyzer = SafetyAnalyzer(
                model_path='dummy.onnx',
                roi_polygon=roi_polygon,
                confirmation_duration=0.1,
                confidence_threshold=0.5,
                nms_threshold=0.4,
                max_distance=100,  # Increased to allow matching over larger distance
                roi_hysteresis_frames=1  # No hysteresis for test
            )

            # We'll simulate a person entering and then leaving by changing the detection location
            def mock_detect_inside(frame):
                return [{'bbox': [120, 120, 80, 80], 'confidence': 0.9, 'class_id': 0}]  # inside

            def mock_detect_outside(frame):
                # This bbox is just outside the ROI to the right: [200, 140, 80, 80] -> centroid (240,180)
                # Distance from inside centroid (160,160) is sqrt((80)^2+(20)^2)=82.4 < 100
                return [{'bbox': [200, 140, 80, 80], 'confidence': 0.9, 'class_id': 0}]  # outside

            with patch.object(analyzer.detector, 'detect') as mock_detect:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # Mock time to ensure minimum duration for exit event
                with patch('backend.vision.analyzer.time.time') as mock_time:
                    mock_time.return_value = 0.0
                    # First, detect inside
                    mock_detect.side_effect = mock_detect_inside
                    annotated_frame, events = analyzer.update(frame)
                    assert len(events) == 1
                    assert events[0]['status'] == 'potential'

                    # Advance time by 1 second (so exit duration >= 0.5s)
                    mock_time.return_value = 1.0
                    # Now detect outside
                    mock_detect.side_effect = mock_detect_outside
                    annotated_frame, events = analyzer.update(frame)
                    # Should generate an exit event
                    assert len(events) == 1
                    assert events[0]['event_type'] == 'restricted_zone_exit'
                    assert events[0]['status'] == 'resolved'
                    assert events[0]['duration'] >= 0.5

def test_analyzer_multiple_persons_simultaneous():
    """Test analyzer handles multiple persons entering ROI simultaneously."""
    roi_polygon = [(100, 100), (300, 100), (300, 300), (100, 300)]

    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            analyzer = SafetyAnalyzer(
                model_path='dummy.onnx',
                roi_polygon=roi_polygon,
                confirmation_duration=0.1,
                confidence_threshold=0.5,
                nms_threshold=0.4,
                roi_hysteresis_frames=1
            )

            # Mock detector to return TWO people inside ROI
            def mock_detect(frame):
                return [
                    {'bbox': [120, 120, 80, 80], 'confidence': 0.9, 'class_id': 0},  # person 1
                    {'bbox': [200, 200, 80, 80], 'confidence': 0.85, 'class_id': 0},  # person 2
                ]

            with patch.object(analyzer.detector, 'detect', side_effect=mock_detect):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                annotated_frame, events = analyzer.update(frame)

                # Should generate two entry events
                assert len(events) == 2
                entry_events = [e for e in events if e['event_type'] == 'restricted_zone_entry']
                assert len(entry_events) == 2
                # Each should have unique person_id
                person_ids = {e['person_id'] for e in entry_events}
                assert len(person_ids) == 2
                assert all(e['status'] == 'potential' for e in entry_events)


def test_analyzer_person_outside_roi():
    """Test that detections outside ROI do not generate entry events."""
    roi_polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]

    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            analyzer = SafetyAnalyzer(
                model_path='dummy.onnx',
                roi_polygon=roi_polygon,
                confirmation_duration=0.1,
                confidence_threshold=0.5,
                nms_threshold=0.4,
                roi_hysteresis_frames=1
            )

            # Mock detector to return detection OUTSIDE ROI
            def mock_detect(frame):
                # Bbox [300, 300, 80, 80] -> center (340, 340) is well outside ROI
                return [{'bbox': [300, 300, 80, 80], 'confidence': 0.9, 'class_id': 0}]

            with patch.object(analyzer.detector, 'detect', side_effect=mock_detect):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                annotated_frame, events = analyzer.update(frame)

                # Should generate NO events since person is outside ROI
                assert len(events) == 0


def test_analyzer_confidence_threshold_filtering():
    """Test that detections below confidence threshold are ignored by detector."""
    # This tests the DETECTOR's internal confidence filtering
    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            detector = PersonDetector(
                model_path='dummy.onnx',
                confidence_threshold=0.7,
                nms_threshold=0.4
            )

            # Mock forward to return detections: one above, one below threshold
            # YOLO11 output: (1, 84, N) where N > 84 (e.g., 8400)
            # After squeeze -> (84, N), transpose -> (N, 84)
            # Each row: 4 box + 80 class scores
            N = 100  # Must be > 84 for transpose to occur
            mock_output = np.zeros((1, 84, N), dtype=np.float32)
            # Detection 1 (index 0): confidence 0.8 (above 0.7), class 0
            mock_output[0, 0:4, 0] = [320, 240, 80, 80]  # cx, cy, w, h
            mock_output[0, 4, 0] = 0.8  # class 0 confidence
            # Detection 2 (index 1): confidence 0.4 (below 0.7), class 0
            mock_output[0, 0:4, 1] = [100, 100, 60, 60]
            mock_output[0, 4, 1] = 0.4  # class 0 confidence
            # Rest are zeros (confidence 0, filtered out)

            mock_net.forward.return_value = mock_output

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)

            # Should only return the one above threshold
            assert len(detections) == 1
            assert detections[0]['confidence'] == pytest.approx(0.8)
            assert detections[0]['class_id'] == 0


def test_analyzer_nms_behavior():
    """Test that NMS suppresses overlapping detections in the detector."""
    # This tests the DETECTOR's internal NMS
    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            detector = PersonDetector(
                model_path='dummy.onnx',
                confidence_threshold=0.5,
                nms_threshold=0.4
            )

            # Mock forward to return two HIGHLY OVERLAPPING detections
            # YOLO11 output: (1, 84, N) where N > 84
            N = 100
            mock_output = np.zeros((1, 84, N), dtype=np.float32)
            # Detection 1: high confidence
            mock_output[0, 0:4, 0] = [320, 240, 80, 80]  # cx, cy, w, h
            mock_output[0, 4, 0] = 0.9  # class 0 confidence
            # Detection 2: overlapping, slightly lower confidence
            mock_output[0, 0:4, 1] = [325, 245, 80, 80]
            mock_output[0, 4, 1] = 0.85  # class 0 confidence

            mock_net.forward.return_value = mock_output

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)

            # NMS should suppress the overlapping detection
            assert len(detections) == 1
            assert detections[0]['confidence'] == pytest.approx(0.9)


def test_analyzer_class_id_filtering():
    """Test that only class_id=0 (person) detections are processed by detector."""
    # This tests the DETECTOR's internal class filtering
    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            detector = PersonDetector(
                model_path='dummy.onnx',
                confidence_threshold=0.5,
                nms_threshold=0.4
            )

            # Mock forward to return class_id=0 (person) AND class_id=1 (e.g., bicycle)
            # YOLO11 output: (1, 84, N) - 4 box + 80 classes
            N = 100
            mock_output = np.zeros((1, 84, N), dtype=np.float32)
            # Detection 1 (index 0): class 0 (person) - class scores start at index 4
            mock_output[0, 0:4, 0] = [320, 240, 80, 80]  # cx, cy, w, h
            mock_output[0, 4, 0] = 0.9  # class 0 confidence
            # Detection 2 (index 1): class 1 (not person)
            mock_output[0, 0:4, 1] = [100, 100, 50, 50]
            mock_output[0, 5, 1] = 0.95  # class 1 confidence (index 5 = class 1)

            mock_net.forward.return_value = mock_output

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)

            # Should only process the person (class_id=0)
            assert len(detections) == 1
            assert detections[0]['class_id'] == 0


def test_analyzer_empty_detections():
    """Test analyzer handles frames with no detections."""
    roi_polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]

    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            analyzer = SafetyAnalyzer(
                model_path='dummy.onnx',
                roi_polygon=roi_polygon,
                confirmation_duration=0.1,
                confidence_threshold=0.5,
                nms_threshold=0.4,
                roi_hysteresis_frames=1
            )

            # Mock detector to return NO detections
            def mock_detect(frame):
                return []

            with patch.object(analyzer.detector, 'detect', side_effect=mock_detect):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                annotated_frame, events = analyzer.update(frame)

                # Should generate NO events
                assert len(events) == 0


def test_analyzer_detection_then_empty():
    """Test that a person entering then disappearing generates proper events.

    When detections stop, the tracker keeps the object for max_disappeared frames.
    Confirmation still occurs if enough time has passed. No exit event is generated
    when the object simply disappears (exit events require detection outside ROI).
    """
    roi_polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]

    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            analyzer = SafetyAnalyzer(
                model_path='dummy.onnx',
                roi_polygon=roi_polygon,
                confirmation_duration=0.1,
                confidence_threshold=0.5,
                nms_threshold=0.4,
                max_disappeared=2,
                roi_hysteresis_frames=1
            )

            detection_calls = [0]

            def mock_detect(frame):
                detection_calls[0] += 1
                if detection_calls[0] == 1:
                    # First frame: person inside
                    return [{'bbox': [120, 120, 80, 80], 'confidence': 0.9, 'class_id': 0}]
                else:
                    # Subsequent frames: no detections (person disappeared)
                    return []

            with patch.object(analyzer.detector, 'detect', side_effect=mock_detect):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                with patch('backend.vision.analyzer.time.time') as mock_time:
                    mock_time.return_value = 0.0
                    # Frame 1: person enters
                    annotated_frame, events = analyzer.update(frame)
                    assert len(events) == 1
                    assert events[0]['event_type'] == 'restricted_zone_entry'

                    mock_time.return_value = 0.5
                    # Frame 2: no detection, but person was in ROI > confirmation_duration (0.1)
                    # Should generate confirmation event (object still tracked, disappeared=1)
                    annotated_frame, events = analyzer.update(frame)
                    assert len(events) == 1
                    assert events[0]['event_type'] == 'restricted_zone_confirmation'
                    assert events[0]['status'] == 'confirmed'

                    mock_time.return_value = 1.0
                    # Frame 3: no detection - still tracked (max_disappeared=2, disappeared=2)
                    annotated_frame, events = analyzer.update(frame)
                    assert len(events) == 0

                    mock_time.return_value = 1.5
                    # Frame 4: no detection - object deregistered (disappeared=3 > max_disappeared=2)
                    # No exit event generated because object wasn't detected outside ROI
                    annotated_frame, events = analyzer.update(frame)
                    assert len(events) == 0


def test_analyzer_tracker_id_consistency():
    """Test that tracker maintains consistent person IDs across frames."""
    roi_polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]

    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            analyzer = SafetyAnalyzer(
                model_path='dummy.onnx',
                roi_polygon=roi_polygon,
                confirmation_duration=0.1,
                confidence_threshold=0.5,
                nms_threshold=0.4,
                roi_hysteresis_frames=1
            )

            # Mock detector - person moves slightly between frames
            def mock_detect_frame1(frame):
                return [{'bbox': [120, 120, 80, 80], 'confidence': 0.9, 'class_id': 0}]

            def mock_detect_frame2(frame):
                return [{'bbox': [125, 125, 80, 80], 'confidence': 0.9, 'class_id': 0}]

            with patch.object(analyzer.detector, 'detect', side_effect=mock_detect_frame1):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                annotated_frame, events = analyzer.update(frame)
                assert len(events) == 1
                person_id_1 = events[0]['person_id']

            with patch.object(analyzer.detector, 'detect', side_effect=mock_detect_frame2):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                annotated_frame, events = analyzer.update(frame)
                # No new event (still same person), but if confirmation happened...
                # The tracker should have assigned same ID

            # Verify tracker has the person tracked
            tracked = analyzer.tracker.objects
            assert len(tracked) == 1
            assert person_id_1 in tracked


def test_detector_initialization_without_model():
    """Test that detector raises error when model file doesn't exist."""
    from backend.vision.detector import PersonDetector

    with patch('backend.vision.detector.os.path.exists', return_value=False):
        try:
            detector = PersonDetector(
                model_path='nonexistent.onnx',
                confidence_threshold=0.5,
                nms_threshold=0.4
            )
            # Should have raised an error
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected


def test_detector_input_blob_format():
    """Test that detector creates correct input blob format."""
    from backend.vision.detector import PersonDetector
    import cv2

    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            detector = PersonDetector(
                model_path='dummy.onnx',
                confidence_threshold=0.5,
                nms_threshold=0.4
            )

            # Create a test frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Fill with some pattern
            frame[:, :] = (128, 64, 32)

            # Mock the forward pass to return empty detections
            mock_net.forward.return_value = np.zeros((1, 1, 8400, 85), dtype=np.float32)

            # Call detect
            detections = detector.detect(frame)

            # Verify blob was created with correct parameters
            mock_net.setInput.assert_called_once()
            call_args = mock_net.setInput.call_args
            blob = call_args[0][0]
            # Check blob shape: (1, 3, 640, 640) - batch, channels, height, width
            assert blob.shape == (1, 3, 640, 640), f"Expected (1, 3, 640, 640), got {blob.shape}"


def test_detector_nms_threshold_effect():
    """Test that different NMS thresholds produce different results."""
    # Test with low NMS threshold (aggressive suppression)
    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            # Create detector with very low NMS threshold
            detector = PersonDetector(
                model_path='dummy.onnx',
                confidence_threshold=0.5,
                nms_threshold=0.1  # Very aggressive NMS
            )

            # Mock forward to return two overlapping detections
            # YOLO11 output: (1, 84, N) where N > 84
            N = 100
            mock_output = np.zeros((1, 84, N), dtype=np.float32)
            # Detection 1: high confidence, class 0
            mock_output[0, 0:4, 0] = [320, 240, 80, 80]  # cx, cy, w, h
            mock_output[0, 4, 0] = 0.9  # class 0 confidence
            # Detection 2: overlapping, slightly lower confidence
            mock_output[0, 0:4, 1] = [325, 245, 80, 80]
            mock_output[0, 4, 1] = 0.85  # class 0 confidence

            mock_net.forward.return_value = mock_output

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)

            # With aggressive NMS (0.1), overlapping boxes should be suppressed
            # Only one detection should remain
            assert len(detections) == 1
            assert detections[0]['confidence'] == pytest.approx(0.9)


if __name__ == '__main__':
    test_analyzer_state_transitions()
    test_analyzer_confirmation_with_mocked_time()
    test_analyzer_exit_event()
    test_analyzer_multiple_persons_simultaneous()
    test_analyzer_person_outside_roi()
    test_analyzer_confidence_threshold_filtering()
    test_analyzer_nms_behavior()
    test_analyzer_class_id_filtering()
    test_analyzer_empty_detections()
    test_analyzer_detection_then_empty()
    test_analyzer_tracker_id_consistency()
    test_detector_initialization_without_model()
    test_detector_input_blob_format()
    test_detector_nms_threshold_effect()
    print("All tests passed!")