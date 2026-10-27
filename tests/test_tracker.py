import numpy as np
import time
from backend.vision.tracker import CentroidTracker

def test_centroid_tracker_register():
    """Test registering a new object."""
    tracker = CentroidTracker()
    centroid = np.array([100, 100])
    bbox = [50, 50, 100, 100]  # x, y, w, h
    confidence = 0.9

    tracker.register(centroid, bbox, confidence)

    assert len(tracker.objects) == 1
    object_id = list(tracker.objects.keys())[0]
    assert object_id == 0
    assert np.array_equal(tracker.objects[object_id]['centroid'], centroid)
    assert tracker.objects[object_id]['bbox'] == bbox
    assert tracker.objects[object_id]['confidence'] == confidence
    assert tracker.disappeared[object_id] == 0

def test_centroid_tracker_update():
    """Test updating the tracker with detections."""
    tracker = CentroidTracker(max_disappeared=5, max_distance=50)

    # Register an object
    centroid1 = np.array([100, 100])
    bbox1 = [50, 50, 100, 100]
    tracker.register(centroid1, bbox1, 0.9)

    # Update with a detection near the same location
    detections = [
        {'bbox': [55, 55, 90, 90], 'confidence': 0.8}  # slightly moved
    ]
    tracked = tracker.update(detections)

    assert len(tracked) == 1
    object_id = list(tracked.keys())[0]
    # The centroid should be updated to the new detection's centroid
    expected_centroid = np.array([55 + 90//2, 55 + 90//2])  # [100, 100]
    assert np.array_equal(tracked[object_id]['centroid'], expected_centroid)
    assert tracked[object_id]['bbox'] == [55, 55, 90, 90]
    assert tracked[object_id]['confidence'] == 0.8
    assert tracker.disappeared[object_id] == 0

def test_centroid_tracker_disappear():
    """Test that objects are removed after disappearing too long."""
    tracker = CentroidTracker(max_disappeared=2, max_distance=50)

    # Register an object
    centroid = np.array([100, 100])
    bbox = [50, 50, 100, 100]
    tracker.register(centroid, bbox, 0.9)

    # Update with no detections for 3 frames
    for i in range(3):
        tracked = tracker.update([])

    # Object should be removed
    assert len(tracker.objects) == 0
    assert len(tracker.disappeared) == 0

def test_centroid_tracker_new_detection():
    """Test that new detections are registered as new objects."""
    tracker = CentroidTracker(max_disappeared=5, max_distance=50)

    # First frame: one detection
    detections1 = [
        {'bbox': [50, 50, 100, 100], 'confidence': 0.9}
    ]
    tracked1 = tracker.update(detections1)
    assert len(tracked1) == 1
    assert list(tracked1.keys())[0] == 0

    # Second frame: two detections (one existing, one new)
    detections2 = [
        {'bbox': [55, 55, 90, 90], 'confidence': 0.8},  # moved slightly
        {'bbox': [300, 300, 50, 50], 'confidence': 0.7}  # new object far away
    ]
    tracked2 = tracker.update(detections2)
    assert len(tracked2) == 2
    # Check that we have two objects with IDs 0 and 1
    assert set(tracked2.keys()) == {0, 1}
    # Object 0 should be the one that moved slightly
    assert tracked2[0]['bbox'] == [55, 55, 90, 90]
    # Object 1 should be the new one
    assert tracked2[1]['bbox'] == [300, 300, 50, 50]