import numpy as np
import time
from backend.vision.motion import MotionAnalyzer

def test_motion_analyzer_update():
    """Test updating motion analyzer with tracked objects."""
    motion = MotionAnalyzer(history_length=5)

    # Simulate tracked objects from tracker
    tracked_objects = {
        0: {
            'centroid': (100, 100),
            'bbox': [50, 50, 100, 100],
            'confidence': 0.9,
            'first_seen': time.time() - 1.0,
            'last_seen': time.time() - 1.0
        }
    }

    # First update: no history yet, so motion should be None
    result = motion.update(tracked_objects)
    assert result[0]['displacement'] is None
    assert result[0]['speed'] is None
    assert result[0]['direction'] is None

    # Second update: move the object
    tracked_objects[0]['centroid'] = (110, 110)
    tracked_objects[0]['last_seen'] = time.time()
    result = motion.update(tracked_objects)

    # Now we should have displacement, speed, and direction
    assert result[0]['displacement'] is not None
    dx, dy = result[0]['displacement']
    assert dx == 10
    assert dy == 10
    assert result[0]['speed'] is not None
    assert result[0]['direction'] is not None
    # Direction should be atan2(10,10) = pi/4
    assert abs(result[0]['direction'] - np.arctan2(10, 10)) < 1e-6

def test_motion_analyzer_history_limit():
    """Test that history is limited to history_length."""
    motion = MotionAnalyzer(history_length=2)

    tracked_objects = {
        0: {
            'centroid': (0, 0),
            'bbox': [0, 0, 10, 10],
            'confidence': 0.9,
            'first_seen': time.time(),
            'last_seen': time.time()
        }
    }

    # Add three observations
    for i in range(3):
        tracked_objects[0]['centroid'] = (i*10, i*10)
        tracked_objects[0]['last_seen'] = time.time() + i
        motion.update(tracked_objects)

    # History should only keep the last 2
    assert len(motion.history[0]) == 2
    # The oldest (0,0) should be dropped
    assert motion.history[0][0][0] == 10  # second observation
    assert motion.history[0][1][0] == 20  # third observation