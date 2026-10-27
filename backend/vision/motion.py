import numpy as np
from typing import Dict, Tuple, Optional
import time

class MotionAnalyzer:
    def __init__(self, history_length: int = 5):
        """
        Initialize the motion analyzer.

        Args:
            history_length: Number of previous positions to keep for each object.
        """
        self.history_length = history_length
        # Store history of centroids and timestamps for each objectID
        self.history: Dict[int, list] = {}  # objectID -> list of (centroid_x, centroid_y, timestamp)

    def update(self, tracked_objects: Dict[int, dict]) -> Dict[int, dict]:
        """
        Update motion information for each tracked object.

        Args:
            tracked_objects: Dictionary from tracker.update() where keys are object IDs and values are dicts with:
                'centroid': (x, y)
                'bbox': [x, y, width, height]
                'confidence': float
                'first_seen': timestamp
                'last_seen': timestamp

        Returns:
            A dictionary with the same keys as tracked_objects, where each value is the original dict
            augmented with motion information:
                'displacement': (dx, dy) since last frame (or None if not enough history)
                'speed': pixels per second (or None)
                'direction': angle in radians (or None)
        """
        motion_info = {}

        for object_id, obj_data in tracked_objects.items():
            centroid = obj_data['centroid']
            timestamp = obj_data['last_seen']

            # Initialize history for new object
            if object_id not in self.history:
                self.history[object_id] = []

            # Append current observation
            self.history[object_id].append((centroid[0], centroid[1], timestamp))

            # Keep only the most recent history_length entries
            if len(self.history[object_id]) > self.history_length:
                self.history[object_id] = self.history[object_id][-self.history_length:]

            # Compute motion if we have at least two points
            if len(self.history[object_id]) >= 2:
                # Get the two most recent points
                (x_prev, y_prev, t_prev) = self.history[object_id][-2]
                (x_curr, y_curr, t_curr) = self.history[object_id][-1]
                dx = x_curr - x_prev
                dy = y_curr - y_prev
                dt = t_curr - t_prev

                if dt > 0:
                    speed = np.sqrt(dx**2 + dy**2) / dt
                    # Avoid division by zero for direction
                    if dx == 0 and dy == 0:
                        direction = 0.0
                    else:
                        direction = np.arctan2(dy, dx)  # in radians
                else:
                    speed = None
                    direction = None
            else:
                dx = dy = dt = speed = direction = None

            # Augment the object data with motion info
            augmented = obj_data.copy()
            augmented['displacement'] = (dx, dy) if dx is not None and dy is not None else None
            augmented['speed'] = speed
            augmented['direction'] = direction
            motion_info[object_id] = augmented

        return motion_info