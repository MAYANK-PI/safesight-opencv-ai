import numpy as np
from collections import OrderedDict
import time
import cv2

class CentroidTracker:
    def __init__(self, max_disappeared: int = 50, max_distance: int = 50, use_appearance: bool = True):
        """
        Initialize the centroid tracker.

        Args:
            max_disappeared: Maximum number of consecutive frames an object can be marked as disappeared before deregistering.
            max_distance: Maximum distance between centroids to associate an object.
            use_appearance: Whether to use color histogram for appearance matching (reduces ID switches).
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # objectID -> dict of object info
        self.disappeared = OrderedDict()  # objectID -> number of consecutive frames not seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.use_appearance = use_appearance
        self.appearance_weight = 0.3  # Weight for appearance vs distance (0.0-1.0)

    def _compute_histogram(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Compute normalized color histogram for a bounding box region."""
        x, y, w, h = bbox
        # Clip to frame bounds
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))

        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return np.zeros(64, dtype=np.float32)

        # Compute HSV histogram (H: 16 bins, S: 4 bins = 64 total)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def _histogram_distance(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compute Bhattacharyya distance between two histograms (0=identical, 1=completely different)."""
        if hist1 is None or hist2 is None or hist1.size == 0 or hist2.size == 0:
            return 1.0
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    def register(self, centroid: np.ndarray, bbox: list, confidence: float, frame: np.ndarray = None):
        """
        Register a new object with the given centroid and bounding box.

        Args:
            centroid: (x, y) centroid of the object.
            bbox: [x, y, width, height] bounding box.
            confidence: Detection confidence.
            frame: Optional frame for appearance histogram computation.
        """
        obj_data = {
            'centroid': centroid,
            'bbox': bbox,
            'confidence': confidence,
            'first_seen': time.time(),
            'last_seen': time.time()
        }

        # Compute appearance histogram if frame provided and appearance enabled
        if self.use_appearance and frame is not None:
            obj_data['histogram'] = self._compute_histogram(frame, bbox)
        else:
            obj_data['histogram'] = None

        self.objects[self.next_object_id] = obj_data
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int):
        """
        Deregister an object by its ID.

        Args:
            object_id: The ID of the object to remove.
        """
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections: list, frame: np.ndarray = None) -> dict:
        """
        Update the tracker with new detections.

        Args:
            detections: List of detections, each detection is a dict with keys:
                        'bbox': [x, y, width, height]
                        'confidence': float
            frame: Optional current frame for appearance histogram computation.

        Returns:
            A dictionary of tracked objects, where keys are object IDs and values are dicts with:
                'centroid': (x, y)
                'bbox': [x, y, width, height]
                'confidence': float
                'first_seen': timestamp
                'last_seen': timestamp
                'histogram': appearance histogram (if use_appearance=True)
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return dict(self.objects)

        # Convert detections to centroids
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)

        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register(input_centroids[i], detections[i]['bbox'], detections[i]['confidence'], frame)
        else:
            # Get current object centroids and IDs
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[obj_id]['centroid'] for obj_id in object_ids])

            # Compute distances between each pair of object centroids and input centroids
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)

            # If using appearance, compute histogram distances too
            if self.use_appearance and frame is not None:
                # Compute histograms for new detections
                detection_histograms = []
                for detection in detections:
                    hist = self._compute_histogram(frame, detection['bbox'])
                    detection_histograms.append(hist)

                # Get existing object histograms
                object_histograms = [self.objects[obj_id].get('histogram') for obj_id in object_ids]

                # Compute appearance cost matrix
                A = np.ones((len(object_ids), len(detections)), dtype=np.float32)
                for i, obj_hist in enumerate(object_histograms):
                    for j, det_hist in enumerate(detection_histograms):
                        if obj_hist is not None and det_hist is not None:
                            A[i, j] = self._histogram_distance(obj_hist, det_hist)

                # Normalize distance matrix to [0, 1] range for combination
                D_norm = D / (self.max_distance if self.max_distance > 0 else 1.0)
                D_norm = np.clip(D_norm, 0.0, 1.0)

                # Combine distance and appearance costs
                combined_cost = (1.0 - self.appearance_weight) * D_norm + self.appearance_weight * A
            else:
                combined_cost = D / (self.max_distance if self.max_distance > 0 else 1.0)
                combined_cost = np.clip(combined_cost, 0.0, 1.0)

            # Find the smallest combined cost for each object
            rows = combined_cost.min(axis=1).argsort()
            cols = combined_cost.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if combined_cost[row, col] > 1.0:  # Normalized threshold
                    continue

                # Assign the detection to the object
                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['bbox'] = detections[col]['bbox']
                self.objects[object_id]['confidence'] = detections[col]['confidence']
                self.objects[object_id]['last_seen'] = time.time()
                self.disappeared[object_id] = 0

                # Update histogram with new detection (exponential moving average)
                if self.use_appearance and frame is not None:
                    new_hist = self._compute_histogram(frame, detections[col]['bbox'])
                    old_hist = self.objects[object_id].get('histogram')
                    if old_hist is not None:
                        # Blend old and new histograms (90% old, 10% new for stability)
                        self.objects[object_id]['histogram'] = 0.9 * old_hist + 0.1 * new_hist
                    else:
                        self.objects[object_id]['histogram'] = new_hist

                used_rows.add(row)
                used_cols.add(col)

            # Handle unassigned objects and detections
            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            # If there are more objects than detections, check for disappeared objects
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More detections than objects, register new ones
                for col in unused_cols:
                    self.register(input_centroids[col], detections[col]['bbox'], detections[col]['confidence'], frame)

        # Return a copy of the current objects (to avoid external modification)
        return dict(self.objects)