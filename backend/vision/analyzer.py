import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any
from .detector import PersonDetector
from .tracker import CentroidTracker
from .roi import RegionOfInterest
from .motion import MotionAnalyzer

class SafetyAnalyzer:
    def __init__(self,
                 model_path: str,
                 roi_polygon: List[Tuple[int, int]],
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 confirmation_duration: float = 3.0,
                 max_disappeared: int = 30,
                 max_distance: int = 50,
                 roi_hysteresis_frames: int = 3):
        """
        Initialize the safety analyzer.

        Args:
            model_path: Path to the ONNX model for person detection.
            roi_polygon: List of (x, y) tuples defining the restricted zone polygon.
            confidence_threshold: Minimum confidence for detection.
            nms_threshold: Non-maximum suppression threshold.
            confirmation_duration: Seconds a person must remain in ROI to be confirmed.
            max_disappeared: Max frames an object can disappear before tracker removes it.
            max_distance: Max distance for centroid association in tracker.
            roi_hysteresis_frames: Number of consecutive frames inside/outside required before state transition.
        """
        self.detector = PersonDetector(model_path, confidence_threshold, nms_threshold)
        self.tracker = CentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)
        self.roi = RegionOfInterest(roi_polygon)
        self.motion_analyzer = MotionAnalyzer()
        self.confirmation_duration = confirmation_duration
        self.roi_hysteresis_frames = roi_hysteresis_frames

        # State for each tracked object: objectID -> dict of state info
        # Status can be: 'OUTSIDE', 'INSIDE_POTENTIAL', 'INSIDE_CONFIRMED'
        self.object_state: Dict[int, dict] = {}

        # Inference metrics
        self.metrics = {
            "total_frames": 0,
            "total_detections": 0,
            "total_events": 0,
            "total_processing_time": 0.0,
            "frame_times": []
        }

    def set_roi(self, roi_polygon: List[Tuple[int, int]]) -> None:
        """Update the ROI polygon for this analysis session."""
        self.roi = RegionOfInterest(roi_polygon)
        # Reset object states since ROI changed
        self.object_state.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Return current inference metrics."""
        metrics = self.metrics.copy()
        if metrics["total_frames"] > 0:
            metrics["avg_frame_time"] = metrics["total_processing_time"] / metrics["total_frames"]
            metrics["avg_fps"] = 1.0 / metrics["avg_frame_time"] if metrics["avg_frame_time"] > 0 else 0.0
        else:
            metrics["avg_frame_time"] = 0.0
            metrics["avg_fps"] = 0.0
        return metrics

    def reset_metrics(self) -> None:
        """Reset inference metrics."""
        self.metrics = {
            "total_frames": 0,
            "total_detections": 0,
            "total_events": 0,
            "total_processing_time": 0.0,
            "frame_times": []
        }

    def update(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a frame and return the annotated frame and list of events.

        Args:
            frame: Input image (BGR format).

        Returns:
            Tuple of (annotated_frame, events)
            annotated_frame: The frame with visualizations drawn.
            events: List of event dictionaries. Each event has at least:
                'event_type': str (e.g., 'restricted_zone_entry', 'restricted_zone_exit')
                'person_id': int
                'confidence': float
                'zone': str (we'll use a fixed zone name for now)
                'timestamp': float (seconds since epoch)
                'duration': float (how long the person has been in the zone, if applicable)
                'status': str (one of 'potential', 'confirmed', 'resolved')
        """
        frame_start = time.perf_counter()

        # Make a copy of the frame for annotation
        annotated = frame.copy()

        # Draw ROI on the annotated frame
        annotated = self.roi.draw(annotated, color=(0, 255, 0), thickness=2)

        # Step 1: Detect persons in the frame
        detections = self.detector.detect(frame)

        # Step 2: Update tracker with detections (pass frame for appearance matching)
        tracked_objects = self.tracker.update(detections, frame)

        # Step 3: Update motion information
        tracked_objects_with_motion = self.motion_analyzer.update(tracked_objects)

        # Step 4: Update state for each tracked object and generate events
        events = []
        current_time = time.time()

        for object_id, obj_data in tracked_objects_with_motion.items():
            centroid = obj_data['centroid']
            bbox = obj_data['bbox']
            confidence = obj_data['confidence']

            # Check if the object's centroid is inside the ROI
            # Convert centroid to tuple for ROI.is_inside
            inside = self.roi.is_inside(tuple(centroid))

            # Get or initialize state for this object
            if object_id not in self.object_state:
                self.object_state[object_id] = {
                    'status': 'OUTSIDE',
                    'entry_time': None,
                    'confirmed_time': None,
                    'inside_count': 0,
                    'outside_count': 0,
                }
            state = self.object_state[object_id]

            if inside:
                # Object is inside the ROI
                state['inside_count'] += 1
                state['outside_count'] = 0

                # Only transition if consistently inside for hysteresis frames
                if state['status'] == 'OUTSIDE' and state['inside_count'] >= self.roi_hysteresis_frames:
                    # Just entered: transition to INSIDE_POTENTIAL
                    state['status'] = 'INSIDE_POTENTIAL'
                    state['entry_time'] = current_time
                    # Generate a potential entry event
                    events.append({
                        'event_type': 'restricted_zone_entry',
                        'person_id': object_id,
                        'confidence': confidence,
                        'zone': 'restricted_zone_1',
                        'timestamp': current_time,
                        'duration': 0.0,
                        'status': 'potential',
                        'bbox': bbox
                    })
                elif state['status'] == 'INSIDE_POTENTIAL':
                    # Check if we have been inside long enough to confirm
                    duration_inside = current_time - state['entry_time']
                    if duration_inside >= self.confirmation_duration:
                        # Transition to INSIDE_CONFIRMED
                        state['status'] = 'INSIDE_CONFIRMED'
                        state['confirmed_time'] = current_time
                        events.append({
                            'event_type': 'restricted_zone_confirmation',
                            'person_id': object_id,
                            'confidence': confidence,
                            'zone': 'restricted_zone_1',
                            'timestamp': current_time,
                            'duration': duration_inside,
                            'status': 'confirmed',
                            'bbox': bbox
                        })
                # If state is INSIDE_CONFIRMED, we do nothing (remain confirmed)
            else:
                # Object is outside the ROI
                state['outside_count'] += 1
                state['inside_count'] = 0

                # Only transition if consistently outside for hysteresis frames
                if state['status'] != 'OUTSIDE' and state['outside_count'] >= self.roi_hysteresis_frames:
                    # Object has left the ROI
                    duration_inside = 0.0
                    if state['entry_time'] is not None:
                        duration_inside = current_time - state['entry_time']

                    # Only generate exit event if duration is meaningful (avoid very short false events)
                    if duration_inside >= 0.5:  # Minimum 0.5 seconds inside to count as real event
                        events.append({
                            'event_type': 'restricted_zone_exit',
                            'person_id': object_id,
                            'confidence': confidence,
                            'zone': 'restricted_zone_1',
                            'timestamp': current_time,
                            'duration': duration_inside,
                            'status': 'resolved',
                            'bbox': bbox
                        })
                    # Reset state
                    state['status'] = 'OUTSIDE'
                    state['entry_time'] = None
                    state['confirmed_time'] = None

        # Step 5: Annotate tracked objects on the frame (optional, for debugging)
        for object_id, obj_data in tracked_objects_with_motion.items():
            bbox = obj_data['bbox']
            confidence = obj_data['confidence']
            x, y, w, h = bbox
            # Draw bounding box
            color = (0, 255, 0)  # Green for tracked
            if object_id in self.object_state:
                state = self.object_state[object_id]['status']
                if state == 'INSIDE_POTENTIAL':
                    color = (0, 255, 255)  # Yellow for potential
                elif state == 'INSIDE_CONFIRMED':
                    color = (0, 0, 255)    # Red for confirmed
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            # Draw ID and confidence
            label = f"ID: {object_id} {confidence:.2f}"
            cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update metrics
        frame_time = time.perf_counter() - frame_start
        self.metrics["total_frames"] += 1
        self.metrics["total_detections"] += len(detections)
        self.metrics["total_events"] += len(events)
        self.metrics["total_processing_time"] += frame_time
        self.metrics["frame_times"].append(frame_time)

        return annotated, events

    def reset(self):
        """Reset the analyzer state for a new video analysis."""
        self.tracker = CentroidTracker(
            max_disappeared=self.tracker.max_disappeared,
            max_distance=self.tracker.max_distance
        )
        self.object_state.clear()
        self.reset_metrics()