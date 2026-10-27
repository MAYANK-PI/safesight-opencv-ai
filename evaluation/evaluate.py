#!/usr/bin/env python
"""
SafeSight AI - Evaluation Framework
Measures precision, recall, F1, latency, cost, and other metrics.
"""
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.vision.analyzer import SafetyAnalyzer
from backend.agent.agent import SafetyAgent


@dataclass
class GroundTruthEvent:
    """Ground truth event for evaluation."""
    frame_number: int
    person_id: int
    event_type: str  # entry, confirmation, exit
    bbox: List[int]  # x, y, w, h
    timestamp: float


@dataclass
class DetectionResult:
    """Detection result for evaluation."""
    frame_number: int
    person_id: int
    event_type: str
    confidence: float
    bbox: List[int]


@dataclass
class EvaluationMetrics:
    """Evaluation metrics container."""
    # Detection metrics
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Tracking metrics
    id_switches: int = 0
    track_fragmentation: int = 0

    # Incident metrics
    incidents_created: int = 0
    duplicate_incidents: int = 0
    false_incidents: int = 0
    missed_incidents: int = 0

    # Latency metrics
    avg_frame_time_ms: float = 0.0
    avg_fps: float = 0.0
    total_processing_time: float = 0.0

    # Cost metrics
    estimated_cost_usd: float = 0.0

    def compute_derived(self):
        """Compute derived metrics."""
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)


class EvaluationFramework:
    """Main evaluation framework for SafeSight AI."""

    def __init__(self, model_path: str, roi_polygon: List[Tuple[int, int]]):
        self.analyzer = SafetyAnalyzer(
            model_path=model_path,
            roi_polygon=roi_polygon,
            confirmation_duration=3.0,
            confidence_threshold=0.5,
            nms_threshold=0.4
        )
        self.agent = SafetyAgent(request_duration=3.0)

    def evaluate_video(
        self,
        video_path: str,
        ground_truth: List[GroundTruthEvent]
    ) -> EvaluationMetrics:
        """Evaluate a single video against ground truth."""
        import cv2

        metrics = EvaluationMetrics()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_number = 0
        all_events = []
        all_decisions = []

        start_time = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.perf_counter()
            _, events = self.analyzer.update(frame)
            frame_time = time.perf_counter() - frame_start

            metrics.total_processing_time += frame_time

            if events:
                all_events.extend([{**e, 'frame': frame_number} for e in events])

                for event in events:
                    decisions = self.agent.update([event])
                    all_decisions.extend(decisions)

            frame_number += 1

        cap.release()

        # Compute latency metrics
        metrics.avg_frame_time_ms = (metrics.total_processing_time / frame_number) * 1000 if frame_number > 0 else 0
        metrics.avg_fps = frame_number / metrics.total_processing_time if metrics.total_processing_time > 0 else 0

        # Evaluate detection/tracking against ground truth
        self._evaluate_detections(ground_truth, all_events, metrics)

        # Evaluate incidents
        self._evaluate_incidents(all_decisions, metrics)

        # Estimate cost
        metrics.estimated_cost_usd = self._estimate_cost(metrics.total_processing_time, frame_number)

        metrics.compute_derived()
        return metrics

    def _evaluate_detections(
        self,
        ground_truth: List[GroundTruthEvent],
        detections: List[Dict],
        metrics: EvaluationMetrics
    ):
        """Evaluate detection accuracy against ground truth."""
        # Group by frame for comparison
        gt_by_frame = defaultdict(list)
        for gt in ground_truth:
            gt_by_frame[gt.frame_number].append(gt)

        det_by_frame = defaultdict(list)
        for det in detections:
            det_by_frame[det.get('frame', 0)].append(det)

        all_frames = set(gt_by_frame.keys()) | set(det_by_frame.keys())

        for frame_num in all_frames:
            gt_events = gt_by_frame.get(frame_num, [])
            det_events = det_by_frame.get(frame_num, [])

            # Match detections to ground truth (simple IoU-based matching)
            matched_gt = set()
            matched_det = set()

            for i, gt in enumerate(gt_events):
                best_iou = 0
                best_det_idx = -1
                for j, det in enumerate(det_events):
                    if j in matched_det:
                        continue
                    iou = self._compute_iou(gt.bbox, det.get('bbox', [0,0,0,0]))
                    if iou > best_iou and iou > 0.5:
                        best_iou = iou
                        best_det_idx = j

                if best_det_idx >= 0:
                    metrics.true_positives += 1
                    matched_gt.add(i)
                    matched_det.add(best_det_idx)
                else:
                    metrics.false_negatives += 1

            # Unmatched detections are false positives
            metrics.false_positives += len(det_events) - len(matched_det)

    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute IoU between two bounding boxes [x, y, w, h]."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _evaluate_incidents(
        self,
        decisions: List[Dict],
        metrics: EvaluationMetrics
    ):
        """Evaluate incident creation quality."""
        incident_creates = [d for d in decisions if d.get('next_action') == 'create_incident']
        metrics.incidents_created = len(incident_creates)

        # Check for duplicates (same person, same zone within short time)
        seen = set()
        for dec in incident_creates:
            key = (dec.get('person_id'), dec.get('parameters', {}).get('zone', 'unknown'))
            if key in seen:
                metrics.duplicate_incidents += 1
            seen.add(key)

    def _estimate_cost(self, processing_time: float, frame_count: int) -> float:
        """Estimate processing cost in USD."""
        # Rough estimates for AWS pricing (us-east-1, 2024)
        # Fargate: $0.04048 per vCPU hour, $0.004445 per GB hour
        # Assuming 2 vCPU, 4GB for 1 minute of processing
        cpu_hours = (2 * processing_time) / 3600
        memory_hours = (4 * processing_time) / 3600
        compute_cost = cpu_hours * 0.04048 + memory_hours * 0.004445

        # S3 storage: ~$0.023/GB/month, assuming 100KB per incident evidence
        storage_cost = 0.0001  # negligible per video

        # CloudWatch: $0.50 per million metrics, $0.50 per GB logs
        monitoring_cost = 0.001

        return compute_cost + storage_cost + monitoring_cost


def create_sample_ground_truth() -> List[GroundTruthEvent]:
    """Create sample ground truth for testing."""
    return [
        GroundTruthEvent(10, 1, "entry", [100, 200, 80, 180], 0.33),
        GroundTruthEvent(30, 1, "confirmation", [110, 205, 85, 185], 1.0),
        GroundTruthEvent(90, 1, "exit", [150, 210, 80, 180], 3.0),
        GroundTruthEvent(50, 2, "entry", [400, 300, 70, 170], 1.67),
        GroundTruthEvent(120, 2, "exit", [410, 305, 75, 175], 4.0),
    ]


def main():
    """Run evaluation on test videos."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SafeSight AI")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--model", type=str, default="models/yolo11n.onnx", help="Path to ONNX model")
    parser.add_argument("--roi", type=str, help="ROI polygon as JSON")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    args = parser.parse_args()

    # Default ROI
    roi = [(0, 350), (800, 350), (800, 800), (0, 800)]
    if args.roi:
        roi = json.loads(args.roi)

    # Ground truth (would normally be loaded from annotation files)
    ground_truth = create_sample_ground_truth()

    # Find test videos
    test_videos = []
    if args.video:
        test_videos = [args.video]
    else:
        # Look for test videos in common locations
        for ext in ['.mp4', '.avi', '.mov']:
            test_videos.extend(Path(".").glob(f"*test*{ext}"))
            test_videos.extend(Path(".").glob(f"*sample*{ext}"))

    if not test_videos:
        print("No test videos found. Create a test video or specify --video")
        return 1

    all_metrics = {}

    for video_path in test_videos:
        print(f"\nEvaluating: {video_path}")
        evaluator = EvaluationFramework(args.model, roi)
        metrics = evaluator.evaluate_video(str(video_path), ground_truth)

        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall: {metrics.recall:.3f}")
        print(f"  F1 Score: {metrics.f1_score:.3f}")
        print(f"  ID Switches: {metrics.id_switches}")
        print(f"  Incidents Created: {metrics.incidents_created}")
        print(f"  Duplicate Incidents: {metrics.duplicate_incidents}")
        print(f"  Avg Frame Time: {metrics.avg_frame_time_ms:.1f}ms")
        print(f"  Avg FPS: {metrics.avg_fps:.1f}")
        print(f"  Estimated Cost: ${metrics.estimated_cost_usd:.6f}")

        all_metrics[str(video_path)] = {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "id_switches": metrics.id_switches,
            "incidents_created": metrics.incidents_created,
            "duplicate_incidents": metrics.duplicate_incidents,
            "avg_frame_time_ms": metrics.avg_frame_time_ms,
            "avg_fps": metrics.avg_fps,
            "estimated_cost_usd": metrics.estimated_cost_usd
        }

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())