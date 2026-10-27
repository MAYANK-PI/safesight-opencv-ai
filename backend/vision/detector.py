import os
from typing import List

import cv2
import numpy as np


class PersonDetector:
    """Person detector for Ultralytics YOLO11 ONNX models."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.person_class_id = 0
        self.input_size = 640

    def detect(self, frame: np.ndarray) -> List[dict]:
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()

        # YOLO11 detection output is typically (1, 84, 8400):
        # 4 box coordinates + 80 class scores, without a separate objectness score.
        predictions = np.asarray(output)
        predictions = np.squeeze(predictions)

        if predictions.ndim != 2:
            raise RuntimeError(
                f"Unexpected YOLO ONNX output shape: {np.asarray(output).shape}"
            )

        # Convert (84, 8400) to (8400, 84) when necessary.
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        if predictions.shape[1] < 5:
            raise RuntimeError(
                f"Unexpected YOLO prediction dimensions: {predictions.shape}"
            )

        boxes = []
        confidences = []

        scale_x = width / self.input_size
        scale_y = height / self.input_size

        for detection in predictions:
            class_scores = detection[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if class_id != self.person_class_id:
                continue
            if not np.isfinite(confidence) or confidence < self.confidence_threshold:
                continue

            cx, cy, box_w, box_h = map(float, detection[:4])

            # YOLO11 coordinates are center_x, center_y, width, height.
            x = int((cx - box_w / 2) * scale_x)
            y = int((cy - box_h / 2) * scale_y)
            box_w = int(box_w * scale_x)
            box_h = int(box_h * scale_y)

            # Clip boxes to the source frame.
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            box_w = max(0, min(box_w, width - x))
            box_h = max(0, min(box_h, height - y))

            if box_w == 0 or box_h == 0:
                continue

            boxes.append([x, y, box_w, box_h])
            confidences.append(confidence)

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )

        detections = []
        if len(indices) > 0:
            for index in np.asarray(indices).reshape(-1):
                detections.append(
                    {
                        "bbox": boxes[int(index)],
                        "confidence": confidences[int(index)],
                        "class_id": self.person_class_id,
                    }
                )

        return detections



