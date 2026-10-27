import cv2
import numpy as np
from backend.vision.detector import PersonDetector

# Create detector
detector = PersonDetector(
    model_path='models/yolo11n.onnx',
    confidence_threshold=0.5,
    nms_threshold=0.4,
)

# Process video
cap = cv2.VideoCapture('test_video.mp4')
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    detections = detector.detect(frame)
    if detections:
        print(f'Frame {frame_count}: {len(detections)} detections')
        for d in detections:
            bbox = d['bbox']
            conf = d['confidence']
            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2
            print(f'  Bbox: {bbox}, Conf: {conf:.3f}, Centroid: ({cx}, {cy})')

cap.release()
print(f'Total frames: {frame_count}')