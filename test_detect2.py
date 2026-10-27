import cv2
import numpy as np
from backend.vision.detector import PersonDetector

detector = PersonDetector(
    model_path='models/yolo11n.onnx',
    confidence_threshold=0.1,
    nms_threshold=0.4,
)

img = cv2.imread('evidence-test.jpg')
print(f'Evidence image shape: {img.shape}')
detections = detector.detect(img)
print(f'Detections on evidence image: {len(detections)}')
for d in detections:
    print(f'  Bbox: {d["bbox"]}, Conf: {d["confidence"]:.3f}')