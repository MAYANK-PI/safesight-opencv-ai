import cv2
import time
from backend.vision.analyzer import SafetyAnalyzer
from backend.agent.agent import SafetyAgent

# Create analyzer
roi_polygon = [(0, 350), (800, 350), (800, 800), (0, 800)]
analyzer = SafetyAnalyzer(
    model_path='models/yolo11n.onnx',
    roi_polygon=roi_polygon,
    confirmation_duration=3.0,
    confidence_threshold=0.5,
    nms_threshold=0.4,
)

# Create agent
agent = SafetyAgent(request_duration=3.0)

# Process video
cap = cv2.VideoCapture('test_video.mp4')
frame_count = 0
all_events = []
all_decisions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    annotated, events = analyzer.update(frame)
    if events:
        all_events.extend(events)
        decisions = agent.update(events, frame=annotated, frame_number=frame_count)
        all_decisions.extend(decisions)
        for d in decisions:
            print(f'Frame {frame_count}: Decision for person {d["person_id"]}: {d["decision"]} risk={d["risk_level"]} reason={d["reason"]}')

cap.release()
print(f'Total frames: {frame_count}')
print(f'Total events: {len(all_events)}')
print(f'Total decisions: {len(all_decisions)}')