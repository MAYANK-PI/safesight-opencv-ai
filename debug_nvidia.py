import os
import base64
import cv2
import numpy as np
import time
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('.env', override=True)

key = os.getenv('NVIDIA_API_KEY')
model = os.getenv('VISION_MODEL')
base_url = os.getenv('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')

print(f'Model: {model}')
print(f'Base URL: {base_url}')

# Load actual video frame
cap = cv2.VideoCapture(r'E:\opencv5-pro\opencv-agentic-vision\test_data\bus_test.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video frame")
    exit(1)

# Crop a person region (using bbox from first detection)
x, y, w, h = 49, 395, 183, 510
crop = frame[y:y+h, x:x+w]
print(f'Crop shape: {crop.shape}')

ok, encoded = cv2.imencode('.jpg', crop)
image_data = base64.b64encode(encoded.tobytes()).decode('ascii')
data_url = 'data:image/jpeg;base64,' + image_data

client = OpenAI(api_key=key, base_url=base_url, timeout=120, max_retries=0)

# Build the same prompt as vision_client
prompt = '''Analyze this cropped image from an industrial safety camera.
Person ID: 0
Bounding box: [49, 395, 183, 510]
Context: {}

Task: Determine if there are any safety violations visible in this crop.
Look for: missing PPE (hard hat, safety vest, gloves, boots), unsafe posture,
proximity to hazards, equipment misuse, or any other safety concerns.

Return a JSON object with the exact schema provided. Be concise and evidence-based.
If the image is ambiguous or insufficient, indicate uncertainty.'''

schema = {
  "type": "object",
  "properties": {
    "task": {"type": "string"},
    "person_id": {"type": "integer"},
    "bbox": {"type": "array", "items": {"type": "integer"}},
    "confidence_boost": {"type": "number"},
    "risk_level_change": {"type": "string", "enum": ["lower", "same", "higher"]},
    "details": {"type": "string"},
    "violations_detected": {"type": "array", "items": {"type": "string"}},
    "ppe_status": {"type": "object", "additionalProperties": {"type": "boolean"}},
    "result": {"type": "string", "enum": ["completed", "vision_api_unavailable", "failed"]},
    "model_used": {"type": "string"},
    "inspection_error": {"type": "object"}
  },
  "required": ["task", "person_id", "bbox", "confidence_boost", "risk_level_change", "details", "violations_detected", "result"]
}

schema_str = json.dumps(schema, indent=2)
full_prompt = (
    f"{prompt}\n\n"
    f"IMPORTANT: Return a JSON OBJECT INSTANCE with actual values (e.g., {{\"task\": \"inspect_crop\", \"person_id\": 1, ...}}), "
    f"NOT the schema definition. The schema below describes the required structure:\n{schema_str}\n\n"
    f"Output ONLY the JSON object instance, no explanations, no markdown."
)

print(f'\nPrompt length: {len(full_prompt)}')
print(f'First 200 chars of prompt: {full_prompt[:200]}')

start = time.time()
try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': 'You are an industrial safety vision analyst. Output ONLY a valid JSON object instance with actual values matching the provided schema. No schema definitions, no explanations, no markdown code fences.'},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': full_prompt},
                {'type': 'image_url', 'image_url': {'url': data_url}}
            ]}
        ],
        max_tokens=500,
        temperature=0.1,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    print(f'\nSUCCESS ({elapsed:.2f}s)')
    print(f'Finish reason: {response.choices[0].finish_reason}')
    print(f'Content length: {len(content) if content else 0}')
    print(f'Raw content: {repr(content[:500])}')

    if content:
        text = content.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].strip()
        print(f'\nExtracted JSON text: {repr(text[:500])}')
        try:
            parsed = json.loads(text)
            print(f'\nParsed JSON successfully!')
            print(f'Keys: {list(parsed.keys())}')
            print(f'Result: {parsed.get("result")}')
        except json.JSONDecodeError as e:
            print(f'\nJSON parse error: {e}')
            print(f'Full text: {text}')
except Exception as exc:
    elapsed = time.time() - start
    print(f'FAILED ({elapsed:.2f}s): {type(exc).__name__}: {str(exc)[:500]}')