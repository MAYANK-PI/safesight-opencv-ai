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
print(f'Timeout: {os.getenv("VISION_TIMEOUT_SECONDS")}')

# Simple test image
img = np.full((256, 256, 3), 255, dtype=np.uint8)
cv2.rectangle(img, (60, 60), (190, 190), (0, 0, 255), -1)
ok, encoded = cv2.imencode('.jpg', img)
image_data = base64.b64encode(encoded.tobytes()).decode('ascii')
data_url = 'data:image/jpeg;base64,' + image_data

client = OpenAI(api_key=key, base_url=base_url, timeout=120, max_retries=0)

# Test 1: Plain text (baseline)
print('\n=== Test 1: Plain text ===')
start = time.time()
try:
    response = client.chat.completions.create(
        model=model,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Describe the image in one sentence.'},
                {'type': 'image_url', 'image_url': {'url': data_url}}
            ]
        }],
        max_tokens=50,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    print(f'SUCCESS ({elapsed:.2f}s): {content[:200]}')
    print(f'Finish reason: {response.choices[0].finish_reason}')
except Exception as exc:
    elapsed = time.time() - start
    print(f'FAILED ({elapsed:.2f}s): {type(exc).__name__}: {str(exc)[:500]}')

# Test 2: JSON schema prompt with IMPROVED prompt (explicit about JSON instance)
print('\n=== Test 2: JSON schema prompt (IMPROVED) ===')
prompt = '''Analyze this cropped image from an industrial safety camera.
Person ID: 1
Bounding box: [100, 100, 50, 50]
Context: {}

Task: Determine if there are any safety violations visible in this crop.
Look for: missing PPE (hard hat, safety vest, gloves, boots), unsafe posture,
proximity to hazards, equipment misuse, or any other safety concerns.

Return a JSON object with the exact schema provided. Be concise and evidence-based.
If the image is ambiguous or insufficient, indicate uncertainty.'''

# Add schema instruction EXPLICITLY asking for JSON INSTANCE
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
    print(f'SUCCESS ({elapsed:.2f}s)')
    print(f'Finish reason: {response.choices[0].finish_reason}')
    print(f'Content length: {len(content)}')
    print(f'First 500 chars: {content[:500]}')

    # Try to parse JSON
    if content:
        # Extract JSON from potential markdown
        text = content.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].strip()
        try:
            parsed = json.loads(text)
            print(f'\nParsed JSON successfully!')
            print(f'Keys: {list(parsed.keys())}')
            print(f'Result: {parsed.get("result")}')
            print(f'Task: {parsed.get("task")}')
            print(f'Person ID: {parsed.get("person_id")}')
            print(f'Violations: {parsed.get("violations_detected")}')
            print(f'PPE Status: {parsed.get("ppe_status")}')
        except json.JSONDecodeError as e:
            print(f'\nJSON parse error: {e}')
except Exception as exc:
    elapsed = time.time() - start
    print(f'FAILED ({elapsed:.2f}s): {type(exc).__name__}: {str(exc)[:500]}')