import os
import base64
import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env", override=True)

key = os.getenv("NVIDIA_API_KEY")
model = os.getenv("VISION_MODEL")
base_url = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
)

if not key:
    raise SystemExit("NVIDIA_API_KEY missing from .env")

if not model or "your_" in model.lower():
    raise SystemExit("VISION_MODEL is missing or still a placeholder")

# Generate a simple synthetic image for the vision test.
img = np.full((256, 256, 3), 255, dtype=np.uint8)
cv2.rectangle(img, (60, 60), (190, 190), (0, 0, 255), -1)

ok, encoded = cv2.imencode(".jpg", img)
if not ok:
    raise SystemExit("Could not create test image")

image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
data_url = "data:image/jpeg;base64," + image_data

client = OpenAI(
    api_key=key,
    base_url=base_url,
    timeout=30,
    max_retries=0,
)

print("Provider: NVIDIA")
print("Model:", model)
print("Sending synthetic image...")

try:
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the visible shape and its color."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                }
            ]
        }],
        max_tokens=100,
    )
    print("Vision request: SUCCESS")
    print("Response:", response.choices[0].message.content)

except Exception as exc:
    print("Vision request: FAILED")
    print("Error type:", type(exc).__name__)
    print("Details:", str(exc)[:700])
