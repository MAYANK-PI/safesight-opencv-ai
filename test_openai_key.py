

import os
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError

load_dotenv(".env")

if not os.getenv("OPENAI_API_KEY"):
    print("FAILED: OPENAI_API_KEY is missing.")
    raise SystemExit(1)

model = os.getenv("VISION_MODEL", "").strip()

if not model:
    print("FAILED: VISION_MODEL is not configured in .env.")
    raise SystemExit(1)

try:
    client = OpenAI(timeout=30.0, max_retries=0)
    response = client.responses.create(
        model=model,
        input="Reply with exactly: API connection successful.",
        max_output_tokens=30,
    )
    print("SUCCESS: API key and model request worked.")
    print("Model:", model)
    print("Response:", response.output_text)

except APIStatusError as exc:
    print("HTTP status:", exc.status_code)
    # Print the API error details, but never request or print credentials.
    try:
        data = exc.response.json()
        error = data.get("error", {})
        print("Error code:", error.get("code"))
        print("Error type:", error.get("type"))
        print("Error message:", error.get("message"))
    except Exception:
        print("Could not parse API error details.")

except APIConnectionError:
    print("FAILED: Network connection to the API failed.")

except Exception as exc:
    print("FAILED:", type(exc).__name__)
