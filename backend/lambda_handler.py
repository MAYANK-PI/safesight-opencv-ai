"""
Lambda handler for SafeSight AI.
Handles API Gateway events for video analysis.
Note: Lambda has 15-minute timeout, suitable for short videos only.
"""
import json
import os
import tempfile
import base64
import boto3
from typing import Dict, Any

# Import the main processing function
from backend.main import _process_video, validate_video_file
from backend.vision.analyzer import SafetyAnalyzer

# Global analyzer (initialized on cold start)
MODEL_PATH = os.getenv("MODEL_PATH", "/var/task/models/yolo11n.onnx")
ROI_POLYGON = [(0, 350), (800, 350), (800, 800), (0, 800)]
CONFIRMATION_DURATION = 3.0

analyzer = None


def _get_analyzer() -> SafetyAnalyzer:
    """Get or create the safety analyzer (singleton for Lambda reuse)."""
    global analyzer
    if analyzer is None:
        analyzer = SafetyAnalyzer(
            model_path=MODEL_PATH,
            roi_polygon=ROI_POLYGON,
            confirmation_duration=CONFIRMATION_DURATION,
            confidence_threshold=0.5,
            nms_threshold=0.4,
        )
    return analyzer


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for API Gateway events.

    Expected event format:
    {
        "body": "base64-encoded video file",
        "isBase64Encoded": true,
        "queryStringParameters": {
            "roi_polygon": "[[100, 100], [500, 100], [500, 400], [100, 400]]"
        }
    }

    Returns:
        API Gateway response format
    """
    try:
        # Parse request
        body = event.get("body", "")
        is_base64 = event.get("isBase64Encoded", False)
        query_params = event.get("queryStringParameters") or {}

        # Parse optional ROI polygon
        roi_polygon = None
        if "roi_polygon" in query_params:
            try:
                roi_polygon = json.loads(query_params["roi_polygon"])
            except json.JSONDecodeError:
                pass

        # Decode video file
        if is_base64:
            video_data = base64.b64decode(body)
        else:
            video_data = body.encode() if isinstance(body, str) else body

        if not video_data:
            return _error_response(400, "No video data provided")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_data)
            temp_path = tmp.name

        try:
            # Validate video
            video_meta = validate_video_file(temp_path)

            # Process video
            analyzer = _get_analyzer()
            response = _process_video(temp_path, analyzer, roi_polygon)

            # Add video metadata
            response_dict = response.model_dump(exclude_none=False)
            response_dict["video_metadata"] = video_meta

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps(response_dict, default=str)
            }

        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    except Exception as e:
        return _error_response(500, f"Processing failed: {str(e)}")


def _error_response(status_code: int, message: str) -> Dict[str, Any]:
    """Create error response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({"error": message})
    }


# For local testing
if __name__ == "__main__":
    # Mock event for testing
    test_event = {
        "body": "",
        "isBase64Encoded": False,
        "queryStringParameters": {}
    }
    result = handler(test_event, None)
    print(json.dumps(result, indent=2))