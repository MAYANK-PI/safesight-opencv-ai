"""
Optional Multimodal Vision API Client for SafeSight AI.
Uses the official OpenAI Python SDK with Responses API for image-based verification.
Supports OpenAI GPT-4V, Anthropic Claude 3.5 Sonnet, and Azure OpenAI.
All external API calls are optional - system works without them.
"""
import os
import json
import time
import base64
import logging
import asyncio
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class VisionProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    NVIDIA = "nvidia"


class VisionTask(str, Enum):
    INSPECT_CROP = "inspect_crop"
    VERIFY_PPE = "verify_ppe"
    ASSESS_SCENE = "assess_scene"


# Structured output schemas for validation
class InspectCropResult(BaseModel):
    """Result of detailed crop inspection."""
    task: Literal["inspect_crop"] = "inspect_crop"
    person_id: int
    bbox: List[int]
    confidence_boost: float = Field(ge=-0.5, le=0.5, default=0.0)
    risk_level_change: Optional[Literal["lower", "same", "higher"]] = "same"
    details: str
    violations_detected: List[str] = Field(default_factory=list)
    ppe_status: Optional[Dict[str, Optional[bool]]] = None
    result: Literal["completed", "vision_api_unavailable", "failed"] = "completed"
    model_used: Optional[str] = None
    inspection_error: Optional[Dict[str, Any]] = None


class VerifyPPEResult(BaseModel):
    """Result of PPE verification."""
    task: Literal["verify_ppe"] = "verify_ppe"
    bbox: List[int]
    ppe_detected: Dict[str, Optional[bool]] = Field(default_factory=dict)
    compliance: Literal["compliant", "non_compliant", "uncertain"] = "uncertain"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    details: str
    missing_items: List[str] = Field(default_factory=list)
    result: Literal["completed", "vision_api_unavailable", "failed"] = "completed"
    model_used: Optional[str] = None
    inspection_error: Optional[Dict[str, Any]] = None


class AssessSceneResult(BaseModel):
    """Result of general scene assessment."""
    task: Literal["assess_scene"] = "assess_scene"
    risk_level: Literal["low", "medium", "high", "critical"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: List[str] = Field(default_factory=list)
    recommended_action: Literal["continue_tracking", "create_incident", "request_human_approval", "no_action"]
    details: str
    result: Literal["completed", "vision_api_unavailable", "failed"] = "completed"
    model_used: Optional[str] = None
    inspection_error: Optional[Dict[str, Any]] = None


VisionResult = InspectCropResult | VerifyPPEResult | AssessSceneResult


@dataclass
class VisionClientConfig:
    """Configuration for vision API client."""
    provider: VisionProvider = VisionProvider.OPENAI
    api_key: Optional[str] = None
    api_base: Optional[str] = None  # For Azure OpenAI or NVIDIA
    api_version: Optional[str] = None  # For Azure OpenAI
    deployment_name: Optional[str] = None  # For Azure OpenAI
    model: str = "gpt-4o-mini"  # or "claude-3-5-sonnet-20241022" or "meta/llama-3.2-11b-vision-instruct"
    max_tokens: int = 500
    temperature: float = 0.1
    timeout_seconds: float = 30.0
    max_retries: int = 2
    max_calls_per_event: int = 3
    max_calls_per_video: int = 10
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "VisionClientConfig":
        """Load configuration from environment variables."""
        provider_str = os.getenv("VISION_PROVIDER", "openai").lower()
        try:
            provider = VisionProvider(provider_str)
        except ValueError:
            provider = VisionProvider.OPENAI

        # Determine API key and base URL based on provider
        api_key = None
        api_base = None
        if provider == VisionProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == VisionProvider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == VisionProvider.AZURE_OPENAI:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        elif provider == VisionProvider.NVIDIA:
            api_key = os.getenv("NVIDIA_API_KEY")
            api_base = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

        return cls(
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            model=os.getenv("VISION_MODEL", "gpt-4o-mini"),
            max_tokens=int(os.getenv("VISION_MAX_TOKENS", "500")),
            temperature=float(os.getenv("VISION_TEMPERATURE", "0.1")),
            timeout_seconds=float(os.getenv("VISION_TIMEOUT_SECONDS", "30.0")),
            max_retries=int(os.getenv("VISION_MAX_RETRIES", "2")),
            max_calls_per_event=int(os.getenv("VISION_MAX_CALLS_PER_EVENT", "3")),
            max_calls_per_video=int(os.getenv("VISION_MAX_CALLS_PER_VIDEO", "10")),
            enabled=os.getenv("VISION_ENABLED", "true").lower() == "true" and api_key is not None,
        )

    def check_config(self) -> Dict[str, Any]:
        """Check configuration status without exposing secrets.

        Returns:
            Dict with configuration status: provider, model, enabled, has_api_key, warnings
        """
        has_key = bool(self.api_key and self.api_key.strip())
        warnings = []

        if not has_key:
            warnings.append("No API key configured - vision API will be disabled")
        if self.enabled and not has_key:
            warnings.append("VISION_ENABLED=true but no API key found")
        if self.timeout_seconds < 5:
            warnings.append("Timeout may be too short for reliable API calls")
        if self.max_calls_per_video < 1:
            warnings.append("max_calls_per_video should be at least 1")

        return {
            "provider": self.provider.value,
            "model": self.model,
            "enabled": self.enabled,
            "has_api_key": has_key,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_calls_per_event": self.max_calls_per_event,
            "max_calls_per_video": self.max_calls_per_video,
            "warnings": warnings,
            "configured": has_key and self.enabled,
        }


class VisionAPIError(Exception):
    """Custom exception for vision API errors."""
    def __init__(self, message: str, error_type: str = "api_error", retryable: bool = False):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


class VisionClient:
    """
    Client for multimodal vision API calls using official OpenAI SDK.
    Gracefully degrades when API is unavailable or disabled.
    """

    def __init__(self, config: Optional[VisionClientConfig] = None):
        self.config = config or VisionClientConfig.from_env()
        self.call_counts: Dict[str, int] = {}  # Track calls per event/task
        self.video_call_count: int = 0  # Track total calls per video session
        self._openai_client = None
        self._initialized = False

    def _get_openai_client(self):
        """Get or create the official OpenAI client (also used for NVIDIA)."""
        if self._openai_client is None and self.config.enabled and self.config.api_key:
            try:
                from openai import OpenAI
                if self.config.provider == VisionProvider.AZURE_OPENAI:
                    from openai import AzureOpenAI
                    self._openai_client = AzureOpenAI(
                        api_key=self.config.api_key,
                        azure_endpoint=self.config.api_base,
                        api_version=self.config.api_version,
                        timeout=self.config.timeout_seconds,
                    )
                elif self.config.provider == VisionProvider.NVIDIA:
                    # NVIDIA uses OpenAI-compatible API with custom base URL
                    self._openai_client = OpenAI(
                        api_key=self.config.api_key,
                        base_url=self.config.api_base,
                        timeout=self.config.timeout_seconds,
                    )
                else:
                    self._openai_client = OpenAI(
                        api_key=self.config.api_key,
                        timeout=self.config.timeout_seconds,
                    )
            except ImportError:
                logger.warning("OpenAI SDK not installed. Vision API will be unavailable.")
                self._openai_client = None
        return self._openai_client

    def _check_rate_limit(self, event_id: str, task: VisionTask) -> bool:
        """Check if we've exceeded max calls for this event."""
        key = f"{event_id}:{task.value}"
        count = self.call_counts.get(key, 0)
        if count >= self.config.max_calls_per_event:
            return False
        # Also check video-level limit
        if self.video_call_count >= self.config.max_calls_per_video:
            return False
        return True

    def _increment_call_count(self, event_id: str, task: VisionTask):
        """Increment call counter for event/task and video."""
        key = f"{event_id}:{task.value}"
        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        self.video_call_count += 1

    def reset_video_counter(self):
        """Reset the per-video call counter (call at start of new video)."""
        self.video_call_count = 0
        self.call_counts.clear()

    def check_config(self) -> Dict[str, Any]:
        """Check configuration status without exposing secrets."""
        return self.config.check_config()

    def _encode_image_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64."""
        return base64.b64encode(image_bytes).decode("utf-8")

    def _build_inspect_crop_prompt(self, person_id: int, bbox: List[int], context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for crop inspection."""
        return f"""Analyze this cropped image from an industrial safety camera.
Person ID: {person_id}
Bounding box: {bbox}
Context: {json.dumps(context or {})}

Task: Determine if there are any safety violations visible in this crop.
Look for: missing PPE (hard hat, safety vest, gloves, boots), unsafe posture,
proximity to hazards, equipment misuse, or any other safety concerns.

Return a JSON object with the exact schema provided. Be concise and evidence-based.
If the image is ambiguous or insufficient, indicate uncertainty.

IMPORTANT: You MUST include a "ppe_status" object with values for ALL of these PPE items:
- hard_hat: true if clearly visible and properly worn, false if clearly missing/not worn, null if cannot be assessed (outside crop view, occluded, too blurry)
- safety_vest: true if clearly visible and properly worn, false if clearly missing/not worn, null if cannot be assessed
- gloves: true if clearly visible and properly worn, false if clearly missing/not worn, null if cannot be assessed
- boots: true if clearly visible and properly worn, false if clearly missing/not worn, null if cannot be assessed

CRITICAL CONSISTENCY RULE: The ppe_status values MUST be consistent with violations_detected and details:
- If violations_detected contains "missing_hard_hat", "missing_safety_vest", "missing_gloves", or "missing_boots" (or "missing_ppe" with specifics in details), the corresponding ppe_status MUST be false (NOT null)
- If details text says an item is "not visible", "not wearing", "missing", or "absent", the corresponding ppe_status MUST be false
- Use null ONLY when the item is truly outside the crop view, heavily occluded, or too blurry to assess
- If you can see enough to determine the item is missing, use false, not null

NEVER omit the ppe_status field. Return exact JSON instance matching the schema."""

    def _build_verify_ppe_prompt(self, bbox: List[int], required_ppe: Optional[List[str]] = None) -> str:
        """Build prompt for PPE verification."""
        required = required_ppe or ["hard_hat", "safety_vest", "gloves", "boots"]
        return f"""Analyze this image region for PPE compliance.
Bounding box: {bbox}
Required PPE items: {', '.join(required)}

For each item, determine if it's clearly visible and properly worn.
IMPORTANT: Return the following for each PPE item in ppe_detected:
- true if the item is clearly visible and properly worn
- false if the item is clearly missing or not properly worn
- null if the item cannot be assessed (outside crop view, occluded, too blurry, ambiguous)

CRITICAL CONSISTENCY RULE: The ppe_detected values MUST be consistent with missing_items and details:
- If missing_items contains an item, the corresponding ppe_detected MUST be false (NOT null)
- If details text says an item is "not visible", "not wearing", "missing", or "absent", the corresponding ppe_detected MUST be false
- Use null ONLY when the item is truly outside the crop view, heavily occluded, or too blurry to assess

If an item is clearly visible and properly worn, return true.
If an item is clearly missing or not properly worn, return false.
If unclear/uncertain/obscured, return null (do NOT guess - be conservative).
Return a JSON object with the exact schema provided."""

    def _build_assess_scene_prompt(self, event_context: Dict[str, Any]) -> str:
        """Build prompt for scene assessment."""
        return f"""Analyze this industrial safety camera frame for risk assessment.
Event context: {json.dumps(event_context)}

Determine the overall risk level and recommended action.
Consider: person count, zone violations, PPE compliance, hazardous materials,
equipment status, environmental conditions.

Return a JSON object with the exact schema provided. Be conservative and evidence-based."""

    def _get_response_schema(self, result_class: type) -> Dict[str, Any]:
        """Get JSON schema for structured output."""
        return result_class.model_json_schema()

    async def _call_openai(
        self,
        images: List[str],
        prompt: str,
        response_schema: Dict[str, Any],
        event_id: str,
        task: VisionTask,
    ) -> Dict[str, Any]:
        """Call OpenAI Responses API with structured output."""
        client = self._get_openai_client()
        if client is None:
            raise VisionAPIError("OpenAI SDK not available or not configured", "not_configured", retryable=False)

        # Build input for Responses API
        content = [{"type": "input_text", "text": prompt}]
        for img_b64 in images:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{img_b64}",
                "detail": "high"
            })

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # Use the Responses API with structured output
                response = client.responses.create(
                    model=self.config.deployment_name if self.config.provider == VisionProvider.AZURE_OPENAI else self.config.model,
                    input=[
                        {"role": "system", "content": "You are an industrial safety vision analyst. Respond only with valid JSON matching the provided schema."},
                        {"role": "user", "content": content}
                    ],
                    text={"format": {"type": "json_schema", "name": "vision_result", "schema": response_schema, "strict": True}},
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )

                # Extract JSON from response
                content_text = response.output_text
                return json.loads(content_text)

            except Exception as e:
                error_str = str(e)
                logger.warning(f"OpenAI Responses API call failed (attempt {attempt + 1}): {error_str}")

                # Categorize errors
                if "401" in error_str or "authentication" in error_str.lower() or "unauthorized" in error_str.lower():
                    last_error = VisionAPIError(f"Authentication failed: {e}", "auth_error", retryable=False)
                    break
                elif "429" in error_str or "rate limit" in error_str.lower() or "rate_limit" in error_str.lower():
                    last_error = VisionAPIError(f"Rate limited: {e}", "rate_limit", retryable=True)
                elif "insufficient_quota" in error_str.lower() or "credit_balance_exhausted" in error_str.lower() or "billing" in error_str.lower():
                    last_error = VisionAPIError(f"Insufficient quota: {e}", "quota_exhausted", retryable=False)
                    break
                elif "timeout" in error_str.lower():
                    last_error = VisionAPIError(f"Request timeout: {e}", "timeout", retryable=True)
                elif "500" in error_str or "502" in error_str or "503" in error_str:
                    last_error = VisionAPIError(f"Server error: {e}", "server_error", retryable=True)
                elif "model" in error_str.lower() and ("not found" in error_str.lower() or "invalid" in error_str.lower()):
                    last_error = VisionAPIError(f"Unsupported model: {e}", "model_error", retryable=False)
                    break
                elif "json" in error_str.lower() or "parse" in error_str.lower() or "schema" in error_str.lower():
                    last_error = VisionAPIError(f"Invalid response format: {e}", "parse_error", retryable=False)
                    break
                else:
                    last_error = VisionAPIError(f"Unexpected error: {e}", "unknown", retryable=False)
                    break

            if attempt < self.config.max_retries and last_error and last_error.retryable:
                wait = 2 ** attempt
                logger.warning(f"Vision API call failed ({last_error.error_type}), retrying in {wait}s...")
                await asyncio.sleep(wait)

        raise last_error or VisionAPIError("Max retries exceeded", "max_retries")

    async def _call_nvidia(
        self,
        images: List[str],
        prompt: str,
        response_schema: Dict[str, Any],
        event_id: str,
        task: VisionTask,
    ) -> Dict[str, Any]:
        """Call NVIDIA API using OpenAI-compatible Chat Completions API with JSON schema."""
        client = self._get_openai_client()
        if client is None:
            raise VisionAPIError("OpenAI SDK not available or not configured", "not_configured", retryable=False)

        # Build messages for Chat Completions API
        content = [{"type": "text", "text": prompt}]
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}
            })

        # Add schema instruction to prompt for NVIDIA (since it may not support structured output natively)
        # Explicitly ask for a JSON INSTANCE (with actual values), not the schema definition
        schema_str = json.dumps(response_schema, indent=2)
        full_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Return a JSON OBJECT INSTANCE with actual values (e.g., {{\"task\": \"inspect_crop\", \"person_id\": 1, ...}}), "
            f"NOT the schema definition. The schema below describes the required structure:\n{schema_str}\n\n"
            f"Output ONLY the JSON object instance, no explanations, no markdown."
        )

        # Build messages with full_prompt that includes schema instructions
        messages = [
            {"role": "system", "content": "You are an industrial safety vision analyst. Output ONLY a valid JSON object instance with actual values matching the provided schema. No schema definitions, no explanations, no markdown code fences."},
            {"role": "user", "content": full_prompt}
        ]
        # Add images to the user message
        for img_b64 in images:
            messages[1]["content"] = [
                {"type": "text", "text": full_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}}
            ]
            break  # Only one image expected for inspect_crop

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # Use Chat Completions API (NVIDIA supports this, not Responses API)
                # NOTE: Do NOT use response_format={"type": "json_object"} as it causes timeouts with this model
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    # response_format={"type": "json_object"} removed - causes timeouts with meta/llama-3.2-11b-vision-instruct
                )

                # Safely extract response content
                if not response.choices or not response.choices[0].message:
                    raise VisionAPIError("Empty response from NVIDIA API", "empty_response", retryable=True)

                content_text = response.choices[0].message.content
                if not content_text or not content_text.strip():
                    raise VisionAPIError("Empty content in NVIDIA API response", "empty_content", retryable=True)

                # Extract JSON from potential markdown code blocks
                content_text = self._extract_json_from_response(content_text)

                # Parse and validate JSON
                try:
                    parsed = json.loads(content_text)
                except json.JSONDecodeError as e:
                    raise VisionAPIError(f"Invalid JSON in response: {e}", "parse_error", retryable=False)

                # Validate against expected schema (basic check for required fields)
                validated = self._validate_response_against_schema(parsed, response_schema, task)
                return validated

            except VisionAPIError:
                # Re-raise our own errors
                raise
            except Exception as e:
                error_str = str(e)
                logger.warning(f"NVIDIA API call failed (attempt {attempt + 1}): {error_str}")

                # Categorize errors
                if "401" in error_str or "authentication" in error_str.lower() or "unauthorized" in error_str.lower():
                    last_error = VisionAPIError(f"Authentication failed: {e}", "auth_error", retryable=False)
                    break
                elif "429" in error_str or "rate limit" in error_str.lower() or "rate_limit" in error_str.lower():
                    last_error = VisionAPIError(f"Rate limited: {e}", "rate_limit", retryable=True)
                elif "timeout" in error_str.lower():
                    last_error = VisionAPIError(f"Request timeout: {e}", "timeout", retryable=True)
                elif "500" in error_str or "502" in error_str or "503" in error_str:
                    last_error = VisionAPIError(f"Server error: {e}", "server_error", retryable=True)
                elif "model" in error_str.lower() and ("not found" in error_str.lower() or "invalid" in error_str.lower()):
                    last_error = VisionAPIError(f"Unsupported model: {e}", "model_error", retryable=False)
                    break
                else:
                    last_error = VisionAPIError(f"Unexpected error: {e}", "unknown", retryable=False)
                    break

            if attempt < self.config.max_retries and last_error and last_error.retryable:
                wait = 2 ** attempt
                logger.warning(f"NVIDIA API call failed ({last_error.error_type}), retrying in {wait}s...")
                await asyncio.sleep(wait)

        raise last_error or VisionAPIError("Max retries exceeded", "max_retries")

    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from response text, handling markdown code fences."""
        if not text:
            return ""
        text = text.strip()
        # Handle markdown code fences
        if "```json" in text:
            parts = text.split("```json")
            if len(parts) > 1:
                text = parts[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1].strip()
        return text

    def _validate_response_against_schema(self, parsed: Dict[str, Any], response_schema: Dict[str, Any], task: VisionTask) -> Dict[str, Any]:
        """Validate parsed response has required fields for the task."""
        required_fields = response_schema.get("required", [])
        missing = [field for field in required_fields if field not in parsed]
        if missing:
            raise VisionAPIError(f"Missing required fields in response: {missing}", "validation_error", retryable=False)
        # Ensure result field is present and valid
        if "result" not in parsed:
            parsed["result"] = "completed"
        return parsed

    async def _call_anthropic(
        self,
        images: List[str],
        prompt: str,
        response_schema: Dict[str, Any],
        event_id: str,
        task: VisionTask,
    ) -> Dict[str, Any]:
        """Call Anthropic Claude API (fallback when OpenAI not available)."""
        # Import anthropic lazily
        try:
            import anthropic
        except ImportError:
            raise VisionAPIError("Anthropic SDK not installed", "not_configured", retryable=False)

        if not self.config.api_key:
            raise VisionAPIError("Anthropic API key not configured", "not_configured", retryable=False)

        client = anthropic.Anthropic(
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
        )

        content = [{"type": "text", "text": prompt}]
        for img_b64 in images:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
            })

        # Add schema instruction to prompt for Anthropic
        schema_str = json.dumps(response_schema, indent=2)
        full_prompt = f"{prompt}\n\nReturn ONLY valid JSON matching this schema:\n{schema_str}"

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": content}],
                )
                content_text = response.content[0].text
                # Extract JSON from potential markdown code blocks
                if "```json" in content_text:
                    content_text = content_text.split("```json")[1].split("```")[0].strip()
                elif "```" in content_text:
                    content_text = content_text.split("```")[1].split("```")[0].strip()
                return json.loads(content_text)
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Anthropic API call failed (attempt {attempt + 1}): {error_str}")

                if "401" in error_str or "authentication" in error_str.lower():
                    last_error = VisionAPIError(f"Authentication failed: {e}", "auth_error", retryable=False)
                    break
                elif "429" in error_str or "rate limit" in error_str.lower():
                    last_error = VisionAPIError(f"Rate limited: {e}", "rate_limit", retryable=True)
                elif "timeout" in error_str.lower():
                    last_error = VisionAPIError(f"Request timeout: {e}", "timeout", retryable=True)
                else:
                    last_error = VisionAPIError(f"API error: {e}", "api_error", retryable=False)
                    break

            if attempt < self.config.max_retries and last_error and last_error.retryable:
                wait = 2 ** attempt
                logger.warning(f"Anthropic API call failed ({last_error.error_type}), retrying in {wait}s...")
                time.sleep(wait)

        raise last_error or VisionAPIError("Max retries exceeded", "max_retries")

    def _call_openai_legacy(
        self,
        payload: Dict[str, Any],
        event_id: str,
    ) -> Dict[str, Any]:
        """Call OpenAI or Azure OpenAI API (legacy method for test compatibility)."""
        # Legacy stub - actual implementation uses async _call_openai with Responses API
        raise NotImplementedError("Use async _call_openai method instead")

    def _build_openai_payload(
        self,
        images: List[str],
        prompt: str,
        response_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build OpenAI-compatible payload with structured output (legacy method for test compatibility)."""
        content = [{"type": "text", "text": prompt}]
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}
            })

        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are an industrial safety vision analyst. Respond only with valid JSON matching the provided schema."},
                {"role": "user", "content": content}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_schema", "json_schema": {"name": "vision_result", "schema": response_schema, "strict": True}},
        }

    def _build_anthropic_payload(
        self,
        images: List[str],
        prompt: str,
    ) -> Dict[str, Any]:
        """Build Anthropic-compatible payload (legacy method for test compatibility)."""
        content = [{"type": "text", "text": prompt}]
        for img_b64 in images:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
            })

        return {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": content}],
        }

    async def _call_api(
        self,
        images: List[str],
        prompt: str,
        response_schema: Dict[str, Any],
        event_id: str,
        task: VisionTask,
    ) -> Dict[str, Any]:
        """Dispatch to appropriate provider."""
        if not self.config.enabled or not self.config.api_key:
            raise VisionAPIError("Vision API not configured or disabled", "not_configured", retryable=False)

        if not self._check_rate_limit(event_id, task):
            raise VisionAPIError(f"Max calls per event exceeded for {task.value}", "rate_limit", retryable=False)

        self._increment_call_count(event_id, task)

        # Log provider and tool call for verification (no secrets)
        logger.info(f"Vision API call: provider={self.config.provider.value}, task={task.value}, event_id={event_id}")

        if self.config.provider == VisionProvider.NVIDIA:
            return await self._call_nvidia(images, prompt, response_schema, event_id, task)
        elif self.config.provider in (VisionProvider.OPENAI, VisionProvider.AZURE_OPENAI):
            return await self._call_openai(images, prompt, response_schema, event_id, task)
        elif self.config.provider == VisionProvider.ANTHROPIC:
            return await self._call_anthropic(images, prompt, response_schema, event_id, task)
        else:
            raise VisionAPIError(f"Unknown provider: {self.config.provider}", "config_error", retryable=False)

    # --- Public API Methods ---

    async def inspect_crop(
        self,
        frame_bytes: bytes,
        bbox: List[int],
        person_id: int,
        event_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> InspectCropResult:
        """
        Inspect a cropped region of interest for detailed safety analysis.
        """
        prompt = self._build_inspect_crop_prompt(person_id, bbox, context)
        schema = self._get_response_schema(InspectCropResult)
        img_b64 = self._encode_image_bytes(frame_bytes)

        try:
            result = await self._call_api([img_b64], prompt, schema, event_id, VisionTask.INSPECT_CROP)
            result["result"] = "completed"
            result["model_used"] = self.config.model
            return InspectCropResult(**result)
        except VisionAPIError as e:
            logger.warning(f"Vision API inspect_crop failed: {e}")
            # Return safe default with structured error info
            return InspectCropResult(
                person_id=person_id,
                bbox=bbox,
                confidence_boost=0.0,
                risk_level_change="same",
                details=f"Vision API unavailable: {e.error_type}",
                violations_detected=[],
                result="vision_api_unavailable" if e.error_type == "not_configured" else "failed",
                model_used=self.config.model,
                inspection_error={"error_type": e.error_type, "message": str(e), "retryable": e.retryable},
            )

    async def verify_ppe(
        self,
        frame_bytes: bytes,
        bbox: List[int],
        event_id: str,
        required_ppe: Optional[List[str]] = None,
    ) -> VerifyPPEResult:
        """
        Verify PPE compliance in a region.
        """
        required = required_ppe or ["hard_hat", "safety_vest", "gloves", "boots"]
        prompt = self._build_verify_ppe_prompt(bbox, required)
        schema = self._get_response_schema(VerifyPPEResult)
        img_b64 = self._encode_image_bytes(frame_bytes)

        try:
            result = await self._call_api([img_b64], prompt, schema, event_id, VisionTask.VERIFY_PPE)
            result["result"] = "completed"
            result["model_used"] = self.config.model
            return VerifyPPEResult(**result)
        except VisionAPIError as e:
            logger.warning(f"Vision API verify_ppe failed: {e}")
            return VerifyPPEResult(
                bbox=bbox,
                ppe_detected={item: None for item in required},
                compliance="uncertain",
                confidence=0.0,
                details=f"Vision API unavailable: {e.error_type}",
                missing_items=required,
                result="vision_api_unavailable" if e.error_type == "not_configured" else "failed",
                model_used=self.config.model,
                inspection_error={"error_type": e.error_type, "message": str(e), "retryable": e.retryable},
            )

    async def assess_scene(
        self,
        frame_bytes: bytes,
        event_id: str,
        event_context: Dict[str, Any],
    ) -> AssessSceneResult:
        """
        Assess overall scene risk level and recommended action.
        """
        prompt = self._build_assess_scene_prompt(event_context)
        schema = self._get_response_schema(AssessSceneResult)
        img_b64 = self._encode_image_bytes(frame_bytes)

        try:
            result = await self._call_api([img_b64], prompt, schema, event_id, VisionTask.ASSESS_SCENE)
            result["result"] = "completed"
            result["model_used"] = self.config.model
            return AssessSceneResult(**result)
        except VisionAPIError as e:
            logger.warning(f"Vision API assess_scene failed: {e}")
            return AssessSceneResult(
                risk_level="low",
                confidence=0.0,
                reason_codes=["vision_api_unavailable"],
                recommended_action="continue_tracking",
                details=f"Vision API unavailable: {e.error_type}",
                result="vision_api_unavailable" if e.error_type == "not_configured" else "failed",
                model_used=self.config.model,
                inspection_error={"error_type": e.error_type, "message": str(e), "retryable": e.retryable},
            )

    async def _call_api_async(
        self,
        images: List[str],
        prompt: str,
        response_schema: Dict[str, Any],
        event_id: str,
        task: VisionTask,
    ) -> Dict[str, Any]:
        """Async wrapper for API calls."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._call_api,
            images, prompt, response_schema, event_id, task
        )

    def is_available(self) -> bool:
        """Check if vision API is configured and enabled."""
        return self.config.enabled and self.config.api_key is not None


# Global client instance (lazy initialization)
_vision_client: Optional[VisionClient] = None


def get_vision_client() -> VisionClient:
    """Get or create the global vision client."""
    global _vision_client
    if _vision_client is None:
        _vision_client = VisionClient()
    return _vision_client


def reset_vision_client():
    """Reset the global vision client (for testing)."""
    global _vision_client
    _vision_client = None


# Synchronous wrapper for use in non-async contexts
import asyncio

def inspect_crop_sync(
    frame_bytes: bytes,
    bbox: List[int],
    person_id: int,
    event_id: str,
    context: Optional[Dict[str, Any]] = None,
) -> InspectCropResult:
    """Synchronous wrapper for inspect_crop."""
    client = get_vision_client()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(client.inspect_crop(frame_bytes, bbox, person_id, event_id, context))


def verify_ppe_sync(
    frame_bytes: bytes,
    bbox: List[int],
    event_id: str,
    required_ppe: Optional[List[str]] = None,
) -> VerifyPPEResult:
    """Synchronous wrapper for verify_ppe."""
    client = get_vision_client()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(client.verify_ppe(frame_bytes, bbox, event_id, required_ppe))


def assess_scene_sync(
    frame_bytes: bytes,
    event_id: str,
    event_context: Dict[str, Any],
) -> AssessSceneResult:
    """Synchronous wrapper for assess_scene."""
    client = get_vision_client()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(client.assess_scene(frame_bytes, event_id, event_context))
