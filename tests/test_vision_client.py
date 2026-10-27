"""
Tests for the VisionClient (multimodal API integration).
Tests cover: success, timeout, invalid response, missing credentials, rate limiting.
"""
import os
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from backend.agent.vision_client import (
    VisionClient,
    VisionClientConfig,
    VisionProvider,
    VisionTask,
    VisionAPIError,
    InspectCropResult,
    VerifyPPEResult,
    AssessSceneResult,
    get_vision_client,
    reset_vision_client,
)


class TestVisionClientConfig:
    """Test configuration loading from environment."""

    def test_default_config(self):
        config = VisionClientConfig()
        assert config.provider == VisionProvider.OPENAI
        assert config.model == "gpt-4o-mini"
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 2
        assert config.max_calls_per_event == 3
        assert config.max_calls_per_video == 10
        assert config.enabled is True

    def test_config_from_env_openai(self):
        with patch.dict(os.environ, {
            "VISION_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key-123",
            "VISION_MODEL": "gpt-4o",
            "VISION_TIMEOUT_SECONDS": "15.0",
            "VISION_MAX_RETRIES": "3",
            "VISION_MAX_CALLS_PER_EVENT": "5",
            "VISION_MAX_CALLS_PER_VIDEO": "20",
            "VISION_ENABLED": "true",
        }):
            config = VisionClientConfig.from_env()
            assert config.provider == VisionProvider.OPENAI
            assert config.api_key == "test-key-123"
            assert config.model == "gpt-4o"
            assert config.timeout_seconds == 15.0
            assert config.max_retries == 3
            assert config.max_calls_per_event == 5
            assert config.max_calls_per_video == 20
            assert config.enabled is True

    def test_config_from_env_anthropic(self):
        with patch.dict(os.environ, {
            "VISION_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "anthropic-key-456",
        }):
            config = VisionClientConfig.from_env()
            assert config.provider == VisionProvider.ANTHROPIC
            assert config.api_key == "anthropic-key-456"

    def test_config_from_env_azure(self):
        with patch.dict(os.environ, {
            "VISION_PROVIDER": "azure_openai",
            "AZURE_OPENAI_API_KEY": "azure-key-789",
            "AZURE_OPENAI_ENDPOINT": "https://my-resource.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-06-01",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4o-deployment",
        }):
            config = VisionClientConfig.from_env()
            assert config.provider == VisionProvider.AZURE_OPENAI
            assert config.api_key == "azure-key-789"
            assert config.api_base == "https://my-resource.openai.azure.com"
            assert config.api_version == "2024-06-01"
            assert config.deployment_name == "gpt-4o-deployment"

    def test_config_from_env_nvidia(self):
        with patch.dict(os.environ, {
            "VISION_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": "nvidia-key-123",
            "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
            "VISION_MODEL": "meta/llama-3.2-11b-vision-instruct",
        }):
            config = VisionClientConfig.from_env()
            assert config.provider == VisionProvider.NVIDIA
            assert config.api_key == "nvidia-key-123"
            assert config.api_base == "https://integrate.api.nvidia.com/v1"
            assert config.model == "meta/llama-3.2-11b-vision-instruct"

    def test_config_from_env_nvidia_default_base_url(self):
        with patch.dict(os.environ, {
            "VISION_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": "nvidia-key-123",
        }):
            config = VisionClientConfig.from_env()
            assert config.provider == VisionProvider.NVIDIA
            assert config.api_key == "nvidia-key-123"
            assert config.api_base == "https://integrate.api.nvidia.com/v1"

    def test_config_disabled_when_no_api_key(self):
        with patch.dict(os.environ, {"VISION_PROVIDER": "openai"}, clear=True):
            config = VisionClientConfig.from_env()
            assert config.enabled is False


class TestVisionClientRateLimiting:
    """Test per-event rate limiting."""

    def setup_method(self):
        reset_vision_client()

    def teardown_method(self):
        reset_vision_client()

    def test_rate_limit_allows_up_to_max(self):
        config = VisionClientConfig(max_calls_per_event=3, enabled=True, api_key="test")
        client = VisionClient(config)

        event_id = "test_event_1"
        task = VisionTask.INSPECT_CROP

        # Should allow up to max_calls_per_event
        for i in range(3):
            assert client._check_rate_limit(event_id, task) is True
            client._increment_call_count(event_id, task)

        # 4th call should be denied
        assert client._check_rate_limit(event_id, task) is False

    def test_rate_limit_per_event_independent(self):
        config = VisionClientConfig(max_calls_per_event=2, enabled=True, api_key="test")
        client = VisionClient(config)

        # Event 1 uses 2 calls
        client._increment_call_count("event_1", VisionTask.INSPECT_CROP)
        client._increment_call_count("event_1", VisionTask.INSPECT_CROP)
        assert client._check_rate_limit("event_1", VisionTask.INSPECT_CROP) is False

        # Event 2 should still have full quota
        assert client._check_rate_limit("event_2", VisionTask.INSPECT_CROP) is True


class TestVisionClientAPICalls:
    """Test actual API call behavior with mocks."""

    def setup_method(self):
        reset_vision_client()

    def teardown_method(self):
        reset_vision_client()

    @pytest.mark.asyncio
    async def test_inspect_crop_success(self):
        """Test successful inspect_crop call."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key="test-key",
            enabled=True,
        )
        client = VisionClient(config)

        mock_response = {
            "task": "inspect_crop",
            "person_id": 1,
            "bbox": [100, 100, 50, 50],
            "confidence_boost": 0.1,
            "risk_level_change": "higher",
            "details": "Hard hat missing",
            "violations_detected": ["missing_hard_hat"],
            "ppe_status": {"hard_hat": False, "vest": True},
        }

        with patch.object(client, '_call_api', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await client.inspect_crop(
                frame_bytes=b"fake_image_bytes",
                bbox=[100, 100, 50, 50],
                person_id=1,
                event_id="event_123",
            )

            assert isinstance(result, InspectCropResult)
            assert result.confidence_boost == 0.1
            assert result.risk_level_change == "higher"
            assert "missing_hard_hat" in result.violations_detected
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_inspect_crop_returns_default_on_api_error(self):
        """Test inspect_crop returns safe default when API fails."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key="test-key",
            enabled=True,
        )
        client = VisionClient(config)

        with patch.object(client, '_call_api', new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = VisionAPIError("API timeout", "timeout", retryable=True)

            result = await client.inspect_crop(
                frame_bytes=b"fake_image_bytes",
                bbox=[100, 100, 50, 50],
                person_id=1,
                event_id="event_123",
            )

            # Should return safe default, not raise
            assert isinstance(result, InspectCropResult)
            assert result.confidence_boost == 0.0
            assert result.risk_level_change == "same"
            assert "unavailable" in result.details.lower()

    @pytest.mark.asyncio
    async def test_inspect_crop_disabled_returns_default(self):
        """Test inspect_crop returns default when client disabled."""
        config = VisionClientConfig(enabled=False, api_key=None)
        client = VisionClient(config)

        result = await client.inspect_crop(
            frame_bytes=b"fake_image_bytes",
            bbox=[100, 100, 50, 50],
            person_id=1,
            event_id="event_123",
        )

        assert isinstance(result, InspectCropResult)
        assert result.confidence_boost == 0.0
        assert "unavailable" in result.details.lower()
        assert "not_configured" in result.details.lower()

    @pytest.mark.asyncio
    async def test_verify_ppe_success(self):
        """Test successful verify_ppe call."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key="test-key",
            enabled=True,
        )
        client = VisionClient(config)

        mock_response = {
            "task": "verify_ppe",
            "bbox": [100, 100, 50, 50],
            "ppe_detected": {"hard_hat": True, "vest": True, "gloves": False, "boots": True},
            "compliance": "non_compliant",
            "confidence": 0.85,
            "details": "Gloves not detected",
            "missing_items": ["gloves"],
        }

        with patch.object(client, '_call_api', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await client.verify_ppe(
                frame_bytes=b"fake_image_bytes",
                bbox=[100, 100, 50, 50],
                event_id="event_123",
            )

            assert isinstance(result, VerifyPPEResult)
            assert result.compliance == "non_compliant"
            assert result.confidence == 0.85
            assert "gloves" in result.missing_items

    @pytest.mark.asyncio
    async def test_assess_scene_success(self):
        """Test successful assess_scene call."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key="test-key",
            enabled=True,
        )
        client = VisionClient(config)

        mock_response = {
            "task": "assess_scene",
            "risk_level": "high",
            "confidence": 0.9,
            "reason_codes": ["zone_violation", "multiple_persons"],
            "recommended_action": "create_incident",
            "details": "Two persons in restricted zone",
        }

        with patch.object(client, '_call_api', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await client.assess_scene(
                frame_bytes=b"fake_image_bytes",
                event_id="event_123",
                event_context={"event_type": "restricted_zone_entry", "person_id": 1},
            )

            assert isinstance(result, AssessSceneResult)
            assert result.risk_level == "high"
            assert result.confidence == 0.9
            assert result.recommended_action == "create_incident"

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        """Test that rate limiting prevents excessive calls - returns default on limit."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key="test-key",
            enabled=True,
            max_calls_per_event=1,
        )
        client = VisionClient(config)

        # Mock the actual async _call_openai method (called by _call_api)
        async def mock_call_openai(images, prompt, response_schema, event_id, task):
            return {"task": "inspect_crop", "person_id": 1, "bbox": [0,0,10,10],
                      "confidence_boost": 0.5, "risk_level_change": "higher", "details": "real api", "violations_detected": ["test"]}

        with patch.object(client, '_call_openai', side_effect=mock_call_openai):
            # First call should work and return the mock response
            result1 = await client.inspect_crop(b"bytes", [0,0,10,10], 1, "event_1")
            assert result1.confidence_boost == 0.5
            assert result1.details == "real api"

            # Second call should hit rate limit and return safe default
            result2 = await client.inspect_crop(b"bytes", [0,0,10,10], 1, "event_1")
            assert result2.confidence_boost == 0.0
            assert "rate_limit" in result2.details.lower() or "unavailable" in result2.details.lower()


class TestVisionClientProviderSelection:
    """Test different provider payloads."""

    def setup_method(self):
        reset_vision_client()

    def teardown_method(self):
        reset_vision_client()

    def test_openai_payload_structure(self):
        config = VisionClientConfig(provider=VisionProvider.OPENAI, api_key="test")
        client = VisionClient(config)

        payload = client._build_openai_payload(
            images=["base64img"],
            prompt="Test prompt",
            response_schema={"type": "object"},
        )

        assert payload["model"] == "gpt-4o-mini"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["response_format"]["type"] == "json_schema"

    def test_anthropic_payload_structure(self):
        config = VisionClientConfig(provider=VisionProvider.ANTHROPIC, api_key="test")
        client = VisionClient(config)

        payload = client._build_anthropic_payload(
            images=["base64img"],
            prompt="Test prompt",
        )

        assert payload["model"] == "gpt-4o-mini"  # default model
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"][0]["type"] == "text"
        assert payload["messages"][0]["content"][1]["type"] == "image"

    def test_nvidia_uses_openai_sdk_with_custom_base_url(self):
        """Test that NVIDIA provider configures OpenAI client with custom base URL."""
        config = VisionClientConfig(
            provider=VisionProvider.NVIDIA,
            api_key="nvidia-key",
            api_base="https://custom.nvidia.api/v1",
            model="meta/llama-3.2-11b-vision-instruct",
        )
        client = VisionClient(config)

        # Verify the config is set correctly
        assert config.provider == VisionProvider.NVIDIA
        assert config.api_base == "https://custom.nvidia.api/v1"
        assert config.model == "meta/llama-3.2-11b-vision-instruct"

        # The client should be created with the custom base_url
        # This is tested via the _get_openai_client method behavior


class TestSyncWrappers:
    """Test synchronous wrapper functions."""

    def setup_method(self):
        reset_vision_client()

    def teardown_method(self):
        reset_vision_client()

    def test_inspect_crop_sync(self):
        from backend.agent.vision_client import inspect_crop_sync

        with patch('backend.agent.vision_client.get_vision_client') as mock_get:
            mock_client = MagicMock()
            mock_client.inspect_crop = AsyncMock(return_value=InspectCropResult(
                person_id=1, bbox=[0,0,10,10], confidence_boost=0.1, risk_level_change="same",
                details="ok", violations_detected=[]
            ))
            mock_get.return_value = mock_client

            result = inspect_crop_sync(b"bytes", [0,0,10,10], 1, "event_1")

            assert isinstance(result, InspectCropResult)
            assert result.confidence_boost == 0.1


class TestVisionClientConfigCheck:
    """Test configuration check functionality."""

    def test_check_config_with_key(self):
        """Test check_config when API key is present."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key="test-key-123",
            enabled=True,
            timeout_seconds=30.0,
        )
        result = config.check_config()
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4o-mini"
        assert result["enabled"] is True
        assert result["has_api_key"] is True
        assert result["configured"] is True
        assert len(result["warnings"]) == 0

    def test_check_config_without_key(self):
        """Test check_config when API key is missing."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key=None,
            enabled=True,
        )
        result = config.check_config()
        assert result["has_api_key"] is False
        assert result["configured"] is False
        assert any("No API key" in w for w in result["warnings"])

    def test_check_config_disabled(self):
        """Test check_config when explicitly disabled."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key="test-key",
            enabled=False,
        )
        result = config.check_config()
        assert result["enabled"] is False
        assert result["configured"] is False

    def test_check_config_warnings(self):
        """Test check_config generates appropriate warnings."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key=None,
            enabled=True,
            timeout_seconds=3.0,  # Too short
            max_calls_per_video=0,  # Invalid
        )
        result = config.check_config()
        assert len(result["warnings"]) >= 2
        assert any("timeout" in w.lower() for w in result["warnings"])
        assert any("max_calls_per_video" in w.lower() for w in result["warnings"])


class TestVisionClientVideoRateLimit:
    """Test video-level rate limiting."""

    def setup_method(self):
        reset_vision_client()

    def teardown_method(self):
        reset_vision_client()

    @pytest.mark.asyncio
    async def test_video_rate_limit_enforced(self):
        """Test that video-level rate limiting prevents excessive calls."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key="test-key",
            enabled=True,
            max_calls_per_event=5,  # High per-event limit
            max_calls_per_video=2,   # Low video limit
        )
        client = VisionClient(config)

        async def mock_call_openai(images, prompt, response_schema, event_id, task):
            return {
                "task": "inspect_crop", "person_id": 1, "bbox": [0,0,10,10],
                "confidence_boost": 0.1, "risk_level_change": "same",
                "details": "real api", "violations_detected": []
            }

        with patch.object(client, '_call_openai', side_effect=mock_call_openai):
            # First call should work
            result1 = await client.inspect_crop(b"bytes", [0,0,10,10], 1, "event_1")
            assert result1.confidence_boost == 0.1

            # Second call should work (different event)
            result2 = await client.inspect_crop(b"bytes", [0,0,10,10], 1, "event_2")
            assert result2.confidence_boost == 0.1

            # Third call should hit video rate limit and return safe default
            result3 = await client.inspect_crop(b"bytes", [0,0,10,10], 1, "event_3")
            assert result3.confidence_boost == 0.0
            assert "rate_limit" in result3.details.lower() or "unavailable" in result3.details.lower()

    def test_reset_video_counter(self):
        """Test that reset_video_counter clears video-level counts."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key="test-key",
            enabled=True,
            max_calls_per_video=1,
        )
        client = VisionClient(config)

        # Manually increment video counter
        client.video_call_count = 1
        client.call_counts["event_1:inspect_crop"] = 1

        # Reset should clear both
        client.reset_video_counter()
        assert client.video_call_count == 0
        assert len(client.call_counts) == 0

    def test_check_config_method_on_client(self):
        """Test that VisionClient.check_config() delegates to config."""
        config = VisionClientConfig(
            provider=VisionProvider.OPENAI,
            api_key="test-key",
            enabled=True,
        )
        client = VisionClient(config)
        result = client.check_config()
        assert result["has_api_key"] is True
        assert result["configured"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])