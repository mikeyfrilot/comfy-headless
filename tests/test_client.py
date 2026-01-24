"""Comprehensive tests for ComfyClient module."""

from unittest.mock import Mock, patch

import pytest

# ============================================================================
# CLIENT INITIALIZATION TESTS
# ============================================================================


class TestComfyClientInitialization:
    """Test client initialization and configuration."""

    def test_client_default_initialization(self):
        """Test client initializes with defaults."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        assert client.base_url is not None
        assert "localhost" in client.base_url or "127.0.0.1" in client.base_url

    def test_client_custom_url(self):
        """Test client with custom URL."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient(base_url="http://192.168.1.100:8188")
        assert client.base_url == "http://192.168.1.100:8188"

    def test_client_url_normalization(self):
        """Test client normalizes URL (strips trailing slash)."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient(base_url="http://localhost:8188/")
        assert not client.base_url.endswith("/")

    def test_client_has_client_id(self):
        """Test client generates unique ID."""
        from comfy_headless.client import ComfyClient

        client1 = ComfyClient()
        client2 = ComfyClient()
        assert client1.client_id != client2.client_id

    def test_client_with_rate_limit(self):
        """Test client with rate limiting."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient(rate_limit=10, rate_limit_per_seconds=1.0)
        assert client._rate_limiter is not None

    def test_client_without_rate_limit(self):
        """Test client without rate limiting."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient(rate_limit=None)
        assert client._rate_limiter is None


# ============================================================================
# SESSION MANAGEMENT TESTS
# ============================================================================


class TestSessionManagement:
    """Test HTTP session management."""

    def test_session_lazy_creation(self):
        """Test session is created lazily."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        assert client._session is None
        # Access session property to trigger creation
        _ = client.session
        assert client._session is not None

    def test_session_reuse(self):
        """Test session is reused."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        session1 = client.session
        session2 = client.session
        assert session1 is session2

    def test_close_session(self):
        """Test session can be closed."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        _ = client.session  # Create session
        client.close()
        assert client._session is None

    def test_context_manager(self):
        """Test context manager closes session."""
        from comfy_headless.client import ComfyClient

        with ComfyClient() as client:
            _ = client.session
            assert client._session is not None
        assert client._session is None


# ============================================================================
# CONNECTION TESTS (MOCKED)
# ============================================================================


class TestConnectionMethods:
    """Test connection methods with mocking."""

    @patch("comfy_headless.client.ComfyClient._get")
    def test_is_online_true(self, mock_get):
        """Test is_online returns True when server responds."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = ComfyClient()
        assert client.is_online() is True
        mock_get.assert_called_once()

    @patch("comfy_headless.client.ComfyClient._get")
    def test_is_online_false_on_error(self, mock_get):
        """Test is_online returns False on error."""
        from comfy_headless.client import ComfyClient

        mock_get.side_effect = Exception("Connection refused")

        client = ComfyClient()
        assert client.is_online() is False

    @patch("comfy_headless.client.ComfyClient._get")
    def test_is_online_false_on_bad_status(self, mock_get):
        """Test is_online returns False on non-200 status."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        client = ComfyClient()
        assert client.is_online() is False

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_system_stats(self, mock_get):
        """Test get_system_stats returns data."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "devices": [{"vram_total": 16 * 1024**3, "vram_free": 8 * 1024**3}]
        }
        mock_get.return_value = mock_response

        client = ComfyClient()
        stats = client.get_system_stats()
        assert stats is not None
        assert "devices" in stats

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_system_stats_returns_none_on_error(self, mock_get):
        """Test get_system_stats returns None on error."""
        from comfy_headless.client import ComfyClient

        mock_get.side_effect = Exception("Connection error")

        client = ComfyClient()
        stats = client.get_system_stats()
        assert stats is None


# ============================================================================
# VRAM DETECTION TESTS
# ============================================================================


class TestVRAMDetection:
    """Test VRAM detection and estimation."""

    @patch("comfy_headless.client.ComfyClient.get_system_stats")
    def test_get_vram_gb(self, mock_stats):
        """Test VRAM detection."""
        from comfy_headless.client import ComfyClient

        mock_stats.return_value = {"devices": [{"vram_total": 16 * 1024**3}]}

        client = ComfyClient()
        vram = client.get_vram_gb()
        assert abs(vram - 16.0) < 0.1

    @patch("comfy_headless.client.ComfyClient.get_system_stats")
    def test_get_vram_gb_fallback(self, mock_stats):
        """Test VRAM fallback when detection fails."""
        from comfy_headless.client import ComfyClient

        mock_stats.return_value = None

        client = ComfyClient()
        vram = client.get_vram_gb()
        assert vram == 8.0  # Default fallback

    @patch("comfy_headless.client.ComfyClient.get_system_stats")
    def test_get_free_vram_gb(self, mock_stats):
        """Test free VRAM detection."""
        from comfy_headless.client import ComfyClient

        mock_stats.return_value = {"devices": [{"vram_free": 8 * 1024**3}]}

        client = ComfyClient()
        vram = client.get_free_vram_gb()
        assert abs(vram - 8.0) < 0.1

    @patch("comfy_headless.client.ComfyClient.get_system_stats")
    def test_get_free_vram_gb_fallback(self, mock_stats):
        """Test free VRAM fallback."""
        from comfy_headless.client import ComfyClient

        mock_stats.return_value = None

        client = ComfyClient()
        vram = client.get_free_vram_gb()
        assert vram == 0.0

    def test_estimate_vram_for_image(self):
        """Test VRAM estimation for images."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        vram = client.estimate_vram_for_image(1024, 1024, 1)
        assert vram > 0

    def test_estimate_vram_for_image_larger(self):
        """Test VRAM increases with resolution."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        small = client.estimate_vram_for_image(512, 512, 1)
        large = client.estimate_vram_for_image(2048, 2048, 1)
        assert large > small

    def test_estimate_vram_for_video(self):
        """Test VRAM estimation for video."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        vram = client.estimate_vram_for_video(512, 512, 16, "animatediff")
        assert vram > 0

    def test_estimate_vram_for_video_models(self):
        """Test VRAM varies by model."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        animatediff = client.estimate_vram_for_video(512, 512, 16, "animatediff")
        hunyuan = client.estimate_vram_for_video(512, 512, 16, "hunyuan")
        # Hunyuan should need more VRAM
        assert hunyuan > animatediff


# ============================================================================
# VRAM CHECKING TESTS
# ============================================================================


class TestVRAMChecking:
    """Test VRAM availability checking."""

    @patch("comfy_headless.client.ComfyClient.get_free_vram_gb")
    def test_check_vram_available_sufficient(self, mock_free):
        """Test VRAM check passes with sufficient memory."""
        from comfy_headless.client import ComfyClient

        mock_free.return_value = 12.0

        client = ComfyClient()
        assert client.check_vram_available(8.0) is True

    @patch("comfy_headless.client.ComfyClient.get_free_vram_gb")
    def test_check_vram_available_insufficient(self, mock_free):
        """Test VRAM check fails with insufficient memory."""
        from comfy_headless.client import ComfyClient

        mock_free.return_value = 4.0

        client = ComfyClient()
        assert client.check_vram_available(8.0) is False

    @patch("comfy_headless.client.ComfyClient.get_free_vram_gb")
    def test_check_vram_available_raises(self, mock_free):
        """Test VRAM check raises exception when requested."""
        from comfy_headless.client import ComfyClient
        from comfy_headless.exceptions import InsufficientVRAMError

        mock_free.return_value = 4.0

        client = ComfyClient()
        with pytest.raises(InsufficientVRAMError):
            client.check_vram_available(8.0, raise_on_insufficient=True)

    @patch("comfy_headless.client.ComfyClient.get_free_vram_gb")
    def test_check_vram_available_unknown(self, mock_free):
        """Test VRAM check passes when VRAM unknown."""
        from comfy_headless.client import ComfyClient

        mock_free.return_value = 0.0  # Unknown

        client = ComfyClient()
        assert client.check_vram_available(8.0) is True


# ============================================================================
# ENSURE ONLINE TESTS
# ============================================================================


class TestEnsureOnline:
    """Test ensure_online method."""

    @patch("comfy_headless.client.ComfyClient.is_online")
    def test_ensure_online_passes(self, mock_online):
        """Test ensure_online doesn't raise when online."""
        from comfy_headless.client import ComfyClient

        mock_online.return_value = True

        client = ComfyClient()
        client.ensure_online()  # Should not raise

    @patch("comfy_headless.client.ComfyClient.is_online")
    def test_ensure_online_raises(self, mock_online):
        """Test ensure_online raises when offline."""
        from comfy_headless.client import ComfyClient
        from comfy_headless.exceptions import ComfyUIOfflineError

        mock_online.return_value = False

        client = ComfyClient()
        with pytest.raises(ComfyUIOfflineError):
            client.ensure_online()


# ============================================================================
# PRESET RECOMMENDATION TESTS
# ============================================================================


class TestPresetRecommendation:
    """Test preset recommendation methods."""

    @patch("comfy_headless.client.ComfyClient.get_vram_gb")
    def test_recommend_image_preset_low_vram(self, mock_vram):
        """Test image preset for low VRAM."""
        from comfy_headless.client import ComfyClient

        mock_vram.return_value = 4.0
        client = ComfyClient()
        assert client.recommend_image_preset() == "draft"

    @patch("comfy_headless.client.ComfyClient.get_vram_gb")
    def test_recommend_image_preset_medium_vram(self, mock_vram):
        """Test image preset for medium VRAM."""
        from comfy_headless.client import ComfyClient

        mock_vram.return_value = 7.0
        client = ComfyClient()
        assert client.recommend_image_preset() == "fast"

    @patch("comfy_headless.client.ComfyClient.get_vram_gb")
    def test_recommend_image_preset_high_vram(self, mock_vram):
        """Test image preset for high VRAM."""
        from comfy_headless.client import ComfyClient

        mock_vram.return_value = 16.0
        client = ComfyClient()
        assert client.recommend_image_preset() == "hd"

    @patch("comfy_headless.client.ComfyClient.get_vram_gb")
    def test_recommend_image_preset_portrait(self, mock_vram):
        """Test image preset with portrait intent."""
        from comfy_headless.client import ComfyClient

        mock_vram.return_value = 10.0
        client = ComfyClient()
        assert client.recommend_image_preset("portrait") == "portrait"

    @patch("comfy_headless.client.ComfyClient.get_vram_gb")
    def test_recommend_video_preset(self, mock_vram):
        """Test video preset recommendation."""
        from comfy_headless.client import ComfyClient

        mock_vram.return_value = 16.0
        client = ComfyClient()
        preset = client.recommend_video_preset()
        assert preset is not None


# ============================================================================
# OBJECT INFO TESTS
# ============================================================================


class TestObjectInfo:
    """Test model/info retrieval methods."""

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_checkpoints(self, mock_get):
        """Test getting checkpoints."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [["model1.safetensors", "model2.safetensors"]]}}
            }
        }
        mock_get.return_value = mock_response

        client = ComfyClient()
        checkpoints = client.get_checkpoints()
        assert "model1.safetensors" in checkpoints

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_samplers(self, mock_get):
        """Test getting samplers."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "KSampler": {
                "input": {"required": {"sampler_name": [["euler", "euler_ancestral", "dpmpp_2m"]]}}
            }
        }
        mock_get.return_value = mock_response

        client = ComfyClient()
        samplers = client.get_samplers()
        assert "euler" in samplers

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_samplers_fallback(self, mock_get):
        """Test samplers fallback on error."""
        from comfy_headless.client import ComfyClient

        mock_get.side_effect = Exception("Error")

        client = ComfyClient()
        samplers = client.get_samplers()
        # Should return default list
        assert "euler" in samplers

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_schedulers(self, mock_get):
        """Test getting schedulers."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "KSampler": {"input": {"required": {"scheduler": [["normal", "karras"]]}}}
        }
        mock_get.return_value = mock_response

        client = ComfyClient()
        schedulers = client.get_schedulers()
        assert "normal" in schedulers

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_loras(self, mock_get):
        """Test getting LoRAs."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "LoraLoader": {"input": {"required": {"lora_name": [["lora1.safetensors"]]}}}
        }
        mock_get.return_value = mock_response

        client = ComfyClient()
        loras = client.get_loras()
        assert "lora1.safetensors" in loras

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_motion_models(self, mock_get):
        """Test getting motion models."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "ADE_LoadAnimateDiffModel": {
                "input": {"required": {"model_name": [["v3_sd15_mm.ckpt"]]}}
            }
        }
        mock_get.return_value = mock_response

        client = ComfyClient()
        models = client.get_motion_models()
        assert "v3_sd15_mm.ckpt" in models


# ============================================================================
# QUEUE MANAGEMENT TESTS
# ============================================================================


class TestQueueManagement:
    """Test queue management methods."""

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_queue(self, mock_get):
        """Test getting queue status."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "queue_running": [["id1", "prompt1"]],
            "queue_pending": [],
        }
        mock_get.return_value = mock_response

        client = ComfyClient()
        queue = client.get_queue()
        assert "queue_running" in queue

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_queue_error(self, mock_get):
        """Test get_queue returns empty on error."""
        from comfy_headless.client import ComfyClient

        mock_get.side_effect = Exception("Error")

        client = ComfyClient()
        queue = client.get_queue()
        assert queue == {"queue_running": [], "queue_pending": []}

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_history(self, mock_get):
        """Test getting history."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"prompt123": {"status": {"completed": True}}}
        mock_get.return_value = mock_response

        client = ComfyClient()
        history = client.get_history("prompt123")
        assert "prompt123" in history

    @patch("comfy_headless.client.ComfyClient._post")
    def test_cancel_current(self, mock_post):
        """Test cancelling current job."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_post.return_value = mock_response

        client = ComfyClient()
        result = client.cancel_current()
        assert result is True
        mock_post.assert_called_with("/interrupt")

    @patch("comfy_headless.client.ComfyClient._post")
    def test_cancel_current_fails(self, mock_post):
        """Test cancel returns False on error."""
        from comfy_headless.client import ComfyClient

        mock_post.side_effect = Exception("Error")

        client = ComfyClient()
        result = client.cancel_current()
        assert result is False

    @patch("comfy_headless.client.ComfyClient._post")
    def test_clear_queue(self, mock_post):
        """Test clearing queue."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_post.return_value = mock_response

        client = ComfyClient()
        result = client.clear_queue()
        assert result is True


# ============================================================================
# PROMPT QUEUEING TESTS
# ============================================================================


class TestPromptQueueing:
    """Test prompt queueing."""

    @patch("comfy_headless.client.ComfyClient._post")
    def test_queue_prompt_success(self, mock_post):
        """Test successful prompt queueing."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"prompt_id": "abc123"}
        mock_post.return_value = mock_response

        client = ComfyClient()
        prompt_id = client.queue_prompt({"test": "workflow"})
        assert prompt_id == "abc123"

    @patch("comfy_headless.client.ComfyClient._post")
    def test_queue_prompt_failure(self, mock_post):
        """Test prompt queueing failure."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        client = ComfyClient()
        prompt_id = client.queue_prompt({"test": "workflow"})
        assert prompt_id is None


# ============================================================================
# FILE DOWNLOAD TESTS
# ============================================================================


class TestFileDownloads:
    """Test file download methods."""

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_image(self, mock_get):
        """Test image download."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_response.content = b"\x89PNG\r\n\x1a\n"
        mock_get.return_value = mock_response

        client = ComfyClient()
        data = client.get_image("test.png")
        assert data == b"\x89PNG\r\n\x1a\n"

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_image_failure(self, mock_get):
        """Test image download failure."""
        from comfy_headless.client import ComfyClient

        mock_get.side_effect = Exception("Download error")

        client = ComfyClient()
        data = client.get_image("test.png")
        assert data is None

    @patch("comfy_headless.client.ComfyClient._get")
    def test_get_video(self, mock_get):
        """Test video download."""
        from comfy_headless.client import ComfyClient

        mock_response = Mock()
        mock_response.ok = True
        mock_response.content = b"\x00\x00\x00\x1cftyp"
        mock_get.return_value = mock_response

        client = ComfyClient()
        data = client.get_video("test.mp4")
        assert data is not None


# ============================================================================
# WORKFLOW BUILDER TESTS
# ============================================================================


class TestWorkflowBuilders:
    """Test workflow building methods."""

    @patch("comfy_headless.client.ComfyClient.get_checkpoints")
    def test_build_txt2img_workflow(self, mock_checkpoints):
        """Test txt2img workflow building."""
        from comfy_headless.client import ComfyClient

        mock_checkpoints.return_value = ["model.safetensors"]

        client = ComfyClient()
        workflow = client.build_txt2img_workflow(
            prompt="a beautiful sunset", width=512, height=512, steps=20
        )

        assert "3" in workflow  # KSampler
        assert "4" in workflow  # CheckpointLoaderSimple
        assert "5" in workflow  # EmptyLatentImage
        assert "6" in workflow  # CLIPTextEncode positive
        assert "9" in workflow  # SaveImage

    @patch("comfy_headless.client.ComfyClient.get_checkpoints")
    def test_build_txt2img_workflow_random_seed(self, mock_checkpoints):
        """Test txt2img workflow generates seed when -1."""
        from comfy_headless.client import ComfyClient

        mock_checkpoints.return_value = ["model.safetensors"]

        client = ComfyClient()
        workflow = client.build_txt2img_workflow(prompt="test", seed=-1)

        seed = workflow["3"]["inputs"]["seed"]
        assert seed != -1

    @patch("comfy_headless.client.ComfyClient.get_checkpoints")
    @patch("comfy_headless.client.ComfyClient.get_motion_models")
    def test_build_video_workflow(self, mock_motion, mock_checkpoints):
        """Test video workflow building."""
        from comfy_headless.client import ComfyClient

        mock_checkpoints.return_value = ["dreamshaper_8.safetensors"]
        mock_motion.return_value = ["v3_sd15_mm.ckpt"]

        client = ComfyClient()
        workflow = client.build_video_workflow(prompt="a cat walking", frames=16, fps=8)

        assert "1" in workflow  # CheckpointLoaderSimple
        assert "2" in workflow  # ADE_LoadAnimateDiffModel
        assert "9" in workflow  # VHS_VideoCombine


# ============================================================================
# HIGH-LEVEL GENERATION TESTS
# ============================================================================


class TestHighLevelGeneration:
    """Test high-level generation methods."""

    @patch("comfy_headless.client.ComfyClient.ensure_online")
    @patch("comfy_headless.client.ComfyClient.queue_prompt")
    def test_generate_image_no_wait(self, mock_queue, mock_online):
        """Test generate_image without waiting."""
        from comfy_headless.client import ComfyClient

        mock_queue.return_value = "prompt123"

        client = ComfyClient()
        result = client.generate_image(prompt="test", wait=False)

        assert result["success"] is True
        assert result["prompt_id"] == "prompt123"

    @patch("comfy_headless.client.ComfyClient.ensure_online")
    @patch("comfy_headless.client.ComfyClient.queue_prompt")
    def test_generate_image_queue_failure(self, mock_queue, mock_online):
        """Test generate_image with queue failure."""
        from comfy_headless.client import ComfyClient

        mock_queue.return_value = None

        client = ComfyClient()
        result = client.generate_image(prompt="test", wait=False)

        assert result["success"] is False
        assert "Failed to queue" in result["error"]

    @patch("comfy_headless.client.ComfyClient.ensure_online")
    def test_generate_image_offline(self, mock_online):
        """Test generate_image when offline."""
        from comfy_headless.client import ComfyClient
        from comfy_headless.exceptions import ComfyUIOfflineError

        mock_online.side_effect = ComfyUIOfflineError(url="http://localhost:8188")

        client = ComfyClient()
        result = client.generate_image(prompt="test")

        assert result["success"] is False
        assert result["error"] is not None

    @patch("comfy_headless.client.ComfyClient.ensure_online")
    @patch("comfy_headless.client.ComfyClient.queue_prompt")
    def test_generate_video_no_wait(self, mock_queue, mock_online):
        """Test generate_video without waiting."""
        from comfy_headless.client import ComfyClient

        mock_queue.return_value = "prompt456"

        client = ComfyClient()
        result = client.generate_video(prompt="a cat walking", preset="quick", wait=False)

        assert result["success"] is True
        assert result["prompt_id"] == "prompt456"


# ============================================================================
# BATCH GENERATION TESTS
# ============================================================================


class TestBatchGeneration:
    """Test batch generation."""

    def test_generate_batch_empty(self):
        """Test batch with no prompts."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        result = client.generate_batch(prompts=[])

        assert result["success"] is False
        assert "No prompts" in result["errors"][0]

    @patch("comfy_headless.client.ComfyClient.generate_image")
    def test_generate_batch_success(self, mock_generate):
        """Test successful batch generation."""
        from comfy_headless.client import ComfyClient

        mock_generate.return_value = {
            "success": True,
            "images": [{"filename": "test.png"}],
            "error": None,
            "seed": 123,
            "prompt_id": "abc",
        }

        client = ComfyClient()
        result = client.generate_batch(prompts=["prompt1", "prompt2"], check_vram=False)

        assert result["success"] is True
        assert result["success_count"] == 2
        assert len(result["results"]) == 2


# ============================================================================
# WAIT FOR COMPLETION TESTS
# ============================================================================


class TestWaitForCompletion:
    """Test wait_for_completion method."""

    @patch("comfy_headless.client.ComfyClient.get_history")
    def test_wait_completes(self, mock_history):
        """Test wait returns on completion."""
        from comfy_headless.client import ComfyClient

        mock_history.return_value = {
            "prompt123": {"status": {"completed": True, "status_str": "success"}, "outputs": {}}
        }

        client = ComfyClient()
        result = client.wait_for_completion("prompt123", timeout=5)

        assert result is not None
        assert result["status"]["completed"] is True

    @patch("comfy_headless.client.ComfyClient.get_history")
    def test_wait_timeout(self, mock_history):
        """Test wait returns None on timeout."""
        from comfy_headless.client import ComfyClient

        mock_history.return_value = {}  # Not in history

        client = ComfyClient()
        result = client.wait_for_completion("prompt123", timeout=0.1, poll_interval=0.05)

        assert result is None


# ============================================================================
# REQUEST METHOD TESTS
# ============================================================================


class TestRequestMethods:
    """Test internal request methods."""

    @patch("requests.Session.request")
    def test_request_connection_error(self, mock_request):
        """Test request raises on connection error."""
        import requests

        from comfy_headless.client import ComfyClient
        from comfy_headless.exceptions import ComfyUIConnectionError

        mock_request.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = ComfyClient()
        with pytest.raises(ComfyUIConnectionError):
            client._request("GET", "/test")

    @patch("requests.Session.request")
    def test_request_timeout(self, mock_request):
        """Test request raises on timeout."""
        import requests

        from comfy_headless.client import ComfyClient
        from comfy_headless.exceptions import ComfyUIConnectionError

        mock_request.side_effect = requests.exceptions.Timeout("Request timed out")

        client = ComfyClient()
        with pytest.raises(ComfyUIConnectionError):
            client._request("GET", "/test", timeout=1)


# ============================================================================
# RATE LIMITING TESTS
# ============================================================================


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter is created correctly."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient(rate_limit=5, rate_limit_per_seconds=2.0)
        assert client._rate_limiter is not None
        assert client._rate_limiter.rate == 5
        assert client._rate_limiter.per_seconds == 2.0

    def test_no_rate_limiter_with_zero(self):
        """Test no rate limiter with 0 limit."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient(rate_limit=0)
        assert client._rate_limiter is None
