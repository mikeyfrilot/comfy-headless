"""
Extended coverage tests for comfy_headless/client.py

Targets specific uncovered lines from coverage report:
- Lines 48-51: _get_video_builder lazy loading
- Lines 172-174: Rate limiter timeout
- Lines 275-276, 297-298: VRAM detection edge cases
- Lines 481-486, 504-516: Preset fallbacks
- Lines 644-648: Queue error handling
- Lines 696, 704-730: Progress callback paths
- Lines 1064-1090: WorkflowCompiler preset handling
- Lines 1120-1152: History extraction
- Lines 1391-1408: Video fallback to legacy
"""

from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# LAZY VIDEO BUILDER TESTS (lines 48-51)
# ============================================================================


class TestLazyVideoBuilder:
    """Test _get_video_builder lazy import."""

    def test_get_video_builder_caches(self):
        """Video builder is cached after first call."""
        import comfy_headless.client as client_module

        # Reset the cached builder
        original = client_module._video_builder
        client_module._video_builder = None

        try:
            with patch("comfy_headless.video.get_video_builder") as mock_get:
                mock_builder = MagicMock()
                mock_get.return_value = mock_builder

                # First call should import
                result1 = client_module._get_video_builder()
                assert result1 is mock_builder

                # Second call should return cached
                result2 = client_module._get_video_builder()
                assert result2 is mock_builder

                # Should only import once
                mock_get.assert_called_once()
        finally:
            client_module._video_builder = original

    def test_get_video_builder_returns_builder(self):
        """_get_video_builder returns a video builder instance."""
        import comfy_headless.client as client_module

        original = client_module._video_builder
        client_module._video_builder = None

        try:
            # Use actual import
            builder = client_module._get_video_builder()
            assert builder is not None
        finally:
            client_module._video_builder = original


# ============================================================================
# RATE LIMITER TIMEOUT TESTS (lines 172-174)
# ============================================================================


class TestRateLimiterTimeout:
    """Test rate limiter timeout handling in _request."""

    def test_rate_limiter_timeout_raises(self):
        """Rate limiter timeout raises ComfyUIConnectionError."""
        from comfy_headless.client import ComfyClient
        from comfy_headless.exceptions import ComfyUIConnectionError

        client = ComfyClient(rate_limit=1, rate_limit_per_seconds=10.0)

        # Mock rate limiter to return False (timeout)
        with patch.object(client._rate_limiter, "acquire", return_value=False):
            with pytest.raises(ComfyUIConnectionError) as exc_info:
                client._request("GET", "/test")

            assert "Rate limit timeout" in str(exc_info.value)

    def test_rate_limiter_success_allows_request(self):
        """Rate limiter success allows request through."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient(rate_limit=10, rate_limit_per_seconds=1.0)

        with patch.object(client._rate_limiter, "acquire", return_value=True):
            with patch.object(client.session, "request") as mock_request:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_request.return_value = mock_response

                response = client._request("GET", "/test")
                assert response.status_code == 200


# ============================================================================
# VRAM DETECTION EDGE CASES (lines 275-276, 297-298)
# ============================================================================


class TestVRAMDetectionEdgeCases:
    """Test VRAM detection edge cases."""

    def test_get_vram_gb_empty_devices(self):
        """get_vram_gb handles empty devices array."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_system_stats") as mock_stats:
            mock_stats.return_value = {"devices": []}

            vram = client.get_vram_gb()
            assert vram == 8.0  # Fallback default

    def test_get_vram_gb_zero_vram(self):
        """get_vram_gb handles zero vram_total."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_system_stats") as mock_stats:
            mock_stats.return_value = {"devices": [{"vram_total": 0}]}

            vram = client.get_vram_gb()
            assert vram == 8.0  # Fallback default

    def test_get_vram_gb_exception(self):
        """get_vram_gb handles exception during detection."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_system_stats") as mock_stats:
            mock_stats.side_effect = Exception("Detection failed")

            vram = client.get_vram_gb()
            assert vram == 8.0  # Fallback default

    def test_get_free_vram_gb_empty_devices(self):
        """get_free_vram_gb handles empty devices array."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_system_stats") as mock_stats:
            mock_stats.return_value = {"devices": []}

            vram = client.get_free_vram_gb()
            assert vram == 0.0

    def test_get_free_vram_gb_zero_free(self):
        """get_free_vram_gb handles zero vram_free."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_system_stats") as mock_stats:
            mock_stats.return_value = {"devices": [{"vram_free": 0}]}

            vram = client.get_free_vram_gb()
            assert vram == 0.0

    def test_get_free_vram_gb_exception(self):
        """get_free_vram_gb handles exception."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_system_stats") as mock_stats:
            mock_stats.side_effect = Exception("Detection failed")

            vram = client.get_free_vram_gb()
            assert vram == 0.0


# ============================================================================
# PRESET FALLBACK TESTS (lines 481-486, 504-516)
# ============================================================================


class TestPresetFallbacks:
    """Test preset recommendation fallbacks."""

    def test_recommend_image_preset_cinematic(self):
        """recommend_image_preset returns cinematic for high VRAM + intent."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_vram_gb", return_value=16.0):
            preset = client.recommend_image_preset("cinematic")
            assert preset == "cinematic"

    def test_recommend_image_preset_film(self):
        """recommend_image_preset returns cinematic for film intent."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_vram_gb", return_value=14.0):
            preset = client.recommend_image_preset("film")
            assert preset == "cinematic"

    def test_recommend_image_preset_landscape(self):
        """recommend_image_preset returns landscape for landscape intent."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_vram_gb", return_value=10.0):
            preset = client.recommend_image_preset("landscape")
            assert preset == "landscape"

    def test_recommend_video_preset_fallback_low_vram(self):
        """recommend_video_preset falls back for low VRAM."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_vram_gb", return_value=6.0):
            with patch("comfy_headless.video.get_recommended_preset", side_effect=ImportError):
                preset = client.recommend_video_preset()
                assert preset == "quick"

    def test_recommend_video_preset_fallback_medium_vram(self):
        """recommend_video_preset falls back for medium VRAM."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_vram_gb", return_value=10.0):
            with patch("comfy_headless.video.get_recommended_preset", side_effect=ImportError):
                preset = client.recommend_video_preset()
                assert preset == "wan_1.3b"

    def test_recommend_video_preset_fallback_high_vram(self):
        """recommend_video_preset falls back for high VRAM."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_vram_gb", return_value=14.0):
            with patch("comfy_headless.video.get_recommended_preset", side_effect=ImportError):
                preset = client.recommend_video_preset()
                assert preset == "ltx_standard"

    def test_recommend_video_preset_fallback_very_high_vram(self):
        """recommend_video_preset falls back for very high VRAM."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_vram_gb", return_value=20.0):
            with patch("comfy_headless.video.get_recommended_preset", side_effect=ImportError):
                preset = client.recommend_video_preset()
                assert preset == "hunyuan15_720p"

    def test_recommend_video_preset_fallback_max_vram(self):
        """recommend_video_preset falls back for max VRAM."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "get_vram_gb", return_value=32.0):
            with patch("comfy_headless.video.get_recommended_preset", side_effect=ImportError):
                preset = client.recommend_video_preset()
                assert preset == "hunyuan15_quality"


# ============================================================================
# QUEUE ERROR HANDLING TESTS (lines 644-648)
# ============================================================================


class TestQueueErrorHandling:
    """Test queue_prompt error handling."""

    def test_queue_prompt_exception_returns_none(self):
        """queue_prompt returns None on non-connection exception."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "_post") as mock_post:
            mock_post.side_effect = ValueError("JSON error")

            result = client.queue_prompt({"test": "workflow"})
            assert result is None

    def test_queue_prompt_connection_error_reraises(self):
        """queue_prompt re-raises ComfyUIConnectionError."""
        from comfy_headless.client import ComfyClient
        from comfy_headless.exceptions import ComfyUIConnectionError

        client = ComfyClient()

        with patch.object(client, "_post") as mock_post:
            mock_post.side_effect = ComfyUIConnectionError(
                message="Connection failed", url="http://localhost:8188"
            )

            with pytest.raises(ComfyUIConnectionError):
                client.queue_prompt({"test": "workflow"})


# ============================================================================
# PROGRESS CALLBACK TESTS (lines 696, 704-730)
# ============================================================================


class TestProgressCallback:
    """Test progress callback in wait_for_completion."""

    def test_progress_callback_on_error(self):
        """Progress callback called with error status."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        progress_calls = []

        def on_progress(progress, status):
            progress_calls.append((progress, status))

        with patch.object(client, "get_history") as mock_history:
            mock_history.return_value = {
                "test-123": {"status": {"status_str": "error", "completed": False}}
            }

            client.wait_for_completion("test-123", timeout=1.0, on_progress=on_progress)

        # Should have called on_progress with "Error" status
        assert any("Error" in call[1] for call in progress_calls)

    def test_progress_callback_execution_messages(self):
        """Progress callback handles execution messages."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        progress_calls = []

        def on_progress(progress, status):
            progress_calls.append((progress, status))

        call_count = [0]

        def mock_history_fn(prompt_id):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "test-123": {
                        "status": {
                            "status_str": "processing",
                            "completed": False,
                            "messages": [["execution_cached", {}], ["executing", "node_5"]],
                        }
                    }
                }
            else:
                return {"test-123": {"status": {"completed": True}}}

        with patch.object(client, "get_history", side_effect=mock_history_fn):
            client.wait_for_completion(
                "test-123", timeout=2.0, poll_interval=0.05, on_progress=on_progress
            )

        # Should have progress calls with status messages
        assert len(progress_calls) > 0

    def test_progress_callback_queued_status(self):
        """Progress callback handles queued status."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        progress_calls = []

        def on_progress(progress, status):
            progress_calls.append((progress, status))

        call_count = [0]

        def mock_history_fn(prompt_id):
            call_count[0] += 1
            if call_count[0] <= 2:
                return {"test-123": {"status": {"status_str": "queued", "completed": False}}}
            else:
                return {"test-123": {"status": {"completed": True}}}

        with patch.object(client, "get_history", side_effect=mock_history_fn):
            client.wait_for_completion(
                "test-123", timeout=2.0, poll_interval=0.05, on_progress=on_progress
            )

        # Should have progress calls with "Queued" status
        assert any("Queued" in str(call[1]) for call in progress_calls)

    def test_progress_callback_queue_position(self):
        """Progress callback shows queue position."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        progress_calls = []

        def on_progress(progress, status):
            progress_calls.append((progress, status))

        call_count = [0]

        def mock_history_fn(prompt_id):
            call_count[0] += 1
            if call_count[0] <= 2:
                return {}  # Not in history yet
            else:
                return {"test-123": {"status": {"completed": True}}}

        def mock_queue_fn():
            return {
                "queue_pending": [["id1", "test-123"]],  # Position 1
                "queue_running": [],
            }

        with patch.object(client, "get_history", side_effect=mock_history_fn):
            with patch.object(client, "get_queue", side_effect=mock_queue_fn):
                client.wait_for_completion(
                    "test-123", timeout=2.0, poll_interval=0.05, on_progress=on_progress
                )

        # Should have progress calls with queue position
        assert any(
            "Queue position" in str(call[1]) or "position" in str(call[1]).lower()
            for call in progress_calls
        )

    def test_progress_callback_running_status(self):
        """Progress callback shows running status."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        progress_calls = []

        def on_progress(progress, status):
            progress_calls.append((progress, status))

        call_count = [0]

        def mock_history_fn(prompt_id):
            call_count[0] += 1
            if call_count[0] == 1:
                return {}  # Not in history
            else:
                return {"test-123": {"status": {"completed": True}}}

        def mock_queue_fn():
            return {
                "queue_pending": [],
                "queue_running": [["id1", "test-123"]],  # Running
            }

        with patch.object(client, "get_history", side_effect=mock_history_fn):
            with patch.object(client, "get_queue", side_effect=mock_queue_fn):
                client.wait_for_completion(
                    "test-123", timeout=2.0, poll_interval=0.05, on_progress=on_progress
                )

        # Should have progress with "Starting" status
        assert any("Starting" in str(call[1]) for call in progress_calls)


# ============================================================================
# WORKFLOW COMPILER TESTS (lines 1064-1090)
# ============================================================================


class TestWorkflowCompiler:
    """Test WorkflowCompiler integration in generate_image."""

    def test_generate_image_uses_preset(self):
        """generate_image uses WorkflowCompiler with preset."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch("comfy_headless.workflows.compile_workflow") as mock_compile:
                    mock_result = MagicMock()
                    mock_result.is_valid = True
                    mock_result.workflow = {"3": {"inputs": {"seed": 42}}}
                    mock_compile.return_value = mock_result

                    result = client.generate_image(prompt="a sunset", preset="quality", wait=False)

                    assert result["success"] is True
                    assert result["seed"] == 42

    def test_generate_image_invalid_preset_fallback(self):
        """generate_image falls back when preset invalid."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch("comfy_headless.workflows.compile_workflow") as mock_compile:
                    mock_result = MagicMock()
                    mock_result.is_valid = False
                    mock_result.errors = ["Invalid preset"]
                    mock_compile.return_value = mock_result

                    result = client.generate_image(prompt="a sunset", preset="quality", wait=False)

                    # Should still succeed using legacy builder
                    assert result["success"] is True

    def test_generate_image_unknown_preset(self):
        """generate_image handles unknown preset."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                result = client.generate_image(
                    prompt="a sunset", preset="nonexistent_preset", wait=False
                )

                # Should use legacy builder
                assert result["success"] is True

    def test_generate_image_compiler_exception(self):
        """generate_image handles compiler exception."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch("comfy_headless.workflows.compile_workflow") as mock_compile:
                    mock_compile.side_effect = Exception("Compiler error")

                    result = client.generate_image(prompt="a sunset", preset="quality", wait=False)

                    # Should fall back to legacy builder
                    assert result["success"] is True


# ============================================================================
# HISTORY EXTRACTION TESTS (lines 1120-1152)
# ============================================================================


class TestHistoryExtraction:
    """Test history extraction in generate_image."""

    def test_generate_image_extracts_images(self):
        """generate_image extracts images from history."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch.object(client, "wait_for_completion") as mock_wait:
                    mock_wait.return_value = {
                        "status": {"completed": True},
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "img1.png", "subfolder": "", "type": "output"},
                                    {
                                        "filename": "img2.png",
                                        "subfolder": "batch",
                                        "type": "output",
                                    },
                                ]
                            }
                        },
                    }

                    result = client.generate_image(prompt="a sunset", wait=True)

                    assert result["success"] is True
                    assert len(result["images"]) == 2
                    assert result["images"][0]["filename"] == "img1.png"

    def test_generate_image_handles_error_status(self):
        """generate_image handles error status in history."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch.object(client, "wait_for_completion") as mock_wait:
                    mock_wait.return_value = {
                        "status": {"status_str": "error", "messages": [["Node 5 failed", {}]]}
                    }

                    result = client.generate_image(prompt="a sunset", wait=True)

                    assert result["success"] is False
                    assert "error" in result["error"].lower() or "Node" in result["error"]

    def test_generate_image_no_images_produced(self):
        """generate_image handles case where no images produced."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch.object(client, "wait_for_completion") as mock_wait:
                    mock_wait.return_value = {
                        "status": {"completed": True},
                        "outputs": {},  # No outputs
                    }

                    result = client.generate_image(prompt="a sunset", wait=True)

                    assert result["success"] is False
                    assert len(result["images"]) == 0

    def test_generate_image_timeout(self):
        """generate_image handles timeout."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch.object(client, "wait_for_completion", return_value=None):
                    result = client.generate_image(prompt="a sunset", wait=True, timeout=1.0)

                    assert result["success"] is False
                    assert "timed out" in result["error"].lower()


# ============================================================================
# VIDEO GENERATION FALLBACK TESTS (lines 1391-1408)
# ============================================================================


class TestVideoGenerationFallback:
    """Test video generation fallback to legacy builder."""

    def test_generate_video_uses_video_module(self):
        """generate_video uses video module when available."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch("comfy_headless.video.build_video_workflow") as mock_build:
                    mock_build.return_value = {
                        "7": {"class_type": "KSampler", "inputs": {"seed": 42}}
                    }

                    result = client.generate_video(
                        prompt="a cat walking", preset="quick", wait=False
                    )

                    assert result["success"] is True

    def test_generate_video_fallback_to_legacy(self):
        """generate_video falls back to legacy builder on error."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch("comfy_headless.video.build_video_workflow") as mock_build:
                    mock_build.side_effect = Exception("Video module error")

                    with patch.object(client, "build_video_workflow") as mock_legacy:
                        mock_legacy.return_value = {"7": {"inputs": {"seed": 99}}}

                        result = client.generate_video(
                            prompt="a cat walking", preset="quick", wait=False
                        )

                        # Should use legacy builder
                        mock_legacy.assert_called_once()
                        assert result["success"] is True

    def test_generate_video_extracts_videos(self):
        """generate_video extracts videos from history."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch("comfy_headless.video.build_video_workflow") as mock_build:
                    mock_build.return_value = {"7": {"inputs": {"seed": 42}}}

                    with patch.object(client, "wait_for_completion") as mock_wait:
                        mock_wait.return_value = {
                            "status": {"completed": True},
                            "outputs": {
                                "9": {
                                    "gifs": [
                                        {"filename": "video.mp4", "subfolder": "", "type": "output"}
                                    ]
                                }
                            },
                        }

                        result = client.generate_video(
                            prompt="a cat walking", preset="quick", wait=True
                        )

                        assert result["success"] is True
                        assert len(result["videos"]) == 1
                        assert result["videos"][0]["filename"] == "video.mp4"

    def test_generate_video_extracts_from_videos_key(self):
        """generate_video extracts from 'videos' key as well."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch("comfy_headless.video.build_video_workflow") as mock_build:
                    mock_build.return_value = {"7": {"inputs": {"seed": 42}}}

                    with patch.object(client, "wait_for_completion") as mock_wait:
                        mock_wait.return_value = {
                            "status": {"completed": True},
                            "outputs": {
                                "9": {
                                    "videos": [
                                        {
                                            "filename": "output.mp4",
                                            "subfolder": "",
                                            "type": "output",
                                        }
                                    ]
                                }
                            },
                        }

                        result = client.generate_video(prompt="a cat walking", wait=True)

                        assert result["success"] is True
                        assert len(result["videos"]) == 1

    def test_generate_video_with_overrides(self):
        """generate_video passes overrides to video module."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "ensure_online"):
            with patch.object(client, "queue_prompt", return_value="prompt-123"):
                with patch("comfy_headless.video.build_video_workflow") as mock_build:
                    mock_build.return_value = {"7": {"inputs": {"seed": 42}}}

                    client.generate_video(
                        prompt="a cat walking",
                        width=640,
                        height=480,
                        frames=24,
                        fps=12,
                        steps=30,
                        cfg=8.0,
                        seed=12345,
                        motion_scale=1.2,
                        wait=False,
                    )

                    # Verify overrides were passed
                    call_kwargs = mock_build.call_args.kwargs
                    assert call_kwargs.get("width") == 640
                    assert call_kwargs.get("height") == 480
                    assert call_kwargs.get("frames") == 24


# ============================================================================
# BATCH GENERATION TESTS (additional coverage)
# ============================================================================


class TestBatchGenerationExtended:
    """Extended batch generation tests."""

    def test_batch_with_partial_seeds(self):
        """Batch handles partial seed list."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "generate_image") as mock_gen:
            mock_gen.return_value = {
                "success": True,
                "images": [{"filename": "test.png"}],
                "error": None,
                "seed": 123,
                "prompt_id": "abc",
            }

            result = client.generate_batch(
                prompts=["prompt1", "prompt2", "prompt3"],
                seeds=[100],  # Only one seed, should pad with -1
                check_vram=False,
            )

            assert result["success"] is True
            assert mock_gen.call_count == 3

    def test_batch_with_progress_callback(self):
        """Batch calls progress callback."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        progress_calls = []

        def on_progress(idx, total, progress, status):
            progress_calls.append((idx, total, progress, status))

        with patch.object(client, "generate_image") as mock_gen:
            mock_gen.return_value = {
                "success": True,
                "images": [{"filename": "test.png"}],
                "error": None,
                "seed": 123,
                "prompt_id": "abc",
            }

            client.generate_batch(
                prompts=["p1", "p2"], check_vram=False, on_progress=on_progress
            )

            assert len(progress_calls) > 0
            # Should have calls for each prompt
            assert any(call[0] == 0 for call in progress_calls)
            assert any(call[0] == 1 for call in progress_calls)

    def test_batch_handles_exception_in_item(self):
        """Batch handles exception in individual item."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        call_count = [0]

        def mock_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Generation failed")
            return {
                "success": True,
                "images": [{"filename": "test.png"}],
                "error": None,
                "seed": 123,
                "prompt_id": "abc",
            }

        with patch.object(client, "generate_image", side_effect=mock_generate):
            result = client.generate_batch(prompts=["p1", "p2", "p3"], check_vram=False)

            assert result["success"] is False
            assert result["success_count"] == 2
            assert len(result["errors"]) == 1


# ============================================================================
# REQUEST METHOD EXTENDED TESTS
# ============================================================================


class TestRequestMethodExtended:
    """Extended tests for _request method."""

    def test_request_generic_exception(self):
        """_request handles generic exceptions."""
        from comfy_headless.client import ComfyClient
        from comfy_headless.exceptions import ComfyUIConnectionError

        client = ComfyClient()

        # Reset circuit breaker to ensure it's not open
        client._circuit.reset()

        with patch.object(client.session, "request") as mock_request:
            mock_request.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(ComfyUIConnectionError) as exc_info:
                client._request("GET", "/test")

            # Check that it's a connection error (could wrap the original)
            assert exc_info.value is not None

    def test_post_method(self):
        """_post delegates to _request."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "_request") as mock_request:
            mock_response = MagicMock()
            mock_request.return_value = mock_response

            client._post("/test", json={"data": 1})

            mock_request.assert_called_once_with("POST", "/test", json={"data": 1})

    def test_get_method(self):
        """_get delegates to _request."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "_request") as mock_request:
            mock_response = MagicMock()
            mock_request.return_value = mock_response

            client._get("/test", params={"q": "value"})

            mock_request.assert_called_once_with("GET", "/test", params={"q": "value"})


# ============================================================================
# CLEAR QUEUE EXTENDED TESTS
# ============================================================================


class TestClearQueueExtended:
    """Extended tests for clear_queue."""

    def test_clear_queue_exception(self):
        """clear_queue returns False on exception."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()

        with patch.object(client, "_post") as mock_post:
            mock_post.side_effect = Exception("Error")

            result = client.clear_queue()
            assert result is False
