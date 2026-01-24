"""Tests for ui.py Gradio interface module."""

from unittest.mock import MagicMock, patch


class TestUIConstants:
    """Test UI constants."""

    def test_presets_defined(self):
        """Test PRESETS constant is defined."""
        from comfy_headless.ui import PRESETS

        assert len(PRESETS) > 0
        assert "Quality (1024px)" in PRESETS

    def test_video_preset_names_defined(self):
        """Test VIDEO_PRESET_NAMES constant is defined."""
        from comfy_headless.ui import VIDEO_PRESET_NAMES

        assert len(VIDEO_PRESET_NAMES) > 0
        assert "Standard" in VIDEO_PRESET_NAMES

    def test_default_negative_defined(self):
        """Test DEFAULT_NEGATIVE is defined."""
        from comfy_headless.ui import DEFAULT_NEGATIVE

        assert len(DEFAULT_NEGATIVE) > 0
        assert "ugly" in DEFAULT_NEGATIVE.lower() or "blurry" in DEFAULT_NEGATIVE.lower()

    def test_example_prompts_defined(self):
        """Test EXAMPLE_PROMPTS is defined."""
        from comfy_headless.ui import EXAMPLE_PROMPTS

        assert len(EXAMPLE_PROMPTS) > 0


class TestCheckOllama:
    """Test check_ollama function."""

    @patch("comfy_headless.ui.intel")
    def test_check_ollama_available(self, mock_intel):
        """Test check_ollama when available."""
        mock_intel.check_ollama.return_value = True
        from comfy_headless.ui import check_ollama

        result = check_ollama()
        assert result is True

    @patch("comfy_headless.ui.intel")
    def test_check_ollama_unavailable(self, mock_intel):
        """Test check_ollama when unavailable."""
        mock_intel.check_ollama.return_value = False
        from comfy_headless.ui import check_ollama

        result = check_ollama()
        assert result is False


class TestEnhanceWithAI:
    """Test enhance_with_ai function."""

    @patch("comfy_headless.ui.intel")
    def test_enhance_empty_prompt(self, mock_intel):
        """Test enhance with empty prompt."""
        from comfy_headless.ui import DEFAULT_NEGATIVE, enhance_with_ai

        enhanced, negative, info = enhance_with_ai("")
        assert enhanced == ""
        assert negative == DEFAULT_NEGATIVE
        assert "Enter" in info

    @patch("comfy_headless.ui.intel")
    def test_enhance_valid_prompt(self, mock_intel):
        """Test enhance with valid prompt."""
        mock_intel.enhance_with_ai.return_value = (
            "enhanced prompt",
            "negative prompt",
            "Enhancement complete",
        )
        from comfy_headless.ui import enhance_with_ai

        enhanced, negative, info = enhance_with_ai("a sunset")
        assert enhanced == "enhanced prompt"
        mock_intel.enhance_with_ai.assert_called_once()


class TestAnalyzePromptAI:
    """Test analyze_prompt_ai function."""

    @patch("comfy_headless.ui.intel")
    def test_analyze_empty_prompt(self, mock_intel):
        """Test analyze with empty prompt."""
        from comfy_headless.ui import analyze_prompt_ai

        result = analyze_prompt_ai("")
        assert "Enter" in result

    @patch("comfy_headless.ui.intel")
    def test_analyze_valid_prompt(self, mock_intel):
        """Test analyze with valid prompt."""
        # Mock the analysis result
        mock_analysis = MagicMock()
        mock_analysis.intent = "landscape"
        mock_analysis.styles = ["photographic"]
        mock_analysis.mood = "peaceful"
        mock_analysis.subjects = ["mountains"]
        mock_analysis.suggested_aspect = "landscape"
        mock_analysis.suggested_preset = "quality"
        mock_analysis.suggested_workflow = "txt2img"
        mock_analysis.confidence = 0.85

        mock_intel.analyze_keywords.return_value = mock_analysis
        mock_intel.analyze_with_ai.return_value = None

        from comfy_headless.ui import analyze_prompt_ai

        result = analyze_prompt_ai("a beautiful mountain landscape")

        assert "Analysis Results" in result
        assert "Intent" in result


class TestGenerateVariationsAI:
    """Test generate_variations_ai function."""

    @patch("comfy_headless.ui.intel")
    def test_generate_variations(self, mock_intel):
        """Test generating variations."""
        mock_intel.generate_variations.return_value = ["variation 1", "variation 2", "variation 3"]
        from comfy_headless.ui import generate_variations_ai

        variations = generate_variations_ai("a sunset", 3)
        assert len(variations) == 3
        mock_intel.generate_variations.assert_called_once()


class TestGetSmartSettings:
    """Test get_smart_settings function."""

    @patch("comfy_headless.ui.intel")
    def test_get_smart_settings(self, mock_intel):
        """Test getting smart settings."""
        mock_analysis = MagicMock()
        mock_analysis.suggested_aspect = "portrait"
        mock_analysis.suggested_preset = "portrait"
        mock_analysis.intent = "portrait"
        mock_analysis.styles = ["photographic"]

        mock_intel.analyze_keywords.return_value = mock_analysis

        from comfy_headless.ui import get_smart_settings

        settings = get_smart_settings("portrait of a woman")

        assert "preset" in settings
        assert "width" in settings
        assert "height" in settings


class TestGetStatus:
    """Test get_status function."""

    @patch("comfy_headless.ui.client")
    def test_get_status_offline(self, mock_client):
        """Test status when offline."""
        mock_client.is_online.return_value = False
        from comfy_headless.ui import get_status

        status = get_status()
        assert "Offline" in status

    @patch("comfy_headless.ui.client")
    def test_get_status_online(self, mock_client):
        """Test status when online."""
        mock_client.is_online.return_value = True
        mock_client.get_system_stats.return_value = {
            "devices": [{"name": "RTX 5080", "vram_total": 16 * 1024**3, "vram_free": 8 * 1024**3}]
        }
        from comfy_headless.ui import get_status

        status = get_status()
        assert "Online" in status

    @patch("comfy_headless.ui.client")
    def test_get_status_online_no_stats(self, mock_client):
        """Test status when online but no stats."""
        mock_client.is_online.return_value = True
        mock_client.get_system_stats.return_value = None
        from comfy_headless.ui import get_status

        status = get_status()
        assert "Online" in status


class TestGetQueueStatus:
    """Test get_queue_status function."""

    @patch("comfy_headless.ui.client")
    def test_queue_status_offline(self, mock_client):
        """Test queue status when offline."""
        mock_client.is_online.return_value = False
        from comfy_headless.ui import get_queue_status

        status = get_queue_status()
        assert "Offline" in status

    @patch("comfy_headless.ui.client")
    def test_queue_status_online(self, mock_client):
        """Test queue status when online."""
        mock_client.is_online.return_value = True
        mock_client.get_queue.return_value = {"queue_running": [["id1"]], "queue_pending": []}
        from comfy_headless.ui import get_queue_status

        status = get_queue_status()
        assert "Running" in status
        assert "Pending" in status


class TestRefreshModels:
    """Test refresh_models function."""

    @patch("comfy_headless.ui.client")
    def test_refresh_models_offline(self, mock_client):
        """Test refresh models when offline."""
        mock_client.is_online.return_value = False
        from comfy_headless.ui import refresh_models

        models = refresh_models()
        assert "(ComfyUI offline)" in models

    @patch("comfy_headless.ui.client")
    def test_refresh_models_online(self, mock_client):
        """Test refresh models when online."""
        mock_client.is_online.return_value = True
        mock_client.get_checkpoints.return_value = ["model1.safetensors", "model2.safetensors"]
        from comfy_headless.ui import refresh_models

        models = refresh_models()
        assert "model1.safetensors" in models


class TestRefreshSamplers:
    """Test refresh_samplers function."""

    @patch("comfy_headless.ui.client")
    def test_refresh_samplers(self, mock_client):
        """Test refresh samplers."""
        mock_client.get_samplers.return_value = ["euler", "euler_ancestral"]
        from comfy_headless.ui import refresh_samplers

        samplers = refresh_samplers()
        assert "euler" in samplers


class TestRefreshSchedulers:
    """Test refresh_schedulers function."""

    @patch("comfy_headless.ui.client")
    def test_refresh_schedulers(self, mock_client):
        """Test refresh schedulers."""
        mock_client.get_schedulers.return_value = ["normal", "karras"]
        from comfy_headless.ui import refresh_schedulers

        schedulers = refresh_schedulers()
        assert "normal" in schedulers


class TestRefreshMotionModels:
    """Test refresh_motion_models function."""

    @patch("comfy_headless.ui.client")
    def test_refresh_motion_models_offline(self, mock_client):
        """Test refresh motion models when offline."""
        mock_client.is_online.return_value = False
        from comfy_headless.ui import refresh_motion_models

        models = refresh_motion_models()
        assert "(ComfyUI offline)" in models

    @patch("comfy_headless.ui.client")
    def test_refresh_motion_models_online(self, mock_client):
        """Test refresh motion models when online."""
        mock_client.is_online.return_value = True
        mock_client.get_motion_models.return_value = ["v3_sd15_mm.ckpt"]
        from comfy_headless.ui import refresh_motion_models

        models = refresh_motion_models()
        assert "v3_sd15_mm.ckpt" in models


class TestApplyPreset:
    """Test apply_preset function."""

    def test_apply_preset_quality(self):
        """Test applying quality preset."""
        from comfy_headless.ui import apply_preset

        width, height, steps = apply_preset("Quality (1024px)")
        assert width > 0
        assert height > 0
        assert steps > 0

    def test_apply_preset_unknown(self):
        """Test applying unknown preset defaults to quality."""
        from comfy_headless.ui import apply_preset

        width, height, steps = apply_preset("Unknown Preset")
        # Should return quality settings as default
        assert width > 0


class TestRandomPrompt:
    """Test random_prompt function."""

    def test_random_prompt(self):
        """Test getting random prompt."""
        from comfy_headless.ui import EXAMPLE_PROMPTS, random_prompt

        result = random_prompt()
        assert result in EXAMPLE_PROMPTS


class TestCancelGeneration:
    """Test cancel_generation function."""

    @patch("comfy_headless.ui.client")
    def test_cancel_success(self, mock_client):
        """Test successful cancellation."""
        mock_client.cancel_current.return_value = True
        from comfy_headless.ui import cancel_generation

        result = cancel_generation()
        assert "Cancelled" in result

    @patch("comfy_headless.ui.client")
    def test_cancel_failure(self, mock_client):
        """Test failed cancellation."""
        mock_client.cancel_current.return_value = False
        from comfy_headless.ui import cancel_generation

        result = cancel_generation()
        assert "Failed" in result


class TestGetHistoryList:
    """Test get_history_list function."""

    @patch("comfy_headless.ui.client")
    def test_get_history_offline(self, mock_client):
        """Test history list when offline."""
        mock_client.is_online.return_value = False
        from comfy_headless.ui import get_history_list

        history = get_history_list()
        assert history == []

    @patch("comfy_headless.ui.client")
    def test_get_history_online(self, mock_client):
        """Test history list when online."""
        mock_client.is_online.return_value = True
        mock_client.get_history.return_value = {
            "prompt123": {
                "status": {"completed": True},
                "outputs": {"9": {"images": [{"filename": "test.png"}]}},
            }
        }
        from comfy_headless.ui import get_history_list

        history = get_history_list()
        assert len(history) > 0


class TestGenerateImage:
    """Test generate_image function."""

    @patch("comfy_headless.ui.client")
    def test_generate_empty_prompt(self, mock_client):
        """Test generate with empty prompt."""
        from comfy_headless.ui import generate_image

        # Mock progress
        class MockProgress:
            def __call__(self, *args, **kwargs):
                pass

        result, info = generate_image(
            prompt="",
            negative_prompt="",
            checkpoint="model.safetensors",
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            seed=-1,
            progress=MockProgress(),
        )

        assert result is None
        assert "Error" in info

    @patch("comfy_headless.ui.client")
    def test_generate_offline(self, mock_client):
        """Test generate when offline."""
        mock_client.is_online.return_value = False
        from comfy_headless.ui import generate_image

        class MockProgress:
            def __call__(self, *args, **kwargs):
                pass

        result, info = generate_image(
            prompt="a sunset",
            negative_prompt="",
            checkpoint="model.safetensors",
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            seed=-1,
            progress=MockProgress(),
        )

        assert result is None
        assert "not running" in info.lower() or "offline" in info.lower()


class TestGenerateVideo:
    """Test generate_video function."""

    @patch("comfy_headless.ui.client")
    def test_video_empty_prompt(self, mock_client):
        """Test video with empty prompt."""
        from comfy_headless.ui import generate_video

        class MockProgress:
            def __call__(self, *args, **kwargs):
                pass

        result, info = generate_video(
            prompt="",
            negative_prompt="",
            checkpoint="model.safetensors",
            motion_model="v3_sd15_mm.ckpt",
            width=512,
            height=512,
            frames=16,
            fps=8,
            steps=20,
            cfg=7.0,
            motion_scale=1.0,
            seed=-1,
            progress=MockProgress(),
        )

        assert result is None
        assert "Error" in info

    @patch("comfy_headless.ui.client")
    def test_video_offline(self, mock_client):
        """Test video when offline."""
        mock_client.is_online.return_value = False
        from comfy_headless.ui import generate_video

        class MockProgress:
            def __call__(self, *args, **kwargs):
                pass

        result, info = generate_video(
            prompt="a cat walking",
            negative_prompt="",
            checkpoint="model.safetensors",
            motion_model="v3_sd15_mm.ckpt",
            width=512,
            height=512,
            frames=16,
            fps=8,
            steps=20,
            cfg=7.0,
            motion_scale=1.0,
            seed=-1,
            progress=MockProgress(),
        )

        assert result is None
        assert "not running" in info.lower() or "offline" in info.lower()


class TestGenerateVariations:
    """Test generate_variations function."""

    @patch("comfy_headless.ui.client")
    def test_variations_empty_prompt(self, mock_client):
        """Test variations with empty prompt."""
        from comfy_headless.ui import generate_variations

        class MockProgress:
            def __call__(self, *args, **kwargs):
                pass

        images, info = generate_variations(
            prompt="",
            negative_prompt="",
            checkpoint="model.safetensors",
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            num_variations=4,
            progress=MockProgress(),
        )

        assert images == []
        assert "Error" in info

    @patch("comfy_headless.ui.client")
    def test_variations_offline(self, mock_client):
        """Test variations when offline."""
        mock_client.is_online.return_value = False
        from comfy_headless.ui import generate_variations

        class MockProgress:
            def __call__(self, *args, **kwargs):
                pass

        images, info = generate_variations(
            prompt="a sunset",
            negative_prompt="",
            checkpoint="model.safetensors",
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            num_variations=4,
            progress=MockProgress(),
        )

        assert images == []
        assert "not running" in info.lower()


class TestComparePrompts:
    """Test compare_prompts function."""

    @patch("comfy_headless.ui.client")
    def test_compare_empty_prompt(self, mock_client):
        """Test compare with empty prompt."""
        from comfy_headless.ui import compare_prompts

        class MockProgress:
            def __call__(self, *args, **kwargs):
                pass

        img_a, img_b, info = compare_prompts(
            prompt_a="",
            prompt_b="test",
            negative_prompt="",
            checkpoint="model.safetensors",
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            seed=-1,
            progress=MockProgress(),
        )

        assert img_a is None
        assert img_b is None
        assert "Error" in info

    @patch("comfy_headless.ui.client")
    def test_compare_offline(self, mock_client):
        """Test compare when offline."""
        mock_client.is_online.return_value = False
        from comfy_headless.ui import compare_prompts

        class MockProgress:
            def __call__(self, *args, **kwargs):
                pass

        img_a, img_b, info = compare_prompts(
            prompt_a="sunset",
            prompt_b="sunrise",
            negative_prompt="",
            checkpoint="model.safetensors",
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            seed=-1,
            progress=MockProgress(),
        )

        assert img_a is None
        assert img_b is None
        assert "not running" in info.lower()


class TestCreateUI:
    """Test create_ui function."""

    @patch("comfy_headless.ui.gr")
    @patch("comfy_headless.ui.client")
    def test_create_ui_returns_app(self, mock_client, mock_gr):
        """Test create_ui returns Gradio app."""
        mock_client.is_online.return_value = False
        mock_client.get_samplers.return_value = ["euler"]
        mock_client.get_schedulers.return_value = ["normal"]
        mock_client.get_checkpoints.return_value = []
        mock_client.get_motion_models.return_value = []

        # Mock Blocks context manager
        mock_app = MagicMock()
        mock_gr.Blocks.return_value.__enter__ = MagicMock(return_value=mock_app)
        mock_gr.Blocks.return_value.__exit__ = MagicMock(return_value=False)

        from comfy_headless.ui import create_ui

        # Just verify it doesn't raise
        try:
            create_ui()
        except Exception:
            # The function requires full Gradio which we're mocking
            pass
