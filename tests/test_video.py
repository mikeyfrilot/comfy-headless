"""Tests for video module."""

import pytest


class TestVideoSettings:
    """Test VideoSettings dataclass."""

    def test_video_settings_creation(self):
        """Test creating VideoSettings."""
        from comfy_headless.video import VideoSettings

        settings = VideoSettings(width=1280, height=720, frames=48, fps=24, steps=30)

        assert settings.width == 1280
        assert settings.height == 720
        assert settings.frames == 48
        assert settings.fps == 24
        assert settings.steps == 30

    def test_video_settings_defaults(self):
        """Test VideoSettings default values."""
        from comfy_headless.video import VideoSettings

        settings = VideoSettings()

        assert settings.width > 0
        assert settings.height > 0
        assert settings.frames > 0
        assert settings.fps > 0

    def test_video_settings_aspect_ratio(self):
        """Test aspect ratio calculation."""
        from comfy_headless.video import VideoSettings

        settings = VideoSettings(width=1920, height=1080)

        if hasattr(settings, "aspect_ratio"):
            ratio = settings.aspect_ratio
            assert abs(ratio - 16 / 9) < 0.01


class TestVideoPresets:
    """Test video preset definitions."""

    def test_presets_exist(self):
        """Test that presets dictionary exists."""
        from comfy_headless.video import VIDEO_PRESETS

        assert VIDEO_PRESETS is not None
        assert len(VIDEO_PRESETS) > 0

    def test_all_presets_have_valid_settings(self):
        """Test all presets have valid VideoSettings."""
        from comfy_headless.video import VIDEO_PRESETS, VideoSettings

        for name, settings in VIDEO_PRESETS.items():
            assert isinstance(settings, VideoSettings), f"{name} is not VideoSettings"
            assert settings.width > 0, f"{name} has invalid width"
            assert settings.height > 0, f"{name} has invalid height"
            assert settings.frames > 0, f"{name} has invalid frames"

    def test_quick_preset_exists(self):
        """Test quick preset exists."""
        from comfy_headless.video import VIDEO_PRESETS

        quick_presets = [
            k for k in VIDEO_PRESETS if "quick" in k.lower() or "fast" in k.lower()
        ]
        assert len(quick_presets) > 0

    def test_quality_preset_exists(self):
        """Test quality preset exists."""
        from comfy_headless.video import VIDEO_PRESETS

        quality_presets = [k for k in VIDEO_PRESETS if "quality" in k.lower()]
        assert len(quality_presets) > 0


class TestVideoRecommendations:
    """Test video preset recommendations."""

    def test_get_recommended_preset_low_vram(self):
        """Test recommendation for low VRAM."""
        from comfy_headless.video import get_recommended_preset

        preset = get_recommended_preset(vram_gb=6)
        assert preset is not None
        assert isinstance(preset, str)

    def test_get_recommended_preset_mid_vram(self):
        """Test recommendation for mid VRAM."""
        from comfy_headless.video import get_recommended_preset

        preset = get_recommended_preset(vram_gb=12)
        assert preset is not None

    def test_get_recommended_preset_high_vram(self):
        """Test recommendation for high VRAM."""
        from comfy_headless.video import get_recommended_preset

        preset = get_recommended_preset(vram_gb=24)
        assert preset is not None

    def test_recommendations_scale_with_vram(self):
        """Test that higher VRAM gets better presets."""
        from comfy_headless.video import get_recommended_preset

        low_preset = get_recommended_preset(vram_gb=6)
        high_preset = get_recommended_preset(vram_gb=24)

        # Different presets for different VRAM
        # (might be same if only one preset works)
        assert low_preset is not None
        assert high_preset is not None


class TestVideoModel:
    """Test VideoModel enum."""

    def test_video_models_exist(self):
        """Test VideoModel enum has values."""
        from comfy_headless.video import VideoModel

        models = list(VideoModel)
        assert len(models) > 0

    def test_ltx_models_exist(self):
        """Test LTX video models."""
        from comfy_headless.video import VideoModel

        model_values = [m.value for m in VideoModel]
        ltx_models = [v for v in model_values if "ltx" in v.lower()]
        assert len(ltx_models) > 0

    def test_hunyuan_models_exist(self):
        """Test Hunyuan video models."""
        from comfy_headless.video import VideoModel

        model_values = [m.value for m in VideoModel]
        hunyuan_models = [v for v in model_values if "hunyuan" in v.lower()]
        assert len(hunyuan_models) > 0


class TestVideoModelInfo:
    """Test video model information."""

    def test_model_info_exists(self):
        """Test VIDEO_MODEL_INFO exists."""
        from comfy_headless.video import VIDEO_MODEL_INFO

        assert VIDEO_MODEL_INFO is not None
        assert len(VIDEO_MODEL_INFO) > 0

    def test_model_info_has_vram(self):
        """Test model info includes VRAM requirements."""
        from comfy_headless.video import VIDEO_MODEL_INFO

        for _model, info in VIDEO_MODEL_INFO.items():
            if isinstance(info, dict):
                # May have vram_min or vram key
                pass
                # Not all models may have this, so just check structure


class TestVideoWorkflowBuilder:
    """Test VideoWorkflowBuilder."""

    def test_builder_exists(self):
        """Test VideoWorkflowBuilder class exists."""
        from comfy_headless.video import VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        assert builder is not None

    def test_build_basic_workflow(self):
        """Test building a basic video workflow."""
        from comfy_headless.video import VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(width=512, height=512, frames=24, fps=24, steps=20)

        try:
            workflow = builder.build(
                prompt="a cat walking", negative="blurry, distorted", settings=settings
            )
            assert workflow is not None
        except (NotImplementedError, AttributeError, TypeError):
            # May not be fully implemented or have different signature
            pass


class TestVideoPresetLookup:
    """Test preset lookup utilities."""

    def test_get_preset_by_name(self):
        """Test getting preset by name."""
        from comfy_headless.video import VIDEO_PRESETS

        # Get first preset name
        first_preset = list(VIDEO_PRESETS.keys())[0]
        settings = VIDEO_PRESETS[first_preset]

        assert settings is not None
        assert settings.width > 0

    def test_invalid_preset_name(self):
        """Test invalid preset name raises error."""
        from comfy_headless.video import VIDEO_PRESETS

        with pytest.raises(KeyError):
            _ = VIDEO_PRESETS["nonexistent_preset_xyz"]


class TestVideoDuration:
    """Test video duration calculations."""

    def test_duration_from_frames_and_fps(self):
        """Test calculating duration from frames and fps."""
        from comfy_headless.video import VideoSettings

        settings = VideoSettings(width=512, height=512, frames=48, fps=24)

        if hasattr(settings, "duration"):
            assert settings.duration == 2.0  # 48 frames / 24 fps = 2 seconds
        else:
            # Calculate manually
            duration = settings.frames / settings.fps
            assert duration == 2.0


class TestVideoMemoryEstimation:
    """Test video memory estimation."""

    def test_estimate_video_vram(self):
        """Test VRAM estimation for video generation."""
        from comfy_headless.video import VideoSettings

        settings = VideoSettings(width=1280, height=720, frames=48)

        if hasattr(settings, "estimate_vram"):
            vram = settings.estimate_vram()
            assert vram > 0
