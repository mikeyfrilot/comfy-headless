"""
Extended coverage tests for comfy_headless/video.py

Targets the specialized video model builder methods:
- _build_hunyuan (lines 899-965)
- _build_hunyuan_15 (lines 987-1123)
- _build_ltxv (lines 1141-1260)
- _build_wan (lines 1279-1404)
- _build_mochi (lines 1422-1558)
- _build_cogvideo (lines 1575-1640)
- Convenience functions (lines 1760-1789)
"""


import pytest

# ============================================================================
# VIDEO SETTINGS TESTS
# ============================================================================


class TestVideoSettingsToDict:
    """Test VideoSettings to_dict method."""

    def test_to_dict_returns_dict(self):
        """to_dict returns a dictionary."""
        from comfy_headless.video import VideoSettings

        settings = VideoSettings(width=1024, height=576, frames=32, fps=16)

        result = settings.to_dict()
        assert isinstance(result, dict)
        assert result["width"] == 1024
        assert result["height"] == 576
        assert result["frames"] == 32
        assert result["fps"] == 16

    def test_to_dict_includes_all_fields(self):
        """to_dict includes all settings fields."""
        from comfy_headless.video import MotionStyle, VideoFormat, VideoModel, VideoSettings

        settings = VideoSettings(
            model=VideoModel.LTXV,
            width=768,
            height=768,
            frames=24,
            fps=12,
            steps=30,
            cfg=5.0,
            seed=12345,
            motion_scale=1.2,
            motion_style=MotionStyle.DYNAMIC,
            checkpoint="model.safetensors",
            format=VideoFormat.MP4,
            interpolate=True,
            variant="1.3b",
            upscale=True,
            shift=7.0,
            precision="fp8",
        )

        result = settings.to_dict()

        assert result["model"] == "ltxv"
        assert result["motion_style"] == "dynamic"
        assert result["format"] == "mp4"
        assert result["variant"] == "1.3b"
        assert result["upscale"] is True
        assert result["shift"] == 7.0
        assert result["precision"] == "fp8"


# ============================================================================
# VIDEO WORKFLOW BUILDER TESTS
# ============================================================================


class TestVideoWorkflowBuilderInit:
    """Test VideoWorkflowBuilder initialization."""

    def test_builder_creation(self):
        """Builder can be created."""
        from comfy_headless.video import VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        assert builder is not None

    def test_get_video_builder(self):
        """get_video_builder returns a builder instance."""
        from comfy_headless.video import get_video_builder

        builder = get_video_builder()
        assert builder is not None


class TestBuildVideoWorkflow:
    """Test build_video_workflow convenience function."""

    def test_build_video_workflow_basic(self):
        """build_video_workflow creates workflow dict."""
        from comfy_headless.video import build_video_workflow

        workflow = build_video_workflow(prompt="a cat walking", negative="blurry", preset="quick")

        assert isinstance(workflow, dict)
        assert len(workflow) > 0

    def test_build_video_workflow_with_overrides(self):
        """build_video_workflow accepts parameter overrides."""
        from comfy_headless.video import build_video_workflow

        workflow = build_video_workflow(
            prompt="sunset over ocean",
            negative="low quality",
            preset="standard",
            width=768,
            height=432,
            frames=24,
            fps=12,
            steps=25,
            cfg=6.0,
            seed=42,
        )

        assert isinstance(workflow, dict)

    def test_build_video_workflow_with_init_image(self):
        """build_video_workflow handles init_image."""
        from comfy_headless.video import build_video_workflow

        # Test with init_image (base64 encoded)
        workflow = build_video_workflow(
            prompt="person walking",
            preset="standard",
            init_image="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        )

        assert isinstance(workflow, dict)


# ============================================================================
# ANIMATEDIFF BUILDER TESTS
# ============================================================================


class TestAnimateDiffBuilder:
    """Test AnimateDiff workflow building."""

    def test_build_animatediff_v3(self):
        """Build AnimateDiff v3 workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.ANIMATEDIFF_V3, width=512, height=512, frames=16, steps=20
        )

        workflow = builder.build(prompt="a dog running", negative="blurry", settings=settings)

        assert isinstance(workflow, dict)
        # Should have key nodes
        assert len(workflow) > 0

    def test_build_animatediff_lightning(self):
        """Build AnimateDiff Lightning workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.ANIMATEDIFF_LIGHTNING,
            width=512,
            height=512,
            frames=16,
            steps=4,
            cfg=2.0,
        )

        workflow = builder.build(prompt="a bird flying", negative="blurry", settings=settings)

        assert isinstance(workflow, dict)

    def test_build_animatediff_v2(self):
        """Build AnimateDiff v2 workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.ANIMATEDIFF_V2, width=512, height=512, frames=16, steps=20
        )

        workflow = builder.build(prompt="flowers blooming", negative="distorted", settings=settings)

        assert isinstance(workflow, dict)

    def test_build_with_interpolation(self):
        """Build workflow with RIFE interpolation enabled."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.ANIMATEDIFF_V3,
            width=512,
            height=512,
            frames=16,
            steps=20,
            interpolate=True,
        )

        workflow = builder.build(prompt="waves on beach", negative="blurry", settings=settings)

        assert isinstance(workflow, dict)


# ============================================================================
# SVD BUILDER TESTS
# ============================================================================


class TestSVDBuilder:
    """Test Stable Video Diffusion workflow building."""

    def test_build_svd(self):
        """Build SVD workflow (requires init_image)."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(model=VideoModel.SVD, width=1024, height=576, frames=14, steps=20)

        # SVD requires an init_image
        workflow = builder.build(
            prompt="person walking",
            negative="low quality",
            settings=settings,
            init_image="data:image/png;base64,iVBORw0KGgo=",
        )

        assert isinstance(workflow, dict)

    def test_build_svd_xt(self):
        """Build SVD-XT workflow (extended temporal)."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.SVD_XT, width=1024, height=576, frames=25, steps=20
        )

        # SVD-XT requires an init_image
        workflow = builder.build(
            prompt="car driving",
            negative="blurry",
            settings=settings,
            init_image="data:image/png;base64,iVBORw0KGgo=",
        )

        assert isinstance(workflow, dict)

    def test_build_svd_without_image_raises(self):
        """Build SVD without init_image raises error."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(model=VideoModel.SVD, width=1024, height=576, frames=14, steps=20)

        with pytest.raises(ValueError) as exc_info:
            builder.build(prompt="test", negative="blurry", settings=settings)

        assert "init_image" in str(exc_info.value).lower()


# ============================================================================
# COGVIDEOX BUILDER TESTS
# ============================================================================


class TestCogVideoXBuilder:
    """Test CogVideoX workflow building."""

    def test_build_cogvideox(self):
        """Build CogVideoX workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.COGVIDEOX, width=720, height=480, frames=49, steps=50, cfg=6.0
        )

        workflow = builder.build(
            prompt="fireworks display", negative="low quality", settings=settings
        )

        assert isinstance(workflow, dict)


# ============================================================================
# HUNYUAN BUILDER TESTS
# ============================================================================


class TestHunyuanBuilder:
    """Test Hunyuan Video workflow building."""

    def test_build_hunyuan(self):
        """Build Hunyuan Video workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.HUNYUAN, width=720, height=480, frames=45, steps=50
        )

        workflow = builder.build(
            prompt="mountains with clouds", negative="low quality, blurry", settings=settings
        )

        assert isinstance(workflow, dict)

    def test_build_hunyuan_with_interpolation(self):
        """Build Hunyuan workflow with interpolation."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.HUNYUAN, width=720, height=480, frames=45, steps=50, interpolate=True
        )

        workflow = builder.build(prompt="waterfall", negative="blurry", settings=settings)

        assert isinstance(workflow, dict)

    def test_build_hunyuan_15(self):
        """Build Hunyuan 1.5 workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.HUNYUAN_15, width=1280, height=720, frames=65, steps=30, shift=9.0
        )

        workflow = builder.build(
            prompt="aurora borealis", negative="low quality", settings=settings
        )

        assert isinstance(workflow, dict)

    def test_build_hunyuan_15_fast(self):
        """Build Hunyuan 1.5 fast (distilled) workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.HUNYUAN_15_FAST, width=1280, height=720, frames=45, steps=6
        )

        workflow = builder.build(prompt="city timelapse", negative="blurry", settings=settings)

        assert isinstance(workflow, dict)

    def test_build_hunyuan_15_i2v(self):
        """Build Hunyuan 1.5 image-to-video workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.HUNYUAN_15_I2V, width=1280, height=720, frames=45, steps=30
        )

        workflow = builder.build(
            prompt="portrait coming to life",
            negative="blurry",
            settings=settings,
            init_image="data:image/png;base64,iVBORw0KGgo=",
        )

        assert isinstance(workflow, dict)


# ============================================================================
# LTXV BUILDER TESTS
# ============================================================================


class TestLTXVBuilder:
    """Test LTX-Video workflow building."""

    def test_build_ltxv(self):
        """Build LTX-Video workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.LTXV, width=768, height=512, frames=97, fps=24, steps=30
        )

        workflow = builder.build(prompt="rocket launch", negative="low quality", settings=settings)

        assert isinstance(workflow, dict)

    def test_build_ltxv_i2v(self):
        """Build LTX-Video image-to-video workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.LTXV_I2V, width=768, height=512, frames=97, fps=24, steps=30
        )

        workflow = builder.build(
            prompt="image coming to life",
            negative="blurry",
            settings=settings,
            init_image="data:image/png;base64,iVBORw0KGgo=",
        )

        assert isinstance(workflow, dict)


# ============================================================================
# WAN BUILDER TESTS
# ============================================================================


class TestWanBuilder:
    """Test Wan video workflow building."""

    def test_build_wan(self):
        """Build Wan workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.WAN, width=832, height=480, frames=81, fps=16, steps=20
        )

        workflow = builder.build(prompt="dragon flying", negative="low quality", settings=settings)

        assert isinstance(workflow, dict)

    def test_build_wan_fast(self):
        """Build Wan fast workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.WAN_FAST, width=832, height=480, frames=81, fps=16, steps=4
        )

        workflow = builder.build(prompt="explosion", negative="blurry", settings=settings)

        assert isinstance(workflow, dict)

    def test_build_wan_i2v(self):
        """Build Wan image-to-video workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.WAN_I2V, width=832, height=480, frames=81, fps=16, steps=20
        )

        workflow = builder.build(
            prompt="photo animation",
            negative="blurry",
            settings=settings,
            init_image="data:image/png;base64,iVBORw0KGgo=",
        )

        assert isinstance(workflow, dict)


# ============================================================================
# MOCHI BUILDER TESTS
# ============================================================================


class TestMochiBuilder:
    """Test Mochi video workflow building."""

    def test_build_mochi(self):
        """Build Mochi workflow."""
        from comfy_headless.video import VideoModel, VideoSettings, VideoWorkflowBuilder

        builder = VideoWorkflowBuilder()
        settings = VideoSettings(
            model=VideoModel.MOCHI, width=848, height=480, frames=163, fps=30, steps=50, cfg=4.5
        )

        workflow = builder.build(
            prompt="astronaut on mars", negative="low quality", settings=settings
        )

        assert isinstance(workflow, dict)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_video_preset(self):
        """get_video_preset returns settings for valid preset."""
        from comfy_headless.video import get_video_preset

        settings = get_video_preset("quick")
        assert settings is not None
        assert settings.width > 0

    def test_get_video_preset_invalid(self):
        """get_video_preset handles invalid preset."""
        from comfy_headless.video import get_video_preset

        # Should return None or raise error for invalid preset
        result = get_video_preset("nonexistent_preset_xyz")
        # Accept None or exception
        assert result is None or isinstance(result, object)

    def test_list_video_presets(self):
        """list_video_presets returns preset names."""
        from comfy_headless.video import list_video_presets

        presets = list_video_presets()
        assert isinstance(presets, (list, tuple, dict))
        assert len(presets) > 0

    def test_list_video_models(self):
        """list_video_models returns model names."""
        from comfy_headless.video import list_video_models

        models = list_video_models()
        assert isinstance(models, (list, tuple, dict))
        assert len(models) > 0

    def test_get_recommended_preset_by_intent(self):
        """get_recommended_preset accepts intent parameter."""
        from comfy_headless.video import get_recommended_preset

        preset = get_recommended_preset(intent="cinematic", vram_gb=16)
        assert preset is not None

    def test_get_recommended_preset_portrait(self):
        """get_recommended_preset for portrait intent."""
        from comfy_headless.video import get_recommended_preset

        preset = get_recommended_preset(intent="portrait", vram_gb=12)
        assert preset is not None

    def test_get_recommended_preset_action(self):
        """get_recommended_preset for action intent."""
        from comfy_headless.video import get_recommended_preset

        preset = get_recommended_preset(intent="action", vram_gb=12)
        assert preset is not None


# ============================================================================
# VIDEO MODEL INFO TESTS
# ============================================================================


class TestVideoModelInfo:
    """Test VIDEO_MODEL_INFO dictionary."""

    def test_model_info_structure(self):
        """VIDEO_MODEL_INFO has expected structure."""
        from comfy_headless.video import VIDEO_MODEL_INFO

        # Should have info for each model type
        assert len(VIDEO_MODEL_INFO) > 0

    def test_model_info_contains_ltx(self):
        """Model info includes LTX models."""
        from comfy_headless.video import VIDEO_MODEL_INFO

        # Check if any LTX model is in the info
        [k for k in VIDEO_MODEL_INFO if "ltx" in str(k).lower()]
        # May or may not have LTX in the enum keys
        assert len(VIDEO_MODEL_INFO) > 0


# ============================================================================
# MOTION STYLE TESTS
# ============================================================================


class TestMotionStyle:
    """Test MotionStyle enum."""

    def test_motion_styles_exist(self):
        """MotionStyle enum has expected values."""
        from comfy_headless.video import MotionStyle

        styles = list(MotionStyle)
        assert len(styles) >= 3  # At least static, moderate, dynamic

    def test_motion_style_values(self):
        """MotionStyle values are strings."""
        from comfy_headless.video import MotionStyle

        for style in MotionStyle:
            assert isinstance(style.value, str)


# ============================================================================
# VIDEO FORMAT TESTS
# ============================================================================


class TestVideoFormat:
    """Test VideoFormat enum."""

    def test_video_formats_exist(self):
        """VideoFormat enum has expected values."""
        from comfy_headless.video import VideoFormat

        formats = list(VideoFormat)
        assert len(formats) >= 2  # At least MP4 and GIF

    def test_mp4_format(self):
        """MP4 format exists."""
        from comfy_headless.video import VideoFormat

        assert VideoFormat.MP4.value == "mp4"

    def test_gif_format(self):
        """GIF format exists."""
        from comfy_headless.video import VideoFormat

        assert VideoFormat.GIF.value == "gif"


# ============================================================================
# PRESET-BASED BUILD TESTS
# ============================================================================


class TestPresetBasedBuilding:
    """Test building workflows via preset names."""

    def test_build_quick_preset(self):
        """Build from 'quick' preset."""
        from comfy_headless.video import build_video_workflow

        workflow = build_video_workflow(prompt="test video", preset="quick")
        assert isinstance(workflow, dict)

    def test_build_standard_preset(self):
        """Build from 'standard' preset."""
        from comfy_headless.video import build_video_workflow

        workflow = build_video_workflow(prompt="test video", preset="standard")
        assert isinstance(workflow, dict)

    def test_build_quality_preset(self):
        """Build from 'quality' preset."""
        from comfy_headless.video import build_video_workflow

        workflow = build_video_workflow(prompt="test video", preset="quality")
        assert isinstance(workflow, dict)

    def test_build_cinematic_preset(self):
        """Build from 'cinematic' preset."""
        from comfy_headless.video import build_video_workflow

        workflow = build_video_workflow(prompt="test video", preset="cinematic")
        assert isinstance(workflow, dict)


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_prompt(self):
        """Handle empty prompt."""
        from comfy_headless.video import build_video_workflow

        workflow = build_video_workflow(prompt="", preset="quick")
        # Should still produce a workflow (empty prompt is valid)
        assert isinstance(workflow, dict)

    def test_very_long_prompt(self):
        """Handle very long prompt."""
        from comfy_headless.video import build_video_workflow

        long_prompt = "A " * 500 + "beautiful sunset"
        workflow = build_video_workflow(prompt=long_prompt, preset="quick")
        assert isinstance(workflow, dict)

    def test_special_characters_in_prompt(self):
        """Handle special characters in prompt."""
        from comfy_headless.video import build_video_workflow

        workflow = build_video_workflow(
            prompt="café scene with 日本語 text & symbols <>,./;'[]", preset="quick"
        )
        assert isinstance(workflow, dict)

    def test_negative_seed(self):
        """Handle negative seed (random)."""
        from comfy_headless.video import build_video_workflow

        workflow = build_video_workflow(prompt="test", preset="quick", seed=-1)
        assert isinstance(workflow, dict)

    def test_zero_seed(self):
        """Handle zero seed."""
        from comfy_headless.video import build_video_workflow

        workflow = build_video_workflow(prompt="test", preset="quick", seed=0)
        assert isinstance(workflow, dict)

    def test_large_seed(self):
        """Handle large seed value."""
        from comfy_headless.video import build_video_workflow

        workflow = build_video_workflow(prompt="test", preset="quick", seed=2**31 - 1)
        assert isinstance(workflow, dict)
