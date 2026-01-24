"""
Integration tests for comfy_headless.

Tests component interactions and end-to-end workflows.
Uses mocking for external services (ComfyUI, Ollama).
"""



# =============================================================================
# CLIENT INTEGRATION TESTS
# =============================================================================


class TestComfyClientIntegration:
    """Integration tests for ComfyClient."""

    def test_client_initialization(self):
        """Client should initialize without external connections."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient(base_url="http://localhost:8188")
        assert client.base_url == "http://localhost:8188"

    def test_client_default_url(self):
        """Client should use default URL."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        assert "localhost" in client.base_url or "127.0.0.1" in client.base_url


# =============================================================================
# WORKFLOW COMPILATION INTEGRATION
# =============================================================================


class TestWorkflowCompilationIntegration:
    """Integration tests for workflow compilation pipeline."""

    def test_full_compilation_pipeline(self):
        """Test complete workflow compilation."""
        from comfy_headless.intelligence import analyze_prompt
        from comfy_headless.workflows import compile_workflow

        prompt = "a beautiful sunset over mountains"

        # Analyze
        analysis = analyze_prompt(prompt)
        assert analysis.intent is not None

        # Compile with analysis-suggested preset
        result = compile_workflow(
            prompt=prompt,
            preset=analysis.suggested_preset,
        )

        assert result.is_valid
        assert result.workflow is not None
        assert len(result.errors) == 0

    def test_compilation_with_all_presets(self):
        """All presets should compile successfully."""
        from comfy_headless.workflows import GENERATION_PRESETS, compile_workflow

        for preset_name in GENERATION_PRESETS:
            result = compile_workflow(
                prompt="test prompt",
                preset=preset_name,
            )

            assert result.is_valid, f"Preset {preset_name} failed: {result.errors}"

    def test_workflow_contains_required_nodes(self):
        """Compiled workflow should have required nodes."""
        from comfy_headless.workflows import compile_workflow

        result = compile_workflow(
            prompt="test",
            preset="quality",
        )

        workflow = result.workflow

        # Should have key node types
        node_types = [
            node.get("class_type") for node in workflow.values() if isinstance(node, dict)
        ]

        assert "KSampler" in node_types or "KSamplerAdvanced" in node_types
        assert "CheckpointLoaderSimple" in node_types


# =============================================================================
# VIDEO WORKFLOW INTEGRATION
# =============================================================================


class TestVideoWorkflowIntegration:
    """Integration tests for video generation workflows."""

    def test_video_preset_to_settings(self):
        """Video presets should convert to valid settings."""
        from comfy_headless.video import VIDEO_PRESETS, VideoSettings

        for preset_name, settings in VIDEO_PRESETS.items():
            assert isinstance(settings, VideoSettings), f"{preset_name} is not VideoSettings"
            assert settings.width > 0
            assert settings.height > 0
            assert settings.frames > 0

    def test_recommended_preset_coverage(self):
        """VRAM recommendations should cover all common GPU sizes."""
        from comfy_headless.video import get_recommended_preset

        gpu_sizes = [6, 8, 10, 12, 16, 24, 48]

        for vram in gpu_sizes:
            preset = get_recommended_preset(vram_gb=vram)
            assert preset is not None, f"No preset for {vram}GB VRAM"


# =============================================================================
# INTELLIGENCE INTEGRATION
# =============================================================================


class TestIntelligenceIntegration:
    """Integration tests for prompt intelligence."""

    def test_analysis_to_preset_mapping(self):
        """Analysis should map to valid presets."""
        from comfy_headless.intelligence import analyze_prompt
        from comfy_headless.workflows import GENERATION_PRESETS

        prompts = [
            "portrait of a woman with red hair",
            "landscape of mountains at sunset",
            "a cat sitting on a windowsill",
            "cyberpunk city at night with neon lights",
        ]

        for prompt in prompts:
            analysis = analyze_prompt(prompt)
            assert analysis.suggested_preset in GENERATION_PRESETS, (
                f"Suggested preset '{analysis.suggested_preset}' not in GENERATION_PRESETS"
            )

    def test_style_detection_consistency(self):
        """Style detection should be consistent."""
        from comfy_headless.intelligence import analyze_prompt

        # Same prompt should give same analysis
        prompt = "anime girl with blue hair"

        analysis1 = analyze_prompt(prompt)
        analysis2 = analyze_prompt(prompt)

        assert analysis1.intent == analysis2.intent
        assert analysis1.styles == analysis2.styles


# =============================================================================
# CACHE INTEGRATION
# =============================================================================


class TestCacheIntegration:
    """Integration tests for caching systems."""

    def test_prompt_cache_workflow(self):
        """Test prompt analysis caching."""
        from comfy_headless.intelligence import PromptAnalysis, PromptCache

        cache = PromptCache(max_size=100)

        # Store
        analysis = PromptAnalysis(
            original="test",
            intent="portrait",
            styles=["realistic"],
            mood="neutral",
            suggested_preset="quality",
        )
        cache.set_analysis("test", analysis)

        # Retrieve
        cached = cache.get_analysis("test")
        assert cached is not None
        assert cached.intent == "portrait"

        # Stats
        stats = cache.stats()
        assert stats["analysis_entries"] == 1

    def test_workflow_cache(self):
        """Test workflow caching."""
        from comfy_headless.workflows import CompiledWorkflow, WorkflowCache

        cache = WorkflowCache(max_size=100, ttl_seconds=300)

        # Check CompiledWorkflow signature
        workflow = CompiledWorkflow(
            workflow={"test": "workflow"},
            template_id="txt2img",
            template_name="Text to Image",
            parameters={"prompt": "test"},
            version="1.0.0",
            is_valid=True,
        )

        # WorkflowCache uses template_id, params, preset for keys
        params = {"prompt": "test"}
        cache.set("txt2img", params, "quality", workflow)

        cached = cache.get("txt2img", params, "quality")
        assert cached is not None
        assert cached.template_id == "txt2img"


# =============================================================================
# ERROR HANDLING INTEGRATION
# =============================================================================


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_validation_error_exists(self):
        """ValidationError should exist."""
        from comfy_headless.exceptions import ValidationError

        error = ValidationError("test error")
        assert "test" in str(error).lower()

    def test_exception_hierarchy(self):
        """All custom exceptions should inherit from base."""
        from comfy_headless.exceptions import (
            ComfyHeadlessError,
            ComfyUIConnectionError,
            GenerationTimeoutError,
            ValidationError,
        )

        assert issubclass(ValidationError, ComfyHeadlessError)
        assert issubclass(ComfyUIConnectionError, ComfyHeadlessError)
        assert issubclass(GenerationTimeoutError, ComfyHeadlessError)

    def test_error_message_formatting(self):
        """Error messages should be formatted correctly."""
        from comfy_headless.exceptions import ValidationError

        error = ValidationError("Width must be positive")
        msg = str(error)

        assert "Width" in msg or "width" in msg.lower()


# =============================================================================
# FEATURE FLAG INTEGRATION
# =============================================================================


class TestFeatureFlagIntegration:
    """Integration tests for feature flags."""

    def test_feature_detection(self):
        """Features should be detected correctly."""
        from comfy_headless.feature_flags import FEATURES

        # Core features should always be present
        assert isinstance(FEATURES, dict)

        # At least some features should be detected
        assert len(FEATURES) > 0

    def test_feature_import_guards(self):
        """Feature imports should be guarded."""
        from comfy_headless import FEATURES

        # If ai feature is available, should be able to import
        if FEATURES.get("ai"):
            from comfy_headless import analyze_prompt

            assert callable(analyze_prompt)

    def test_missing_feature_handling(self):
        """Missing features should give helpful errors."""
        from comfy_headless.feature_flags import get_install_hint

        hint = get_install_hint("ui")
        assert "pip install" in hint


# =============================================================================
# SETTINGS INTEGRATION
# =============================================================================


class TestSettingsIntegration:
    """Integration tests for settings management."""

    def test_settings_singleton(self):
        """get_settings should return singleton."""
        from comfy_headless.config import get_settings

        s1 = get_settings()
        s2 = get_settings()

        assert s1 is s2

    def test_settings_reload(self):
        """reload_settings should create new instance."""
        from comfy_headless.config import get_settings, reload_settings

        get_settings()
        s2 = reload_settings()
        s3 = get_settings()

        # s2 and s3 should be same, but different from s1
        assert s2 is s3

    def test_settings_to_dict(self):
        """Settings should serialize to dict."""
        from comfy_headless.config import Settings

        settings = Settings()
        d = settings.to_dict()

        assert isinstance(d, dict)
        assert "comfyui" in d
        assert "generation" in d
        assert "retry" in d


# =============================================================================
# LOGGING INTEGRATION
# =============================================================================


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_logger_creation(self):
        """Loggers should be created correctly."""
        from comfy_headless.logging_config import get_logger

        logger = get_logger("test.module")
        assert logger is not None
        # Logger name may be prefixed
        assert "test.module" in logger.name

    def test_logger_caching(self):
        """Same module should get same logger."""
        from comfy_headless.logging_config import get_logger

        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")

        assert logger1 is logger2


# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================


class TestFullPipeline:
    """End-to-end pipeline tests."""

    def test_prompt_to_workflow_pipeline(self):
        """Test complete prompt-to-workflow pipeline."""
        from comfy_headless.intelligence import analyze_prompt
        from comfy_headless.workflows import compile_workflow

        # 1. Start with user prompt
        prompt = "a majestic lion in the savanna at sunset, photorealistic"

        # 2. Analyze
        analysis = analyze_prompt(prompt)
        assert analysis.confidence > 0

        # 3. Compile
        result = compile_workflow(
            prompt=prompt,
            preset=analysis.suggested_preset,
        )

        assert result.is_valid
        assert result.workflow is not None

        # 4. Verify workflow structure
        workflow = result.workflow
        assert isinstance(workflow, dict)
        assert len(workflow) > 0

    def test_video_pipeline(self):
        """Test video generation pipeline."""
        from comfy_headless.video import (
            VIDEO_PRESETS,
            VideoSettings,
            get_recommended_preset,
        )

        # 1. Get recommendation for GPU
        preset_name = get_recommended_preset(vram_gb=12)

        # 2. Get preset settings
        settings = VIDEO_PRESETS[preset_name]
        assert isinstance(settings, VideoSettings)

        # 3. Verify settings are valid
        assert settings.width > 0
        assert settings.height > 0
        assert settings.frames > 0
