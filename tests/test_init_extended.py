"""
Extended tests for comfy_headless/__init__.py

Covers:
- Dynamic imports via __getattr__
- Feature-conditional exports
- Lazy loading behavior
- __all__ exports completeness
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


class TestDynamicImports:
    """Test __getattr__ lazy loading behavior."""

    def test_getattr_unknown_attribute_raises(self):
        """Unknown attributes raise AttributeError."""
        import comfy_headless
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = comfy_headless.nonexistent_attribute

    def test_getattr_launch_without_ui_feature(self):
        """launch() raises helpful ImportError when UI not installed."""
        import comfy_headless
        from comfy_headless.feature_flags import FEATURES

        # Only test if UI feature is not available
        if FEATURES.get("ui", False):
            pytest.skip("UI feature is installed, skipping unavailable test")

        with pytest.raises(ImportError, match=r"\[ui\].*Install with"):
            _ = comfy_headless.launch

    def test_lazy_import_entries_in_mapping(self):
        """Verify _LAZY_IMPORTS contains expected entries."""
        import comfy_headless

        expected_features = ["ai", "websocket", "ui", "health"]
        for feature in expected_features:
            # At least one entry should require each feature
            feature_entries = [
                name for name, (feat, _) in comfy_headless._LAZY_IMPORTS.items()
                if feat == feature
            ]
            assert len(feature_entries) > 0 or feature not in comfy_headless.FEATURES, \
                f"Expected lazy import entries for feature '{feature}'"


class TestFeatureConditionalExports:
    """Test exports that depend on feature availability."""

    def test_websockets_available_flag_exported(self):
        """WEBSOCKETS_AVAILABLE flag is always exported."""
        from comfy_headless import WEBSOCKETS_AVAILABLE
        assert isinstance(WEBSOCKETS_AVAILABLE, bool)

    def test_async_http_client_none_without_ai(self):
        """AsyncHttpClient is None when AI feature unavailable."""
        from comfy_headless.feature_flags import FEATURES
        import comfy_headless

        if not FEATURES.get("ai", False):
            assert comfy_headless.AsyncHttpClient is None
            assert comfy_headless.get_async_http_client is None

    def test_intelligence_stubs_without_ai(self):
        """Intelligence module stubs are None without AI feature."""
        from comfy_headless.feature_flags import FEATURES
        import comfy_headless

        if not FEATURES.get("ai", False):
            stub_names = [
                "PromptIntelligence", "PromptAnalysis", "EnhancedPrompt",
                "get_intelligence", "analyze_prompt", "enhance_prompt",
                "quick_enhance", "PromptCache", "get_prompt_cache",
            ]
            for name in stub_names:
                assert getattr(comfy_headless, name) is None, f"{name} should be None without AI"


class TestAllExportsComplete:
    """Test __all__ completeness and accessibility."""

    def test_all_exports_accessible(self):
        """All items in __all__ are actually accessible."""
        import comfy_headless

        for name in comfy_headless.__all__:
            # Should not raise AttributeError
            try:
                _ = getattr(comfy_headless, name)
            except ImportError:
                # ImportError is OK for optional features
                pass

    def test_version_exported(self):
        """__version__ is exported and valid."""
        import comfy_headless
        assert hasattr(comfy_headless, "__version__")
        assert isinstance(comfy_headless.__version__, str)
        # Semantic version format
        parts = comfy_headless.__version__.split(".")
        assert len(parts) >= 2, "Version should be semantic (x.y or x.y.z)"

    def test_core_classes_exported(self):
        """Core classes are always available."""
        from comfy_headless import (
            ComfyClient,
            Settings,
            ComfyHeadlessError,
            CircuitBreaker,
            TempFileManager,
        )

        assert ComfyClient is not None
        assert Settings is not None
        assert ComfyHeadlessError is not None
        assert CircuitBreaker is not None
        assert TempFileManager is not None


class TestModuleImportSafety:
    """Test module can be imported safely under various conditions."""

    def test_import_without_optional_deps(self):
        """Module imports successfully without optional dependencies."""
        # This test validates the import itself doesn't crash
        import comfy_headless
        assert comfy_headless is not None

    def test_features_dict_structure(self):
        """FEATURES dict has expected structure."""
        from comfy_headless import FEATURES

        expected_keys = {"ai", "websocket", "health", "ui", "validation", "observability"}
        assert expected_keys.issubset(set(FEATURES.keys()))

        for key, value in FEATURES.items():
            assert isinstance(value, bool), f"Feature '{key}' should be bool"

    def test_exception_classes_importable(self):
        """All exception classes are importable."""
        from comfy_headless import (
            ComfyHeadlessError,
            ComfyUIConnectionError,
            ComfyUIOfflineError,
            OllamaConnectionError,
            QueueError,
            GenerationTimeoutError,
            GenerationFailedError,
            WorkflowCompilationError,
            TemplateNotFoundError,
            RetryExhaustedError,
            CircuitOpenError,
            ValidationError,
            InvalidPromptError,
            InvalidParameterError,
            SecurityError,
        )

        # All should be subclasses of ComfyHeadlessError
        exception_classes = [
            ComfyUIConnectionError, ComfyUIOfflineError, OllamaConnectionError,
            QueueError, GenerationTimeoutError, GenerationFailedError,
            WorkflowCompilationError, TemplateNotFoundError,
            RetryExhaustedError, CircuitOpenError,
            ValidationError, InvalidPromptError, InvalidParameterError, SecurityError,
        ]

        for exc_class in exception_classes:
            assert issubclass(exc_class, ComfyHeadlessError)


class TestWebSocketImportHandling:
    """Test WebSocket module import handling."""

    def test_websocket_classes_defined_or_none(self):
        """WebSocket classes are either properly imported or None."""
        import comfy_headless

        ws_exports = ["ComfyWSClient", "WSProgress", "WSMessageType"]

        for name in ws_exports:
            value = getattr(comfy_headless, name, "MISSING")
            assert value != "MISSING", f"{name} should be exported"
            # Value is either None (not installed) or a class
            assert value is None or callable(value) or hasattr(value, "__members__")


class TestRetryModuleExports:
    """Test retry module exports."""

    def test_retry_exports_available(self):
        """Retry-related exports are available."""
        from comfy_headless import (
            retry_with_backoff,
            retry_on_exception,
            retry_async,
            CircuitBreaker,
            CircuitState,
            CircuitBreakerRegistry,
            circuit_registry,
            get_circuit_breaker,
            RateLimiter,
            OperationTimeoutError,
            with_timeout,
            async_timeout,
        )

        assert callable(retry_with_backoff)
        assert callable(retry_on_exception)
        assert callable(retry_async)
        assert CircuitBreaker is not None
        assert CircuitState is not None
        assert CircuitBreakerRegistry is not None
        assert circuit_registry is not None
        assert callable(get_circuit_breaker)
        assert RateLimiter is not None


class TestConfigExports:
    """Test configuration exports."""

    def test_config_exports_available(self):
        """Config-related exports are available."""
        from comfy_headless import (
            Settings,
            settings,
            get_temp_dir,
            get_settings,
            reload_settings,
            HttpConfig,
        )

        assert Settings is not None
        assert settings is not None
        assert callable(get_temp_dir)
        assert callable(get_settings)
        assert callable(reload_settings)
        assert HttpConfig is not None

    def test_settings_available(self):
        """settings and get_settings() both work."""
        from comfy_headless import settings, get_settings

        # Both should return valid Settings objects
        assert settings is not None
        assert get_settings() is not None


class TestWorkflowExports:
    """Test workflow module exports."""

    def test_workflow_exports_available(self):
        """Workflow-related exports are available."""
        from comfy_headless import (
            WorkflowCompiler,
            WorkflowTemplate,
            TemplateLibrary,
            get_compiler,
            get_library,
            compile_workflow,
            GENERATION_PRESETS,
            list_presets,
            WorkflowVersion,
            WorkflowSnapshot,
            SnapshotManager,
            get_snapshot_manager,
            WorkflowCache,
            get_workflow_cache,
            compute_workflow_hash,
            DAGValidator,
            validate_workflow_dag,
        )

        assert WorkflowCompiler is not None
        assert callable(get_compiler)
        assert callable(compile_workflow)
        assert isinstance(GENERATION_PRESETS, dict)


class TestVideoExports:
    """Test video module exports."""

    def test_video_exports_available(self):
        """Video-related exports are available."""
        from comfy_headless import (
            VideoWorkflowBuilder,
            VideoSettings,
            VideoModel,
            MotionStyle,
            get_video_builder,
            VIDEO_PRESETS,
            VIDEO_MODEL_INFO,
            build_video_workflow,
            list_video_presets,
            get_recommended_preset,
        )

        assert VideoWorkflowBuilder is not None
        assert VideoSettings is not None
        assert callable(get_video_builder)
        assert isinstance(VIDEO_PRESETS, dict)


class TestValidationExports:
    """Test validation module exports."""

    def test_validation_exports_available(self):
        """Validation-related exports are available."""
        from comfy_headless import (
            validate_prompt,
            sanitize_prompt,
            validate_dimensions,
            clamp_dimensions,
            validate_path,
            validate_in_range,
            validate_choice,
            validate_generation_params,
            validated_prompt,
            validated_dimensions,
        )

        assert callable(validate_prompt)
        assert callable(sanitize_prompt)
        assert callable(validate_dimensions)
        assert callable(clamp_dimensions)


class TestSecretsExports:
    """Test secrets module exports."""

    def test_secrets_exports_available(self):
        """Secrets-related exports are available."""
        from comfy_headless import (
            SecretValue,
            SecretsManager,
            get_secret,
            get_secret_str,
            get_secrets_manager,
            generate_token,
            generate_api_key,
            hash_secret,
            verify_hashed_secret,
            mask_url_credentials,
            redact_dict,
        )

        assert SecretValue is not None
        assert SecretsManager is not None
        assert callable(get_secret)
        assert callable(generate_token)
        assert callable(hash_secret)


class TestHelpSystemExports:
    """Test help system exports."""

    def test_help_exports_available(self):
        """Help-related exports are available."""
        from comfy_headless import (
            HelpLevel,
            HelpTopic,
            get_help,
            get_help_for_error,
            list_topics,
            search_help,
            set_help_level,
            get_help_level,
            format_quick_help,
            format_help_list,
        )

        assert HelpLevel is not None
        assert callable(get_help)
        assert callable(search_help)


class TestLoggingExports:
    """Test logging module exports."""

    def test_logging_exports_available(self):
        """Logging-related exports are available."""
        from comfy_headless import (
            get_logger,
            set_log_level,
            set_request_id,
            clear_request_id,
            LogContext,
            log_timing,
            traced_operation,
            get_tracer,
        )

        assert callable(get_logger)
        assert callable(set_log_level)
        assert callable(set_request_id)


class TestHealthExports:
    """Test health module exports."""

    def test_health_exports_available(self):
        """Health-related exports are available."""
        from comfy_headless import (
            HealthStatus,
            ComponentHealth,
            HealthReport,
            HealthChecker,
            HealthMonitor,
            check_health,
            full_health_check,
            is_healthy,
            get_health_checker,
        )

        assert HealthStatus is not None
        assert HealthChecker is not None
        assert callable(check_health)
        assert callable(is_healthy)


class TestCleanupExports:
    """Test cleanup module exports."""

    def test_cleanup_exports_available(self):
        """Cleanup-related exports are available."""
        from comfy_headless import (
            TempFileManager,
            CleanupThread,
            get_temp_manager,
            cleanup_temp_files,
            cleanup_all,
            register_shutdown_handlers,
            register_cleanup_callback,
            save_temp_image,
            save_temp_video,
        )

        assert TempFileManager is not None
        assert callable(get_temp_manager)
        assert callable(cleanup_temp_files)
        assert callable(save_temp_image)
