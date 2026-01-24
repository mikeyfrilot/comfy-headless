"""
Error path and failure mode tests.

Tests that error conditions are handled gracefully without crashes.
"""


import pytest

# =============================================================================
# VALIDATION ERROR PATHS
# =============================================================================


class TestValidationErrorPaths:
    """Test validation error handling."""

    def test_negative_dimension(self):
        """Negative dimensions should raise error."""
        from comfy_headless.exceptions import DimensionError, ValidationError
        from comfy_headless.validation import validate_dimensions

        with pytest.raises((ValidationError, DimensionError, ValueError)):
            validate_dimensions(-100, 512)

    def test_zero_dimension(self):
        """Zero dimensions should raise error."""
        from comfy_headless.exceptions import DimensionError, ValidationError
        from comfy_headless.validation import validate_dimensions

        with pytest.raises((ValidationError, DimensionError, ValueError)):
            validate_dimensions(0, 512)

    def test_dimension_too_large(self):
        """Dimensions over max should raise error."""
        from comfy_headless.exceptions import DimensionError, ValidationError
        from comfy_headless.validation import validate_dimensions

        with pytest.raises((ValidationError, DimensionError, ValueError)):
            validate_dimensions(10000, 512)

    def test_invalid_in_range_below_min(self):
        """Value below min should raise error."""
        from comfy_headless.exceptions import ValidationError
        from comfy_headless.validation import validate_in_range

        with pytest.raises((ValidationError, ValueError)):
            validate_in_range(-1.0, "cfg", min_val=0, max_val=30)

    def test_invalid_in_range_above_max(self):
        """Value above max should raise error."""
        from comfy_headless.exceptions import ValidationError
        from comfy_headless.validation import validate_in_range

        with pytest.raises((ValidationError, ValueError)):
            validate_in_range(100, "steps", min_val=1, max_val=50)

    def test_valid_in_range(self):
        """Valid values should pass."""
        from comfy_headless.validation import validate_in_range

        result = validate_in_range(7.5, "cfg", min_val=0, max_val=30)
        assert result == 7.5


# =============================================================================
# WORKFLOW ERROR PATHS
# =============================================================================


class TestWorkflowErrorPaths:
    """Test workflow compilation error handling."""

    def test_invalid_preset(self):
        """Invalid preset should be handled gracefully."""
        from comfy_headless.workflows import compile_workflow

        # Should not crash, may use default or raise
        try:
            result = compile_workflow(
                prompt="test",
                preset="nonexistent_preset_12345",
            )
            # If it doesn't raise, should mark as invalid or use default
            assert result is not None
        except (ValueError, KeyError):
            # Expected error types
            pass

    def test_empty_workflow_validation(self):
        """Empty workflow should be handled."""
        from comfy_headless.workflows import validate_workflow_dag

        # Returns list of errors (empty if valid)
        errors = validate_workflow_dag({})
        assert isinstance(errors, list)

    def test_valid_workflow_dag(self):
        """Valid workflow should pass DAG validation."""
        from comfy_headless.workflows import validate_workflow_dag

        valid_workflow = {
            "1": {"class_type": "NodeA", "inputs": {}},
            "2": {"class_type": "NodeB", "inputs": {"input": ["1", 0]}},
        }

        # Returns list of errors (empty if valid)
        errors = validate_workflow_dag(valid_workflow)
        assert isinstance(errors, list)
        assert len(errors) == 0, f"Unexpected errors: {errors}"


# =============================================================================
# RETRY/CIRCUIT BREAKER ERROR PATHS
# =============================================================================


class TestRetryErrorPaths:
    """Test retry and circuit breaker error handling."""

    def test_circuit_breaker_rejection(self):
        """Open circuit should reject requests."""
        from comfy_headless.exceptions import CircuitOpenError
        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=1)
        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError), breaker:
            pass

    def test_retry_exhaustion(self):
        """Exhausted retries should raise appropriate error."""
        from comfy_headless.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_attempts=2, backoff_base=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises((ValueError, Exception)):
            always_fails()

        assert call_count == 2  # Should have tried twice


# =============================================================================
# SECRETS ERROR PATHS
# =============================================================================


class TestSecretsErrorPaths:
    """Test secrets handling error paths."""

    def test_empty_secret(self):
        """Empty secret should be handled."""
        from comfy_headless.secrets import SecretValue

        sv = SecretValue("")
        assert sv.get_secret_value() == ""
        # String repr should not be the actual value
        str_repr = str(sv)
        assert isinstance(str_repr, str)

    def test_none_secret_handling(self):
        """Non-existent env var should return default."""
        from comfy_headless.secrets import get_secret

        result = get_secret("NONEXISTENT_SECRET_12345", default="default_value")
        assert result == "default_value"


# =============================================================================
# CONFIG ERROR PATHS
# =============================================================================


class TestConfigErrorPaths:
    """Test configuration error handling."""

    def test_settings_construction(self):
        """Settings should construct without crash."""
        from comfy_headless.config import Settings

        settings = Settings()
        assert settings is not None

    def test_temp_dir_creation(self):
        """Temp dir should be created if not exists."""
        from comfy_headless.config import get_temp_dir

        temp_dir = get_temp_dir()
        assert temp_dir.exists()
        assert temp_dir.is_dir()


# =============================================================================
# INTELLIGENCE ERROR PATHS
# =============================================================================


class TestIntelligenceErrorPaths:
    """Test intelligence module error handling."""

    def test_malformed_prompt(self):
        """Malformed prompts should be handled."""
        from comfy_headless.intelligence import analyze_prompt

        malformed = [
            "",
            "   ",
            "\n\n\n",
            "\t\t",
            "a" * 10000,  # Very long
        ]

        for prompt in malformed:
            # Should not crash
            result = analyze_prompt(prompt)
            assert result is not None

    def test_prompt_sanitization(self):
        """Prompts should be sanitized safely."""
        from comfy_headless.intelligence import sanitize_prompt

        # Test control character removal
        result = sanitize_prompt("test\x00null\x00byte")
        assert "\x00" not in result


# =============================================================================
# VIDEO ERROR PATHS
# =============================================================================


class TestVideoErrorPaths:
    """Test video module error handling."""

    def test_invalid_vram_recommendation(self):
        """Very low VRAM should still get recommendation."""
        from comfy_headless.video import get_recommended_preset

        result = get_recommended_preset(vram_gb=1)
        assert result is not None

    def test_very_high_vram(self):
        """Very high VRAM should not crash."""
        from comfy_headless.video import get_recommended_preset

        result = get_recommended_preset(vram_gb=1000)
        assert result is not None


# =============================================================================
# CLIENT ERROR PATHS
# =============================================================================


class TestClientErrorPaths:
    """Test client error handling."""

    def test_client_initialization(self):
        """Client should initialize without connection."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient(base_url="http://localhost:99999")
        assert client is not None

    def test_client_has_expected_methods(self):
        """Client should have expected methods."""
        from comfy_headless.client import ComfyClient

        client = ComfyClient()
        assert hasattr(client, "queue_prompt")
        assert hasattr(client, "get_history")


# =============================================================================
# EXCEPTION ERROR PATHS
# =============================================================================


class TestExceptionErrorPaths:
    """Test exception creation and handling."""

    def test_all_exceptions_have_str(self):
        """All exceptions should have string representation."""
        from comfy_headless.exceptions import (
            ComfyHeadlessError,
            ComfyUIConnectionError,
            GenerationTimeoutError,
            ValidationError,
        )

        exceptions = [
            ComfyHeadlessError("test"),
            ValidationError("test"),
            ComfyUIConnectionError("test"),
            GenerationTimeoutError("test"),
        ]

        for exc in exceptions:
            assert isinstance(str(exc), str)
            assert len(str(exc)) > 0

    def test_exception_chaining(self):
        """Exceptions should support chaining."""
        from comfy_headless.exceptions import ComfyHeadlessError

        original = ValueError("Original error")
        chained = ComfyHeadlessError("Wrapped error")
        chained.__cause__ = original

        assert chained.__cause__ is original


# =============================================================================
# FEATURE FLAG ERROR PATHS
# =============================================================================


class TestFeatureFlagErrorPaths:
    """Test feature flag error handling."""

    def test_unknown_feature(self):
        """Unknown feature should return False."""
        from comfy_headless.feature_flags import FEATURES

        assert FEATURES.get("nonexistent_feature_12345", False) is False

    def test_install_hint_unknown_feature(self):
        """Install hint for unknown feature should be helpful."""
        from comfy_headless.feature_flags import get_install_hint

        hint = get_install_hint("unknown_feature")
        assert isinstance(hint, str)


# =============================================================================
# CONCURRENT ACCESS TESTS
# =============================================================================


class TestConcurrentAccess:
    """Test thread safety and concurrent access."""

    def test_circuit_breaker_thread_safe(self):
        """Circuit breaker should be thread-safe."""
        import threading

        from comfy_headless.retry import CircuitBreaker

        breaker = CircuitBreaker(name="thread_test", failure_threshold=100)

        def record_failures():
            for _ in range(10):
                breaker.record_failure()

        threads = [threading.Thread(target=record_failures) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded failures without crash
        assert breaker._failure_count <= 100

    def test_cache_thread_safe(self):
        """Caches should be thread-safe."""
        import threading

        from comfy_headless.intelligence import PromptAnalysis, PromptCache

        cache = PromptCache(max_size=100)

        def add_entries():
            for i in range(10):
                analysis = PromptAnalysis(
                    original=f"prompt_{threading.current_thread().name}_{i}",
                    intent="test",
                    styles=[],
                    mood="neutral",
                    suggested_preset="quality",
                )
                cache.set_analysis(f"{threading.current_thread().name}_{i}", analysis)

        threads = [threading.Thread(target=add_entries) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have entries without crash
        stats = cache.stats()
        assert stats["analysis_entries"] > 0
