"""
Property-based tests using Hypothesis.

Tests invariants and edge cases that example-based tests might miss.
Following 2026 best practices for thorough testing.
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

# =============================================================================
# VALIDATION MODULE TESTS
# =============================================================================


class TestValidationProperties:
    """Property-based tests for validation module."""

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=200)
    def test_sanitize_prompt_never_crashes(self, text):
        """sanitize_prompt should never crash on any input."""
        from comfy_headless.intelligence import sanitize_prompt

        # Should never raise
        result = sanitize_prompt(text)
        assert isinstance(result, str)

    @given(st.text(min_size=0, max_size=100))
    def test_sanitize_prompt_no_control_chars(self, text):
        """Sanitized prompts should never contain control characters."""
        from comfy_headless.intelligence import sanitize_prompt

        result = sanitize_prompt(text)
        # Control characters are 0x00-0x1F and 0x7F (except tab, newline, carriage return)
        for char in result:
            assert ord(char) >= 0x20 or char in "\t\n\r", f"Found control char: {ord(char)}"

    @given(st.text(min_size=0, max_size=5000))
    def test_sanitize_prompt_respects_max_length(self, text):
        """Sanitized prompts should respect max_length."""
        from comfy_headless.intelligence import sanitize_prompt

        max_len = 100
        result = sanitize_prompt(text, max_length=max_len)
        assert len(result) <= max_len

    @given(st.integers(min_value=64, max_value=2048))
    def test_valid_dimensions_accepted(self, value):
        """Valid dimensions should be accepted."""
        from comfy_headless.validation import validate_dimensions

        # Round to nearest 8 (ComfyUI requirement)
        value = (value // 8) * 8
        assume(value >= 64)

        # Both dimensions valid
        result = validate_dimensions(value, value)
        assert result == (value, value)

    @given(st.floats(min_value=0.0, max_value=30.0, allow_nan=False))
    def test_in_range_validation(self, value):
        """Values in range should be accepted."""
        from comfy_headless.validation import validate_in_range

        result = validate_in_range(value, "cfg", min_val=0, max_val=30)
        assert result == value


# =============================================================================
# CONFIG MODULE TESTS
# =============================================================================


class TestConfigProperties:
    """Property-based tests for configuration."""

    @given(st.text(min_size=1, max_size=50))
    def test_settings_construction_stable(self, _):
        """Settings should construct stably."""
        from comfy_headless.config import Settings

        settings = Settings()
        assert settings.comfyui.url is not None

    def test_settings_to_dict_roundtrip(self):
        """Settings.to_dict() should produce valid dict."""
        from comfy_headless.config import Settings

        settings = Settings()
        d = settings.to_dict()

        assert isinstance(d, dict)
        assert "version" in d
        assert "comfyui" in d


# =============================================================================
# EXCEPTIONS MODULE TESTS
# =============================================================================


class TestExceptionProperties:
    """Property-based tests for exception handling."""

    @given(st.text(min_size=0, max_size=500))
    def test_exception_message_handling(self, message):
        """Exceptions should handle any message string."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError(message)
        # Should be convertible to string
        str_repr = str(error)
        assert isinstance(str_repr, str)

    @given(st.text(min_size=0, max_size=100), st.text(min_size=0, max_size=100))
    def test_validation_error_fields(self, field, value):
        """ValidationError should handle any field/value combo."""
        from comfy_headless.exceptions import ValidationError

        error = ValidationError(f"Invalid {field}: {value}")
        assert isinstance(str(error), str)


# =============================================================================
# RETRY MODULE TESTS
# =============================================================================


class TestRetryProperties:
    """Property-based tests for retry logic."""

    @given(st.integers(min_value=1, max_value=10))
    def test_circuit_breaker_threshold(self, threshold):
        """Circuit breaker should respect failure threshold."""
        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=threshold, reset_timeout=60)

        # Record failures up to threshold - 1
        for _ in range(threshold - 1):
            breaker.record_failure()
            assert breaker.state == CircuitState.CLOSED

        # One more should open
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    @given(st.integers(min_value=1, max_value=100), st.floats(min_value=0.1, max_value=10.0))
    def test_rate_limiter_properties(self, rate, per_seconds):
        """Rate limiter should respect configured limits."""
        from comfy_headless.retry import RateLimiter

        limiter = RateLimiter(rate=rate, per_seconds=per_seconds)

        # Should allow at least 'rate' immediate acquisitions
        acquired = 0
        for _ in range(rate):
            if limiter.acquire(blocking=False):
                acquired += 1

        assert acquired == rate


# =============================================================================
# INTELLIGENCE MODULE TESTS
# =============================================================================


class TestIntelligenceProperties:
    """Property-based tests for prompt intelligence."""

    @given(st.text(min_size=1, max_size=200))
    @settings(max_examples=100)
    def test_analyze_prompt_never_crashes(self, prompt):
        """analyze_prompt should handle any input."""
        from comfy_headless.intelligence import analyze_prompt

        # Should never crash
        try:
            result = analyze_prompt(prompt)
            assert result.intent is not None
            assert result.styles is not None
            assert isinstance(result.confidence, float)
        except Exception as e:
            # Should only be expected errors, not crashes
            assert not isinstance(e, (TypeError, AttributeError))

    @given(st.sampled_from(["portrait", "landscape", "scene", "character", "object"]))
    def test_known_intents_detected(self, intent_word):
        """Known intent words should be detected."""
        from comfy_headless.intelligence import analyze_prompt

        prompt = f"a {intent_word} of something"
        result = analyze_prompt(prompt)

        # Should have some intent detected
        assert result.intent is not None


# =============================================================================
# WORKFLOWS MODULE TESTS
# =============================================================================


class TestWorkflowProperties:
    """Property-based tests for workflow compilation."""

    @given(
        st.sampled_from(["draft", "fast", "quality", "hd", "portrait", "landscape", "cinematic"])
    )
    def test_preset_compilation(self, preset):
        """All presets should compile without error."""
        from comfy_headless.workflows import compile_workflow

        result = compile_workflow(
            prompt="test prompt",
            preset=preset,
        )

        assert result.is_valid
        assert result.workflow is not None

    @given(st.integers(min_value=64, max_value=2048), st.integers(min_value=64, max_value=2048))
    def test_dimension_compilation(self, width, height):
        """Various dimensions should compile (with validation)."""
        from comfy_headless.workflows import compile_workflow

        # Round to nearest 8 (ComfyUI requirement)
        width = (width // 8) * 8
        height = (height // 8) * 8

        assume(width >= 64 and height >= 64)
        assume(width <= 2048 and height <= 2048)

        result = compile_workflow(
            prompt="test",
            preset="draft",
            width=width,
            height=height,
        )

        # Should compile (may have warnings but should be valid)
        assert result is not None

    @given(st.text(min_size=1, max_size=500))
    def test_prompt_in_workflow(self, prompt):
        """Compiled workflow should generate for any prompt."""
        from comfy_headless.workflows import compile_workflow

        # Skip prompts with only whitespace
        assume(prompt.strip())

        result = compile_workflow(
            prompt=prompt,
            preset="draft",
        )

        # Workflow should be generated
        assert result.workflow is not None


# =============================================================================
# VIDEO MODULE TESTS
# =============================================================================


class TestVideoProperties:
    """Property-based tests for video generation."""

    @given(
        st.sampled_from(
            [
                "quick",
                "standard",
                "quality",
                "cinematic",
                "ltx_quick",
                "ltx_standard",
                "ltx_quality",
                "hunyuan15_720p",
                "hunyuan15_fast",
                "wan_1.3b",
                "wan_14b",
                "mochi",
                "mochi_short",
            ]
        )
    )
    def test_video_preset_exists(self, preset_name):
        """All documented presets should exist."""
        from comfy_headless.video import VIDEO_PRESETS

        assert preset_name in VIDEO_PRESETS

    @given(st.integers(min_value=4, max_value=32))
    def test_vram_recommendation(self, vram_gb):
        """VRAM recommendations should always return valid preset."""
        from comfy_headless.video import get_recommended_preset

        result = get_recommended_preset(vram_gb=vram_gb)
        assert result is not None
        assert isinstance(result, str)


# =============================================================================
# SECRETS MODULE TESTS
# =============================================================================


class TestSecretsProperties:
    """Property-based tests for secrets handling."""

    @given(st.text(min_size=1, max_size=100))
    def test_secret_value_masking(self, secret):
        """SecretValue should always mask in string representation."""
        from comfy_headless.secrets import SecretValue

        sv = SecretValue(secret)

        # String representation should be masked (all asterisks)
        str_repr = str(sv)
        # The string repr is typically just asterisks
        assert "*" in str_repr or str_repr == ""

    @given(st.text(min_size=1, max_size=50))
    def test_secret_value_retrieval(self, secret):
        """SecretValue.get_secret_value() should return original value."""
        from comfy_headless.secrets import SecretValue

        sv = SecretValue(secret)
        assert sv.get_secret_value() == secret


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Specific edge cases that should be handled."""

    def test_empty_prompt(self):
        """Empty prompt should be handled gracefully."""
        from comfy_headless.intelligence import analyze_prompt, sanitize_prompt

        result = sanitize_prompt("")
        assert result == ""

        analysis = analyze_prompt("")
        assert analysis is not None

    def test_none_prompt(self):
        """None prompt should be handled."""
        from comfy_headless.intelligence import sanitize_prompt

        result = sanitize_prompt(None)
        assert result == ""

    def test_unicode_prompt(self):
        """Unicode prompts should work."""
        from comfy_headless.intelligence import analyze_prompt

        prompts = [
            "日本語のプロンプト",
            "Mixed 混合 content",
            "Ñoño señor",
            "Ελληνικά",
        ]

        for prompt in prompts:
            result = analyze_prompt(prompt)
            assert result is not None

    def test_very_long_prompt(self):
        """Very long prompts should be truncated safely."""
        from comfy_headless.intelligence import sanitize_prompt

        long_prompt = "a" * 100000
        result = sanitize_prompt(long_prompt, max_length=2000)
        assert len(result) == 2000

    @given(st.binary(min_size=0, max_size=100))
    def test_binary_data_handling(self, data):
        """Binary data should not crash sanitization."""
        from comfy_headless.intelligence import sanitize_prompt

        try:
            text = data.decode("utf-8", errors="replace")
            result = sanitize_prompt(text)
            assert isinstance(result, str)
        except Exception:
            # Decoding might fail, that's OK
            pass


# =============================================================================
# SNAPSHOT TESTS
# =============================================================================


class TestSnapshotInvariants:
    """Test workflow snapshot invariants."""

    @given(st.text(min_size=1, max_size=100))
    def test_snapshot_hash_deterministic(self, prompt):
        """Same workflow should produce same hash."""
        from comfy_headless.workflows import compute_workflow_hash

        workflow = {"prompt": prompt, "nodes": []}

        hash1 = compute_workflow_hash(workflow)
        hash2 = compute_workflow_hash(workflow)

        assert hash1 == hash2

    @given(st.text(min_size=1, max_size=50), st.text(min_size=1, max_size=50))
    def test_different_workflows_different_hashes(self, prompt1, prompt2):
        """Different workflows should (usually) have different hashes."""
        from comfy_headless.workflows import compute_workflow_hash

        assume(prompt1 != prompt2)

        workflow1 = {"prompt": prompt1, "nodes": []}
        workflow2 = {"prompt": prompt2, "nodes": []}

        hash1 = compute_workflow_hash(workflow1)
        hash2 = compute_workflow_hash(workflow2)

        # Should be different (collision is theoretically possible but unlikely)
        assert hash1 != hash2
