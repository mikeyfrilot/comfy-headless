"""
Extended tests for comfy_headless/exceptions.py

Covers:
- Error formatting (lines 841-871)
- Verbosity levels
- Exception groups (Python 3.11+)
- Result wrapper
- All exception to_dict() methods
"""

import os
from unittest.mock import patch

import pytest


class TestErrorLevelEnum:
    """Test ErrorLevel enum."""

    def test_error_level_values(self):
        """ErrorLevel has expected values."""
        from comfy_headless.exceptions import ErrorLevel

        assert ErrorLevel.DEBUG.value == "debug"
        assert ErrorLevel.INFO.value == "info"
        assert ErrorLevel.WARNING.value == "warning"
        assert ErrorLevel.ERROR.value == "error"
        assert ErrorLevel.CRITICAL.value == "critical"


class TestVerbosityLevel:
    """Test VerbosityLevel enum."""

    def test_verbosity_level_values(self):
        """VerbosityLevel has expected values."""
        from comfy_headless.exceptions import VerbosityLevel

        assert VerbosityLevel.ELI5.value == "eli5"
        assert VerbosityLevel.CASUAL.value == "casual"
        assert VerbosityLevel.DEVELOPER.value == "developer"


class TestSetGetVerbosity:
    """Test set_verbosity and get_verbosity."""

    def test_default_verbosity(self):
        """Default verbosity is CASUAL."""
        from comfy_headless.exceptions import VerbosityLevel, get_verbosity, set_verbosity

        # Reset to ensure clean state
        set_verbosity(None)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COMFY_HEADLESS_VERBOSITY", None)
            level = get_verbosity()
            # Default should be CASUAL
            assert level == VerbosityLevel.CASUAL

    def test_set_verbosity(self):
        """set_verbosity changes the global verbosity."""
        from comfy_headless.exceptions import VerbosityLevel, get_verbosity, set_verbosity

        set_verbosity(VerbosityLevel.DEVELOPER)
        assert get_verbosity() == VerbosityLevel.DEVELOPER

        set_verbosity(VerbosityLevel.ELI5)
        assert get_verbosity() == VerbosityLevel.ELI5

        # Clean up
        set_verbosity(None)

    def test_verbosity_from_env(self):
        """Verbosity can be set via environment variable."""
        from comfy_headless.exceptions import VerbosityLevel, _get_verbosity, set_verbosity

        set_verbosity(None)  # Clear global setting

        with patch.dict(os.environ, {"COMFY_HEADLESS_VERBOSITY": "developer"}):
            level = _get_verbosity()
            assert level == VerbosityLevel.DEVELOPER


class TestComfyHeadlessError:
    """Test base ComfyHeadlessError class."""

    def test_basic_creation(self):
        """Error can be created with message."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test error message")
        assert str(error) is not None
        assert error.message == "Test error message"

    def test_error_code(self):
        """Error has correct code."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test", code="CUSTOM_CODE")
        assert error.code == "CUSTOM_CODE"

    def test_default_code_from_class_name(self):
        """Default code is class name."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test")
        assert error.code == "ComfyHeadlessError"

    def test_error_details(self):
        """Error stores details."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test", details={"key": "value"})
        assert error.details["key"] == "value"

    def test_error_request_id(self):
        """Error stores request_id."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test", request_id="req-123")
        assert error.request_id == "req-123"
        assert error.details["request_id"] == "req-123"

    def test_error_cause(self):
        """Error chains cause."""
        from comfy_headless.exceptions import ComfyHeadlessError

        original = ValueError("Original error")
        error = ComfyHeadlessError("Wrapped", cause=original)

        assert error.cause is original
        assert error.__cause__ is original

    def test_error_suggestions(self):
        """Error stores suggestions."""
        from comfy_headless.exceptions import ComfyHeadlessError

        suggestions = ["Try this", "Or that"]
        error = ComfyHeadlessError("Test", suggestions=suggestions)
        assert error.suggestions == suggestions


class TestErrorMessages:
    """Test different message types."""

    def test_user_message(self):
        """user_message returns user-friendly message."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Technical error", user_message="Something went wrong")
        assert error.user_message == "Something went wrong"

    def test_eli5_message(self):
        """eli5_message returns simple message."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Technical", eli5_message="Oops!")
        assert error.eli5_message == "Oops!"

    def test_developer_message(self):
        """developer_message includes full details."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError(
            "Technical error", code="ERR_001", details={"extra": "info"}, request_id="req-123"
        )

        dev_msg = error.developer_message
        assert "ERR_001" in dev_msg
        assert "req-123" in dev_msg
        assert "extra=info" in dev_msg

    def test_get_message_verbosity(self):
        """get_message respects verbosity level."""
        from comfy_headless.exceptions import ComfyHeadlessError, VerbosityLevel

        error = ComfyHeadlessError("Technical", user_message="User friendly", eli5_message="Simple")

        assert error.get_message(VerbosityLevel.ELI5) == "Simple"
        assert error.get_message(VerbosityLevel.CASUAL) == "User friendly"
        assert "Technical" in error.get_message(VerbosityLevel.DEVELOPER)


class TestErrorChaining:
    """Test error chaining methods."""

    def test_add_context(self):
        """add_context adds to details."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test")
        error.add_context("key1", "value1")
        error.add_context("key2", "value2")

        assert error.details["key1"] == "value1"
        assert error.details["key2"] == "value2"

    def test_add_context_chainable(self):
        """add_context is chainable."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test")
        result = error.add_context("key", "value")

        assert result is error

    def test_add_suggestion(self):
        """add_suggestion adds suggestion."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test")
        error.add_suggestion("Try this")
        error.add_suggestion("Or that")

        assert "Try this" in error.suggestions
        assert "Or that" in error.suggestions

    def test_add_suggestion_chainable(self):
        """add_suggestion is chainable."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test")
        result = error.add_suggestion("Suggestion")

        assert result is error


class TestErrorToDict:
    """Test to_dict() serialization."""

    def test_to_dict_basic(self):
        """to_dict returns dict with expected keys."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test error")
        d = error.to_dict()

        assert "error" in d
        assert d["error"] is True
        assert "code" in d
        assert "message" in d

    def test_to_dict_with_request_id(self):
        """to_dict includes request_id."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test", request_id="req-123")
        d = error.to_dict()

        assert d["request_id"] == "req-123"

    def test_to_dict_production_mode(self):
        """to_dict hides details in production."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test", details={"secret": "value"})

        with patch.dict(os.environ, {"COMFY_HEADLESS_ENV": "production"}):
            d = error.to_dict(include_internal=False)
            # In production without include_internal, details hidden
            assert "details" not in d or d.get("details") is None

    def test_to_dict_include_internal(self):
        """to_dict includes details when requested."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test", details={"key": "value"})
        d = error.to_dict(include_internal=True)

        assert "details" in d
        assert d["details"]["key"] == "value"


class TestSpecificExceptions:
    """Test specific exception subclasses."""

    def test_comfyui_connection_error(self):
        """ComfyUIConnectionError has correct defaults."""
        from comfy_headless.exceptions import ComfyUIConnectionError

        error = ComfyUIConnectionError(url="http://localhost:8188")
        assert error.code == "COMFYUI_CONNECTION_ERROR"
        assert error.details["url"] == "http://localhost:8188"
        assert len(error.suggestions) > 0

    def test_comfyui_offline_error(self):
        """ComfyUIOfflineError has correct defaults."""
        from comfy_headless.exceptions import ComfyUIOfflineError

        error = ComfyUIOfflineError()
        assert error.code == "COMFYUI_OFFLINE"

    def test_ollama_connection_error(self):
        """OllamaConnectionError has correct defaults."""
        from comfy_headless.exceptions import OllamaConnectionError

        error = OllamaConnectionError(url="http://localhost:11434")
        assert error.code == "OLLAMA_CONNECTION_ERROR"

    def test_queue_error(self):
        """QueueError has correct defaults."""
        from comfy_headless.exceptions import QueueError

        error = QueueError(prompt_id="test-123")
        assert error.code == "QUEUE_ERROR"
        assert error.details["prompt_id"] == "test-123"

    def test_generation_timeout_error(self):
        """GenerationTimeoutError has correct defaults."""
        from comfy_headless.exceptions import GenerationTimeoutError

        error = GenerationTimeoutError(timeout=60.0, prompt_id="test-123")
        assert error.code == "GENERATION_TIMEOUT"
        assert error.details["timeout_seconds"] == 60.0

    def test_generation_failed_error(self):
        """GenerationFailedError has correct defaults."""
        from comfy_headless.exceptions import GenerationFailedError

        error = GenerationFailedError(comfy_error="Node failed")
        assert error.code == "GENERATION_FAILED"
        assert error.details["comfy_error"] == "Node failed"

    def test_workflow_compilation_error(self):
        """WorkflowCompilationError has correct defaults."""
        from comfy_headless.exceptions import WorkflowCompilationError

        error = WorkflowCompilationError(template_id="txt2img", errors=["Error 1"])
        assert error.code == "WORKFLOW_COMPILATION_ERROR"

    def test_template_not_found_error(self):
        """TemplateNotFoundError has correct defaults."""
        from comfy_headless.exceptions import TemplateNotFoundError

        error = TemplateNotFoundError("missing_template")
        assert error.code == "TEMPLATE_NOT_FOUND"
        assert "missing_template" in error.message

    def test_retry_exhausted_error(self):
        """RetryExhaustedError has correct defaults."""
        from comfy_headless.exceptions import RetryExhaustedError

        last_err = ValueError("Last error")
        error = RetryExhaustedError(attempts=3, last_error=last_err)
        assert error.code == "RETRY_EXHAUSTED"
        assert error.details["attempts"] == 3
        assert error.cause is last_err

    def test_circuit_open_error(self):
        """CircuitOpenError has correct defaults."""
        from comfy_headless.exceptions import CircuitOpenError

        error = CircuitOpenError(service="comfyui")
        assert error.code == "CIRCUIT_OPEN"
        assert error.details["service"] == "comfyui"

    def test_validation_error(self):
        """ValidationError has correct defaults."""
        from comfy_headless.exceptions import ValidationError

        error = ValidationError("Invalid input")
        assert issubclass(type(error), Exception)

    def test_invalid_prompt_error(self):
        """InvalidPromptError has correct defaults."""
        from comfy_headless.exceptions import InvalidPromptError

        error = InvalidPromptError()
        assert error.code == "INVALID_PROMPT"
        assert len(error.suggestions) > 0

    def test_invalid_parameter_error(self):
        """InvalidParameterError has correct defaults."""
        from comfy_headless.exceptions import InvalidParameterError

        error = InvalidParameterError("width", 5000, reason="too large", allowed_values=[512, 1024])
        assert error.code == "INVALID_PARAMETER"
        assert error.details["parameter"] == "width"
        assert error.details["value"] == "5000"

    def test_security_error(self):
        """SecurityError has correct defaults."""
        from comfy_headless.exceptions import SecurityError

        error = SecurityError("Injection detected")
        assert error.code == "SECURITY_ERROR"


class TestResult:
    """Test Result wrapper class."""

    def test_result_success(self):
        """Result.success creates successful result."""
        from comfy_headless.exceptions import Result

        result = Result.success("value")

        assert result.ok is True
        assert result.failed is False
        assert result.value == "value"
        assert result.error is None

    def test_result_failure(self):
        """Result.failure creates failed result."""
        from comfy_headless.exceptions import ComfyHeadlessError, Result

        error = ComfyHeadlessError("Test error")
        result = Result.failure(error)

        assert result.ok is False
        assert result.failed is True
        assert result.error is error

    def test_result_value_raises_on_failure(self):
        """Accessing value on failure raises."""
        from comfy_headless.exceptions import ComfyHeadlessError, Result

        error = ComfyHeadlessError("Error")
        result = Result.failure(error)

        with pytest.raises(ComfyHeadlessError):
            _ = result.value

    def test_result_value_or(self):
        """value_or returns default on failure."""
        from comfy_headless.exceptions import ComfyHeadlessError, Result

        result = Result.failure(ComfyHeadlessError("Error"))

        assert result.value_or("default") == "default"

    def test_result_value_or_success(self):
        """value_or returns value on success."""
        from comfy_headless.exceptions import Result

        result = Result.success("actual")

        assert result.value_or("default") == "actual"

    def test_result_map(self):
        """Result.map transforms successful value."""
        from comfy_headless.exceptions import Result

        result = Result.success(5)
        mapped = result.map(lambda x: x * 2)

        assert mapped.value == 10

    def test_result_map_on_failure(self):
        """Result.map returns same failure."""
        from comfy_headless.exceptions import ComfyHeadlessError, Result

        error = ComfyHeadlessError("Error")
        result = Result.failure(error)
        mapped = result.map(lambda x: x * 2)

        assert mapped.failed
        assert mapped.error is error

    def test_result_flat_map(self):
        """Result.flat_map chains results."""
        from comfy_headless.exceptions import Result

        result = Result.success(5)
        chained = result.flat_map(lambda x: Result.success(x * 2))

        assert chained.value == 10

    def test_result_on_error(self):
        """Result.on_error calls callback on failure."""
        from comfy_headless.exceptions import ComfyHeadlessError, Result

        errors = []
        error = ComfyHeadlessError("Error")
        result = Result.failure(error)

        result.on_error(lambda e: errors.append(e))

        assert len(errors) == 1
        assert errors[0] is error

    def test_result_on_error_success(self):
        """Result.on_error doesn't call callback on success."""
        from comfy_headless.exceptions import Result

        errors = []
        result = Result.success("value")

        result.on_error(lambda e: errors.append(e))

        assert len(errors) == 0

    def test_result_to_dict(self):
        """Result.to_dict serializes correctly."""
        from comfy_headless.exceptions import Result

        success = Result.success("value")
        d = success.to_dict()

        assert d["success"] is True
        assert d["value"] == "value"

    def test_result_from_exception(self):
        """Result.from_exception wraps function calls."""
        from comfy_headless.exceptions import Result

        result = Result.from_exception(lambda: 42)
        assert result.ok
        assert result.value == 42

    def test_result_from_exception_catches_error(self):
        """Result.from_exception catches exceptions."""
        from comfy_headless.exceptions import ComfyHeadlessError, Result

        def failing():
            raise ComfyHeadlessError("Failed")

        result = Result.from_exception(failing)

        assert result.failed

    def test_result_repr(self):
        """Result has useful repr."""
        from comfy_headless.exceptions import Result

        success = Result.success("value")
        Result.failure(None)

        assert "success" in repr(success)


class TestExceptionGroups:
    """Test ComfyHeadlessExceptionGroup."""

    def test_exception_group_creation(self):
        """ExceptionGroup can be created."""
        from comfy_headless.exceptions import (
            ComfyHeadlessExceptionGroup,
            InvalidPromptError,
            ValidationError,
        )

        errors = [InvalidPromptError("Empty"), ValidationError("Invalid")]

        group = ComfyHeadlessExceptionGroup("Multiple errors", errors)

        assert len(group.exceptions) == 2

    def test_exception_group_user_messages(self):
        """ExceptionGroup collects user messages."""
        from comfy_headless.exceptions import ComfyHeadlessExceptionGroup, InvalidPromptError

        errors = [InvalidPromptError("Error 1")]
        group = ComfyHeadlessExceptionGroup("Errors", errors)

        messages = group.user_messages
        assert len(messages) >= 1

    def test_exception_group_all_suggestions(self):
        """ExceptionGroup collects all suggestions."""
        from comfy_headless.exceptions import ComfyHeadlessExceptionGroup, InvalidPromptError

        errors = [InvalidPromptError()]
        group = ComfyHeadlessExceptionGroup("Errors", errors)

        suggestions = group.all_suggestions
        assert isinstance(suggestions, list)


class TestFormatErrorForUser:
    """Test format_error_for_user utility."""

    def test_format_comfy_error(self):
        """format_error_for_user handles ComfyHeadlessError."""
        from comfy_headless.exceptions import (
            ComfyHeadlessError,
            VerbosityLevel,
            format_error_for_user,
        )

        error = ComfyHeadlessError("Technical", user_message="User friendly")

        result = format_error_for_user(error, VerbosityLevel.CASUAL)
        assert result == "User friendly"

    def test_format_generic_error_eli5(self):
        """format_error_for_user handles generic error at ELI5."""
        from comfy_headless.exceptions import VerbosityLevel, format_error_for_user

        error = ValueError("Some error")

        result = format_error_for_user(error, VerbosityLevel.ELI5)
        assert result == "Something went wrong"

    def test_format_generic_error_casual(self):
        """format_error_for_user handles generic error at CASUAL."""
        from comfy_headless.exceptions import VerbosityLevel, format_error_for_user

        error = ValueError("Some error")

        result = format_error_for_user(error, VerbosityLevel.CASUAL)
        assert "ValueError" in result

    def test_format_generic_error_developer(self):
        """format_error_for_user handles generic error at DEVELOPER."""
        from comfy_headless.exceptions import VerbosityLevel, format_error_for_user

        error = ValueError("Some error")

        result = format_error_for_user(error, VerbosityLevel.DEVELOPER)
        assert "ValueError" in result
        assert "Some error" in result


class TestAllExports:
    """Test __all__ exports."""

    def test_all_exports_defined(self):
        """All expected exports are in __all__."""
        from comfy_headless import exceptions

        expected = [
            "ErrorLevel",
            "VerbosityLevel",
            "set_verbosity",
            "get_verbosity",
            "Result",
            "ComfyHeadlessError",
            "ComfyUIConnectionError",
            "ComfyUIOfflineError",
            "OllamaConnectionError",
            "QueueError",
            "GenerationTimeoutError",
            "GenerationFailedError",
            "WorkflowCompilationError",
            "TemplateNotFoundError",
            "RetryExhaustedError",
            "CircuitOpenError",
            "ValidationError",
            "InvalidPromptError",
            "InvalidParameterError",
            "SecurityError",
            "ComfyHeadlessExceptionGroup",
            "format_error_for_user",
        ]

        for name in expected:
            assert name in exceptions.__all__

    def test_all_exports_accessible(self):
        """All items in __all__ are accessible."""
        from comfy_headless import exceptions

        for name in exceptions.__all__:
            assert hasattr(exceptions, name)
