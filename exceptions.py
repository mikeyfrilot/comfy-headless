"""
Comfy Headless - Exception Hierarchy
=====================================

Modern exception classes with:
- User-friendly vs developer messages (environment-aware)
- Exception notes (PEP 678) for adding context after catch
- Exception groups support (PEP 654) for concurrent errors
- Structured error codes for programmatic handling
- Recovery suggestions for common errors

Usage:
    from comfy_headless.exceptions import (
        ComfyHeadlessError,
        ComfyUIConnectionError,
        GenerationError,
        ErrorLevel,
    )

    try:
        client.generate_image(...)
    except ComfyUIConnectionError as e:
        # User-friendly message for UI
        print(e.user_message)
        # Full details for logs
        logger.error(e.developer_message)
        # Recovery suggestions
        for suggestion in e.suggestions:
            print(f"Try: {suggestion}")
"""

import os
import sys
from enum import Enum
from typing import Any

__all__ = [
    # Error levels and verbosity
    "ErrorLevel",
    "VerbosityLevel",
    "set_verbosity",
    "get_verbosity",
    # Result wrapper
    "Result",
    # Base exceptions
    "ComfyHeadlessError",
    # Connection errors
    "ComfyUIConnectionError",
    "ComfyUIOfflineError",
    "OllamaConnectionError",
    "OllamaOfflineError",
    # Generation errors
    "GenerationError",
    "QueueError",
    "GenerationTimeoutError",
    "GenerationFailedError",
    # Workflow errors
    "WorkflowError",
    "WorkflowCompilationError",
    "TemplateNotFoundError",
    # Retry/circuit errors
    "RetryExhaustedError",
    "CircuitOpenError",
    # Validation errors
    "ValidationError",
    "InvalidPromptError",
    "InvalidParameterError",
    "DimensionError",
    "SecurityError",
    # Exception groups
    "ComfyHeadlessExceptionGroup",
    # Utilities
    "format_error_for_user",
    # Constants
    "EXCEPTION_GROUPS_AVAILABLE",
]

# Check Python version for exception groups support
EXCEPTION_GROUPS_AVAILABLE = sys.version_info >= (3, 11)


# =============================================================================
# ERROR LEVELS AND VERBOSITY
# =============================================================================


class ErrorLevel(Enum):
    """Error severity levels for filtering and display."""

    DEBUG = "debug"  # Developer-only details
    INFO = "info"  # Informational
    WARNING = "warning"  # Recoverable issues
    ERROR = "error"  # Operation failed
    CRITICAL = "critical"  # System-level failure


class VerbosityLevel(Enum):
    """
    Output verbosity levels for different audiences.

    ELI5: Simple explanations for non-technical users
    CASUAL: User-friendly messages for general users
    DEVELOPER: Full technical details for debugging
    """

    ELI5 = "eli5"  # Explain Like I'm 5
    CASUAL = "casual"  # Regular user
    DEVELOPER = "developer"  # Full technical details


_current_verbosity: VerbosityLevel | None = None


def _get_verbosity() -> VerbosityLevel:
    """Get current verbosity from environment or global setting."""
    global _current_verbosity
    if _current_verbosity is not None:
        return _current_verbosity
    level = os.environ.get("COMFY_HEADLESS_VERBOSITY", "casual").lower()
    try:
        return VerbosityLevel(level)
    except ValueError:
        return VerbosityLevel.CASUAL


def set_verbosity(level: VerbosityLevel):
    """Set the global verbosity level."""
    global _current_verbosity
    _current_verbosity = level


def get_verbosity() -> VerbosityLevel:
    """Get the current verbosity level."""
    return _get_verbosity()


def _is_production() -> bool:
    """Check if running in production mode."""
    return os.environ.get("COMFY_HEADLESS_ENV", "development").lower() == "production"


# =============================================================================
# BASE EXCEPTION
# =============================================================================


class ComfyHeadlessError(Exception):
    """
    Base exception for all comfy_headless errors.

    Features:
    - Dual messages: user-friendly and developer-detailed
    - Recovery suggestions for common errors
    - Structured error codes
    - Exception notes support (PEP 678)
    - Environment-aware output (production hides details)

    Attributes:
        message: Technical error message
        user_message: User-friendly explanation
        code: Error code for programmatic handling
        details: Dict with additional context
        suggestions: List of recovery suggestions
        level: Error severity level
    """

    # Default messages (can be overridden by subclasses)
    _default_user_message = "An error occurred"
    _default_eli5_message = "Something went wrong"
    _default_suggestions: list[str] = []

    def __init__(
        self,
        message: str,
        *,
        user_message: str | None = None,
        eli5_message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
        suggestions: list[str] | None = None,
        level: ErrorLevel = ErrorLevel.ERROR,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self._user_message = user_message
        self._eli5_message = eli5_message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self._suggestions = suggestions
        self.level = level
        self.request_id = request_id

        # Add request_id to details for logging/serialization
        if request_id:
            self.details["request_id"] = request_id

        # Chain cause if provided
        if cause:
            self.__cause__ = cause

    @property
    def user_message(self) -> str:
        """Get user-friendly message (for UI display)."""
        return self._user_message or self._default_user_message

    @property
    def eli5_message(self) -> str:
        """Get simple explanation (for non-technical users)."""
        return self._eli5_message or self._default_eli5_message

    @property
    def developer_message(self) -> str:
        """Get full technical message (for logs/debugging)."""
        prefix = f"[{self.code}]"
        if self.request_id:
            prefix = f"[{self.code}:{self.request_id}]"
        msg = f"{prefix} {self.message}"
        # Exclude request_id from details display since it's in prefix
        filtered_details = {k: v for k, v in self.details.items() if k != "request_id"}
        if filtered_details:
            details_str = ", ".join(f"{k}={v}" for k, v in filtered_details.items())
            msg += f" ({details_str})"
        if self.cause:
            msg += f" [caused by: {type(self.cause).__name__}: {self.cause}]"
        return msg

    @property
    def suggestions(self) -> list[str]:
        """Get recovery suggestions."""
        return self._suggestions or self._default_suggestions

    def get_message(self, verbosity: VerbosityLevel | None = None) -> str:
        """
        Get message appropriate for the verbosity level.

        Args:
            verbosity: Override auto-detected verbosity

        Returns:
            Appropriate message for the audience
        """
        level = verbosity or _get_verbosity()

        if level == VerbosityLevel.ELI5:
            return self.eli5_message
        elif level == VerbosityLevel.CASUAL:
            return self.user_message
        else:  # DEVELOPER
            return self.developer_message

    def add_context(self, key: str, value: Any) -> "ComfyHeadlessError":
        """Add context information (chainable)."""
        self.details[key] = value
        return self

    def add_suggestion(self, suggestion: str) -> "ComfyHeadlessError":
        """Add a recovery suggestion (chainable)."""
        if self._suggestions is None:
            self._suggestions = []
        self._suggestions.append(suggestion)
        return self

    def to_dict(self, include_internal: bool = False) -> dict[str, Any]:
        """
        Convert exception to dictionary for JSON serialization.

        Args:
            include_internal: Include developer details (False in production)
        """
        result = {
            "error": True,
            "code": self.code,
            "message": self.user_message,
            "suggestions": self.suggestions,
        }

        # Always include request_id for correlation
        if self.request_id:
            result["request_id"] = self.request_id

        # Include details only if not production or explicitly requested
        if include_internal or not _is_production():
            result["details"] = self.details
            result["developer_message"] = self.developer_message
            if self.cause:
                result["cause"] = str(self.cause)

        return result

    def __str__(self) -> str:
        """Return appropriate message based on environment."""
        if _is_production():
            return self.user_message
        return self.developer_message


# =============================================================================
# CONNECTION ERRORS
# =============================================================================


class ConnectionError(ComfyHeadlessError):
    """Base class for connection-related errors."""

    _default_user_message = "Connection failed"
    _default_eli5_message = "Can't connect to the service"


class ComfyUIConnectionError(ConnectionError):
    """Failed to connect to ComfyUI."""

    _default_user_message = "Unable to connect to ComfyUI"
    _default_eli5_message = "The image generator isn't responding"
    _default_suggestions = [
        "Check if ComfyUI is running",
        "Verify the URL in settings",
        "Check firewall settings",
    ]

    def __init__(
        self, message: str = "Failed to connect to ComfyUI", url: str | None = None, **kwargs
    ):
        details = kwargs.pop("details", {})
        if url:
            details["url"] = url
        super().__init__(message, code="COMFYUI_CONNECTION_ERROR", details=details, **kwargs)


class ComfyUIOfflineError(ConnectionError):
    """ComfyUI is not running or unreachable."""

    _default_user_message = "ComfyUI is not running"
    _default_eli5_message = "The image generator is turned off"
    _default_suggestions = [
        "Start ComfyUI",
        "Wait a few seconds and try again",
    ]

    def __init__(self, message: str = "ComfyUI is offline", url: str | None = None, **kwargs):
        details = kwargs.pop("details", {})
        if url:
            details["url"] = url
        super().__init__(message, code="COMFYUI_OFFLINE", details=details, **kwargs)


class OllamaConnectionError(ConnectionError):
    """Failed to connect to Ollama."""

    _default_user_message = "Unable to connect to Ollama AI"
    _default_eli5_message = "The AI helper isn't responding"
    _default_suggestions = [
        "Check if Ollama is running",
        "Run 'ollama serve' in terminal",
    ]

    def __init__(
        self, message: str = "Failed to connect to Ollama", url: str | None = None, **kwargs
    ):
        details = kwargs.pop("details", {})
        if url:
            details["url"] = url
        super().__init__(message, code="OLLAMA_CONNECTION_ERROR", details=details, **kwargs)


class OllamaOfflineError(ConnectionError):
    """Ollama is not running or unreachable."""

    _default_user_message = "Ollama AI is not available"
    _default_eli5_message = "The AI helper is turned off"

    def __init__(self, message: str = "Ollama is offline", **kwargs):
        super().__init__(message, code="OLLAMA_OFFLINE", **kwargs)


# =============================================================================
# GENERATION ERRORS
# =============================================================================


class GenerationError(ComfyHeadlessError):
    """Base class for generation-related errors."""

    _default_user_message = "Generation failed"
    _default_eli5_message = "Couldn't create the image"


class QueueError(GenerationError):
    """Failed to queue a prompt."""

    _default_user_message = "Unable to start generation"
    _default_eli5_message = "Couldn't start making the image"
    _default_suggestions = [
        "Check if ComfyUI is responding",
        "Try a simpler prompt",
    ]

    def __init__(
        self, message: str = "Failed to queue prompt", prompt_id: str | None = None, **kwargs
    ):
        details = kwargs.pop("details", {})
        if prompt_id:
            details["prompt_id"] = prompt_id
        super().__init__(message, code="QUEUE_ERROR", details=details, **kwargs)


class GenerationTimeoutError(GenerationError):
    """Generation timed out."""

    _default_user_message = "Generation took too long"
    _default_eli5_message = "Making the image took too long"
    _default_suggestions = [
        "Try a smaller image size",
        "Reduce the number of steps",
        "Use a faster model",
    ]

    def __init__(
        self,
        message: str = "Generation timed out",
        timeout: float | None = None,
        prompt_id: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if timeout:
            details["timeout_seconds"] = timeout
        if prompt_id:
            details["prompt_id"] = prompt_id
        super().__init__(message, code="GENERATION_TIMEOUT", details=details, **kwargs)


class GenerationFailedError(GenerationError):
    """Generation failed with an error."""

    _default_user_message = "Generation failed"
    _default_eli5_message = "Something went wrong while making the image"
    _default_suggestions = [
        "Try a different prompt",
        "Check your model and settings",
    ]

    def __init__(
        self,
        message: str = "Generation failed",
        comfy_error: str | None = None,
        prompt_id: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if comfy_error:
            details["comfy_error"] = comfy_error
        if prompt_id:
            details["prompt_id"] = prompt_id
        super().__init__(message, code="GENERATION_FAILED", details=details, **kwargs)


class NoOutputError(GenerationError):
    """Generation completed but produced no output."""

    _default_user_message = "No image was generated"
    _default_eli5_message = "The image generator finished but didn't make anything"

    def __init__(
        self,
        message: str = "Generation produced no output",
        prompt_id: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if prompt_id:
            details["prompt_id"] = prompt_id
        super().__init__(message, code="NO_OUTPUT", details=details, **kwargs)


# =============================================================================
# WORKFLOW ERRORS
# =============================================================================


class WorkflowError(ComfyHeadlessError):
    """Base class for workflow-related errors."""

    _default_user_message = "Workflow error"
    _default_eli5_message = "The recipe for making the image has a problem"


class WorkflowCompilationError(WorkflowError):
    """Failed to compile a workflow."""

    _default_user_message = "Unable to prepare the workflow"
    _default_eli5_message = "Couldn't set up the image recipe"

    def __init__(
        self,
        message: str = "Failed to compile workflow",
        template_id: str | None = None,
        errors: list | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if template_id:
            details["template_id"] = template_id
        if errors:
            details["errors"] = errors
        super().__init__(message, code="WORKFLOW_COMPILATION_ERROR", details=details, **kwargs)


class WorkflowValidationError(WorkflowError):
    """Workflow validation failed."""

    _default_user_message = "Workflow settings are invalid"
    _default_eli5_message = "Some of the image settings aren't quite right"

    def __init__(
        self, message: str = "Workflow validation failed", errors: list | None = None, **kwargs
    ):
        details = kwargs.pop("details", {})
        if errors:
            details["validation_errors"] = errors
        super().__init__(message, code="WORKFLOW_VALIDATION_ERROR", details=details, **kwargs)


class TemplateNotFoundError(WorkflowError):
    """Requested workflow template not found."""

    _default_user_message = "Template not found"
    _default_eli5_message = "Can't find that image recipe"

    def __init__(self, template_id: str, message: str | None = None, **kwargs):
        msg = message or f"Template not found: {template_id}"
        details = kwargs.pop("details", {})
        details["template_id"] = template_id
        super().__init__(msg, code="TEMPLATE_NOT_FOUND", details=details, **kwargs)


class MissingParameterError(WorkflowError):
    """Required parameter is missing."""

    _default_user_message = "Missing required setting"
    _default_eli5_message = "You forgot to fill in something important"

    def __init__(self, parameter: str, message: str | None = None, **kwargs):
        msg = message or f"Missing required parameter: {parameter}"
        details = kwargs.pop("details", {})
        details["parameter"] = parameter
        super().__init__(msg, code="MISSING_PARAMETER", details=details, **kwargs)


# =============================================================================
# VALIDATION ERRORS
# =============================================================================


class ValidationError(ComfyHeadlessError):
    """Base class for validation errors."""

    _default_user_message = "Invalid input"
    _default_eli5_message = "Something you entered isn't quite right"


class InvalidPromptError(ValidationError):
    """Prompt is invalid or empty."""

    _default_user_message = "Please enter a valid prompt"
    _default_eli5_message = "You need to describe what image you want"
    _default_suggestions = [
        "Enter a description of the image you want",
        "Example: 'a sunset over mountains'",
    ]

    def __init__(self, message: str = "Invalid or empty prompt", **kwargs):
        super().__init__(message, code="INVALID_PROMPT", **kwargs)


class InvalidParameterError(ValidationError):
    """Parameter value is invalid."""

    def __init__(
        self,
        parameter: str,
        value: Any,
        reason: str | None = None,
        allowed_values: list | None = None,
        **kwargs,
    ):
        msg = f"Invalid value for '{parameter}': {value}"
        if reason:
            msg += f" ({reason})"

        details = kwargs.pop("details", {})
        details["parameter"] = parameter
        details["value"] = str(value)
        if reason:
            details["reason"] = reason
        if allowed_values:
            details["allowed_values"] = allowed_values

        user_msg = f"Invalid {parameter}"
        if allowed_values:
            user_msg += f". Choose from: {', '.join(str(v) for v in allowed_values[:5])}"

        super().__init__(
            msg, code="INVALID_PARAMETER", user_message=user_msg, details=details, **kwargs
        )


class DimensionError(ValidationError):
    """Image/video dimensions are invalid."""

    _default_user_message = "Invalid image size"
    _default_eli5_message = "The image size isn't right"
    _default_suggestions = [
        "Use dimensions divisible by 8",
        "Try standard sizes: 512x512, 1024x1024",
    ]

    def __init__(self, width: int, height: int, reason: str | None = None, **kwargs):
        msg = f"Invalid dimensions: {width}x{height}"
        if reason:
            msg += f" ({reason})"
        details = kwargs.pop("details", {})
        details["width"] = width
        details["height"] = height
        if reason:
            details["reason"] = reason
        super().__init__(msg, code="DIMENSION_ERROR", details=details, **kwargs)


class SecurityError(ValidationError):
    """Security-related validation failure."""

    _default_user_message = "Security check failed"
    _default_eli5_message = "That's not allowed for safety reasons"

    def __init__(self, message: str = "Security validation failed", **kwargs):
        super().__init__(message, code="SECURITY_ERROR", level=ErrorLevel.WARNING, **kwargs)


# =============================================================================
# RESOURCE ERRORS
# =============================================================================


class ResourceError(ComfyHeadlessError):
    """Base class for resource-related errors."""

    _default_user_message = "Resource unavailable"
    _default_eli5_message = "Something we need isn't available"


class ModelNotFoundError(ResourceError):
    """Requested model not found."""

    _default_user_message = "Model not available"
    _default_eli5_message = "The AI model we need isn't installed"
    _default_suggestions = [
        "Check if the model is installed",
        "Download the model from HuggingFace or CivitAI",
    ]

    def __init__(self, model_name: str, model_type: str = "checkpoint", **kwargs):
        msg = f"{model_type.capitalize()} not found: {model_name}"
        details = kwargs.pop("details", {})
        details["model_name"] = model_name
        details["model_type"] = model_type
        super().__init__(msg, code="MODEL_NOT_FOUND", details=details, **kwargs)


class InsufficientVRAMError(ResourceError):
    """Not enough VRAM for the operation."""

    _default_user_message = "Not enough GPU memory"
    _default_eli5_message = "Your computer doesn't have enough power for this"
    _default_suggestions = [
        "Try a smaller image size",
        "Use a lighter model",
        "Close other GPU-intensive applications",
    ]

    def __init__(self, required_gb: float, available_gb: float | None = None, **kwargs):
        msg = f"Insufficient VRAM: requires {required_gb:.1f}GB"
        if available_gb is not None:
            msg += f", available {available_gb:.1f}GB"
        details = kwargs.pop("details", {})
        details["required_gb"] = required_gb
        if available_gb is not None:
            details["available_gb"] = available_gb
        super().__init__(msg, code="INSUFFICIENT_VRAM", details=details, **kwargs)


class FileNotFoundResourceError(ResourceError):
    """File not found. (Named to avoid shadowing builtin FileNotFoundError)"""

    _default_user_message = "File not found"
    _default_eli5_message = "Can't find that file"

    def __init__(self, filepath: str, **kwargs):
        msg = f"File not found: {filepath}"
        details = kwargs.pop("details", {})
        details["filepath"] = filepath
        super().__init__(msg, code="FILE_NOT_FOUND", details=details, **kwargs)


# Backwards compatibility alias (deprecated - use FileNotFoundResourceError)
ResourceFileNotFoundError = FileNotFoundResourceError


# =============================================================================
# RESILIENCE ERRORS
# =============================================================================


class ResilienceError(ComfyHeadlessError):
    """Base class for resilience-related errors."""

    _default_user_message = "Service temporarily unavailable"
    _default_eli5_message = "The service is having trouble right now"


class RetryExhaustedError(ResilienceError):
    """All retry attempts exhausted."""

    _default_user_message = "Operation failed after multiple attempts"
    _default_eli5_message = "We tried several times but it didn't work"
    _default_suggestions = [
        "Wait a moment and try again",
        "Check your connection",
    ]

    def __init__(
        self,
        message: str = "All retry attempts exhausted",
        attempts: int | None = None,
        last_error: Exception | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if attempts:
            details["attempts"] = attempts
        super().__init__(
            message, code="RETRY_EXHAUSTED", details=details, cause=last_error, **kwargs
        )


class CircuitOpenError(ResilienceError):
    """Circuit breaker is open, requests blocked."""

    _default_user_message = "Service temporarily blocked for safety"
    _default_eli5_message = "We're giving the service a break because it wasn't working"
    _default_suggestions = [
        "Wait 30-60 seconds",
        "The service may be overloaded or down",
    ]

    def __init__(self, service: str, message: str | None = None, **kwargs):
        msg = message or f"Circuit breaker open for {service}"
        details = kwargs.pop("details", {})
        details["service"] = service
        super().__init__(msg, code="CIRCUIT_OPEN", details=details, **kwargs)


# =============================================================================
# EXCEPTION GROUPS (PEP 654 - Python 3.11+)
# =============================================================================

if EXCEPTION_GROUPS_AVAILABLE:

    class ComfyHeadlessExceptionGroup(ExceptionGroup):
        """
        Group of related exceptions (e.g., multiple validation errors).

        Usage:
            errors = []
            if not prompt:
                errors.append(InvalidPromptError())
            if width < 64:
                errors.append(DimensionError(width, height, "too small"))

            if errors:
                raise ComfyHeadlessExceptionGroup("Validation failed", errors)
        """

        def __new__(cls, message: str, exceptions: list[ComfyHeadlessError]):
            return super().__new__(cls, message, exceptions)

        @property
        def user_messages(self) -> list[str]:
            """Get all user-friendly messages."""
            return [e.user_message for e in self.exceptions if isinstance(e, ComfyHeadlessError)]

        @property
        def all_suggestions(self) -> list[str]:
            """Get all recovery suggestions (deduplicated)."""
            suggestions = []
            for e in self.exceptions:
                if isinstance(e, ComfyHeadlessError):
                    for s in e.suggestions:
                        if s not in suggestions:
                            suggestions.append(s)
            return suggestions

else:
    # Fallback for Python < 3.11
    class ComfyHeadlessExceptionGroup(ComfyHeadlessError):
        """Fallback exception group for Python < 3.11."""

        def __init__(self, message: str, exceptions: list[ComfyHeadlessError]):
            self.exceptions = exceptions
            details = {"exception_count": len(exceptions)}
            super().__init__(message, code="EXCEPTION_GROUP", details=details)

        @property
        def user_messages(self) -> list[str]:
            return [e.user_message for e in self.exceptions if isinstance(e, ComfyHeadlessError)]

        @property
        def all_suggestions(self) -> list[str]:
            suggestions = []
            for e in self.exceptions:
                if isinstance(e, ComfyHeadlessError):
                    for s in e.suggestions:
                        if s not in suggestions:
                            suggestions.append(s)
            return suggestions


# =============================================================================
# RESULT CLASS (for structured returns instead of exceptions)
# =============================================================================


class Result:
    """
    A result object that can be either success or failure.

    Provides a functional approach to error handling without exceptions.

    Usage:
        result = generate_image(...)
        if result.ok:
            image = result.value
        else:
            logger.error(result.error)
            print(result.error.user_message)
    """

    def __init__(self, value: Any = None, error: ComfyHeadlessError | None = None):
        self._value = value
        self._error = error

    @property
    def ok(self) -> bool:
        """True if this is a successful result."""
        return self._error is None

    @property
    def failed(self) -> bool:
        """True if this is a failed result."""
        return self._error is not None

    @property
    def value(self) -> Any:
        """Get the value. Raises if this is an error result."""
        if self._error:
            raise self._error
        return self._value

    @property
    def error(self) -> ComfyHeadlessError | None:
        """Get the error if any."""
        return self._error

    def value_or(self, default: Any) -> Any:
        """Get the value or a default if this is an error."""
        return self._value if self.ok else default

    def map(self, fn):
        """Apply a function to the value if successful."""
        if self.ok:
            return Result(value=fn(self._value))
        return self

    def flat_map(self, fn):
        """Apply a function that returns a Result if successful."""
        if self.ok:
            return fn(self._value)
        return self

    def on_error(self, fn) -> "Result":
        """Call function with error if failed (chainable)."""
        if self.failed:
            fn(self._error)
        return self

    def to_dict(self, include_internal: bool = False) -> dict[str, Any]:
        """Convert to dictionary."""
        if self.ok:
            return {"success": True, "value": self._value}
        return {"success": False, **self._error.to_dict(include_internal)}

    @classmethod
    def success(cls, value: Any) -> "Result":
        """Create a successful result."""
        return cls(value=value)

    @classmethod
    def failure(cls, error: ComfyHeadlessError) -> "Result":
        """Create a failed result."""
        return cls(error=error)

    @classmethod
    def from_exception(cls, fn, *args, **kwargs) -> "Result":
        """
        Execute a function and wrap exceptions in Result.

        Usage:
            result = Result.from_exception(risky_function, arg1, arg2)
        """
        try:
            return cls.success(fn(*args, **kwargs))
        except ComfyHeadlessError as e:
            return cls.failure(e)
        except Exception as e:
            return cls.failure(ComfyHeadlessError(str(e), cause=e))

    def __repr__(self) -> str:
        if self.ok:
            return f"Result.success({self._value!r})"
        return f"Result.failure({self._error!r})"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def format_error_for_user(error: Exception, verbosity: VerbosityLevel | None = None) -> str:
    """
    Format any exception for user display.

    Args:
        error: The exception to format
        verbosity: Output verbosity level

    Returns:
        Formatted error message appropriate for the audience
    """
    level = verbosity or _get_verbosity()

    if isinstance(error, ComfyHeadlessError):
        return error.get_message(level)

    # Generic exception
    if level == VerbosityLevel.ELI5:
        return "Something went wrong"
    elif level == VerbosityLevel.CASUAL:
        return f"Error: {type(error).__name__}"
    else:
        return f"{type(error).__name__}: {error}"


def collect_suggestions(error: Exception) -> list[str]:
    """Collect all recovery suggestions from an exception."""
    if isinstance(error, ComfyHeadlessError):
        return error.suggestions
    return []
