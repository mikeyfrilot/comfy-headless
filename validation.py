"""
Comfy Headless - Input Validation and Sanitization
====================================================

Secure input validation using Pydantic with:
- Whitelist-based validation (safer than blacklist)
- Prompt injection detection
- Path traversal prevention
- Dimension and parameter bounds checking

Usage:
    from comfy_headless.validation import (
        validate_prompt,
        validate_dimensions,
        GenerationRequest,
    )

    # Simple validation
    safe_prompt = validate_prompt(user_input)

    # Pydantic model validation
    request = GenerationRequest(
        prompt="a sunset",
        width=1024,
        height=1024,
    )
"""

import html
import re
from pathlib import Path
from typing import Any

from .config import settings
from .exceptions import (
    DimensionError,
    InvalidParameterError,
    InvalidPromptError,
    SecurityError,
    ValidationError,
)
from .logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    # Core validation
    "validate_prompt",
    "sanitize_prompt",
    "validate_dimensions",
    "clamp_dimensions",
    "validate_path",
    "validate_in_range",
    "validate_choice",
    "validate_generation_params",
    # Decorators
    "validated_prompt",
    "validated_dimensions",
    # Pydantic models (if available)
    "GenerationRequest",
    "ImageGenerationRequest",
    "VideoGenerationRequest",
    # Security patterns
    "PROMPT_INJECTION_PATTERNS",
    # Constants
    "PYDANTIC_AVAILABLE",
]

# Try to import Pydantic for schema validation
try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        SecretStr,
        field_validator,
        model_validator,
    )

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object


# =============================================================================
# SECURITY PATTERNS
# =============================================================================

# Patterns that may indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"disregard\s+(previous|above|all)",
    r"forget\s+(everything|all|previous)",
    r"new\s+instructions?:",
    r"system\s*:",
    r"<\|.*?\|>",  # Special tokens
    r"\[INST\]",  # Instruction markers
    r"<<SYS>>",  # System prompt markers
]

# Characters that could be problematic in prompts
DANGEROUS_CHARS = [
    "\x00",  # Null byte
    "\x1b",  # Escape
    "\r\n",  # CRLF
]

# Allowed characters in prompts (whitelist approach)
PROMPT_ALLOWED_PATTERN = re.compile(r"^[\w\s\.,!?'\"()\-:;@#$%&*+=/<>\[\]{}|~`\n\r]+$", re.UNICODE)

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e",
    r"%252e",
    r"\.\.%2f",
    r"\.\.%5c",
]


# =============================================================================
# PROMPT VALIDATION
# =============================================================================


def validate_prompt(
    prompt: str,
    *,
    max_length: int = 10000,
    min_length: int = 1,
    allow_html: bool = False,
    check_injection: bool = True,
) -> str:
    """
    Validate and sanitize a user prompt.

    Args:
        prompt: The user's input prompt
        max_length: Maximum allowed length
        min_length: Minimum required length
        allow_html: Whether to allow HTML (escaped if False)
        check_injection: Whether to check for injection patterns

    Returns:
        Sanitized prompt string

    Raises:
        InvalidPromptError: If prompt fails validation
        SecurityError: If injection attempt detected
    """
    if not prompt:
        raise InvalidPromptError("Prompt cannot be empty")

    # Strip whitespace
    prompt = prompt.strip()

    if len(prompt) < min_length:
        raise InvalidPromptError(f"Prompt too short (minimum {min_length} characters)")

    if len(prompt) > max_length:
        raise InvalidPromptError(f"Prompt too long (maximum {max_length} characters)")

    # Remove dangerous characters
    for char in DANGEROUS_CHARS:
        prompt = prompt.replace(char, "")

    # Check for injection patterns BEFORE escaping (so patterns can match raw input)
    if check_injection:
        prompt_lower = prompt.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                logger.warning(
                    "Potential prompt injection detected",
                    extra={"pattern": pattern, "prompt_preview": prompt[:100]},
                )
                raise SecurityError(
                    "Prompt contains potentially unsafe content",
                    suggestions=[
                        "Remove special instruction-like text",
                        "Use natural language only",
                    ],
                )

    # Escape HTML if not allowed (after injection check)
    if not allow_html:
        prompt = html.escape(prompt)

    return prompt


def sanitize_prompt(prompt: str) -> str:
    """
    Sanitize a prompt without raising exceptions.

    Returns empty string if prompt is invalid.
    """
    try:
        return validate_prompt(prompt, check_injection=False)
    except (InvalidPromptError, SecurityError):
        return ""


# =============================================================================
# DIMENSION VALIDATION
# =============================================================================


def validate_dimensions(
    width: int,
    height: int,
    *,
    min_size: int = 64,
    max_size: int | None = None,
    must_be_divisible_by: int = 8,
) -> tuple:
    """
    Validate image dimensions.

    Args:
        width: Image width
        height: Image height
        min_size: Minimum dimension
        max_size: Maximum dimension (default from settings)
        must_be_divisible_by: Required divisibility

    Returns:
        Tuple of (width, height)

    Raises:
        DimensionError: If dimensions are invalid
    """
    max_size = max_size or settings.generation.max_width

    # Type check
    if not isinstance(width, int) or not isinstance(height, int):
        raise DimensionError(
            int(width) if width else 0, int(height) if height else 0, "Dimensions must be integers"
        )

    # Minimum size
    if width < min_size or height < min_size:
        raise DimensionError(width, height, f"Minimum size is {min_size}x{min_size}")

    # Maximum size
    if width > max_size or height > max_size:
        raise DimensionError(width, height, f"Maximum size is {max_size}x{max_size}")

    # Divisibility
    if must_be_divisible_by > 0:
        if width % must_be_divisible_by != 0 or height % must_be_divisible_by != 0:
            raise DimensionError(
                width, height, f"Dimensions must be divisible by {must_be_divisible_by}"
            )

    return (width, height)


def clamp_dimensions(
    width: int,
    height: int,
    *,
    min_size: int = 64,
    max_size: int | None = None,
    divisible_by: int = 8,
) -> tuple:
    """
    Clamp and adjust dimensions to valid values without raising.

    Returns dimensions adjusted to be valid.
    """
    max_size = max_size or settings.generation.max_width

    # Clamp to range
    width = max(min_size, min(width, max_size))
    height = max(min_size, min(height, max_size))

    # Round to nearest divisible value
    if divisible_by > 0:
        width = round(width / divisible_by) * divisible_by
        height = round(height / divisible_by) * divisible_by

    return (width, height)


# =============================================================================
# PATH VALIDATION
# =============================================================================


def validate_path(
    path: str,
    *,
    must_exist: bool = False,
    allowed_extensions: list[str] | None = None,
    base_directory: Path | None = None,
) -> Path:
    """
    Validate a file path for security.

    Args:
        path: The path to validate
        must_exist: Whether the path must exist
        allowed_extensions: List of allowed extensions (e.g., [".png", ".jpg"])
        base_directory: If set, path must be within this directory

    Returns:
        Validated Path object

    Raises:
        SecurityError: If path traversal detected
        ValidationError: If path is invalid
    """
    if not path:
        raise ValidationError("Path cannot be empty")

    # Check for traversal patterns
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if re.search(pattern, path, re.IGNORECASE):
            logger.warning("Path traversal attempt detected", extra={"path": path})
            raise SecurityError("Invalid path: possible traversal attempt")

    try:
        filepath = Path(path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid path: {e}")

    # Check extension
    if allowed_extensions:
        ext = filepath.suffix.lower()
        if ext not in [e.lower() for e in allowed_extensions]:
            raise ValidationError(
                f"Invalid file type: {ext}",
                suggestions=[f"Allowed types: {', '.join(allowed_extensions)}"],
            )

    # Check base directory containment
    if base_directory:
        base = Path(base_directory).resolve()
        try:
            filepath.relative_to(base)
        except ValueError:
            raise SecurityError(
                "Path is outside allowed directory",
                suggestions=["Use paths within the allowed directory only"],
            )

    # Check existence
    if must_exist and not filepath.exists():
        raise ValidationError(f"Path does not exist: {filepath}")

    return filepath


# =============================================================================
# PARAMETER VALIDATION
# =============================================================================


def validate_in_range(
    value: int | float,
    name: str,
    *,
    min_val: int | float | None = None,
    max_val: int | float | None = None,
) -> int | float:
    """
    Validate that a numeric value is within range.

    Args:
        value: The value to check
        name: Parameter name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        The validated value

    Raises:
        InvalidParameterError: If out of range
    """
    if min_val is not None and value < min_val:
        raise InvalidParameterError(name, value, f"must be at least {min_val}")

    if max_val is not None and value > max_val:
        raise InvalidParameterError(name, value, f"must be at most {max_val}")

    return value


def validate_choice(
    value: Any,
    name: str,
    choices: list[Any],
) -> Any:
    """
    Validate that a value is one of the allowed choices.

    Args:
        value: The value to check
        name: Parameter name for error messages
        choices: List of allowed values

    Returns:
        The validated value

    Raises:
        InvalidParameterError: If not in choices
    """
    if value not in choices:
        raise InvalidParameterError(name, value, "not a valid choice", allowed_values=choices)

    return value


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

if PYDANTIC_AVAILABLE:

    class GenerationRequest(BaseModel):
        """
        Validated generation request using Pydantic.

        All inputs are validated and sanitized automatically.
        """

        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_default=True,
            extra="forbid",  # Reject unknown fields
        )

        prompt: str = Field(
            ..., min_length=1, max_length=10000, description="The generation prompt"
        )
        negative_prompt: str = Field(default="", max_length=5000, description="Negative prompt")
        width: int = Field(default=1024, ge=64, le=4096, description="Image width")
        height: int = Field(default=1024, ge=64, le=4096, description="Image height")
        steps: int = Field(default=25, ge=1, le=150, description="Number of steps")
        cfg: float = Field(default=7.0, ge=1.0, le=30.0, description="CFG scale")
        seed: int = Field(default=-1, ge=-1, description="Random seed (-1 for random)")

        @field_validator("prompt", "negative_prompt")
        @classmethod
        def sanitize_prompts(cls, v: str) -> str:
            """Sanitize prompt content."""
            if not v:
                return v
            # Remove dangerous characters
            for char in DANGEROUS_CHARS:
                v = v.replace(char, "")
            return html.escape(v)

        @field_validator("prompt")
        @classmethod
        def check_prompt_injection(cls, v: str) -> str:
            """Check for injection patterns."""
            v_lower = v.lower()
            for pattern in INJECTION_PATTERNS:
                if re.search(pattern, v_lower, re.IGNORECASE):
                    raise ValueError("Prompt contains potentially unsafe content")
            return v

        @model_validator(mode="after")
        def validate_dimensions(self) -> "GenerationRequest":
            """Ensure dimensions are divisible by 8."""
            if self.width % 8 != 0:
                self.width = round(self.width / 8) * 8
            if self.height % 8 != 0:
                self.height = round(self.height / 8) * 8
            return self

    class VideoRequest(BaseModel):
        """Validated video generation request."""

        model_config = ConfigDict(
            str_strip_whitespace=True,
            extra="forbid",
        )

        prompt: str = Field(..., min_length=1, max_length=10000)
        negative_prompt: str = Field(default="", max_length=5000)
        width: int = Field(default=512, ge=64, le=2048)
        height: int = Field(default=512, ge=64, le=2048)
        frames: int = Field(default=16, ge=1, le=128)
        fps: int = Field(default=8, ge=1, le=60)
        motion_bucket_id: int = Field(default=127, ge=1, le=255)

        @field_validator("prompt")
        @classmethod
        def sanitize_prompt(cls, v: str) -> str:
            for char in DANGEROUS_CHARS:
                v = v.replace(char, "")
            return html.escape(v)

    class ModelReference(BaseModel):
        """Validated model reference."""

        model_config = ConfigDict(extra="forbid")

        name: str = Field(..., min_length=1, max_length=500)
        type: str = Field(default="checkpoint")

        @field_validator("name")
        @classmethod
        def validate_model_name(cls, v: str) -> str:
            """Prevent path traversal in model names."""
            for pattern in PATH_TRAVERSAL_PATTERNS:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError("Invalid model name")
            return v


# =============================================================================
# VALIDATION DECORATORS
# =============================================================================


def validated_prompt(func):
    """
    Decorator that validates the first 'prompt' argument.

    Usage:
        @validated_prompt
        def generate(prompt: str, **kwargs):
            ...
    """
    import functools

    @functools.wraps(func)
    def wrapper(prompt, *args, **kwargs):
        validated = validate_prompt(prompt)
        return func(validated, *args, **kwargs)

    return wrapper


def validated_dimensions(func):
    """
    Decorator that validates width/height keyword arguments.

    Usage:
        @validated_dimensions
        def generate(prompt: str, *, width: int, height: int):
            ...
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "width" in kwargs and "height" in kwargs:
            width, height = validate_dimensions(kwargs["width"], kwargs["height"])
            kwargs["width"] = width
            kwargs["height"] = height
        return func(*args, **kwargs)

    return wrapper


# =============================================================================
# BATCH VALIDATION
# =============================================================================


def validate_generation_params(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    cfg: float = 7.0,
    seed: int = -1,
    negative_prompt: str = "",
) -> dict[str, Any]:
    """
    Validate all generation parameters at once.

    Returns validated parameters dict.
    Raises ValidationError with all issues if any fail.
    """
    from .exceptions import ComfyHeadlessExceptionGroup

    errors = []

    # Validate prompt
    try:
        prompt = validate_prompt(prompt)
    except (InvalidPromptError, SecurityError) as e:
        errors.append(e)

    # Validate dimensions
    try:
        width, height = validate_dimensions(width, height)
    except DimensionError as e:
        errors.append(e)

    # Validate steps
    try:
        steps = validate_in_range(steps, "steps", min_val=1, max_val=150)
    except InvalidParameterError as e:
        errors.append(e)

    # Validate CFG
    try:
        cfg = validate_in_range(cfg, "cfg", min_val=1.0, max_val=30.0)
    except InvalidParameterError as e:
        errors.append(e)

    # Validate negative prompt (optional, just sanitize)
    if negative_prompt:
        try:
            negative_prompt = validate_prompt(negative_prompt, check_injection=False)
        except InvalidPromptError as e:
            errors.append(e)

    # If any errors, raise them all
    if errors:
        raise ComfyHeadlessExceptionGroup("Validation failed", errors)

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
    }
