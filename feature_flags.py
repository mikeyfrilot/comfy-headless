"""
Comfy Headless - Feature Flags
==============================

Detects which optional features are available based on installed dependencies.
Provides decorators and utilities for graceful degradation.

Usage:
    from comfy_headless.feature_flags import FEATURES, require_feature

    # Check if feature is available
    if FEATURES["websocket"]:
        from comfy_headless import ComfyWSClient

    # Decorator to require a feature
    @require_feature("ai")
    def enhance_prompt(prompt: str) -> str:
        ...

Installation commands for each feature:
    pip install comfy-headless              # Core only
    pip install comfy-headless[ai]          # + Ollama intelligence
    pip install comfy-headless[websocket]   # + WebSocket real-time updates
    pip install comfy-headless[health]      # + System health monitoring
    pip install comfy-headless[ui]          # + Gradio web UI
    pip install comfy-headless[validation]  # + Pydantic config validation
    pip install comfy-headless[observability] # + OpenTelemetry tracing
    pip install comfy-headless[standard]    # ai + websocket (recommended)
    pip install comfy-headless[full]        # Everything
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

# Use standard logging to avoid circular import with logging_config
logger = logging.getLogger(__name__)

__all__ = [
    "FEATURES",
    "check_feature",
    "require_feature",
    "get_install_hint",
    "list_available_features",
    "list_missing_features",
]

# Feature detection results
FEATURES: dict[str, bool] = {
    "ai": False,
    "websocket": False,
    "health": False,
    "ui": False,
    "validation": False,
    "observability": False,
}

# Installation hints for each feature
INSTALL_HINTS: dict[str, str] = {
    "ai": "pip install comfy-headless[ai]",
    "websocket": "pip install comfy-headless[websocket]",
    "health": "pip install comfy-headless[health]",
    "ui": "pip install comfy-headless[ui]",
    "validation": "pip install comfy-headless[validation]",
    "observability": "pip install comfy-headless[observability]",
}

# Feature descriptions
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "ai": "AI-powered prompt intelligence via Ollama (httpx)",
    "websocket": "Real-time progress updates via WebSocket",
    "health": "System health monitoring (psutil)",
    "ui": "Gradio web interface",
    "validation": "Pydantic configuration validation",
    "observability": "OpenTelemetry distributed tracing",
}


# =============================================================================
# FEATURE DETECTION
# =============================================================================


def _detect_features() -> None:
    """Detect which optional features are available."""
    global FEATURES

    # AI feature (httpx for Ollama)
    try:
        import httpx  # noqa: F401

        FEATURES["ai"] = True
        logger.debug("Feature 'ai' available: httpx installed")
    except ImportError:
        logger.debug("Feature 'ai' unavailable: httpx not installed")

    # WebSocket feature
    try:
        import websockets  # noqa: F401

        FEATURES["websocket"] = True
        logger.debug("Feature 'websocket' available: websockets installed")
    except ImportError:
        logger.debug("Feature 'websocket' unavailable: websockets not installed")

    # Health monitoring feature
    try:
        import psutil  # noqa: F401

        FEATURES["health"] = True
        logger.debug("Feature 'health' available: psutil installed")
    except ImportError:
        logger.debug("Feature 'health' unavailable: psutil not installed")

    # UI feature (Gradio)
    try:
        import gradio  # noqa: F401

        FEATURES["ui"] = True
        logger.debug("Feature 'ui' available: gradio installed")
    except ImportError:
        logger.debug("Feature 'ui' unavailable: gradio not installed")

    # Validation feature (Pydantic)
    try:
        import pydantic  # noqa: F401
        import pydantic_settings  # noqa: F401

        FEATURES["validation"] = True
        logger.debug("Feature 'validation' available: pydantic installed")
    except ImportError:
        logger.debug("Feature 'validation' unavailable: pydantic/pydantic_settings not installed")

    # Observability feature (OpenTelemetry)
    try:
        import opentelemetry  # noqa: F401

        FEATURES["observability"] = True
        logger.debug("Feature 'observability' available: opentelemetry installed")
    except ImportError:
        logger.debug("Feature 'observability' unavailable: opentelemetry not installed")

    # Log summary of detected features
    available = [name for name, enabled in FEATURES.items() if enabled]
    missing = [name for name, enabled in FEATURES.items() if not enabled]
    logger.debug(f"Feature detection complete: {len(available)} available, {len(missing)} missing")


# Run detection on module load
_detect_features()


# =============================================================================
# PUBLIC API
# =============================================================================


def check_feature(feature: str) -> bool:
    """
    Check if a feature is available.

    Args:
        feature: Feature name (ai, websocket, health, ui, validation, observability)

    Returns:
        True if the feature is installed
    """
    return FEATURES.get(feature, False)


def get_install_hint(feature: str) -> str:
    """
    Get installation command for a feature.

    Args:
        feature: Feature name

    Returns:
        pip install command string
    """
    return INSTALL_HINTS.get(feature, f"pip install comfy-headless[{feature}]")


def list_available_features() -> dict[str, str]:
    """
    List all installed features with descriptions.

    Returns:
        Dict of feature name -> description for installed features
    """
    return {
        name: FEATURE_DESCRIPTIONS.get(name, "")
        for name, available in FEATURES.items()
        if available
    }


def list_missing_features() -> dict[str, str]:
    """
    List all missing features with install hints.

    Returns:
        Dict of feature name -> install hint for missing features
    """
    return {
        name: INSTALL_HINTS.get(name, "") for name, available in FEATURES.items() if not available
    }


# Type variable for generic function preservation
F = TypeVar("F", bound=Callable[..., Any])


def require_feature(feature: str) -> Callable[[F], F]:
    """
    Decorator to require a feature for a function.

    Raises ImportError with install hint if feature is not available.

    Args:
        feature: Feature name to require

    Usage:
        @require_feature("ai")
        def enhance_prompt(prompt: str) -> str:
            # Uses httpx internally
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not FEATURES.get(feature, False):
                hint = INSTALL_HINTS.get(feature, f"pip install comfy-headless[{feature}]")
                raise ImportError(
                    f"Feature '{feature}' is required but not installed. Install with: {hint}"
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class FeatureNotAvailable(ImportError):
    """Raised when trying to use a feature that isn't installed."""

    def __init__(self, feature: str, message: str = ""):
        self.feature = feature
        self.install_hint = get_install_hint(feature)
        if not message:
            message = f"Feature '{feature}' is not available. Install with: {self.install_hint}"
        super().__init__(message)


def ensure_feature(feature: str) -> None:
    """
    Ensure a feature is available, raising FeatureNotAvailable if not.

    Args:
        feature: Feature name to check

    Raises:
        FeatureNotAvailable: If the feature is not installed
    """
    if not FEATURES.get(feature, False):
        raise FeatureNotAvailable(feature)
