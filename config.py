"""
Comfy Headless - Configuration Management
==========================================

Modern configuration using pydantic-settings for type-safe environment variable parsing.
All settings can be overridden via environment variables with COMFY_HEADLESS_ prefix.

Example:
    COMFY_HEADLESS_COMFYUI__URL=http://192.168.1.100:8188
    COMFY_HEADLESS_LOGGING__LEVEL=DEBUG
    COMFY_HEADLESS_RETRY__MAX_RETRIES=5

Features:
- Type-safe configuration with automatic validation
- Nested config via double underscore delimiter (__)
- .env file support
- SecretStr for sensitive values
- Cached settings instance via @lru_cache
"""

from functools import lru_cache
from pathlib import Path

__all__ = [
    # Config classes (vary based on pydantic availability)
    "Settings",
    "settings",
    "get_settings",
    "reload_settings",
    "get_temp_dir",
    # Sub-configs
    "ComfyUIConfig",
    "OllamaConfig",
    "RetryConfig",
    "LoggingConfig",
    "UIConfig",
    "GenerationConfig",
    "HttpConfig",
    # Constants
    "PYDANTIC_SETTINGS_AVAILABLE",
]

try:
    from pydantic import SecretStr
    from pydantic_settings import BaseSettings, SettingsConfigDict

    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    # Fallback for environments without pydantic-settings
    PYDANTIC_SETTINGS_AVAILABLE = False
    SecretStr = str  # type: ignore


# =============================================================================
# CONFIGURATION CLASSES (Pydantic v2 Style)
# =============================================================================

if PYDANTIC_SETTINGS_AVAILABLE:

    class ComfyUIConfig(BaseSettings):
        """ComfyUI connection configuration."""

        model_config = SettingsConfigDict(
            env_prefix="COMFY_HEADLESS_COMFYUI__",
            env_ignore_empty=True,
        )

        url: str = "http://localhost:8188"
        timeout_connect: float = 5.0
        timeout_read: float = 30.0
        timeout_queue: float = 10.0
        timeout_image: float = 60.0
        timeout_video: float = 120.0

    class OllamaConfig(BaseSettings):
        """Ollama AI configuration."""

        model_config = SettingsConfigDict(
            env_prefix="COMFY_HEADLESS_OLLAMA__",
            env_ignore_empty=True,
        )

        url: str = "http://localhost:11434"
        # v2.5.1: Use smaller, faster model by default for concise output
        model: str = "qwen2.5:1.5b"  # Fast model for quick enhancement
        quality_model: str = "qwen2.5:7b"  # Quality model for detailed enhancement
        timeout_analysis: float = 10.0  # Reduced for faster response
        timeout_enhancement: float = 20.0  # Reduced for faster response
        timeout_connect: float = 2.0
        # Custom few-shot examples file path (optional)
        # Format: JSON array of {"input": "...", "output": "...", "style": "..."}
        few_shot_examples_path: str | None = None

    class RetryConfig(BaseSettings):
        """Retry and resilience configuration."""

        model_config = SettingsConfigDict(
            env_prefix="COMFY_HEADLESS_RETRY__",
            env_ignore_empty=True,
        )

        max_retries: int = 3
        backoff_base: float = 1.5
        backoff_max: float = 30.0
        backoff_jitter: bool = True  # NEW: Add jitter to prevent thundering herd
        circuit_breaker_threshold: int = 5
        circuit_breaker_reset: float = 60.0

    class LoggingConfig(BaseSettings):
        """Logging configuration with OpenTelemetry support."""

        model_config = SettingsConfigDict(
            env_prefix="COMFY_HEADLESS_LOGGING__",
            env_ignore_empty=True,
        )

        level: str = "INFO"
        format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        date_format: str = "%Y-%m-%d %H:%M:%S"
        file: str | None = None
        json_output: bool = False
        # NEW: OpenTelemetry integration
        otel_enabled: bool = False
        otel_service_name: str = "comfy-headless"
        otel_endpoint: str | None = None  # OTLP endpoint

    class UIConfig(BaseSettings):
        """
        Gradio UI configuration.

        Security Note:
            Default host is 127.0.0.1 (localhost only) for security.
            Set COMFY_HEADLESS_UI__HOST=0.0.0.0 to expose on network.
        """

        model_config = SettingsConfigDict(
            env_prefix="COMFY_HEADLESS_UI__",
            env_ignore_empty=True,
        )

        port: int = 7861
        # Security: Default to localhost-only to prevent accidental network exposure
        host: str = "127.0.0.1"
        share: bool = False
        auto_open: bool = True
        temp_cleanup_interval: int = 3600

    class GenerationConfig(BaseSettings):
        """Default generation settings."""

        model_config = SettingsConfigDict(
            env_prefix="COMFY_HEADLESS_GENERATION__",
            env_ignore_empty=True,
        )

        default_width: int = 1024
        default_height: int = 1024
        default_steps: int = 25
        default_cfg: float = 7.0
        max_width: int = 2048
        max_height: int = 2048
        max_steps: int = 100
        generation_timeout: float = 300.0
        video_timeout: float = 600.0

    class HttpConfig(BaseSettings):
        """HTTP client configuration (NEW: for httpx support)."""

        model_config = SettingsConfigDict(
            env_prefix="COMFY_HEADLESS_HTTP__",
            env_ignore_empty=True,
        )

        # Connection pooling
        max_connections: int = 100
        max_keepalive_connections: int = 20
        keepalive_expiry: float = 5.0

        # HTTP/2 support
        http2: bool = True

        # Timeouts (httpx-style)
        connect_timeout: float = 5.0
        read_timeout: float = 30.0
        write_timeout: float = 30.0
        pool_timeout: float = 10.0

    class Settings(BaseSettings):
        """
        Main settings container using pydantic-settings.

        All settings are loaded from environment variables with COMFY_HEADLESS_ prefix.
        Nested settings use double underscore (__) as delimiter.

        Usage:
            from comfy_headless.config import get_settings

            settings = get_settings()
            print(settings.comfyui.url)
            print(settings.retry.max_retries)

        Environment Examples:
            COMFY_HEADLESS_COMFYUI__URL=http://192.168.1.100:8188
            COMFY_HEADLESS_LOGGING__LEVEL=DEBUG
            COMFY_HEADLESS_RETRY__MAX_RETRIES=5
        """

        model_config = SettingsConfigDict(
            env_prefix="COMFY_HEADLESS_",
            env_file=".env",
            env_file_encoding="utf-8",
            env_ignore_empty=True,
            env_nested_delimiter="__",
            extra="ignore",
        )

        # Nested configs
        comfyui: ComfyUIConfig = ComfyUIConfig()
        ollama: OllamaConfig = OllamaConfig()
        retry: RetryConfig = RetryConfig()
        logging: LoggingConfig = LoggingConfig()
        ui: UIConfig = UIConfig()
        generation: GenerationConfig = GenerationConfig()
        http: HttpConfig = HttpConfig()

        # Package info
        version: str = "2.5.1"
        name: str = "comfy_headless"

        def to_dict(self) -> dict:
            """Export settings as dictionary."""
            return {
                "version": self.version,
                "comfyui": {
                    "url": self.comfyui.url,
                    "timeout_connect": self.comfyui.timeout_connect,
                    "timeout_read": self.comfyui.timeout_read,
                },
                "ollama": {
                    "url": self.ollama.url,
                    "model": self.ollama.model,
                },
                "retry": {
                    "max_retries": self.retry.max_retries,
                    "backoff_base": self.retry.backoff_base,
                    "backoff_jitter": self.retry.backoff_jitter,
                },
                "logging": {
                    "level": self.logging.level,
                    "otel_enabled": self.logging.otel_enabled,
                },
                "http": {
                    "http2": self.http.http2,
                    "max_connections": self.http.max_connections,
                    "max_keepalive_connections": self.http.max_keepalive_connections,
                    "keepalive_expiry": self.http.keepalive_expiry,
                    "connect_timeout": self.http.connect_timeout,
                    "read_timeout": self.http.read_timeout,
                    "write_timeout": self.http.write_timeout,
                    "pool_timeout": self.http.pool_timeout,
                },
                "ui": {
                    "port": self.ui.port,
                    "host": self.ui.host,
                },
                "generation": {
                    "default_width": self.generation.default_width,
                    "default_height": self.generation.default_height,
                    "max_width": self.generation.max_width,
                    "max_height": self.generation.max_height,
                },
            }

else:
    # Fallback implementation using dataclasses (for compatibility)
    import os
    from dataclasses import dataclass, field

    def _get_env(key: str, default: str = None) -> str | None:
        return os.environ.get(f"COMFY_HEADLESS_{key}", default)

    def _get_env_int(key: str, default: int) -> int:
        val = _get_env(key)
        return int(val) if val else default

    def _get_env_float(key: str, default: float) -> float:
        val = _get_env(key)
        return float(val) if val else default

    def _get_env_bool(key: str, default: bool) -> bool:
        val = _get_env(key)
        return val.lower() in ("true", "1", "yes") if val else default

    @dataclass
    class ComfyUIConfig:
        url: str = field(default_factory=lambda: _get_env("COMFYUI__URL", "http://localhost:8188"))
        timeout_connect: float = field(
            default_factory=lambda: _get_env_float("COMFYUI__TIMEOUT_CONNECT", 5.0)
        )
        timeout_read: float = field(
            default_factory=lambda: _get_env_float("COMFYUI__TIMEOUT_READ", 30.0)
        )
        timeout_queue: float = field(
            default_factory=lambda: _get_env_float("COMFYUI__TIMEOUT_QUEUE", 10.0)
        )
        timeout_image: float = field(
            default_factory=lambda: _get_env_float("COMFYUI__TIMEOUT_IMAGE", 60.0)
        )
        timeout_video: float = field(
            default_factory=lambda: _get_env_float("COMFYUI__TIMEOUT_VIDEO", 120.0)
        )

    @dataclass
    class OllamaConfig:
        url: str = field(default_factory=lambda: _get_env("OLLAMA__URL", "http://localhost:11434"))
        model: str = field(default_factory=lambda: _get_env("OLLAMA__MODEL", "qwen2.5:7b"))
        timeout_analysis: float = field(
            default_factory=lambda: _get_env_float("OLLAMA__TIMEOUT_ANALYSIS", 15.0)
        )
        timeout_enhancement: float = field(
            default_factory=lambda: _get_env_float("OLLAMA__TIMEOUT_ENHANCEMENT", 30.0)
        )
        timeout_connect: float = field(
            default_factory=lambda: _get_env_float("OLLAMA__TIMEOUT_CONNECT", 2.0)
        )
        few_shot_examples_path: str | None = field(
            default_factory=lambda: _get_env("OLLAMA__FEW_SHOT_EXAMPLES_PATH", None)
        )

    @dataclass
    class RetryConfig:
        max_retries: int = field(default_factory=lambda: _get_env_int("RETRY__MAX_RETRIES", 3))
        backoff_base: float = field(
            default_factory=lambda: _get_env_float("RETRY__BACKOFF_BASE", 1.5)
        )
        backoff_max: float = field(
            default_factory=lambda: _get_env_float("RETRY__BACKOFF_MAX", 30.0)
        )
        backoff_jitter: bool = field(
            default_factory=lambda: _get_env_bool("RETRY__BACKOFF_JITTER", True)
        )
        circuit_breaker_threshold: int = field(
            default_factory=lambda: _get_env_int("RETRY__CIRCUIT_BREAKER_THRESHOLD", 5)
        )
        circuit_breaker_reset: float = field(
            default_factory=lambda: _get_env_float("RETRY__CIRCUIT_BREAKER_RESET", 60.0)
        )

    @dataclass
    class LoggingConfig:
        level: str = field(default_factory=lambda: _get_env("LOGGING__LEVEL", "INFO"))
        format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        date_format: str = "%Y-%m-%d %H:%M:%S"
        file: str | None = None
        json_output: bool = False
        otel_enabled: bool = False
        otel_service_name: str = "comfy-headless"
        otel_endpoint: str | None = None

    @dataclass
    class UIConfig:
        """Security: Default to localhost-only."""

        port: int = field(default_factory=lambda: _get_env_int("UI__PORT", 7861))
        host: str = field(default_factory=lambda: _get_env("UI__HOST", "127.0.0.1"))
        share: bool = False
        auto_open: bool = True
        temp_cleanup_interval: int = 3600

    @dataclass
    class GenerationConfig:
        default_width: int = 1024
        default_height: int = 1024
        default_steps: int = 25
        default_cfg: float = 7.0
        max_width: int = 2048
        max_height: int = 2048
        max_steps: int = 100
        generation_timeout: float = 300.0
        video_timeout: float = 600.0

    @dataclass
    class HttpConfig:
        max_connections: int = 100
        max_keepalive_connections: int = 20
        keepalive_expiry: float = 5.0
        http2: bool = True
        connect_timeout: float = 5.0
        read_timeout: float = 30.0
        write_timeout: float = 30.0
        pool_timeout: float = 10.0

    @dataclass
    class Settings:
        comfyui: ComfyUIConfig = field(default_factory=ComfyUIConfig)
        ollama: OllamaConfig = field(default_factory=OllamaConfig)
        retry: RetryConfig = field(default_factory=RetryConfig)
        logging: LoggingConfig = field(default_factory=LoggingConfig)
        ui: UIConfig = field(default_factory=UIConfig)
        generation: GenerationConfig = field(default_factory=GenerationConfig)
        http: HttpConfig = field(default_factory=HttpConfig)
        version: str = "2.5.1"
        name: str = "comfy_headless"

        def to_dict(self) -> dict:
            return {"version": self.version}


# =============================================================================
# CACHED SETTINGS INSTANCE
# =============================================================================


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses @lru_cache to avoid re-reading environment/.env on every call.
    Call get_settings.cache_clear() to reload settings.
    """
    return Settings()


# Backwards-compatible singleton
settings = get_settings()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def reload_settings() -> Settings:
    """Reload all settings from environment variables."""
    get_settings.cache_clear()
    global settings
    settings = get_settings()
    return settings


def get_temp_dir() -> Path:
    """Get the temp directory for this package."""
    import tempfile

    temp_dir = Path(tempfile.gettempdir()) / "comfy_headless"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def get_cache_dir() -> Path:
    """Get the cache directory for this package."""
    cache_dir = Path.home() / ".cache" / "comfy_headless"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
