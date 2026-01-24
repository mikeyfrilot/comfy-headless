"""
Comfy Headless - Logging Configuration
=======================================

Structured logging with OpenTelemetry trace correlation support.

Features:
- Structured JSON output for production
- Human-readable format for development
- OpenTelemetry trace/span ID correlation
- Request ID tracking across operations
- Performance timing utilities

Usage:
    from comfy_headless.logging_config import get_logger, LogContext

    logger = get_logger(__name__)
    logger.info("Starting generation", extra={"prompt_id": "abc123"})

    # With request tracing
    with LogContext("request-123"):
        logger.info("Processing request")  # Includes request_id in all logs

    # With OpenTelemetry (when enabled)
    # Logs automatically include trace_id and span_id
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import settings

__all__ = [
    # Formatter
    "StructuredFormatter",
    # Logger setup
    "get_logger",
    "set_log_level",
    # Request context
    "set_request_id",
    "clear_request_id",
    "LogContext",
    # Timing utilities
    "log_timing",
    "timed_operation",
    # Tracing
    "traced_operation",
    "get_tracer",
    # Constants
    "OTEL_AVAILABLE",
]

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


# =============================================================================
# CUSTOM FORMATTER
# =============================================================================


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs structured log messages.

    Supports both text and JSON output formats.
    Includes OpenTelemetry trace correlation when available.
    """

    def __init__(
        self, fmt: str | None = None, datefmt: str | None = None, json_output: bool = False
    ):
        super().__init__(fmt, datefmt)
        self.json_output = json_output

    def format(self, record: logging.LogRecord) -> str:
        if self.json_output:
            return self._format_json(record)
        return super().format(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        """Format as JSON for structured logging."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add OpenTelemetry trace context if available
        if OTEL_AVAILABLE and settings.logging.otel_enabled:
            span = trace.get_current_span()
            if span and span.is_recording():
                ctx = span.get_span_context()
                log_data["trace_id"] = format(ctx.trace_id, "032x")
                log_data["span_id"] = format(ctx.span_id, "016x")

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields (excluding standard LogRecord attributes)
        skip_keys = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "message",
            "asctime",
            "taskName",
        }

        for key, value in record.__dict__.items():
            if key not in skip_keys and not key.startswith("_"):
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data)


class OtelFormatter(logging.Formatter):
    """
    Formatter that includes OpenTelemetry trace context in log output.

    Format: timestamp | level | logger | [trace_id:span_id] | message
    """

    def format(self, record: logging.LogRecord) -> str:
        # Add trace context if available
        trace_context = ""
        if OTEL_AVAILABLE and settings.logging.otel_enabled:
            span = trace.get_current_span()
            if span and span.is_recording():
                ctx = span.get_span_context()
                trace_id = format(ctx.trace_id, "032x")[-12:]  # Last 12 chars
                span_id = format(ctx.span_id, "016x")[-8:]  # Last 8 chars
                trace_context = f"[{trace_id}:{span_id}] "

        # Build the message
        record.trace_context = trace_context
        return super().format(record)


# =============================================================================
# CONTEXT FILTER
# =============================================================================


class ContextFilter(logging.Filter):
    """
    Filter that adds context information to log records.

    Adds:
    - request_id: For tracing across operations
    - component: The component name
    - trace_id/span_id: From OpenTelemetry when available
    """

    def __init__(self, component: str = "comfy_headless"):
        super().__init__()
        self.component = component
        self._request_id: str | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        record.component = self.component
        record.request_id = self._request_id or "-"

        # Add OpenTelemetry context if available
        if OTEL_AVAILABLE and settings.logging.otel_enabled:
            span = trace.get_current_span()
            if span and span.is_recording():
                ctx = span.get_span_context()
                record.trace_id = format(ctx.trace_id, "032x")
                record.span_id = format(ctx.span_id, "016x")
            else:
                record.trace_id = "-"
                record.span_id = "-"
        else:
            record.trace_id = "-"
            record.span_id = "-"

        return True

    def set_request_id(self, request_id: str):
        """Set the current request ID for tracing."""
        self._request_id = request_id

    def clear_request_id(self):
        """Clear the current request ID."""
        self._request_id = None


# =============================================================================
# OPENTELEMETRY SETUP
# =============================================================================

_otel_initialized = False


def _setup_opentelemetry():
    """Initialize OpenTelemetry if enabled and available."""
    global _otel_initialized

    if _otel_initialized or not OTEL_AVAILABLE or not settings.logging.otel_enabled:
        return

    try:
        # Create resource with service name
        resource = Resource.create(
            {
                "service.name": settings.logging.otel_service_name,
                "service.version": settings.version,
            }
        )

        # Set up tracer provider
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        # Optionally set up OTLP exporter
        if settings.logging.otel_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter = OTLPSpanExporter(endpoint=settings.logging.otel_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except ImportError:
                pass  # OTLP exporter not installed

        # Instrument logging to add trace context automatically
        LoggingInstrumentor().instrument()

        _otel_initialized = True

    except Exception as e:
        # Don't fail if OTel setup fails
        logging.getLogger("comfy_headless").warning(f"OpenTelemetry setup failed: {e}")


# =============================================================================
# LOGGER MANAGEMENT
# =============================================================================

_loggers: dict = {}
_initialized: bool = False
_context_filter: ContextFilter | None = None


def _setup_logging():
    """Initialize the logging system."""
    global _initialized, _context_filter

    if _initialized:
        return

    config = settings.logging

    # Set up OpenTelemetry if enabled
    _setup_opentelemetry()

    # Create context filter
    _context_filter = ContextFilter()

    # Create formatter based on configuration
    if config.json_output:
        formatter = StructuredFormatter(json_output=True)
    elif OTEL_AVAILABLE and config.otel_enabled:
        # Use OTel-aware formatter
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(trace_context)s%(message)s"
        formatter = OtelFormatter(fmt=fmt, datefmt=config.date_format)
    else:
        fmt = config.format
        formatter = StructuredFormatter(fmt=fmt, datefmt=config.date_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_context_filter)

    # File handler (if configured)
    file_handler = None
    if config.file:
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_context_filter)

    # Configure root logger for our package
    root_logger = logging.getLogger("comfy_headless")
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    root_logger.propagate = False

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.

    Args:
        name: Usually __name__ of the calling module

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Starting operation")
    """
    _setup_logging()

    # Ensure name is under our namespace
    if not name.startswith("comfy_headless"):
        name = f"comfy_headless.{name}"

    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger

    return _loggers[name]


def set_log_level(level: str):
    """
    Change the log level at runtime.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    _setup_logging()
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger("comfy_headless")
    root_logger.setLevel(numeric_level)

    for logger in _loggers.values():
        logger.setLevel(numeric_level)


def set_request_id(request_id: str):
    """Set the current request ID for log tracing."""
    _setup_logging()
    if _context_filter:
        _context_filter.set_request_id(request_id)


def clear_request_id():
    """Clear the current request ID."""
    if _context_filter:
        _context_filter.clear_request_id()


# =============================================================================
# LOGGING CONTEXT MANAGER
# =============================================================================


class LogContext:
    """
    Context manager for request-scoped logging.

    Usage:
        with LogContext("abc123"):
            logger.info("Processing request")
            # All logs will include request_id=abc123
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.previous_id: str | None = None

    def __enter__(self):
        _setup_logging()
        if _context_filter:
            self.previous_id = _context_filter._request_id
            _context_filter.set_request_id(self.request_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _context_filter:
            if self.previous_id:
                _context_filter.set_request_id(self.previous_id)
            else:
                _context_filter.clear_request_id()
        return False


# =============================================================================
# OPENTELEMETRY SPAN CONTEXT
# =============================================================================


@contextmanager
def traced_operation(name: str, attributes: dict[str, Any] | None = None):
    """
    Context manager for creating an OpenTelemetry span.

    Usage:
        with traced_operation("generate_image", {"prompt": "sunset"}):
            result = generate(prompt)

    Falls back to no-op if OpenTelemetry is not available or disabled.
    """
    if OTEL_AVAILABLE and settings.logging.otel_enabled:
        tracer = trace.get_tracer(settings.logging.otel_service_name)
        with tracer.start_as_current_span(name, attributes=attributes or {}):
            yield
    else:
        yield


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def log_exception(
    logger: logging.Logger, message: str, exception: Exception, level: int = logging.ERROR, **extra
):
    """
    Log an exception with consistent formatting.

    Args:
        logger: The logger to use
        message: Context message
        exception: The exception to log
        level: Log level (default ERROR)
        **extra: Additional context fields
    """
    logger.log(
        level, f"{message}: {type(exception).__name__}: {exception}", exc_info=True, extra=extra
    )


def log_operation(
    logger: logging.Logger,
    operation: str,
    success: bool,
    duration_ms: float | None = None,
    **extra,
):
    """
    Log an operation result.

    Args:
        logger: The logger to use
        operation: Name of the operation
        success: Whether it succeeded
        duration_ms: Duration in milliseconds
        **extra: Additional context fields
    """
    status = "completed" if success else "failed"
    msg = f"{operation} {status}"
    if duration_ms is not None:
        msg += f" ({duration_ms:.1f}ms)"

    level = logging.INFO if success else logging.WARNING
    logger.log(level, msg, extra={"operation": operation, "success": success, **extra})


# =============================================================================
# PERFORMANCE LOGGING
# =============================================================================


@contextmanager
def log_timing(logger: logging.Logger, operation: str, **extra):
    """
    Context manager to log operation timing.

    Usage:
        with log_timing(logger, "image_generation"):
            result = generate_image(...)
    """
    start = time.perf_counter()
    try:
        yield
        success = True
    except Exception:
        success = False
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        log_operation(logger, operation, success, duration_ms, **extra)


# =============================================================================
# GET TRACER (for manual span creation)
# =============================================================================


def get_tracer(name: str | None = None):
    """
    Get an OpenTelemetry tracer for manual span creation.

    Args:
        name: Tracer name (default: service name from settings)

    Returns:
        OpenTelemetry tracer or None if not available

    Example:
        tracer = get_tracer()
        if tracer:
            with tracer.start_as_current_span("my_operation"):
                do_work()
    """
    if OTEL_AVAILABLE and settings.logging.otel_enabled:
        return trace.get_tracer(name or settings.logging.otel_service_name)
    return None
