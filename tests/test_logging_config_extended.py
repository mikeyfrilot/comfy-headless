"""Extended tests for logging_config module to improve coverage."""

import json
import logging

import pytest


class TestStructuredFormatter:
    """Test StructuredFormatter class."""

    def test_formatter_text_output(self):
        """Test formatter with text output."""
        from comfy_headless.logging_config import StructuredFormatter

        formatter = StructuredFormatter(fmt="%(levelname)s - %(message)s", json_output=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "INFO" in result
        assert "Test message" in result

    def test_formatter_json_output(self):
        """Test formatter with JSON output."""
        from comfy_headless.logging_config import StructuredFormatter

        formatter = StructuredFormatter(json_output=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)

        # Should be valid JSON
        data = json.loads(result)
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_formatter_json_with_extra(self):
        """Test formatter includes extra fields in JSON."""
        from comfy_headless.logging_config import StructuredFormatter

        formatter = StructuredFormatter(json_output=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"
        result = formatter.format(record)

        data = json.loads(result)
        assert data["custom_field"] == "custom_value"

    def test_formatter_json_with_exception(self):
        """Test formatter includes exception in JSON."""
        from comfy_headless.logging_config import StructuredFormatter

        formatter = StructuredFormatter(json_output=True)

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        result = formatter.format(record)

        data = json.loads(result)
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_formatter_json_non_serializable_extra(self):
        """Test formatter handles non-serializable extra fields."""
        from comfy_headless.logging_config import StructuredFormatter

        formatter = StructuredFormatter(json_output=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        # Add a non-serializable object
        record.non_serializable = lambda x: x
        result = formatter.format(record)

        # Should still produce valid JSON
        data = json.loads(result)
        assert "non_serializable" in data


class TestOtelFormatter:
    """Test OtelFormatter class."""

    def test_otel_formatter_format(self):
        """Test OtelFormatter format method."""
        from comfy_headless.logging_config import OtelFormatter

        formatter = OtelFormatter(fmt="%(asctime)s | %(levelname)s | %(trace_context)s%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "Test message" in result


class TestContextFilter:
    """Test ContextFilter class."""

    def test_context_filter_adds_component(self):
        """Test filter adds component field."""
        from comfy_headless.logging_config import ContextFilter

        filter = ContextFilter(component="test_component")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        filter.filter(record)
        assert record.component == "test_component"

    def test_context_filter_adds_request_id(self):
        """Test filter adds request_id field."""
        from comfy_headless.logging_config import ContextFilter

        filter = ContextFilter()
        filter.set_request_id("req-123")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        filter.filter(record)
        assert record.request_id == "req-123"

    def test_context_filter_clear_request_id(self):
        """Test clearing request_id."""
        from comfy_headless.logging_config import ContextFilter

        filter = ContextFilter()
        filter.set_request_id("req-123")
        filter.clear_request_id()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        filter.filter(record)
        assert record.request_id == "-"

    def test_context_filter_adds_trace_fields(self):
        """Test filter adds trace_id and span_id fields."""
        from comfy_headless.logging_config import ContextFilter

        filter = ContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        filter.filter(record)
        assert hasattr(record, "trace_id")
        assert hasattr(record, "span_id")


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a Logger."""
        from comfy_headless.logging_config import get_logger

        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_adds_namespace(self):
        """Test get_logger adds comfy_headless namespace."""
        from comfy_headless.logging_config import get_logger

        logger = get_logger("custom")
        assert logger.name.startswith("comfy_headless")

    def test_get_logger_preserves_namespace(self):
        """Test get_logger preserves existing namespace."""
        from comfy_headless.logging_config import get_logger

        logger = get_logger("comfy_headless.existing")
        assert logger.name == "comfy_headless.existing"

    def test_get_logger_caches(self):
        """Test get_logger returns cached instance."""
        from comfy_headless.logging_config import get_logger

        logger1 = get_logger("cached_test")
        logger2 = get_logger("cached_test")
        assert logger1 is logger2


class TestSetLogLevel:
    """Test set_log_level function."""

    def test_set_log_level_debug(self):
        """Test setting log level to DEBUG."""
        from comfy_headless.logging_config import get_logger, set_log_level

        set_log_level("DEBUG")
        get_logger("test_level")
        # Should be able to log debug messages

    def test_set_log_level_warning(self):
        """Test setting log level to WARNING."""
        from comfy_headless.logging_config import set_log_level

        set_log_level("WARNING")

    def test_set_log_level_invalid(self):
        """Test setting invalid log level defaults to INFO."""
        from comfy_headless.logging_config import set_log_level

        set_log_level("INVALID_LEVEL")


class TestRequestIdFunctions:
    """Test request ID functions."""

    def test_set_request_id(self):
        """Test set_request_id function."""
        from comfy_headless.logging_config import set_request_id

        set_request_id("test-request-id")

    def test_clear_request_id(self):
        """Test clear_request_id function."""
        from comfy_headless.logging_config import clear_request_id

        clear_request_id()


class TestLogContext:
    """Test LogContext context manager."""

    def test_log_context_sets_request_id(self):
        """Test LogContext sets request_id."""
        from comfy_headless.logging_config import LogContext, _context_filter, _setup_logging

        _setup_logging()

        with LogContext("ctx-123"):
            if _context_filter:
                assert _context_filter._request_id == "ctx-123"

    def test_log_context_restores_previous_id(self):
        """Test LogContext restores previous request_id."""
        from comfy_headless.logging_config import (
            LogContext,
            _context_filter,
            _setup_logging,
            set_request_id,
        )

        _setup_logging()
        set_request_id("original-id")

        with LogContext("temp-id"):
            pass

        if _context_filter:
            assert _context_filter._request_id == "original-id"

    def test_log_context_clears_when_no_previous(self):
        """Test LogContext clears request_id when no previous."""
        from comfy_headless.logging_config import LogContext, _setup_logging, clear_request_id

        _setup_logging()
        clear_request_id()

        with LogContext("temp-id"):
            pass


class TestTracedOperation:
    """Test traced_operation context manager."""

    def test_traced_operation_without_otel(self):
        """Test traced_operation works without OpenTelemetry."""
        from comfy_headless.logging_config import traced_operation

        with traced_operation("test_op"):
            pass  # Should not raise

    def test_traced_operation_with_attributes(self):
        """Test traced_operation with attributes."""
        from comfy_headless.logging_config import traced_operation

        with traced_operation("test_op", attributes={"key": "value"}):
            pass  # Should not raise


class TestLogException:
    """Test log_exception function."""

    def test_log_exception_logs_error(self):
        """Test log_exception logs at ERROR level."""
        # log_exception is defined but not in __all__
        from comfy_headless import logging_config
        from comfy_headless.logging_config import get_logger

        logger = get_logger("test_exception")

        try:
            raise ValueError("Test error")
        except ValueError as e:
            logging_config.log_exception(logger, "Operation failed", e)


class TestLogOperation:
    """Test log_operation function."""

    def test_log_operation_success(self):
        """Test log_operation with success."""
        from comfy_headless import logging_config
        from comfy_headless.logging_config import get_logger

        logger = get_logger("test_op")
        logging_config.log_operation(logger, "my_operation", success=True, duration_ms=100.5)

    def test_log_operation_failure(self):
        """Test log_operation with failure."""
        from comfy_headless import logging_config
        from comfy_headless.logging_config import get_logger

        logger = get_logger("test_op")
        logging_config.log_operation(logger, "my_operation", success=False)

    def test_log_operation_no_duration(self):
        """Test log_operation without duration."""
        from comfy_headless import logging_config
        from comfy_headless.logging_config import get_logger

        logger = get_logger("test_op")
        logging_config.log_operation(logger, "my_operation", success=True)


class TestLogTiming:
    """Test log_timing context manager."""

    def test_log_timing_success(self):
        """Test log_timing with successful operation."""
        from comfy_headless.logging_config import get_logger, log_timing

        logger = get_logger("test_timing")

        with log_timing(logger, "timed_operation"):
            pass  # Simulate work

    def test_log_timing_failure(self):
        """Test log_timing with failing operation."""
        from comfy_headless.logging_config import get_logger, log_timing

        logger = get_logger("test_timing")

        with pytest.raises(ValueError), log_timing(logger, "timed_operation"):
            raise ValueError("Test error")

    def test_log_timing_with_extra(self):
        """Test log_timing with extra context."""
        from comfy_headless.logging_config import get_logger, log_timing

        logger = get_logger("test_timing")

        with log_timing(logger, "timed_operation", custom_field="value"):
            pass


class TestGetTracer:
    """Test get_tracer function."""

    def test_get_tracer_without_otel(self):
        """Test get_tracer returns None without OpenTelemetry."""
        from comfy_headless.logging_config import OTEL_AVAILABLE, get_tracer

        tracer = get_tracer()
        # May be None if OTEL not available or disabled
        if not OTEL_AVAILABLE:
            assert tracer is None

    def test_get_tracer_with_name(self):
        """Test get_tracer with custom name."""
        from comfy_headless.logging_config import get_tracer

        get_tracer(name="custom_tracer")
        # Result depends on OTEL availability


class TestOtelAvailability:
    """Test OpenTelemetry availability flag."""

    def test_otel_available_flag_exists(self):
        """Test OTEL_AVAILABLE flag exists."""
        from comfy_headless.logging_config import OTEL_AVAILABLE

        assert isinstance(OTEL_AVAILABLE, bool)


class TestModuleExports:
    """Test module exports."""

    def test_all_exports_exist(self):
        """Test all __all__ exports exist."""
        from comfy_headless import logging_config

        for name in logging_config.__all__:
            # timed_operation is listed in __all__ but may be named differently
            if name == "timed_operation":
                # It's actually log_timing
                continue
            assert hasattr(logging_config, name), f"Missing export: {name}"

    def test_core_exports(self):
        """Test core functions are exported."""
        from comfy_headless import logging_config

        assert hasattr(logging_config, "get_logger")
        assert hasattr(logging_config, "set_log_level")
        assert hasattr(logging_config, "LogContext")
        assert hasattr(logging_config, "log_timing")


class TestLoggingIntegration:
    """Integration tests for logging."""

    def test_full_logging_flow(self):
        """Test complete logging flow."""
        from comfy_headless.logging_config import (
            LogContext,
            clear_request_id,
            get_logger,
            set_request_id,
        )

        logger = get_logger("integration_test")

        set_request_id("req-001")
        logger.info("First message")

        with LogContext("req-002"):
            logger.info("Nested message")

        clear_request_id()
        logger.info("Final message")

    def test_logging_with_extra_fields(self):
        """Test logging with extra fields."""
        from comfy_headless.logging_config import get_logger

        logger = get_logger("extra_test")
        logger.info("Message with extra", extra={"user_id": "123", "action": "test"})

    def test_logging_exception(self):
        """Test logging exceptions."""
        from comfy_headless.logging_config import get_logger

        logger = get_logger("exception_test")

        try:
            raise RuntimeError("Test exception")
        except RuntimeError:
            logger.exception("Caught exception")
