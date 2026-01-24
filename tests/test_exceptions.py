"""Tests for exceptions module."""



class TestComfyHeadlessError:
    """Test base exception class."""

    def test_basic_error(self):
        """Test basic error creation."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test error")
        assert str(error) is not None
        assert error.code == "ComfyHeadlessError"

    def test_with_details(self):
        """Test error with details."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test error", details={"key": "value"}, code="TEST_ERROR")
        assert error.details["key"] == "value"
        assert error.code == "TEST_ERROR"

    def test_with_request_id(self):
        """Test error with request ID."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test error", request_id="abc123")
        assert error.request_id == "abc123"
        assert "abc123" in error.developer_message
        assert error.details["request_id"] == "abc123"

    def test_user_message(self):
        """Test user-friendly message."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Technical error details", user_message="Something went wrong")
        assert error.user_message == "Something went wrong"
        assert "Technical" in error.developer_message

    def test_suggestions(self):
        """Test recovery suggestions."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test", suggestions=["Try this", "Or this"])
        assert len(error.suggestions) == 2
        assert "Try this" in error.suggestions

    def test_add_context(self):
        """Test adding context is chainable."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test")
        result = error.add_context("foo", "bar")
        assert result is error
        assert error.details["foo"] == "bar"

    def test_to_dict(self):
        """Test JSON serialization."""
        from comfy_headless.exceptions import ComfyHeadlessError

        error = ComfyHeadlessError("Test", code="TEST", request_id="req123")
        data = error.to_dict()

        assert data["error"] is True
        assert data["code"] == "TEST"
        assert data["request_id"] == "req123"


class TestConnectionErrors:
    """Test connection-related errors."""

    def test_comfyui_connection_error(self):
        """Test ComfyUI connection error."""
        from comfy_headless.exceptions import ComfyUIConnectionError

        error = ComfyUIConnectionError(url="http://localhost:8188")
        assert error.details["url"] == "http://localhost:8188"
        assert len(error.suggestions) > 0

    def test_comfyui_offline_error(self):
        """Test ComfyUI offline error."""
        from comfy_headless.exceptions import ComfyUIOfflineError

        error = ComfyUIOfflineError()
        assert "offline" in error.message.lower() or "not running" in error.user_message.lower()

    def test_ollama_connection_error(self):
        """Test Ollama connection error."""
        from comfy_headless.exceptions import OllamaConnectionError

        error = OllamaConnectionError(url="http://localhost:11434")
        assert error.details["url"] == "http://localhost:11434"


class TestGenerationErrors:
    """Test generation-related errors."""

    def test_queue_error(self):
        """Test queue error."""
        from comfy_headless.exceptions import QueueError

        error = QueueError(prompt_id="abc123")
        assert error.details["prompt_id"] == "abc123"

    def test_generation_timeout(self):
        """Test timeout error."""
        from comfy_headless.exceptions import GenerationTimeoutError

        error = GenerationTimeoutError(timeout=60.0, prompt_id="xyz")
        assert error.details["timeout_seconds"] == 60.0
        assert error.details["prompt_id"] == "xyz"

    def test_generation_failed(self):
        """Test failed generation error."""
        from comfy_headless.exceptions import GenerationFailedError

        error = GenerationFailedError(comfy_error="Node error")
        assert "Node error" in error.details["comfy_error"]


class TestResult:
    """Test Result wrapper class."""

    def test_success_result(self):
        """Test successful result."""
        from comfy_headless.exceptions import Result

        result = Result.success("value")
        assert result.ok
        assert not result.failed
        assert result.value == "value"

    def test_failure_result(self):
        """Test failed result."""
        from comfy_headless.exceptions import ComfyHeadlessError, Result

        error = ComfyHeadlessError("Test error")
        result = Result.failure(error)
        assert result.failed
        assert not result.ok
        assert result.error is error

    def test_value_or(self):
        """Test value_or default."""
        from comfy_headless.exceptions import ComfyHeadlessError, Result

        success = Result.success("value")
        failure = Result.failure(ComfyHeadlessError("error"))

        assert success.value_or("default") == "value"
        assert failure.value_or("default") == "default"

    def test_map(self):
        """Test map transformation."""
        from comfy_headless.exceptions import Result

        result = Result.success(5)
        mapped = result.map(lambda x: x * 2)
        assert mapped.value == 10

    def test_from_exception(self):
        """Test wrapping function calls."""
        from comfy_headless.exceptions import Result

        def succeeds():
            return 42

        def fails():
            raise ValueError("oops")

        success = Result.from_exception(succeeds)
        failure = Result.from_exception(fails)

        assert success.ok
        assert success.value == 42
        assert failure.failed
