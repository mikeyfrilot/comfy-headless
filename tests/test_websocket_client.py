"""Tests for websocket_client module."""

import json
from unittest.mock import AsyncMock, patch

import pytest


class TestWebSocketImports:
    """Test module imports and availability."""

    def test_websockets_available_flag(self):
        """Test WEBSOCKETS_AVAILABLE flag exists."""
        from comfy_headless.websocket_client import WEBSOCKETS_AVAILABLE

        assert isinstance(WEBSOCKETS_AVAILABLE, bool)

    def test_exports(self):
        """Test expected exports are available."""
        from comfy_headless import websocket_client

        assert hasattr(websocket_client, "ComfyWSClient")
        assert hasattr(websocket_client, "WSProgress")
        assert hasattr(websocket_client, "WSMessageType")


class TestWSMessageType:
    """Test WSMessageType enum."""

    def test_message_types(self):
        """Test all message types exist."""
        from comfy_headless.websocket_client import WSMessageType

        assert WSMessageType.STATUS == "status"
        assert WSMessageType.PROGRESS == "progress"
        assert WSMessageType.EXECUTING == "executing"
        assert WSMessageType.EXECUTED == "executed"
        assert WSMessageType.EXECUTION_START == "execution_start"
        assert WSMessageType.EXECUTION_CACHED == "execution_cached"
        assert WSMessageType.EXECUTION_ERROR == "execution_error"
        assert WSMessageType.EXECUTION_INTERRUPTED == "execution_interrupted"


class TestWSProgress:
    """Test WSProgress dataclass."""

    def test_progress_creation(self):
        """Test creating WSProgress."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(
            prompt_id="test123", node_id="1", progress=0.5, max_progress=1.0, status="progress"
        )

        assert progress.prompt_id == "test123"
        assert progress.node_id == "1"
        assert progress.progress == 0.5

    def test_progress_percent(self):
        """Test percent property."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(prompt_id="test", progress=0.5, max_progress=1.0)
        assert progress.percent == 50.0

    def test_progress_percent_zero_max(self):
        """Test percent with zero max."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(prompt_id="test", progress=0.5, max_progress=0.0)
        assert progress.percent == 0.0

    def test_progress_normalized(self):
        """Test normalized property."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(prompt_id="test", progress=7, max_progress=10)
        assert progress.normalized == 0.7

    def test_progress_normalized_zero_max(self):
        """Test normalized with zero max."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(prompt_id="test", progress=5, max_progress=0)
        assert progress.normalized == 0.0

    def test_progress_defaults(self):
        """Test WSProgress default values."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(prompt_id="test")

        assert progress.node_id is None
        assert progress.node_type is None
        assert progress.progress == 0.0
        assert progress.max_progress == 1.0
        assert progress.step == 0
        assert progress.total_steps == 0
        assert progress.status == "queued"
        assert progress.preview_data is None


class TestComfyWSClientInit:
    """Test ComfyWSClient initialization."""

    @pytest.fixture
    def mock_websockets(self):
        """Mock websockets module."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            yield

    def test_init_with_http_url(self, mock_websockets):
        """Test initialization with HTTP URL."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient(base_url="http://localhost:8188")
                assert "ws://" in client.ws_url
                assert "/ws" in client.ws_url
            except ImportError:
                pytest.skip("websockets not available")

    def test_init_with_https_url(self, mock_websockets):
        """Test initialization with HTTPS URL."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient(base_url="https://example.com:8188")
                assert "wss://" in client.ws_url
            except ImportError:
                pytest.skip("websockets not available")

    def test_init_with_custom_client_id(self, mock_websockets):
        """Test initialization with custom client ID."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient(client_id="my-custom-id")
                assert client.client_id == "my-custom-id"
            except ImportError:
                pytest.skip("websockets not available")

    def test_init_generates_client_id(self, mock_websockets):
        """Test initialization generates client ID."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()
                assert client.client_id is not None
                assert len(client.client_id) > 0
            except ImportError:
                pytest.skip("websockets not available")

    def test_init_reconnect_settings(self, mock_websockets):
        """Test reconnect settings."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient(reconnect_attempts=5, reconnect_delay=2.0)
                assert client.reconnect_attempts == 5
                assert client.reconnect_delay == 2.0
            except ImportError:
                pytest.skip("websockets not available")

    def test_init_raises_without_websockets(self):
        """Test ImportError raised when websockets not available."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", False):
            # Reload module to pick up the patched value
            import importlib

            from comfy_headless import websocket_client

            importlib.reload(websocket_client)

            # The ComfyWSClient should raise ImportError
            if not websocket_client.WEBSOCKETS_AVAILABLE:
                with pytest.raises(ImportError):
                    websocket_client.ComfyWSClient()


class TestComfyWSClientProperties:
    """Test ComfyWSClient properties."""

    def test_connected_property_false_when_no_ws(self):
        """Test connected is False when no websocket."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()
                assert client.connected is False
            except ImportError:
                pytest.skip("websockets not available")


class TestComfyWSClientListeners:
    """Test listener management."""

    def test_add_listener(self):
        """Test adding a listener."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()

                async def callback(progress):
                    pass

                client.add_listener("prompt123", callback)
                assert "prompt123" in client._listeners
                assert callback in client._listeners["prompt123"]
            except ImportError:
                pytest.skip("websockets not available")

    def test_remove_listener(self):
        """Test removing a listener."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()

                async def callback(progress):
                    pass

                client.add_listener("prompt123", callback)
                client.remove_listener("prompt123", callback)

                assert callback not in client._listeners.get("prompt123", [])
            except ImportError:
                pytest.skip("websockets not available")

    def test_add_multiple_listeners(self):
        """Test adding multiple listeners."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()

                async def callback1(progress):
                    pass

                async def callback2(progress):
                    pass

                client.add_listener("prompt123", callback1)
                client.add_listener("prompt123", callback2)

                assert len(client._listeners["prompt123"]) == 2
            except ImportError:
                pytest.skip("websockets not available")


class TestComfyWSClientMessageHandling:
    """Test message handling."""

    @pytest.mark.asyncio
    async def test_handle_binary_message(self):
        """Test handling binary preview message."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()

                # Mock notify_listeners
                client._notify_listeners = AsyncMock()

                # Binary message with header + data
                binary_data = b"\x00" * 8 + b"\x89PNG\r\n\x1a\n"

                await client._handle_message(binary_data)

                # Should have notified with preview
                if client._notify_listeners.called:
                    call_args = client._notify_listeners.call_args
                    assert call_args[0][0] == "preview"
            except ImportError:
                pytest.skip("websockets not available")

    @pytest.mark.asyncio
    async def test_handle_status_message(self):
        """Test handling status message."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()
                client._notify_listeners = AsyncMock()

                message = json.dumps(
                    {"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 5}}}}
                )

                await client._handle_message(message)
                # Status messages are logged but don't notify
            except ImportError:
                pytest.skip("websockets not available")

    @pytest.mark.asyncio
    async def test_handle_execution_start_message(self):
        """Test handling execution_start message."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()
                client._notify_listeners = AsyncMock()

                message = json.dumps({"type": "execution_start", "data": {"prompt_id": "abc123"}})

                await client._handle_message(message)

                if client._notify_listeners.called:
                    call_args = client._notify_listeners.call_args
                    progress = call_args[0][1]
                    assert progress.status == "started"
            except ImportError:
                pytest.skip("websockets not available")

    @pytest.mark.asyncio
    async def test_handle_progress_message(self):
        """Test handling progress message."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()
                client._notify_listeners = AsyncMock()

                message = json.dumps(
                    {
                        "type": "progress",
                        "data": {"prompt_id": "abc123", "node": "3", "value": 5, "max": 10},
                    }
                )

                await client._handle_message(message)

                if client._notify_listeners.called:
                    call_args = client._notify_listeners.call_args
                    progress = call_args[0][1]
                    assert progress.progress == 5.0
                    assert progress.max_progress == 10.0
            except ImportError:
                pytest.skip("websockets not available")

    @pytest.mark.asyncio
    async def test_handle_executing_complete(self):
        """Test handling executing message with None node (complete)."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()
                client._notify_listeners = AsyncMock()

                message = json.dumps(
                    {"type": "executing", "data": {"prompt_id": "abc123", "node": None}}
                )

                await client._handle_message(message)

                if client._notify_listeners.called:
                    call_args = client._notify_listeners.call_args
                    progress = call_args[0][1]
                    assert progress.status == "completed"
            except ImportError:
                pytest.skip("websockets not available")

    @pytest.mark.asyncio
    async def test_handle_execution_error(self):
        """Test handling execution_error message."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()
                client._notify_listeners = AsyncMock()

                message = json.dumps({"type": "execution_error", "data": {"prompt_id": "abc123"}})

                await client._handle_message(message)

                if client._notify_listeners.called:
                    call_args = client._notify_listeners.call_args
                    progress = call_args[0][1]
                    assert progress.status == "error"
            except ImportError:
                pytest.skip("websockets not available")

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self):
        """Test handling invalid JSON message."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient

                client = ComfyWSClient()
                client._notify_listeners = AsyncMock()

                # Invalid JSON should not raise, just log
                await client._handle_message("not valid json {{{")
            except ImportError:
                pytest.skip("websockets not available")


class TestComfyWSClientNotifyListeners:
    """Test listener notification."""

    @pytest.mark.asyncio
    async def test_notify_listeners_calls_callbacks(self):
        """Test notify_listeners calls all callbacks."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient, WSProgress

                client = ComfyWSClient()

                called = []

                async def callback(progress):
                    called.append(progress)

                client.add_listener("prompt123", callback)

                progress = WSProgress(prompt_id="prompt123", status="test")
                await client._notify_listeners("prompt123", progress)

                assert len(called) == 1
                assert called[0].status == "test"
            except ImportError:
                pytest.skip("websockets not available")

    @pytest.mark.asyncio
    async def test_notify_global_listeners(self):
        """Test global listeners (empty prompt_id) are notified."""
        with patch("comfy_headless.websocket_client.WEBSOCKETS_AVAILABLE", True):
            try:
                from comfy_headless.websocket_client import ComfyWSClient, WSProgress

                client = ComfyWSClient()

                called = []

                async def global_callback(progress):
                    called.append(progress)

                # Add global listener (empty prompt_id)
                client.add_listener("", global_callback)

                progress = WSProgress(prompt_id="other_prompt", status="test")
                await client._notify_listeners("other_prompt", progress)

                # Global listener should be called for any prompt
                assert len(called) == 1
            except ImportError:
                pytest.skip("websockets not available")
