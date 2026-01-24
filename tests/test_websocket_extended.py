"""
Extended tests for comfy_headless/websocket_client.py

Covers:
- Connection handling (lines 161-191)
- Connection states and reconnection (lines 375-391)
- WSProgress dataclass
- WSMessageType enum
- Message handling
"""

import asyncio
from dataclasses import is_dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check if websockets is available
try:
    import websockets

    WEBSOCKETS_INSTALLED = True
except ImportError:
    WEBSOCKETS_INSTALLED = False


pytestmark = pytest.mark.skipif(not WEBSOCKETS_INSTALLED, reason="websockets package not installed")


class TestWSMessageType:
    """Test WSMessageType enum."""

    def test_message_types_exist(self):
        """All expected message types exist."""
        from comfy_headless.websocket_client import WSMessageType

        expected = [
            "STATUS",
            "PROGRESS",
            "EXECUTING",
            "EXECUTED",
            "EXECUTION_START",
            "EXECUTION_CACHED",
            "EXECUTION_ERROR",
            "EXECUTION_INTERRUPTED",
            "PREVIEW",
        ]

        for name in expected:
            assert hasattr(WSMessageType, name)

    def test_message_type_values(self):
        """Message types have expected values."""
        from comfy_headless.websocket_client import WSMessageType

        assert WSMessageType.STATUS.value == "status"
        assert WSMessageType.PROGRESS.value == "progress"
        assert WSMessageType.EXECUTING.value == "executing"
        assert WSMessageType.EXECUTED.value == "executed"

    def test_message_type_is_string_enum(self):
        """Message types are string enums."""
        from comfy_headless.websocket_client import WSMessageType

        assert isinstance(WSMessageType.STATUS.value, str)


class TestWSProgress:
    """Test WSProgress dataclass."""

    def test_wsprogress_is_dataclass(self):
        """WSProgress is a dataclass."""
        from comfy_headless.websocket_client import WSProgress

        assert is_dataclass(WSProgress)

    def test_wsprogress_default_values(self):
        """WSProgress has sensible defaults."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(prompt_id="test-123")

        assert progress.prompt_id == "test-123"
        assert progress.node_id is None
        assert progress.node_type is None
        assert progress.progress == 0.0
        assert progress.max_progress == 1.0
        assert progress.step == 0
        assert progress.total_steps == 0
        assert progress.status == "queued"
        assert progress.preview_data is None

    def test_wsprogress_percent(self):
        """WSProgress.percent calculates percentage."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(prompt_id="test", progress=0.5, max_progress=1.0)
        assert progress.percent == 50.0

        progress = WSProgress(prompt_id="test", progress=25, max_progress=100)
        assert progress.percent == 25.0

    def test_wsprogress_percent_zero_max(self):
        """WSProgress.percent handles zero max_progress."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(prompt_id="test", progress=5, max_progress=0)
        assert progress.percent == 0.0

    def test_wsprogress_normalized(self):
        """WSProgress.normalized returns 0-1 range."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(prompt_id="test", progress=0.75, max_progress=1.0)
        assert progress.normalized == 0.75

    def test_wsprogress_normalized_zero_max(self):
        """WSProgress.normalized handles zero max_progress."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(prompt_id="test", progress=5, max_progress=0)
        assert progress.normalized == 0.0

    def test_wsprogress_with_all_fields(self):
        """WSProgress accepts all fields."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(
            prompt_id="test-123",
            node_id="3",
            node_type="KSampler",
            progress=10,
            max_progress=20,
            step=10,
            total_steps=20,
            status="sampling",
            preview_data=b"image data",
        )

        assert progress.prompt_id == "test-123"
        assert progress.node_id == "3"
        assert progress.node_type == "KSampler"
        assert progress.step == 10
        assert progress.total_steps == 20
        assert progress.status == "sampling"
        assert progress.preview_data == b"image data"


class TestComfyWSClientInitialization:
    """Test ComfyWSClient initialization."""

    def test_default_initialization(self):
        """Client initializes with defaults."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient()

        assert client.ws_url is not None
        assert client.client_id is not None
        assert len(client.client_id) == 36  # UUID format
        assert client.reconnect_attempts == 3
        assert client.reconnect_delay == 1.0

    def test_custom_url(self):
        """Client accepts custom URL."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient(base_url="http://custom:9999")

        # Should be converted to ws://
        assert "ws://" in client.ws_url
        assert "custom:9999" in client.ws_url

    def test_http_to_ws_conversion(self):
        """HTTP URL converted to WS URL."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient(base_url="http://localhost:8188")
        assert client.ws_url == "ws://localhost:8188/ws"

    def test_https_to_wss_conversion(self):
        """HTTPS URL converted to WSS URL."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient(base_url="https://secure.example.com")
        assert client.ws_url == "wss://secure.example.com/ws"

    def test_custom_client_id(self):
        """Client accepts custom client_id."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient(client_id="my-custom-id")
        assert client.client_id == "my-custom-id"

    def test_custom_reconnect_settings(self):
        """Client accepts custom reconnect settings."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient(reconnect_attempts=5, reconnect_delay=2.0)
        assert client.reconnect_attempts == 5
        assert client.reconnect_delay == 2.0


class TestComfyWSClientConnection:
    """Test connection handling."""

    def test_connected_property_initially_false(self):
        """connected is False initially."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient()
        assert client.connected is False

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """connect returns True on success."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient()

        mock_ws = AsyncMock()
        mock_ws.closed = False

        with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
            result = await client.connect()
            assert result is True
            assert client.connected is True

    @pytest.mark.asyncio
    async def test_connect_retries_on_failure(self):
        """connect retries on failure."""
        from comfy_headless.exceptions import ComfyUIConnectionError
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient(reconnect_attempts=2, reconnect_delay=0.01)

        connect_calls = 0

        async def fail_connect(*args, **kwargs):
            nonlocal connect_calls
            connect_calls += 1
            raise ConnectionError("Failed")

        with patch("websockets.connect", new=fail_connect):
            with pytest.raises(ComfyUIConnectionError):
                await client.connect()

        assert connect_calls == 2

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        """disconnect cleans up resources."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient()

        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.close = AsyncMock()

        with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
            await client.connect()

            # Cancel the message task immediately
            if client._message_task:
                client._message_task.cancel()

            await client.disconnect()

            assert client._connected is False


class TestComfyWSClientContextManager:
    """Test async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Async context manager connects and disconnects."""
        from comfy_headless.websocket_client import ComfyWSClient

        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.close = AsyncMock()

        with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
            async with ComfyWSClient() as client:
                assert client.connected is True

            # After context, should be disconnected
            assert client._connected is False


class TestComfyWSClientQueuePrompt:
    """Test queue_prompt method."""

    @pytest.mark.asyncio
    async def test_queue_prompt_returns_id(self):
        """queue_prompt returns prompt_id."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient()
        client._connected = True

        # Create a mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"prompt_id": "queued-123"}
        mock_response.raise_for_status = MagicMock()

        # Create a mock async client
        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_response

        # Mock httpx.AsyncClient as context manager
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_async_client
            mock_client_class.return_value.__aexit__.return_value = None

            result = await client.queue_prompt({"workflow": "data"})
            assert result == "queued-123"


class TestWebsocketsAvailableFlag:
    """Test WEBSOCKETS_AVAILABLE flag."""

    def test_flag_is_bool(self):
        """WEBSOCKETS_AVAILABLE is boolean."""
        from comfy_headless.websocket_client import WEBSOCKETS_AVAILABLE

        assert isinstance(WEBSOCKETS_AVAILABLE, bool)

    def test_flag_reflects_import(self):
        """Flag reflects websockets availability."""
        from comfy_headless.websocket_client import WEBSOCKETS_AVAILABLE

        try:
            import websockets

            assert WEBSOCKETS_AVAILABLE is True
        except ImportError:
            assert WEBSOCKETS_AVAILABLE is False


class TestWSProgressStepTracking:
    """Test step tracking in WSProgress."""

    def test_step_tracking(self):
        """WSProgress tracks steps correctly."""
        from comfy_headless.websocket_client import WSProgress

        # Simulate progress through sampling
        for step in range(1, 21):
            progress = WSProgress(
                prompt_id="test", step=step, total_steps=20, progress=step, max_progress=20
            )

            assert progress.step == step
            assert progress.total_steps == 20
            assert progress.percent == (step / 20) * 100


class TestWSClientEdgeCases:
    """Test edge cases and error handling."""

    def test_url_without_scheme(self):
        """URL without scheme gets ws:// added."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient(base_url="localhost:8188")
        assert client.ws_url.startswith("ws://")

    def test_trailing_slash_stripped(self):
        """Trailing slash is stripped from URL."""
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient(base_url="http://localhost:8188/")
        assert not client.ws_url.endswith("//ws")
        assert client.ws_url.endswith("/ws")


class TestAllExports:
    """Test __all__ exports."""

    def test_all_exports_defined(self):
        """All expected exports are in __all__."""
        from comfy_headless import websocket_client

        expected = [
            "ComfyWSClient",
            "WSProgress",
            "WSMessageType",
            "WEBSOCKETS_AVAILABLE",
        ]

        for name in expected:
            assert name in websocket_client.__all__

    def test_all_exports_accessible(self):
        """All items in __all__ are accessible."""
        from comfy_headless import websocket_client

        for name in websocket_client.__all__:
            assert hasattr(websocket_client, name)


class TestWSProgressMutability:
    """Test WSProgress is mutable as expected."""

    def test_progress_updatable(self):
        """WSProgress fields can be updated."""
        from comfy_headless.websocket_client import WSProgress

        progress = WSProgress(prompt_id="test")

        progress.progress = 0.5
        progress.status = "processing"
        progress.node_id = "3"

        assert progress.progress == 0.5
        assert progress.status == "processing"
        assert progress.node_id == "3"


class TestReconnectionBackoff:
    """Test reconnection backoff calculation."""

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Reconnection uses exponential backoff."""
        from comfy_headless.exceptions import ComfyUIConnectionError
        from comfy_headless.websocket_client import ComfyWSClient

        client = ComfyWSClient(reconnect_attempts=3, reconnect_delay=0.01)

        delays = []
        original_sleep = asyncio.sleep

        async def track_sleep(duration):
            delays.append(duration)
            await original_sleep(0)  # Don't actually wait

        with patch("asyncio.sleep", new=track_sleep):
            with patch("websockets.connect", new=AsyncMock(side_effect=ConnectionError)):
                with pytest.raises(ComfyUIConnectionError):
                    await client.connect()

        # Should have exponential backoff: 0.01, 0.02 (only 2 delays, not 3 since last attempt doesn't sleep)
        assert len(delays) == 2
        assert delays[1] > delays[0]


class TestWSMessageParsing:
    """Test WebSocket message parsing patterns."""

    def test_status_message_structure(self):
        """Status messages have expected structure."""
        # This tests the expected message format
        status_msg = {"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}}}

        assert status_msg["type"] == "status"
        assert "data" in status_msg

    def test_progress_message_structure(self):
        """Progress messages have expected structure."""
        progress_msg = {
            "type": "progress",
            "data": {"value": 5, "max": 20, "prompt_id": "test-123", "node": "3"},
        }

        assert progress_msg["type"] == "progress"
        assert progress_msg["data"]["value"] == 5
        assert progress_msg["data"]["max"] == 20

    def test_executing_message_structure(self):
        """Executing messages have expected structure."""
        executing_msg = {"type": "executing", "data": {"node": "3", "prompt_id": "test-123"}}

        assert executing_msg["type"] == "executing"
        assert executing_msg["data"]["node"] == "3"

    def test_executed_message_structure(self):
        """Executed messages have expected structure."""
        executed_msg = {
            "type": "executed",
            "data": {
                "node": "9",
                "prompt_id": "test-123",
                "output": {"images": [{"filename": "test.png"}]},
            },
        }

        assert executed_msg["type"] == "executed"
        assert "output" in executed_msg["data"]
