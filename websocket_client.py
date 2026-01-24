"""
Comfy Headless - WebSocket Client for Real-Time Progress
=========================================================

Connects to ComfyUI's WebSocket endpoint for real-time progress updates
instead of polling. Provides significantly faster progress feedback.

Usage:
    from comfy_headless import ComfyWSClient

    async def progress_handler(progress: float, node: str, msg: str):
        print(f"[{progress*100:.0f}%] {node}: {msg}")

    async with ComfyWSClient() as client:
        prompt_id = await client.queue_prompt(workflow)
        result = await client.wait_for_completion(
            prompt_id,
            on_progress=progress_handler
        )

Features:
- Real-time progress via WebSocket
- Automatic reconnection with backoff
- Node-level progress tracking
- Preview image support
- Async-native design
"""

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import websockets
    from websockets.asyncio.client import ClientConnection

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    ClientConnection = None

import contextlib

from .config import settings
from .exceptions import ComfyUIConnectionError
from .logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "ComfyWSClient",
    "WSProgress",
    "WSMessageType",
    "WEBSOCKETS_AVAILABLE",
]


class WSMessageType(str, Enum):
    """ComfyUI WebSocket message types."""

    STATUS = "status"
    PROGRESS = "progress"
    EXECUTING = "executing"
    EXECUTED = "executed"
    EXECUTION_START = "execution_start"
    EXECUTION_CACHED = "execution_cached"
    EXECUTION_ERROR = "execution_error"
    EXECUTION_INTERRUPTED = "execution_interrupted"
    PREVIEW = "b]"  # Binary preview image


@dataclass
class WSProgress:
    """Progress information from WebSocket."""

    prompt_id: str
    node_id: str | None = None
    node_type: str | None = None
    progress: float = 0.0
    max_progress: float = 1.0
    step: int = 0
    total_steps: int = 0
    status: str = "queued"
    preview_data: bytes | None = None

    @property
    def percent(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.max_progress > 0:
            return (self.progress / self.max_progress) * 100
        return 0.0

    @property
    def normalized(self) -> float:
        """Get progress as 0.0-1.0."""
        if self.max_progress > 0:
            return self.progress / self.max_progress
        return 0.0


# Type alias for progress callback
ProgressCallback = Callable[[WSProgress], Awaitable[None]]


class ComfyWSClient:
    """
    WebSocket client for real-time ComfyUI communication.

    Provides instant progress updates instead of polling.

    Args:
        base_url: ComfyUI server URL (ws:// or http://)
        client_id: Unique client identifier
        reconnect_attempts: Max reconnection attempts
        reconnect_delay: Base delay between reconnects
        max_message_size: Maximum WebSocket message size (security limit)

    Security Notes:
        - Use wss:// (HTTPS) in production to encrypt WebSocket traffic
        - max_message_size prevents DoS via large messages
        - Listener count is limited to prevent memory exhaustion
    """

    # Security: Maximum listeners per prompt to prevent memory exhaustion
    MAX_LISTENERS_PER_PROMPT = 100
    # Security: Default maximum message size (1MB) to prevent DoS
    DEFAULT_MAX_MESSAGE_SIZE = 1_048_576

    def __init__(
        self,
        base_url: str | None = None,
        client_id: str | None = None,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        *,
        max_message_size: int | None = None,
    ):
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required. Install with: pip install websockets")

        # Convert HTTP URL to WebSocket URL
        http_url = (base_url or settings.comfyui.url).rstrip("/")
        uses_encryption = False

        if http_url.startswith("http://"):
            self.ws_url = http_url.replace("http://", "ws://") + "/ws"
        elif http_url.startswith("https://"):
            self.ws_url = http_url.replace("https://", "wss://") + "/ws"
            uses_encryption = True
        else:
            self.ws_url = f"ws://{http_url}/ws"

        # Security: Warn about unencrypted connections
        if not uses_encryption:
            logger.warning(
                "WebSocket using unencrypted ws:// protocol. "
                "Consider using wss:// (HTTPS) for production environments "
                "to prevent man-in-the-middle attacks.",
                extra={"ws_url": self.ws_url},
            )

        self.client_id = client_id or str(uuid.uuid4())
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.max_message_size = max_message_size or self.DEFAULT_MAX_MESSAGE_SIZE

        self._ws: ClientConnection | None = None
        self._connected = False
        self._listeners: dict[str, list[ProgressCallback]] = {}
        self._message_task: asyncio.Task | None = None

        logger.debug("ComfyWSClient initialized", extra={"ws_url": self.ws_url})

    @property
    def connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected and self._ws is not None and not self._ws.closed

    async def connect(self) -> bool:
        """
        Connect to ComfyUI WebSocket.

        Returns:
            True if connected successfully
        """
        url = f"{self.ws_url}?clientId={self.client_id}"

        for attempt in range(self.reconnect_attempts):
            try:
                self._ws = await websockets.connect(
                    url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=self.max_message_size,  # Security: Prevent DoS via large messages
                )
                self._connected = True

                # Start message handler
                self._message_task = asyncio.create_task(self._message_loop())

                logger.info("WebSocket connected", extra={"client_id": self.client_id[:8]})
                return True

            except Exception as e:
                delay = self.reconnect_delay * (2**attempt)
                logger.warning(
                    f"WebSocket connection failed (attempt {attempt + 1})",
                    extra={"error": str(e), "retry_in": delay},
                )
                if attempt < self.reconnect_attempts - 1:
                    await asyncio.sleep(delay)

        raise ComfyUIConnectionError(
            message="Failed to connect to ComfyUI WebSocket", url=self.ws_url
        )

    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self._message_task:
            self._message_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._message_task
            self._message_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._connected = False
        logger.debug("WebSocket disconnected")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def _message_loop(self):
        """Background task to process incoming messages."""
        try:
            async for message in self._ws:
                await self._handle_message(message)
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self._connected = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self._connected = False

    async def _handle_message(self, message: bytes | str):
        """Process a WebSocket message."""
        # Handle binary preview images
        if isinstance(message, bytes):
            # Binary data is preview image
            # First 4 bytes might be message type, rest is image
            if len(message) > 8:
                await self._notify_listeners(
                    "preview",
                    WSProgress(
                        prompt_id="",
                        status="preview",
                        preview_data=message[8:],  # Skip header
                    ),
                )
            return

        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            # Log with more context for debugging malformed messages
            truncated_msg = message[:200] if len(message) > 200 else message
            logger.warning(
                "Malformed WebSocket message received",
                extra={
                    "error": str(e),
                    "message_preview": truncated_msg,
                    "message_length": len(message),
                },
            )
            return

        msg_type = data.get("type", "")
        msg_data = data.get("data", {})

        # Extract prompt_id
        prompt_id = msg_data.get("prompt_id", "")

        if msg_type == WSMessageType.STATUS:
            # Queue status update
            queue_info = msg_data.get("status", {}).get("exec_info", {})
            queue_remaining = queue_info.get("queue_remaining", 0)
            logger.debug(f"Queue status: {queue_remaining} remaining")

        elif msg_type == WSMessageType.EXECUTION_START:
            progress = WSProgress(
                prompt_id=prompt_id,
                status="started",
                progress=0.0,
            )
            await self._notify_listeners(prompt_id, progress)

        elif msg_type == WSMessageType.EXECUTING:
            node_id = msg_data.get("node")
            if node_id is None:
                # None node means execution complete
                progress = WSProgress(
                    prompt_id=prompt_id,
                    status="completed",
                    progress=1.0,
                    max_progress=1.0,
                )
            else:
                progress = WSProgress(
                    prompt_id=prompt_id,
                    node_id=node_id,
                    status="executing",
                )
            await self._notify_listeners(prompt_id, progress)

        elif msg_type == WSMessageType.PROGRESS:
            progress = WSProgress(
                prompt_id=prompt_id,
                node_id=msg_data.get("node"),
                progress=float(msg_data.get("value", 0)),
                max_progress=float(msg_data.get("max", 1)),
                status="progress",
            )
            await self._notify_listeners(prompt_id, progress)

        elif msg_type == WSMessageType.EXECUTED:
            node_id = msg_data.get("node")
            progress = WSProgress(
                prompt_id=prompt_id,
                node_id=node_id,
                status="node_complete",
            )
            await self._notify_listeners(prompt_id, progress)

        elif msg_type == WSMessageType.EXECUTION_CACHED:
            msg_data.get("nodes", [])
            progress = WSProgress(
                prompt_id=prompt_id,
                status="cached",
            )
            await self._notify_listeners(prompt_id, progress)

        elif msg_type == WSMessageType.EXECUTION_ERROR:
            progress = WSProgress(
                prompt_id=prompt_id,
                status="error",
            )
            await self._notify_listeners(prompt_id, progress)

        elif msg_type == WSMessageType.EXECUTION_INTERRUPTED:
            progress = WSProgress(
                prompt_id=prompt_id,
                status="interrupted",
            )
            await self._notify_listeners(prompt_id, progress)

    def add_listener(self, prompt_id: str, callback: ProgressCallback):
        """
        Add a progress listener for a prompt.

        Security: Limited to MAX_LISTENERS_PER_PROMPT to prevent memory exhaustion.
        """
        if prompt_id not in self._listeners:
            self._listeners[prompt_id] = []

        # Security: Prevent memory exhaustion from too many listeners
        if len(self._listeners[prompt_id]) >= self.MAX_LISTENERS_PER_PROMPT:
            logger.warning(
                f"Maximum listeners ({self.MAX_LISTENERS_PER_PROMPT}) reached for prompt",
                extra={"prompt_id": prompt_id[:8] if prompt_id else "global"},
            )
            return

        self._listeners[prompt_id].append(callback)

    def remove_listener(self, prompt_id: str, callback: ProgressCallback):
        """Remove a progress listener."""
        if prompt_id in self._listeners:
            self._listeners[prompt_id] = [cb for cb in self._listeners[prompt_id] if cb != callback]

    async def _notify_listeners(self, prompt_id: str, progress: WSProgress):
        """Notify all listeners for a prompt."""
        if prompt_id in self._listeners:
            for callback in self._listeners[prompt_id]:
                try:
                    await callback(progress)
                except Exception as e:
                    logger.warning(f"Listener error: {e}")

        # Also notify global listeners (empty prompt_id)
        if "" in self._listeners:
            for callback in self._listeners[""]:
                try:
                    await callback(progress)
                except Exception as e:
                    logger.warning(f"Global listener error: {e}")

    async def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """
        Queue a workflow for execution.

        Args:
            workflow: ComfyUI workflow dict

        Returns:
            prompt_id
        """
        import httpx

        http_url = self.ws_url.replace("ws://", "http://").replace("wss://", "https://")
        http_url = http_url.replace("/ws", "")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{http_url}/prompt",
                json={"prompt": workflow, "client_id": self.client_id},
                timeout=settings.comfyui.timeout_queue,
            )
            response.raise_for_status()
            data = response.json()

        prompt_id = data.get("prompt_id")
        logger.info("Queued prompt", extra={"prompt_id": prompt_id[:8]})
        return prompt_id

    async def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> WSProgress:
        """
        Wait for a prompt to complete using WebSocket updates.

        Args:
            prompt_id: The prompt ID to wait for
            timeout: Maximum wait time in seconds
            on_progress: Callback for progress updates

        Returns:
            Final progress state
        """
        timeout = timeout or settings.generation.generation_timeout
        completed = asyncio.Event()
        final_progress: WSProgress = WSProgress(prompt_id=prompt_id)

        async def handler(progress: WSProgress):
            nonlocal final_progress
            final_progress = progress

            if on_progress:
                await on_progress(progress)

            if progress.status in ("completed", "error", "interrupted"):
                completed.set()

        self.add_listener(prompt_id, handler)

        try:
            await asyncio.wait_for(completed.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            final_progress = WSProgress(
                prompt_id=prompt_id,
                status="timeout",
            )
        finally:
            self.remove_listener(prompt_id, handler)

        return final_progress

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        """Get execution history for a prompt."""
        import httpx

        http_url = self.ws_url.replace("ws://", "http://").replace("wss://", "https://")
        http_url = http_url.replace("/ws", "")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{http_url}/history/{prompt_id}",
                timeout=settings.comfyui.timeout_read,
            )
            response.raise_for_status()
            return response.json()

    async def get_image(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes | None:
        """Download a generated image."""
        import httpx

        http_url = self.ws_url.replace("ws://", "http://").replace("wss://", "https://")
        http_url = http_url.replace("/ws", "")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{http_url}/view",
                params={"filename": filename, "subfolder": subfolder, "type": folder_type},
                timeout=settings.comfyui.timeout_image,
            )
            if response.status_code == 200:
                return response.content
        return None
