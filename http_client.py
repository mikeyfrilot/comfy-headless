"""
Comfy Headless - Modern HTTP Client
====================================

Dual sync/async HTTP client using httpx with HTTP/2 support.

Features:
- Sync and async APIs with identical interface
- HTTP/2 support for better performance
- Connection pooling
- Automatic retry with circuit breaker
- Full type annotations

Usage:
    # Sync usage
    from comfy_headless.http_client import HttpClient

    client = HttpClient(base_url="http://localhost:8188")
    response = client.get("/system_stats")

    # Async usage
    async with AsyncHttpClient(base_url="http://localhost:8188") as client:
        response = await client.get("/system_stats")

    # Or use the convenience functions
    from comfy_headless.http_client import get_http_client, get_async_http_client
"""

from typing import Any

from .config import settings
from .exceptions import ComfyUIConnectionError
from .logging_config import get_logger
from .retry import get_circuit_breaker

logger = get_logger(__name__)

__all__ = [
    # Sync client
    "HttpClient",
    "get_http_client",
    # Async client
    "AsyncHttpClient",
    "get_async_http_client",
    # Cleanup
    "close_all_clients",
    # Constants
    "HTTPX_AVAILABLE",
    "REQUESTS_AVAILABLE",
]

# Try to import httpx, fall back to requests
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

# Fallback to requests if httpx unavailable
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


# =============================================================================
# SYNC HTTP CLIENT
# =============================================================================


class HttpClient:
    """
    Modern sync HTTP client using httpx (with requests fallback).

    Features:
    - Connection pooling
    - HTTP/2 support (httpx only)
    - Circuit breaker integration
    - Automatic retry

    Usage:
        client = HttpClient(base_url="http://localhost:8188")
        response = client.get("/api/endpoint")
        data = response.json()
    """

    def __init__(
        self,
        base_url: str,
        timeout: float | None = None,
        circuit_name: str | None = None,
    ):
        """
        Initialize HTTP client.

        Args:
            base_url: Base URL for all requests
            timeout: Default timeout in seconds
            circuit_name: Name for circuit breaker (None to disable)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout or settings.http.read_timeout
        self._circuit = get_circuit_breaker(circuit_name) if circuit_name else None
        self._client: Any | None = None

        logger.debug(
            "HttpClient initialized", extra={"base_url": self.base_url, "httpx": HTTPX_AVAILABLE}
        )

    @property
    def client(self):
        """Lazy-initialize the underlying HTTP client."""
        if self._client is None:
            if HTTPX_AVAILABLE:
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(
                        connect=settings.http.connect_timeout,
                        read=settings.http.read_timeout,
                        write=settings.http.write_timeout,
                        pool=settings.http.pool_timeout,
                    ),
                    http2=settings.http.http2,
                    limits=httpx.Limits(
                        max_connections=settings.http.max_connections,
                        max_keepalive_connections=settings.http.max_keepalive_connections,
                        keepalive_expiry=settings.http.keepalive_expiry,
                    ),
                )
            elif REQUESTS_AVAILABLE:
                self._client = requests.Session()
            else:
                raise ImportError("Neither httpx nor requests is installed")
        return self._client

    def _request(
        self, method: str, endpoint: str, timeout: float | None = None, **kwargs
    ) -> Any:
        """
        Make an HTTP request with circuit breaker protection.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (will be appended to base_url)
            timeout: Request timeout (overrides default)
            **kwargs: Additional arguments passed to the HTTP client

        Returns:
            Response object (httpx.Response or requests.Response)
        """
        url = f"{self.base_url}{endpoint}" if endpoint.startswith("/") else endpoint
        _timeout = timeout or self.timeout

        def make_request():
            if HTTPX_AVAILABLE:
                return self.client.request(method, endpoint, timeout=_timeout, **kwargs)
            else:
                return self.client.request(method, url, timeout=_timeout, **kwargs)

        try:
            if self._circuit:
                with self._circuit:
                    return make_request()
            else:
                return make_request()

        except Exception as e:
            if HTTPX_AVAILABLE and isinstance(e, httpx.ConnectError) or REQUESTS_AVAILABLE and isinstance(e, requests.exceptions.ConnectionError):
                raise ComfyUIConnectionError(
                    f"Failed to connect to {self.base_url}: {e}",
                    url=self.base_url,
                )
            raise

    def get(self, endpoint: str, **kwargs) -> Any:
        """Make a GET request."""
        return self._request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> Any:
        """Make a POST request."""
        return self._request("POST", endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs) -> Any:
        """Make a PUT request."""
        return self._request("PUT", endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Any:
        """Make a DELETE request."""
        return self._request("DELETE", endpoint, **kwargs)

    def post_json(self, endpoint: str, data: dict, **kwargs) -> Any:
        """Make a POST request with JSON body."""
        if HTTPX_AVAILABLE:
            return self._request("POST", endpoint, json=data, **kwargs)
        else:
            return self._request("POST", endpoint, json=data, **kwargs)

    def close(self):
        """Close the underlying client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# ASYNC HTTP CLIENT
# =============================================================================


class AsyncHttpClient:
    """
    Modern async HTTP client using httpx.

    Features:
    - Full async/await support
    - Connection pooling
    - HTTP/2 support
    - Circuit breaker integration

    Usage:
        async with AsyncHttpClient(base_url="http://localhost:8188") as client:
            response = await client.get("/api/endpoint")
            data = response.json()
    """

    def __init__(
        self,
        base_url: str,
        timeout: float | None = None,
        circuit_name: str | None = None,
    ):
        """
        Initialize async HTTP client.

        Args:
            base_url: Base URL for all requests
            timeout: Default timeout in seconds
            circuit_name: Name for circuit breaker (None to disable)
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for async HTTP client. Install with: pip install httpx"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout or settings.http.read_timeout
        self._circuit = get_circuit_breaker(circuit_name) if circuit_name else None
        self._client: httpx.AsyncClient | None = None

        logger.debug("AsyncHttpClient initialized", extra={"base_url": self.base_url})

    @property
    def client(self) -> "httpx.AsyncClient":
        """Lazy-initialize the underlying async client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=settings.http.connect_timeout,
                    read=settings.http.read_timeout,
                    write=settings.http.write_timeout,
                    pool=settings.http.pool_timeout,
                ),
                http2=settings.http.http2,
                limits=httpx.Limits(
                    max_connections=settings.http.max_connections,
                    max_keepalive_connections=settings.http.max_keepalive_connections,
                    keepalive_expiry=settings.http.keepalive_expiry,
                ),
            )
        return self._client

    async def _request(
        self, method: str, endpoint: str, timeout: float | None = None, **kwargs
    ) -> httpx.Response:
        """Make an async HTTP request with circuit breaker protection."""
        _timeout = timeout or self.timeout

        async def make_request():
            return await self.client.request(method, endpoint, timeout=_timeout, **kwargs)

        try:
            if self._circuit:
                async with self._circuit:
                    return await make_request()
            else:
                return await make_request()

        except httpx.ConnectError as e:
            raise ComfyUIConnectionError(
                f"Failed to connect to {self.base_url}: {e}",
                url=self.base_url,
            )

    async def get(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make a GET request."""
        return await self._request("GET", endpoint, **kwargs)

    async def post(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make a POST request."""
        return await self._request("POST", endpoint, **kwargs)

    async def put(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make a PUT request."""
        return await self._request("PUT", endpoint, **kwargs)

    async def delete(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make a DELETE request."""
        return await self._request("DELETE", endpoint, **kwargs)

    async def post_json(self, endpoint: str, data: dict, **kwargs) -> httpx.Response:
        """Make a POST request with JSON body."""
        return await self._request("POST", endpoint, json=data, **kwargs)

    async def close(self):
        """Close the underlying client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_sync_clients: dict[str, HttpClient] = {}
_async_clients: dict[str, AsyncHttpClient] = {}


def get_http_client(
    base_url: str | None = None,
    circuit_name: str | None = None,
) -> HttpClient:
    """
    Get a shared sync HTTP client instance.

    Args:
        base_url: Base URL (default: ComfyUI URL from settings)
        circuit_name: Circuit breaker name

    Returns:
        Shared HttpClient instance
    """
    url = base_url or settings.comfyui.url
    key = f"{url}:{circuit_name}"

    if key not in _sync_clients:
        _sync_clients[key] = HttpClient(
            base_url=url,
            circuit_name=circuit_name,
        )

    return _sync_clients[key]


def get_async_http_client(
    base_url: str | None = None,
    circuit_name: str | None = None,
) -> AsyncHttpClient:
    """
    Get a shared async HTTP client instance.

    Args:
        base_url: Base URL (default: ComfyUI URL from settings)
        circuit_name: Circuit breaker name

    Returns:
        Shared AsyncHttpClient instance
    """
    url = base_url or settings.comfyui.url
    key = f"{url}:{circuit_name}"

    if key not in _async_clients:
        _async_clients[key] = AsyncHttpClient(
            base_url=url,
            circuit_name=circuit_name,
        )

    return _async_clients[key]


def close_all_clients():
    """Close all shared HTTP clients."""
    for client in _sync_clients.values():
        client.close()
    _sync_clients.clear()

    # Note: async clients need to be closed in an async context
    # This just clears references
    _async_clients.clear()


# =============================================================================
# HTTPX-SPECIFIC UTILITIES
# =============================================================================

if HTTPX_AVAILABLE:

    def create_httpx_client(base_url: str, http2: bool = True, **kwargs) -> httpx.Client:
        """
        Create a configured httpx.Client with sensible defaults.

        Args:
            base_url: Base URL for requests
            http2: Enable HTTP/2 (default True)
            **kwargs: Additional httpx.Client arguments

        Returns:
            Configured httpx.Client
        """
        return httpx.Client(
            base_url=base_url,
            http2=http2,
            timeout=httpx.Timeout(
                connect=settings.http.connect_timeout,
                read=settings.http.read_timeout,
                write=settings.http.write_timeout,
                pool=settings.http.pool_timeout,
            ),
            limits=httpx.Limits(
                max_connections=settings.http.max_connections,
                max_keepalive_connections=settings.http.max_keepalive_connections,
            ),
            **kwargs,
        )

    def create_async_httpx_client(base_url: str, http2: bool = True, **kwargs) -> httpx.AsyncClient:
        """
        Create a configured httpx.AsyncClient with sensible defaults.

        Args:
            base_url: Base URL for requests
            http2: Enable HTTP/2 (default True)
            **kwargs: Additional httpx.AsyncClient arguments

        Returns:
            Configured httpx.AsyncClient
        """
        return httpx.AsyncClient(
            base_url=base_url,
            http2=http2,
            timeout=httpx.Timeout(
                connect=settings.http.connect_timeout,
                read=settings.http.read_timeout,
                write=settings.http.write_timeout,
                pool=settings.http.pool_timeout,
            ),
            limits=httpx.Limits(
                max_connections=settings.http.max_connections,
                max_keepalive_connections=settings.http.max_keepalive_connections,
            ),
            **kwargs,
        )
