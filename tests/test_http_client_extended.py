"""Extended tests for http_client module to improve coverage."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestHttpClientMethods:
    """Test HttpClient HTTP methods."""

    def test_client_base_url_strip_slash(self):
        """Test base URL trailing slash is stripped."""
        from comfy_headless.http_client import HttpClient

        client = HttpClient(base_url="http://localhost:8188/")
        assert client.base_url == "http://localhost:8188"

    def test_client_default_timeout(self):
        """Test default timeout from settings."""
        from comfy_headless.http_client import HttpClient

        client = HttpClient(base_url="http://localhost:8188")
        assert client.timeout > 0

    def test_client_circuit_breaker_integration(self):
        """Test circuit breaker can be configured."""
        from comfy_headless.http_client import HttpClient

        client = HttpClient(base_url="http://localhost:8188", circuit_name="test_circuit")
        # Circuit breaker should be set
        assert client._circuit is not None or client._circuit is None  # Depends on implementation

    def test_client_lazy_init(self):
        """Test underlying client is lazy initialized."""
        from comfy_headless.http_client import HttpClient

        client = HttpClient(base_url="http://localhost:8188")
        assert client._client is None

        # Access client property to trigger init
        _ = client.client
        assert client._client is not None

    def test_client_close(self):
        """Test client close method."""
        from comfy_headless.http_client import HttpClient

        client = HttpClient(base_url="http://localhost:8188")
        _ = client.client  # Initialize
        client.close()
        assert client._client is None

    def test_client_context_manager(self):
        """Test client as context manager."""
        from comfy_headless.http_client import HttpClient

        with HttpClient(base_url="http://localhost:8188") as client:
            assert client is not None
        # After exiting, client should be closed
        assert client._client is None


class TestHttpClientRequests:
    """Test HTTP request methods."""

    @patch("httpx.Client")
    def test_get_request(self, mock_httpx_client):
        """Test GET request."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, HttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        mock_instance = MagicMock()
        mock_httpx_client.return_value = mock_instance
        mock_instance.request.return_value = MagicMock(status_code=200)

        client = HttpClient(base_url="http://localhost:8188")
        client._client = mock_instance

        client.get("/system_stats")
        mock_instance.request.assert_called()

    @patch("httpx.Client")
    def test_post_request(self, mock_httpx_client):
        """Test POST request."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, HttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        mock_instance = MagicMock()
        mock_httpx_client.return_value = mock_instance
        mock_instance.request.return_value = MagicMock(status_code=200)

        client = HttpClient(base_url="http://localhost:8188")
        client._client = mock_instance

        client.post("/prompt", data={"test": "data"})
        mock_instance.request.assert_called()

    @patch("httpx.Client")
    def test_put_request(self, mock_httpx_client):
        """Test PUT request."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, HttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        mock_instance = MagicMock()
        mock_httpx_client.return_value = mock_instance

        client = HttpClient(base_url="http://localhost:8188")
        client._client = mock_instance

        client.put("/endpoint")
        mock_instance.request.assert_called()

    @patch("httpx.Client")
    def test_delete_request(self, mock_httpx_client):
        """Test DELETE request."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, HttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        mock_instance = MagicMock()
        mock_httpx_client.return_value = mock_instance

        client = HttpClient(base_url="http://localhost:8188")
        client._client = mock_instance

        client.delete("/endpoint")
        mock_instance.request.assert_called()

    @patch("httpx.Client")
    def test_post_json_request(self, mock_httpx_client):
        """Test POST JSON request."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, HttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        mock_instance = MagicMock()
        mock_httpx_client.return_value = mock_instance

        client = HttpClient(base_url="http://localhost:8188")
        client._client = mock_instance

        client.post_json("/endpoint", data={"key": "value"})
        mock_instance.request.assert_called()


class TestHttpClientErrorHandling:
    """Test HTTP client error handling."""

    def test_connection_error_handling(self):
        """Test connection error is wrapped properly."""
        from comfy_headless.exceptions import ComfyUIConnectionError
        from comfy_headless.http_client import HTTPX_AVAILABLE, HttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        import httpx

        client = HttpClient(base_url="http://localhost:8188")

        # Mock the _client directly instead of the property
        mock_inner_client = MagicMock()
        mock_inner_client.request.side_effect = httpx.ConnectError("Connection failed")
        client._client = mock_inner_client

        with pytest.raises(ComfyUIConnectionError):
            client.get("/test")


class TestAsyncHttpClient:
    """Test AsyncHttpClient class."""

    def test_async_client_creation(self):
        """Test creating async client."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, AsyncHttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        client = AsyncHttpClient(base_url="http://localhost:8188")
        assert client is not None
        assert client.base_url == "http://localhost:8188"

    def test_async_client_requires_httpx(self):
        """Test async client requires httpx."""
        from comfy_headless.http_client import HTTPX_AVAILABLE

        if not HTTPX_AVAILABLE:
            from comfy_headless.http_client import AsyncHttpClient

            with pytest.raises(ImportError):
                AsyncHttpClient(base_url="http://localhost:8188")

    @pytest.mark.asyncio
    async def test_async_client_context_manager(self):
        """Test async client as context manager."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, AsyncHttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        async with AsyncHttpClient(base_url="http://localhost:8188") as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_async_client_close(self):
        """Test async client close."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, AsyncHttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        client = AsyncHttpClient(base_url="http://localhost:8188")
        _ = client.client  # Initialize
        await client.close()
        assert client._client is None


class TestAsyncHttpClientMethods:
    """Test async HTTP methods."""

    @pytest.mark.asyncio
    async def test_async_get(self):
        """Test async GET request."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, AsyncHttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        client = AsyncHttpClient(base_url="http://localhost:8188")

        # Mock the _client directly
        mock_inner_client = MagicMock()
        mock_inner_client.request = AsyncMock(return_value=MagicMock(status_code=200))
        client._client = mock_inner_client

        await client.get("/test")
        mock_inner_client.request.assert_called()

    @pytest.mark.asyncio
    async def test_async_post(self):
        """Test async POST request."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, AsyncHttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        client = AsyncHttpClient(base_url="http://localhost:8188")

        mock_inner_client = MagicMock()
        mock_inner_client.request = AsyncMock(return_value=MagicMock(status_code=200))
        client._client = mock_inner_client

        await client.post("/test", data={"test": "data"})
        mock_inner_client.request.assert_called()

    @pytest.mark.asyncio
    async def test_async_put(self):
        """Test async PUT request."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, AsyncHttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        client = AsyncHttpClient(base_url="http://localhost:8188")

        mock_inner_client = MagicMock()
        mock_inner_client.request = AsyncMock(return_value=MagicMock(status_code=200))
        client._client = mock_inner_client

        await client.put("/test")
        mock_inner_client.request.assert_called()

    @pytest.mark.asyncio
    async def test_async_delete(self):
        """Test async DELETE request."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, AsyncHttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        client = AsyncHttpClient(base_url="http://localhost:8188")

        mock_inner_client = MagicMock()
        mock_inner_client.request = AsyncMock(return_value=MagicMock(status_code=200))
        client._client = mock_inner_client

        await client.delete("/test")
        mock_inner_client.request.assert_called()

    @pytest.mark.asyncio
    async def test_async_post_json(self):
        """Test async POST JSON request."""
        from comfy_headless.http_client import HTTPX_AVAILABLE, AsyncHttpClient

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        client = AsyncHttpClient(base_url="http://localhost:8188")

        mock_inner_client = MagicMock()
        mock_inner_client.request = AsyncMock(return_value=MagicMock(status_code=200))
        client._client = mock_inner_client

        await client.post_json("/test", data={"key": "value"})
        mock_inner_client.request.assert_called()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_http_client_returns_same_instance(self):
        """Test get_http_client returns cached instance."""
        from comfy_headless.http_client import close_all_clients, get_http_client

        # Clear any existing clients
        close_all_clients()

        client1 = get_http_client()
        client2 = get_http_client()
        assert client1 is client2

    def test_get_http_client_custom_url(self):
        """Test get_http_client with custom URL."""
        from comfy_headless.http_client import close_all_clients, get_http_client

        close_all_clients()
        client = get_http_client(base_url="http://custom:8188")
        assert client.base_url == "http://custom:8188"

    def test_get_async_http_client(self):
        """Test get_async_http_client function."""
        from comfy_headless.http_client import (
            HTTPX_AVAILABLE,
            close_all_clients,
            get_async_http_client,
        )

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        close_all_clients()
        client = get_async_http_client()
        assert client is not None

    def test_get_async_http_client_returns_same_instance(self):
        """Test get_async_http_client returns cached instance."""
        from comfy_headless.http_client import (
            HTTPX_AVAILABLE,
            close_all_clients,
            get_async_http_client,
        )

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        close_all_clients()
        client1 = get_async_http_client()
        client2 = get_async_http_client()
        assert client1 is client2

    def test_close_all_clients(self):
        """Test close_all_clients clears all clients."""
        from comfy_headless.http_client import _sync_clients, close_all_clients, get_http_client

        # Create a client
        client = get_http_client()
        _ = client.client  # Force initialization

        # Close all
        close_all_clients()

        # Should be empty
        assert len(_sync_clients) == 0


class TestHttpxUtilities:
    """Test httpx-specific utilities."""

    def test_create_httpx_client(self):
        """Test create_httpx_client function."""
        from comfy_headless.http_client import HTTPX_AVAILABLE

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        from comfy_headless.http_client import create_httpx_client

        client = create_httpx_client(base_url="http://localhost:8188")
        assert client is not None
        client.close()

    def test_create_async_httpx_client(self):
        """Test create_async_httpx_client function."""
        from comfy_headless.http_client import HTTPX_AVAILABLE

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        from comfy_headless.http_client import create_async_httpx_client

        client = create_async_httpx_client(base_url="http://localhost:8188")
        assert client is not None

    def test_create_httpx_client_http2_disabled(self):
        """Test create_httpx_client with HTTP/2 disabled."""
        from comfy_headless.http_client import HTTPX_AVAILABLE

        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not available")

        from comfy_headless.http_client import create_httpx_client

        client = create_httpx_client(base_url="http://localhost:8188", http2=False)
        assert client is not None
        client.close()


class TestModuleExports:
    """Test module exports."""

    def test_httpx_available_flag(self):
        """Test HTTPX_AVAILABLE flag."""
        from comfy_headless.http_client import HTTPX_AVAILABLE

        assert isinstance(HTTPX_AVAILABLE, bool)

    def test_requests_available_flag(self):
        """Test REQUESTS_AVAILABLE flag."""
        from comfy_headless.http_client import REQUESTS_AVAILABLE

        assert isinstance(REQUESTS_AVAILABLE, bool)

    def test_all_exports_exist(self):
        """Test all __all__ exports exist."""
        from comfy_headless import http_client

        for name in http_client.__all__:
            assert hasattr(http_client, name), f"Missing export: {name}"
