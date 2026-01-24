"""Tests for http_client module."""



class TestHttpClientInitialization:
    """Test HTTP client initialization."""

    def test_client_creation(self):
        """Test creating HTTP client."""
        from comfy_headless.http_client import HttpClient

        client = HttpClient(base_url="http://localhost:8188")
        assert client is not None

    def test_client_with_base_url(self):
        """Test client with base URL."""
        from comfy_headless.http_client import HttpClient

        client = HttpClient(base_url="http://localhost:8188")
        assert client.base_url == "http://localhost:8188"

    def test_client_with_timeout(self):
        """Test client with custom timeout."""
        from comfy_headless.http_client import HttpClient

        client = HttpClient(base_url="http://localhost:8188", timeout=60)
        assert client.timeout == 60


class TestHttpClientHelpers:
    """Test HTTP client helper functions."""

    def test_get_http_client(self):
        """Test get_http_client factory function."""
        from comfy_headless.http_client import get_http_client

        client = get_http_client()
        assert client is not None

    def test_close_all_clients(self):
        """Test close_all_clients doesn't raise."""
        from comfy_headless.http_client import close_all_clients

        # Should not raise
        close_all_clients()


class TestAsyncHttpClient:
    """Test async HTTP client."""

    def test_async_client_exists(self):
        """Test AsyncHttpClient class exists."""
        from comfy_headless.http_client import AsyncHttpClient

        # Just verify the class exists
        assert AsyncHttpClient is not None
