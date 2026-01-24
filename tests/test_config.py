"""Tests for config module."""



class TestSettings:
    """Test Settings configuration."""

    def test_default_settings(self):
        """Test default settings values."""
        from comfy_headless.config import Settings

        settings = Settings()

        assert settings.comfyui.url == "http://localhost:8188"
        assert settings.ollama.url == "http://localhost:11434"
        assert settings.retry.max_retries == 3
        assert settings.generation.default_width == 1024
        assert settings.generation.default_height == 1024

    def test_version(self):
        """Test version is set."""
        from comfy_headless.config import Settings

        settings = Settings()
        assert settings.version == "2.5.0"

    def test_to_dict(self):
        """Test settings serialization."""
        from comfy_headless.config import Settings

        settings = Settings()
        data = settings.to_dict()

        assert "version" in data
        assert "comfyui" in data
        assert data["comfyui"]["url"] == "http://localhost:8188"


class TestGetSettings:
    """Test get_settings caching."""

    def test_get_settings_returns_same_instance(self):
        """Test that get_settings returns cached instance."""
        from comfy_headless.config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload_settings(self):
        """Test settings can be reloaded."""
        from comfy_headless.config import get_settings, reload_settings

        get_settings()
        s2 = reload_settings()
        # After reload, should be different instance
        s3 = get_settings()
        assert s2 is s3


class TestTempDir:
    """Test temp directory utilities."""

    def test_get_temp_dir_creates_directory(self):
        """Test temp directory is created."""
        from comfy_headless.config import get_temp_dir

        temp_dir = get_temp_dir()
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        assert temp_dir.name == "comfy_headless"
