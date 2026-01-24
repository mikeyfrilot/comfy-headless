"""Tests for cleanup module."""

import os
from pathlib import Path


class TestTempFileManager:
    """Test TempFileManager class."""

    def test_temp_file_manager_exists(self):
        """Test TempFileManager class exists."""
        from comfy_headless.cleanup import TempFileManager

        manager = TempFileManager()
        assert manager is not None

    def test_temp_dir_created(self):
        """Test temp directory is created."""
        from comfy_headless.cleanup import TempFileManager

        manager = TempFileManager()
        assert manager.base_dir.exists()

    def test_create_temp_file(self):
        """Test creating a temp file."""
        from comfy_headless.cleanup import TempFileManager

        manager = TempFileManager()
        temp_file = manager.create_temp_file(suffix=".txt")

        assert temp_file is not None
        assert Path(temp_file).exists()

        # Cleanup
        if Path(temp_file).exists():
            os.unlink(temp_file)


class TestGetTempManager:
    """Test get_temp_manager singleton."""

    def test_get_temp_manager_returns_manager(self):
        """Test get_temp_manager returns TempFileManager."""
        from comfy_headless.cleanup import TempFileManager, get_temp_manager

        manager = get_temp_manager()
        assert isinstance(manager, TempFileManager)

    def test_get_temp_manager_singleton(self):
        """Test get_temp_manager returns same instance."""
        from comfy_headless.cleanup import get_temp_manager

        m1 = get_temp_manager()
        m2 = get_temp_manager()
        assert m1 is m2


class TestCleanupFunctions:
    """Test cleanup utility functions."""

    def test_cleanup_temp_files(self):
        """Test cleanup_temp_files doesn't raise."""
        from comfy_headless.cleanup import cleanup_temp_files

        # Should not raise
        cleanup_temp_files()

    def test_cleanup_all(self):
        """Test cleanup_all doesn't raise."""
        from comfy_headless.cleanup import cleanup_all

        # Should not raise
        cleanup_all()

    def test_register_cleanup_callback(self):
        """Test registering cleanup callback."""
        from comfy_headless.cleanup import register_cleanup_callback

        called = []

        def callback():
            called.append(True)

        register_cleanup_callback(callback)
        # Callback registered, not immediately called


class TestSaveTempFunctions:
    """Test save_temp_* helper functions."""

    def test_save_temp_image(self):
        """Test save_temp_image."""
        from comfy_headless.cleanup import save_temp_image

        # Create simple PNG bytes (1x1 pixel)
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        path = save_temp_image(png_data)
        assert path is not None
        assert Path(path).exists()
        assert path.endswith(".png")

        # Cleanup
        if Path(path).exists():
            os.unlink(path)

    def test_save_temp_video(self):
        """Test save_temp_video."""
        from comfy_headless.cleanup import save_temp_video

        video_data = b"\x00" * 100

        path = save_temp_video(video_data, format="mp4")
        assert path is not None
        assert Path(path).exists()
        assert path.endswith(".mp4")

        # Cleanup
        if Path(path).exists():
            os.unlink(path)


class TestCleanupThread:
    """Test CleanupThread."""

    def test_cleanup_thread_exists(self):
        """Test CleanupThread class exists."""
        from comfy_headless.cleanup import CleanupThread

        assert CleanupThread is not None

    def test_cleanup_thread_creation(self):
        """Test creating CleanupThread."""
        from comfy_headless.cleanup import CleanupThread

        thread = CleanupThread()
        assert thread is not None
        assert not thread.is_alive()
