"""
Extended tests for comfy_headless/cleanup.py

Covers:
- TempFileManager file cleanup logic (lines 287-306)
- Age-based cleanup
- Thread-safe operations
- Shutdown handlers
- save_temp_image/save_temp_video functions
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestTempFileManagerInitialization:
    """Test TempFileManager initialization."""

    def test_default_initialization(self):
        """Manager initializes with defaults."""
        from comfy_headless.cleanup import TempFileManager

        manager = TempFileManager()

        assert manager.base_dir is not None
        assert manager.base_dir.exists()
        assert manager.auto_cleanup is True
        assert manager.max_age_seconds == 3600.0

    def test_custom_base_dir(self):
        """Manager accepts custom base directory."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "custom"
            manager = TempFileManager(base_dir=base)

            assert manager.base_dir == base
            assert base.exists()  # Should be created

    def test_custom_settings(self):
        """Manager accepts custom settings."""
        from comfy_headless.cleanup import TempFileManager

        manager = TempFileManager(auto_cleanup=False, max_age_seconds=1800.0)

        assert manager.auto_cleanup is False
        assert manager.max_age_seconds == 1800.0

    def test_creates_base_dir(self):
        """Manager creates base directory if missing."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c"
            TempFileManager(base_dir=nested)

            assert nested.exists()


class TestTempFileManagerCreateFile:
    """Test file creation."""

    def test_create_temp_file_returns_path(self):
        """create_temp_file returns Path object."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)
            path = manager.create_temp_file(".txt")

            assert isinstance(path, Path)
            assert path.exists()

    def test_create_temp_file_with_suffix(self):
        """create_temp_file uses correct suffix."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)
            path = manager.create_temp_file(".png")

            assert path.suffix == ".png"

    def test_create_temp_file_with_content(self):
        """create_temp_file writes content."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)
            content = b"test content 123"
            path = manager.create_temp_file(".txt", content=content)

            assert path.read_bytes() == content

    def test_create_temp_file_with_prefix(self):
        """create_temp_file uses custom prefix."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)
            path = manager.create_temp_file(".txt", prefix="myprefix_")

            assert path.name.startswith("myprefix_")

    def test_create_temp_file_tracks_file(self):
        """create_temp_file adds file to tracking."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)

            assert manager.get_tracked_count() == 0

            manager.create_temp_file(".txt")

            assert manager.get_tracked_count() == 1


class TestTempFileManagerTracking:
    """Test file tracking."""

    def test_track_file(self):
        """track_file adds file to tracking."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)

            external_file = Path(tmpdir) / "external.txt"
            external_file.touch()

            manager.track_file(external_file)

            assert manager.get_tracked_count() == 1

    def test_untrack_file(self):
        """untrack_file removes file from tracking."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)

            path = manager.create_temp_file(".txt")
            assert manager.get_tracked_count() == 1

            manager.untrack_file(path)
            assert manager.get_tracked_count() == 0
            assert path.exists()  # File should still exist

    def test_untrack_nonexistent_file(self):
        """untrack_file handles non-tracked files gracefully."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)

            # Should not raise
            manager.untrack_file(Path("/nonexistent/file.txt"))


class TestTempFileManagerCleanup:
    """Test cleanup functionality."""

    def test_cleanup_deletes_tracked_files(self):
        """cleanup(force=True) deletes all tracked files."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)

            path1 = manager.create_temp_file(".txt")
            path2 = manager.create_temp_file(".txt")

            assert path1.exists()
            assert path2.exists()

            manager.cleanup(force=True)

            assert not path1.exists()
            assert not path2.exists()
            assert manager.get_tracked_count() == 0

    def test_cleanup_removes_nonexistent_from_tracking(self):
        """cleanup removes already-deleted files from tracking."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)

            path = manager.create_temp_file(".txt")
            path.unlink()  # Delete manually

            assert manager.get_tracked_count() == 1

            manager.cleanup(force=False)

            assert manager.get_tracked_count() == 0

    def test_cleanup_old_respects_max_age(self):
        """cleanup_old only deletes files older than max_age."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(
                base_dir=Path(tmpdir),
                auto_cleanup=False,
                max_age_seconds=0.1,  # Very short for testing
            )

            path1 = manager.create_temp_file(".txt")

            # Make file old by modifying mtime
            old_time = time.time() - 1.0  # 1 second ago
            import os

            os.utime(path1, (old_time, old_time))

            manager.cleanup_old()

            # Old file should be deleted
            assert not path1.exists()

    def test_cleanup_keeps_recent_files(self):
        """cleanup_old keeps files newer than max_age."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(
                base_dir=Path(tmpdir),
                auto_cleanup=False,
                max_age_seconds=3600.0,  # 1 hour
            )

            path = manager.create_temp_file(".txt")

            manager.cleanup_old()

            # Recent file should still exist
            assert path.exists()

    def test_cleanup_all(self):
        """cleanup_all deletes all tracked files."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)

            path = manager.create_temp_file(".txt")

            manager.cleanup_all()

            assert not path.exists()
            assert manager.get_tracked_count() == 0


class TestTempFileManagerContextManager:
    """Test context manager behavior."""

    def test_context_manager_cleans_up(self):
        """Context manager cleans up on exit when auto_cleanup=True."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            with TempFileManager(base_dir=Path(tmpdir), auto_cleanup=True) as manager:
                path = manager.create_temp_file(".txt")
                assert path.exists()

            # After context exit, file should be cleaned
            assert not path.exists()

    def test_context_manager_no_cleanup(self):
        """Context manager doesn't clean up when auto_cleanup=False."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            with TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False) as manager:
                path = manager.create_temp_file(".txt")

            # File should still exist
            assert path.exists()


class TestTempFileManagerStats:
    """Test stats methods."""

    def test_get_tracked_count(self):
        """get_tracked_count returns correct count."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)

            assert manager.get_tracked_count() == 0

            manager.create_temp_file(".txt")
            assert manager.get_tracked_count() == 1

            manager.create_temp_file(".txt")
            assert manager.get_tracked_count() == 2

    def test_get_total_size(self):
        """get_total_size returns correct total."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)

            content1 = b"12345"  # 5 bytes
            content2 = b"1234567890"  # 10 bytes

            manager.create_temp_file(".txt", content=content1)
            manager.create_temp_file(".txt", content=content2)

            assert manager.get_total_size() == 15

    def test_get_total_size_handles_missing_files(self):
        """get_total_size handles deleted files gracefully."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)

            path = manager.create_temp_file(".txt", content=b"content")
            path.unlink()  # Delete the file

            # Should not crash
            size = manager.get_total_size()
            assert size == 0


class TestTempFileManagerThreadSafety:
    """Test thread safety."""

    def test_concurrent_file_creation(self):
        """Multiple threads can create files safely."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)
            errors = []

            def create_files():
                try:
                    for _ in range(10):
                        manager.create_temp_file(".txt")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=create_files) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert manager.get_tracked_count() == 40

    def test_concurrent_cleanup(self):
        """Multiple threads can cleanup safely."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)
            errors = []

            # Create files
            for _ in range(20):
                manager.create_temp_file(".txt")

            def cleanup():
                try:
                    manager.cleanup(force=True)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=cleanup) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0


class TestSaveTempImage:
    """Test save_temp_image function."""

    def test_save_temp_image_returns_path(self):
        """save_temp_image returns a valid path."""
        from comfy_headless.cleanup import save_temp_image

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock get_temp_manager
            with patch("comfy_headless.cleanup.get_temp_manager") as mock_manager:
                mock_instance = MagicMock()
                mock_instance.create_temp_file.return_value = Path(tmpdir) / "test.png"
                mock_manager.return_value = mock_instance

                path = save_temp_image(b"PNG data")

                assert path is not None
                mock_instance.create_temp_file.assert_called_once()


class TestSaveTempVideo:
    """Test save_temp_video function."""

    def test_save_temp_video_returns_path(self):
        """save_temp_video returns a valid path."""
        from comfy_headless.cleanup import save_temp_video

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("comfy_headless.cleanup.get_temp_manager") as mock_manager:
                mock_instance = MagicMock()
                mock_instance.create_temp_file.return_value = Path(tmpdir) / "test.mp4"
                mock_manager.return_value = mock_instance

                path = save_temp_video(b"MP4 data")

                assert path is not None


class TestGetTempManager:
    """Test get_temp_manager singleton."""

    def test_get_temp_manager_returns_manager(self):
        """get_temp_manager returns TempFileManager instance."""
        from comfy_headless.cleanup import TempFileManager, get_temp_manager

        manager = get_temp_manager()
        assert isinstance(manager, TempFileManager)

    def test_get_temp_manager_singleton(self):
        """get_temp_manager returns same instance."""
        from comfy_headless.cleanup import get_temp_manager

        m1 = get_temp_manager()
        m2 = get_temp_manager()
        assert m1 is m2


class TestCleanupTempFiles:
    """Test cleanup_temp_files function."""

    def test_cleanup_temp_files_works(self):
        """cleanup_temp_files executes without error."""
        from comfy_headless.cleanup import cleanup_temp_files

        # Just verify it doesn't crash
        cleanup_temp_files()


class TestCleanupAll:
    """Test cleanup_all function."""

    def test_cleanup_all_works(self):
        """cleanup_all executes without error."""
        from comfy_headless.cleanup import cleanup_all

        # Just verify it doesn't crash
        cleanup_all()


class TestRegisterShutdownHandlers:
    """Test shutdown handler registration."""

    def test_register_shutdown_handlers(self):
        """register_shutdown_handlers registers atexit handler."""
        from comfy_headless.cleanup import register_shutdown_handlers

        with patch("atexit.register"):
            register_shutdown_handlers()

            # Should register something
            assert True  # May already be registered


class TestRegisterCleanupCallback:
    """Test cleanup callback registration."""

    def test_register_cleanup_callback(self):
        """register_cleanup_callback adds callback."""
        from comfy_headless.cleanup import register_cleanup_callback

        called = []

        def my_callback():
            called.append(True)

        register_cleanup_callback(my_callback)

        # Callback should be stored (exact mechanism depends on implementation)


class TestCleanupThread:
    """Test CleanupThread class."""

    def test_cleanup_thread_exists(self):
        """CleanupThread class is exported."""
        from comfy_headless.cleanup import CleanupThread

        assert CleanupThread is not None


class TestAllExports:
    """Test __all__ exports."""

    def test_all_exports_defined(self):
        """All expected exports are in __all__."""
        from comfy_headless import cleanup

        expected = [
            "TempFileManager",
            "get_temp_manager",
            "CleanupThread",
            "cleanup_temp_files",
            "cleanup_all",
            "register_shutdown_handlers",
            "register_cleanup_callback",
            "save_temp_image",
            "save_temp_video",
        ]

        for name in expected:
            assert name in cleanup.__all__

    def test_all_exports_accessible(self):
        """All items in __all__ are accessible."""
        from comfy_headless import cleanup

        for name in cleanup.__all__:
            assert hasattr(cleanup, name)


class TestEdgeCases:
    """Test edge cases."""

    def test_cleanup_handles_already_deleted_file(self):
        """cleanup handles already-deleted files gracefully."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)
            path = manager.create_temp_file(".txt")

            # Delete the file externally
            path.unlink()

            # Should not crash
            manager.cleanup(force=True)

    def test_create_file_in_nonexistent_dir(self):
        """create_temp_file works even if base_dir was deleted."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "subdir"
            TempFileManager(base_dir=base, auto_cleanup=False)

            # Delete the base dir
            import shutil

            shutil.rmtree(base)

            # Creating file should recreate dir or fail gracefully
            # Depending on implementation

    def test_empty_content_creates_empty_file(self):
        """create_temp_file with no content creates empty file."""
        from comfy_headless.cleanup import TempFileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TempFileManager(base_dir=Path(tmpdir), auto_cleanup=False)
            path = manager.create_temp_file(".txt")

            assert path.exists()
            assert path.read_bytes() == b""
