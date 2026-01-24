"""
Comfy Headless - Resource Cleanup
==================================

Handles cleanup of temporary files, connections, and other resources.

Usage:
    from comfy_headless.cleanup import (
        TempFileManager,
        cleanup_temp_files,
        register_shutdown_handlers,
    )

    # Use temp file manager
    with TempFileManager() as manager:
        path = manager.create_temp_file(".png", content)
        # File will be cleaned up when context exits

    # Or for Gradio UI - files persist but are tracked
    manager = TempFileManager(auto_cleanup=False)
    path = manager.create_temp_file(".png", content)
    # Later: manager.cleanup()
"""

import atexit
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .config import get_temp_dir, settings
from .logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    # Temp file manager
    "TempFileManager",
    "get_temp_manager",
    # Cleanup thread
    "CleanupThread",
    # Convenience functions
    "cleanup_temp_files",
    "cleanup_all",
    "register_shutdown_handlers",
    "register_cleanup_callback",
    "save_temp_image",
    "save_temp_video",
]


# =============================================================================
# TEMP FILE MANAGER
# =============================================================================


class TempFileManager:
    """
    Manages temporary files with automatic cleanup.

    Features:
    - Tracks all created temp files
    - Optional auto-cleanup on context exit
    - Age-based cleanup for old files
    - Thread-safe operations
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        auto_cleanup: bool = True,
        max_age_seconds: float = 3600.0,
    ):
        """
        Initialize the temp file manager.

        Args:
            base_dir: Base directory for temp files (default: system temp/comfy_headless)
            auto_cleanup: Whether to clean up files on context exit
            max_age_seconds: Maximum age for files during cleanup sweeps
        """
        self.base_dir = base_dir or get_temp_dir()
        self.auto_cleanup = auto_cleanup
        self.max_age_seconds = max_age_seconds

        self._files: set[Path] = set()
        self._lock = threading.Lock()

        # Ensure directory exists (with proper error handling)
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"Permission denied creating temp directory: {self.base_dir}")
            raise
        except OSError as e:
            logger.error(f"Failed to create temp directory {self.base_dir}: {e}")
            raise

        logger.debug(
            "TempFileManager initialized",
            extra={"base_dir": str(self.base_dir), "auto_cleanup": auto_cleanup},
        )

    def create_temp_file(
        self, suffix: str, content: bytes | None = None, prefix: str = "comfy_"
    ) -> Path:
        """
        Create a temporary file.

        Args:
            suffix: File extension (e.g., ".png")
            content: Optional content to write
            prefix: Filename prefix

        Returns:
            Path to the created file
        """
        import uuid

        filename = f"{prefix}{uuid.uuid4().hex[:8]}{suffix}"
        filepath = self.base_dir / filename

        if content:
            filepath.write_bytes(content)
        else:
            filepath.touch()

        with self._lock:
            self._files.add(filepath)

        logger.debug(f"Created temp file: {filepath.name}")
        return filepath

    def track_file(self, filepath: Path):
        """Track an externally created file for cleanup."""
        with self._lock:
            self._files.add(Path(filepath))

    def untrack_file(self, filepath: Path):
        """Remove a file from tracking (but don't delete it)."""
        with self._lock:
            self._files.discard(Path(filepath))

    def cleanup(self, force: bool = False):
        """
        Clean up tracked temp files.

        Args:
            force: If True, delete all tracked files regardless of age

        Note:
            Uses atomic operations to avoid TOCTOU race conditions.
            Files that disappear between check and delete are handled gracefully.
        """
        with self._lock:
            files_to_remove = set()

            for filepath in self._files:
                try:
                    if force:
                        # Attempt deletion directly - handles race condition
                        # where file is deleted by another process
                        try:
                            filepath.unlink()
                            files_to_remove.add(filepath)
                            logger.debug(f"Deleted temp file: {filepath.name}")
                        except FileNotFoundError:
                            # File already deleted - expected race condition
                            files_to_remove.add(filepath)
                            logger.debug(f"Temp file already deleted: {filepath.name}")
                    else:
                        # Check age - use try/except to handle race condition
                        try:
                            stat_info = filepath.stat()
                            age = time.time() - stat_info.st_mtime
                            if age > self.max_age_seconds:
                                try:
                                    filepath.unlink()
                                    files_to_remove.add(filepath)
                                    logger.debug(f"Deleted old temp file: {filepath.name}")
                                except FileNotFoundError:
                                    # Deleted between stat and unlink
                                    files_to_remove.add(filepath)
                        except FileNotFoundError:
                            # File doesn't exist anymore
                            files_to_remove.add(filepath)

                except PermissionError as e:
                    logger.warning(f"Permission denied deleting {filepath}: {e}")
                except OSError as e:
                    logger.error(f"OS error cleaning up {filepath}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error cleaning up {filepath}: {e}", exc_info=True)

            self._files -= files_to_remove

        if files_to_remove:
            logger.info(f"Cleaned up {len(files_to_remove)} temp files")

    def cleanup_all(self):
        """Clean up all tracked files regardless of age."""
        self.cleanup(force=True)

    def cleanup_old(self):
        """Clean up only old files based on max_age_seconds."""
        self.cleanup(force=False)

    def get_tracked_count(self) -> int:
        """Get the number of tracked files."""
        with self._lock:
            return len(self._files)

    def get_total_size(self) -> int:
        """Get total size of tracked files in bytes."""
        total = 0
        with self._lock:
            for filepath in self._files:
                try:
                    if filepath.exists():
                        total += filepath.stat().st_size
                except (OSError, PermissionError):
                    # File may have been deleted or be inaccessible - skip silently
                    pass
        return total

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_cleanup:
            self.cleanup_all()
        return False


# =============================================================================
# GLOBAL CLEANUP
# =============================================================================

_global_manager: TempFileManager | None = None
_cleanup_callbacks: list[Callable[[], None]] = []
_shutdown_registered = False


def get_temp_manager() -> TempFileManager:
    """Get the global temp file manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = TempFileManager(auto_cleanup=False)
    return _global_manager


def register_cleanup_callback(callback: Callable[[], None]):
    """Register a callback to be called during cleanup."""
    _cleanup_callbacks.append(callback)


def cleanup_temp_files(max_age_seconds: float | None = None):
    """
    Clean up old temp files in the comfy_headless temp directory.

    Args:
        max_age_seconds: Maximum file age (default from settings)

    Note:
        Handles TOCTOU race conditions gracefully - files that disappear
        between listing and deletion are simply skipped.
    """
    max_age = max_age_seconds or settings.ui.temp_cleanup_interval
    temp_dir = get_temp_dir()

    if not temp_dir.exists():
        return

    now = time.time()
    cleaned = 0

    for filepath in temp_dir.glob("*"):
        try:
            # Use try/except instead of is_file() check to handle race conditions
            try:
                stat_info = filepath.stat()
                # Check if it's a file (not directory) and old enough
                if stat_info.st_mode & 0o100000:  # S_IFREG - regular file
                    age = now - stat_info.st_mtime
                    if age > max_age:
                        try:
                            filepath.unlink()
                            cleaned += 1
                        except FileNotFoundError:
                            # File deleted between stat and unlink - that's fine
                            pass
            except FileNotFoundError:
                # File disappeared between glob and stat - that's fine
                pass
        except PermissionError as e:
            logger.debug(f"Permission denied accessing {filepath}: {e}")
        except OSError as e:
            logger.debug(f"OS error accessing {filepath}: {e}")
        except Exception as e:
            logger.debug(f"Failed to cleanup {filepath}: {e}")

    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} old temp files")


def cleanup_all():
    """Run all cleanup operations."""
    logger.info("Running full cleanup")

    # Clean global temp manager
    if _global_manager:
        _global_manager.cleanup_all()

    # Clean temp directory
    cleanup_temp_files(max_age_seconds=0)

    # Run registered callbacks
    for callback in _cleanup_callbacks:
        try:
            callback()
        except Exception as e:
            logger.warning(f"Cleanup callback failed: {e}")


def register_shutdown_handlers():
    """Register cleanup handlers for graceful shutdown."""
    global _shutdown_registered

    if _shutdown_registered:
        return

    def shutdown_handler(signum=None, frame=None):
        logger.info("Shutdown signal received, cleaning up...")
        cleanup_all()

    # Register atexit handler
    atexit.register(cleanup_all)

    # Register signal handlers (if possible)
    try:
        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)
    except (OSError, ValueError):
        # Signal handlers may not work in all environments (e.g., Windows threads)
        pass

    _shutdown_registered = True
    logger.debug("Shutdown handlers registered")


# =============================================================================
# BACKGROUND CLEANUP THREAD
# =============================================================================


class CleanupThread(threading.Thread):
    """
    Background thread for periodic cleanup.

    Usage:
        cleaner = CleanupThread(interval=3600)
        cleaner.start()
        # Later
        cleaner.stop()
    """

    def __init__(self, interval: float = 3600.0, max_file_age: float = 7200.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.max_file_age = max_file_age
        self._stop_event = threading.Event()

    def run(self):
        logger.info(f"Cleanup thread started (interval: {self.interval}s)")

        while not self._stop_event.wait(self.interval):
            try:
                cleanup_temp_files(max_age_seconds=self.max_file_age)
            except Exception as e:
                logger.error(f"Cleanup thread error: {e}")

    def stop(self):
        """Stop the cleanup thread."""
        self._stop_event.set()
        self.join(timeout=5)
        logger.info("Cleanup thread stopped")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR UI
# =============================================================================


def save_temp_image(content: bytes) -> str:
    """
    Save image content to a temp file and return the path.

    This is the main function the UI should use for saving generated images.
    """
    manager = get_temp_manager()
    path = manager.create_temp_file(".png", content)
    return str(path)


def save_temp_video(content: bytes, format: str = "mp4") -> str:
    """
    Save video content to a temp file and return the path.

    This is the main function the UI should use for saving generated videos.
    """
    manager = get_temp_manager()
    path = manager.create_temp_file(f".{format}", content)
    return str(path)
