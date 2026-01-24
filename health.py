"""
Comfy Headless - Health Checks and Self-Healing
=================================================

Provides health monitoring and automatic recovery capabilities.

Usage:
    from comfy_headless.health import HealthChecker, check_health

    # Quick check
    status = check_health()
    print(status.is_healthy)

    # Detailed health checker
    checker = HealthChecker()
    report = checker.full_report()
"""

import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import requests

from .config import get_temp_dir, settings
from .logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    # Enums
    "HealthStatus",
    # Data classes
    "ComponentHealth",
    "HealthReport",
    # Individual checks
    "check_comfyui_health",
    "check_ollama_health",
    "check_disk_space",
    "check_memory",
    "check_temp_files",
    "check_circuit_breakers",
    # Health checker
    "HealthChecker",
    "get_health_checker",
    # Health monitor
    "HealthMonitor",
    # Convenience functions
    "check_health",
    "full_health_check",
    "is_healthy",
]


# =============================================================================
# HEALTH STATUS TYPES
# =============================================================================


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class HealthReport:
    """Overall health report."""

    status: HealthStatus
    components: list[ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = field(default_factory=lambda: settings.version)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "healthy": self.is_healthy,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "components": {c.name: c.to_dict() for c in self.components},
        }


# =============================================================================
# INDIVIDUAL HEALTH CHECKS
# =============================================================================


def check_comfyui_health() -> ComponentHealth:
    """Check ComfyUI connectivity."""
    start = time.perf_counter()
    try:
        url = settings.comfyui.url
        response = requests.get(f"{url}/system_stats", timeout=settings.comfyui.timeout_connect)
        latency = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            devices = data.get("devices", [])

            # Check VRAM if available
            details = {}
            if devices:
                gpu = devices[0]
                vram_total = gpu.get("vram_total", 0)
                vram_free = gpu.get("vram_free", 0)
                if vram_total > 0:
                    vram_used_pct = (vram_total - vram_free) / vram_total * 100
                    details["gpu"] = gpu.get("name", "Unknown")
                    details["vram_used_pct"] = round(vram_used_pct, 1)

            return ComponentHealth(
                name="comfyui",
                status=HealthStatus.HEALTHY,
                message="Connected",
                latency_ms=latency,
                details=details,
            )
        else:
            return ComponentHealth(
                name="comfyui",
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP {response.status_code}",
                latency_ms=latency,
            )

    except requests.exceptions.Timeout:
        return ComponentHealth(
            name="comfyui", status=HealthStatus.UNHEALTHY, message="Connection timeout"
        )
    except requests.exceptions.ConnectionError:
        return ComponentHealth(
            name="comfyui",
            status=HealthStatus.UNHEALTHY,
            message="Connection refused - ComfyUI not running",
        )
    except Exception as e:
        return ComponentHealth(
            name="comfyui", status=HealthStatus.UNKNOWN, message=f"Check failed: {e}"
        )


def check_ollama_health() -> ComponentHealth:
    """Check Ollama connectivity."""
    start = time.perf_counter()
    try:
        url = settings.ollama.url
        response = requests.get(f"{url}/api/tags", timeout=settings.ollama.timeout_connect)
        latency = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check if our preferred model is available (exact match or base name match)
            # Model names can be like "qwen2.5:7b" or "qwen2.5:7b-instruct"
            preferred = settings.ollama.model
            # Exact match first, then base name match (before the colon variation)
            has_model = any(
                name == preferred  # Exact match
                or name.startswith(f"{preferred}:")  # Preferred is base, name has tag
                or preferred.startswith(f"{name}:")  # Name is base, preferred has tag
                for name in model_names
            )
            # Also check if any model starts with the same base (e.g., qwen2.5)
            if not has_model:
                base_preferred = preferred.split(":")[0]
                has_model = any(name.split(":")[0] == base_preferred for name in model_names)

            return ComponentHealth(
                name="ollama",
                status=HealthStatus.HEALTHY if has_model else HealthStatus.DEGRADED,
                message="Connected" if has_model else f"Model {preferred} not found",
                latency_ms=latency,
                details={"models": model_names[:5], "model_available": has_model},
            )
        else:
            return ComponentHealth(
                name="ollama",
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP {response.status_code}",
                latency_ms=latency,
            )

    except requests.exceptions.Timeout:
        return ComponentHealth(
            name="ollama",
            status=HealthStatus.DEGRADED,
            message="Connection timeout - AI features unavailable",
        )
    except requests.exceptions.ConnectionError:
        return ComponentHealth(
            name="ollama",
            status=HealthStatus.DEGRADED,
            message="Not running - AI features unavailable",
        )
    except Exception as e:
        return ComponentHealth(
            name="ollama", status=HealthStatus.UNKNOWN, message=f"Check failed: {e}"
        )


def check_disk_space() -> ComponentHealth:
    """Check available disk space."""
    try:
        temp_dir = get_temp_dir()
        usage = shutil.disk_usage(temp_dir)
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        used_pct = (usage.used / usage.total) * 100

        if free_gb < 1:
            status = HealthStatus.UNHEALTHY
            message = f"Critical: Only {free_gb:.1f}GB free"
        elif free_gb < 5:
            status = HealthStatus.DEGRADED
            message = f"Low: {free_gb:.1f}GB free"
        else:
            status = HealthStatus.HEALTHY
            message = f"{free_gb:.1f}GB free"

        return ComponentHealth(
            name="disk",
            status=status,
            message=message,
            details={
                "free_gb": round(free_gb, 1),
                "total_gb": round(total_gb, 1),
                "used_pct": round(used_pct, 1),
                "path": str(temp_dir),
            },
        )

    except Exception as e:
        return ComponentHealth(
            name="disk", status=HealthStatus.UNKNOWN, message=f"Check failed: {e}"
        )


def check_memory() -> ComponentHealth:
    """Check available memory."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        used_pct = mem.percent

        if available_gb < 1:
            status = HealthStatus.UNHEALTHY
            message = f"Critical: Only {available_gb:.1f}GB available"
        elif available_gb < 4:
            status = HealthStatus.DEGRADED
            message = f"Low: {available_gb:.1f}GB available"
        else:
            status = HealthStatus.HEALTHY
            message = f"{available_gb:.1f}GB available"

        return ComponentHealth(
            name="memory",
            status=status,
            message=message,
            details={
                "available_gb": round(available_gb, 1),
                "total_gb": round(total_gb, 1),
                "used_pct": round(used_pct, 1),
            },
        )

    except ImportError:
        return ComponentHealth(
            name="memory", status=HealthStatus.UNKNOWN, message="psutil not installed"
        )
    except Exception as e:
        return ComponentHealth(
            name="memory", status=HealthStatus.UNKNOWN, message=f"Check failed: {e}"
        )


def check_temp_files() -> ComponentHealth:
    """Check temp file usage and cleanup status."""
    try:
        temp_dir = get_temp_dir()
        files = list(temp_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        size_mb = total_size / (1024**2)

        if size_mb > 1000:
            status = HealthStatus.DEGRADED
            message = f"{len(files)} files, {size_mb:.0f}MB - cleanup recommended"
        else:
            status = HealthStatus.HEALTHY
            message = f"{len(files)} files, {size_mb:.0f}MB"

        return ComponentHealth(
            name="temp_files",
            status=status,
            message=message,
            details={
                "file_count": len(files),
                "size_mb": round(size_mb, 1),
                "path": str(temp_dir),
            },
        )

    except Exception as e:
        return ComponentHealth(
            name="temp_files", status=HealthStatus.UNKNOWN, message=f"Check failed: {e}"
        )


def check_circuit_breakers() -> ComponentHealth:
    """Check circuit breaker status."""
    try:
        from .retry import circuit_registry

        status_map = circuit_registry.status()

        open_circuits = [name for name, s in status_map.items() if s["state"] == "open"]

        if open_circuits:
            status = HealthStatus.DEGRADED
            message = f"Open circuits: {', '.join(open_circuits)}"
        else:
            status = HealthStatus.HEALTHY
            message = "All circuits closed"

        return ComponentHealth(name="circuits", status=status, message=message, details=status_map)

    except Exception as e:
        return ComponentHealth(
            name="circuits", status=HealthStatus.UNKNOWN, message=f"Check failed: {e}"
        )


# =============================================================================
# HEALTH CHECKER CLASS
# =============================================================================


class HealthChecker:
    """
    Comprehensive health checker with optional auto-recovery.

    Usage:
        checker = HealthChecker()
        report = checker.full_report()

        if not report.is_healthy:
            checker.attempt_recovery()
    """

    def __init__(self):
        self._checks: dict[str, Callable[[], ComponentHealth]] = {
            "comfyui": check_comfyui_health,
            "ollama": check_ollama_health,
            "disk": check_disk_space,
            "memory": check_memory,
            "temp_files": check_temp_files,
            "circuits": check_circuit_breakers,
        }
        self._recovery_actions: dict[str, Callable[[], bool]] = {
            "temp_files": self._cleanup_temp_files,
            "circuits": self._reset_circuits,
        }

    def check(self, component: str) -> ComponentHealth:
        """Check a single component."""
        if component not in self._checks:
            return ComponentHealth(
                name=component,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown component: {component}",
            )
        return self._checks[component]()

    def quick_check(self) -> HealthReport:
        """Quick health check (ComfyUI only)."""
        components = [check_comfyui_health()]
        overall = self._compute_overall_status(components)
        return HealthReport(status=overall, components=components)

    def full_report(self) -> HealthReport:
        """Full health check of all components."""
        components = [check() for check in self._checks.values()]
        overall = self._compute_overall_status(components)
        return HealthReport(status=overall, components=components)

    def _compute_overall_status(self, components: list[ComponentHealth]) -> HealthStatus:
        """Compute overall status from component statuses."""
        statuses = [c.status for c in components]

        # Critical components
        comfyui = next((c for c in components if c.name == "comfyui"), None)
        if comfyui and comfyui.status == HealthStatus.UNHEALTHY:
            return HealthStatus.UNHEALTHY

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if HealthStatus.UNKNOWN in statuses:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def attempt_recovery(self, component: str | None = None) -> dict[str, bool]:
        """
        Attempt to recover unhealthy components.

        Args:
            component: Specific component to recover, or None for all

        Returns:
            Dict mapping component names to recovery success
        """
        results = {}

        if component:
            if component in self._recovery_actions:
                try:
                    results[component] = self._recovery_actions[component]()
                except Exception as e:
                    logger.error(f"Recovery failed for {component}: {e}")
                    results[component] = False
        else:
            for name, action in self._recovery_actions.items():
                try:
                    results[name] = action()
                except Exception as e:
                    logger.error(f"Recovery failed for {name}: {e}")
                    results[name] = False

        return results

    def _cleanup_temp_files(self) -> bool:
        """Clean up temp files."""
        try:
            temp_dir = get_temp_dir()
            now = time.time()
            cleaned = 0

            for f in temp_dir.glob("*"):
                if f.is_file():
                    # Delete files older than 1 hour
                    age = now - f.stat().st_mtime
                    if age > 3600:
                        f.unlink()
                        cleaned += 1

            logger.info(f"Cleaned up {cleaned} temp files")
            return True

        except Exception as e:
            logger.error(f"Temp file cleanup failed: {e}")
            return False

    def _reset_circuits(self) -> bool:
        """Reset all circuit breakers."""
        try:
            from .retry import circuit_registry

            circuit_registry.reset_all()
            logger.info("Reset all circuit breakers")
            return True
        except Exception as e:
            logger.error(f"Circuit reset failed: {e}")
            return False


# =============================================================================
# BACKGROUND HEALTH MONITOR
# =============================================================================


class HealthMonitor:
    """
    Background health monitoring with automatic recovery.

    Usage:
        monitor = HealthMonitor(interval=60)
        monitor.start()

        # Later
        monitor.stop()
    """

    def __init__(
        self,
        interval: float = 60.0,
        auto_recover: bool = True,
        on_unhealthy: Callable[[HealthReport], None] | None = None,
    ):
        self.interval = interval
        self.auto_recover = auto_recover
        self.on_unhealthy = on_unhealthy

        self._checker = HealthChecker()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_report: HealthReport | None = None

    @property
    def last_report(self) -> HealthReport | None:
        return self._last_report

    def start(self):
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Health monitor started (interval: {self.interval}s)")

    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Health monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                report = self._checker.full_report()
                self._last_report = report

                if not report.is_healthy:
                    logger.warning(
                        f"Health check: {report.status.value}", extra={"report": report.to_dict()}
                    )

                    if self.on_unhealthy:
                        self.on_unhealthy(report)

                    if self.auto_recover:
                        self._checker.attempt_recovery()

                else:
                    logger.debug("Health check: healthy")

            except Exception as e:
                logger.error(f"Health monitor error: {e}")

            time.sleep(self.interval)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get the singleton health checker."""
    global _checker
    if _checker is None:
        _checker = HealthChecker()
    return _checker


def check_health() -> HealthReport:
    """Quick health check."""
    return get_health_checker().quick_check()


def full_health_check() -> HealthReport:
    """Full health check of all components."""
    return get_health_checker().full_report()


def is_healthy() -> bool:
    """Simple boolean health check."""
    return check_health().is_healthy
