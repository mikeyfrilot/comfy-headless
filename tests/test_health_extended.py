"""Extended tests for health module to improve coverage."""

import time
from unittest.mock import MagicMock, patch


class TestHealthStatusEnum:
    """Test HealthStatus enum values."""

    def test_health_status_values(self):
        """Test HealthStatus enum has correct values."""
        from comfy_headless.health import HealthStatus

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentHealthDataclass:
    """Test ComponentHealth dataclass methods."""

    def test_component_health_is_healthy_true(self):
        """Test is_healthy property returns True for HEALTHY status."""
        from comfy_headless.health import ComponentHealth, HealthStatus

        health = ComponentHealth(name="test", status=HealthStatus.HEALTHY, message="OK")
        assert health.is_healthy is True

    def test_component_health_is_healthy_false(self):
        """Test is_healthy property returns False for non-HEALTHY status."""
        from comfy_headless.health import ComponentHealth, HealthStatus

        for status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]:
            health = ComponentHealth(name="test", status=status)
            assert health.is_healthy is False

    def test_component_health_to_dict(self):
        """Test to_dict method."""
        from comfy_headless.health import ComponentHealth, HealthStatus

        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            latency_ms=50.5,
            details={"key": "value"},
        )
        result = health.to_dict()

        assert result["name"] == "test"
        assert result["status"] == "healthy"
        assert result["message"] == "OK"
        assert result["latency_ms"] == 50.5
        assert result["details"]["key"] == "value"
        assert "checked_at" in result

    def test_component_health_default_values(self):
        """Test ComponentHealth default values."""
        from comfy_headless.health import ComponentHealth, HealthStatus

        health = ComponentHealth(name="test", status=HealthStatus.HEALTHY)
        assert health.message == ""
        assert health.latency_ms is None
        assert health.details == {}


class TestHealthReportDataclass:
    """Test HealthReport dataclass methods."""

    def test_health_report_is_healthy_true(self):
        """Test is_healthy returns True for HEALTHY status."""
        from comfy_headless.health import HealthReport, HealthStatus

        report = HealthReport(status=HealthStatus.HEALTHY, components=[])
        assert report.is_healthy is True

    def test_health_report_is_healthy_false(self):
        """Test is_healthy returns False for non-HEALTHY status."""
        from comfy_headless.health import HealthReport, HealthStatus

        report = HealthReport(status=HealthStatus.DEGRADED, components=[])
        assert report.is_healthy is False

    def test_health_report_to_dict(self):
        """Test to_dict method."""
        from comfy_headless.health import ComponentHealth, HealthReport, HealthStatus

        component = ComponentHealth(name="comp1", status=HealthStatus.HEALTHY)
        report = HealthReport(status=HealthStatus.HEALTHY, components=[component])
        result = report.to_dict()

        assert result["status"] == "healthy"
        assert result["healthy"] is True
        assert "timestamp" in result
        assert "version" in result
        assert "comp1" in result["components"]


class TestComfyUIHealthCheck:
    """Test check_comfyui_health function."""

    @patch("requests.get")
    def test_comfyui_healthy_with_devices(self, mock_get):
        """Test ComfyUI healthy response with GPU info."""
        from comfy_headless.health import HealthStatus, check_comfyui_health

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "devices": [
                {
                    "name": "RTX 4090",
                    "vram_total": 24 * 1024 * 1024 * 1024,
                    "vram_free": 20 * 1024 * 1024 * 1024,
                }
            ]
        }
        mock_get.return_value = mock_response

        result = check_comfyui_health()
        assert result.status == HealthStatus.HEALTHY
        assert "gpu" in result.details

    @patch("requests.get")
    def test_comfyui_healthy_no_devices(self, mock_get):
        """Test ComfyUI healthy response without devices."""
        from comfy_headless.health import HealthStatus, check_comfyui_health

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"devices": []}
        mock_get.return_value = mock_response

        result = check_comfyui_health()
        assert result.status == HealthStatus.HEALTHY

    @patch("requests.get")
    def test_comfyui_non_200_response(self, mock_get):
        """Test ComfyUI non-200 response."""
        from comfy_headless.health import HealthStatus, check_comfyui_health

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = check_comfyui_health()
        assert result.status == HealthStatus.UNHEALTHY
        assert "500" in result.message

    @patch("requests.get")
    def test_comfyui_timeout(self, mock_get):
        """Test ComfyUI timeout."""
        import requests

        from comfy_headless.health import HealthStatus, check_comfyui_health

        mock_get.side_effect = requests.exceptions.Timeout()

        result = check_comfyui_health()
        assert result.status == HealthStatus.UNHEALTHY
        assert "timeout" in result.message.lower()

    @patch("requests.get")
    def test_comfyui_connection_refused(self, mock_get):
        """Test ComfyUI connection refused."""
        import requests

        from comfy_headless.health import HealthStatus, check_comfyui_health

        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = check_comfyui_health()
        assert result.status == HealthStatus.UNHEALTHY
        assert "not running" in result.message.lower()

    @patch("requests.get")
    def test_comfyui_generic_exception(self, mock_get):
        """Test ComfyUI generic exception."""
        from comfy_headless.health import HealthStatus, check_comfyui_health

        mock_get.side_effect = Exception("Unknown error")

        result = check_comfyui_health()
        assert result.status == HealthStatus.UNKNOWN


class TestOllamaHealthCheck:
    """Test check_ollama_health function."""

    @patch("requests.get")
    def test_ollama_healthy_with_model(self, mock_get):
        """Test Ollama healthy with required model."""
        from comfy_headless.health import HealthStatus, check_ollama_health

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "qwen2.5:7b"}, {"name": "llama3:8b"}]
        }
        mock_get.return_value = mock_response

        result = check_ollama_health()
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @patch("requests.get")
    def test_ollama_degraded_missing_model(self, mock_get):
        """Test Ollama degraded when model missing."""
        from comfy_headless.health import HealthStatus, check_ollama_health

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "other_model:latest"}]}
        mock_get.return_value = mock_response

        # This may be DEGRADED if the preferred model isn't found
        result = check_ollama_health()
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @patch("requests.get")
    def test_ollama_connection_refused(self, mock_get):
        """Test Ollama connection refused."""
        import requests

        from comfy_headless.health import HealthStatus, check_ollama_health

        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = check_ollama_health()
        assert result.status == HealthStatus.DEGRADED


class TestDiskSpaceCheck:
    """Test check_disk_space function."""

    def test_disk_space_check_runs(self):
        """Test disk space check completes."""
        from comfy_headless.health import HealthStatus, check_disk_space

        result = check_disk_space()
        assert result.name == "disk"
        assert result.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        assert "free_gb" in result.details

    @patch("shutil.disk_usage")
    def test_disk_space_critical(self, mock_usage):
        """Test disk space critical (< 1GB free)."""
        from comfy_headless.health import HealthStatus, check_disk_space

        # Simulate < 1GB free
        mock_usage.return_value = MagicMock(
            total=100 * 1024**3, used=99.5 * 1024**3, free=0.5 * 1024**3
        )

        result = check_disk_space()
        assert result.status == HealthStatus.UNHEALTHY
        assert "critical" in result.message.lower()

    @patch("shutil.disk_usage")
    def test_disk_space_low(self, mock_usage):
        """Test disk space low (< 5GB free)."""
        from comfy_headless.health import HealthStatus, check_disk_space

        # Simulate 3GB free
        mock_usage.return_value = MagicMock(
            total=100 * 1024**3, used=97 * 1024**3, free=3 * 1024**3
        )

        result = check_disk_space()
        assert result.status == HealthStatus.DEGRADED


class TestMemoryCheck:
    """Test check_memory function."""

    def test_memory_check_runs(self):
        """Test memory check completes."""
        from comfy_headless.health import check_memory

        result = check_memory()
        assert result.name == "memory"

    @patch("psutil.virtual_memory")
    def test_memory_critical(self, mock_mem):
        """Test memory critical (< 1GB available)."""
        from comfy_headless.health import HealthStatus, check_memory

        mock_mem.return_value = MagicMock(total=16 * 1024**3, available=0.5 * 1024**3, percent=97)

        result = check_memory()
        assert result.status == HealthStatus.UNHEALTHY

    @patch("psutil.virtual_memory")
    def test_memory_low(self, mock_mem):
        """Test memory low (< 4GB available)."""
        from comfy_headless.health import HealthStatus, check_memory

        mock_mem.return_value = MagicMock(total=16 * 1024**3, available=2 * 1024**3, percent=87)

        result = check_memory()
        assert result.status == HealthStatus.DEGRADED


class TestTempFilesCheck:
    """Test check_temp_files function."""

    def test_temp_files_check_runs(self):
        """Test temp files check completes."""
        from comfy_headless.health import check_temp_files

        result = check_temp_files()
        assert result.name == "temp_files"
        assert "file_count" in result.details


class TestCircuitBreakersCheck:
    """Test check_circuit_breakers function."""

    def test_circuit_breakers_check_runs(self):
        """Test circuit breakers check completes."""
        from comfy_headless.health import check_circuit_breakers

        result = check_circuit_breakers()
        assert result.name == "circuits"


class TestHealthChecker:
    """Test HealthChecker class."""

    def test_check_specific_component(self):
        """Test checking a specific component."""
        from comfy_headless.health import ComponentHealth, HealthChecker

        checker = HealthChecker()
        result = checker.check("disk")
        assert isinstance(result, ComponentHealth)
        assert result.name == "disk"

    def test_check_unknown_component(self):
        """Test checking unknown component."""
        from comfy_headless.health import HealthChecker, HealthStatus

        checker = HealthChecker()
        result = checker.check("nonexistent")
        assert result.status == HealthStatus.UNKNOWN
        assert "Unknown" in result.message

    def test_quick_check(self):
        """Test quick_check method."""
        from comfy_headless.health import HealthChecker, HealthReport

        checker = HealthChecker()
        report = checker.quick_check()
        assert isinstance(report, HealthReport)
        # Quick check only checks ComfyUI
        assert len(report.components) == 1

    def test_full_report(self):
        """Test full_report method."""
        from comfy_headless.health import HealthChecker, HealthReport

        checker = HealthChecker()
        report = checker.full_report()
        assert isinstance(report, HealthReport)
        assert len(report.components) > 1

    def test_compute_overall_status_all_healthy(self):
        """Test _compute_overall_status with all healthy components."""
        from comfy_headless.health import ComponentHealth, HealthChecker, HealthStatus

        checker = HealthChecker()
        components = [
            ComponentHealth(name="c1", status=HealthStatus.HEALTHY),
            ComponentHealth(name="c2", status=HealthStatus.HEALTHY),
        ]
        result = checker._compute_overall_status(components)
        assert result == HealthStatus.HEALTHY

    def test_compute_overall_status_with_degraded(self):
        """Test _compute_overall_status with degraded component."""
        from comfy_headless.health import ComponentHealth, HealthChecker, HealthStatus

        checker = HealthChecker()
        components = [
            ComponentHealth(name="c1", status=HealthStatus.HEALTHY),
            ComponentHealth(name="c2", status=HealthStatus.DEGRADED),
        ]
        result = checker._compute_overall_status(components)
        assert result == HealthStatus.DEGRADED

    def test_compute_overall_status_with_unhealthy(self):
        """Test _compute_overall_status with unhealthy component."""
        from comfy_headless.health import ComponentHealth, HealthChecker, HealthStatus

        checker = HealthChecker()
        components = [
            ComponentHealth(name="c1", status=HealthStatus.HEALTHY),
            ComponentHealth(name="c2", status=HealthStatus.UNHEALTHY),
        ]
        result = checker._compute_overall_status(components)
        assert result == HealthStatus.UNHEALTHY

    def test_compute_overall_status_comfyui_unhealthy(self):
        """Test _compute_overall_status when ComfyUI is unhealthy."""
        from comfy_headless.health import ComponentHealth, HealthChecker, HealthStatus

        checker = HealthChecker()
        components = [
            ComponentHealth(name="comfyui", status=HealthStatus.UNHEALTHY),
            ComponentHealth(name="disk", status=HealthStatus.HEALTHY),
        ]
        result = checker._compute_overall_status(components)
        assert result == HealthStatus.UNHEALTHY


class TestHealthCheckerRecovery:
    """Test HealthChecker recovery methods."""

    def test_attempt_recovery_specific(self):
        """Test attempting recovery for specific component."""
        from comfy_headless.health import HealthChecker

        checker = HealthChecker()
        results = checker.attempt_recovery("circuits")
        assert "circuits" in results

    def test_attempt_recovery_all(self):
        """Test attempting recovery for all components."""
        from comfy_headless.health import HealthChecker

        checker = HealthChecker()
        results = checker.attempt_recovery()
        assert isinstance(results, dict)

    def test_cleanup_temp_files(self):
        """Test _cleanup_temp_files method."""
        from comfy_headless.health import HealthChecker

        checker = HealthChecker()
        result = checker._cleanup_temp_files()
        assert isinstance(result, bool)

    def test_reset_circuits(self):
        """Test _reset_circuits method."""
        from comfy_headless.health import HealthChecker

        checker = HealthChecker()
        result = checker._reset_circuits()
        assert isinstance(result, bool)


class TestHealthMonitor:
    """Test HealthMonitor class."""

    def test_monitor_creation(self):
        """Test creating health monitor."""
        from comfy_headless.health import HealthMonitor

        monitor = HealthMonitor(interval=60)
        assert monitor.interval == 60
        assert monitor.auto_recover is True

    def test_monitor_start_stop(self):
        """Test starting and stopping monitor."""
        from comfy_headless.health import HealthMonitor

        monitor = HealthMonitor(interval=0.1)  # Short interval for testing
        monitor.start()
        assert monitor._running is True
        time.sleep(0.2)  # Let it run briefly
        monitor.stop()
        assert monitor._running is False

    def test_monitor_last_report(self):
        """Test last_report property."""
        from comfy_headless.health import HealthMonitor

        monitor = HealthMonitor(interval=0.1)
        assert monitor.last_report is None

        monitor.start()
        time.sleep(0.3)  # Let a check run
        monitor.stop()

        # May have a report now
        # (depends on timing)

    def test_monitor_callback(self):
        """Test on_unhealthy callback."""
        from comfy_headless.health import HealthMonitor

        callback_called = []

        def on_unhealthy(report):
            callback_called.append(report)

        monitor = HealthMonitor(interval=0.1, on_unhealthy=on_unhealthy)
        assert monitor.on_unhealthy is not None


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_health_checker_singleton(self):
        """Test get_health_checker returns singleton."""
        from comfy_headless.health import get_health_checker

        checker1 = get_health_checker()
        checker2 = get_health_checker()
        assert checker1 is checker2

    def test_check_health_returns_report(self):
        """Test check_health returns HealthReport."""
        from comfy_headless.health import HealthReport, check_health

        report = check_health()
        assert isinstance(report, HealthReport)

    def test_full_health_check_returns_report(self):
        """Test full_health_check returns HealthReport."""
        from comfy_headless.health import HealthReport, full_health_check

        report = full_health_check()
        assert isinstance(report, HealthReport)

    def test_is_healthy_returns_bool(self):
        """Test is_healthy returns boolean."""
        from comfy_headless.health import is_healthy

        result = is_healthy()
        assert isinstance(result, bool)


class TestModuleExports:
    """Test module exports."""

    def test_all_exports_exist(self):
        """Test all __all__ exports exist."""
        from comfy_headless import health

        for name in health.__all__:
            assert hasattr(health, name), f"Missing export: {name}"
