"""Tests for health module."""



class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_health_status_enum_exists(self):
        """Test HealthStatus enum exists."""
        from comfy_headless.health import HealthStatus

        assert HealthStatus.HEALTHY is not None
        assert HealthStatus.DEGRADED is not None
        assert HealthStatus.UNHEALTHY is not None


class TestComponentHealth:
    """Test ComponentHealth dataclass."""

    def test_component_health_creation(self):
        """Test creating ComponentHealth."""
        from comfy_headless.health import ComponentHealth, HealthStatus

        health = ComponentHealth(name="test", status=HealthStatus.HEALTHY, message="All good")

        assert health.name == "test"
        assert health.status == HealthStatus.HEALTHY


class TestHealthReport:
    """Test HealthReport dataclass."""

    def test_health_report_creation(self):
        """Test creating HealthReport."""
        from comfy_headless.health import HealthReport, HealthStatus

        report = HealthReport(status=HealthStatus.HEALTHY, components=[])

        assert report.status == HealthStatus.HEALTHY


class TestHealthCheckFunctions:
    """Test individual health check functions."""

    def test_check_comfyui_health(self):
        """Test check_comfyui_health function."""
        from comfy_headless.health import ComponentHealth, check_comfyui_health

        result = check_comfyui_health()
        assert isinstance(result, ComponentHealth)
        assert result.name == "comfyui"

    def test_check_ollama_health(self):
        """Test check_ollama_health function."""
        from comfy_headless.health import ComponentHealth, check_ollama_health

        result = check_ollama_health()
        assert isinstance(result, ComponentHealth)
        assert result.name == "ollama"

    def test_check_disk_space(self):
        """Test check_disk_space function."""
        from comfy_headless.health import ComponentHealth, check_disk_space

        result = check_disk_space()
        assert isinstance(result, ComponentHealth)
        assert result.name == "disk"

    def test_check_memory(self):
        """Test check_memory function."""
        from comfy_headless.health import ComponentHealth, check_memory

        result = check_memory()
        assert isinstance(result, ComponentHealth)
        assert result.name == "memory"

    def test_check_temp_files(self):
        """Test check_temp_files function."""
        from comfy_headless.health import ComponentHealth, check_temp_files

        result = check_temp_files()
        assert isinstance(result, ComponentHealth)
        assert result.name == "temp_files"

    def test_check_circuit_breakers(self):
        """Test check_circuit_breakers function."""
        from comfy_headless.health import ComponentHealth, check_circuit_breakers

        result = check_circuit_breakers()
        assert isinstance(result, ComponentHealth)
        assert result.name == "circuits"


class TestHealthChecker:
    """Test HealthChecker class."""

    def test_health_checker_exists(self):
        """Test HealthChecker class exists."""
        from comfy_headless.health import HealthChecker

        checker = HealthChecker()
        assert checker is not None

    def test_health_checker_full_report(self):
        """Test full_report method."""
        from comfy_headless.health import HealthChecker, HealthReport

        checker = HealthChecker()
        report = checker.full_report()

        assert isinstance(report, HealthReport)


class TestHealthHelperFunctions:
    """Test health helper functions."""

    def test_get_health_checker(self):
        """Test get_health_checker returns singleton."""
        from comfy_headless.health import HealthChecker, get_health_checker

        checker = get_health_checker()
        assert isinstance(checker, HealthChecker)

    def test_check_health(self):
        """Test check_health convenience function."""
        from comfy_headless.health import HealthReport, check_health

        report = check_health()
        assert isinstance(report, HealthReport)

    def test_full_health_check(self):
        """Test full_health_check function."""
        from comfy_headless.health import HealthReport, full_health_check

        report = full_health_check()
        assert isinstance(report, HealthReport)

    def test_is_healthy(self):
        """Test is_healthy convenience function."""
        from comfy_headless.health import is_healthy

        result = is_healthy()
        assert isinstance(result, bool)
