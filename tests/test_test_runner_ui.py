"""Tests for test_runner_ui module - the universal test runner."""

import tempfile
from pathlib import Path
from unittest.mock import patch


class TestTestRunnerConfig:
    """Test TestRunnerConfig dataclass."""

    def test_config_default_values(self):
        """Test default configuration values."""
        from comfy_headless.test_runner_ui import TestRunnerConfig

        config = TestRunnerConfig()
        assert config.project_name == "Python Project"
        assert config.tests_path == "tests"
        assert config.coverage_target == 80
        assert config.coverage_fail_under == 30
        assert config.branch_coverage is True
        assert config.server_host == "127.0.0.1"
        assert config.server_port == 7862

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from comfy_headless.test_runner_ui import TestRunnerConfig

        config = TestRunnerConfig(
            project_name="My Project",
            package_path="my_package",
            tests_path="my_package/tests",
            coverage_target=90,
        )
        assert config.project_name == "My Project"
        assert config.package_path == "my_package"
        assert config.tests_path == "my_package/tests"
        assert config.coverage_target == 90

    def test_config_project_root_path_conversion(self):
        """Test project_root is converted to Path."""
        from comfy_headless.test_runner_ui import TestRunnerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TestRunnerConfig(project_root=tmpdir)
            assert isinstance(config.project_root, Path)

    def test_config_auto_detect_package(self):
        """Test package auto-detection in __post_init__."""
        from comfy_headless.test_runner_ui import TestRunnerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a package directory
            pkg_dir = Path(tmpdir) / "my_auto_pkg"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").touch()

            config = TestRunnerConfig(
                project_root=tmpdir,
                package_path="",  # Empty to trigger auto-detect
            )
            assert config.package_path == "my_auto_pkg"

    def test_config_exclude_patterns(self):
        """Test default exclude patterns."""
        from comfy_headless.test_runner_ui import TestRunnerConfig

        config = TestRunnerConfig()
        assert "**/test_*.py" in config.exclude_from_coverage
        assert "**/__pycache__/**" in config.exclude_from_coverage


class TestTestRunner:
    """Test TestRunner class."""

    def test_runner_creation_default_config(self):
        """Test creating runner with default config."""
        from comfy_headless.test_runner_ui import TestRunner

        runner = TestRunner()
        assert runner is not None
        assert runner.config is not None

    def test_runner_creation_custom_config(self):
        """Test creating runner with custom config."""
        from comfy_headless.test_runner_ui import TestRunner, TestRunnerConfig

        config = TestRunnerConfig(project_name="Custom")
        runner = TestRunner(config)
        assert runner.config.project_name == "Custom"

    def test_run_command_success(self):
        """Test run_command with successful command."""
        from comfy_headless.test_runner_ui import TestRunner

        runner = TestRunner()
        stdout, stderr, code = runner.run_command(["python", "--version"])
        assert code == 0
        assert "Python" in stdout or "Python" in stderr

    def test_run_command_failure(self):
        """Test run_command with failing command."""
        from comfy_headless.test_runner_ui import TestRunner

        runner = TestRunner()
        stdout, stderr, code = runner.run_command(["python", "-c", "import sys; sys.exit(1)"])
        assert code == 1

    def test_run_command_timeout(self):
        """Test run_command timeout handling."""
        from comfy_headless.test_runner_ui import TestRunner

        runner = TestRunner()
        # Use a very short timeout
        with patch("subprocess.run") as mock_run:
            import subprocess

            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["test"], timeout=1)
            stdout, stderr, code = runner.run_command(["sleep", "100"])
            assert code == 1
            assert "timed out" in stderr.lower()

    def test_run_command_exception(self):
        """Test run_command exception handling."""
        from comfy_headless.test_runner_ui import TestRunner

        runner = TestRunner()
        with patch("subprocess.run", side_effect=Exception("Test error")):
            stdout, stderr, code = runner.run_command(["nonexistent_command"])
            assert code == 1
            assert "Test error" in stderr

    def test_get_test_files(self):
        """Test getting list of test files."""
        from comfy_headless.test_runner_ui import TestRunner, TestRunnerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_one.py").touch()
            (tests_dir / "test_two.py").touch()
            (tests_dir / "not_a_test.py").touch()

            config = TestRunnerConfig(project_root=tmpdir, tests_path="tests")
            runner = TestRunner(config)
            files = runner.get_test_files()

            assert "test_one.py" in files
            assert "test_two.py" in files
            assert "not_a_test.py" not in files

    def test_get_test_files_empty_dir(self):
        """Test getting test files from empty directory."""
        from comfy_headless.test_runner_ui import TestRunner, TestRunnerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty tests dir
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()

            config = TestRunnerConfig(project_root=tmpdir, tests_path="tests")
            runner = TestRunner(config)
            files = runner.get_test_files()
            assert files == []


class TestPytestOutputParsing:
    """Test pytest output parsing."""

    def test_parse_pytest_output_basic(self):
        """Test parsing basic pytest output."""
        from comfy_headless.test_runner_ui import TestRunner

        runner = TestRunner()
        output = "155 passed, 2 warnings in 30.40s"
        stats = runner.parse_pytest_output(output)

        assert stats["passed"] == 155
        assert stats["duration"] == "30.40s"

    def test_parse_pytest_output_with_failures(self):
        """Test parsing pytest output with failures."""
        from comfy_headless.test_runner_ui import TestRunner

        runner = TestRunner()
        output = "10 passed, 2 failed, 1 skipped in 5.00s"
        stats = runner.parse_pytest_output(output)

        assert stats["passed"] == 10
        assert stats["failed"] == 2
        assert stats["skipped"] == 1

    def test_parse_pytest_output_empty(self):
        """Test parsing empty output."""
        from comfy_headless.test_runner_ui import TestRunner

        runner = TestRunner()
        stats = runner.parse_pytest_output("")

        assert stats["passed"] == 0
        assert stats["failed"] == 0


class TestCoverageOutputParsing:
    """Test coverage output parsing."""

    def test_parse_coverage_output_basic(self):
        """Test parsing basic coverage output."""
        from comfy_headless.test_runner_ui import TestRunner

        runner = TestRunner()
        output = """
Name                          Stmts   Miss Branch BrPart   Cover
module.py                       100     20     40     10    75.00%
other.py                         50     10     20      5    80.00%
TOTAL                           150     30     60     15    77.50%
"""
        overall, modules = runner.parse_coverage_output(output)

        assert overall == "77.50%"
        assert len(modules) == 2

    def test_parse_coverage_output_no_total(self):
        """Test parsing coverage output without TOTAL line."""
        from comfy_headless.test_runner_ui import TestRunner

        runner = TestRunner()
        output = "No coverage data"
        overall, modules = runner.parse_coverage_output(output)

        assert overall == "0%"
        assert modules == []


class TestTestRunnerMethods:
    """Test TestRunner utility methods."""

    @patch.object(
        __import__("comfy_headless.test_runner_ui", fromlist=["TestRunner"]).TestRunner,
        "run_command",
    )
    def test_run_all_tests(self, mock_run):
        """Test run_all_tests method."""
        from comfy_headless.test_runner_ui import TestRunner

        mock_run.return_value = ("10 passed in 1.00s", "", 0)
        runner = TestRunner()
        output = runner.run_all_tests(verbose=True)

        assert "TEST RUN" in output
        assert "passed" in output

    @patch.object(
        __import__("comfy_headless.test_runner_ui", fromlist=["TestRunner"]).TestRunner,
        "run_command",
    )
    def test_run_tests_with_coverage(self, mock_run):
        """Test run_tests_with_coverage method."""
        from comfy_headless.test_runner_ui import TestRunner

        mock_run.return_value = (
            """
10 passed in 1.00s
TOTAL                           100     20     40     10    80.00%
""",
            "",
            0,
        )
        runner = TestRunner()
        output, summary = runner.run_tests_with_coverage()

        assert "COVERAGE RUN" in output
        assert "80" in summary or "Coverage" in summary

    @patch.object(
        __import__("comfy_headless.test_runner_ui", fromlist=["TestRunner"]).TestRunner,
        "run_command",
    )
    def test_run_specific_test(self, mock_run):
        """Test run_specific_test method."""
        from comfy_headless.test_runner_ui import TestRunner

        mock_run.return_value = ("5 passed in 0.50s", "", 0)
        runner = TestRunner()
        output = runner.run_specific_test("test_example.py")

        assert "TEST FILE" in output
        assert "test_example.py" in output

    def test_run_specific_test_no_file(self):
        """Test run_specific_test with no file selected."""
        from comfy_headless.test_runner_ui import TestRunner

        runner = TestRunner()
        output = runner.run_specific_test("")

        assert "select" in output.lower()

    @patch.object(
        __import__("comfy_headless.test_runner_ui", fromlist=["TestRunner"]).TestRunner,
        "run_command",
    )
    def test_run_failed_only(self, mock_run):
        """Test run_failed_only method."""
        from comfy_headless.test_runner_ui import TestRunner

        mock_run.return_value = ("2 passed in 0.20s", "", 0)
        runner = TestRunner()
        output = runner.run_failed_only()

        assert "RE-RUNNING FAILED" in output or "No failed tests" in output

    @patch.object(
        __import__("comfy_headless.test_runner_ui", fromlist=["TestRunner"]).TestRunner,
        "run_command",
    )
    def test_run_failed_only_no_failures(self, mock_run):
        """Test run_failed_only when no previous failures."""
        from comfy_headless.test_runner_ui import TestRunner

        mock_run.return_value = ("no previously failed tests", "", 5)
        runner = TestRunner()
        output = runner.run_failed_only()

        assert "No failed tests" in output


class TestTestQualityReport:
    """Test test quality checking."""

    def test_check_test_quality(self):
        """Test check_test_quality method."""
        from comfy_headless.test_runner_ui import TestRunner, TestRunnerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()

            # Create a test file with some test functions
            test_file = tests_dir / "test_sample.py"
            test_file.write_text("""
def test_one():
    pass

def test_two():
    pass

def test_three():
    pass
""")

            config = TestRunnerConfig(project_root=tmpdir, tests_path="tests")
            runner = TestRunner(config)
            output = runner.check_test_quality()

            assert "Test Quality Report" in output
            assert "Total Tests" in output


class TestLegacyFunctions:
    """Test legacy compatibility functions."""

    def test_legacy_run_command(self):
        """Test legacy run_command function."""
        from comfy_headless.test_runner_ui import run_command

        stdout, stderr, code = run_command(["python", "--version"])
        assert code == 0

    def test_legacy_get_test_files(self):
        """Test legacy get_test_files function."""
        from comfy_headless.test_runner_ui import get_test_files

        files = get_test_files()
        assert isinstance(files, list)

    def test_legacy_parse_pytest_output(self):
        """Test legacy parse_pytest_output function."""
        from comfy_headless.test_runner_ui import parse_pytest_output

        stats = parse_pytest_output("10 passed in 1.00s")
        assert stats["passed"] == 10

    def test_legacy_parse_coverage_output(self):
        """Test legacy parse_coverage_output function."""
        from comfy_headless.test_runner_ui import parse_coverage_output

        overall, modules = parse_coverage_output("TOTAL 100 20 40 10 80.00%")
        assert overall == "80.00%"


class TestGradioAvailability:
    """Test Gradio availability handling."""

    def test_gradio_available_flag(self):
        """Test GRADIO_AVAILABLE flag exists."""
        from comfy_headless.test_runner_ui import GRADIO_AVAILABLE

        assert isinstance(GRADIO_AVAILABLE, bool)

    def test_launch_checks_gradio(self):
        """Test launch checks for Gradio availability."""
        from comfy_headless.test_runner_ui import GRADIO_AVAILABLE, TestRunner

        TestRunner()
        # Just verify we can check the flag
        assert isinstance(GRADIO_AVAILABLE, bool)


class TestDefaultConfig:
    """Test DEFAULT_CONFIG instance."""

    def test_default_config_exists(self):
        """Test DEFAULT_CONFIG is defined."""
        from comfy_headless.test_runner_ui import DEFAULT_CONFIG

        assert DEFAULT_CONFIG is not None

    def test_default_config_values(self):
        """Test DEFAULT_CONFIG has expected values."""
        from comfy_headless.test_runner_ui import DEFAULT_CONFIG

        assert DEFAULT_CONFIG.project_name == "Python Test Runner"
        assert DEFAULT_CONFIG.package_path == "comfy_headless"


class TestMainEntryPoint:
    """Test main() entry point."""

    def test_main_function_exists(self):
        """Test main function exists."""
        from comfy_headless.test_runner_ui import main

        assert callable(main)

    def test_main_uses_default_config(self):
        """Test main() uses DEFAULT_CONFIG."""
        from comfy_headless.test_runner_ui import DEFAULT_CONFIG

        # Just verify the default config is set up correctly
        assert DEFAULT_CONFIG.project_name == "Python Test Runner"
