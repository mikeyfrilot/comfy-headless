"""
Universal Python Test Runner - Gradio Interface

A configurable visual interface for running tests, viewing coverage, and monitoring code quality.
Can be used with ANY Python project by configuring the settings.

Launch with: python -m comfy_headless.test_runner_ui
Or import and use: from comfy_headless.test_runner_ui import TestRunner
"""

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TestRunnerConfig:
    """Configuration for the test runner.

    Example usage:
        config = TestRunnerConfig(
            project_name="My Project",
            package_path="my_package",
            tests_path="my_package/tests",
            coverage_target=80,
        )
        runner = TestRunner(config)
        runner.launch()
    """

    # Project identification
    project_name: str = "Python Project"

    # Paths (relative to project root)
    project_root: Path = field(default_factory=lambda: Path.cwd())
    package_path: str = ""  # e.g., "my_package" - for coverage
    tests_path: str = "tests"  # e.g., "my_package/tests"

    # Coverage settings
    coverage_target: int = 80  # Target percentage
    coverage_fail_under: int = 30  # Minimum required
    branch_coverage: bool = True  # Enable branch coverage

    # UI settings
    server_host: str = "127.0.0.1"
    server_port: int = 7862

    # Custom patterns for parsing (regex)
    test_file_pattern: str = "test_*.py"
    module_coverage_pattern: str = r"(\S+\.py)\s+\d+\s+\d+\s+\d+\s+\d+\s+([\d.]+%)"

    # Pytest options
    pytest_args: list[str] = field(default_factory=list)

    # Exclusions
    exclude_from_coverage: list[str] = field(
        default_factory=lambda: ["**/test_*.py", "**/__pycache__/**", "**/conftest.py"]
    )

    def __post_init__(self):
        self.project_root = Path(self.project_root)
        if not self.package_path:
            # Try to auto-detect package
            for item in self.project_root.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    self.package_path = item.name
                    break


# Default configuration for comfy_headless (backward compatibility)
DEFAULT_CONFIG = TestRunnerConfig(
    project_name="Python Test Runner",
    package_path="comfy_headless",
    tests_path="comfy_headless/tests",
    project_root=Path(__file__).parent.parent,
    coverage_target=70,
    coverage_fail_under=30,
)


# =============================================================================
# TEST RUNNER CLASS
# =============================================================================


class TestRunner:
    """Universal test runner with Gradio UI."""

    def __init__(self, config: TestRunnerConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.project_root = self.config.project_root
        self.tests_dir = self.project_root / self.config.tests_path

    def run_command(self, cmd: list[str], cwd: Path | None = None) -> tuple[str, str, int]:
        """Run a command and return stdout, stderr, return code."""
        try:
            result = subprocess.run(
                cmd, cwd=cwd or self.project_root, capture_output=True, text=True, timeout=600
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out after 10 minutes", 1
        except Exception as e:
            return "", str(e), 1

    def get_test_files(self) -> list[str]:
        """Get list of test files."""
        test_files = []
        if self.tests_dir.exists():
            for f in self.tests_dir.glob(self.config.test_file_pattern):
                test_files.append(f.name)
        return sorted(test_files)

    def parse_pytest_output(self, output: str) -> dict:
        """Parse pytest output for summary stats."""
        stats = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "warnings": 0,
            "duration": "0.00s",
        }

        # Match patterns like "155 passed, 2 warnings in 30.40s"
        summary_match = re.search(r"(\d+)\s+passed.*?in\s+([\d.]+s)", output)
        if summary_match:
            stats["passed"] = int(summary_match.group(1))
            stats["duration"] = summary_match.group(2)

        for key in ["failed", "skipped", "error", "warning"]:
            match = re.search(rf"(\d+)\s+{key}", output)
            if match:
                target_key = "errors" if key == "error" else key + "s" if key == "warning" else key
                stats[target_key] = int(match.group(1))

        return stats

    def parse_coverage_output(self, output: str) -> tuple[str, list[dict]]:
        """Parse coverage output for stats."""
        overall = "0%"
        modules = []

        # Find overall coverage
        total_match = re.search(r"TOTAL\s+\d+\s+\d+\s+\d+\s+\d+\s+([\d.]+%)", output)
        if total_match:
            overall = total_match.group(1)

        # Parse individual module coverage
        for line in output.split("\n"):
            match = re.match(self.config.module_coverage_pattern, line)
            if match:
                modules.append({"module": match.group(1), "coverage": match.group(2)})

        return overall, modules

    def run_all_tests(self, verbose: bool = True) -> str:
        """Run all tests."""
        cmd = [sys.executable, "-m", "pytest", self.config.tests_path]
        if verbose:
            cmd.append("-v")
        cmd.extend(self.config.pytest_args)

        stdout, stderr, code = self.run_command(cmd)

        output = f"{'=' * 60}\n"
        output += f"TEST RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"{'=' * 60}\n\n"

        if stdout:
            output += stdout
        if stderr and "warning" not in stderr.lower():
            output += f"\n\nSTDERR:\n{stderr}"

        stats = self.parse_pytest_output(stdout)

        output += f"\n\n{'=' * 60}\n"
        output += f"SUMMARY: {stats['passed']} passed, {stats['failed']} failed, "
        output += f"{stats['skipped']} skipped in {stats['duration']}\n"

        return output

    def run_tests_with_coverage(self) -> tuple[str, str]:
        """Run tests with coverage report."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            self.config.tests_path,
            f"--cov={self.config.package_path}",
            "--cov-report=term-missing",
            "-v",
        ]

        if self.config.branch_coverage:
            cmd.append("--cov-branch")

        cmd.extend(self.config.pytest_args)

        stdout, stderr, code = self.run_command(cmd)

        output = f"{'=' * 60}\n"
        output += f"COVERAGE RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"{'=' * 60}\n\n"
        output += stdout

        overall, modules = self.parse_coverage_output(stdout)
        overall_pct = float(overall.rstrip("%")) if overall != "0%" else 0

        # Create coverage summary
        target = self.config.coverage_target
        status = (
            "✅"
            if overall_pct >= target
            else "⚠️"
            if overall_pct >= self.config.coverage_fail_under
            else "❌"
        )

        summary = f"## {status} Overall Coverage: {overall}\n"
        summary += f"*Target: {target}% | Minimum: {self.config.coverage_fail_under}%*\n\n"

        # Coverage gaps section
        summary += "### Coverage Gaps (Lowest First)\n"
        summary += "| Module | Coverage | Status |\n"
        summary += "|--------|----------|--------|\n"

        # Sort by coverage (lowest first)
        modules_sorted = sorted(modules, key=lambda x: float(x["coverage"].rstrip("%")))

        for mod in modules_sorted[:15]:  # Show bottom 15
            cov = float(mod["coverage"].rstrip("%"))
            emoji = "✅" if cov >= 80 else "⚠️" if cov >= 50 else "❌"
            summary += f"| {mod['module']} | {mod['coverage']} | {emoji} |\n"

        return output, summary

    def run_specific_test(self, test_file: str, verbose: bool = True) -> str:
        """Run a specific test file."""
        if not test_file:
            return "Please select a test file."

        test_path = f"{self.config.tests_path}/{test_file}"
        cmd = [sys.executable, "-m", "pytest", test_path]
        if verbose:
            cmd.append("-v")

        stdout, stderr, code = self.run_command(cmd)

        output = f"{'=' * 60}\n"
        output += f"TEST FILE: {test_file}\n"
        output += f"{'=' * 60}\n\n"
        output += stdout

        if stderr and "warning" not in stderr.lower():
            output += f"\n\nSTDERR:\n{stderr}"

        return output

    def run_failed_only(self) -> str:
        """Re-run only failed tests."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            self.config.tests_path,
            "--lf",  # Last failed
            "-v",
        ]

        stdout, stderr, code = self.run_command(cmd)

        if "no previously failed tests" in stdout.lower() or code == 5:
            return "✅ No failed tests to re-run!"

        output = f"{'=' * 60}\n"
        output += "RE-RUNNING FAILED TESTS\n"
        output += f"{'=' * 60}\n\n"
        output += stdout

        return output

    def generate_html_report(self) -> str:
        """Generate HTML coverage report."""
        html_dir = self.project_root / self.config.package_path / "htmlcov"

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            self.config.tests_path,
            f"--cov={self.config.package_path}",
            f"--cov-report=html:{html_dir}",
            "-q",
        ]

        stdout, stderr, code = self.run_command(cmd)

        html_path = html_dir / "index.html"
        if html_path.exists():
            return f"✅ HTML report generated!\n\nOpen in browser:\nfile://{html_path}"
        else:
            return "❌ Failed to generate HTML report"

    def get_coverage_gaps(self) -> str:
        """Analyze coverage gaps in detail."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            self.config.tests_path,
            f"--cov={self.config.package_path}",
            "--cov-report=term-missing",
            "--cov-branch",
            "-q",
        ]

        stdout, stderr, code = self.run_command(cmd)

        output = "## Coverage Gap Analysis\n\n"

        # Parse missing lines
        gaps = []
        for line in stdout.split("\n"):
            match = re.match(r"(\S+\.py)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+%)\s+(.+)", line)
            if match:
                module = match.group(1)
                stmts = int(match.group(2))
                miss = int(match.group(3))
                branch = int(match.group(4))
                br_miss = int(match.group(5))
                cover = match.group(6)
                missing = match.group(7)

                if miss > 0 or br_miss > 0:
                    gaps.append(
                        {
                            "module": module,
                            "statements": stmts,
                            "missing": miss,
                            "branches": branch,
                            "branch_miss": br_miss,
                            "coverage": cover,
                            "lines": missing,
                        }
                    )

        # Sort by missing statements
        gaps.sort(key=lambda x: x["missing"], reverse=True)

        output += "### Modules Needing Tests (sorted by gap size)\n\n"

        for gap in gaps[:10]:
            pct = float(gap["coverage"].rstrip("%"))
            priority = "🔴 HIGH" if pct < 50 else "🟡 MED" if pct < 70 else "🟢 LOW"

            output += f"#### {gap['module']} - {gap['coverage']} ({priority})\n"
            output += f"- Missing: {gap['missing']} statements, {gap['branch_miss']} branches\n"
            output += f"- Lines needing tests: `{gap['lines'][:100]}{'...' if len(gap['lines']) > 100 else ''}`\n\n"

        return output

    def check_test_quality(self) -> str:
        """Check test quality metrics."""
        output = ""

        # Count test functions
        total_tests = 0
        test_counts = {}

        for test_file in self.get_test_files():
            file_path = self.tests_dir / test_file
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Count test functions
                tests = len(re.findall(r"\n\s+def test_", content))
                total_tests += tests
                test_counts[test_file] = tests
            except Exception:
                continue

        output += "## Test Quality Report\n\n"
        output += f"**Total Tests:** {total_tests}\n"
        output += f"**Test Files:** {len(test_counts)}\n\n"

        output += "### Tests per File\n"
        output += "| File | Tests |\n"
        output += "|------|-------|\n"

        for file, count in sorted(test_counts.items(), key=lambda x: -x[1]):
            output += f"| {file} | {count} |\n"

        # Check for test patterns
        output += "\n### Test Patterns Detected\n"

        patterns = {
            "test_property_based.py": "✅ Property-based testing (Hypothesis)",
            "test_integration.py": "✅ Integration tests",
            "test_error_paths.py": "✅ Error path testing",
            "test_edge_cases.py": "✅ Edge case testing",
            "conftest.py": "✅ Fixtures configured",
        }

        for filename, desc in patterns.items():
            if (self.tests_dir / filename).exists():
                output += f"- {desc}\n"

        return output

    def create_interface(self):
        """Create the Gradio interface."""

        with gr.Blocks(title="Test Runner", theme=gr.themes.Soft()) as app:
            gr.Markdown(f"""
            ## 🧪 Test Runner

            *Testing: `{self.config.package_path}` | Target: {self.config.coverage_target}% coverage*
            """)

            with gr.Tabs():
                # Tab 1: Run All Tests
                with gr.Tab("▶️ Run Tests"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            run_all_btn = gr.Button("Run All Tests", variant="primary", size="lg")
                            run_failed_btn = gr.Button("Re-run Failed", variant="secondary")
                            verbose_check = gr.Checkbox(label="Verbose output", value=True)

                        with gr.Column(scale=3):
                            test_output = gr.Textbox(
                                label="Test Output", lines=25, max_lines=50, show_copy_button=True
                            )

                    run_all_btn.click(
                        fn=self.run_all_tests, inputs=[verbose_check], outputs=[test_output]
                    )

                    run_failed_btn.click(fn=self.run_failed_only, outputs=[test_output])

                # Tab 2: Coverage
                with gr.Tab("📊 Coverage"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            run_coverage_btn = gr.Button(
                                "Run Coverage", variant="primary", size="lg"
                            )
                            gen_html_btn = gr.Button("Generate HTML Report", variant="secondary")
                            analyze_gaps_btn = gr.Button("Analyze Gaps", variant="secondary")

                        with gr.Column(scale=2):
                            coverage_summary = gr.Markdown(label="Coverage Summary")

                    coverage_output = gr.Textbox(
                        label="Coverage Details", lines=20, max_lines=40, show_copy_button=True
                    )

                    html_link = gr.Textbox(label="HTML Report Location", lines=2)
                    gaps_output = gr.Markdown(label="Gap Analysis")

                    run_coverage_btn.click(
                        fn=self.run_tests_with_coverage, outputs=[coverage_output, coverage_summary]
                    )

                    gen_html_btn.click(fn=self.generate_html_report, outputs=[html_link])

                    analyze_gaps_btn.click(fn=self.get_coverage_gaps, outputs=[gaps_output])

                # Tab 3: Run Specific Tests
                with gr.Tab("🎯 Specific Tests"):
                    with gr.Row():
                        test_file_dropdown = gr.Dropdown(
                            choices=self.get_test_files(),
                            label="Select Test File",
                            interactive=True,
                        )
                        refresh_btn = gr.Button("🔄")
                        run_specific_btn = gr.Button("Run Selected", variant="primary")

                    specific_output = gr.Textbox(
                        label="Test Output", lines=25, max_lines=50, show_copy_button=True
                    )

                    run_specific_btn.click(
                        fn=self.run_specific_test,
                        inputs=[test_file_dropdown],
                        outputs=[specific_output],
                    )

                    refresh_btn.click(
                        fn=lambda: gr.Dropdown(choices=self.get_test_files()),
                        outputs=[test_file_dropdown],
                    )

                # Tab 4: Quality Report
                with gr.Tab("📈 Quality Report"):
                    check_quality_btn = gr.Button(
                        "Check Test Quality", variant="primary", size="lg"
                    )
                    quality_output = gr.Markdown(label="Quality Report")

                    check_quality_btn.click(fn=self.check_test_quality, outputs=[quality_output])

                # Tab 5: Configuration
                with gr.Tab("⚙️ Config"):
                    gr.Markdown(f"""
                    ### Current Configuration

                    | Setting | Value |
                    |---------|-------|
                    | Project Root | `{self.config.project_root}` |
                    | Package | `{self.config.package_path}` |
                    | Tests Path | `{self.config.tests_path}` |
                    | Coverage Target | {self.config.coverage_target}% |
                    | Minimum Coverage | {self.config.coverage_fail_under}% |
                    | Branch Coverage | {self.config.branch_coverage} |

                    ### Usage

                    ```python
                    from comfy_headless.test_runner_ui import TestRunner, TestRunnerConfig

                    # Configure for your project
                    config = TestRunnerConfig(
                        project_name="My App",
                        package_path="my_app",
                        tests_path="my_app/tests",
                        coverage_target=80,
                    )

                    # Launch the UI
                    runner = TestRunner(config)
                    runner.launch()
                    ```
                    """)

            gr.Markdown("""
            ---
            **Tips for Improving Coverage:**
            - Run "Analyze Gaps" to see which modules need tests
            - Focus on modules with < 50% coverage first
            - Add tests for error paths and edge cases
            - Use property-based testing for input validation
            """)

        return app

    def launch(self, **kwargs):
        """Launch the test runner UI."""
        if not GRADIO_AVAILABLE:
            print("Error: Gradio is not installed.")
            print("Install with: pip install gradio")
            print("\nAlternatively, run tests directly:")
            print(f"  python -m pytest {self.config.tests_path} -v")
            sys.exit(1)

        app = self.create_interface()
        app.launch(
            server_name=kwargs.get("server_name", self.config.server_host),
            server_port=kwargs.get("server_port", self.config.server_port),
            share=kwargs.get("share", False),
            show_error=True,
        )


# =============================================================================
# LEGACY FUNCTIONS (for backward compatibility)
# =============================================================================

# Create a default runner instance
_default_runner = TestRunner(DEFAULT_CONFIG)


# Export legacy functions
def run_command(cmd, cwd=None):
    return _default_runner.run_command(cmd, cwd)


def get_test_files():
    return _default_runner.get_test_files()


def parse_pytest_output(output):
    return _default_runner.parse_pytest_output(output)


def parse_coverage_output(output):
    return _default_runner.parse_coverage_output(output)


def run_all_tests(verbose=True):
    return _default_runner.run_all_tests(verbose)


def run_tests_with_coverage():
    return _default_runner.run_tests_with_coverage()


def run_specific_test(test_file, verbose=True):
    return _default_runner.run_specific_test(test_file, verbose)


def run_failed_only():
    return _default_runner.run_failed_only()


def get_coverage_html_link():
    return _default_runner.generate_html_report()


def check_test_quality():
    return _default_runner.check_test_quality()


def create_interface():
    return _default_runner.create_interface()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Launch the test runner UI."""
    runner = TestRunner(DEFAULT_CONFIG)
    runner.launch()


if __name__ == "__main__":
    main()
