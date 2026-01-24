#!/usr/bin/env python3
"""
Fuzz Testing Runner for comfy_headless

This script runs Hypothesis-based fuzz tests with configurable parameters.
Works on all platforms (Windows, Linux, macOS).

For coverage-guided fuzzing with Atheris (Linux/macOS only):
    python tests/atheris_runner.py [target]

Usage:
    python tests/run_fuzz.py                    # Run all fuzz tests (default settings)
    python tests/run_fuzz.py --examples 5000   # More examples per test
    python tests/run_fuzz.py --time 600        # Run for 10 minutes total
    python tests/run_fuzz.py --security        # Focus on security tests only
    python tests/run_fuzz.py --stress          # Run stress tests
    python tests/run_fuzz.py --coverage        # Run with coverage measurement
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add comfy_headless parent directory to path
_this_file = Path(__file__).resolve()
_comfy_headless_dir = _this_file.parent.parent
_project_root = _comfy_headless_dir.parent
sys.path.insert(0, str(_project_root))


def run_hypothesis_fuzz(
    max_examples: int = 1000,
    max_time: int = 300,
    security_only: bool = False,
    stress_only: bool = False,
    with_coverage: bool = False,
    verbose: bool = True,
):
    """Run Hypothesis-based fuzz testing."""
    import subprocess

    # Build pytest command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "comfy_headless/tests/test_fuzz.py",
        "-v" if verbose else "-q",
        f"--hypothesis-seed={int(time.time())}",
    ]

    # Add coverage if requested
    if with_coverage:
        cmd.extend(
            [
                "--cov=comfy_headless",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov_fuzz",
            ]
        )

    # Filter tests
    if security_only:
        cmd.extend(["-k", "Security"])
    elif stress_only:
        cmd.extend(["-k", "Stress"])

    # Set hypothesis profile via environment
    env = os.environ.copy()
    env["HYPOTHESIS_MAX_EXAMPLES"] = str(max_examples)

    print("Running Hypothesis fuzz tests...")
    print(f"  Max examples per test: {max_examples}")
    print(f"  Security only: {security_only}")
    print(f"  Stress only: {stress_only}")
    print(f"  With coverage: {with_coverage}")
    print("-" * 60)

    # Run pytest from project root so imports work
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=_project_root,
        env=env,
    )
    elapsed = time.time() - start

    print("-" * 60)
    print(f"Completed in {elapsed:.1f} seconds")

    if with_coverage:
        print("Coverage report: htmlcov_fuzz/index.html")

    return result.returncode


def run_targeted_fuzz(target: str, iterations: int = 10000, verbose: bool = True):
    """Run targeted fuzzing on a specific component."""
    from hypothesis import HealthCheck, Phase, given, settings
    from hypothesis import strategies as st

    # Import targets from test_fuzz
    from comfy_headless.tests.test_fuzz import (
        fuzz_dimension_validation,
        fuzz_prompt_validation,
        fuzz_workflow_compilation,
    )

    targets = {
        "prompt": (fuzz_prompt_validation, st.binary(min_size=0, max_size=4096)),
        "dimensions": (fuzz_dimension_validation, st.binary(min_size=8, max_size=100)),
        "workflow": (fuzz_workflow_compilation, st.binary(min_size=1, max_size=2048)),
    }

    if target not in targets:
        print(f"Unknown target: {target}")
        print(f"Available: {', '.join(targets.keys())}")
        return 1

    func, strategy = targets[target]

    print(f"Fuzzing target: {target}")
    print(f"Iterations: {iterations}")
    print("-" * 60)

    @given(strategy)
    @settings(
        max_examples=iterations,
        phases=[Phase.generate],
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def fuzz_test(data):
        func(data)

    start = time.time()
    try:
        fuzz_test()
        print(f"PASSED: No crashes in {iterations} iterations")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return 1
    finally:
        elapsed = time.time() - start
        print(f"Completed in {elapsed:.1f} seconds ({iterations / elapsed:.0f} iter/sec)")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Fuzz testing runner for comfy_headless",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--examples",
        "-n",
        type=int,
        default=1000,
        help="Max examples per Hypothesis test (default: 1000)",
    )
    parser.add_argument(
        "--time",
        "-t",
        type=int,
        default=300,
        help="Max total time in seconds (default: 300)",
    )
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run security-focused tests only",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run stress tests only",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage measurement",
    )
    parser.add_argument(
        "--target",
        choices=["prompt", "dimensions", "workflow"],
        help="Run targeted fuzzing on a specific component",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Less verbose output",
    )

    args = parser.parse_args()

    if args.target:
        return run_targeted_fuzz(
            target=args.target,
            iterations=args.examples,
            verbose=not args.quiet,
        )
    else:
        return run_hypothesis_fuzz(
            max_examples=args.examples,
            max_time=args.time,
            security_only=args.security,
            stress_only=args.stress,
            with_coverage=args.coverage,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    sys.exit(main())
