#!/usr/bin/env python3
"""
Atheris Fuzz Runner for comfy_headless

Atheris is a coverage-guided Python fuzzer from Google.
It only supports Linux and macOS (not Windows natively).

To run on Windows, use Docker or WSL:
    docker build -t comfy-fuzz -f tests/Dockerfile.atheris .
    docker run --rm comfy-fuzz

Or on Linux/macOS:
    pip install atheris
    python tests/atheris_runner.py [target] [corpus_dir]

Available targets:
    prompt      - Fuzz prompt validation/sanitization
    dimensions  - Fuzz dimension validation
    workflow    - Fuzz workflow compilation
    path        - Fuzz path validation
    all         - Run all targets sequentially

Example:
    python tests/atheris_runner.py prompt ./corpus/prompt -max_total_time=300
"""

import contextlib
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_atheris():
    """Check if Atheris is available."""
    try:
        import atheris

        return True
    except ImportError:
        return False


# =============================================================================
# FUZZ TARGETS
# =============================================================================


def fuzz_prompt(data: bytes) -> None:
    """Fuzz target for prompt validation and sanitization."""
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    from comfy_headless.exceptions import InvalidPromptError, SecurityError
    from comfy_headless.validation import sanitize_prompt, validate_prompt

    # Test sanitization
    with contextlib.suppress(Exception):
        sanitize_prompt(text)

    # Test validation with various settings
    for check_injection in [True, False]:
        for allow_html in [True, False]:
            try:
                validate_prompt(
                    text,
                    check_injection=check_injection,
                    allow_html=allow_html,
                    max_length=10000,
                )
            except (InvalidPromptError, SecurityError, TypeError, ValueError):
                pass
            except Exception as e:
                # Unexpected exception - this is interesting
                print(f"Unexpected exception in validate_prompt: {type(e).__name__}: {e}")


def fuzz_dimensions(data: bytes) -> None:
    """Fuzz target for dimension validation."""
    if len(data) < 8:
        return

    try:
        width = int.from_bytes(data[:4], "little", signed=True)
        height = int.from_bytes(data[4:8], "little", signed=True)
    except Exception:
        return

    from comfy_headless.exceptions import DimensionError
    from comfy_headless.validation import clamp_dimensions, validate_dimensions

    try:
        clamp_dimensions(width, height)
    except (DimensionError, TypeError, ValueError, OverflowError):
        pass
    except Exception as e:
        print(f"Unexpected exception in clamp_dimensions: {type(e).__name__}: {e}")

    try:
        validate_dimensions(width, height)
    except (DimensionError, TypeError, ValueError, OverflowError):
        pass
    except Exception as e:
        print(f"Unexpected exception in validate_dimensions: {type(e).__name__}: {e}")


def fuzz_workflow(data: bytes) -> None:
    """Fuzz target for workflow compilation."""
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    if not text.strip():
        return

    from comfy_headless.workflows import compile_workflow

    # Try various presets
    presets = ["draft", "balanced", "quality", "ultra"]
    for preset in presets:
        with contextlib.suppress(Exception):
            compile_workflow(prompt=text, preset=preset)


def fuzz_path(data: bytes) -> None:
    """Fuzz target for path validation."""
    try:
        path = data.decode("utf-8", errors="replace")
    except Exception:
        return

    if not path:
        return

    from comfy_headless.exceptions import SecurityError, ValidationError
    from comfy_headless.validation import validate_path

    try:
        validate_path(path)
    except (SecurityError, ValidationError, TypeError, ValueError, OSError):
        pass
    except Exception as e:
        print(f"Unexpected exception in validate_path: {type(e).__name__}: {e}")


def fuzz_json_workflow(data: bytes) -> None:
    """Fuzz target for JSON workflow parsing."""
    try:
        import json

        workflow = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    if not isinstance(workflow, dict):
        return

    from comfy_headless.workflows import WorkflowBuilder

    try:
        builder = WorkflowBuilder()
        # Try to process the workflow
        if "prompt" in workflow:
            builder.txt2img(
                prompt=str(workflow.get("prompt", "")),
                negative_prompt=str(workflow.get("negative_prompt", "")),
            )
    except Exception:
        pass


# =============================================================================
# ATHERIS RUNNER
# =============================================================================

TARGETS = {
    "prompt": fuzz_prompt,
    "dimensions": fuzz_dimensions,
    "workflow": fuzz_workflow,
    "path": fuzz_path,
    "json": fuzz_json_workflow,
}


def run_atheris(target_name: str, corpus_dir: str = None, extra_args: list = None):
    """Run Atheris with the specified target."""
    if not check_atheris():
        print("ERROR: Atheris is not installed.")
        print("Atheris only supports Linux and macOS.")
        print("")
        print("On Linux/macOS: pip install atheris")
        print("On Windows: Use Docker or WSL")
        print("")
        print("Docker example:")
        print("  docker build -t comfy-fuzz -f tests/Dockerfile.atheris .")
        print("  docker run --rm comfy-fuzz prompt")
        sys.exit(1)

    import atheris

    if target_name not in TARGETS:
        print(f"ERROR: Unknown target '{target_name}'")
        print(f"Available targets: {', '.join(TARGETS.keys())}")
        sys.exit(1)

    target_func = TARGETS[target_name]

    # Setup corpus directory
    if corpus_dir:
        os.makedirs(corpus_dir, exist_ok=True)
        sys.argv = [sys.argv[0], corpus_dir] + (extra_args or [])
    else:
        sys.argv = [sys.argv[0]] + (extra_args or [])

    print(f"Starting Atheris fuzzing for target: {target_name}")
    print(f"Corpus directory: {corpus_dir or 'in-memory'}")
    print("-" * 60)

    # Instrument modules
    atheris.instrument_all()

    # Run fuzzer
    atheris.Setup(sys.argv, target_func)
    atheris.Fuzz()


def run_all_targets(duration_per_target: int = 60):
    """Run all fuzz targets sequentially."""
    if not check_atheris():
        print("ERROR: Atheris is not installed.")
        sys.exit(1)

    import atheris

    for name, target in TARGETS.items():
        print(f"\n{'=' * 60}")
        print(f"Fuzzing target: {name}")
        print(f"Duration: {duration_per_target} seconds")
        print(f"{'=' * 60}\n")

        corpus_dir = f"./corpus/{name}"
        os.makedirs(corpus_dir, exist_ok=True)

        # Fork for each target to get clean state
        pid = os.fork() if hasattr(os, "fork") else 0
        if pid == 0:
            sys.argv = [sys.argv[0], corpus_dir, f"-max_total_time={duration_per_target}"]
            atheris.instrument_all()
            atheris.Setup(sys.argv, target)
            with contextlib.suppress(SystemExit):
                atheris.Fuzz()
            if hasattr(os, "fork"):
                os._exit(0)
        else:
            os.waitpid(pid, 0)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"\nAvailable targets: {', '.join(TARGETS.keys())}, all")
        sys.exit(0)

    target = sys.argv[1]
    corpus_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else None
    extra_args = [a for a in sys.argv[2:] if a.startswith("-")]

    if target == "all":
        duration = 60
        for arg in extra_args:
            if arg.startswith("-max_total_time="):
                duration = int(arg.split("=")[1])
        run_all_targets(duration)
    else:
        run_atheris(target, corpus_dir, extra_args)


if __name__ == "__main__":
    main()
