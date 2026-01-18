# Contributing to Comfy Headless

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites
- Python 3.10 or higher
- Git
- A running ComfyUI instance (for integration tests)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/mcp-tool-shop/comfy-headless.git
   cd comfy-headless
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,full]"
   ```

4. **Install pre-commit hooks** (optional but recommended)
   ```bash
   pre-commit install
   ```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check linting
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Style Guidelines
- Follow PEP 8
- Use type hints for all function signatures
- Write docstrings for public functions and classes
- Keep functions focused and under 50 lines when possible
- Prefer composition over inheritance

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=comfy_headless --cov-report=term-missing

# Run specific test file
pytest tests/test_client.py

# Run tests matching a pattern
pytest -k "test_generate"

# Skip slow/integration tests
pytest -m "not slow and not integration"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures from `conftest.py`
- Mock external services (ComfyUI, Ollama) in unit tests
- Use `@pytest.mark.integration` for tests requiring external services

Example:
```python
import pytest
from comfy_headless import ComfyClient

@pytest.fixture
def mock_client(mocker):
    mocker.patch('comfy_headless.client.requests.get')
    return ComfyClient(base_url="http://localhost:8188")

def test_client_initialization(mock_client):
    assert mock_client.base_url == "http://localhost:8188"

@pytest.mark.integration
async def test_real_connection():
    """Requires running ComfyUI instance."""
    client = ComfyClient()
    assert await client.is_online()
```

## Type Checking

We use mypy for static type checking:

```bash
mypy comfy_headless
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Update documentation if needed
   - Follow the code style guidelines

3. **Run checks locally**
   ```bash
   ruff check .
   ruff format --check .
   mypy comfy_headless
   pytest
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "feat: add support for new video model"
   ```

   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation only
   - `style:` Formatting, no code change
   - `refactor:` Code change that neither fixes a bug nor adds a feature
   - `test:` Adding or updating tests
   - `chore:` Maintenance tasks

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

6. **PR Requirements**
   - All CI checks must pass
   - At least one approval from maintainers
   - No merge conflicts
   - Changelog entry for user-facing changes

## Reporting Bugs

When reporting bugs, please include:

1. **Environment info**
   - Python version
   - OS and version
   - comfy-headless version
   - ComfyUI version (if applicable)

2. **Steps to reproduce**
   - Minimal code example
   - Expected behavior
   - Actual behavior

3. **Error messages**
   - Full stack trace
   - Log output (set `COMFY_HEADLESS_LOG_LEVEL=DEBUG`)

## Feature Requests

For feature requests, please:

1. Check existing issues to avoid duplicates
2. Describe the use case clearly
3. Explain why existing features don't meet the need
4. Consider if you'd be willing to implement it

## Architecture Overview

```
comfy_headless/
├── __init__.py          # Package exports, lazy loading
├── __main__.py          # CLI entry point
├── config.py            # Settings (Pydantic/dataclass)
├── client.py            # Main ComfyClient
├── websocket_client.py  # WebSocket for real-time progress
├── workflows.py         # Workflow compilation
├── video.py             # Video generation presets
├── intelligence.py      # Ollama AI integration
├── ui.py                # Gradio interface
├── theme.py             # UI theming
├── exceptions.py        # Exception hierarchy
├── validation.py        # Input validation
├── retry.py             # Retry/circuit breaker
├── logging_config.py    # Structured logging
├── feature_flags.py     # Runtime feature detection
├── secrets_manager.py   # Secrets handling
├── cleanup.py           # Temp file management
├── help_system.py       # Context-aware help
└── http_client.py       # HTTP utilities
```

### Key Design Principles

1. **Modular by default** - Features are optional via extras
2. **Fail gracefully** - Missing dependencies don't crash
3. **Type-safe** - Full type hints with mypy strict
4. **User-friendly errors** - Helpful messages with recovery suggestions
5. **Minimal core** - Only requests + tenacity required

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
