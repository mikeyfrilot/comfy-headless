# Comfy Headless - Makefile for Development Convenience
# Simplifies common development tasks

.PHONY: help venv dev test test-fast lint format typecheck security clean coverage install-dev install-full

help:  ## Show this help message
	@echo "Comfy Headless Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv:  ## Create virtual environment
	python -m venv .venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source .venv/bin/activate  (Linux/Mac)"
	@echo "  .venv\\Scripts\\activate     (Windows)"

dev: venv  ## Set up development environment
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -e ".[dev,ai,websocket,validation]"
	. .venv/bin/activate && pre-commit install
	@echo "Development environment ready!"

install-dev:  ## Install with dev dependencies (assumes venv active)
	pip install -e ".[dev,ai,websocket,validation]"

install-full:  ## Install with all optional dependencies (assumes venv active)
	pip install -e ".[full,dev]"

test:  ## Run all tests with coverage
	pytest tests/ -v --cov=comfy_headless --cov-report=html --cov-report=term-missing

test-fast:  ## Run tests without slow/integration tests
	pytest tests/ -v -m "not slow and not integration"

test-integration:  ## Run only integration tests
	pytest tests/ -v -m integration

lint:  ## Run linting checks
	ruff check .

format:  ## Format code with ruff
	ruff format .
	ruff check . --fix

typecheck:  ## Run type checking with mypy
	mypy comfy_headless

security:  ## Run security checks
	pip-audit
	bandit -c pyproject.toml -r comfy_headless

pre-commit:  ## Run all pre-commit hooks
	pre-commit run --all-files

coverage:  ## Generate coverage report
	pytest tests/ --cov=comfy_headless --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

clean:  ## Clean build artifacts and cache
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

verify:  ## Run all verification steps (lint, typecheck, test)
	@echo "Running linting..."
	@$(MAKE) lint
	@echo "Running type checking..."
	@$(MAKE) typecheck
	@echo "Running tests..."
	@$(MAKE) test-fast
	@echo "All checks passed!"
