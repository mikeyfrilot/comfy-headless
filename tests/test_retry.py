"""Tests for retry module."""

import time

import pytest


class TestRetryWithBackoff:
    """Test retry decorator."""

    def test_succeeds_first_try(self):
        """Test function that succeeds immediately."""
        from comfy_headless.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_attempts=3)
        def succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retries_on_failure(self):
        """Test retry on transient failure."""
        from comfy_headless.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_attempts=3, backoff_base=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert call_count == 3

    def test_exhausts_retries(self):
        """Test retry exhaustion."""
        from comfy_headless.exceptions import RetryExhaustedError
        from comfy_headless.retry import retry_with_backoff

        @retry_with_backoff(max_attempts=2, backoff_base=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises((RetryExhaustedError, ValueError)):
            always_fails()


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_circuit_starts_closed(self):
        """Test circuit breaker initial state."""
        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=3)
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_opens_on_failures(self):
        """Test circuit opens after threshold failures."""
        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=2, reset_timeout=60)

        # Record failures
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_circuit_rejects_when_open(self):
        """Test circuit rejects calls when open."""
        from comfy_headless.exceptions import CircuitOpenError
        from comfy_headless.retry import CircuitBreaker

        breaker = CircuitBreaker(name="test", failure_threshold=1)
        breaker.record_failure()

        with pytest.raises(CircuitOpenError), breaker:
            pass

    def test_circuit_resets_on_success(self):
        """Test circuit resets after success in half-open."""
        from comfy_headless.retry import CircuitBreaker, CircuitState

        # success_threshold=1 so one success is enough to close
        breaker = CircuitBreaker(
            name="test", failure_threshold=1, reset_timeout=0.01, success_threshold=1
        )

        # Open the circuit
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for half-open
        time.sleep(0.02)

        # Trigger state check to move to HALF_OPEN
        _ = breaker.state  # This triggers _maybe_transition_to_half_open

        # Record success should close it
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED


class TestRateLimiter:
    """Test rate limiter."""

    def test_allows_within_limit(self):
        """Test rate limiter allows requests within limit."""
        from comfy_headless.retry import RateLimiter

        limiter = RateLimiter(rate=10, per_seconds=1.0)

        # Should allow 10 immediate requests
        for _ in range(10):
            assert limiter.acquire(blocking=False)

    def test_blocks_over_limit(self):
        """Test rate limiter blocks when over limit."""
        from comfy_headless.retry import RateLimiter

        limiter = RateLimiter(rate=2, per_seconds=1.0)

        # Use up tokens
        limiter.acquire(blocking=False)
        limiter.acquire(blocking=False)

        # Should not acquire more immediately
        assert not limiter.acquire(blocking=False)


class TestOperationTimeoutError:
    """Test timeout error."""

    def test_timeout_error_exists(self):
        """Test OperationTimeoutError is importable."""
        from comfy_headless.retry import OperationTimeoutError

        error = OperationTimeoutError("Timed out")
        assert "Timed out" in str(error)
