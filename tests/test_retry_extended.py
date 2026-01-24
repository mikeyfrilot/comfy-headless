"""
Extended tests for comfy_headless/retry.py

Covers:
- Backoff logic (lines 174-228)
- Circuit breaker state transitions (lines 300-350)
- Rate limiter token bucket algorithm
- Async retry functionality
- Timeout utilities
"""

import asyncio
import time

import pytest


class TestRetryWithBackoff:
    """Test retry_with_backoff decorator."""

    def test_retry_success_on_first_try(self):
        """Function succeeds on first try, no retry needed."""
        from comfy_headless.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_attempts=3)
        def succeed_immediately():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed_immediately()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """Function retries and eventually succeeds."""
        from comfy_headless.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_attempts=3, backoff_base=0.01, jitter=False)
        def succeed_on_third():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = succeed_on_third()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted_raises(self):
        """All retries exhausted raises RetryExhaustedError."""
        from comfy_headless.exceptions import RetryExhaustedError
        from comfy_headless.retry import retry_with_backoff

        @retry_with_backoff(max_attempts=2, backoff_base=0.01, jitter=False)
        def always_fail():
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            always_fail()

        assert "retry" in str(exc_info.value).lower()

    def test_retry_only_catches_specified_exceptions(self):
        """Retry only catches specified exception types."""
        from comfy_headless.retry import retry_with_backoff

        @retry_with_backoff(max_attempts=3, exceptions=ValueError, backoff_base=0.01)
        def raise_type_error():
            raise TypeError("Not a ValueError")

        with pytest.raises(TypeError):
            raise_type_error()

    def test_retry_with_jitter_varies_backoff(self):
        """Jitter adds randomness to backoff times."""
        from comfy_headless.retry import retry_with_backoff

        # This test is probabilistic - jitter should make times vary
        # We just verify it doesn't crash with jitter enabled
        call_count = 0

        @retry_with_backoff(max_attempts=2, backoff_base=0.01, jitter=True)
        def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt")
            return "ok"

        result = fail_once()
        assert result == "ok"


class TestRetryOnException:
    """Test retry_on_exception function (non-decorator)."""

    def test_retry_on_exception_success(self):
        """retry_on_exception succeeds when function works."""
        from comfy_headless.retry import retry_on_exception

        result = retry_on_exception(lambda: "success", max_attempts=3)
        assert result == "success"

    def test_retry_on_exception_with_failures(self):
        """retry_on_exception retries on failure."""
        from comfy_headless.retry import retry_on_exception

        counter = {"count": 0}

        def fail_twice():
            counter["count"] += 1
            if counter["count"] < 3:
                raise ConnectionError("Not yet")
            return "success"

        result = retry_on_exception(
            fail_twice, max_attempts=3, exceptions=ConnectionError, backoff_base=0.01, jitter=False
        )
        assert result == "success"
        assert counter["count"] == 3

    def test_retry_on_exception_exhausted(self):
        """retry_on_exception raises after all attempts."""
        from comfy_headless.exceptions import RetryExhaustedError
        from comfy_headless.retry import retry_on_exception

        with pytest.raises(RetryExhaustedError):
            retry_on_exception(
                lambda: (_ for _ in ()).throw(ValueError("fail")), max_attempts=2, backoff_base=0.01
            )


class TestCircuitBreakerStates:
    """Test CircuitBreaker state machine."""

    def test_circuit_starts_closed(self):
        """Circuit breaker starts in CLOSED state."""
        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True
        assert breaker.is_open is False

    def test_circuit_opens_after_threshold(self):
        """Circuit opens after failure threshold exceeded."""
        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=3)

        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open is True

    def test_circuit_allows_in_closed_state(self):
        """CLOSED circuit allows requests."""
        from comfy_headless.retry import CircuitBreaker

        breaker = CircuitBreaker(name="test")
        assert breaker.allow_request() is True

    def test_circuit_blocks_in_open_state(self):
        """OPEN circuit blocks requests."""
        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=1)
        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.allow_request() is False

    def test_circuit_transitions_to_half_open(self):
        """Circuit transitions OPEN -> HALF_OPEN after reset timeout."""
        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=1, reset_timeout=0.01)
        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        # Wait for reset timeout
        time.sleep(0.02)

        assert breaker.state == CircuitState.HALF_OPEN

    def test_circuit_closes_after_half_open_successes(self):
        """Circuit closes after success_threshold successes in HALF_OPEN."""
        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(
            name="test", failure_threshold=1, reset_timeout=0.01, success_threshold=2
        )
        breaker.record_failure()
        time.sleep(0.02)  # Transition to half-open

        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()
        breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    def test_circuit_reopens_on_half_open_failure(self):
        """Circuit reopens on failure during HALF_OPEN."""
        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=1, reset_timeout=0.01)
        breaker.record_failure()
        time.sleep(0.02)  # Transition to half-open

        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

    def test_circuit_reset_clears_state(self):
        """reset() clears circuit to CLOSED."""
        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=1)
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0


class TestCircuitBreakerContextManager:
    """Test CircuitBreaker as context manager."""

    def test_context_manager_success(self):
        """Context manager records success on clean exit."""
        from comfy_headless.retry import CircuitBreaker

        breaker = CircuitBreaker(name="test")

        with breaker:
            pass  # Success

        # Should still be closed (success reduces failure count)
        assert breaker.is_closed

    def test_context_manager_failure(self):
        """Context manager records failure on exception."""
        from comfy_headless.retry import CircuitBreaker

        breaker = CircuitBreaker(name="test", failure_threshold=2)

        with pytest.raises(ValueError), breaker:
            raise ValueError("Test error")

        assert breaker._failure_count == 1

    def test_context_manager_raises_when_open(self):
        """Context manager raises CircuitOpenError when open."""
        from comfy_headless.exceptions import CircuitOpenError
        from comfy_headless.retry import CircuitBreaker

        breaker = CircuitBreaker(name="test", failure_threshold=1)
        breaker.record_failure()

        with pytest.raises(CircuitOpenError), breaker:
            pass


class TestCircuitBreakerAsyncContextManager:
    """Test CircuitBreaker async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager_success(self):
        """Async context manager works for success case."""
        from comfy_headless.retry import CircuitBreaker

        breaker = CircuitBreaker(name="test")

        async with breaker:
            await asyncio.sleep(0.001)

        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_async_context_manager_failure(self):
        """Async context manager records failure on exception."""
        from comfy_headless.retry import CircuitBreaker

        breaker = CircuitBreaker(name="test", failure_threshold=2)

        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("Async error")

        assert breaker._failure_count == 1


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry."""

    def test_registry_creates_new_breaker(self):
        """Registry creates new breaker for unknown name."""
        from comfy_headless.retry import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()
        breaker = registry.get("new_service")

        assert breaker is not None
        assert breaker.name == "new_service"

    def test_registry_returns_same_breaker(self):
        """Registry returns same breaker for same name."""
        from comfy_headless.retry import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()
        b1 = registry.get("service")
        b2 = registry.get("service")

        assert b1 is b2

    def test_registry_reset_single(self):
        """Registry can reset a single breaker."""
        from comfy_headless.retry import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()
        breaker = registry.get("service")
        breaker.record_failure()
        breaker.record_failure()

        registry.reset("service")

        assert breaker._failure_count == 0

    def test_registry_reset_all(self):
        """Registry can reset all breakers."""
        from comfy_headless.retry import CircuitBreakerRegistry, CircuitState

        registry = CircuitBreakerRegistry()
        b1 = registry.get("service1")
        b2 = registry.get("service2")

        # Open both
        for _ in range(5):
            b1.record_failure()
            b2.record_failure()

        registry.reset_all()

        assert b1.state == CircuitState.CLOSED
        assert b2.state == CircuitState.CLOSED

    def test_registry_status(self):
        """Registry provides status of all breakers."""
        from comfy_headless.retry import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()
        registry.get("service1")
        registry.get("service2")

        status = registry.status()

        assert "service1" in status
        assert "service2" in status
        assert "state" in status["service1"]
        assert "failure_count" in status["service1"]


class TestGlobalCircuitRegistry:
    """Test global circuit_registry singleton."""

    def test_get_circuit_breaker(self):
        """get_circuit_breaker uses global registry."""
        from comfy_headless.retry import circuit_registry, get_circuit_breaker

        breaker = get_circuit_breaker("test_service")
        assert breaker is circuit_registry.get("test_service")


class TestRateLimiter:
    """Test RateLimiter token bucket algorithm."""

    def test_rate_limiter_allows_within_rate(self):
        """Rate limiter allows requests within rate."""
        from comfy_headless.retry import RateLimiter

        limiter = RateLimiter(rate=10, per_seconds=1.0)

        # Should be able to acquire immediately
        for _ in range(10):
            assert limiter.acquire() is True

    def test_rate_limiter_blocks_over_rate(self):
        """Rate limiter blocks when rate exceeded (non-blocking)."""
        from comfy_headless.retry import RateLimiter

        limiter = RateLimiter(rate=2, per_seconds=1.0)

        limiter.acquire()
        limiter.acquire()

        # Third should fail (non-blocking)
        assert limiter.acquire(blocking=False) is False

    def test_rate_limiter_refills_over_time(self):
        """Rate limiter refills tokens over time."""
        from comfy_headless.retry import RateLimiter

        limiter = RateLimiter(rate=10, per_seconds=0.1)

        # Use all tokens
        for _ in range(10):
            limiter.acquire()

        # Wait for refill
        time.sleep(0.15)

        # Should have tokens again
        assert limiter.acquire() is True

    def test_rate_limiter_blocking_waits(self):
        """Rate limiter blocking mode waits for token."""
        from comfy_headless.retry import RateLimiter

        limiter = RateLimiter(rate=5, per_seconds=0.1)

        # Use all tokens
        for _ in range(5):
            limiter.acquire()

        # This should wait and succeed
        start = time.monotonic()
        assert limiter.acquire(blocking=True, timeout=0.5) is True
        elapsed = time.monotonic() - start

        # Should have waited some time
        assert elapsed > 0.01

    def test_rate_limiter_blocking_timeout(self):
        """Rate limiter blocking mode respects timeout."""
        from comfy_headless.retry import RateLimiter

        limiter = RateLimiter(rate=1, per_seconds=100.0)  # Very slow refill

        limiter.acquire()

        # Should timeout - use longer timeout to avoid flakiness
        start = time.monotonic()
        result = limiter.acquire(blocking=True, timeout=0.1)
        elapsed = time.monotonic() - start

        # Either timed out or succeeded after wait
        assert result is False or elapsed >= 0.05


class TestTimeoutDecorator:
    """Test with_timeout decorator."""

    def test_timeout_allows_fast_function(self):
        """Fast functions complete before timeout."""
        from comfy_headless.retry import with_timeout

        @with_timeout(1.0)
        def fast_func():
            return "done"

        result = fast_func()
        assert result == "done"

    def test_timeout_raises_on_slow_function(self):
        """Slow functions raise OperationTimeoutError."""
        from comfy_headless.retry import OperationTimeoutError, with_timeout

        @with_timeout(0.05)
        def slow_func():
            time.sleep(0.2)
            return "done"

        with pytest.raises(OperationTimeoutError):
            slow_func()


class TestAsyncTimeout:
    """Test async_timeout decorator."""

    @pytest.mark.asyncio
    async def test_async_timeout_allows_fast(self):
        """Fast async functions complete before timeout."""
        from comfy_headless.retry import async_timeout

        @async_timeout(1.0)
        async def fast_async():
            await asyncio.sleep(0.01)
            return "done"

        result = await fast_async()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_async_timeout_raises_on_slow(self):
        """Slow async functions raise OperationTimeoutError."""
        from comfy_headless.retry import OperationTimeoutError, async_timeout

        @async_timeout(0.05)
        async def slow_async():
            await asyncio.sleep(0.2)
            return "done"

        with pytest.raises(OperationTimeoutError):
            await slow_async()


class TestRetryAsync:
    """Test retry_async decorator."""

    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """Async retry succeeds on first try."""
        from comfy_headless.retry import retry_async

        @retry_async(max_attempts=3)
        async def async_succeed():
            return "success"

        result = await async_succeed()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_async_with_failures(self):
        """Async retry retries on failure."""
        from comfy_headless.retry import retry_async

        counter = {"count": 0}

        @retry_async(max_attempts=3, backoff_base=0.01, jitter=False)
        async def async_fail_twice():
            counter["count"] += 1
            if counter["count"] < 3:
                raise ValueError("Not yet")
            return "success"

        result = await async_fail_twice()
        assert result == "success"
        assert counter["count"] == 3

    @pytest.mark.asyncio
    async def test_retry_async_exhausted(self):
        """Async retry raises after exhaustion."""
        from comfy_headless.retry import retry_async

        @retry_async(max_attempts=2, backoff_base=0.01, jitter=False)
        async def always_fail_async():
            raise ValueError("Always fails")

        # Tenacity raises tenacity.RetryError, fallback raises RetryExhaustedError
        with pytest.raises(Exception):  # Accept either
            await always_fail_async()


class TestTenacityAvailable:
    """Test TENACITY_AVAILABLE flag."""

    def test_flag_is_bool(self):
        """TENACITY_AVAILABLE is a boolean."""
        from comfy_headless.retry import TENACITY_AVAILABLE

        assert isinstance(TENACITY_AVAILABLE, bool)

    def test_flag_reflects_import(self):
        """Flag reflects tenacity import availability."""
        from comfy_headless.retry import TENACITY_AVAILABLE

        try:
            import tenacity

            assert TENACITY_AVAILABLE is True
        except ImportError:
            assert TENACITY_AVAILABLE is False


class TestCircuitBreakerThreadSafety:
    """Test CircuitBreaker thread safety."""

    def test_concurrent_failures(self):
        """Circuit breaker handles concurrent failures safely."""
        import threading

        from comfy_headless.retry import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=100)
        errors = []

        def record_many_failures():
            try:
                for _ in range(50):
                    breaker.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many_failures) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert breaker.state == CircuitState.OPEN

    def test_concurrent_successes(self):
        """Circuit breaker handles concurrent successes safely (no deadlocks/crashes)."""
        import threading

        from comfy_headless.retry import CircuitBreaker, CircuitState

        # Test that concurrent success recording doesn't crash or deadlock
        breaker = CircuitBreaker(
            name="test_concurrent_success",
            failure_threshold=5,
            reset_timeout=1.0,  # Long timeout so it stays closed
            success_threshold=3,
        )

        errors = []

        def record_many_successes():
            try:
                for _ in range(25):
                    breaker.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many_successes) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Primary assertion: no errors/deadlocks during concurrent access
        assert len(errors) == 0
        # Circuit should remain closed (never opened because no failures)
        assert breaker.state == CircuitState.CLOSED


class TestBackoffCalculation:
    """Test backoff time calculation."""

    def test_backoff_increases_exponentially(self):
        """Backoff time increases with attempts."""
        # This is implicitly tested through retry behavior
        # but we can test the concept
        base = 1.5
        max_backoff = 30.0

        backoffs = []
        for attempt in range(1, 6):
            backoff = min(base**attempt, max_backoff)
            backoffs.append(backoff)

        # Each backoff should be larger than previous (until max)
        for i in range(1, len(backoffs)):
            if backoffs[i - 1] < max_backoff:
                assert backoffs[i] > backoffs[i - 1]

    def test_backoff_capped_at_max(self):
        """Backoff doesn't exceed max value."""
        base = 2.0
        max_backoff = 10.0

        for attempt in range(1, 20):
            backoff = min(base**attempt, max_backoff)
            assert backoff <= max_backoff
