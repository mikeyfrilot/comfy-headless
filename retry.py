"""
Comfy Headless - Retry and Resilience Utilities
=================================================

Modern retry patterns using tenacity library with circuit breaker support.

Provides:
- Retry with exponential backoff + jitter (prevents thundering herd)
- Circuit breaker pattern (fail fast when systems are down)
- Rate limiting (token bucket algorithm)
- Timeout handling

Usage:
    from comfy_headless.retry import retry_with_backoff, get_circuit_breaker

    @retry_with_backoff(max_attempts=3)
    def fetch_data():
        ...

    breaker = get_circuit_breaker("comfyui")
    with breaker:
        client.request(...)
"""

import asyncio
import concurrent.futures
import functools
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TypeVar, Union

from .config import settings
from .exceptions import CircuitOpenError, RetryExhaustedError
from .logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    # Retry decorators
    "retry_with_backoff",
    "retry_on_exception",
    "retry_async",
    # Circuit breaker
    "CircuitState",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "circuit_registry",
    "get_circuit_breaker",
    # Rate limiter
    "RateLimiter",
    # Timeout
    "OperationTimeoutError",
    "with_timeout",
    "async_timeout",
    # Constants
    "TENACITY_AVAILABLE",
]

# Try to import tenacity, fall back to custom implementation
try:
    import tenacity
    from tenacity import (
        after_log,
        before_sleep_log,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
        wait_exponential_jitter,
    )

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    tenacity = None


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

T = TypeVar("T")
ExceptionTypes = Union[type[Exception], tuple[type[Exception], ...]]


# =============================================================================
# RETRY DECORATOR (Tenacity-based when available)
# =============================================================================


def retry_with_backoff(
    max_attempts: int | None = None,
    backoff_base: float | None = None,
    backoff_max: float | None = None,
    jitter: bool | None = None,
    exceptions: ExceptionTypes = Exception,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable:
    """
    Decorator for retry with exponential backoff and optional jitter.

    Uses tenacity library when available for production-grade retry logic.
    Falls back to custom implementation if tenacity is not installed.

    Args:
        max_attempts: Maximum number of attempts (default from settings)
        backoff_base: Base for exponential backoff (default from settings)
        backoff_max: Maximum backoff time (default from settings)
        jitter: Add randomness to prevent thundering herd (default from settings)
        exceptions: Exception types to catch and retry
        on_retry: Optional callback called on each retry with (attempt, exception)

    Returns:
        Decorated function

    Example:
        @retry_with_backoff(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
        def fetch_data():
            ...

        # With jitter (recommended for distributed systems)
        @retry_with_backoff(max_attempts=5, jitter=True)
        async def api_call():
            ...
    """
    _max_attempts = max_attempts or settings.retry.max_retries
    _backoff_base = backoff_base or settings.retry.backoff_base
    _backoff_max = backoff_max or settings.retry.backoff_max
    _jitter = jitter if jitter is not None else settings.retry.backoff_jitter

    if TENACITY_AVAILABLE:
        # Use tenacity for production-grade retry
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            # Choose wait strategy based on jitter setting
            if _jitter:
                wait_strategy = wait_exponential_jitter(
                    initial=_backoff_base,
                    max=_backoff_max,
                    jitter=_backoff_max / 2,  # Up to half of max as jitter
                )
            else:
                wait_strategy = wait_exponential(
                    multiplier=_backoff_base,
                    max=_backoff_max,
                )

            # Build retry decorator
            retry_decorator = retry(
                stop=stop_after_attempt(_max_attempts),
                wait=wait_strategy,
                retry=retry_if_exception_type(exceptions),
                before_sleep=before_sleep_log(logger, log_level=20),  # INFO
                reraise=True,
            )

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                try:
                    return retry_decorator(func)(*args, **kwargs)
                except tenacity.RetryError as e:
                    raise RetryExhaustedError(
                        message=f"All {_max_attempts} retry attempts exhausted for {func.__name__}",
                        attempts=_max_attempts,
                        last_error=e.last_attempt.exception() if e.last_attempt else None,
                    )

            return wrapper

        return decorator

    else:
        # Fallback: Custom implementation with jitter
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                last_exception: Exception | None = None

                for attempt in range(1, _max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e

                        if attempt == _max_attempts:
                            logger.warning(
                                f"Retry exhausted for {func.__name__} after {attempt} attempts",
                                extra={"function": func.__name__, "attempts": attempt},
                            )
                            raise RetryExhaustedError(
                                message=f"All {_max_attempts} retry attempts exhausted for {func.__name__}",
                                attempts=_max_attempts,
                                last_error=last_exception,
                            )

                        # Calculate backoff with optional jitter
                        backoff = min(_backoff_base**attempt, _backoff_max)
                        if _jitter:
                            # Add up to 50% jitter
                            jitter_amount = backoff * random.uniform(0, 0.5)
                            backoff = backoff + jitter_amount

                        logger.debug(
                            f"Retry {attempt}/{_max_attempts} for {func.__name__}, "
                            f"waiting {backoff:.1f}s: {e}",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt,
                                "backoff": backoff,
                                "jitter": _jitter,
                                "error": str(e),
                            },
                        )

                        if on_retry:
                            on_retry(attempt, e)

                        time.sleep(backoff)

                # Should never reach here
                raise RetryExhaustedError(
                    message=f"Retry failed for {func.__name__}",
                    attempts=_max_attempts,
                    last_error=last_exception,
                )

            return wrapper

        return decorator


def retry_on_exception(
    func: Callable[..., T],
    max_attempts: int = 3,
    exceptions: ExceptionTypes = Exception,
    backoff_base: float = 1.5,
    jitter: bool = True,
) -> T:
    """
    Execute a function with retry (non-decorator version).

    Args:
        func: Function to execute
        max_attempts: Maximum attempts
        exceptions: Exceptions to catch
        backoff_base: Backoff multiplier
        jitter: Add randomness to backoff

    Returns:
        Function result

    Example:
        result = retry_on_exception(
            lambda: requests.get(url),
            max_attempts=3,
            exceptions=(ConnectionError,)
        )
    """
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts:
                backoff = min(backoff_base**attempt, 30.0)
                if jitter:
                    backoff += backoff * random.uniform(0, 0.5)
                time.sleep(backoff)

    raise RetryExhaustedError(
        message=f"All {max_attempts} retry attempts exhausted",
        attempts=max_attempts,
        last_error=last_exception,
    )


# =============================================================================
# ASYNC RETRY (for httpx/aiohttp)
# =============================================================================


def retry_async(
    max_attempts: int | None = None,
    backoff_base: float | None = None,
    backoff_max: float | None = None,
    jitter: bool | None = None,
    exceptions: ExceptionTypes = Exception,
) -> Callable:
    """
    Async-compatible retry decorator.

    Works with both tenacity (native async support) and asyncio.sleep fallback.

    Example:
        @retry_async(max_attempts=3)
        async def fetch_data():
            async with httpx.AsyncClient() as client:
                return await client.get(url)
    """
    import asyncio

    _max_attempts = max_attempts or settings.retry.max_retries
    _backoff_base = backoff_base or settings.retry.backoff_base
    _backoff_max = backoff_max or settings.retry.backoff_max
    _jitter = jitter if jitter is not None else settings.retry.backoff_jitter

    if TENACITY_AVAILABLE:

        def decorator(func):
            if _jitter:
                wait_strategy = wait_exponential_jitter(
                    initial=_backoff_base,
                    max=_backoff_max,
                )
            else:
                wait_strategy = wait_exponential(
                    multiplier=_backoff_base,
                    max=_backoff_max,
                )

            return retry(
                stop=stop_after_attempt(_max_attempts),
                wait=wait_strategy,
                retry=retry_if_exception_type(exceptions),
                reraise=True,
            )(func)

        return decorator
    else:
        # Fallback async implementation
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(1, _max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < _max_attempts:
                            backoff = min(_backoff_base**attempt, _backoff_max)
                            if _jitter:
                                backoff += backoff * random.uniform(0, 0.5)
                            await asyncio.sleep(backoff)

                raise RetryExhaustedError(
                    message=f"All {_max_attempts} attempts exhausted",
                    attempts=_max_attempts,
                    last_error=last_exception,
                )

            return wrapper

        return decorator


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when a service is down.

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Service is failing, requests are rejected immediately
        HALF_OPEN: Testing recovery, limited requests pass through

    Usage:
        breaker = CircuitBreaker("comfyui")

        # As context manager
        with breaker:
            result = client.request()

        # Manual usage
        if breaker.allow_request():
            try:
                result = client.request()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """

    name: str
    failure_threshold: int = field(default_factory=lambda: settings.retry.circuit_breaker_threshold)
    reset_timeout: float = field(default_factory=lambda: settings.retry.circuit_breaker_reset)
    success_threshold: int = 3  # Successes needed in half-open to close

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: datetime | None = field(default=None, init=False)
    _success_count_half_open: int = field(default=0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def _maybe_transition_to_half_open(self):
        """Check if we should transition from OPEN to HALF_OPEN."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            if elapsed >= self.reset_timeout:
                logger.info(
                    f"Circuit {self.name}: OPEN -> HALF_OPEN after {elapsed:.1f}s",
                    extra={"circuit": self.name, "transition": "open_to_half_open"},
                )
                self._state = CircuitState.HALF_OPEN
                self._success_count_half_open = 0

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                return True
            else:  # OPEN
                return False

    def record_success(self):
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count_half_open += 1
                if self._success_count_half_open >= self.success_threshold:
                    logger.info(
                        f"Circuit {self.name}: HALF_OPEN -> CLOSED after {self._success_count_half_open} successes",
                        extra={"circuit": self.name, "transition": "half_open_to_closed"},
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0

            elif self._state == CircuitState.CLOSED:
                # Reduce failure count on success (gradual recovery)
                if self._failure_count > 0:
                    self._failure_count -= 1

    def record_failure(self):
        """Record a failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                logger.warning(
                    f"Circuit {self.name}: HALF_OPEN -> OPEN after failure",
                    extra={"circuit": self.name, "transition": "half_open_to_open"},
                )
                self._state = CircuitState.OPEN

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.warning(
                        f"Circuit {self.name}: CLOSED -> OPEN after {self._failure_count} failures",
                        extra={
                            "circuit": self.name,
                            "transition": "closed_to_open",
                            "failures": self._failure_count,
                        },
                    )
                    self._state = CircuitState.OPEN

    def reset(self):
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._success_count_half_open = 0
            logger.info(
                f"Circuit {self.name}: manually reset to CLOSED", extra={"circuit": self.name}
            )

    def __enter__(self):
        """Context manager entry - check if request allowed."""
        if not self.allow_request():
            raise CircuitOpenError(
                service=self.name, message=f"Circuit breaker {self.name} is open"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record success/failure."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return self.__exit__(exc_type, exc_val, exc_tb)


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================


class CircuitBreakerRegistry:
    """
    Registry for circuit breakers.

    Provides centralized access to circuit breakers by service name.
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a service."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name)
            return self._breakers[name]

    def reset(self, name: str):
        """Reset a specific circuit breaker."""
        with self._lock:
            if name in self._breakers:
                self._breakers[name].reset()

    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def status(self) -> dict:
        """Get status of all circuit breakers."""
        with self._lock:
            return {
                name: {
                    "state": breaker.state.value,
                    "failure_count": breaker._failure_count,
                }
                for name, breaker in self._breakers.items()
            }


# Singleton registry
circuit_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Get a circuit breaker from the registry."""
    return circuit_registry.get(name)


# =============================================================================
# RATE LIMITER
# =============================================================================


@dataclass
class RateLimiter:
    """
    Simple rate limiter using token bucket algorithm.

    Usage:
        limiter = RateLimiter(rate=10, per_seconds=1)  # 10 requests/second

        if limiter.acquire():
            make_request()
        else:
            # Rate limited, wait or skip

        # Or blocking
        limiter.acquire(blocking=True)
        make_request()
    """

    rate: int  # Number of tokens
    per_seconds: float = 1.0  # Time window

    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self._tokens = float(self.rate)
        self._last_update = time.monotonic()

    def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.rate, self._tokens + (elapsed * self.rate / self.per_seconds))
        self._last_update = now

    def acquire(self, blocking: bool = False, timeout: float | None = None) -> bool:
        """
        Acquire a token.

        Args:
            blocking: If True, wait for a token
            timeout: Maximum time to wait if blocking

        Returns:
            True if token acquired, False otherwise
        """
        start = time.monotonic()

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

            if not blocking:
                return False

            if timeout is not None and time.monotonic() - start >= timeout:
                return False

            # Wait a bit before retrying
            time.sleep(self.per_seconds / self.rate)


# =============================================================================
# TIMEOUT UTILITIES
# =============================================================================


class OperationTimeoutError(Exception):
    """
    Operation timed out.

    Named to avoid shadowing the builtin TimeoutError.
    """

    pass


def with_timeout(timeout: float):
    """
    Decorator to add timeout to a function.

    Note: This only works for I/O-bound operations, not CPU-bound.
    For true timeout support, use concurrent.futures or asyncio.

    Args:
        timeout: Timeout in seconds

    Example:
        @with_timeout(10.0)
        def slow_operation():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    raise OperationTimeoutError(
                        f"Operation {func.__name__} timed out after {timeout}s"
                    )

        return wrapper

    return decorator


def async_timeout(timeout: float):
    """
    Async-compatible timeout decorator.

    Example:
        @async_timeout(10.0)
        async def slow_operation():
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                raise OperationTimeoutError(f"Operation {func.__name__} timed out after {timeout}s")

        return wrapper

    return decorator
