"""Retry utilities with exponential backoff for resilient API calls."""

from __future__ import annotations

import asyncio
import functools
from typing import Callable, TypeVar, Any

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """Decorator to retry async functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds between retries
        max_delay: Maximum delay cap in seconds
        exponential: Use exponential backoff (doubles each retry)
        exceptions: Tuple of exception types to catch and retry
        
    Example:
        >>> @async_retry(max_attempts=3, exceptions=(ConnectionError,))
        >>> async def fetch_data():
        >>>     return await api_call()
        
    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt >= max_attempts:
                        logger.error(
                            "All retry attempts failed for %s",
                            func.__name__,
                            extra={
                                "function": func.__name__,
                                "attempts": max_attempts,
                                "error": str(e),
                            },
                        )
                        break

                    # Calculate delay
                    if exponential:
                        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    else:
                        delay = base_delay

                    logger.warning(
                        "Retry attempt %d/%d for %s failed, waiting %.1fs: %s",
                        attempt,
                        max_attempts,
                        func.__name__,
                        delay,
                        str(e),
                        extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "delay": delay,
                            "error_type": type(e).__name__,
                        },
                    )

                    await asyncio.sleep(delay)

            # All attempts failed, raise last exception
            raise last_exception

        return wrapper

    return decorator


class RetryContext:
    """Context manager for manual retry control.
    
    Useful when you need more fine-grained control over retry logic.
    
    Example:
        >>> retry = RetryContext(max_attempts=3, base_delay=1.0)
        >>> async with retry:
        >>>     for attempt in retry:
        >>>         try:
        >>>             result = await risky_operation()
        >>>             break
        >>>         except Exception as e:
        >>>             if not await retry.should_retry(e):
        >>>                 raise
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential = exponential
        self.current_attempt = 0
        self.last_exception: Exception | None = None

    def __iter__(self):
        """Iterate through retry attempts."""
        self.current_attempt = 0
        return self

    def __next__(self) -> int:
        """Get next attempt number."""
        if self.current_attempt >= self.max_attempts:
            raise StopIteration
        self.current_attempt += 1
        return self.current_attempt

    async def should_retry(self, exception: Exception) -> bool:
        """Determine if should retry after exception.
        
        Args:
            exception: Exception that occurred
            
        Returns:
            True if should retry, False otherwise
        """
        self.last_exception = exception

        if self.current_attempt >= self.max_attempts:
            return False

        # Calculate and apply delay
        if self.exponential:
            delay = min(
                self.base_delay * (2 ** (self.current_attempt - 1)), self.max_delay
            )
        else:
            delay = self.base_delay

        logger.debug(
            "Retrying after %.1fs (attempt %d/%d)",
            delay,
            self.current_attempt,
            self.max_attempts,
        )

        await asyncio.sleep(delay)
        return True

    async def __aenter__(self):
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        # Don't suppress exceptions
        return False
