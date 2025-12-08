"""Retry decorators with exponential backoff for LLM calls."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

P = ParamSpec("P")
T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: tuple[type[Exception], ...] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for LLM-safe retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        retry_on: Tuple of exception types to retry on. If None, retries on all exceptions.

    Returns:
        Decorated function with retry logic.

    Example:
        @retry_with_backoff(max_attempts=3, base_delay=1.0)
        async def call_llm(prompt: str) -> str:
            return await api.generate(prompt)

    """
    retry_exceptions = retry_on or (Exception,)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if asyncio.iscoroutinefunction(func):

            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=base_delay, min=base_delay, max=max_delay),
                retry=retry_if_exception_type(retry_exceptions),
                reraise=True,
            )
            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Retry attempt for {func.__name__}: {e}")
                    raise

            return async_wrapper  # type: ignore[return-value]

        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_delay, min=base_delay, max=max_delay),
            retry=retry_if_exception_type(retry_exceptions),
            reraise=True,
        )
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Retry attempt for {func.__name__}: {e}")
                raise

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def with_timeout(timeout_seconds: float) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add timeout to async functions.

    Args:
        timeout_seconds: Maximum execution time in seconds.

    Returns:
        Decorated function with timeout.

    Raises:
        asyncio.TimeoutError: If function exceeds timeout.

    Example:
        @with_timeout(30.0)
        async def slow_operation() -> str:
            await asyncio.sleep(100)
            return "done"

    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("with_timeout can only be applied to async functions")

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout_seconds,
            )

        return wrapper  # type: ignore[return-value]

    return decorator
