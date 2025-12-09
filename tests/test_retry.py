"""Unit tests for src/utils/retry.py."""

from __future__ import annotations

import asyncio

import pytest

from src.utils.retry import retry_with_backoff, with_timeout


class TestRetryWithBackoff:
    """Test retry_with_backoff decorator."""

    def test_sync_function_success(self) -> None:
        """Test sync function that succeeds immediately."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def always_succeeds() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_sync_function_retry_then_success(self) -> None:
        """Test sync function that fails then succeeds."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def fails_then_succeeds() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"

        result = fails_then_succeeds()
        assert result == "success"
        assert call_count == 2

    def test_sync_function_max_retries_exceeded(self) -> None:
        """Test sync function that always fails."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent failure")

        with pytest.raises(ValueError, match="Persistent failure"):
            always_fails()

        assert call_count == 3  # Initial + 2 retries

    def test_sync_function_preserves_args(self) -> None:
        """Test sync function receives correct arguments."""

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def echo_args(a: int, b: str, c: int = 10) -> tuple[int, str, int]:
            return (a, b, c)

        result = echo_args(1, "hello", c=20)
        assert result == (1, "hello", 20)

    def test_sync_function_specific_exception_type(self) -> None:
        """Test retry only on specific exception types."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01, retry_on=(ValueError,))
        def raises_type_error() -> str:
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retried")

        # TypeError should not be retried
        with pytest.raises(TypeError):
            raises_type_error()

        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_async_function_success(self) -> None:
        """Test async function that succeeds immediately."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        async def async_succeeds() -> str:
            nonlocal call_count
            call_count += 1
            return "async success"

        result = await async_succeeds()
        assert result == "async success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_function_retry_then_success(self) -> None:
        """Test async function that fails then succeeds."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        async def async_fails_then_succeeds() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary async failure")
            return "async success"

        result = await async_fails_then_succeeds()
        assert result == "async success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_function_max_retries_exceeded(self) -> None:
        """Test async function that always fails."""
        call_count = 0

        @retry_with_backoff(max_attempts=2, base_delay=0.01)
        async def async_always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent async failure")

        with pytest.raises(ValueError, match="Persistent async failure"):
            await async_always_fails()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_function_preserves_args(self) -> None:
        """Test async function receives correct arguments."""

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        async def async_echo(x: int, y: str) -> str:
            return f"{x}-{y}"

        result = await async_echo(42, "test")
        assert result == "42-test"

    def test_decorator_preserves_function_metadata(self) -> None:
        """Test decorator preserves original function name."""

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def named_function() -> str:
            """Docstring preserved."""
            return "named"

        assert named_function.__name__ == "named_function"
        # Note: docstring may not be preserved with tenacity wrapper


class TestWithTimeout:
    """Test with_timeout decorator."""

    @pytest.mark.asyncio
    async def test_function_completes_within_timeout(self) -> None:
        """Test function that completes before timeout."""

        @with_timeout(1.0)
        async def quick_function() -> str:
            await asyncio.sleep(0.01)
            return "done"

        result = await quick_function()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_function_exceeds_timeout(self) -> None:
        """Test function that exceeds timeout raises TimeoutError."""

        @with_timeout(0.1)
        async def slow_function() -> str:
            await asyncio.sleep(10)
            return "never reached"

        with pytest.raises(asyncio.TimeoutError):
            await slow_function()

    @pytest.mark.asyncio
    async def test_function_preserves_args(self) -> None:
        """Test timeout wrapper preserves arguments."""

        @with_timeout(1.0)
        async def echo_with_timeout(value: str) -> str:
            return f"echoed: {value}"

        result = await echo_with_timeout("hello")
        assert result == "echoed: hello"

    def test_cannot_apply_to_sync_function(self) -> None:
        """Test with_timeout cannot be applied to sync functions."""
        with pytest.raises(TypeError, match="async functions"):

            @with_timeout(1.0)
            def sync_function() -> str:
                return "sync"

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self) -> None:
        """Test decorator preserves function name."""

        @with_timeout(1.0)
        async def named_async_function() -> str:
            """Async docstring."""
            return "named"

        assert named_async_function.__name__ == "named_async_function"

    @pytest.mark.asyncio
    async def test_timeout_cancels_properly(self) -> None:
        """Test that cancelled tasks are handled properly."""
        side_effect_executed = False

        @with_timeout(0.1)
        async def function_with_side_effect() -> str:
            nonlocal side_effect_executed
            await asyncio.sleep(10)
            side_effect_executed = True  # Should never execute
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await function_with_side_effect()

        # Give a moment for any lingering tasks
        await asyncio.sleep(0.05)
        assert not side_effect_executed


class TestRetryWithBackoffConfig:
    """Test retry_with_backoff configuration options."""

    def test_default_retries_all_exceptions(self) -> None:
        """Test default configuration retries any exception."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def raises_runtime_error() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Custom error")
            return "ok"

        result = raises_runtime_error()
        assert result == "ok"
        assert call_count == 2

    def test_custom_max_delay(self) -> None:
        """Test custom max_delay is accepted."""

        @retry_with_backoff(max_attempts=2, base_delay=0.01, max_delay=0.1)
        def simple_function() -> str:
            return "done"

        result = simple_function()
        assert result == "done"

    def test_retry_on_multiple_exception_types(self) -> None:
        """Test retry_on accepts multiple exception types."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01, retry_on=(ValueError, RuntimeError))
        def raises_different_errors() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First")
            if call_count == 2:
                raise RuntimeError("Second")
            return "success"

        result = raises_different_errors()
        assert result == "success"
        assert call_count == 3
