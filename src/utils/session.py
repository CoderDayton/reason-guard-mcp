"""Session manager base classes with sync and async support.

Provides common session management functionality for all reasoning tools.
Includes AsyncSessionManager with uvloop for high-performance async operations.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from typing import Generic, Protocol, TypeVar, runtime_checkable

# Install uvloop as default event loop policy (Linux/macOS only)
# Note: For Python 3.12+, prefer using uvloop.run() to wrap your main coroutine
# For module-level installation (needed for library code), we use install() with warning suppression
if sys.platform != "win32":
    try:
        import warnings

        import uvloop

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvloop")
            uvloop.install()
    except ImportError:
        pass  # uvloop not installed, use default asyncio


@runtime_checkable
class HasUpdatedAt(Protocol):
    """Protocol for objects with an updated_at timestamp."""

    updated_at: datetime


T = TypeVar("T")


class SessionNotFoundError(Exception):
    """Raised when a session ID is not found."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class SessionManager(Generic[T]):
    """Thread-safe base class for session management.

    Provides:
    - Thread-safe session storage with RLock
    - Common `_get_session()` lookup with error handling
    - `@contextmanager` helper for atomic session operations

    Usage:
        class MyManager(SessionManager[MyState]):
            def do_something(self, session_id: str) -> dict:
                with self.session(session_id) as state:
                    state.value = "updated"
                    return {"status": "ok"}
    """

    def __init__(self) -> None:
        """Initialize session manager with empty sessions and lock."""
        self._sessions: dict[str, T] = {}
        self._lock = threading.RLock()

    def _get_session(self, session_id: str) -> T:
        """Get session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session state object.

        Raises:
            SessionNotFoundError: If session doesn't exist.

        Note:
            This method does NOT acquire the lock. Caller must hold lock
            or use the `session()` context manager.

        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        return self._sessions[session_id]

    @contextmanager
    def session(self, session_id: str) -> Generator[T, None, None]:
        """Context manager for atomic session operations.

        Acquires lock, retrieves session, yields it, and releases lock
        even if an exception occurs.

        Args:
            session_id: Session identifier.

        Yields:
            Session state object.

        Raises:
            SessionNotFoundError: If session doesn't exist.

        Example:
            with self.session(session_id) as state:
                state.status = Status.COMPLETED
                state.updated_at = datetime.now()

        """
        with self._lock:
            yield self._get_session(session_id)

    @contextmanager
    def locked(self) -> Generator[dict[str, T], None, None]:
        """Context manager for operations on all sessions.

        Acquires lock and yields the sessions dict for bulk operations.

        Yields:
            The sessions dictionary.

        Example:
            with self.locked() as sessions:
                for sid, state in sessions.items():
                    ...

        """
        with self._lock:
            yield self._sessions

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists (thread-safe).

        Args:
            session_id: Session identifier.

        Returns:
            True if session exists.

        """
        with self._lock:
            return session_id in self._sessions

    def session_count(self) -> int:
        """Get number of active sessions (thread-safe).

        Returns:
            Number of sessions.

        """
        with self._lock:
            return len(self._sessions)

    def _register_session(self, session_id: str, state: T) -> None:
        """Register a new session (thread-safe).

        Args:
            session_id: Session identifier.
            state: Session state object.

        """
        with self._lock:
            self._sessions[session_id] = state

    def _remove_session(self, session_id: str) -> T | None:
        """Remove a session (thread-safe).

        Args:
            session_id: Session identifier.

        Returns:
            Removed session state, or None if not found.

        """
        with self._lock:
            return self._sessions.pop(session_id, None)

    def cleanup_stale(
        self,
        max_age: timedelta,
        *,
        now: datetime | None = None,
        predicate: Callable[[T], bool] | None = None,
    ) -> list[str]:
        """Remove sessions older than max_age (thread-safe).

        Prevents memory leaks in long-running servers by removing
        sessions that haven't been updated recently.

        Args:
            max_age: Maximum age for sessions. Sessions with
                `updated_at` older than `now - max_age` are removed.
            now: Reference time (defaults to datetime.now()).
            predicate: Optional additional filter. If provided, only
                sessions where `predicate(state)` returns True are
                eligible for removal. Use this to skip active sessions.

        Returns:
            List of removed session IDs.

        Raises:
            TypeError: If session state doesn't have `updated_at` attribute.

        Example:
            # Remove sessions inactive for more than 1 hour
            removed = manager.cleanup_stale(timedelta(hours=1))

            # Remove only completed sessions older than 30 minutes
            removed = manager.cleanup_stale(
                timedelta(minutes=30),
                predicate=lambda s: s.status == Status.COMPLETED
            )

        """
        if now is None:
            now = datetime.now()

        cutoff = now - max_age
        removed: list[str] = []

        with self._lock:
            # Collect stale session IDs first to avoid dict mutation during iteration
            stale_ids: list[str] = []
            for session_id, state in self._sessions.items():
                # Check if state has updated_at
                if not isinstance(state, HasUpdatedAt):
                    raise TypeError(
                        f"Session state {type(state).__name__} must have 'updated_at' attribute"
                    )

                # Combined check: updated_at exists, is stale, and passes predicate
                updated_at = getattr(state, "updated_at", None)
                is_stale = updated_at is not None and updated_at < cutoff
                passes_predicate = predicate is None or predicate(state)  # type: ignore[arg-type]
                if is_stale and passes_predicate:
                    stale_ids.append(session_id)

            # Remove stale sessions
            for session_id in stale_ids:
                del self._sessions[session_id]
                removed.append(session_id)

        return removed


class AsyncSessionManager(Generic[T]):
    """Async-native session manager using asyncio.Lock.

    Designed for high-performance async servers with uvloop.
    Does NOT block the event loop during lock acquisition.

    Provides:
    - Non-blocking session storage with asyncio.Lock
    - Async context manager for atomic session operations
    - Full API compatibility with SessionManager

    Usage:
        class MyAsyncManager(AsyncSessionManager[MyState]):
            async def do_something(self, session_id: str) -> dict:
                async with self.session(session_id) as state:
                    state.value = "updated"
                    return {"status": "ok"}

    Performance:
        With uvloop (auto-installed on Linux/macOS):
        - 2-4x faster than threading.Lock in async contexts
        - No GIL contention for lock operations
        - Better scaling under high concurrency
    """

    def __init__(self) -> None:
        """Initialize async session manager with empty sessions and async lock."""
        self._sessions: dict[str, T] = {}
        self._lock = asyncio.Lock()

    def _get_session_unsafe(self, session_id: str) -> T:
        """Get session by ID without lock (caller must hold lock).

        Args:
            session_id: Session identifier.

        Returns:
            Session state object.

        Raises:
            SessionNotFoundError: If session doesn't exist.

        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        return self._sessions[session_id]

    @asynccontextmanager
    async def session(self, session_id: str) -> AsyncGenerator[T, None]:
        """Async context manager for atomic session operations.

        Acquires async lock, retrieves session, yields it, and releases lock.
        Does NOT block the event loop while waiting for the lock.

        Args:
            session_id: Session identifier.

        Yields:
            Session state object.

        Raises:
            SessionNotFoundError: If session doesn't exist.

        Example:
            async with self.session(session_id) as state:
                state.status = Status.COMPLETED
                state.updated_at = datetime.now()

        """
        async with self._lock:
            yield self._get_session_unsafe(session_id)

    @asynccontextmanager
    async def locked(self) -> AsyncGenerator[dict[str, T], None]:
        """Async context manager for operations on all sessions.

        Acquires async lock and yields the sessions dict for bulk operations.

        Yields:
            The sessions dictionary.

        Example:
            async with self.locked() as sessions:
                for sid, state in sessions.items():
                    ...

        """
        async with self._lock:
            yield self._sessions

    async def session_exists(self, session_id: str) -> bool:
        """Check if session exists (async-safe).

        Args:
            session_id: Session identifier.

        Returns:
            True if session exists.

        """
        async with self._lock:
            return session_id in self._sessions

    async def session_count(self) -> int:
        """Get number of active sessions (async-safe).

        Returns:
            Number of sessions.

        """
        async with self._lock:
            return len(self._sessions)

    async def register_session(self, session_id: str, state: T) -> None:
        """Register a new session (async-safe).

        Args:
            session_id: Session identifier.
            state: Session state object.

        """
        async with self._lock:
            self._sessions[session_id] = state

    async def remove_session(self, session_id: str) -> T | None:
        """Remove a session (async-safe).

        Args:
            session_id: Session identifier.

        Returns:
            Removed session state, or None if not found.

        """
        async with self._lock:
            return self._sessions.pop(session_id, None)

    async def get_session(self, session_id: str) -> T:
        """Get session by ID (async-safe).

        Args:
            session_id: Session identifier.

        Returns:
            Session state object.

        Raises:
            SessionNotFoundError: If session doesn't exist.

        """
        async with self._lock:
            return self._get_session_unsafe(session_id)

    async def cleanup_stale(
        self,
        max_age: timedelta,
        *,
        now: datetime | None = None,
        predicate: Callable[[T], bool] | None = None,
    ) -> list[str]:
        """Remove sessions older than max_age (async-safe).

        Prevents memory leaks in long-running servers by removing
        sessions that haven't been updated recently.

        Args:
            max_age: Maximum age for sessions. Sessions with
                `updated_at` older than `now - max_age` are removed.
            now: Reference time (defaults to datetime.now()).
            predicate: Optional additional filter. If provided, only
                sessions where `predicate(state)` returns True are
                eligible for removal. Use this to skip active sessions.

        Returns:
            List of removed session IDs.

        Raises:
            TypeError: If session state doesn't have `updated_at` attribute.

        Example:
            # Remove sessions inactive for more than 1 hour
            removed = await manager.cleanup_stale(timedelta(hours=1))

            # Remove only completed sessions older than 30 minutes
            removed = await manager.cleanup_stale(
                timedelta(minutes=30),
                predicate=lambda s: s.status == Status.COMPLETED
            )

        """
        if now is None:
            now = datetime.now()

        cutoff = now - max_age
        removed: list[str] = []

        async with self._lock:
            # Collect stale session IDs first to avoid dict mutation during iteration
            stale_ids: list[str] = []
            for session_id, state in self._sessions.items():
                # Check if state has updated_at
                if not isinstance(state, HasUpdatedAt):
                    raise TypeError(
                        f"Session state {type(state).__name__} must have 'updated_at' attribute"
                    )

                # Combined check: updated_at exists, is stale, and passes predicate
                updated_at = getattr(state, "updated_at", None)
                is_stale = updated_at is not None and updated_at < cutoff
                passes_predicate = predicate is None or predicate(state)  # type: ignore[arg-type]
                if is_stale and passes_predicate:
                    stale_ids.append(session_id)

            # Remove stale sessions
            for session_id in stale_ids:
                del self._sessions[session_id]
                removed.append(session_id)

        return removed

    def get_all_sessions_snapshot(self) -> dict[str, T]:
        """Get a snapshot of all sessions (non-blocking, eventual consistency).

        Returns a shallow copy of the sessions dict without acquiring the lock.
        Use this for read-only operations where eventual consistency is acceptable.

        Returns:
            Shallow copy of sessions dictionary.

        Warning:
            The returned dict may be slightly stale. For guaranteed consistency,
            use `async with self.locked()` instead.

        """
        return dict(self._sessions)


def is_uvloop_installed() -> bool:
    """Check if uvloop is installed and active.

    Returns:
        True if uvloop is the active event loop policy.

    """
    if sys.platform == "win32":
        return False
    try:
        import uvloop

        policy = asyncio.get_event_loop_policy()
        return isinstance(policy, uvloop.EventLoopPolicy)
    except ImportError:
        return False
