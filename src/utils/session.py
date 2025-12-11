"""Thread-safe session manager base class.

Provides common session management functionality for all reasoning tools.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Generic, Protocol, TypeVar, runtime_checkable


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
