"""Unit tests for SessionManager base class."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest

from src.utils.session import SessionManager, SessionNotFoundError


@dataclass
class MockState:
    """Mock state object for testing."""

    session_id: str
    value: str = ""
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class MockManager(SessionManager[MockState]):
    """Concrete implementation for testing."""

    def create_session(self, session_id: str, value: str = "") -> MockState:
        """Create and register a new session."""
        state = MockState(session_id=session_id, value=value)
        self._register_session(session_id, state)
        return state


class TestSessionManagerBasics:
    """Tests for basic SessionManager operations."""

    def test_init_empty(self) -> None:
        """New manager should have no sessions."""
        mgr = MockManager()
        assert mgr.session_count() == 0

    def test_register_session(self) -> None:
        """_register_session should add session to storage."""
        mgr = MockManager()
        state = MockState(session_id="test1")
        mgr._register_session("test1", state)
        assert mgr.session_count() == 1
        assert mgr.session_exists("test1")

    def test_session_exists_false(self) -> None:
        """session_exists should return False for unknown IDs."""
        mgr = MockManager()
        assert mgr.session_exists("nonexistent") is False

    def test_get_session_success(self) -> None:
        """_get_session should return registered session."""
        mgr = MockManager()
        state = MockState(session_id="test1", value="hello")
        mgr._register_session("test1", state)

        retrieved = mgr._get_session("test1")
        assert retrieved.value == "hello"

    def test_get_session_not_found(self) -> None:
        """_get_session should raise SessionNotFoundError."""
        mgr = MockManager()
        with pytest.raises(SessionNotFoundError) as exc_info:
            mgr._get_session("nonexistent")
        assert exc_info.value.session_id == "nonexistent"

    def test_remove_session_success(self) -> None:
        """_remove_session should remove and return session."""
        mgr = MockManager()
        state = MockState(session_id="test1")
        mgr._register_session("test1", state)

        removed = mgr._remove_session("test1")
        assert removed is state
        assert mgr.session_count() == 0
        assert not mgr.session_exists("test1")

    def test_remove_session_not_found(self) -> None:
        """_remove_session should return None for unknown IDs."""
        mgr = MockManager()
        removed = mgr._remove_session("nonexistent")
        assert removed is None


class TestSessionContextManager:
    """Tests for session() context manager."""

    def test_session_yields_state(self) -> None:
        """session() should yield the session state."""
        mgr = MockManager()
        mgr.create_session("test1", "original")

        with mgr.session("test1") as state:
            assert state.value == "original"

    def test_session_allows_mutation(self) -> None:
        """Mutations inside session() should persist."""
        mgr = MockManager()
        mgr.create_session("test1", "original")

        with mgr.session("test1") as state:
            state.value = "modified"

        with mgr.session("test1") as state:
            assert state.value == "modified"

    def test_session_not_found(self) -> None:
        """session() should raise SessionNotFoundError."""
        mgr = MockManager()
        with pytest.raises(SessionNotFoundError), mgr.session("nonexistent"):
            pass

    def test_session_exception_releases_lock(self) -> None:
        """Lock should be released even if exception occurs."""
        mgr = MockManager()
        mgr.create_session("test1")

        with pytest.raises(ValueError), mgr.session("test1") as state:
            raise ValueError("test error")

        # Lock should be released - this should not deadlock
        with mgr.session("test1") as state:
            state.value = "after_exception"

        assert mgr._get_session("test1").value == "after_exception"


class TestLockedContextManager:
    """Tests for locked() context manager."""

    def test_locked_yields_sessions_dict(self) -> None:
        """locked() should yield the sessions dictionary."""
        mgr = MockManager()
        mgr.create_session("test1")
        mgr.create_session("test2")

        with mgr.locked() as sessions:
            assert len(sessions) == 2
            assert "test1" in sessions
            assert "test2" in sessions

    def test_locked_allows_iteration(self) -> None:
        """Should be able to iterate over sessions."""
        mgr = MockManager()
        mgr.create_session("a", "value_a")
        mgr.create_session("b", "value_b")

        values = []
        with mgr.locked() as sessions:
            for sid, state in sessions.items():
                values.append((sid, state.value))

        assert sorted(values) == [("a", "value_a"), ("b", "value_b")]

    def test_locked_exception_releases_lock(self) -> None:
        """Lock should be released even if exception occurs."""
        mgr = MockManager()
        mgr.create_session("test1")

        with pytest.raises(RuntimeError), mgr.locked():
            raise RuntimeError("test error")

        # Lock should be released - this should not deadlock
        with mgr.locked() as sessions:
            assert len(sessions) == 1


class TestCleanupStale:
    """Tests for cleanup_stale() method."""

    def test_cleanup_removes_old_sessions(self) -> None:
        """Old sessions should be removed."""
        mgr = MockManager()

        # Create old session
        old_state = MockState(session_id="old")
        old_state.updated_at = datetime.now() - timedelta(hours=2)
        mgr._register_session("old", old_state)

        # Create recent session
        new_state = MockState(session_id="new")
        new_state.updated_at = datetime.now()
        mgr._register_session("new", new_state)

        removed = mgr.cleanup_stale(timedelta(hours=1))

        assert removed == ["old"]
        assert mgr.session_count() == 1
        assert mgr.session_exists("new")
        assert not mgr.session_exists("old")

    def test_cleanup_with_custom_now(self) -> None:
        """Should use custom 'now' parameter."""
        mgr = MockManager()

        state = MockState(session_id="test")
        state.updated_at = datetime(2024, 1, 1, 12, 0, 0)
        mgr._register_session("test", state)

        # Using custom 'now' that makes session old
        removed = mgr.cleanup_stale(timedelta(hours=1), now=datetime(2024, 1, 1, 14, 0, 0))

        assert removed == ["test"]

    def test_cleanup_with_predicate(self) -> None:
        """Predicate should filter which sessions are eligible."""
        mgr = MockManager()

        # Both old, but different statuses
        completed = MockState(session_id="completed", status="completed")
        completed.updated_at = datetime.now() - timedelta(hours=2)
        mgr._register_session("completed", completed)

        active = MockState(session_id="active", status="active")
        active.updated_at = datetime.now() - timedelta(hours=2)
        mgr._register_session("active", active)

        # Only remove completed sessions
        removed = mgr.cleanup_stale(timedelta(hours=1), predicate=lambda s: s.status == "completed")

        assert removed == ["completed"]
        assert mgr.session_exists("active")
        assert not mgr.session_exists("completed")

    def test_cleanup_empty_manager(self) -> None:
        """Cleanup on empty manager should return empty list."""
        mgr = MockManager()
        removed = mgr.cleanup_stale(timedelta(hours=1))
        assert removed == []

    def test_cleanup_nothing_stale(self) -> None:
        """Should return empty list if no sessions are stale."""
        mgr = MockManager()
        mgr.create_session("test1")
        mgr.create_session("test2")

        removed = mgr.cleanup_stale(timedelta(hours=1))
        assert removed == []
        assert mgr.session_count() == 2

    def test_cleanup_all_stale(self) -> None:
        """Should handle removing all sessions."""
        mgr = MockManager()

        for i in range(5):
            state = MockState(session_id=f"test{i}")
            state.updated_at = datetime.now() - timedelta(hours=2)
            mgr._register_session(f"test{i}", state)

        removed = mgr.cleanup_stale(timedelta(hours=1))

        assert len(removed) == 5
        assert mgr.session_count() == 0


class TestSessionNotFoundError:
    """Tests for SessionNotFoundError exception."""

    def test_error_message(self) -> None:
        """Error should include session_id in message."""
        error = SessionNotFoundError("abc123")
        assert "abc123" in str(error)
        assert error.session_id == "abc123"

    def test_error_is_exception(self) -> None:
        """Should be a proper Exception subclass."""
        error = SessionNotFoundError("test")
        assert isinstance(error, Exception)


class TestThreadSafety:
    """Tests for thread-safety guarantees."""

    def test_reentrant_lock(self) -> None:
        """RLock should allow reentrant access."""
        mgr = MockManager()
        mgr.create_session("test")

        # Nested lock acquisition should work with RLock
        with mgr._lock, mgr._lock:
            state = mgr._get_session("test")
            assert state is not None

    def test_register_idempotent(self) -> None:
        """Re-registering same ID should overwrite."""
        mgr = MockManager()

        state1 = MockState(session_id="test", value="first")
        mgr._register_session("test", state1)

        state2 = MockState(session_id="test", value="second")
        mgr._register_session("test", state2)

        assert mgr.session_count() == 1
        assert mgr._get_session("test").value == "second"
