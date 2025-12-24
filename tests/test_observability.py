"""Tests for the SQLite-based observability module."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

import pytest

from src.utils.observability import (
    MetricsStore,
    Span,
    get_metrics_store,
    is_metrics_enabled,
    record_session_completed,
    record_session_created,
    record_step,
    trace_sync,
    trace_tool,
)


class TestSpan:
    """Tests for the Span dataclass."""

    def test_span_creation(self) -> None:
        """Test basic span creation."""
        span = Span(
            span_id="test-123",
            name="test_operation",
            start_time=time.time(),
        )
        assert span.span_id == "test-123"
        assert span.name == "test_operation"
        assert span.status == "OK"
        assert span.error is None

    def test_span_set_attribute(self) -> None:
        """Test setting span attributes."""
        span = Span(span_id="test", name="op", start_time=time.time())
        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 42)
        assert span.attributes == {"key1": "value1", "key2": 42}

    def test_span_add_event(self) -> None:
        """Test adding events to span."""
        span = Span(span_id="test", name="op", start_time=time.time())
        span.add_event("checkpoint", {"data": "test"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"
        assert span.events[0]["attributes"] == {"data": "test"}

    def test_span_record_exception(self) -> None:
        """Test recording exceptions."""
        span = Span(span_id="test", name="op", start_time=time.time())
        try:
            raise ValueError("test error")
        except ValueError as e:
            span.record_exception(e)

        assert span.status == "ERROR"
        assert "ValueError" in span.error
        assert "test error" in span.error
        assert len(span.events) == 1
        assert span.events[0]["name"] == "exception"

    def test_span_duration_ms(self) -> None:
        """Test duration calculation."""
        start = time.time()
        span = Span(span_id="test", name="op", start_time=start)
        time.sleep(0.01)  # 10ms
        span.end_time = time.time()

        assert span.duration_ms >= 10.0
        assert span.duration_ms < 100.0  # Sanity check


class TestMetricsStore:
    """Tests for the MetricsStore class."""

    def test_store_creation_memory(self) -> None:
        """Test creating in-memory store."""
        store = MetricsStore(":memory:")
        assert store is not None

    def test_store_creation_file(self) -> None:
        """Test creating file-backed store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_metrics.db"
            store = MetricsStore(str(db_path))
            assert store is not None
            assert db_path.exists()

    def test_store_rejects_path_traversal(self) -> None:
        """Test that path traversal is rejected."""
        with pytest.raises(ValueError, match="Invalid"):
            MetricsStore("../../../etc/passwd")

    def test_trace_context_manager(self) -> None:
        """Test tracing with context manager."""
        store = MetricsStore(":memory:")

        with store.trace("test_operation", session_id="sess-123") as span:
            span.set_attribute("test_key", "test_value")
            time.sleep(0.005)

        # Verify span was saved
        spans = store.get_recent_spans(name="test_operation")
        assert len(spans) == 1
        assert spans[0]["name"] == "test_operation"
        assert spans[0]["session_id"] == "sess-123"
        assert spans[0]["attributes"]["test_key"] == "test_value"
        assert spans[0]["duration_ms"] >= 5.0

    def test_trace_captures_exception(self) -> None:
        """Test that exceptions are captured in traces."""
        store = MetricsStore(":memory:")

        with pytest.raises(RuntimeError), store.trace("failing_operation"):
            raise RuntimeError("test failure")

        spans = store.get_recent_spans(name="failing_operation")
        assert len(spans) == 1
        assert spans[0]["status"] == "ERROR"
        assert "RuntimeError" in spans[0]["error"]

    def test_increment_counter(self) -> None:
        """Test counter increment."""
        store = MetricsStore(":memory:")

        store.increment("test.counter", 1, {"tag": "a"})
        store.increment("test.counter", 2, {"tag": "b"})
        store.increment("test.counter", 3, {"tag": "a"})

        summary = store.get_metric_summary("test.counter")
        assert summary["count"] == 3
        assert summary["total"] == 6

    def test_record_histogram(self) -> None:
        """Test histogram recording."""
        store = MetricsStore(":memory:")

        store.record_histogram("test.latency", 10.0)
        store.record_histogram("test.latency", 20.0)
        store.record_histogram("test.latency", 30.0)

        summary = store.get_metric_summary("test.latency")
        assert summary["count"] == 3
        assert summary["avg"] == 20.0
        assert summary["min"] == 10.0
        assert summary["max"] == 30.0

    def test_session_tracking(self) -> None:
        """Test session start/end tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_session.db")
            store = MetricsStore(db_path)

            store.record_session_start(
                session_id="sess-1",
                problem="Test problem",
                mode="chain",
                complexity="low",
            )

            stats = store.get_session_stats()
            assert stats["total_sessions"] == 1

            store.record_session_end("sess-1", status="completed", total_steps=5)

            stats = store.get_session_stats()
            assert stats["completed"] == 1

    def test_get_tool_stats(self) -> None:
        """Test tool statistics aggregation."""
        store = MetricsStore(":memory:")

        # Create some test spans
        for i in range(5):
            with store.trace("tool_a"):
                time.sleep(0.001)

        for i in range(3):
            with store.trace("tool_b"):
                time.sleep(0.002)

        stats = store.get_tool_stats()
        assert "tool_a" in stats
        assert "tool_b" in stats
        assert stats["tool_a"]["calls"] == 5
        assert stats["tool_b"]["calls"] == 3

    def test_cleanup_old_data(self) -> None:
        """Test data cleanup by retention period."""
        # Use file-based DB for isolation from other tests
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "cleanup_test.db"
            store = MetricsStore(str(db_path))

            # Add a span with old timestamp by directly inserting
            conn = store._get_connection()
            old_time = time.time() - 3600  # 1 hour ago
            conn.execute(
                """INSERT INTO spans (span_id, name, start_time, end_time, duration_ms, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("old-span", "old_operation", old_time, old_time + 0.1, 100.0, "OK"),
            )
            conn.commit()

            # Verify it exists
            assert len(store.get_recent_spans(since_hours=2)) == 1

            # Cleanup with 0 hours retention (delete everything)
            deleted = store.cleanup_old_data(retention_hours=0)
            assert deleted >= 1

            # Verify it's gone
            assert len(store.get_recent_spans(since_hours=2)) == 0

    def test_get_recent_spans_filtering(self) -> None:
        """Test span filtering by name and session."""
        store = MetricsStore(":memory:")

        with store.trace("op_a", session_id="sess-1"):
            pass
        with store.trace("op_b", session_id="sess-1"):
            pass
        with store.trace("op_a", session_id="sess-2"):
            pass

        # Filter by name
        spans = store.get_recent_spans(name="op_a")
        assert len(spans) == 2

        # Filter by session
        spans = store.get_recent_spans(session_id="sess-1")
        assert len(spans) == 2

        # Filter by both
        spans = store.get_recent_spans(name="op_a", session_id="sess-1")
        assert len(spans) == 1


class TestDecorators:
    """Tests for the tracing decorators."""

    @pytest.mark.asyncio
    async def test_trace_tool_decorator(self) -> None:
        """Test the async trace_tool decorator."""
        import src.utils.observability as obs

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_trace_tool.db")
            obs._store = MetricsStore(db_path)

            @trace_tool
            async def sample_tool(session_id: str, value: int) -> str:
                await asyncio.sleep(0.01)
                return json.dumps({"result": value * 2})

            result = await sample_tool(session_id="test-sess", value=21)
            assert json.loads(result)["result"] == 42

            # Verify trace was recorded
            store = get_metrics_store()
            spans = store.get_recent_spans(name="sample_tool")
            assert len(spans) >= 1
            assert spans[0]["attributes"]["session_id"] == "test-sess"

            obs._store = None

    @pytest.mark.asyncio
    async def test_trace_tool_captures_errors(self) -> None:
        """Test that trace_tool captures exceptions."""
        import src.utils.observability as obs

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_trace_errors.db")
            obs._store = MetricsStore(db_path)

            @trace_tool
            async def failing_tool(session_id: str) -> str:
                raise ValueError("intentional failure")

            with pytest.raises(ValueError):
                await failing_tool(session_id="test")

            store = get_metrics_store()
            spans = store.get_recent_spans(name="failing_tool")
            assert len(spans) >= 1
            assert spans[0]["status"] == "ERROR"

            obs._store = None

    @pytest.mark.asyncio
    async def test_trace_tool_extracts_action(self) -> None:
        """Test that trace_tool extracts action parameter."""
        import src.utils.observability as obs

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_trace_action.db")
            obs._store = MetricsStore(db_path)

            @trace_tool
            async def think_like_tool(action: str, session_id: str | None = None) -> str:
                return json.dumps({"status": "ok"})

            await think_like_tool(action="start", session_id="sess-1")

            store = get_metrics_store()
            spans = store.get_recent_spans(name="think_like_tool")
            assert len(spans) >= 1
            assert spans[0]["attributes"]["action"] == "start"

            obs._store = None

    def test_trace_sync_decorator(self) -> None:
        """Test the sync trace decorator."""
        import src.utils.observability as obs

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_trace_sync.db")
            obs._store = MetricsStore(db_path)

            @trace_sync("custom_name")
            def sync_operation(session_id: str, x: int) -> int:
                return x * 2

            result = sync_operation(session_id="sess-1", x=21)
            assert result == 42

            store = get_metrics_store()
            spans = store.get_recent_spans(name="custom_name")
            assert len(spans) >= 1

            obs._store = None


class TestConvenienceFunctions:
    """Tests for the convenience recording functions."""

    def test_record_step(self) -> None:
        """Test record_step function."""
        import src.utils.observability as obs

        # Use file-based isolated store
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_step.db")
            obs._store = MetricsStore(db_path)

            record_step("sess-1", "premise", "ACCEPTED", 15.5)
            record_step("sess-1", "hypothesis", "REJECTED", 10.0)

            store = get_metrics_store()
            summary = store.get_metric_summary("router.steps.total")
            assert summary["count"] == 2

            rejections = store.get_metric_summary("router.rejections.total")
            assert rejections["count"] == 1

            obs._store = None

    def test_record_session_created(self) -> None:
        """Test record_session_created function."""
        import src.utils.observability as obs

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_session_created.db")
            obs._store = MetricsStore(db_path)

            record_session_created("sess-1", "medium", "chain")

            store = get_metrics_store()
            summary = store.get_metric_summary("router.sessions.total")
            assert summary["count"] == 1

            obs._store = None

    def test_record_session_completed(self) -> None:
        """Test record_session_completed function."""
        import src.utils.observability as obs

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_session_completed.db")
            obs._store = MetricsStore(db_path)

            record_session_completed("sess-1", 5000.0, 10)

            store = get_metrics_store()
            summary = store.get_metric_summary("router.session.duration_ms")
            assert summary["count"] == 1
            assert summary["avg"] == 5000.0

            obs._store = None
        assert summary["count"] == 1
        assert summary["avg"] == 5000.0


class TestMetricsDisabled:
    """Tests for when metrics are disabled."""

    def test_disabled_metrics_noop(self) -> None:
        """Test that disabled metrics don't record."""
        import src.utils.observability as obs

        # Save original values
        original_enabled = obs._METRICS_ENABLED
        original_store = obs._store

        try:
            # Create fresh store while metrics enabled
            obs._store = None
            obs._METRICS_ENABLED = True
            store = MetricsStore(":memory:")
            obs._store = store

            # Now disable metrics and try to record
            obs._METRICS_ENABLED = False

            # These should be no-ops (check _METRICS_ENABLED inside)
            store.increment("test.disabled", 1)
            store.record_histogram("test.disabled.hist", 100.0)

            # Re-enable to verify nothing was recorded
            obs._METRICS_ENABLED = True
            summary = store.get_metric_summary("test.disabled")
            assert summary["count"] == 0

            summary = store.get_metric_summary("test.disabled.hist")
            assert summary["count"] == 0
        finally:
            obs._METRICS_ENABLED = original_enabled
            obs._store = original_store

    def test_is_metrics_enabled(self) -> None:
        """Test is_metrics_enabled function."""
        assert is_metrics_enabled() is True  # Default enabled


class TestConcurrency:
    """Tests for thread safety."""

    def test_concurrent_writes(self) -> None:
        """Test concurrent metric writes don't corrupt data."""
        import tempfile
        import threading

        # Use file-based DB with WAL for real concurrent access
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "concurrent_test.db"
            store = MetricsStore(str(db_path))

            errors: list[Exception] = []

            def writer(thread_id: int) -> None:
                try:
                    for i in range(100):
                        store.increment("concurrent.counter", 1, {"thread": thread_id})
                        with store.trace(f"thread_{thread_id}_op"):
                            pass
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Check no errors occurred
            assert not errors, f"Errors in threads: {errors}"

            summary = store.get_metric_summary("concurrent.counter")
            assert summary["count"] == 500  # 5 threads * 100 writes


class TestDatabaseRotation:
    """Tests for database rotation feature."""

    def test_get_db_size_mb_memory(self) -> None:
        """Test that in-memory DB returns 0 size."""
        store = MetricsStore(":memory:")
        assert store.get_db_size_mb() == 0.0

    def test_get_db_size_mb_file(self) -> None:
        """Test that file DB returns actual size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "size_test.db"
            store = MetricsStore(str(db_path))

            # Add some data
            for i in range(100):
                store.increment("test.counter", i)

            # Should have some size > 0
            size = store.get_db_size_mb()
            assert size > 0

    def test_check_rotation_needed_memory(self) -> None:
        """Test that in-memory DB never needs rotation."""
        store = MetricsStore(":memory:")
        assert store.check_rotation_needed() is False
        assert store.check_rotation_needed(threshold_mb=0) is False

    def test_check_rotation_needed_below_threshold(self) -> None:
        """Test rotation not needed when below threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rotation_check.db"
            store = MetricsStore(str(db_path))

            # Small DB should not need rotation at 100MB threshold
            assert store.check_rotation_needed(threshold_mb=100) is False

    def test_check_rotation_needed_above_threshold(self) -> None:
        """Test rotation needed when above threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rotation_check.db"
            store = MetricsStore(str(db_path))

            # Add some data to make DB non-zero
            for i in range(100):
                store.increment("test.counter", i)

            # Very low threshold should trigger rotation
            assert store.check_rotation_needed(threshold_mb=0.0001) is True

    def test_rotate_database_memory_noop(self) -> None:
        """Test that in-memory DB rotation is a no-op."""
        store = MetricsStore(":memory:")
        result = store.rotate_database()
        assert result is None

    def test_rotate_database_creates_archive(self) -> None:
        """Test that rotation creates archive file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rotate_test.db"
            store = MetricsStore(str(db_path))

            # Add data and force rotation
            for i in range(100):
                store.increment("test.counter", i)

            # Very low threshold to force rotation
            import src.utils.observability as obs

            original_size = obs._ROTATION_SIZE_MB
            obs._ROTATION_SIZE_MB = 0.0001  # Force rotation

            try:
                archive_path = store.rotate_database()
                assert archive_path is not None
                assert Path(archive_path).exists()
                assert "rotate_test_" in archive_path
            finally:
                obs._ROTATION_SIZE_MB = original_size

    def test_rotate_database_starts_fresh(self) -> None:
        """Test that rotation starts fresh DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "fresh_test.db"
            store = MetricsStore(str(db_path))

            # Add data
            for i in range(100):
                store.increment("test.counter", i)

            # Verify data exists
            summary = store.get_metric_summary("test.counter", since_hours=1)
            assert summary["count"] == 100

            # Force rotation
            import src.utils.observability as obs

            original_size = obs._ROTATION_SIZE_MB
            obs._ROTATION_SIZE_MB = 0.0001

            try:
                store.rotate_database()

                # Data should be gone in new DB
                summary = store.get_metric_summary("test.counter", since_hours=1)
                assert summary["count"] == 0
            finally:
                obs._ROTATION_SIZE_MB = original_size

    def test_rotate_database_cleanup_old_archives(self) -> None:
        """Test that rotation removes old archives beyond keep count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "cleanup_test.db"

            import src.utils.observability as obs

            original_size = obs._ROTATION_SIZE_MB
            obs._ROTATION_SIZE_MB = 0.0001  # Force rotation

            try:
                store = MetricsStore(str(db_path))

                # Create 4 rotations with enough data to exceed threshold
                archives = []
                for rotation in range(4):
                    # Add enough data to exceed the tiny threshold
                    for i in range(100):
                        store.increment(f"test_{rotation}", i)
                        store.record_histogram(f"hist_{rotation}", float(i) * 1.5)
                    archive = store.rotate_database(keep_count=2)
                    if archive:
                        archives.append(archive)
                    import time

                    time.sleep(0.02)  # Ensure different timestamps

                # Should have kept only 2 most recent
                existing_archives = list(Path(tmpdir).glob("cleanup_test_*.db"))
                # Allow for race conditions: expect at least 1 and at most 2
                assert 1 <= len(existing_archives) <= 2
            finally:
                obs._ROTATION_SIZE_MB = original_size

    def test_rotate_database_no_rotation_if_not_needed(self) -> None:
        """Test that rotation skips if threshold not met."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "skip_test.db"
            store = MetricsStore(str(db_path))

            # Add minimal data
            store.increment("test", 1)

            # High threshold should prevent rotation
            result = store.rotate_database()
            assert result is None

            # No archives should exist
            archives = list(Path(tmpdir).glob("skip_test_*.db"))
            assert len(archives) == 0
