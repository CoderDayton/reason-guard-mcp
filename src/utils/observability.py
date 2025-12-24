"""Lightweight SQLite-based observability for Reason Guard MCP.

Zero external dependencies beyond stdlib + sqlite. Provides:
- Tool call tracing with timing
- Metrics (counters, histograms)
- Session tracking
- Query interface for analysis

Usage:
    from src.utils.observability import trace_tool, get_metrics_store

    @trace_tool
    async def my_tool(arg1: str) -> str:
        ...  # Automatically traced

    # Or manual:
    store = get_metrics_store()
    with store.trace("my_operation", session_id="abc") as span:
        span.set_attribute("key", "value")
        ...

Environment Variables:
    REASONGUARD_METRICS_DB - SQLite path (default: :memory:, use file for persistence)
    REASONGUARD_METRICS_ENABLED - Set to "false" to disable (default: true)
    REASONGUARD_METRICS_RETENTION_HOURS - Hours to retain data (default: 24)
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

# Configuration
_METRICS_ENABLED = os.environ.get("REASONGUARD_METRICS_ENABLED", "true").lower() != "false"
_METRICS_DB = os.environ.get("REASONGUARD_METRICS_DB", ":memory:")
_RETENTION_HOURS = int(os.environ.get("REASONGUARD_METRICS_RETENTION_HOURS", "24"))
_ROTATION_SIZE_MB = int(os.environ.get("REASONGUARD_METRICS_ROTATION_MB", "100"))
_ROTATION_KEEP_COUNT = int(os.environ.get("REASONGUARD_METRICS_ROTATION_KEEP", "3"))

# Type vars for decorator
P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class Span:
    """A trace span representing a unit of work."""

    span_id: str
    name: str
    start_time: float
    end_time: float | None = None
    parent_id: str | None = None
    session_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    error: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self.events.append({"name": name, "timestamp": time.time(), "attributes": attributes or {}})

    def set_status(self, status: str, error: str | None = None) -> None:
        """Set span status."""
        self.status = status
        self.error = error

    def record_exception(self, exc: Exception) -> None:
        """Record an exception."""
        self.status = "ERROR"
        self.error = f"{type(exc).__name__}: {exc}"
        self.add_event("exception", {"type": type(exc).__name__, "message": str(exc)})

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000


class MetricsStore:
    """SQLite-backed metrics and tracing store.

    Thread-safe with connection pooling per thread.
    Supports automatic rotation when DB exceeds size threshold.
    """

    _instance: MetricsStore | None = None
    _lock = threading.Lock()

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize the metrics store.

        Args:
            db_path: Path to SQLite database, or ":memory:" for in-memory.

        """
        self._db_path = db_path
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False
        self._rotation_lock = threading.Lock()
        # For in-memory DBs with multi-thread access, use shared cache
        self._is_shared_memory = False

        # Validate path if not memory
        if db_path != ":memory:":
            # Security: prevent path traversal BEFORE resolving
            if ".." in db_path:
                raise ValueError("Invalid database path: path traversal not allowed")
            path = Path(db_path).resolve()
            # Double-check after resolve
            if ".." in str(path):
                raise ValueError("Invalid database path")
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(path)
        else:
            # Use shared cache for in-memory DBs so all threads see same data
            self._db_path = "file::memory:?cache=shared"
            self._is_shared_memory = True

        self._ensure_schema()

    @classmethod
    def get_instance(cls, db_path: str | None = None) -> MetricsStore:
        """Get or create the singleton metrics store."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path or _METRICS_DB)
        return cls._instance

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            # For shared memory, use URI mode
            if self._is_shared_memory:
                self._local.conn = sqlite3.connect(
                    self._db_path,
                    uri=True,
                    check_same_thread=False,
                    timeout=30.0,
                )
            else:
                self._local.conn = sqlite3.connect(
                    self._db_path,
                    check_same_thread=False,
                    timeout=30.0,
                )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL for better concurrency (file-based only, not shared memory)
            if not self._is_shared_memory and self._db_path != ":memory:":
                self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        conn: sqlite3.Connection = self._local.conn  # type: ignore[assignment]
        return conn

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database schema on a connection."""
        conn.executescript(
            """
            -- Spans table for tracing
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                parent_id TEXT,
                session_id TEXT,
                start_time REAL NOT NULL,
                end_time REAL,
                duration_ms REAL,
                status TEXT DEFAULT 'OK',
                error TEXT,
                attributes TEXT,  -- JSON
                events TEXT,      -- JSON
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Metrics table for counters/histograms
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,  -- counter, histogram
                value REAL NOT NULL,
                attributes TEXT,     -- JSON
                timestamp REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Sessions table for session tracking
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                problem TEXT,
                mode TEXT,
                complexity TEXT,
                start_time REAL NOT NULL,
                end_time REAL,
                status TEXT DEFAULT 'active',
                total_steps INTEGER DEFAULT 0,
                metadata TEXT,  -- JSON
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_spans_session ON spans(session_id);
            CREATE INDEX IF NOT EXISTS idx_spans_name ON spans(name);
            CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time);
            CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
            """
        )
        conn.commit()

    def _ensure_schema(self) -> None:
        """Create database schema if needed."""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            conn = self._get_connection()
            self._create_schema(conn)
            self._initialized = True

    @contextmanager
    def trace(
        self,
        name: str,
        session_id: str | None = None,
        parent_id: str | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager for tracing an operation.

        Args:
            name: Operation name (e.g., "think", "submit_step")
            session_id: Optional session ID to associate with
            parent_id: Optional parent span ID

        Yields:
            Span object for adding attributes/events

        """
        span = Span(
            span_id=uuid.uuid4().hex[:16],
            name=name,
            start_time=time.time(),
            session_id=session_id,
            parent_id=parent_id,
        )

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end_time = time.time()
            self._save_span(span)

    def _save_span(self, span: Span) -> None:
        """Save a span to the database."""
        if not _METRICS_ENABLED:
            return

        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO spans (
                span_id, name, parent_id, session_id, start_time, end_time,
                duration_ms, status, error, attributes, events
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                span.span_id,
                span.name,
                span.parent_id,
                span.session_id,
                span.start_time,
                span.end_time,
                span.duration_ms,
                span.status,
                span.error,
                json.dumps(span.attributes),
                json.dumps(span.events),
            ),
        )
        conn.commit()

    def increment(
        self,
        name: str,
        value: int = 1,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Increment a counter metric."""
        if not _METRICS_ENABLED:
            return

        conn = self._get_connection()
        conn.execute(
            "INSERT INTO metrics (name, type, value, attributes, timestamp) VALUES (?, ?, ?, ?, ?)",
            (name, "counter", value, json.dumps(attributes or {}), time.time()),
        )
        conn.commit()

    def record_histogram(
        self,
        name: str,
        value: float,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record a histogram value."""
        if not _METRICS_ENABLED:
            return

        conn = self._get_connection()
        conn.execute(
            "INSERT INTO metrics (name, type, value, attributes, timestamp) VALUES (?, ?, ?, ?, ?)",
            (name, "histogram", value, json.dumps(attributes or {}), time.time()),
        )
        conn.commit()

    def record_session_start(
        self,
        session_id: str,
        problem: str,
        mode: str,
        complexity: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record session start."""
        if not _METRICS_ENABLED:
            return

        conn = self._get_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions
            (session_id, problem, mode, complexity, start_time, status, metadata)
            VALUES (?, ?, ?, ?, ?, 'active', ?)
            """,
            (
                session_id,
                problem[:1000],  # Truncate for storage
                mode,
                complexity,
                time.time(),
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        self.increment("sessions.started", attributes={"mode": mode})

    def record_session_end(
        self,
        session_id: str,
        status: str = "completed",
        total_steps: int = 0,
    ) -> None:
        """Record session end."""
        if not _METRICS_ENABLED:
            return

        conn = self._get_connection()
        now = time.time()

        # Get session start time for duration calculation
        row = conn.execute(
            "SELECT start_time, mode FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        if row:
            duration_ms = (now - row["start_time"]) * 1000
            self.record_histogram(
                "session.duration_ms",
                duration_ms,
                {"session_id": session_id, "mode": row["mode"]},
            )

        conn.execute(
            "UPDATE sessions SET end_time = ?, status = ?, total_steps = ? WHERE session_id = ?",
            (now, status, total_steps, session_id),
        )
        conn.commit()
        self.increment("sessions.completed", attributes={"status": status})

    def cleanup_old_data(self, retention_hours: int | None = None) -> int:
        """Remove data older than retention period.

        Returns:
            Number of rows deleted.

        """
        hours = retention_hours if retention_hours is not None else _RETENTION_HOURS
        cutoff = time.time() - (hours * 3600)

        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM spans WHERE start_time < ?", (cutoff,))
        spans_deleted = cursor.rowcount

        cursor = conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff,))
        metrics_deleted = cursor.rowcount

        cursor = conn.execute(
            "DELETE FROM sessions WHERE start_time < ? AND status != 'active'",
            (cutoff,),
        )
        sessions_deleted = cursor.rowcount

        conn.commit()

        # Vacuum if significant cleanup
        total = spans_deleted + metrics_deleted + sessions_deleted
        if total > 1000:
            conn.execute("VACUUM")

        return total

    def get_db_size_mb(self) -> float:
        """Get current database file size in MB.

        Returns:
            Size in MB, or 0.0 for in-memory databases.

        """
        if self._is_shared_memory or self._db_path == ":memory:":
            return 0.0

        try:
            path = Path(self._db_path)
            if path.exists():
                return path.stat().st_size / (1024 * 1024)
        except OSError:
            pass
        return 0.0

    def check_rotation_needed(self, threshold_mb: float | None = None) -> bool:
        """Check if database rotation is needed.

        Args:
            threshold_mb: Size threshold in MB (default: REASONGUARD_METRICS_ROTATION_MB)

        Returns:
            True if rotation is needed.

        """
        if self._is_shared_memory or self._db_path == ":memory:":
            return False

        threshold = threshold_mb if threshold_mb is not None else _ROTATION_SIZE_MB
        return self.get_db_size_mb() >= threshold

    def rotate_database(self, keep_count: int | None = None) -> str | None:
        """Rotate database if size threshold exceeded.

        Archives current DB with timestamp, starts fresh, removes old archives.

        Args:
            keep_count: Number of archives to keep (default: REASONGUARD_METRICS_ROTATION_KEEP)

        Returns:
            Path to archive file if rotated, None otherwise.

        """
        if self._is_shared_memory or self._db_path == ":memory:":
            return None

        with self._rotation_lock:
            if not self.check_rotation_needed():
                return None

            keep = keep_count if keep_count is not None else _ROTATION_KEEP_COUNT
            db_path = Path(self._db_path)

            # Generate archive name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{db_path.stem}_{timestamp}{db_path.suffix}"
            archive_path = db_path.parent / archive_name

            # Close all connections in current thread
            if hasattr(self._local, "conn") and self._local.conn is not None:
                with contextlib.suppress(Exception):
                    self._local.conn.close()
                self._local.conn = None

            # Rename current DB to archive
            try:
                db_path.rename(archive_path)
            except OSError:
                # If rename fails, try copy+delete
                import shutil

                shutil.copy2(db_path, archive_path)
                db_path.unlink()

            # Also move WAL/SHM files if they exist
            for suffix in ["-wal", "-shm"]:
                wal_path = Path(str(db_path) + suffix)
                if wal_path.exists():
                    with contextlib.suppress(OSError):
                        wal_path.unlink()

            # Reset initialization state to recreate schema
            self._initialized = False
            self._ensure_schema()

            # Cleanup old archives
            self._cleanup_old_archives(db_path, keep)

            return str(archive_path)

    def _cleanup_old_archives(self, db_path: Path, keep_count: int) -> None:
        """Remove old archive files, keeping only the most recent ones.

        Args:
            db_path: Original database path
            keep_count: Number of archives to keep

        """
        if keep_count < 0:
            return

        # Find all archive files matching pattern: name_YYYYMMDD_HHMMSS.db
        pattern = f"{db_path.stem}_*{db_path.suffix}"
        archives = sorted(
            db_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        )

        # Remove oldest archives beyond keep_count
        for archive in archives[keep_count:]:
            with contextlib.suppress(OSError):
                archive.unlink()

    # --- Query Methods ---

    def get_recent_spans(
        self,
        name: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
        since_hours: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Get recent spans with optional filtering."""
        conn = self._get_connection()
        cutoff = time.time() - (since_hours * 3600)

        query = "SELECT * FROM spans WHERE start_time > ?"
        params: list[Any] = [cutoff]

        if name:
            query += " AND name = ?"
            params.append(name)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [
            {
                **dict(row),
                "attributes": json.loads(row["attributes"] or "{}"),
                "events": json.loads(row["events"] or "[]"),
            }
            for row in rows
        ]

    def get_metric_summary(
        self,
        name: str,
        since_hours: float = 1.0,
    ) -> dict[str, Any]:
        """Get summary statistics for a metric."""
        conn = self._get_connection()
        cutoff = time.time() - (since_hours * 3600)

        row = conn.execute(
            """
            SELECT
                COUNT(*) as count,
                SUM(value) as total,
                AVG(value) as avg,
                MIN(value) as min,
                MAX(value) as max
            FROM metrics
            WHERE name = ? AND timestamp > ?
            """,
            (name, cutoff),
        ).fetchone()

        return dict(row) if row else {"count": 0}

    def get_tool_stats(self, since_hours: float = 1.0) -> dict[str, Any]:
        """Get statistics for all traced tools."""
        conn = self._get_connection()
        cutoff = time.time() - (since_hours * 3600)

        rows = conn.execute(
            """
            SELECT
                name,
                COUNT(*) as calls,
                AVG(duration_ms) as avg_duration_ms,
                MIN(duration_ms) as min_duration_ms,
                MAX(duration_ms) as max_duration_ms,
                SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) as errors
            FROM spans
            WHERE start_time > ?
            GROUP BY name
            ORDER BY calls DESC
            """,
            (cutoff,),
        ).fetchall()

        return {row["name"]: dict(row) for row in rows}

    def get_session_stats(self, since_hours: float = 24.0) -> dict[str, Any]:
        """Get session statistics."""
        conn = self._get_connection()
        cutoff = time.time() - (since_hours * 3600)

        row = conn.execute(
            """
            SELECT
                COUNT(*) as total_sessions,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                AVG(total_steps) as avg_steps,
                AVG(CASE WHEN end_time IS NOT NULL THEN (end_time - start_time) * 1000 END) as avg_duration_ms
            FROM sessions
            WHERE start_time > ?
            """,
            (cutoff,),
        ).fetchone()

        mode_rows = conn.execute(
            """
            SELECT mode, COUNT(*) as count
            FROM sessions
            WHERE start_time > ?
            GROUP BY mode
            """,
            (cutoff,),
        ).fetchall()

        return {
            **dict(row),
            "by_mode": {r["mode"]: r["count"] for r in mode_rows},
        }


# --- Singleton access ---

_store: MetricsStore | None = None


def get_metrics_store() -> MetricsStore:
    """Get the global metrics store instance."""
    global _store
    if _store is None:
        _store = MetricsStore.get_instance()
    return _store


# --- Decorators ---


def trace_tool(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Decorator to trace an async tool function.

    Automatically captures:
    - Tool name
    - Duration
    - Session ID (if in kwargs)
    - Errors

    Usage:
        @trace_tool
        async def my_tool(session_id: str, ...) -> str:
            ...

    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not _METRICS_ENABLED:
            return await func(*args, **kwargs)

        store = get_metrics_store()
        tool_name = func.__name__
        session_id_raw = kwargs.get("session_id")
        session_id: str | None = str(session_id_raw) if session_id_raw is not None else None

        with store.trace(tool_name, session_id=session_id) as span:
            # Add common attributes
            span.set_attribute("tool", tool_name)
            if session_id:
                span.set_attribute("session_id", session_id)

            # Extract action if present (for think tool)
            action = kwargs.get("action")
            if action:
                span.set_attribute("action", action)

            try:
                result = await func(*args, **kwargs)

                # Try to extract status from result
                if isinstance(result, str):
                    try:
                        data = json.loads(result)
                        if "error" in data:
                            span.set_status("ERROR", data.get("error"))
                        elif "status" in data:
                            span.set_attribute("result_status", data["status"])
                    except json.JSONDecodeError:
                        pass

                return result

            except Exception as e:
                span.record_exception(e)
                raise

    return wrapper


def trace_sync(
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for tracing synchronous functions.

    Args:
        name: Optional custom name (defaults to function name)

    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not _METRICS_ENABLED:
                return func(*args, **kwargs)

            store = get_metrics_store()
            span_name = name or func.__name__
            session_id_raw = kwargs.get("session_id")
            session_id: str | None = str(session_id_raw) if session_id_raw is not None else None

            with store.trace(span_name, session_id=session_id) as span:
                span.set_attribute("function", func.__name__)
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


# --- Convenience functions for recording metrics ---


def record_step(
    session_id: str,
    step_type: str,
    status: str,
    duration_ms: float | None = None,
) -> None:
    """Record a reasoning step."""
    if not _METRICS_ENABLED:
        return

    store = get_metrics_store()
    attrs = {"session_id": session_id, "step_type": step_type, "status": status}
    store.increment("router.steps.total", attributes=attrs)

    if status == "REJECTED":
        store.increment("router.rejections.total", attributes=attrs)

    if duration_ms is not None:
        store.record_histogram("router.step.duration_ms", duration_ms, attrs)


def record_session_created(session_id: str, complexity: str, mode: str = "unknown") -> None:
    """Record session creation."""
    if not _METRICS_ENABLED:
        return

    store = get_metrics_store()
    store.increment("router.sessions.total", attributes={"complexity": complexity, "mode": mode})


def record_session_completed(session_id: str, duration_ms: float, steps: int) -> None:
    """Record session completion."""
    if not _METRICS_ENABLED:
        return

    store = get_metrics_store()
    store.record_histogram(
        "router.session.duration_ms",
        duration_ms,
        {"session_id": session_id, "total_steps": steps},
    )


def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled."""
    return _METRICS_ENABLED


# --- Legacy compatibility (for existing OTel-style code) ---


# No-op versions for when OTel code expects these
class NoopSpan:
    """No-op span for OTel compatibility."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        pass

    def __enter__(self) -> NoopSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class NoopTracer:
    """No-op tracer for OTel compatibility."""

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Generator[NoopSpan, None, None]:
        yield NoopSpan()

    def start_span(self, name: str, **kwargs: Any) -> NoopSpan:
        return NoopSpan()


# Provide OTel-compatible interface
tracer = NoopTracer()
meter = None  # Not needed with SQLite approach


def is_otel_available() -> bool:
    """Check if OpenTelemetry is available (always False, using SQLite instead)."""
    return False


def setup_telemetry(
    service_name: str = "reasonguard-router",
    otlp_endpoint: str | None = None,
) -> bool:
    """Setup telemetry - initializes SQLite store.

    Note: OTel parameters are ignored; using SQLite-based metrics.

    Returns:
        True if metrics are enabled.

    """
    if not _METRICS_ENABLED:
        return False

    # Initialize the store
    get_metrics_store()
    return True
