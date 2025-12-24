"""Tests for the metrics HTTP server."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

import src.utils.observability as obs
from src.metrics_server import app
from src.utils.observability import MetricsStore


@pytest.fixture
def metrics_store():
    """Create isolated metrics store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_metrics.db")
        store = MetricsStore(db_path)
        original_store = obs._store
        obs._store = store
        yield store
        obs._store = original_store


@pytest.fixture
def client():
    """Create test client for metrics server."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200_when_enabled(self, client, metrics_store) -> None:
        """Health check returns 200 when metrics enabled."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert data["metrics_enabled"] is True

    def test_health_returns_503_when_disabled(self, client) -> None:
        """Health check returns 503 when metrics disabled."""
        with patch.object(obs, "_METRICS_ENABLED", False):
            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "degraded"
            assert data["reason"] == "metrics_disabled"


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_returns_summary(self, client, metrics_store) -> None:
        """Metrics endpoint returns tool and session stats."""
        # Add some test data
        metrics_store.increment("test.counter", 5)
        with metrics_store.trace("test_tool"):
            pass

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "tools" in data
        assert "sessions" in data
        assert "db_size_mb" in data

    def test_metrics_respects_since_hours(self, client, metrics_store) -> None:
        """Metrics endpoint respects since_hours parameter."""
        response = client.get("/metrics?since_hours=24")
        assert response.status_code == 200
        data = response.json()
        assert data["since_hours"] == 24.0

    def test_metrics_returns_503_when_disabled(self, client) -> None:
        """Metrics endpoint returns 503 when disabled."""
        with patch.object(obs, "_METRICS_ENABLED", False):
            response = client.get("/metrics")
            assert response.status_code == 503


class TestSessionsEndpoint:
    """Tests for /sessions endpoint."""

    def test_sessions_returns_stats(self, client, metrics_store) -> None:
        """Sessions endpoint returns session statistics."""
        metrics_store.record_session_start("sess-1", "test problem", "chain")
        metrics_store.record_session_end("sess-1", "completed", 5)

        response = client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "total_sessions" in data
        assert data["total_sessions"] >= 1


class TestToolsEndpoint:
    """Tests for /tools endpoint."""

    def test_tools_returns_stats(self, client, metrics_store) -> None:
        """Tools endpoint returns tool call statistics."""
        for _ in range(3):
            with metrics_store.trace("my_tool"):
                pass

        response = client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "my_tool" in data["tools"]
        assert data["tools"]["my_tool"]["calls"] == 3


class TestSpansEndpoint:
    """Tests for /spans endpoint."""

    def test_spans_returns_recent(self, client, metrics_store) -> None:
        """Spans endpoint returns recent spans."""
        with metrics_store.trace("op1", session_id="sess-1"):
            pass
        with metrics_store.trace("op2", session_id="sess-2"):
            pass

        response = client.get("/spans")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["spans"]) == 2

    def test_spans_filters_by_name(self, client, metrics_store) -> None:
        """Spans endpoint filters by operation name."""
        with metrics_store.trace("op1"):
            pass
        with metrics_store.trace("op2"):
            pass

        response = client.get("/spans?name=op1")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["spans"][0]["name"] == "op1"

    def test_spans_filters_by_session(self, client, metrics_store) -> None:
        """Spans endpoint filters by session_id."""
        with metrics_store.trace("op", session_id="sess-1"):
            pass
        with metrics_store.trace("op", session_id="sess-2"):
            pass

        response = client.get("/spans?session_id=sess-1")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["spans"][0]["session_id"] == "sess-1"


class TestRotateEndpoint:
    """Tests for /rotate endpoint."""

    def test_rotate_skips_when_below_threshold(self, client, metrics_store) -> None:
        """Rotate endpoint skips when DB below threshold."""
        response = client.post("/rotate")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "skipped"
        assert data["reason"] == "threshold_not_met"

    def test_rotate_with_custom_keep_count(self, client, metrics_store) -> None:
        """Rotate endpoint accepts custom keep_count."""
        # Force rotation by setting very low threshold
        original = obs._ROTATION_SIZE_MB
        obs._ROTATION_SIZE_MB = 0.0001

        try:
            # Add data to create non-empty DB
            for i in range(100):
                metrics_store.increment("test", i)

            response = client.post("/rotate", json={"keep_count": 5})
            assert response.status_code == 200
            data = response.json()
            # Either rotated or skipped is valid
            assert data["status"] in ("rotated", "skipped")
        finally:
            obs._ROTATION_SIZE_MB = original


class TestCleanupEndpoint:
    """Tests for /cleanup endpoint."""

    def test_cleanup_removes_old_data(self, client, metrics_store) -> None:
        """Cleanup endpoint removes old data."""
        response = client.post("/cleanup", json={"retention_hours": 0})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "rows_deleted" in data

    def test_cleanup_uses_default_retention(self, client, metrics_store) -> None:
        """Cleanup endpoint uses default retention when not specified."""
        response = client.post("/cleanup")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
