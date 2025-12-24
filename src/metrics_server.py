"""Lightweight metrics HTTP server for Reason Guard observability.

Runs separately from the MCP server, exposing metrics via REST endpoints.
Uses Starlette (already bundled with fastmcp) for minimal overhead.

Usage:
    # Start metrics server
    python -m src.metrics_server

    # Or with custom port
    METRICS_PORT=9090 python -m src.metrics_server

    # Endpoints:
    # GET /health     - Health check
    # GET /metrics    - All metrics summary
    # GET /sessions   - Session statistics
    # GET /tools      - Tool call statistics
    # POST /rotate    - Trigger DB rotation
    # POST /cleanup   - Trigger data cleanup

Environment Variables:
    METRICS_HOST: Bind address (default: 127.0.0.1)
    METRICS_PORT: Port (default: 9090)
    REASONGUARD_METRICS_DB: Path to metrics database
"""

from __future__ import annotations

import json
import os
import time

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from src.utils.observability import MetricsStore, get_metrics_store, is_metrics_enabled

# Configuration
_HOST = os.environ.get("METRICS_HOST", "127.0.0.1")
_PORT = int(os.environ.get("METRICS_PORT", "9090"))

# Server start time for uptime calculation
_START_TIME = time.time()


def _get_store() -> MetricsStore | None:
    """Get metrics store if enabled."""
    if not is_metrics_enabled():
        return None
    return get_metrics_store()


async def health(request: Request) -> Response:
    """Health check endpoint.

    Returns:
        200 with status info if healthy
        503 if metrics disabled

    """
    store = _get_store()
    uptime = time.time() - _START_TIME

    if store is None:
        return JSONResponse(
            {"status": "degraded", "reason": "metrics_disabled", "uptime_seconds": uptime},
            status_code=503,
        )

    return JSONResponse(
        {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "metrics_enabled": True,
            "db_size_mb": round(store.get_db_size_mb(), 2),
        }
    )


async def metrics(request: Request) -> Response:
    """Get all metrics summary.

    Query params:
        since_hours: Time window (default: 1)

    """
    store = _get_store()
    if store is None:
        return JSONResponse({"error": "metrics_disabled"}, status_code=503)

    since_hours = float(request.query_params.get("since_hours", "1"))

    tool_stats = store.get_tool_stats(since_hours=since_hours)
    session_stats = store.get_session_stats(since_hours=since_hours)

    return JSONResponse(
        {
            "timestamp": time.time(),
            "since_hours": since_hours,
            "tools": tool_stats,
            "sessions": session_stats,
            "db_size_mb": round(store.get_db_size_mb(), 2),
        }
    )


async def sessions(request: Request) -> Response:
    """Get session statistics.

    Query params:
        since_hours: Time window (default: 24)

    """
    store = _get_store()
    if store is None:
        return JSONResponse({"error": "metrics_disabled"}, status_code=503)

    since_hours = float(request.query_params.get("since_hours", "24"))
    stats = store.get_session_stats(since_hours=since_hours)

    return JSONResponse(
        {
            "timestamp": time.time(),
            "since_hours": since_hours,
            **stats,
        }
    )


async def tools(request: Request) -> Response:
    """Get tool call statistics.

    Query params:
        since_hours: Time window (default: 1)

    """
    store = _get_store()
    if store is None:
        return JSONResponse({"error": "metrics_disabled"}, status_code=503)

    since_hours = float(request.query_params.get("since_hours", "1"))
    stats = store.get_tool_stats(since_hours=since_hours)

    return JSONResponse(
        {
            "timestamp": time.time(),
            "since_hours": since_hours,
            "tools": stats,
        }
    )


async def spans(request: Request) -> Response:
    """Get recent spans/traces.

    Query params:
        name: Filter by operation name
        session_id: Filter by session
        limit: Max results (default: 100)
        since_hours: Time window (default: 1)

    """
    store = _get_store()
    if store is None:
        return JSONResponse({"error": "metrics_disabled"}, status_code=503)

    name = request.query_params.get("name")
    session_id = request.query_params.get("session_id")
    limit = int(request.query_params.get("limit", "100"))
    since_hours = float(request.query_params.get("since_hours", "1"))

    spans_data = store.get_recent_spans(
        name=name,
        session_id=session_id,
        limit=limit,
        since_hours=since_hours,
    )

    return JSONResponse(
        {
            "timestamp": time.time(),
            "count": len(spans_data),
            "spans": spans_data,
        }
    )


async def rotate(request: Request) -> Response:
    """Trigger database rotation.

    POST body (optional):
        keep_count: Number of archives to keep

    """
    store = _get_store()
    if store is None:
        return JSONResponse({"error": "metrics_disabled"}, status_code=503)

    keep_count = None
    if request.method == "POST":
        try:
            body = await request.json()
            keep_count = body.get("keep_count")
        except (json.JSONDecodeError, ValueError):
            pass

    archive_path = store.rotate_database(keep_count=keep_count)

    if archive_path:
        return JSONResponse(
            {
                "status": "rotated",
                "archive_path": archive_path,
                "new_db_size_mb": round(store.get_db_size_mb(), 2),
            }
        )
    else:
        return JSONResponse(
            {
                "status": "skipped",
                "reason": "threshold_not_met",
                "current_size_mb": round(store.get_db_size_mb(), 2),
            }
        )


async def cleanup(request: Request) -> Response:
    """Trigger data cleanup.

    POST body (optional):
        retention_hours: Hours of data to keep

    """
    store = _get_store()
    if store is None:
        return JSONResponse({"error": "metrics_disabled"}, status_code=503)

    retention_hours = None
    if request.method == "POST":
        try:
            body = await request.json()
            retention_hours = body.get("retention_hours")
        except (json.JSONDecodeError, ValueError):
            pass

    deleted = store.cleanup_old_data(retention_hours=retention_hours)

    return JSONResponse(
        {
            "status": "completed",
            "rows_deleted": deleted,
            "db_size_mb": round(store.get_db_size_mb(), 2),
        }
    )


# Routes
routes = [
    Route("/health", health, methods=["GET"]),
    Route("/metrics", metrics, methods=["GET"]),
    Route("/sessions", sessions, methods=["GET"]),
    Route("/tools", tools, methods=["GET"]),
    Route("/spans", spans, methods=["GET"]),
    Route("/rotate", rotate, methods=["POST"]),
    Route("/cleanup", cleanup, methods=["POST"]),
]

# Application
app = Starlette(routes=routes)


def main() -> None:
    """Run the metrics server."""
    import uvicorn

    print(f"Starting metrics server on {_HOST}:{_PORT}")
    print(f"Metrics enabled: {is_metrics_enabled()}")

    if is_metrics_enabled():
        store = get_metrics_store()
        print(f"Database size: {store.get_db_size_mb():.2f} MB")

    uvicorn.run(
        app,
        host=_HOST,
        port=_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
