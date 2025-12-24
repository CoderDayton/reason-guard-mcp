"""Structured logging utilities for Reason Guard MCP.

Provides a consistent logging interface with:
- Structured JSON logging for production
- Human-readable format for development
- Context injection for trace IDs and request tracking
- Log level configuration from environment/config
- Automatic redaction of sensitive data
"""

from __future__ import annotations

import json
import os
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

# Context variables for request tracking
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_tool_name: ContextVar[str | None] = ContextVar("tool_name", default=None)


class LogFormat(str, Enum):
    """Supported log output formats."""

    JSON = "json"
    TEXT = "text"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Sensitive keys to redact from logs
SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "password",
        "secret",
        "token",
        "authorization",
        "auth",
        "credential",
        "private_key",
        "privatekey",
        "openai_api_key",
        "anthropic_api_key",
    }
)


def redact_sensitive(data: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Recursively redact sensitive values from a dictionary.

    Args:
        data: Dictionary to redact.
        depth: Current recursion depth (prevents infinite recursion).

    Returns:
        Dictionary with sensitive values replaced with "[REDACTED]".

    """
    if depth > 10:
        return data

    result: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower().replace("_", "").replace("-", "")
        if any(sensitive in key_lower for sensitive in SENSITIVE_KEYS):
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = redact_sensitive(value, depth + 1)
        elif isinstance(value, list):
            result[key] = [
                redact_sensitive(item, depth + 1) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def json_serializer(record: Record) -> str:
    """Serialize log record to JSON format.

    Args:
        record: Loguru record dictionary.

    Returns:
        JSON string representation of the log entry.

    """
    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }

    # Add context variables if set
    if trace_id := _trace_id.get():
        log_entry["trace_id"] = trace_id
    if request_id := _request_id.get():
        log_entry["request_id"] = request_id
    if session_id := _session_id.get():
        log_entry["session_id"] = session_id
    if tool_name := _tool_name.get():
        log_entry["tool"] = tool_name

    # Add extra data if present
    if record.get("extra"):
        extra = redact_sensitive(dict(record["extra"]))
        log_entry["extra"] = extra

    # Add exception info if present
    if record["exception"]:
        exc_info = record["exception"]
        log_entry["exception"] = {
            "type": exc_info.type.__name__ if exc_info.type else None,
            "value": str(exc_info.value) if exc_info.value else None,
            "traceback": exc_info.traceback is not None,
        }

    return json.dumps(log_entry, default=str, ensure_ascii=False)


def text_format(record: Record) -> str:
    """Format log record as human-readable text.

    Args:
        record: Loguru record dictionary.

    Returns:
        Formatted string for console output.

    """
    # Build context prefix
    context_parts = []
    if trace_id := _trace_id.get():
        context_parts.append(f"trace={trace_id[:8]}")
    if session_id := _session_id.get():
        context_parts.append(f"sess={session_id[:8]}")
    if tool_name := _tool_name.get():
        context_parts.append(f"tool={tool_name}")

    context = f"[{' '.join(context_parts)}] " if context_parts else ""

    return (
        f"<green>{record['time']:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        f"<level>{record['level'].name:8}</level> | "
        f"<cyan>{record['name']}</cyan>:<cyan>{record['function']}</cyan>:"
        f"<cyan>{record['line']}</cyan> | "
        f"{context}"
        f"<level>{record['message']}</level>\n"
    )


class StructuredLogger:
    """Structured logging wrapper with context injection.

    Provides a consistent interface for logging with automatic
    context injection and format configuration.

    Example:
        log = StructuredLogger("my_module")
        log.info("Processing request", user_id=123, action="query")

        # With context
        with log.context(trace_id="abc123", tool_name="compress"):
            log.info("Starting compression")

    """

    def __init__(
        self,
        name: str,
        level: LogLevel | str = LogLevel.INFO,
        log_format: LogFormat | str = LogFormat.TEXT,
        log_file: str | Path | None = None,
    ) -> None:
        """Initialize structured logger.

        Args:
            name: Logger name (usually module name).
            level: Minimum log level.
            log_format: Output format (json or text).
            log_file: Optional file path for log output.

        """
        self.name = name
        self.level = LogLevel(level) if isinstance(level, str) else level
        self.log_format = LogFormat(log_format) if isinstance(log_format, str) else log_format

        # Configure loguru
        self._configure_logger(log_file)

    def _configure_logger(self, log_file: str | Path | None = None) -> None:
        """Configure loguru with appropriate handlers."""
        # Remove default handler
        logger.remove()

        # Add console handler
        if self.log_format == LogFormat.JSON:
            # Use serialize=True for native JSON output
            logger.add(
                sys.stderr,
                format="{message}",
                level=self.level.value,
                serialize=True,
            )
        else:
            # Use standard loguru format string for text output
            format_str = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
            logger.add(
                sys.stderr,
                format=format_str,
                level=self.level.value,
                colorize=True,
            )

        # Add file handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                log_path,
                format="{message}",
                level=self.level.value,
                serialize=True,
                rotation="100 MB",
                retention="7 days",
                compression="gz",
            )

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Internal logging method with context injection."""
        # Bind extra context
        bound_logger = logger.bind(**kwargs)
        getattr(bound_logger.opt(depth=2), level)(message)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log("error", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log("critical", message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        logger.opt(depth=1, exception=True).error(message, **kwargs)

    class _ContextManager:
        """Context manager for scoped logging context."""

        def __init__(
            self,
            trace_id: str | None = None,
            request_id: str | None = None,
            session_id: str | None = None,
            tool_name: str | None = None,
        ) -> None:
            self.trace_id = trace_id
            self.request_id = request_id
            self.session_id = session_id
            self.tool_name = tool_name
            self._tokens: list[Any] = []

        def __enter__(self) -> StructuredLogger._ContextManager:
            if self.trace_id:
                self._tokens.append(_trace_id.set(self.trace_id))
            if self.request_id:
                self._tokens.append(_request_id.set(self.request_id))
            if self.session_id:
                self._tokens.append(_session_id.set(self.session_id))
            if self.tool_name:
                self._tokens.append(_tool_name.set(self.tool_name))
            return self

        def __exit__(self, *args: Any) -> None:
            for token in self._tokens:
                # Each token knows which ContextVar it came from
                token.var.reset(token)

    def context(
        self,
        trace_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        tool_name: str | None = None,
    ) -> _ContextManager:
        """Create a context manager for scoped logging context.

        Args:
            trace_id: Trace ID for distributed tracing.
            request_id: Request ID for request tracking.
            session_id: Session ID for reasoning session correlation.
            tool_name: Name of the tool being executed.

        Returns:
            Context manager that sets the logging context.

        Example:
            with log.context(session_id="abc123", tool_name="think"):
                log.info("Processing")  # Includes session_id and tool_name

        """
        return self._ContextManager(trace_id, request_id, session_id, tool_name)


def set_trace_id(trace_id: str) -> None:
    """Set the trace ID for the current context.

    Args:
        trace_id: Trace ID string.

    """
    _trace_id.set(trace_id)


def set_request_id(request_id: str) -> None:
    """Set the request ID for the current context.

    Args:
        request_id: Request ID string.

    """
    _request_id.set(request_id)


def set_session_id(session_id: str | None) -> None:
    """Set the session ID for the current context.

    Args:
        session_id: Session ID string, or None to clear.

    """
    _session_id.set(session_id)


def get_session_id() -> str | None:
    """Get the current session ID from context.

    Returns:
        Current session ID or None if not set.

    """
    return _session_id.get()


def set_tool_name(tool_name: str) -> None:
    """Set the tool name for the current context.

    Args:
        tool_name: Tool name string.

    """
    _tool_name.set(tool_name)


def get_logger(
    name: str,
    level: LogLevel | str | None = None,
    log_format: LogFormat | str | None = None,
) -> StructuredLogger:
    """Get a configured structured logger.

    Reads configuration from environment variables if not specified:
    - LOG_LEVEL: Minimum log level (DEBUG, INFO, WARNING, ERROR)
    - LOG_FORMAT: Output format (json, text)
    - LOG_FILE: Optional file path for log output

    Args:
        name: Logger name (usually __name__).
        level: Minimum log level (default: from env or INFO).
        log_format: Output format (default: from env or TEXT).

    Returns:
        Configured StructuredLogger instance.

    Example:
        log = get_logger(__name__)
        log.info("Application started", version="1.0.0")

    """
    # Read from environment if not specified
    env_level = os.getenv("LOG_LEVEL", "INFO")
    env_format = os.getenv("LOG_FORMAT", "text")
    env_file = os.getenv("LOG_FILE")

    return StructuredLogger(
        name=name,
        level=level or LogLevel(env_level.upper()),
        log_format=log_format or LogFormat(env_format.lower()),
        log_file=env_file,
    )


# Module-level convenience logger
_default_logger: StructuredLogger | None = None


def get_default_logger() -> StructuredLogger:
    """Get the default application logger.

    Returns:
        Default StructuredLogger instance.

    """
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger("reason_guard_mcp")
    return _default_logger


# Convenience exports
log = get_default_logger
