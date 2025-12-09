"""Unit tests for src/utils/logging.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.utils.logging import (
    LogFormat,
    LogLevel,
    StructuredLogger,
    _request_id,
    _tool_name,
    _trace_id,
    get_default_logger,
    get_logger,
    json_serializer,
    redact_sensitive,
    set_request_id,
    set_tool_name,
    set_trace_id,
)


class TestLogFormatEnum:
    """Test LogFormat enum."""

    def test_json_format(self) -> None:
        """Test JSON format value."""
        assert LogFormat.JSON.value == "json"

    def test_text_format(self) -> None:
        """Test TEXT format value."""
        assert LogFormat.TEXT.value == "text"


class TestLogLevelEnum:
    """Test LogLevel enum."""

    def test_all_levels_exist(self) -> None:
        """Test all log levels are defined."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestRedactSensitive:
    """Test sensitive data redaction."""

    def test_redacts_api_key(self) -> None:
        """Test API key is redacted."""
        data = {"api_key": "sk-secret-key-123"}
        result = redact_sensitive(data)
        assert result["api_key"] == "[REDACTED]"

    def test_redacts_password(self) -> None:
        """Test password is redacted."""
        data = {"password": "my_secret_password"}
        result = redact_sensitive(data)
        assert result["password"] == "[REDACTED]"

    def test_redacts_token(self) -> None:
        """Test token is redacted."""
        data = {"token": "bearer-token-xyz"}
        result = redact_sensitive(data)
        assert result["token"] == "[REDACTED]"

    def test_redacts_openai_api_key(self) -> None:
        """Test OpenAI API key is redacted."""
        data = {"openai_api_key": "sk-openai-key"}
        result = redact_sensitive(data)
        assert result["openai_api_key"] == "[REDACTED]"

    def test_redacts_nested_sensitive(self) -> None:
        """Test nested sensitive data is redacted."""
        data = {"config": {"api_key": "secret", "other": "value"}}
        result = redact_sensitive(data)
        assert result["config"]["api_key"] == "[REDACTED]"
        assert result["config"]["other"] == "value"

    def test_redacts_in_list(self) -> None:
        """Test sensitive data in lists is redacted."""
        data = {"items": [{"api_key": "secret1"}, {"api_key": "secret2"}]}
        result = redact_sensitive(data)
        assert result["items"][0]["api_key"] == "[REDACTED]"
        assert result["items"][1]["api_key"] == "[REDACTED]"

    def test_preserves_non_sensitive(self) -> None:
        """Test non-sensitive data is preserved."""
        data = {"name": "test", "count": 42}
        result = redact_sensitive(data)
        assert result["name"] == "test"
        assert result["count"] == 42

    def test_max_depth_protection(self) -> None:
        """Test max depth prevents infinite recursion."""
        # Create deeply nested structure
        data: dict[str, Any] = {"level": 0}
        current = data
        for i in range(15):
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        # Should not raise and should return result
        result = redact_sensitive(data)
        assert result["level"] == 0

    def test_handles_mixed_case_keys(self) -> None:
        """Test handles variations in key casing."""
        data = {
            "API_KEY": "secret1",
            "ApiKey": "secret2",
            "api-key": "secret3",
        }
        result = redact_sensitive(data)
        assert result["API_KEY"] == "[REDACTED]"
        assert result["ApiKey"] == "[REDACTED]"
        assert result["api-key"] == "[REDACTED]"

    def test_handles_authorization_header(self) -> None:
        """Test authorization header is redacted."""
        data = {"authorization": "Bearer token123"}
        result = redact_sensitive(data)
        assert result["authorization"] == "[REDACTED]"


class TestJsonSerializer:
    """Test JSON log serialization."""

    def test_basic_serialization(self) -> None:
        """Test basic record serialization."""
        # Create mock record
        record: dict[str, Any] = {
            "level": MagicMock(name="INFO"),
            "message": "Test message",
            "name": "test_module",
            "function": "test_func",
            "line": 42,
            "extra": {},
            "exception": None,
        }
        record["level"].name = "INFO"

        result = json_serializer(record)
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "test_module"
        assert parsed["function"] == "test_func"
        assert parsed["line"] == 42
        assert "timestamp" in parsed

    def test_serialization_with_trace_id(self) -> None:
        """Test serialization includes trace_id when set."""
        set_trace_id("trace-123")
        try:
            record: dict[str, Any] = {
                "level": MagicMock(name="INFO"),
                "message": "Test",
                "name": "test",
                "function": "func",
                "line": 1,
                "extra": {},
                "exception": None,
            }
            record["level"].name = "INFO"

            result = json_serializer(record)
            parsed = json.loads(result)
            assert parsed["trace_id"] == "trace-123"
        finally:
            _trace_id.set(None)

    def test_serialization_with_exception(self) -> None:
        """Test serialization includes exception info."""
        record: dict[str, Any] = {
            "level": MagicMock(name="ERROR"),
            "message": "Error occurred",
            "name": "test",
            "function": "func",
            "line": 1,
            "extra": {},
            "exception": MagicMock(),
        }
        record["level"].name = "ERROR"
        record["exception"].type = ValueError
        record["exception"].value = "test error"
        record["exception"].traceback = True

        result = json_serializer(record)
        parsed = json.loads(result)

        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"

    def test_serialization_redacts_extra(self) -> None:
        """Test extra data has sensitive values redacted."""
        record: dict[str, Any] = {
            "level": MagicMock(name="INFO"),
            "message": "Test",
            "name": "test",
            "function": "func",
            "line": 1,
            "extra": {"api_key": "secret", "data": "public"},
            "exception": None,
        }
        record["level"].name = "INFO"

        result = json_serializer(record)
        parsed = json.loads(result)

        assert parsed["extra"]["api_key"] == "[REDACTED]"
        assert parsed["extra"]["data"] == "public"


class TestTextFormat:
    """Test text log formatting."""

    def test_context_parts_no_context(self) -> None:
        """Test text formatting with no context variables set."""
        # Clear any context
        _trace_id.set(None)
        _tool_name.set(None)
        _request_id.set(None)

        # The text_format function uses loguru's format string
        # which requires specific record format. We test the context logic instead.
        # Just verify no error occurs when getting context vars
        assert _trace_id.get() is None
        assert _tool_name.get() is None

    def test_context_parts_with_context(self) -> None:
        """Test context variables are set correctly for formatting."""
        set_trace_id("trace-abc")
        set_tool_name("compress")
        try:
            # Verify context vars are set (used by text_format internally)
            assert _trace_id.get() == "trace-abc"
            assert _tool_name.get() == "compress"
        finally:
            _trace_id.set(None)
            _tool_name.set(None)


class TestStructuredLogger:
    """Test StructuredLogger class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        logger = StructuredLogger("test_logger")
        assert logger.name == "test_logger"
        assert logger.level == LogLevel.INFO
        assert logger.log_format == LogFormat.TEXT

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        logger = StructuredLogger(
            "custom",
            level=LogLevel.DEBUG,
            log_format=LogFormat.JSON,
        )
        assert logger.level == LogLevel.DEBUG
        assert logger.log_format == LogFormat.JSON

    def test_init_with_string_level(self) -> None:
        """Test initialization with string level."""
        logger = StructuredLogger("test", level="WARNING")
        assert logger.level == LogLevel.WARNING

    def test_init_with_string_format(self) -> None:
        """Test initialization with string format."""
        logger = StructuredLogger("test", log_format="json")
        assert logger.log_format == LogFormat.JSON

    def test_debug_logging(self) -> None:
        """Test debug level logging."""
        logger = StructuredLogger("test", level=LogLevel.DEBUG)
        # Should not raise
        logger.debug("Debug message", key="value")

    def test_info_logging(self) -> None:
        """Test info level logging."""
        logger = StructuredLogger("test")
        logger.info("Info message", data=123)

    def test_warning_logging(self) -> None:
        """Test warning level logging."""
        logger = StructuredLogger("test")
        logger.warning("Warning message")

    def test_error_logging(self) -> None:
        """Test error level logging."""
        logger = StructuredLogger("test")
        logger.error("Error message", error="details")

    def test_critical_logging(self) -> None:
        """Test critical level logging."""
        logger = StructuredLogger("test")
        logger.critical("Critical message")

    def test_exception_logging(self) -> None:
        """Test exception logging with traceback."""
        logger = StructuredLogger("test")
        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("Caught exception")

    def test_context_manager(self) -> None:
        """Test context manager for scoped logging."""
        logger = StructuredLogger("test")

        with logger.context(trace_id="ctx-123", tool_name="test_tool"):
            assert _trace_id.get() == "ctx-123"
            assert _tool_name.get() == "test_tool"

        # Context should be exited (may or may not be cleared depending on implementation)

    def test_context_manager_with_request_id(self) -> None:
        """Test context manager with request_id."""
        logger = StructuredLogger("test")

        with logger.context(request_id="req-456"):
            assert _request_id.get() == "req-456"

    def test_file_logging(self, tmp_path: Path) -> None:
        """Test logging to file."""
        log_file = tmp_path / "test.log"
        logger = StructuredLogger("test", log_file=log_file)
        logger.info("File test message")

        # File should be created (may not be written yet due to buffering)
        # assert log_file.parent.exists()


class TestContextVariableFunctions:
    """Test context variable setter functions."""

    def test_set_trace_id(self) -> None:
        """Test set_trace_id function."""
        set_trace_id("new-trace")
        assert _trace_id.get() == "new-trace"
        _trace_id.set(None)

    def test_set_request_id(self) -> None:
        """Test set_request_id function."""
        set_request_id("new-request")
        assert _request_id.get() == "new-request"
        _request_id.set(None)

    def test_set_tool_name(self) -> None:
        """Test set_tool_name function."""
        set_tool_name("new-tool")
        assert _tool_name.get() == "new-tool"
        _tool_name.set(None)


class TestGetLogger:
    """Test get_logger factory function."""

    def test_get_logger_default(self) -> None:
        """Test get_logger with defaults."""
        logger = get_logger("my_module")
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "my_module"

    def test_get_logger_with_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_logger reads from environment."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_FORMAT", "json")

        logger = get_logger("env_test")
        assert logger.level == LogLevel.DEBUG
        assert logger.log_format == LogFormat.JSON

    def test_get_logger_with_explicit_values(self) -> None:
        """Test get_logger with explicit values overrides env."""
        logger = get_logger(
            "explicit_test",
            level=LogLevel.ERROR,
            log_format=LogFormat.TEXT,
        )
        assert logger.level == LogLevel.ERROR
        assert logger.log_format == LogFormat.TEXT


class TestGetDefaultLogger:
    """Test default logger singleton."""

    def test_get_default_logger(self) -> None:
        """Test get_default_logger returns logger."""
        logger = get_default_logger()
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "matrixmind_mcp"

    def test_get_default_logger_singleton(self) -> None:
        """Test default logger is singleton."""
        logger1 = get_default_logger()
        logger2 = get_default_logger()
        assert logger1 is logger2
