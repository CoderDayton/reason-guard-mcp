"""Custom exceptions for Reason Guard MCP."""

from __future__ import annotations

import warnings
from typing import Any


class ReasonGuardException(Exception):
    """Base exception for Reason Guard MCP."""

    pass


def _get_deprecated_matrixmind_exception() -> type[ReasonGuardException]:
    """Return ReasonGuardException with deprecation warning."""
    warnings.warn(
        "MatrixMindException is deprecated, use ReasonGuardException instead. "
        "MatrixMindException will be removed in version 1.0.0.",
        DeprecationWarning,
        stacklevel=3,
    )
    return ReasonGuardException


# Backwards compatibility - lazy deprecation via __getattr__
def __getattr__(name: str) -> type[ReasonGuardException]:
    if name == "MatrixMindException":
        return _get_deprecated_matrixmind_exception()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class CompressionException(ReasonGuardException):
    """Raised during compression failures."""

    pass


class ReasoningException(ReasonGuardException):
    """Raised during reasoning failures."""

    pass


class VerificationException(ReasonGuardException):
    """Raised during verification failures."""

    pass


class LLMException(ReasonGuardException):
    """Raised during LLM API calls."""

    pass


class ConfigException(ReasonGuardException):
    """Raised during configuration issues."""

    pass


class ModelNotReadyException(ReasonGuardException):
    """Raised when a tool is called before the embedding model is ready.

    This typically occurs when:
    - The model is still downloading (first run)
    - The model is loading into memory
    - The model failed to load due to network/disk issues
    """

    pass


class InvalidActionError(ReasonGuardException):
    """Raised when an invalid action is provided to a tool."""

    pass


class SessionNotFoundError(ReasonGuardException):
    """Raised when a session ID is not found."""

    pass


class ToolExecutionError(Exception):
    """Raised when tool execution fails in MCP context.

    Provides structured error information that can be returned
    to the LLM client in a parseable format.
    """

    def __init__(
        self,
        tool_name: str,
        error_message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize tool execution error.

        Args:
            tool_name: Name of the tool that failed.
            error_message: Human-readable error message.
            details: Optional dictionary with additional error details.

        """
        self.tool_name = tool_name
        self.error_message = error_message
        self.details = details or {}
        super().__init__(f"Tool {tool_name} failed: {error_message}")

    def to_mcp_error(self) -> str:
        """Convert to MCP-compatible error format.

        Returns:
            Formatted error string for MCP response.

        """
        return f"[{self.tool_name}] {self.error_message}. Details: {self.details}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the error.

        """
        return {
            "error": True,
            "tool": self.tool_name,
            "message": self.error_message,
            "details": self.details,
        }
