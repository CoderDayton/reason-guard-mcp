"""Custom exceptions for Enhanced CoT MCP."""

from __future__ import annotations

from typing import Any


class EnhancedCoTException(Exception):
    """Base exception for Enhanced CoT MCP."""

    pass


class CompressionException(EnhancedCoTException):
    """Raised during compression failures."""

    pass


class ReasoningException(EnhancedCoTException):
    """Raised during reasoning failures."""

    pass


class VerificationException(EnhancedCoTException):
    """Raised during verification failures."""

    pass


class LLMException(EnhancedCoTException):
    """Raised during LLM API calls."""

    pass


class ConfigException(EnhancedCoTException):
    """Raised during configuration issues."""

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
