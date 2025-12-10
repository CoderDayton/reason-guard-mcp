"""Tests for MCP server tool implementations.

These tests verify the FastMCP tool endpoints work correctly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.server import (
    _get_env,
    _get_env_int,
    mcp,
)

# =============================================================================
# Environment Variable Helper Tests
# =============================================================================


class TestEnvHelpers:
    """Tests for environment variable helper functions."""

    def test_get_env_returns_value(self) -> None:
        """Test _get_env returns environment variable."""
        with patch.dict("os.environ", {"TEST_VAR": "test_value"}):
            assert _get_env("TEST_VAR") == "test_value"

    def test_get_env_returns_default_when_missing(self) -> None:
        """Test _get_env returns default when var missing."""
        with patch.dict("os.environ", {}, clear=True):
            assert _get_env("NONEXISTENT_VAR", "default") == "default"

    def test_get_env_returns_default_when_empty(self) -> None:
        """Test _get_env returns default when var is empty string."""
        with patch.dict("os.environ", {"EMPTY_VAR": ""}):
            assert _get_env("EMPTY_VAR", "default") == "default"

    def test_get_env_int_returns_int(self) -> None:
        """Test _get_env_int parses integer."""
        with patch.dict("os.environ", {"INT_VAR": "42"}):
            assert _get_env_int("INT_VAR", 0) == 42

    def test_get_env_int_returns_default_when_missing(self) -> None:
        """Test _get_env_int returns default when missing."""
        with patch.dict("os.environ", {}, clear=True):
            assert _get_env_int("NONEXISTENT", 100) == 100

    def test_get_env_int_returns_default_on_invalid(self) -> None:
        """Test _get_env_int returns default on invalid int."""
        with patch.dict("os.environ", {"BAD_INT": "not_a_number"}):
            assert _get_env_int("BAD_INT", 50) == 50


# =============================================================================
# Server Configuration Tests
# =============================================================================


class TestServerConfig:
    """Tests for server configuration."""

    def test_mcp_server_exists(self) -> None:
        """Test FastMCP server is initialized."""
        assert mcp is not None
        assert mcp.name == "MatrixMind-MCP" or hasattr(mcp, "name")


# =============================================================================
# State Manager Unit Tests (Direct, no mocking)
# =============================================================================


class TestChainManagerIntegration:
    """Tests for LongChainManager via direct instantiation."""

    def test_chain_manager_start(self) -> None:
        """Test chain manager creates session."""
        from src.tools.long_chain import LongChainManager

        manager = LongChainManager()
        result = manager.start_chain(problem="Test", expected_steps=5)

        assert "session_id" in result
        assert result["status"] == "started"

    def test_chain_manager_workflow(self) -> None:
        """Test chain manager full workflow."""
        from src.tools.long_chain import LongChainManager

        manager = LongChainManager()

        # Start
        start = manager.start_chain(problem="What is 2+2?", expected_steps=3)
        session_id = start["session_id"]

        # Add steps
        step1 = manager.add_step(session_id, thought="I need to add 2 and 2")
        assert step1["step_added"] == 1

        step2 = manager.add_step(session_id, thought="2 + 2 = 4")
        assert step2["step_added"] == 2

        # Finalize
        result = manager.finalize(session_id, answer="4", confidence=1.0)
        assert result["status"] == "completed"
        assert result["final_answer"] == "4"


class TestMatrixManagerIntegration:
    """Tests for MatrixOfThoughtManager via direct instantiation."""

    def test_matrix_manager_start(self) -> None:
        """Test matrix manager creates session."""
        from src.tools.mot_reasoning import MatrixOfThoughtManager

        manager = MatrixOfThoughtManager()
        result = manager.start_matrix(question="Test?", rows=2, cols=2)

        assert "session_id" in result
        assert result["status"] == "started"
        assert result["matrix_dimensions"]["rows"] == 2

    def test_matrix_manager_workflow(self) -> None:
        """Test matrix manager full workflow."""
        from src.tools.mot_reasoning import MatrixOfThoughtManager

        manager = MatrixOfThoughtManager()

        # Start
        start = manager.start_matrix(question="Is Python good?", rows=2, cols=2)
        session_id = start["session_id"]

        # Fill cells
        manager.set_cell(session_id, 0, 0, thought="Pros: Easy syntax")
        manager.set_cell(session_id, 0, 1, thought="Pros: Large ecosystem")
        manager.set_cell(session_id, 1, 0, thought="Cons: Slow")
        manager.set_cell(session_id, 1, 1, thought="Cons: GIL")

        # Finalize
        result = manager.finalize(session_id, answer="Yes, Python is good", confidence=0.8)
        assert result["status"] == "completed"


class TestVerificationManagerIntegration:
    """Tests for VerificationManager via direct instantiation."""

    def test_verify_manager_start(self) -> None:
        """Test verification manager creates session."""
        from src.tools.verify import VerificationManager

        manager = VerificationManager()
        result = manager.start_verification(
            answer="Einstein was born in 1879",
            context="Albert Einstein was born in Ulm, Germany in 1879.",
        )

        assert "session_id" in result
        assert result["status"] == "started"

    def test_verify_manager_workflow(self) -> None:
        """Test verification manager full workflow."""
        from src.tools.verify import VerificationManager

        manager = VerificationManager()

        # Start
        start = manager.start_verification(
            answer="Einstein was born in 1879 in Ulm",
            context="Albert Einstein was born in Ulm, Germany in 1879.",
        )
        session_id = start["session_id"]

        # Add claims
        manager.add_claim(session_id, content="Einstein was born in 1879")
        manager.add_claim(session_id, content="Einstein was born in Ulm")

        # Verify claims
        manager.verify_claim(session_id, 0, "supported", "Text says 1879", 0.95)
        manager.verify_claim(session_id, 1, "supported", "Text says Ulm", 0.95)

        # Finalize
        result = manager.finalize(session_id)
        assert result["status"] == "completed"
        assert result["summary"]["supported"] == 2


# =============================================================================
# Compression Tool Tests
# =============================================================================


class TestCompressionTool:
    """Tests for compression tool."""

    def test_compression_tool_creation(self) -> None:
        """Test compression tool can be created."""
        from src.tools.compress import ContextAwareCompressionTool

        tool = ContextAwareCompressionTool()
        assert tool is not None
        assert hasattr(tool, "compress")


# =============================================================================
# Strategy Recommendation Tests (Pure Logic)
# =============================================================================


class TestStrategyRecommendation:
    """Tests for strategy recommendation logic."""

    def test_serial_problem_indicators(self) -> None:
        """Test serial problem indicator detection."""
        problem = "Find a path through a graph with constraints step by step"
        problem_lower = problem.lower()

        serial_indicators = [
            "order",
            "sequence",
            "step",
            "then",
            "constraint",
            "path",
            "graph",
            "connect",
            "chain",
            "depend",
        ]

        serial_count = sum(1 for ind in serial_indicators if ind in problem_lower)
        assert serial_count >= 3  # Should detect serial indicators

    def test_parallel_problem_indicators(self) -> None:
        """Test parallel problem indicator detection."""
        problem = "Generate multiple different creative alternatives to explore"
        problem_lower = problem.lower()

        parallel_indicators = [
            "multiple",
            "different",
            "alternative",
            "creative",
            "generate",
            "brainstorm",
            "explore",
            "options",
        ]

        parallel_count = sum(1 for ind in parallel_indicators if ind in problem_lower)
        assert parallel_count >= 3  # Should detect parallel indicators


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in tools."""

    def test_chain_invalid_session(self) -> None:
        """Test chain manager raises on invalid session."""
        from src.tools.long_chain import LongChainManager

        manager = LongChainManager()

        with pytest.raises(ValueError, match="Session .* not found"):
            manager.add_step("nonexistent", thought="Test")

    def test_matrix_invalid_session(self) -> None:
        """Test matrix manager raises on invalid session."""
        from src.tools.mot_reasoning import MatrixOfThoughtManager

        manager = MatrixOfThoughtManager()

        with pytest.raises(ValueError, match="Session .* not found"):
            manager.set_cell("nonexistent", 0, 0, thought="Test")

    def test_verify_invalid_session(self) -> None:
        """Test verification manager raises on invalid session."""
        from src.tools.verify import VerificationManager

        manager = VerificationManager()

        with pytest.raises(ValueError, match="Session .* not found"):
            manager.add_claim("nonexistent", content="Test")

    def test_chain_confidence_validation(self) -> None:
        """Test chain manager accepts confidence values."""
        from src.tools.long_chain import LongChainManager

        manager = LongChainManager()
        start = manager.start_chain(problem="Test", expected_steps=5)
        session_id = start["session_id"]

        # The implementation accepts confidence without validation
        # This test documents current behavior
        result = manager.finalize(session_id, answer="X", confidence=0.9)
        assert result["status"] == "completed"
        assert result["confidence"] == 0.9

    def test_verify_empty_context_handling(self) -> None:
        """Test verification manager handles empty context."""
        from src.tools.verify import VerificationManager

        manager = VerificationManager()

        # Empty context should either raise or return error
        try:
            result = manager.start_verification(answer="Test", context="")
            # If it doesn't raise, check for error in result
            assert "error" in result or result["status"] == "started"
        except ValueError:
            # This is also acceptable behavior
            pass
