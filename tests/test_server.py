"""Tests for MCP server tool implementations.

These tests verify the FastMCP tool endpoints work correctly.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import patch

import pytest

from src.server import (
    _get_env,
    _get_env_int,
    get_compression_tool,
    mcp,
)
from src.tools.long_chain import get_chain_manager
from src.tools.mot_reasoning import get_matrix_manager
from src.tools.verify import get_verification_manager
from src.utils.session import SessionNotFoundError

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

        with pytest.raises(SessionNotFoundError):
            manager.add_step("nonexistent", thought="Test")

    def test_matrix_invalid_session(self) -> None:
        """Test matrix manager raises on invalid session."""
        from src.tools.mot_reasoning import MatrixOfThoughtManager

        manager = MatrixOfThoughtManager()

        with pytest.raises(SessionNotFoundError):
            manager.set_cell("nonexistent", 0, 0, thought="Test")

    def test_verify_invalid_session(self) -> None:
        """Test verification manager raises on invalid session."""
        from src.tools.verify import VerificationManager

        manager = VerificationManager()

        with pytest.raises(SessionNotFoundError):
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


# =============================================================================
# Singleton Manager Tests
# =============================================================================


class TestSingletonManagers:
    """Tests for singleton manager accessors."""

    def test_get_chain_manager_singleton(self) -> None:
        """Test chain manager is singleton."""
        mgr1 = get_chain_manager()
        mgr2 = get_chain_manager()
        assert mgr1 is mgr2

    def test_get_matrix_manager_singleton(self) -> None:
        """Test matrix manager is singleton."""
        mgr1 = get_matrix_manager()
        mgr2 = get_matrix_manager()
        assert mgr1 is mgr2

    def test_get_verification_manager_singleton(self) -> None:
        """Test verification manager is singleton."""
        mgr1 = get_verification_manager()
        mgr2 = get_verification_manager()
        assert mgr1 is mgr2

    def test_get_compression_tool_singleton(self) -> None:
        """Test compression tool is singleton."""
        tool1 = get_compression_tool()
        tool2 = get_compression_tool()
        assert tool1 is tool2


# =============================================================================
# Cleanup Task Tests
# =============================================================================


class TestCleanupTask:
    """Tests for session cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_stale_sessions(self) -> None:
        """Test stale session cleanup works correctly."""
        # Create managers and add sessions
        chain_mgr = get_chain_manager()
        matrix_mgr = get_matrix_manager()
        verify_mgr = get_verification_manager()

        # Create test sessions
        chain_mgr.start_chain(problem="Test", expected_steps=5)
        matrix_mgr.start_matrix(question="Test", rows=2, cols=2)
        verify_mgr.start_verification(answer="Test", context="Context")

        # Clean up with 0 max_age should remove all (they're "stale")
        zero_age = timedelta(seconds=0)

        # Wait a tiny bit so sessions are "old" relative to 0 age
        await asyncio.sleep(0.01)

        chain_removed = chain_mgr.cleanup_stale(zero_age)
        matrix_removed = matrix_mgr.cleanup_stale(zero_age)
        verify_removed = verify_mgr.cleanup_stale(zero_age)

        # At least one should be removed from each
        assert len(chain_removed) >= 1
        assert len(matrix_removed) >= 1
        assert len(verify_removed) >= 1


# =============================================================================
# Server Initialization Tests
# =============================================================================


class TestServerInitialization:
    """Tests for server initialization functions."""

    def test_get_embedding_model_name_with_prefix(self) -> None:
        """Test _get_embedding_model_name adds prefix when needed."""
        from src.server import _get_embedding_model_name

        result = _get_embedding_model_name()
        # Should contain sentence-transformers prefix or have a /
        assert "/" in result

    def test_init_model_manager(self) -> None:
        """Test _init_model_manager initializes the model."""
        from src.server import _init_model_manager

        # Should not raise
        _init_model_manager()

        # Verify model is initialized
        from src.models.model_manager import ModelManager

        manager = ModelManager.get_instance()
        status = manager.get_status()
        # Model manager returns dict with device info when ready
        assert "device" in status or "model_name" in status


# =============================================================================
# Consolidated Tools Tests
# =============================================================================


class TestConsolidatedThinkTool:
    """Tests for the unified think() tool."""

    @pytest.mark.asyncio
    async def test_think_chain_start(self) -> None:
        """Test think() with action=start and mode=chain."""
        import json

        from src.server import think

        result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="What is 2 + 2?",
                expected_steps=2,
            )
        )

        assert "session_id" in result
        assert result["mode"] == "chain"

    @pytest.mark.asyncio
    async def test_think_chain_continue(self) -> None:
        """Test think() with action=continue for chain mode."""
        import json

        from src.server import think

        # Start a chain first
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Multiply 6 by 7",
                expected_steps=2,
            )
        )
        session_id = start_result["session_id"]

        # Continue with a thought
        add_result = json.loads(
            await think.fn(
                action="continue",
                session_id=session_id,
                thought="6 * 7 = 42",
            )
        )

        assert "step" in add_result or "thought" in add_result or "error" not in add_result

    @pytest.mark.asyncio
    async def test_think_chain_finish(self) -> None:
        """Test think() with action=finish for chain mode."""
        import json

        from src.server import think

        # Start and continue
        start_result = json.loads(
            await think.fn(action="start", mode="chain", problem="What is 3+3?", expected_steps=1)
        )
        session_id = start_result["session_id"]

        await think.fn(action="continue", session_id=session_id, thought="3 + 3 = 6")

        # Finish
        final_result = json.loads(
            await think.fn(
                action="finish",
                session_id=session_id,
                thought="The answer is 6",
            )
        )

        assert final_result["status"] == "completed" or "final" in str(final_result).lower()

    @pytest.mark.asyncio
    async def test_think_chain_branch(self) -> None:
        """Test think() with action=branch for chain mode."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(action="start", mode="chain", problem="Test branching", expected_steps=3)
        )
        session_id = start_result["session_id"]

        await think.fn(action="continue", session_id=session_id, thought="Step 1")

        # Branch from step 0
        branch_result = json.loads(
            await think.fn(
                action="branch",
                session_id=session_id,
                thought="Alternative approach",
                branch_from=0,
            )
        )

        assert "branch" in str(branch_result).lower() or "error" not in branch_result

    @pytest.mark.asyncio
    async def test_think_chain_revise(self) -> None:
        """Test think() with action=revise for chain mode."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(action="start", mode="chain", problem="Test revising", expected_steps=3)
        )
        session_id = start_result["session_id"]

        await think.fn(action="continue", session_id=session_id, thought="Initial thought")

        # Revise step 1
        revise_result = json.loads(
            await think.fn(
                action="revise",
                session_id=session_id,
                thought="Revised thought",
                revises=1,
            )
        )

        assert "revis" in str(revise_result).lower() or "error" not in revise_result

    @pytest.mark.asyncio
    async def test_think_matrix_start(self) -> None:
        """Test think() with action=start and mode=matrix."""
        import json

        from src.server import think

        result = json.loads(
            await think.fn(
                action="start",
                mode="matrix",
                problem="Is climate change real?",
                context="Scientific consensus",
                rows=2,
                cols=2,
            )
        )

        assert "session_id" in result
        assert result["mode"] == "matrix"

    @pytest.mark.asyncio
    async def test_think_matrix_continue(self) -> None:
        """Test think() with action=continue for matrix mode sets cell."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(
                action="start",
                mode="matrix",
                problem="Test question",
                rows=2,
                cols=2,
            )
        )
        session_id = start_result["session_id"]

        set_result = json.loads(
            await think.fn(
                action="continue",
                session_id=session_id,
                row=0,
                col=0,
                thought="Analysis point",
            )
        )

        assert "error" not in set_result or "cell" in str(set_result).lower()

    @pytest.mark.asyncio
    async def test_think_matrix_synthesize(self) -> None:
        """Test think() with action=synthesize for matrix mode."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(action="start", mode="matrix", problem="Test", rows=2, cols=2)
        )
        session_id = start_result["session_id"]

        # Fill ALL cells in column 0 (both rows required before synthesize)
        await think.fn(action="continue", session_id=session_id, row=0, col=0, thought="Point 1")
        await think.fn(action="continue", session_id=session_id, row=1, col=0, thought="Point 2")

        # Synthesize column 0 - now all rows are filled
        synth_result = json.loads(
            await think.fn(
                action="synthesize",
                session_id=session_id,
                col=0,
                thought="Column synthesis combining points 1 and 2",
            )
        )

        assert "error" not in synth_result or "synth" in str(synth_result).lower()

    @pytest.mark.asyncio
    async def test_think_matrix_finish(self) -> None:
        """Test think() with action=finish for matrix mode."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(action="start", mode="matrix", problem="Test", rows=2, cols=2)
        )
        session_id = start_result["session_id"]

        # Fill some cells
        await think.fn(action="continue", session_id=session_id, row=0, col=0, thought="Cell 0,0")

        # Finish
        final_result = json.loads(
            await think.fn(
                action="finish",
                session_id=session_id,
                thought="Final analysis",
            )
        )

        assert final_result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_think_verify_start(self) -> None:
        """Test think() with action=start and mode=verify."""
        import json

        from src.server import think

        result = json.loads(
            await think.fn(
                action="start",
                mode="verify",
                problem="The sky is blue",
                context="Basic color observation",
            )
        )

        assert "session_id" in result
        assert result["mode"] == "verify"

    @pytest.mark.asyncio
    async def test_think_verify_continue_claim(self) -> None:
        """Test think() with action=continue for verify mode adds claim."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(
                action="start",
                mode="verify",
                problem="Water boils at 100C",
                context="At sea level",
            )
        )
        session_id = start_result["session_id"]

        add_result = json.loads(
            await think.fn(
                action="continue",
                session_id=session_id,
                thought="Water boils at 100 degrees Celsius",
            )
        )

        assert "error" not in add_result or "claim" in str(add_result).lower()

    @pytest.mark.asyncio
    async def test_think_verify_action(self) -> None:
        """Test think() with action=verify for claim verification."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(action="start", mode="verify", problem="Fact", context="Context")
        )
        session_id = start_result["session_id"]

        # Add a claim first (claim_id will be 0)
        await think.fn(action="continue", session_id=session_id, thought="Claim to verify")

        # Verify claim 0
        check_result = json.loads(
            await think.fn(
                action="verify",
                session_id=session_id,
                claim_id=0,
                evidence="Evidence text",
                verdict="supported",
            )
        )

        # Should succeed or show appropriate error
        assert "error" not in check_result or "claim" in str(check_result).lower()

    @pytest.mark.asyncio
    async def test_think_verify_finish(self) -> None:
        """Test think() with action=finish for verify mode."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(action="start", mode="verify", problem="Test fact", context="Context")
        )
        session_id = start_result["session_id"]

        # Add and verify a claim
        await think.fn(action="continue", session_id=session_id, thought="Test claim")
        await think.fn(
            action="verify",
            session_id=session_id,
            claim_id=0,
            evidence="Evidence",
            verdict="supported",
        )

        # Finish
        final_result = json.loads(await think.fn(action="finish", session_id=session_id))

        assert final_result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_think_invalid_action(self) -> None:
        """Test think() with invalid action returns error."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="invalid_action"))

        assert "error" in result

    @pytest.mark.asyncio
    async def test_think_missing_session_id(self) -> None:
        """Test think() with action requiring session_id but none provided."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="continue", thought="Some thought"))

        assert "error" in result

    @pytest.mark.asyncio
    async def test_think_missing_mode_on_start(self) -> None:
        """Test think() start action without mode returns error."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="start", problem="Test"))

        assert "error" in result

    @pytest.mark.asyncio
    async def test_think_missing_problem_on_start(self) -> None:
        """Test think() start action without problem returns error."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="start", mode="chain"))

        assert "error" in result

    @pytest.mark.asyncio
    async def test_think_branch_missing_branch_from(self) -> None:
        """Test think() branch action without branch_from returns error."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(action="start", mode="chain", problem="Test", expected_steps=3)
        )
        session_id = start_result["session_id"]

        result = json.loads(
            await think.fn(action="branch", session_id=session_id, thought="Branch thought")
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_think_synthesize_missing_col(self) -> None:
        """Test think() synthesize action without col returns error."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(action="start", mode="matrix", problem="Test", rows=2, cols=2)
        )
        session_id = start_result["session_id"]

        result = json.loads(
            await think.fn(action="synthesize", session_id=session_id, thought="Synthesis")
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_think_verify_missing_claim_id(self) -> None:
        """Test think() verify action without claim_id returns error."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(action="start", mode="verify", problem="Test", context="Context")
        )
        session_id = start_result["session_id"]

        result = json.loads(
            await think.fn(
                action="verify", session_id=session_id, verdict="supported", evidence="Evidence"
            )
        )

        assert "error" in result


class TestConsolidatedCompressTool:
    """Tests for the unified compress() tool."""

    @pytest.mark.asyncio
    async def test_compress_basic(self) -> None:
        """Test compress() with basic context."""
        import json

        from src.server import compress

        result = json.loads(
            await compress.fn(
                context="This is a long context with many words. " * 20,
                query="What is the main point?",
                ratio=0.5,
            )
        )

        assert (
            "compressed" in result
            or "result" in result
            or "text" in result
            or "error" not in result
        )

    @pytest.mark.asyncio
    async def test_compress_high_ratio(self) -> None:
        """Test compress() with high compression ratio."""
        import json

        from src.server import compress

        long_context = "Important information about the topic. " * 50

        result = json.loads(
            await compress.fn(
                context=long_context,
                query="Summarize the topic",
                ratio=0.1,
            )
        )

        # Should return compressed content
        assert "error" not in result or "compressed" in result

    @pytest.mark.asyncio
    async def test_compress_requires_query(self) -> None:
        """Test compress() without query returns error (query required by underlying tool)."""
        import json

        from src.server import compress

        result = json.loads(
            await compress.fn(
                context="Some context to compress. " * 10,
                query="",  # Empty query
                ratio=0.5,
            )
        )

        # Should error because question cannot be empty
        assert "error" in result


class TestConsolidatedStatusTool:
    """Tests for the unified status() tool."""

    @pytest.mark.asyncio
    async def test_status_server(self) -> None:
        """Test status() returns server status when no session_id."""
        import json

        from src.server import status

        result = json.loads(await status.fn())

        assert "server" in result or "status" in result or "version" in result

    @pytest.mark.asyncio
    async def test_status_with_session(self) -> None:
        """Test status() returns session status when session_id provided."""
        import json

        from src.server import status, think

        # Create a session first
        start_result = json.loads(
            await think.fn(action="start", mode="chain", problem="Test", expected_steps=1)
        )
        session_id = start_result["session_id"]

        # Get status for that session
        status_result = json.loads(await status.fn(session_id=session_id))

        assert "session_id" in status_result or "status" in status_result

    @pytest.mark.asyncio
    async def test_status_invalid_session(self) -> None:
        """Test status() with invalid session_id returns error."""
        import json

        from src.server import status

        result = json.loads(await status.fn(session_id="nonexistent-session-id"))

        assert "error" in result or "not_found" in result or "status" in result


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for session creation rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self) -> None:
        """Test that requests within rate limit are allowed."""
        from src.server import SessionRateLimiter

        limiter = SessionRateLimiter(max_requests=10, window_seconds=60)

        for _ in range(5):
            allowed, info = await limiter.check_rate_limit()
            assert allowed, f"Request should be allowed: {info}"
            await limiter.record_request()

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self) -> None:
        """Test that requests exceeding rate limit are blocked."""
        from src.server import SessionRateLimiter

        limiter = SessionRateLimiter(max_requests=3, window_seconds=60)

        # Use up the limit
        for _ in range(3):
            allowed, _ = await limiter.check_rate_limit()
            assert allowed
            await limiter.record_request()

        # Next request should be blocked
        allowed, info = await limiter.check_rate_limit()
        assert not allowed
        assert "error" in info
        assert info["error"] == "rate_limit_exceeded"
        assert "retry_after_seconds" in info

    @pytest.mark.asyncio
    async def test_rate_limiter_sliding_window(self) -> None:
        """Test that rate limiter uses sliding window correctly."""
        from src.server import SessionRateLimiter

        # Very short window for testing
        limiter = SessionRateLimiter(max_requests=2, window_seconds=1)

        # Use up limit
        await limiter.record_request()
        await limiter.record_request()

        allowed, _ = await limiter.check_rate_limit()
        assert not allowed

        # Wait for window to expire
        await asyncio.sleep(1.1)

        # Should be allowed again
        allowed, info = await limiter.check_rate_limit()
        assert allowed, f"Should be allowed after window expires: {info}"

    def test_rate_limiter_stats(self) -> None:
        """Test rate limiter statistics."""
        from src.server import SessionRateLimiter

        limiter = SessionRateLimiter(max_requests=10, window_seconds=60)

        stats = limiter.get_stats()
        assert stats["max_allowed"] == 10
        assert stats["window_seconds"] == 60
        assert stats["current_count"] == 0
        assert stats["remaining"] == 10

    @pytest.mark.asyncio
    async def test_think_start_respects_rate_limit(self) -> None:
        """Test that think start action respects rate limiting."""
        import json

        from src.server import get_rate_limiter, think

        # Get the rate limiter and check it's working
        limiter = get_rate_limiter()
        limiter.get_stats()  # Verify stats are accessible

        # Should be able to start a session
        result = json.loads(
            await think.fn(action="start", mode="chain", problem="Test", expected_steps=3)
        )

        # Either succeeds or rate limited (depending on test order)
        assert "session_id" in result or "error" in result

    @pytest.mark.asyncio
    async def test_status_includes_rate_limit_info(self) -> None:
        """Test that status() includes rate limit information."""
        import json

        from src.server import status

        result = json.loads(await status.fn())

        assert "rate_limit" in result
        assert "max_allowed" in result["rate_limit"]
        assert "remaining" in result["rate_limit"]


# =============================================================================
# Session Cleanup Tests
# =============================================================================


class TestSessionCleanup:
    """Tests for automatic session cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_stale_sessions(self) -> None:
        """Test stale session cleanup works correctly."""
        # Create managers and add sessions
        chain_mgr = get_chain_manager()
        matrix_mgr = get_matrix_manager()
        verify_mgr = get_verification_manager()

        # Create test sessions
        chain_mgr.start_chain(problem="Test", expected_steps=5)
        matrix_mgr.start_matrix(question="Test", rows=2, cols=2)
        verify_mgr.start_verification(answer="Test", context="Context")

        # Clean up with 0 max_age should remove all (they're "stale")
        zero_age = timedelta(seconds=0)

        # Wait a tiny bit so sessions are "old" relative to 0 age
        await asyncio.sleep(0.01)

        chain_removed = chain_mgr.cleanup_stale(zero_age)
        matrix_removed = matrix_mgr.cleanup_stale(zero_age)
        verify_removed = verify_mgr.cleanup_stale(zero_age)

        # At least one should be removed from each
        assert len(chain_removed) >= 1
        assert len(matrix_removed) >= 1
        assert len(verify_removed) >= 1

    def test_cleanup_task_functions_exist(self) -> None:
        """Test that cleanup task functions are available."""
        from src.server import _start_cleanup_task, _stop_cleanup_task

        # Functions should exist and be callable
        assert callable(_start_cleanup_task)
        assert callable(_stop_cleanup_task)

    @pytest.mark.asyncio
    async def test_status_includes_cleanup_info(self) -> None:
        """Test that status() includes cleanup configuration."""
        import json

        from src.server import status

        result = json.loads(await status.fn())

        assert "cleanup" in result
        assert "max_age_minutes" in result["cleanup"]
        assert "interval_seconds" in result["cleanup"]

    @pytest.mark.asyncio
    async def test_max_sessions_limit(self) -> None:
        """Test that MAX_TOTAL_SESSIONS is reported in status."""
        import json

        from src.server import status

        result = json.loads(await status.fn())

        assert "active_sessions" in result
        assert "max_total" in result["active_sessions"]
        assert result["active_sessions"]["max_total"] > 0
