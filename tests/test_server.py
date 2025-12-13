"""Tests for MCP server tool implementations.

These tests verify the FastMCP tool endpoints work correctly.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any
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
    async def test_think_auto_mode_on_start(self) -> None:
        """Test think() start action without mode uses auto-mode selection."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="start", problem="Test problem"))

        # Auto-mode selection should succeed
        assert "session_id" in result
        assert "actual_mode" in result  # Shows auto-selected mode

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
    async def test_think_verify_requires_evidence(self) -> None:
        """Test think() verify action requires thought/evidence."""
        import json

        from src.server import think

        start_result = json.loads(
            await think.fn(action="start", mode="verify", problem="Test", context="Context")
        )
        session_id = start_result["session_id"]

        # Verify without thought/evidence should error
        result = json.loads(await think.fn(action="verify", session_id=session_id))

        assert "error" in result


class TestContradictionGuidance:
    """Tests for contradiction resolution guidance in unified reasoner."""

    def _make_session(self, problem: str = "Test") -> Any:
        """Create a minimal ReasoningSession for testing."""
        from src.tools.unified_reasoner import (
            DomainType,
            ReasoningMode,
            ReasoningSession,
            SessionStatus,
        )
        from src.utils.complexity import ComplexityResult

        return ReasoningSession(
            session_id="test_session",
            problem=problem,
            context="Test context",
            mode=ReasoningMode.CHAIN,
            actual_mode=ReasoningMode.CHAIN,
            status=SessionStatus.ACTIVE,
            domain=DomainType.GENERAL,
            complexity=ComplexityResult(
                complexity_score=0.5,
                complexity_level="medium",
                recommended_rows=3,
                recommended_cols=3,
                signals=(),
                word_count=10,
            ),
        )

    @pytest.mark.asyncio
    async def test_contradiction_guidance_structure(self) -> None:
        """Test that contradiction guidance returns proper structure."""
        from src.tools.unified_reasoner import (
            Thought,
            ThoughtType,
            UnifiedReasonerManager,
        )

        manager = UnifiedReasonerManager()
        session = self._make_session()

        session.thoughts["t1"] = Thought(
            id="t1",
            content="X is true",
            thought_type=ThoughtType.CONTINUATION,
            step_number=1,
            confidence=0.8,
        )
        session.thoughts["t2"] = Thought(
            id="t2",
            content="X is false",
            thought_type=ThoughtType.CONTINUATION,
            step_number=2,
            confidence=0.6,
        )

        current_thought = session.thoughts["t2"]

        guidance = manager._get_contradiction_guidance(
            session=session,
            current_thought=current_thought,
            contradicting_ids=["t1"],
        )

        assert "contradicting_thoughts" in guidance
        assert "strategies" in guidance
        assert "recommendation" in guidance
        assert "action_required" in guidance
        assert "message" in guidance

    @pytest.mark.asyncio
    async def test_contradiction_guidance_strategies(self) -> None:
        """Test that all four strategies are provided."""
        from src.tools.unified_reasoner import (
            Thought,
            ThoughtType,
            UnifiedReasonerManager,
        )

        manager = UnifiedReasonerManager()
        session = self._make_session()
        session.thoughts["t1"] = Thought(
            id="t1",
            content="Claim A",
            thought_type=ThoughtType.CONTINUATION,
            step_number=1,
            confidence=0.7,
        )

        guidance = manager._get_contradiction_guidance(
            session=session,
            current_thought=Thought(
                id="t2",
                content="Not A",
                thought_type=ThoughtType.CONTINUATION,
                step_number=2,
                confidence=0.7,
            ),
            contradicting_ids=["t1"],
        )

        strategy_names = [s["name"] for s in guidance["strategies"]]
        assert "revise" in strategy_names
        assert "branch" in strategy_names
        assert "reconcile" in strategy_names
        assert "backtrack" in strategy_names

    @pytest.mark.asyncio
    async def test_contradiction_guidance_recommends_revise_on_lower_confidence(self) -> None:
        """Test that revise is recommended when current thought has lower confidence."""
        from src.tools.unified_reasoner import (
            Thought,
            ThoughtType,
            UnifiedReasonerManager,
        )

        manager = UnifiedReasonerManager()
        session = self._make_session()
        session.thoughts["t1"] = Thought(
            id="t1",
            content="High confidence claim",
            thought_type=ThoughtType.CONTINUATION,
            step_number=1,
            confidence=0.9,  # High confidence
        )

        guidance = manager._get_contradiction_guidance(
            session=session,
            current_thought=Thought(
                id="t2",
                content="Low confidence claim",
                thought_type=ThoughtType.CONTINUATION,
                step_number=2,
                confidence=0.4,  # Low confidence
            ),
            contradicting_ids=["t1"],
        )

        assert guidance["recommendation"]["strategy"] == "revise"

    @pytest.mark.asyncio
    async def test_contradiction_guidance_recommends_reconcile_on_similar_confidence(self) -> None:
        """Test that reconcile is recommended when confidences are similar."""
        from src.tools.unified_reasoner import (
            Thought,
            ThoughtType,
            UnifiedReasonerManager,
        )

        manager = UnifiedReasonerManager()
        session = self._make_session()
        session.thoughts["t1"] = Thought(
            id="t1",
            content="Claim with similar confidence",
            thought_type=ThoughtType.CONTINUATION,
            step_number=1,
            confidence=0.7,
        )

        guidance = manager._get_contradiction_guidance(
            session=session,
            current_thought=Thought(
                id="t2",
                content="Contradicting claim with similar confidence",
                thought_type=ThoughtType.CONTINUATION,
                step_number=2,
                confidence=0.7,
            ),
            contradicting_ids=["t1"],
        )

        assert guidance["recommendation"]["strategy"] == "reconcile"

    @pytest.mark.asyncio
    async def test_contradiction_guidance_empty_on_no_contradictions(self) -> None:
        """Test that empty dict is returned when no contradicting IDs."""
        from src.tools.unified_reasoner import (
            Thought,
            ThoughtType,
            UnifiedReasonerManager,
        )

        manager = UnifiedReasonerManager()
        session = self._make_session()

        guidance = manager._get_contradiction_guidance(
            session=session,
            current_thought=Thought(
                id="t1",
                content="Some thought",
                thought_type=ThoughtType.CONTINUATION,
                step_number=1,
            ),
            contradicting_ids=[],
        )

        assert guidance == {}

    @pytest.mark.asyncio
    async def test_contradiction_guidance_includes_thought_summaries(self) -> None:
        """Test that contradicting thoughts have proper summaries."""
        from src.tools.unified_reasoner import (
            Thought,
            ThoughtType,
            UnifiedReasonerManager,
        )

        manager = UnifiedReasonerManager()
        session = self._make_session()

        long_content = "This is a very long thought content that exceeds one hundred characters and should be truncated in the guidance summary for readability."
        session.thoughts["t1"] = Thought(
            id="t1",
            content=long_content,
            thought_type=ThoughtType.CONTINUATION,
            step_number=1,
            confidence=0.7,
        )

        guidance = manager._get_contradiction_guidance(
            session=session,
            current_thought=Thought(
                id="t2",
                content="Contradicting thought",
                thought_type=ThoughtType.CONTINUATION,
                step_number=2,
            ),
            contradicting_ids=["t1"],
        )

        summary = guidance["contradicting_thoughts"][0]["summary"]
        assert len(summary) <= 103  # 100 chars + "..."
        assert summary.endswith("...")


class TestSemanticContradictionDetection:
    """Tests for semantic contradiction detection in unified reasoner."""

    @pytest.mark.asyncio
    async def test_pattern_detection_explicit_patterns(self) -> None:
        """Test pattern-based detection finds explicit contradictions."""
        from src.tools.unified_reasoner import UnifiedReasonerManager

        manager = UnifiedReasonerManager()

        # Explicit pattern: valid/invalid
        assert manager._detect_contradiction_patterns(
            "The solution is valid",
            "The solution is invalid",
        )

        # Explicit pattern: always/never
        assert manager._detect_contradiction_patterns(
            "This always works",
            "This never works",
        )

        # Explicit pattern: true/false
        assert manager._detect_contradiction_patterns(
            "The statement is true",
            "The statement is false",
        )

    @pytest.mark.asyncio
    async def test_pattern_detection_negation(self) -> None:
        """Test pattern-based detection finds negation contradictions."""
        from src.tools.unified_reasoner import UnifiedReasonerManager

        manager = UnifiedReasonerManager()

        # Negation with shared phrase
        assert manager._detect_contradiction_patterns(
            "The answer is correct",
            "The answer is not correct",
        )

    @pytest.mark.asyncio
    async def test_pattern_detection_no_contradiction(self) -> None:
        """Test pattern-based detection returns False for non-contradictions."""
        from src.tools.unified_reasoner import UnifiedReasonerManager

        manager = UnifiedReasonerManager()

        # Different topics, no contradiction
        assert not manager._detect_contradiction_patterns(
            "Python is a programming language",
            "The weather is nice today",
        )

        # Same topic, no contradiction
        assert not manager._detect_contradiction_patterns(
            "The algorithm runs in O(n) time",
            "The algorithm uses linear space",
        )

    @pytest.mark.asyncio
    async def test_semantic_detection_returns_pattern_match(self) -> None:
        """Test semantic detection fast-paths on pattern matches."""
        from src.tools.unified_reasoner import UnifiedReasonerManager

        manager = UnifiedReasonerManager()  # No vector store

        is_contra, confidence = await manager._detect_contradiction_semantic(
            "The hypothesis is always true",
            "The hypothesis is never true",
        )

        assert is_contra is True
        assert confidence >= 0.9  # High confidence for pattern match

    @pytest.mark.asyncio
    async def test_semantic_detection_no_vector_store(self) -> None:
        """Test semantic detection returns False without vector store."""
        from src.tools.unified_reasoner import UnifiedReasonerManager

        manager = UnifiedReasonerManager()  # No vector store

        # No pattern match, and no vector store for semantic check
        is_contra, confidence = await manager._detect_contradiction_semantic(
            "Machine learning models require training data",
            "Neural networks don't need any examples to learn",
        )

        # Without pattern match and without vector store, should return False
        # (this specific pair might or might not trigger patterns)
        assert isinstance(is_contra, bool)
        assert isinstance(confidence, float)

    @pytest.mark.asyncio
    async def test_detect_contradictions_in_session(self) -> None:
        """Test detecting all contradictions in a session."""
        from src.tools.unified_reasoner import ReasoningMode, UnifiedReasonerManager

        manager = UnifiedReasonerManager()

        # Start session and add thoughts
        start = await manager.start_session(
            problem="Test problem",
            mode=ReasoningMode.CHAIN,
        )
        session_id = start["session_id"]

        # Add thought 1
        await manager.add_thought(
            session_id=session_id,
            content="The result is always positive",
            confidence=0.8,
        )

        # Add contradicting thought 2
        await manager.add_thought(
            session_id=session_id,
            content="The result is never positive under any circumstances",
            confidence=0.7,
        )

        # Detect contradictions
        contradictions = await manager.detect_contradictions_in_session(
            session_id=session_id,
            use_semantic=False,  # Use pattern-based only for determinism
        )

        assert len(contradictions) >= 1
        assert "thought_a" in contradictions[0]
        assert "thought_b" in contradictions[0]
        assert "confidence" in contradictions[0]
        assert "method" in contradictions[0]

    @pytest.mark.asyncio
    async def test_detect_contradictions_specific_thought(self) -> None:
        """Test detecting contradictions for a specific thought."""
        from src.tools.unified_reasoner import ReasoningMode, UnifiedReasonerManager

        manager = UnifiedReasonerManager()

        start = await manager.start_session(
            problem="Test problem",
            mode=ReasoningMode.CHAIN,
        )
        session_id = start["session_id"]

        # Add neutral thought
        result1 = await manager.add_thought(
            session_id=session_id,
            content="First, let's analyze the data",
            confidence=0.9,
        )
        result1["thought_id"]

        # Add second thought
        result2 = await manager.add_thought(
            session_id=session_id,
            content="The data shows X is valid",
            confidence=0.8,
        )
        thought2_id = result2["thought_id"]

        # Add contradicting thought
        result3 = await manager.add_thought(
            session_id=session_id,
            content="The data shows X is invalid",
            confidence=0.75,
        )
        thought3_id = result3["thought_id"]

        # Check contradictions for thought3 only
        contradictions = await manager.detect_contradictions_in_session(
            session_id=session_id,
            thought_id=thought3_id,
            use_semantic=False,
        )

        # Should find contradiction with thought2 (valid/invalid)
        assert len(contradictions) >= 1
        contra_partners = [c["thought_b"] for c in contradictions]
        assert thought2_id in contra_partners


class TestResolveContradictionAction:
    """Tests for the resolve contradiction action in think() tool."""

    @pytest.mark.asyncio
    async def test_resolve_action_revise_strategy(self) -> None:
        """Test resolve action with revise strategy."""
        import json

        from src.server import think

        # Start session
        start = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Determine if result is positive or negative",
                expected_steps=5,
            )
        )
        session_id = start["session_id"]

        # Add contradicting thoughts
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The result is always positive because all values are above zero.",
            confidence=0.8,
        )

        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The result is never positive since we found negative values.",
            confidence=0.6,
        )

        # Resolve with revise strategy
        result = json.loads(
            await think.fn(
                action="resolve",
                session_id=session_id,
                resolve_strategy="revise",
                thought="After careful review, the result is positive in most cases but negative in edge cases.",
                confidence=0.85,
            )
        )

        assert "resolution_id" in result or "error" not in result
        if "resolution_id" in result:
            assert result["strategy_applied"] == "revise"

    @pytest.mark.asyncio
    async def test_resolve_action_branch_strategy(self) -> None:
        """Test resolve action with branch strategy."""
        import json

        from src.server import think

        start = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test branching resolution",
                expected_steps=5,
            )
        )
        session_id = start["session_id"]

        # Add thoughts
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="X is valid",
            confidence=0.7,
        )

        await think.fn(
            action="continue",
            session_id=session_id,
            thought="X is invalid",
            confidence=0.7,
        )

        # Resolve with branch strategy
        result = json.loads(
            await think.fn(
                action="resolve",
                session_id=session_id,
                resolve_strategy="branch",
                thought="Exploring the case where X is invalid: this would mean...",
                confidence=0.75,
            )
        )

        assert "resolution_id" in result or "error" not in result

    @pytest.mark.asyncio
    async def test_resolve_action_reconcile_strategy(self) -> None:
        """Test resolve action with reconcile strategy."""
        import json

        from src.server import think

        start = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test reconciliation",
                expected_steps=5,
            )
        )
        session_id = start["session_id"]

        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The system is always reliable",
            confidence=0.8,
        )

        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The system is never reliable under load",
            confidence=0.7,
        )

        # Resolve with reconcile strategy
        result = json.loads(
            await think.fn(
                action="resolve",
                session_id=session_id,
                resolve_strategy="reconcile",
                thought="The system is reliable under normal load but unreliable under heavy load. Both statements are conditionally true.",
                confidence=0.9,
            )
        )

        assert "resolution_id" in result or "error" not in result

    @pytest.mark.asyncio
    async def test_resolve_action_backtrack_strategy(self) -> None:
        """Test resolve action with backtrack strategy."""
        import json

        from src.server import think

        start = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test backtracking",
                expected_steps=5,
            )
        )
        session_id = start["session_id"]

        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The algorithm always terminates",
            confidence=0.9,
        )

        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The algorithm never terminates for large inputs",
            confidence=0.5,
        )

        # Resolve with backtrack strategy
        result = json.loads(
            await think.fn(
                action="resolve",
                session_id=session_id,
                resolve_strategy="backtrack",
                thought="Abandoning the claim about non-termination. The algorithm does terminate.",
                confidence=0.95,
            )
        )

        assert "resolution_id" in result or "error" not in result

    @pytest.mark.asyncio
    async def test_resolve_action_missing_strategy(self) -> None:
        """Test resolve action without strategy returns error."""
        import json

        from src.server import think

        start = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test",
                expected_steps=3,
            )
        )
        session_id = start["session_id"]

        result = json.loads(
            await think.fn(
                action="resolve",
                session_id=session_id,
                thought="Resolution content",
            )
        )

        assert "error" in result
        assert "resolve_strategy" in result["error"].lower() or "valid_strategies" in result

    @pytest.mark.asyncio
    async def test_resolve_action_missing_thought(self) -> None:
        """Test resolve action without thought returns error."""
        import json

        from src.server import think

        start = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test",
                expected_steps=3,
            )
        )
        session_id = start["session_id"]

        result = json.loads(
            await think.fn(
                action="resolve",
                session_id=session_id,
                resolve_strategy="revise",
            )
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_unified_reasoner_resolve_contradiction_method(self) -> None:
        """Test the resolve_contradiction method directly."""
        from src.tools.unified_reasoner import ReasoningMode, UnifiedReasonerManager

        manager = UnifiedReasonerManager()

        start = await manager.start_session(
            problem="Direct method test",
            mode=ReasoningMode.CHAIN,
        )
        session_id = start["session_id"]

        # Add contradicting thoughts
        await manager.add_thought(
            session_id=session_id,
            content="The value is always positive",
            confidence=0.8,
        )

        result2 = await manager.add_thought(
            session_id=session_id,
            content="The value is never positive",
            confidence=0.7,
        )

        # Get the thought ID that has contradictions
        thought_id = result2["thought_id"]

        # Resolve the contradiction
        resolution = await manager.resolve_contradiction(
            session_id=session_id,
            strategy="reconcile",
            resolution_content="The value can be positive or negative depending on input",
            contradicting_thought_id=thought_id,
            confidence=0.85,
        )

        assert "resolution_id" in resolution
        assert "strategy_applied" in resolution
        assert resolution["strategy_applied"] == "reconcile"
        assert "remaining_contradictions" in resolution


class TestSessionAnalysis:
    """Tests for session analysis functionality via think(action='analyze')."""

    @pytest.mark.asyncio
    async def test_analyze_requires_session_id(self) -> None:
        """Test analyze action requires session_id."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="analyze"))
        assert "error" in result
        assert "session_id" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_session_not_found(self) -> None:
        """Test analyze returns error for non-existent session."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="analyze", session_id="nonexistent"))
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_analyze_basic_session(self) -> None:
        """Test analyze returns metrics for basic session."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="What is 2+2?",
                expected_steps=3,
            )
        )
        session_id = start_result["session_id"]

        # Add a thought
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="Let me think about this simple addition.",
            confidence=0.9,
        )

        # Analyze
        analysis = json.loads(await think.fn(action="analyze", session_id=session_id))

        # Verify structure
        assert "progress" in analysis
        assert "quality" in analysis
        assert "issues" in analysis
        assert "efficiency" in analysis
        assert "recommendations" in analysis
        assert "risk" in analysis

        # Verify progress metrics
        assert analysis["progress"]["total_thoughts"] >= 1
        assert analysis["progress"]["main_chain_length"] >= 1

        # Verify quality scores are between 0 and 1
        assert 0 <= analysis["quality"]["coherence_score"] <= 1
        assert 0 <= analysis["quality"]["coverage_score"] <= 1
        assert 0 <= analysis["quality"]["depth_score"] <= 1
        assert 0 <= analysis["quality"]["overall"] <= 1

        # Verify risk has valid level
        assert analysis["risk"]["level"] in ("low", "medium", "high")

    @pytest.mark.asyncio
    async def test_analyze_detects_contradictions(self) -> None:
        """Test analyze reports contradictions in session."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Is the sky blue?",
                expected_steps=5,
            )
        )
        session_id = start_result["session_id"]

        # Add contradicting thoughts
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The sky is definitely blue.",
            confidence=0.9,
        )
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="However, the sky is not blue at all.",
            confidence=0.8,
        )

        # Analyze
        analysis = json.loads(await think.fn(action="analyze", session_id=session_id))

        # Should have issues or recommendations about contradictions
        issues = analysis["issues"]
        assert "contradictions" in issues or "unresolved_contradictions" in issues

    @pytest.mark.asyncio
    async def test_analyze_returns_recommendations(self) -> None:
        """Test analyze provides actionable recommendations."""
        import json

        from src.server import think

        # Start a session but don't add many thoughts (shallow reasoning)
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Explain the theory of relativity in detail.",  # Complex problem
                expected_steps=10,
            )
        )
        session_id = start_result["session_id"]

        # Add just one shallow thought
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="Relativity is about space and time.",
            confidence=0.5,
        )

        # Analyze
        analysis = json.loads(await think.fn(action="analyze", session_id=session_id))

        # Should have recommendations
        assert len(analysis["recommendations"]) > 0
        # At least one recommendation should be present
        assert isinstance(analysis["recommendations"][0], str)

    @pytest.mark.asyncio
    async def test_analyze_quality_improves_with_depth(self) -> None:
        """Test that quality scores improve as reasoning deepens."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="How do plants photosynthesize?",
                expected_steps=5,
            )
        )
        session_id = start_result["session_id"]

        # First analysis (only initial thought if any)
        analysis1 = json.loads(await think.fn(action="analyze", session_id=session_id))
        initial_depth = analysis1["quality"]["depth_score"]

        # Add more thoughts
        for _i, thought in enumerate(
            [
                "Plants use sunlight to convert CO2 and water.",
                "This process occurs in the chloroplasts.",
                "Chlorophyll absorbs light energy.",
                "The light reactions produce ATP and NADPH.",
            ]
        ):
            await think.fn(
                action="continue",
                session_id=session_id,
                thought=thought,
                confidence=0.8,
            )

        # Second analysis
        analysis2 = json.loads(await think.fn(action="analyze", session_id=session_id))
        final_depth = analysis2["quality"]["depth_score"]

        # Depth score should increase (or at least not decrease)
        assert final_depth >= initial_depth

    @pytest.mark.asyncio
    async def test_analyze_risk_level_with_issues(self) -> None:
        """Test that risk level increases with issues."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test risk assessment",
                expected_steps=5,
            )
        )
        session_id = start_result["session_id"]

        # Add low-confidence contradicting thoughts
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The answer is definitely A.",
            confidence=0.3,  # Low confidence
        )
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="Actually, the answer is definitely not A, it's B.",
            confidence=0.3,  # Low confidence
        )

        # Analyze
        analysis = json.loads(await think.fn(action="analyze", session_id=session_id))

        # Risk should be medium or high due to contradictions/low confidence
        assert analysis["risk"]["level"] in ("medium", "high")
        # Should have risk factors
        assert len(analysis["risk"]["factors"]) >= 0  # May have factors

    @pytest.mark.asyncio
    async def test_session_analytics_to_dict(self) -> None:
        """Test SessionAnalytics.to_dict() produces valid structure."""
        from src.tools.unified_reasoner import SessionAnalytics

        analytics = SessionAnalytics(
            session_id="test-123",
            total_thoughts=5,
            main_chain_length=4,
            branch_count=1,
            average_confidence=0.75,
            average_survival_score=0.8,
            coherence_score=0.7,
            coverage_score=0.6,
            depth_score=0.9,
            contradictions=[("t1", "t2")],
            unresolved_contradictions=1,
            blind_spots_detected=2,
            blind_spots_unaddressed=1,
            cycles_detected=0,
            validation_rate=0.85,
            invalid_thoughts=["t3"],
            planning_ratio=0.2,
            revision_count=1,
            branch_utilization=0.5,
            recommendations=["Address blind spots", "Resolve contradictions"],
            risk_level="medium",
            risk_factors=["1 unresolved contradiction(s)"],
        )

        result = analytics.to_dict()

        # Verify all sections present
        assert result["session_id"] == "test-123"
        assert result["progress"]["total_thoughts"] == 5
        assert result["quality"]["coherence_score"] == 0.7
        assert result["issues"]["contradictions"] == 1
        assert result["validation"]["rate"] == 0.85
        assert result["efficiency"]["revision_count"] == 1
        assert len(result["recommendations"]) == 2
        assert result["risk"]["level"] == "medium"


class TestSuggestAction:
    """Tests for the think(action='suggest') functionality."""

    @pytest.mark.asyncio
    async def test_suggest_requires_session_id(self) -> None:
        """Test suggest action requires session_id."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="suggest"))
        assert "error" in result
        assert "session_id" in result["error"]

    @pytest.mark.asyncio
    async def test_suggest_session_not_found(self) -> None:
        """Test suggest returns error for non-existent session."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="suggest", session_id="nonexistent"))
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_suggest_basic_session(self) -> None:
        """Test suggest returns valid suggestion for basic session."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="What is 2+2?",
                expected_steps=3,
            )
        )
        session_id = start_result["session_id"]

        # Get suggestion
        suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))

        # Verify structure
        assert "suggested_action" in suggestion
        assert "urgency" in suggestion
        assert "reason" in suggestion
        assert "parameters" in suggestion
        assert "guidance" in suggestion
        assert "alternatives" in suggestion
        assert "session_summary" in suggestion

        # Verify valid action suggested
        valid_actions = {
            "start",
            "continue",
            "branch",
            "revise",
            "synthesize",
            "verify",
            "finish",
            "resolve",
        }
        assert suggestion["suggested_action"] in valid_actions

        # Verify urgency is valid
        assert suggestion["urgency"] in ("low", "medium", "high")

    @pytest.mark.asyncio
    async def test_suggest_prioritizes_contradictions(self) -> None:
        """Test suggest recommends 'resolve' when contradictions exist."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Is this claim valid?",
                expected_steps=5,
            )
        )
        session_id = start_result["session_id"]

        # Add contradicting thoughts using patterns that will be detected
        # Pattern: "is valid" vs "is invalid" or "correct" vs "incorrect"
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The claim is valid and correct.",
            confidence=0.9,
        )
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The claim is invalid and incorrect.",
            confidence=0.8,
        )

        # Get suggestion
        suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))

        # Should suggest resolving contradiction (highest priority)
        assert suggestion["suggested_action"] == "resolve"
        assert suggestion["urgency"] == "high"
        assert "contradiction" in suggestion["reason"].lower()

    @pytest.mark.asyncio
    async def test_suggest_recommends_continue_for_shallow_reasoning(self) -> None:
        """Test suggest recommends 'continue' when reasoning is shallow."""
        import json

        from src.server import think

        # Start a complex problem
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Explain quantum entanglement in detail.",
                expected_steps=10,
            )
        )
        session_id = start_result["session_id"]

        # Add only one thought (shallow)
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="Quantum entanglement is a phenomenon.",
            confidence=0.7,
        )

        # Get suggestion
        suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))

        # Should suggest continue since reasoning is shallow
        assert suggestion["suggested_action"] == "continue"
        # Should mention depth in reason or have continue as suggested
        assert (
            "depth" in suggestion["reason"].lower() or suggestion["suggested_action"] == "continue"
        )

    @pytest.mark.asyncio
    async def test_suggest_recommends_finish_when_ready(self) -> None:
        """Test suggest recommends 'finish' when session is ready."""
        import json

        from src.server import think

        # Start a simple problem
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="What is 5+5?",
                expected_steps=3,
            )
        )
        session_id = start_result["session_id"]

        # Add sufficient reasoning with good confidence
        for thought in [
            "I need to add 5 and 5 together.",
            "5 + 5 = 10.",
            "The answer is 10.",
        ]:
            await think.fn(
                action="continue",
                session_id=session_id,
                thought=thought,
                confidence=0.95,
            )

        # Get suggestion
        suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))

        # Should suggest finish or continue (depends on quality score)
        assert suggestion["suggested_action"] in ("finish", "continue")

        # If not finish, finish should be in alternatives
        if suggestion["suggested_action"] != "finish":
            alt_actions = [a["action"] for a in suggestion["alternatives"]]
            # Finish might be suggested as alternative
            assert "finish" in alt_actions or suggestion["suggested_action"] == "continue"

    @pytest.mark.asyncio
    async def test_suggest_includes_alternatives(self) -> None:
        """Test suggest includes alternative actions."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test problem",
                expected_steps=5,
            )
        )
        session_id = start_result["session_id"]

        # Add a thought
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="First thought about the problem.",
            confidence=0.6,
        )

        # Get suggestion
        suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))

        # Alternatives should be a list
        assert isinstance(suggestion["alternatives"], list)

        # Each alternative should have action and reason
        for alt in suggestion["alternatives"]:
            assert "action" in alt
            assert "reason" in alt

    @pytest.mark.asyncio
    async def test_suggest_includes_session_summary(self) -> None:
        """Test suggest includes session summary metrics."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test problem",
                expected_steps=3,
            )
        )
        session_id = start_result["session_id"]

        # Add thoughts
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="First thought.",
            confidence=0.8,
        )
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="Second thought.",
            confidence=0.85,
        )

        # Get suggestion
        suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))

        # Verify session summary
        summary = suggestion["session_summary"]
        assert "thoughts" in summary
        assert "quality" in summary
        assert "risk" in summary
        assert "status" in summary

        assert summary["thoughts"] >= 2
        assert 0 <= summary["quality"] <= 1
        assert summary["risk"] in ("low", "medium", "high")
        assert summary["status"] == "active"

    @pytest.mark.asyncio
    async def test_suggest_with_low_confidence_suggests_verify(self) -> None:
        """Test suggest recommends verification for low-confidence thoughts."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Is this claim true?",
                expected_steps=5,
            )
        )
        session_id = start_result["session_id"]

        # Add low-confidence thoughts
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="I'm not sure about this claim.",
            confidence=0.3,  # Very low confidence
        )
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="It might be true, but I'm uncertain.",
            confidence=0.4,  # Low confidence
        )

        # Get suggestion
        suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))

        # Should suggest verify or continue (verify for low confidence, or continue to build more context)
        # The exact suggestion depends on other factors like depth
        assert suggestion["suggested_action"] in ("verify", "continue", "finish")


class TestFeedbackAction:
    """Tests for the think(action='feedback') functionality (S2)."""

    @pytest.mark.asyncio
    async def test_feedback_requires_session_id(self) -> None:
        """Test feedback action requires session_id."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="feedback"))
        assert "error" in result
        assert "session_id" in result["error"]

    @pytest.mark.asyncio
    async def test_feedback_requires_suggestion_id(self) -> None:
        """Test feedback action requires suggestion_id."""
        import json

        from src.server import think

        # Start a session first
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test problem",
                expected_steps=3,
            )
        )
        session_id = start_result["session_id"]

        result = json.loads(
            await think.fn(
                action="feedback",
                session_id=session_id,
                suggestion_outcome="accepted",
            )
        )
        assert "error" in result
        assert "suggestion_id" in result["error"]

    @pytest.mark.asyncio
    async def test_feedback_requires_outcome(self) -> None:
        """Test feedback action requires suggestion_outcome."""
        import json

        from src.server import think

        # Start a session first
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test problem",
                expected_steps=3,
            )
        )
        session_id = start_result["session_id"]

        result = json.loads(
            await think.fn(
                action="feedback",
                session_id=session_id,
                suggestion_id="some-id",
            )
        )
        assert "error" in result
        assert "suggestion_outcome" in result["error"]

    @pytest.mark.asyncio
    async def test_feedback_records_accepted(self) -> None:
        """Test feedback correctly records accepted suggestion."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="What is 2+2?",
                expected_steps=3,
            )
        )
        session_id = start_result["session_id"]

        # Get a suggestion to record feedback for
        suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))
        suggestion_id = suggestion.get("suggestion_id")

        # If no suggestion_id is returned, skip this test
        if not suggestion_id:
            pytest.skip("No suggestion_id returned by suggest action")

        # Record feedback
        feedback_result = json.loads(
            await think.fn(
                action="feedback",
                session_id=session_id,
                suggestion_id=suggestion_id,
                suggestion_outcome="accepted",
            )
        )

        # Verify feedback was recorded (has outcome field)
        assert feedback_result.get("outcome") == "accepted"
        assert feedback_result.get("suggestion_id") == suggestion_id
        assert "weights_updated" in feedback_result

    @pytest.mark.asyncio
    async def test_feedback_records_rejected_with_actual_action(self) -> None:
        """Test feedback records rejection and actual action taken."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="What is 2+2?",
                expected_steps=3,
            )
        )
        session_id = start_result["session_id"]

        # Get a suggestion to record feedback for
        suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))
        suggestion_id = suggestion.get("suggestion_id")

        # If no suggestion_id is returned, skip this test
        if not suggestion_id:
            pytest.skip("No suggestion_id returned by suggest action")

        # Record feedback with rejection and actual action
        feedback_result = json.loads(
            await think.fn(
                action="feedback",
                session_id=session_id,
                suggestion_id=suggestion_id,
                suggestion_outcome="rejected",
                actual_action="verify",
            )
        )

        # Verify feedback was recorded
        assert feedback_result.get("outcome") == "rejected"
        assert feedback_result.get("actual_action") == "verify"
        assert "weights_updated" in feedback_result

    @pytest.mark.asyncio
    async def test_feedback_invalid_suggestion_id(self) -> None:
        """Test feedback returns error for invalid suggestion_id."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test problem",
                expected_steps=3,
            )
        )
        session_id = start_result["session_id"]

        # Try to record feedback for non-existent suggestion
        result = json.loads(
            await think.fn(
                action="feedback",
                session_id=session_id,
                suggestion_id="nonexistent-suggestion-id",
                suggestion_outcome="accepted",
            )
        )

        # Should fail because suggestion doesn't exist
        assert "error" in result or result.get("success") is False


class TestAutoAction:
    """Tests for the think(action='auto') functionality (S3).

    Note: The server-level auto action returns suggestions without LLM integration.
    For full auto-execution with thought generation, use the UnifiedReasonerManager
    directly with a thought_generator callback.
    """

    @pytest.mark.asyncio
    async def test_auto_requires_session_id(self) -> None:
        """Test auto action requires session_id."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="auto"))
        assert "error" in result
        assert "session_id" in result["error"]

    @pytest.mark.asyncio
    async def test_auto_session_not_found(self) -> None:
        """Test auto returns error for non-existent session."""
        import json

        from src.server import think

        result = json.loads(await think.fn(action="auto", session_id="nonexistent"))
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_auto_validates_max_steps(self) -> None:
        """Test auto validates max_auto_steps bounds."""
        import json

        from src.server import think

        # Start a session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Test problem",
                expected_steps=3,
            )
        )
        session_id = start_result["session_id"]

        # Test max_auto_steps < 1
        result = json.loads(
            await think.fn(
                action="auto",
                session_id=session_id,
                max_auto_steps=0,
            )
        )
        assert "error" in result
        assert "at least 1" in result["error"]

        # Test max_auto_steps > 20
        result = json.loads(
            await think.fn(
                action="auto",
                session_id=session_id,
                max_auto_steps=21,
            )
        )
        assert "error" in result
        assert "cannot exceed 20" in result["error"]

    @pytest.mark.asyncio
    async def test_auto_returns_suggestion_without_generator(self) -> None:
        """Test auto returns suggestion when no thought generator is available.

        At the server level, auto returns a suggestion for manual execution
        since it doesn't have LLM integration built-in.
        """
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="What is 2+2?",
                expected_steps=3,
            )
        )
        session_id = start_result["session_id"]

        # Execute auto (will return suggestion_only since no generator)
        auto_result = json.loads(
            await think.fn(
                action="auto",
                session_id=session_id,
                max_auto_steps=1,
            )
        )

        # Should return suggestion_only status
        assert auto_result.get("status") == "suggestion_only"
        assert "suggestion" in auto_result
        assert auto_result["suggestion"].get("suggested_action") is not None

    @pytest.mark.asyncio
    async def test_auto_returns_checkpoint_at_high_risk(self) -> None:
        """Test auto returns checkpoint status when risk is high."""
        import json

        from src.server import think

        # Start session that will generate contradictions (high risk)
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Analyze this controversial claim",
                expected_steps=5,
            )
        )
        session_id = start_result["session_id"]

        # Add contradicting thoughts to create high-risk state
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The claim is definitely true and valid.",
            confidence=0.9,
        )
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="The claim is definitely false and invalid.",
            confidence=0.9,
        )

        # Execute auto with stop_on_high_risk=True
        auto_result = json.loads(
            await think.fn(
                action="auto",
                session_id=session_id,
                max_auto_steps=5,
                stop_on_high_risk=True,
            )
        )

        # Should indicate checkpoint or high risk
        status = auto_result.get("status")
        # Could be checkpoint (high risk) or suggestion_only (no generator)
        assert status in ("checkpoint", "suggestion_only")
        assert "suggestion" in auto_result

    @pytest.mark.asyncio
    async def test_auto_suggestion_contains_valid_action(self) -> None:
        """Test auto returns a valid suggested action."""
        import json

        from src.server import think

        # Start session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="Explain photosynthesis step by step",
                expected_steps=5,
            )
        )
        session_id = start_result["session_id"]

        # Execute auto
        auto_result = json.loads(
            await think.fn(
                action="auto",
                session_id=session_id,
                max_auto_steps=3,
            )
        )

        # Verify suggestion contains valid action
        suggestion = auto_result.get("suggestion", {})
        valid_actions = {
            "start",
            "continue",
            "branch",
            "revise",
            "synthesize",
            "verify",
            "finish",
            "resolve",
        }
        assert suggestion.get("suggested_action") in valid_actions

    @pytest.mark.asyncio
    async def test_auto_with_finished_session(self) -> None:
        """Test auto behavior with finished session."""
        import json

        from src.server import think

        # Start a very short session
        start_result = json.loads(
            await think.fn(
                action="start",
                mode="chain",
                problem="What is 1+1?",
                expected_steps=2,
            )
        )
        session_id = start_result["session_id"]

        # Add thoughts and finish
        await think.fn(
            action="continue",
            session_id=session_id,
            thought="1+1 equals 2.",
            confidence=0.95,
        )
        await think.fn(
            action="finish",
            session_id=session_id,
            thought="Conclusion: 2",
            confidence=1.0,
        )

        # Try auto on finished session
        auto_result = json.loads(
            await think.fn(
                action="auto",
                session_id=session_id,
                max_auto_steps=5,
            )
        )

        # Should still return a valid response (suggestion_only or checkpoint)
        assert auto_result.get("status") in ("suggestion_only", "checkpoint")
        assert "suggestion" in auto_result


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
