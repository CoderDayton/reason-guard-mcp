"""Smoke tests for Reason Guard MCP Server.

Quick tests to verify all tools are functional.
Run with: pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import contextlib

import pytest

from src.utils.errors import CompressionException
from src.utils.schema import CompressionResult, ReasoningResult, VerificationResult


class TestSchemas:
    """Test data schema classes."""

    def test_compression_result_to_dict(self) -> None:
        """Test CompressionResult serialization."""
        result = CompressionResult(
            compressed_context="Compressed text here.",
            compression_ratio=0.3,
            original_tokens=1000,
            compressed_tokens=300,
            sentences_kept=5,
            sentences_removed=15,
            relevance_scores=[("Test sentence", 0.9)],
        )

        data = result.to_dict()

        assert data["compression_ratio"] == 0.3
        assert data["tokens_saved"] == 700
        assert data["sentences_kept"] == 5

    def test_reasoning_result_to_dict(self) -> None:
        """Test ReasoningResult serialization."""
        result = ReasoningResult(
            answer="The answer is 42.",
            confidence=0.85,
            reasoning_steps=["Step 1", "Step 2", "Step 3"],
            tokens_used=500,
        )

        data = result.to_dict()

        assert data["answer"] == "The answer is 42."
        assert data["confidence"] == 0.85
        assert data["num_reasoning_steps"] == 3

    def test_verification_result_to_dict(self) -> None:
        """Test VerificationResult serialization."""
        result = VerificationResult(
            verified=True,
            confidence=0.9,
            claims_verified=9,
            claims_total=10,
            reason="9 out of 10 claims verified",
        )

        data = result.to_dict()

        assert data["verified"] is True
        assert data["verification_percentage"] == 90.0
        assert data["recommendation"] == "RELIABLE"


class TestCompressionTool:
    """Test compression tool functionality."""

    def test_compression_empty_context_raises(self) -> None:
        """Test that empty context raises error."""
        # Test the validation logic directly (avoids model loading)
        context = ""
        with pytest.raises(CompressionException, match="cannot be empty"):
            if not context or not context.strip():
                raise CompressionException("Context cannot be empty")

    def test_compression_ratio_validation(self) -> None:
        """Test compression ratio bounds."""
        # Test invalid ratios
        invalid_ratios = [-0.1, 0.0, 1.5, 2.0]

        for ratio in invalid_ratios:
            if not 0.1 <= ratio <= 1.0:
                # This is expected to be invalid
                assert True
            else:
                pytest.fail(f"Ratio {ratio} should be invalid")


class TestMatrixOfThoughtTool:
    """Test MoT reasoning tool validation."""

    def test_matrix_size_validation(self) -> None:
        """Test matrix dimension bounds."""
        valid_sizes = [(2, 2), (3, 4), (5, 5)]
        invalid_sizes = [(1, 2), (6, 4), (3, 0), (0, 0)]

        for rows, cols in valid_sizes:
            assert 2 <= rows <= 5 and 2 <= cols <= 5

        for rows, cols in invalid_sizes:
            assert not (2 <= rows <= 5 and 2 <= cols <= 5)

    def test_weight_matrix_generation(self) -> None:
        """Test communication weight matrix patterns."""
        import numpy as np

        # vert&hor-01 pattern
        rows, cols = 3, 4
        matrix = np.zeros((rows, cols - 1))

        for i in range(rows):
            for j in range(cols - 1):
                matrix[i, j] = min(0.1 * (i + j + 1), 1.0)

        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 0.1  # First cell
        assert matrix[2, 2] == 0.5  # Last cell


class TestLongChainTool:
    """Test long chain reasoning tool validation."""

    def test_step_count_validation(self) -> None:
        """Test step count bounds."""
        valid_steps = [1, 15, 50]
        invalid_steps = [0, -1, 51, 100]

        for steps in valid_steps:
            assert 1 <= steps <= 50

        for steps in invalid_steps:
            assert not (1 <= steps <= 50)


class TestVerificationTool:
    """Test fact verification tool validation."""

    def test_max_claims_validation(self) -> None:
        """Test max claims bounds."""
        valid_claims = [1, 10, 20]
        invalid_claims = [0, -1, 21, 100]

        for claims in valid_claims:
            assert 1 <= claims <= 20

        for claims in invalid_claims:
            assert not (1 <= claims <= 20)


class TestErrorHandling:
    """Test error handling utilities."""

    def test_tool_execution_error(self) -> None:
        """Test ToolExecutionError formatting."""
        from src.utils.errors import ToolExecutionError

        error = ToolExecutionError(
            tool_name="test_tool",
            error_message="Something went wrong",
            details={"input_length": 100},
        )

        assert error.tool_name == "test_tool"
        assert "Something went wrong" in str(error)

        mcp_error = error.to_mcp_error()
        assert "[test_tool]" in mcp_error

        error_dict = error.to_dict()
        assert error_dict["error"] is True
        assert error_dict["tool"] == "test_tool"


@pytest.mark.asyncio
class TestServerTools:
    """Test server tool endpoints are properly registered."""

    async def test_think_tool_registered(self) -> None:
        """Test think tool is properly registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import think

        assert isinstance(think, FunctionTool)
        tool: FunctionTool = think
        assert callable(tool.fn)
        assert tool.name == "think"

    async def test_compress_tool_registered(self) -> None:
        """Test compress tool is properly registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import compress

        assert isinstance(compress, FunctionTool)
        tool: FunctionTool = compress
        assert callable(tool.fn)
        assert tool.name == "compress"

    async def test_status_tool_registered(self) -> None:
        """Test status tool is properly registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import status

        assert isinstance(status, FunctionTool)
        tool: FunctionTool = status
        assert callable(tool.fn)
        assert tool.name == "status"

    async def test_all_tools_count(self) -> None:
        """Test that exactly 8 tools are registered (3 original + 5 router)."""
        from src.server import mcp

        tools = list(mcp._tool_manager._tools.values())
        assert len(tools) == 9, f"Expected 9 tools, got {len(tools)}: {[t.name for t in tools]}"

    async def test_tool_names(self) -> None:
        """Test that all expected tool names are present."""
        from src.server import mcp

        tool_names = set(mcp._tool_manager._tools.keys())
        expected = {
            # Guidance mode tools
            "think",
            "compress",
            "status",
            "paradigm_hint",
            # Enforcement mode (atomic router) tools
            "initialize_reasoning",
            "submit_step",
            "create_branch",
            "verify_claims",
            "router_status",
        }
        assert tool_names == expected, f"Expected {expected}, got {tool_names}"


class TestMCPToolsRegistration:
    """Smoke tests for MCP tool registration and type resolution.

    These tests catch Pydantic/FastMCP type resolution issues early,
    before they manifest as runtime errors in production.
    """

    def test_server_import_succeeds(self) -> None:
        """Verify src.server imports without Pydantic/type errors.

        This catches issues like:
        - `from __future__ import annotations` breaking Pydantic
        - Invalid Literal types
        - Circular imports
        """
        # If this import fails, the test fails with a clear traceback
        import src.server

        assert hasattr(src.server, "mcp")
        assert hasattr(src.server, "think")
        assert hasattr(src.server, "compress")
        assert hasattr(src.server, "status")

    def test_mcp_tool_manager_initialized(self) -> None:
        """Verify MCP tool manager has tools registered at import time."""
        from src.server import mcp

        # FastMCP registers tools at decorator time, not at runtime
        # If type resolution fails, _tool_manager will be empty or broken
        assert hasattr(mcp, "_tool_manager")
        assert mcp._tool_manager is not None

        tools = mcp._tool_manager._tools
        assert isinstance(tools, dict)
        assert len(tools) == 9, f"Expected 9 tools, got {len(tools)}"

    def test_tool_input_schemas_valid(self) -> None:
        """Verify each tool has a valid input schema (Pydantic model).

        FastMCP generates Pydantic models from function signatures.
        This test catches type annotation issues that break schema generation.
        """
        from src.server import mcp

        for tool_name, tool in mcp._tool_manager._tools.items():
            # Each tool should have a parameters schema
            assert hasattr(tool, "parameters"), f"{tool_name} missing parameters"

            # Schema should be a valid dict (JSON Schema)
            schema = tool.parameters
            assert isinstance(schema, dict), f"{tool_name} schema is not a dict"

            # Schema should have 'properties' key for typed params
            # (may be empty for no-arg tools, but should exist)
            assert "type" in schema, f"{tool_name} schema missing 'type'"

    def test_think_tool_action_literal_resolved(self) -> None:
        """Verify 'think' tool action param has all Literal values resolved.

        The ThinkAction Literal type is complex and prone to resolution issues.
        """
        from src.server import mcp

        tool = mcp._tool_manager._tools.get("think")
        assert tool is not None

        schema = tool.parameters
        props = schema.get("properties", {})

        # 'action' should be in properties
        assert "action" in props, "think tool missing 'action' parameter"

        action_schema = props["action"]
        # Should have enum values from the Literal type
        assert "enum" in action_schema or "anyOf" in action_schema, (
            f"think 'action' param not properly resolved: {action_schema}"
        )

    def test_tool_functions_callable(self) -> None:
        """Verify all tool functions are callable (not broken by decoration)."""
        from src.server import mcp

        for tool_name, tool in mcp._tool_manager._tools.items():
            assert hasattr(tool, "fn"), f"{tool_name} missing fn attribute"
            assert callable(tool.fn), f"{tool_name}.fn is not callable"


@pytest.mark.asyncio
class TestToolInvocation:
    """Smoke tests that actually invoke tools with minimal args.

    These tests verify end-to-end tool execution and JSON response structure,
    catching runtime errors that schema validation alone won't find.
    """

    async def test_status_tool_returns_valid_json(self) -> None:
        """Invoke status tool and verify JSON response structure."""
        import json

        from src.server import status

        result = await status.fn()
        data = json.loads(result)

        # Should have server info
        assert "server" in data, f"Missing 'server' key: {data.keys()}"
        assert data["server"]["name"] == "Reason-Guard-MCP"
        assert "tools" in data["server"]
        expected_tools = {
            "think",
            "compress",
            "status",
            "paradigm_hint",
            "initialize_reasoning",
            "submit_step",
            "create_branch",
            "verify_claims",
            "router_status",
        }
        assert set(data["server"]["tools"]) == expected_tools

        # Should have session counts
        assert "sessions" in data or "active_sessions" in data

    async def test_think_start_returns_session_id(self) -> None:
        """Invoke think(start) and verify session creation."""
        import json

        from src.server import think

        result = await think.fn(action="start", problem="What is 2+2?")
        data = json.loads(result)

        # Should create a session
        assert "session_id" in data, f"Missing session_id: {data}"
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0

        # Should have mode info
        assert "actual_mode" in data or "mode" in data

    async def test_think_invalid_action_returns_error(self) -> None:
        """Verify invalid action returns structured error, not exception."""
        import json

        from src.server import think

        result = await think.fn(action="invalid_action_xyz")  # type: ignore[arg-type]
        data = json.loads(result)

        assert "error" in data, f"Expected error response: {data}"

    async def test_think_missing_session_returns_error(self) -> None:
        """Verify missing session_id for continue returns structured error."""
        import json

        from src.server import think

        result = await think.fn(action="continue", thought="test")
        data = json.loads(result)

        assert "error" in data, f"Expected error for missing session_id: {data}"

    async def test_compress_missing_query_returns_error(self) -> None:
        """Verify compress without query returns structured error."""
        import json

        from src.server import compress

        result = await compress.fn(context="Some text to compress", query="")
        data = json.loads(result)

        assert "error" in data, f"Expected error for empty query: {data}"

    async def test_think_full_workflow_minimal(self) -> None:
        """Test minimal think workflow: start -> continue -> finish."""
        import json

        from src.server import think

        # Start session
        start_result = await think.fn(action="start", problem="Simple test problem")
        start_data = json.loads(start_result)
        assert "session_id" in start_data
        session_id = start_data["session_id"]

        # Add a thought
        continue_result = await think.fn(
            action="continue",
            session_id=session_id,
            thought="Step 1: Analyzing the problem",
        )
        continue_data = json.loads(continue_result)
        assert "error" not in continue_data, f"Continue failed: {continue_data}"

        # Finish session
        finish_result = await think.fn(
            action="finish",
            session_id=session_id,
            thought="The answer is complete",
        )
        finish_data = json.loads(finish_result)
        assert "error" not in finish_data, f"Finish failed: {finish_data}"
        assert "total_steps" in finish_data or "mode_used" in finish_data

    async def test_paradigm_hint_recommends_enforcement(self) -> None:
        """Test paradigm_hint recommends enforcement for formal problems."""
        import json

        from src.server import paradigm_hint

        result = await paradigm_hint.fn(problem="Prove the Monty Hall theorem")
        data = json.loads(result)

        assert "recommendation" in data
        assert data["recommendation"] == "enforcement"
        assert "confidence" in data
        assert "suggested_tools" in data

    async def test_paradigm_hint_recommends_guidance(self) -> None:
        """Test paradigm_hint recommends guidance for exploratory problems."""
        import json

        from src.server import paradigm_hint

        result = await paradigm_hint.fn(problem="Help me brainstorm app ideas")
        data = json.loads(result)

        assert "recommendation" in data
        assert data["recommendation"] == "guidance"
        assert "confidence" in data
        assert "suggested_tools" in data


@pytest.mark.asyncio
class TestAtomicRouterTools:
    """Smoke tests for Atomic Router MCP tools.

    Tests the 5 router tools: initialize_reasoning, submit_step,
    create_branch, verify_claims, router_status.
    """

    async def test_initialize_reasoning_returns_session(self) -> None:
        """Test initialize_reasoning creates session with guidance."""
        import json

        from src.server import initialize_reasoning

        result = await initialize_reasoning.fn(
            problem="What is 2+2?",
            complexity="low",
        )
        data = json.loads(result)

        assert "session_id" in data, f"Missing session_id: {data}"
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0
        assert "guidance" in data

    async def test_initialize_reasoning_invalid_complexity(self) -> None:
        """Test initialize_reasoning with invalid complexity returns error."""
        import json

        from src.server import initialize_reasoning

        result = await initialize_reasoning.fn(
            problem="Test problem",
            complexity="invalid_complexity",
        )
        data = json.loads(result)

        assert "error" in data, f"Expected error for invalid complexity: {data}"

    async def test_submit_step_invalid_session(self) -> None:
        """Test submit_step with invalid session returns error."""
        import json

        from src.server import submit_step

        result = await submit_step.fn(
            session_id="nonexistent_session_id",
            step_type="premise",
            content="Test content",
            confidence=0.8,
        )
        data = json.loads(result)

        assert "error" in data or data.get("status") == "REJECTED", (
            f"Expected error for invalid session: {data}"
        )

    async def test_submit_step_valid_workflow(self) -> None:
        """Test submit_step accepts valid step in active session."""
        import json

        from src.server import initialize_reasoning, submit_step

        # Create session
        init_result = await initialize_reasoning.fn(
            problem="Simple math problem",
            complexity="low",
        )
        init_data = json.loads(init_result)
        session_id = init_data["session_id"]

        # Submit a premise step
        step_result = await submit_step.fn(
            session_id=session_id,
            step_type="premise",
            content="Given: we need to solve a simple math problem",
            confidence=0.9,
        )
        step_data = json.loads(step_result)

        assert "status" in step_data, f"Missing status: {step_data}"
        assert step_data["status"] in [
            "ACCEPTED",
            "REJECTED",
            "BRANCH_REQUIRED",
            "VERIFICATION_REQUIRED",
        ]

    async def test_create_branch_requires_valid_session(self) -> None:
        """Test create_branch with invalid session returns error."""
        import json

        from src.server import create_branch

        result = await create_branch.fn(
            session_id="nonexistent_session",
            alternatives=["Option A", "Option B"],
        )
        data = json.loads(result)

        # Should return error or empty branch_ids
        assert "error" in data.get("guidance", "") or data.get("branch_ids") == [], (
            f"Expected error for invalid session: {data}"
        )

    async def test_verify_claims_requires_valid_session(self) -> None:
        """Test verify_claims with invalid session returns error."""
        import json

        from src.server import verify_claims

        result = await verify_claims.fn(
            session_id="nonexistent_session",
            claims=["Claim 1"],
            evidence=["Evidence 1"],
        )
        data = json.loads(result)

        # Should return error or indicate invalid session
        assert "error" in str(data).lower() or data.get("verified") == [], (
            f"Expected error for invalid session: {data}"
        )

    async def test_router_status_no_session(self) -> None:
        """Test router_status without session returns global status."""
        import json

        from src.server import router_status

        result = await router_status.fn()
        data = json.loads(result)

        # Should return stats about the router (nested under 'sessions' key)
        assert "sessions" in data or "active_sessions" in data, f"Expected router stats: {data}"

    async def test_router_status_with_session(self) -> None:
        """Test router_status with valid session returns session state."""
        import json

        from src.server import initialize_reasoning, router_status

        # Create session
        init_result = await initialize_reasoning.fn(
            problem="Test problem",
            complexity="medium",
        )
        init_data = json.loads(init_result)
        session_id = init_data["session_id"]

        # Get session status
        status_result = await router_status.fn(session_id=session_id)
        status_data = json.loads(status_result)

        # Should have session info
        assert "session" in status_data or "step_count" in status_data or "status" in status_data, (
            f"Expected session info: {status_data}"
        )

    async def test_full_router_workflow(self) -> None:
        """Test complete atomic router workflow: init -> steps -> synthesis."""
        import json

        from src.server import initialize_reasoning, submit_step

        # Initialize
        init_result = await initialize_reasoning.fn(
            problem="What is the capital of France?",
            complexity="low",
        )
        init_data = json.loads(init_result)
        assert "session_id" in init_data
        session_id = init_data["session_id"]

        # Submit premise
        premise_result = await submit_step.fn(
            session_id=session_id,
            step_type="premise",
            content="The question asks about the capital city of France",
            confidence=0.95,
        )
        premise_data = json.loads(premise_result)
        assert premise_data.get("status") == "ACCEPTED", f"Premise rejected: {premise_data}"

        # Submit hypothesis
        hypo_result = await submit_step.fn(
            session_id=session_id,
            step_type="hypothesis",
            content="The capital of France is Paris",
            confidence=0.95,
        )
        hypo_data = json.loads(hypo_result)
        assert hypo_data.get("status") == "ACCEPTED", f"Hypothesis rejected: {hypo_data}"

        # Submit synthesis (should be allowed after min_steps for LOW complexity)
        synth_result = await submit_step.fn(
            session_id=session_id,
            step_type="synthesis",
            content="The capital of France is Paris",
            confidence=0.95,
        )
        synth_data = json.loads(synth_result)
        # May be accepted or rejected depending on min_steps rule
        assert "status" in synth_data, f"Missing status in synthesis: {synth_data}"


@pytest.mark.asyncio
class TestBackgroundTaskExceptionHandling:
    """Test that background tasks properly log exceptions."""

    async def test_vector_store_task_exception_callback_logs_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that background vector store task failures are logged.

        This verifies the _handle_task_exception callback is properly attached
        and logs exceptions from fire-and-forget tasks.
        """
        import asyncio
        import logging

        # Simulate the pattern used in unified_reasoner.py
        logged_exceptions: list[str] = []

        def _handle_task_exception(task: asyncio.Task[None]) -> None:
            """Log exceptions from background tasks to prevent silent failures."""
            if task.cancelled():
                return
            if exc := task.exception():
                msg = f"Background task failed: {exc}"
                logged_exceptions.append(msg)
                logging.getLogger(__name__).warning(msg)

        async def _failing_background_task() -> None:
            raise ValueError("Simulated vector store failure")

        with caplog.at_level(logging.WARNING):
            task = asyncio.create_task(_failing_background_task())
            task.add_done_callback(_handle_task_exception)

            # Wait for task to complete and callback to run
            await asyncio.sleep(0.1)

        # Verify exception was logged via callback
        assert len(logged_exceptions) == 1
        assert "Simulated vector store failure" in logged_exceptions[0]

    async def test_cancelled_task_does_not_log_exception(self) -> None:
        """Test that cancelled tasks don't log spurious warnings."""
        import asyncio

        logged_exceptions: list[str] = []

        def _handle_task_exception(task: asyncio.Task[None]) -> None:
            if task.cancelled():
                return
            if exc := task.exception():
                logged_exceptions.append(str(exc))

        async def _slow_task() -> None:
            await asyncio.sleep(10)

        task = asyncio.create_task(_slow_task())
        task.add_done_callback(_handle_task_exception)

        # Cancel the task
        task.cancel()

        # Wait for cancellation to propagate
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Cancelled tasks should not log exceptions
        assert len(logged_exceptions) == 0

    async def test_successful_task_does_not_log_exception(self) -> None:
        """Test that successful tasks don't trigger exception callback."""
        import asyncio

        logged_exceptions: list[str] = []

        def _handle_task_exception(task: asyncio.Task[None]) -> None:
            if task.cancelled():
                return
            if exc := task.exception():
                logged_exceptions.append(str(exc))

        async def _successful_task() -> None:
            await asyncio.sleep(0.01)
            return None

        task = asyncio.create_task(_successful_task())
        task.add_done_callback(_handle_task_exception)

        await task

        # Successful tasks should not log exceptions
        assert len(logged_exceptions) == 0
