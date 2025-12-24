"""End-to-end tests for MCP protocol integration.

These tests verify the full MCP protocol flow using FastMCP Client.
"""

from __future__ import annotations

import json

import pytest


class TestMCPProtocol:
    """Test MCP protocol compliance and tool registration."""

    @pytest.mark.asyncio
    async def test_tools_registered(self) -> None:
        """Test that all expected tools are registered."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            tools = await client.list_tools()

            tool_names = {tool.name for tool in tools}
            expected_tools = {
                # Guidance mode tools (4)
                "think",
                "compress",
                "status",
                "paradigm_hint",
                # Enforcement mode tools (5)
                "initialize_reasoning",
                "submit_step",
                "create_branch",
                "verify_claims",
                "router_status",
            }

            missing = expected_tools - tool_names
            extra = tool_names - expected_tools
            assert tool_names == expected_tools, f"Missing: {missing}, Extra: {extra}"

    @pytest.mark.asyncio
    async def test_status_tool(self) -> None:
        """Test status tool returns server info via MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            result = await client.call_tool("status", {})
            assert not result.is_error, f"Tool returned error: {result.data}"

            response = json.loads(result.data)
            assert "server" in response
            assert "active_sessions" in response

    @pytest.mark.asyncio
    async def test_think_chain_workflow_via_mcp(self) -> None:
        """Test complete chain workflow through MCP protocol using think tool."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Step 1: Start chain
            result = await client.call_tool(
                "think",
                {
                    "action": "start",
                    "mode": "chain",
                    "problem": "What is 2 + 2?",
                    "expected_steps": 3,
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "active"  # Unified reasoner uses "active"
            assert response["mode"] == "chain" or response["actual_mode"] == "chain"
            session_id = response["session_id"]

            # Step 2: Add steps with continue action
            result = await client.call_tool(
                "think",
                {
                    "action": "continue",
                    "session_id": session_id,
                    "thought": "First, identify the operands: 2 and 2",
                },
            )
            assert not result.is_error

            result = await client.call_tool(
                "think",
                {
                    "action": "continue",
                    "session_id": session_id,
                    "thought": "Apply addition: 2 + 2 = 4",
                },
            )
            assert not result.is_error

            # Step 3: Finish
            result = await client.call_tool(
                "think",
                {"action": "finish", "session_id": session_id, "thought": "The answer is 4"},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "completed"

    @pytest.mark.asyncio
    async def test_think_matrix_workflow_via_mcp(self) -> None:
        """Test complete matrix workflow through MCP protocol using think tool."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Step 1: Start matrix
            result = await client.call_tool(
                "think",
                {
                    "action": "start",
                    "mode": "matrix",
                    "problem": "What is AI?",
                    "rows": 2,
                    "cols": 2,
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "active"  # Unified reasoner uses "active"
            assert response["mode"] == "matrix" or response["actual_mode"] == "matrix"
            session_id = response["session_id"]

            # Step 2: Set cells with continue action
            for row in range(2):
                for col in range(2):
                    result = await client.call_tool(
                        "think",
                        {
                            "action": "continue",
                            "session_id": session_id,
                            "row": row,
                            "col": col,
                            "thought": f"Thought at ({row}, {col})",
                        },
                    )
                    assert not result.is_error

            # Step 3: Synthesize columns
            result = await client.call_tool(
                "think",
                {
                    "action": "synthesize",
                    "session_id": session_id,
                    "col": 0,
                    "thought": "Column 0 synthesis",
                },
            )
            assert not result.is_error

            result = await client.call_tool(
                "think",
                {
                    "action": "synthesize",
                    "session_id": session_id,
                    "col": 1,
                    "thought": "Column 1 synthesis",
                },
            )
            assert not result.is_error

            # Step 4: Finish
            result = await client.call_tool(
                "think",
                {
                    "action": "finish",
                    "session_id": session_id,
                    "thought": "AI is the simulation of human intelligence",
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "completed"

    @pytest.mark.asyncio
    async def test_think_verify_workflow_via_mcp(self) -> None:
        """Test complete verification workflow through MCP protocol using think tool."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Step 1: Start verification (uses chain mode internally)
            result = await client.call_tool(
                "think",
                {
                    "action": "start",
                    "mode": "verify",
                    "problem": "Einstein was a physicist",
                    "context": "Albert Einstein was a theoretical physicist.",
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "active"  # Unified reasoner uses "active"
            assert response["mode"] == "verify"  # Preserved for backwards compatibility
            session_id = response["session_id"]

            # Step 2: Add reasoning step with continue action
            result = await client.call_tool(
                "think",
                {
                    "action": "continue",
                    "session_id": session_id,
                    "thought": "Einstein was a physicist",
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            # Unified reasoner returns thought_id, not claim_id
            assert "thought_id" in response or "step" in response

            # Step 3: Add verification step
            result = await client.call_tool(
                "think",
                {
                    "action": "verify",
                    "session_id": session_id,
                    "evidence": "Context confirms this claim about Einstein",
                },
            )
            assert not result.is_error

            # Step 4: Finish
            result = await client.call_tool(
                "think",
                {"action": "finish", "session_id": session_id, "thought": "Verified"},
            )
            assert not result.is_error
            response = json.loads(result.data)
            # Unified reasoner returns "completed" status, not "verified"
            assert response["status"] == "completed"

    @pytest.mark.asyncio
    async def test_think_error_handling(self) -> None:
        """Test that think tool errors are properly propagated through MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Call with invalid session_id
            result = await client.call_tool(
                "think",
                {"action": "continue", "session_id": "invalid-session-id", "thought": "Test step"},
            )

            # Should return error in response
            response = json.loads(result.data)
            assert "error" in response

    @pytest.mark.asyncio
    async def test_think_invalid_action(self) -> None:
        """Test that invalid action is rejected by FastMCP validation."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # FastMCP validates Literal types at call time and raises an exception
            with pytest.raises(Exception):  # noqa: B017 - ValidationError wrapped by FastMCP
                await client.call_tool(
                    "think",
                    {"action": "invalid_action"},
                )

    @pytest.mark.asyncio
    async def test_status_with_session(self) -> None:
        """Test status tool with session_id returns session status."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Start a chain session
            result = await client.call_tool(
                "think",
                {"action": "start", "mode": "chain", "problem": "Test problem"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]

            # Check session status
            result = await client.call_tool(
                "status",
                {"session_id": session_id},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "active"

    @pytest.mark.asyncio
    async def test_compress_tool(self) -> None:
        """Test compress tool via MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            long_text = "This is important information about the topic. " * 30

            result = await client.call_tool(
                "compress",
                {"context": long_text, "query": "What is important?", "ratio": 0.3},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert "compressed" in response
            assert response["compression_ratio"] <= 0.5  # Should be compressed


class TestEnforcementModeWorkflow:
    """Test enforcement mode (atomic router) workflow via MCP protocol."""

    @pytest.mark.asyncio
    async def test_enforcement_complete_workflow(self) -> None:
        """Test complete enforcement workflow: initialize -> submit_step -> verify_claims."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Step 1: Initialize reasoning session
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Prove that 2 + 2 = 4", "complexity": "low"},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert "session_id" in response
            assert response["complexity"] == "low"
            assert "min_steps" in response
            assert "max_steps" in response
            session_id = response["session_id"]

            # Step 2: Submit premise step
            result = await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "premise",
                    "content": "We define 2 as the successor of 1",
                    "confidence": 0.95,
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "ACCEPTED"
            assert "valid_next_steps" in response

            # Step 3: Submit hypothesis step
            result = await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "hypothesis",
                    "content": "2 + 2 equals the successor of successor of 2",
                    "confidence": 0.9,
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "ACCEPTED"

            # Step 4: Submit verification step
            result = await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "verification",
                    "content": "By Peano axioms, S(S(2)) = S(3) = 4",
                    "confidence": 0.95,
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "ACCEPTED"

            # Step 5: Verify claims before synthesis
            result = await client.call_tool(
                "verify_claims",
                {
                    "session_id": session_id,
                    "claims": ["2 + 2 = 4"],
                    "evidence": ["Peano axioms define successor function"],
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert "verified" in response

            # Step 6: Submit synthesis step
            result = await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "synthesis",
                    "content": "Therefore, 2 + 2 = 4 is proven",
                    "confidence": 0.99,
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            # Synthesis should be accepted (low complexity = 2-5 steps, we did 4)
            assert response["status"] == "ACCEPTED"

    @pytest.mark.asyncio
    async def test_enforcement_rejects_early_synthesis(self) -> None:
        """Test that enforcement mode rejects synthesis before min_steps."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize with medium complexity (min 4 steps)
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Explain the Monty Hall problem", "complexity": "medium"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]
            min_steps = response["min_steps"]

            # Try to synthesize immediately (should be rejected)
            result = await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "synthesis",
                    "content": "The answer is to switch doors",
                    "confidence": 0.9,
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "REJECTED"
            assert "rejection_reason" in response
            # Should mention needing more steps
            assert (
                str(min_steps) in response["rejection_reason"]
                or "step" in response["rejection_reason"].lower()
            )

    @pytest.mark.asyncio
    async def test_enforcement_branch_required_for_low_confidence(self) -> None:
        """Test that low confidence triggers BRANCH_REQUIRED."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize session
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Solve a probability puzzle", "complexity": "medium"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]
            confidence_threshold = response["confidence_threshold"]

            # Submit premise with high confidence
            await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "premise",
                    "content": "Initial premise",
                    "confidence": 0.95,
                },
            )

            # Submit hypothesis with LOW confidence (below threshold)
            result = await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "hypothesis",
                    "content": "Uncertain hypothesis",
                    "confidence": confidence_threshold - 0.2,  # Below threshold
                },
            )
            response = json.loads(result.data)
            assert response["status"] == "BRANCH_REQUIRED"

            # Create branch to resolve
            result = await client.call_tool(
                "create_branch",
                {
                    "session_id": session_id,
                    "alternatives": ["Alternative A", "Alternative B"],
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert "branch_ids" in response
            assert len(response["branch_ids"]) == 2

    @pytest.mark.asyncio
    async def test_router_status_returns_session_state(self) -> None:
        """Test router_status returns session state when session_id provided."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize session
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Test problem", "complexity": "low"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]

            # Get session state via router_status
            result = await client.call_tool(
                "router_status",
                {"session_id": session_id},
            )
            assert not result.is_error
            response = json.loads(result.data)
            # Session state includes these fields
            assert "complexity" in response
            assert "step_count" in response
            assert "can_synthesize" in response
            assert "valid_next_steps" in response

    @pytest.mark.asyncio
    async def test_router_status_global_stats(self) -> None:
        """Test router_status returns global stats without session_id."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            result = await client.call_tool("router_status", {})
            assert not result.is_error
            response = json.loads(result.data)
            assert "router" in response
            assert "sessions" in response
            assert "tools" in response["router"]
            assert "rules" in response["router"]


class TestParadigmHint:
    """Test paradigm_hint tool for recommending guidance vs enforcement."""

    @pytest.mark.asyncio
    async def test_paradigm_hint_recommends_enforcement_for_proofs(self) -> None:
        """Test paradigm_hint recommends enforcement for mathematical proofs."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            result = await client.call_tool(
                "paradigm_hint",
                {"problem": "Prove that the Monty Hall solution is correct mathematically"},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["recommendation"] == "enforcement"
            assert response["confidence"] >= 0.6
            assert "enforcement_signals" in response
            assert "suggested_tools" in response
            assert "initialize_reasoning" in response["suggested_tools"]

    @pytest.mark.asyncio
    async def test_paradigm_hint_recommends_guidance_for_creative(self) -> None:
        """Test paradigm_hint recommends guidance for creative/exploratory tasks."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            result = await client.call_tool(
                "paradigm_hint",
                {"problem": "Brainstorm ideas for improving user engagement"},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["recommendation"] == "guidance"
            assert response["confidence"] >= 0.6
            assert "guidance_signals" in response
            assert "suggested_tools" in response
            assert "think" in response["suggested_tools"]

    @pytest.mark.asyncio
    async def test_paradigm_hint_includes_workflow_hint(self) -> None:
        """Test paradigm_hint includes workflow_hint for both paradigms."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Test enforcement workflow hint
            result = await client.call_tool(
                "paradigm_hint",
                {"problem": "Verify that the theorem is valid"},
            )
            response = json.loads(result.data)
            assert "workflow_hint" in response
            if response["recommendation"] == "enforcement":
                assert "initialize_reasoning" in response["workflow_hint"]
                assert "submit_step" in response["workflow_hint"]

            # Test guidance workflow hint
            result = await client.call_tool(
                "paradigm_hint",
                {"problem": "Help me debug this code issue"},
            )
            response = json.loads(result.data)
            assert "workflow_hint" in response
            if response["recommendation"] == "guidance":
                assert "think" in response["workflow_hint"]

    @pytest.mark.asyncio
    async def test_paradigm_hint_handles_empty_problem(self) -> None:
        """Test paradigm_hint handles empty problem gracefully."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            result = await client.call_tool(
                "paradigm_hint",
                {"problem": ""},
            )
            response = json.loads(result.data)
            assert "error" in response

    @pytest.mark.asyncio
    async def test_paradigm_hint_neutral_problem(self) -> None:
        """Test paradigm_hint handles neutral problems with default guidance."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # A neutral problem with no strong signals
            result = await client.call_tool(
                "paradigm_hint",
                {"problem": "Tell me about elephants"},
            )
            assert not result.is_error
            response = json.loads(result.data)
            # Should default to guidance for flexibility
            assert "recommendation" in response
            assert "confidence" in response
            assert "suggested_tools" in response


class TestMCPServerInitialization:
    """Test server initialization and configuration."""

    @pytest.mark.asyncio
    async def test_server_has_instructions(self) -> None:
        """Test that server provides instructions for LLM clients."""
        from src.server import mcp

        # FastMCP stores instructions in the server
        assert mcp.instructions is not None
        assert "think" in mcp.instructions.lower()

    @pytest.mark.asyncio
    async def test_server_name_configured(self) -> None:
        """Test that server name is properly configured."""
        from src.server import mcp

        assert mcp.name == "Reason-Guard-MCP"

    @pytest.mark.asyncio
    async def test_client_can_ping_server(self) -> None:
        """Test basic connectivity via ping."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # If we get here without exception, connection works
            assert client.is_connected()


class TestTrustedProxiesValidation:
    """Test TRUSTED_PROXIES parsing and validation."""

    def test_parse_trusted_proxies_valid_entries(self) -> None:
        """Test that valid IPs and CIDRs are accepted."""
        from src.server import _parse_trusted_proxies

        result = _parse_trusted_proxies("10.0.0.1,192.168.1.0/24,::1")
        assert "10.0.0.1" in result
        assert "192.168.1.0/24" in result
        assert "::1" in result
        assert len(result) == 3

    def test_parse_trusted_proxies_empty_string(self) -> None:
        """Test that empty string returns empty frozenset."""
        from src.server import _parse_trusted_proxies

        result = _parse_trusted_proxies("")
        assert result == frozenset()

    def test_parse_trusted_proxies_whitespace_handling(self) -> None:
        """Test that whitespace is properly stripped."""
        from src.server import _parse_trusted_proxies

        result = _parse_trusted_proxies("  10.0.0.1 , 192.168.1.1  ")
        assert "10.0.0.1" in result
        assert "192.168.1.1" in result

    def test_parse_trusted_proxies_invalid_entries_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid entries emit warning and are excluded."""
        import logging

        from src.server import _parse_trusted_proxies

        with caplog.at_level(logging.WARNING):
            result = _parse_trusted_proxies("10.0.0.1,invalid-ip,256.1.1.1,192.168.1.0/24")

        # Valid entries should be included
        assert "10.0.0.1" in result
        assert "192.168.1.0/24" in result

        # Invalid entries should be excluded
        assert "invalid-ip" not in result
        assert "256.1.1.1" not in result
        assert len(result) == 2

        # Warnings should be logged for invalid entries
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("invalid-ip" in msg for msg in warning_messages)
        assert any("256.1.1.1" in msg for msg in warning_messages)

    def test_parse_trusted_proxies_invalid_cidr_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid CIDR notation emits warning."""
        import logging

        from src.server import _parse_trusted_proxies

        with caplog.at_level(logging.WARNING):
            result = _parse_trusted_proxies("10.0.0.0/33,172.16.0.0/12")

        # Valid CIDR should be included
        assert "172.16.0.0/12" in result

        # Invalid CIDR should be excluded
        assert "10.0.0.0/33" not in result

        # Warning should be logged
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("10.0.0.0/33" in msg for msg in warning_messages)


class TestEnforcementModeRejectionPaths:
    """Test enforcement mode rejection paths and edge cases."""

    @pytest.mark.asyncio
    async def test_rejects_synthesis_at_step_zero(self) -> None:
        """Test that synthesis is rejected when no steps have been submitted."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize session
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Test problem", "complexity": "low"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]

            # Try to synthesize at step 0 (should be rejected)
            result = await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "synthesis",
                    "content": "Premature synthesis",
                    "confidence": 0.9,
                },
            )
            response = json.loads(result.data)
            assert response["status"] == "REJECTED"
            assert "rejection_reason" in response

    @pytest.mark.asyncio
    async def test_rejects_invalid_step_type(self) -> None:
        """Test that invalid step types are rejected by Pydantic validation."""
        from fastmcp import Client
        from fastmcp.exceptions import ToolError

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize session
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Test problem", "complexity": "low"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]

            # Try invalid step type - FastMCP validates Literal types and raises ToolError
            with pytest.raises(ToolError):
                await client.call_tool(
                    "submit_step",
                    {
                        "session_id": session_id,
                        "step_type": "invalid_type",
                        "content": "Test content",
                        "confidence": 0.9,
                    },
                )

    @pytest.mark.asyncio
    async def test_rejects_nonexistent_session(self) -> None:
        """Test that operations on nonexistent sessions return REJECTED status."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            result = await client.call_tool(
                "submit_step",
                {
                    "session_id": "nonexistent-session-id",
                    "step_type": "premise",
                    "content": "Test content",
                    "confidence": 0.9,
                },
            )
            response = json.loads(result.data)
            assert response["status"] == "REJECTED"
            assert "not found" in response.get("rejection_reason", "").lower()

    @pytest.mark.asyncio
    async def test_high_complexity_requires_more_steps(self) -> None:
        """Test that high complexity problems require more steps before synthesis."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize with high complexity
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Prove Fermat's Last Theorem", "complexity": "high"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]
            min_steps = response["min_steps"]

            # High complexity should have min_steps >= 6
            assert min_steps >= 6

            # Submit one premise
            await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "premise",
                    "content": "Initial premise",
                    "confidence": 0.95,
                },
            )

            # Try to synthesize (should be rejected - only 1 step)
            result = await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "synthesis",
                    "content": "Premature synthesis",
                    "confidence": 0.9,
                },
            )
            response = json.loads(result.data)
            assert response["status"] == "REJECTED"

    @pytest.mark.asyncio
    async def test_very_low_confidence_triggers_branch(self) -> None:
        """Test that very low confidence (< 0.5) triggers BRANCH_REQUIRED."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize session
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Uncertain problem", "complexity": "low"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]

            # First submit a premise (required before hypothesis)
            await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "premise",
                    "content": "Initial premise",
                    "confidence": 0.95,
                },
            )

            # Submit hypothesis with very low confidence
            result = await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "hypothesis",
                    "content": "Very uncertain hypothesis",
                    "confidence": 0.3,  # Very low
                },
            )
            response = json.loads(result.data)
            assert response["status"] == "BRANCH_REQUIRED"

    @pytest.mark.asyncio
    async def test_branch_creation_with_invalid_count(self) -> None:
        """Test that create_branch validates alternative count."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize session
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Test problem", "complexity": "low"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]

            # Try to create branch with only one alternative
            result = await client.call_tool(
                "create_branch",
                {
                    "session_id": session_id,
                    "alternatives": ["Only one"],
                },
            )
            response = json.loads(result.data)
            # Should error - need at least 2 alternatives
            assert "error" in response or response.get("branch_ids", []) == []

    @pytest.mark.asyncio
    async def test_verify_claims_empty_session(self) -> None:
        """Test verify_claims on session with no steps."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize session
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Test problem", "complexity": "low"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]

            # Verify claims on empty session
            result = await client.call_tool(
                "verify_claims",
                {
                    "session_id": session_id,
                    "claims": ["Test claim"],
                    "evidence": ["Test evidence"],
                },
            )
            response = json.loads(result.data)
            # Should succeed but with no supporting steps
            assert "verified" in response or "error" not in response

    @pytest.mark.asyncio
    async def test_step_feedback_provides_actionable_guidance(self) -> None:
        """Test that rejection feedback provides actionable guidance."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize with medium complexity
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Complex problem", "complexity": "medium"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]

            # Try to synthesize immediately
            result = await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "synthesis",
                    "content": "Premature synthesis",
                    "confidence": 0.9,
                },
            )
            response = json.loads(result.data)
            assert response["status"] == "REJECTED"
            # Feedback should include valid next steps
            assert "valid_next_steps" in response or "rejection_reason" in response

    @pytest.mark.asyncio
    async def test_session_state_tracks_rejection_history(self) -> None:
        """Test that session state tracks step count correctly after rejections."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize session
            result = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Test problem", "complexity": "low"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]

            # Try invalid synthesis (rejected)
            await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "synthesis",
                    "content": "Invalid synthesis",
                    "confidence": 0.9,
                },
            )

            # Check state - step count should still be 0
            result = await client.call_tool(
                "router_status",
                {"session_id": session_id},
            )
            response = json.loads(result.data)
            assert response["step_count"] == 0

            # Now submit valid step
            await client.call_tool(
                "submit_step",
                {
                    "session_id": session_id,
                    "step_type": "premise",
                    "content": "Valid premise",
                    "confidence": 0.95,
                },
            )

            # Check state - step count should be 1
            result = await client.call_tool(
                "router_status",
                {"session_id": session_id},
            )
            response = json.loads(result.data)
            assert response["step_count"] == 1

    @pytest.mark.asyncio
    async def test_complexity_auto_detection_consistency(self) -> None:
        """Test that complexity auto-detection is consistent across similar problems."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Initialize without explicit complexity (auto-detect)
            result1 = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Prove mathematical theorem X"},
            )
            response1 = json.loads(result1.data)

            result2 = await client.call_tool(
                "initialize_reasoning",
                {"problem": "Prove mathematical theorem Y"},
            )
            response2 = json.loads(result2.data)

            # Similar problems should get similar complexity
            # At minimum, both should have valid min_steps and max_steps
            assert "min_steps" in response1
            assert "max_steps" in response1
            assert "min_steps" in response2
            assert "max_steps" in response2
            assert response1["min_steps"] <= response1["max_steps"]
            assert response2["min_steps"] <= response2["max_steps"]
