"""End-to-end tests for MCP protocol compliance.

Tests the actual MCP server through the protocol layer using FastMCP's Client.
This verifies that tools are properly exposed and callable via MCP.
"""

from __future__ import annotations

import json

import pytest


class TestMCPProtocol:
    """Test MCP protocol compliance through actual server communication."""

    @pytest.mark.asyncio
    async def test_server_lists_all_tools(self) -> None:
        """Test that server exposes all expected tools via MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            tools = await client.list_tools()

            tool_names = {tool.name for tool in tools}
            expected_tools = {
                # Compression
                "compress_prompt",
                # Long Chain (multi-call)
                "chain_start",
                "chain_add_step",
                "chain_finalize",
                "chain_get",
                # Matrix of Thought (multi-call)
                "matrix_start",
                "matrix_set_cell",
                "matrix_synthesize",
                "matrix_finalize",
                "matrix_get",
                # Verification (multi-call)
                "verify_start",
                "verify_add_claim",
                "verify_claim",
                "verify_finalize",
                "verify_get",
                # Utilities
                "recommend_reasoning_strategy",
                "check_status",
            }

            missing = expected_tools - tool_names
            extra = tool_names - expected_tools
            assert tool_names == expected_tools, f"Missing: {missing}, Extra: {extra}"

    @pytest.mark.asyncio
    async def test_tool_schemas_are_valid(self) -> None:
        """Test that all tools have valid JSON schemas for their parameters."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            tools = await client.list_tools()

            for tool in tools:
                # Each tool should have an inputSchema
                assert tool.inputSchema is not None, f"Tool {tool.name} missing inputSchema"
                assert "type" in tool.inputSchema, f"Tool {tool.name} schema missing type"
                assert tool.inputSchema["type"] == "object"

                # Verify required fields are defined
                if "required" in tool.inputSchema:
                    assert isinstance(tool.inputSchema["required"], list)

    @pytest.mark.asyncio
    async def test_recommend_strategy_tool_callable(self) -> None:
        """Test recommend_reasoning_strategy tool via MCP protocol.

        This tool doesn't require session state, so it's a good E2E test target.
        """
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            result = await client.call_tool(
                "recommend_reasoning_strategy",
                {"problem": "What is 2 + 2?", "token_budget": 2000},
            )

            # CallToolResult has .content, .data, .is_error
            assert not result.is_error, f"Tool returned error: {result.data}"
            assert len(result.content) == 1

            # Parse the JSON response
            response = json.loads(result.data)

            # Verify response structure
            assert "recommended_strategy" in response
            assert "explanation" in response
            assert response["recommended_strategy"] in [
                "long_chain",
                "matrix",
                "parallel_voting",
            ]

    @pytest.mark.asyncio
    async def test_chain_workflow_via_mcp(self) -> None:
        """Test complete chain workflow through MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Step 1: Start chain (parameter is 'expected_steps', not 'planned_steps')
            result = await client.call_tool(
                "chain_start",
                {"problem": "What is 2 + 2?", "expected_steps": 3},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "started"
            session_id = response["session_id"]

            # Step 2: Add steps (parameter is 'thought', not 'step_content')
            result = await client.call_tool(
                "chain_add_step",
                {"session_id": session_id, "thought": "First, identify the operands: 2 and 2"},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert "current_step" in response or "instruction" in response  # Verify valid response

            result = await client.call_tool(
                "chain_add_step",
                {"session_id": session_id, "thought": "Apply addition: 2 + 2 = 4"},
            )
            assert not result.is_error

            # Step 3: Finalize (parameter is 'answer', not 'final_answer')
            result = await client.call_tool(
                "chain_finalize",
                {"session_id": session_id, "answer": "The answer is 4"},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "completed"
            assert "chain" in response  # Chain contains the reasoning steps

    @pytest.mark.asyncio
    async def test_matrix_workflow_via_mcp(self) -> None:
        """Test complete matrix workflow through MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Step 1: Start matrix
            result = await client.call_tool(
                "matrix_start",
                {"question": "What is AI?", "rows": 2, "cols": 2},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "started"
            session_id = response["session_id"]

            # Step 2: Set cells (parameter is 'thought', not 'content')
            for row in range(2):
                for col in range(2):
                    result = await client.call_tool(
                        "matrix_set_cell",
                        {
                            "session_id": session_id,
                            "row": row,
                            "col": col,
                            "thought": f"Thought at ({row}, {col})",
                        },
                    )
                    assert not result.is_error

            # Step 3: Synthesize (requires col parameter)
            result = await client.call_tool(
                "matrix_synthesize",
                {"session_id": session_id, "col": 0, "synthesis": "Column 0 synthesis"},
            )
            assert not result.is_error

            result = await client.call_tool(
                "matrix_synthesize",
                {"session_id": session_id, "col": 1, "synthesis": "Column 1 synthesis"},
            )
            assert not result.is_error

            # Step 4: Finalize (parameter is 'answer', not 'final_answer')
            result = await client.call_tool(
                "matrix_finalize",
                {"session_id": session_id, "answer": "AI is the simulation of human intelligence"},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "completed"

    @pytest.mark.asyncio
    async def test_verify_workflow_via_mcp(self) -> None:
        """Test complete verification workflow through MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Step 1: Start verification
            result = await client.call_tool(
                "verify_start",
                {
                    "answer": "Einstein was a physicist",
                    "context": "Albert Einstein was a theoretical physicist.",
                },
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "started"
            session_id = response["session_id"]

            # Step 2: Add claim
            result = await client.call_tool(
                "verify_add_claim",
                {"session_id": session_id, "claim": "Einstein was a physicist"},
            )
            assert not result.is_error
            response = json.loads(result.data)
            claim_id = response["claim_id"]

            # Step 3: Verify claim
            result = await client.call_tool(
                "verify_claim",
                {
                    "session_id": session_id,
                    "claim_id": claim_id,
                    "status": "supported",
                    "evidence": "Context confirms this",
                },
            )
            assert not result.is_error

            # Step 4: Finalize verification session
            result = await client.call_tool(
                "verify_finalize",
                {"session_id": session_id},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["verified"] is True

    @pytest.mark.asyncio
    async def test_tool_error_handling(self) -> None:
        """Test that tool errors are properly propagated through MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Call with invalid session_id (use 'thought' param)
            result = await client.call_tool(
                "chain_add_step",
                {"session_id": "invalid-session-id", "thought": "Test step"},
            )

            # Should return error
            response = json.loads(result.data)
            assert "error" in response

    @pytest.mark.asyncio
    async def test_check_status_tool(self) -> None:
        """Test check_status tool via MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # check_status returns general server status with active_sessions
            result = await client.call_tool(
                "check_status",
                {},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert "active_sessions" in response
            assert "server_info" in response

    @pytest.mark.asyncio
    async def test_chain_get_session_status(self) -> None:
        """Test chain_get tool to check session status via MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Start a chain session
            result = await client.call_tool(
                "chain_start",
                {"problem": "Test problem"},
            )
            response = json.loads(result.data)
            session_id = response["session_id"]

            # Check session status via chain_get
            result = await client.call_tool(
                "chain_get",
                {"session_id": session_id},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["status"] == "active"


class TestMCPServerInitialization:
    """Test server initialization and configuration."""

    @pytest.mark.asyncio
    async def test_server_has_instructions(self) -> None:
        """Test that server provides instructions for LLM clients."""
        from src.server import mcp

        # FastMCP stores instructions in the server
        assert mcp.instructions is not None
        assert "chain" in mcp.instructions.lower() or "matrix" in mcp.instructions.lower()

    @pytest.mark.asyncio
    async def test_server_name_configured(self) -> None:
        """Test that server name is properly configured."""
        from src.server import mcp

        assert mcp.name == "MatrixMind-MCP"

    @pytest.mark.asyncio
    async def test_client_can_ping_server(self) -> None:
        """Test basic connectivity via ping."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # If we get here without exception, connection works
            assert client.is_connected()
