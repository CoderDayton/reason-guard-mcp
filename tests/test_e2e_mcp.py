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
                # Consolidated tools (3 total)
                "think",
                "compress",
                "status",
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
            assert response["status"] == "started"
            assert response["mode"] == "chain"
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
            assert response["status"] == "started"
            assert response["mode"] == "matrix"
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
            # Step 1: Start verification
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
            assert response["status"] == "started"
            assert response["mode"] == "verify"
            session_id = response["session_id"]

            # Step 2: Add claim with continue action
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
            claim_id = response["claim_id"]

            # Step 3: Verify claim
            result = await client.call_tool(
                "think",
                {
                    "action": "verify",
                    "session_id": session_id,
                    "claim_id": claim_id,
                    "verdict": "supported",
                    "evidence": "Context confirms this",
                },
            )
            assert not result.is_error

            # Step 4: Finish verification
            result = await client.call_tool(
                "think",
                {"action": "finish", "session_id": session_id},
            )
            assert not result.is_error
            response = json.loads(result.data)
            assert response["verified"] is True

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

        assert mcp.name == "MatrixMind-MCP"

    @pytest.mark.asyncio
    async def test_client_can_ping_server(self) -> None:
        """Test basic connectivity via ping."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # If we get here without exception, connection works
            assert client.is_connected()
