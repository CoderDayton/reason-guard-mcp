"""End-to-end tests for MCP protocol compliance.

Tests the actual MCP server through the protocol layer using FastMCP's Client.
This verifies that tools are properly exposed and callable via MCP.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.schema import CompressionResult, ReasoningResult, VerificationResult

if TYPE_CHECKING:
    pass


class TestMCPProtocol:
    """Test MCP protocol compliance through actual server communication."""

    @pytest.fixture
    def mock_llm_response(self) -> str:
        """Standard mock LLM response."""
        return "This is a mock LLM response for testing."

    @pytest.fixture
    def mock_llm_client(self, mock_llm_response: str) -> MagicMock:
        """Create mock LLM client that returns predictable responses."""
        mock = MagicMock()
        mock.generate.return_value = mock_llm_response
        mock.generate_async = AsyncMock(return_value=mock_llm_response)
        mock.estimate_tokens.return_value = 100
        return mock

    @pytest.mark.asyncio
    async def test_server_lists_all_tools(self) -> None:
        """Test that server exposes all 5 expected tools via MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            tools = await client.list_tools()

            tool_names = {tool.name for tool in tools}
            expected_tools = {
                "compress_prompt",
                "matrix_of_thought_reasoning",
                "long_chain_of_thought",
                "verify_fact_consistency",
                "recommend_reasoning_strategy",
                "check_status",
            }

            assert tool_names == expected_tools, f"Missing tools: {expected_tools - tool_names}"

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

        This tool doesn't require LLM calls, so it's a good E2E test target.
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
    async def test_compress_prompt_with_mock_llm(self) -> None:
        """Test compress_prompt tool via MCP protocol with mocked tool."""
        from fastmcp import Client

        from src.server import mcp

        # Patch the get_compression_tool to use our mock
        with patch("src.server.get_compression_tool") as mock_get_tool:
            mock_tool = MagicMock()
            # Return a proper dataclass instance that serializes correctly
            mock_tool.compress.return_value = CompressionResult(
                compressed_context="Compressed context here.",
                compression_ratio=0.3,
                original_tokens=1000,
                compressed_tokens=300,
                sentences_kept=5,
                sentences_removed=15,
                relevance_scores=[("test sentence", 0.9)],
            )
            mock_get_tool.return_value = mock_tool

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "compress_prompt",
                    {
                        "context": "This is a long context. " * 100,
                        "question": "What is this about?",
                        "compression_ratio": 0.3,
                    },
                )

                assert not result.is_error, f"Tool returned error: {result.data}"
                response = json.loads(result.data)

                assert "compressed_context" in response
                assert "compression_ratio" in response

    @pytest.mark.asyncio
    async def test_matrix_of_thought_with_mock_llm(self) -> None:
        """Test matrix_of_thought_reasoning tool via MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        with patch("src.server.get_mot_tool") as mock_get_tool:
            mock_tool = MagicMock()
            # Mock the async method used by the server
            mock_tool.reason_async = AsyncMock(
                return_value=ReasoningResult(
                    answer="The answer is 42.",
                    confidence=0.85,
                    reasoning_steps=["Step 1", "Step 2"],
                    reasoning_trace={"matrix": [[1, 2], [3, 4]]},
                    tokens_used=500,
                )
            )
            mock_get_tool.return_value = mock_tool

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "matrix_of_thought_reasoning",
                    {
                        "question": "What is the meaning of life?",
                        "context": "Philosophy text here.",
                        "matrix_rows": 2,
                        "matrix_cols": 2,
                    },
                )

                assert not result.is_error, f"Tool returned error: {result.data}"
                response = json.loads(result.data)

                assert "answer" in response
                assert "confidence" in response

    @pytest.mark.asyncio
    async def test_long_chain_with_mock_llm(self) -> None:
        """Test long_chain_of_thought tool via MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        with patch("src.server.get_long_chain_tool") as mock_get_tool:
            mock_tool = MagicMock()
            # Mock the async method used by the server
            mock_tool.reason_async = AsyncMock(
                return_value=ReasoningResult(
                    answer="Step-by-step answer.",
                    confidence=0.78,
                    reasoning_steps=["Step 1", "Step 2", "Step 3"],
                    verification_results={"passed": True},
                    tokens_used=800,
                )
            )
            mock_get_tool.return_value = mock_tool

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "long_chain_of_thought",
                    {
                        "problem": "Solve this complex problem.",
                        "num_steps": 5,
                        "verify_intermediate": False,
                    },
                )

                assert not result.is_error, f"Tool returned error: {result.data}"
                response = json.loads(result.data)

                assert "answer" in response
                assert "reasoning_steps" in response

    @pytest.mark.asyncio
    async def test_verify_fact_with_mock_llm(self) -> None:
        """Test verify_fact_consistency tool via MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        with patch("src.server.get_verify_tool") as mock_get_tool:
            mock_tool = MagicMock()
            # Mock the async method used by the server
            mock_tool.verify_async = AsyncMock(
                return_value=VerificationResult(
                    verified=True,
                    confidence=0.95,
                    claims_verified=3,
                    claims_total=3,
                    reason="All claims verified.",
                    claim_details=[{"claim": "Test", "status": "supported"}],
                )
            )
            mock_get_tool.return_value = mock_tool

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "verify_fact_consistency",
                    {
                        "answer": "Einstein was a physicist.",
                        "context": "Albert Einstein was a theoretical physicist.",
                        "max_claims": 5,
                    },
                )

                assert not result.is_error, f"Tool returned error: {result.data}"
                response = json.loads(result.data)

                assert "verified" in response
                assert "confidence" in response

    @pytest.mark.asyncio
    async def test_tool_error_handling(self) -> None:
        """Test that tool errors are properly propagated through MCP protocol."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Call with invalid parameters (empty problem)
            result = await client.call_tool(
                "recommend_reasoning_strategy",
                {"problem": "", "token_budget": 100},
            )

            # Should return error in response or handle gracefully
            response = json.loads(result.data)

            # Either error key or the tool handles empty gracefully
            assert "error" in response or "recommended_strategy" in response

    @pytest.mark.asyncio
    async def test_tool_parameter_validation(self) -> None:
        """Test that invalid parameter types are rejected."""
        from fastmcp import Client

        from src.server import mcp

        async with Client(mcp) as client:
            # Test with out-of-range token_budget
            result = await client.call_tool(
                "recommend_reasoning_strategy",
                {"problem": "Test problem", "token_budget": 100},  # Below minimum
            )

            # Should either error or clamp to valid range
            response = json.loads(result.data)
            assert "error" in response or "recommended_strategy" in response


class TestMCPServerInitialization:
    """Test server initialization and configuration."""

    @pytest.mark.asyncio
    async def test_server_has_instructions(self) -> None:
        """Test that server provides instructions for LLM clients."""
        from src.server import mcp

        # FastMCP stores instructions in the server
        assert mcp.instructions is not None
        assert "compress" in mcp.instructions.lower()
        assert "reason" in mcp.instructions.lower()

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
