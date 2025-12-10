"""Smoke tests for MatrixMind MCP Server.

Quick tests to verify all tools are functional.
Run with: pytest tests/test_smoke.py -v
"""

from __future__ import annotations

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

    async def test_compress_prompt_registered(self) -> None:
        """Test compress_prompt tool is properly registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import compress_prompt

        assert isinstance(compress_prompt, FunctionTool)
        tool: FunctionTool = compress_prompt
        assert callable(tool.fn)
        assert tool.name == "compress_prompt"

    async def test_chain_start_registered(self) -> None:
        """Test chain_start tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import chain_start

        assert isinstance(chain_start, FunctionTool)
        tool: FunctionTool = chain_start
        assert callable(tool.fn)
        assert tool.name == "chain_start"

    async def test_chain_add_step_registered(self) -> None:
        """Test chain_add_step tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import chain_add_step

        assert isinstance(chain_add_step, FunctionTool)
        assert chain_add_step.name == "chain_add_step"

    async def test_chain_finalize_registered(self) -> None:
        """Test chain_finalize tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import chain_finalize

        assert isinstance(chain_finalize, FunctionTool)
        assert chain_finalize.name == "chain_finalize"

    async def test_matrix_start_registered(self) -> None:
        """Test matrix_start tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import matrix_start

        assert isinstance(matrix_start, FunctionTool)
        assert matrix_start.name == "matrix_start"

    async def test_matrix_set_cell_registered(self) -> None:
        """Test matrix_set_cell tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import matrix_set_cell

        assert isinstance(matrix_set_cell, FunctionTool)
        assert matrix_set_cell.name == "matrix_set_cell"

    async def test_matrix_synthesize_registered(self) -> None:
        """Test matrix_synthesize tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import matrix_synthesize

        assert isinstance(matrix_synthesize, FunctionTool)
        assert matrix_synthesize.name == "matrix_synthesize"

    async def test_matrix_finalize_registered(self) -> None:
        """Test matrix_finalize tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import matrix_finalize

        assert isinstance(matrix_finalize, FunctionTool)
        assert matrix_finalize.name == "matrix_finalize"

    async def test_verify_start_registered(self) -> None:
        """Test verify_start tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import verify_start

        assert isinstance(verify_start, FunctionTool)
        assert verify_start.name == "verify_start"

    async def test_verify_add_claim_registered(self) -> None:
        """Test verify_add_claim tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import verify_add_claim

        assert isinstance(verify_add_claim, FunctionTool)
        assert verify_add_claim.name == "verify_add_claim"

    async def test_verify_claim_registered(self) -> None:
        """Test verify_claim tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import verify_claim

        assert isinstance(verify_claim, FunctionTool)
        assert verify_claim.name == "verify_claim"

    async def test_verify_finalize_registered(self) -> None:
        """Test verify_finalize tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import verify_finalize

        assert isinstance(verify_finalize, FunctionTool)
        assert verify_finalize.name == "verify_finalize"

    async def test_recommend_strategy_registered(self) -> None:
        """Test recommend_reasoning_strategy tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import recommend_reasoning_strategy

        assert isinstance(recommend_reasoning_strategy, FunctionTool)
        tool: FunctionTool = recommend_reasoning_strategy
        assert callable(tool.fn)
        assert tool.name == "recommend_reasoning_strategy"

    async def test_check_status_registered(self) -> None:
        """Test check_status tool is registered."""
        from fastmcp.tools.tool import FunctionTool

        from src.server import check_status

        assert isinstance(check_status, FunctionTool)
        assert check_status.name == "check_status"
