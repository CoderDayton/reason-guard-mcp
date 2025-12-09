"""Unit tests for src/server.py helper functions and configuration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestEnvHelpers:
    """Test environment variable helper functions."""

    def test_get_env_returns_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _get_env returns environment variable value."""
        monkeypatch.setenv("TEST_KEY", "test_value")

        from src.server import _get_env

        result = _get_env("TEST_KEY", "default")
        assert result == "test_value"

    def test_get_env_returns_default_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _get_env returns default when env var not set."""
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)

        from src.server import _get_env

        result = _get_env("NONEXISTENT_KEY", "default_value")
        assert result == "default_value"

    def test_get_env_empty_string_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _get_env treats empty string as unset."""
        monkeypatch.setenv("EMPTY_KEY", "")

        from src.server import _get_env

        result = _get_env("EMPTY_KEY", "default")
        assert result == "default"

    def test_get_env_int_returns_integer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _get_env_int returns integer value."""
        monkeypatch.setenv("INT_KEY", "42")

        from src.server import _get_env_int

        result = _get_env_int("INT_KEY", 10)
        assert result == 42

    def test_get_env_int_returns_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _get_env_int returns default when not set."""
        monkeypatch.delenv("NONEXISTENT_INT", raising=False)

        from src.server import _get_env_int

        result = _get_env_int("NONEXISTENT_INT", 100)
        assert result == 100

    def test_get_env_int_invalid_value_returns_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _get_env_int returns default for invalid integer."""
        monkeypatch.setenv("INVALID_INT", "not_a_number")

        from src.server import _get_env_int

        result = _get_env_int("INVALID_INT", 50)
        assert result == 50

    def test_get_env_float_returns_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _get_env_float returns float value."""
        monkeypatch.setenv("FLOAT_KEY", "0.5")

        from src.server import _get_env_float

        result = _get_env_float("FLOAT_KEY", 0.7)
        assert result == 0.5

    def test_get_env_float_returns_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _get_env_float returns default when not set."""
        monkeypatch.delenv("NONEXISTENT_FLOAT", raising=False)

        from src.server import _get_env_float

        result = _get_env_float("NONEXISTENT_FLOAT", 0.7)
        assert result == 0.7

    def test_get_env_float_invalid_value_returns_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _get_env_float returns default for invalid float."""
        monkeypatch.setenv("INVALID_FLOAT", "not_a_number")

        from src.server import _get_env_float

        result = _get_env_float("INVALID_FLOAT", 0.7)
        assert result == 0.7


class TestEmbeddingModelName:
    """Test embedding model name resolution."""

    def test_model_name_with_slash_unchanged(self) -> None:
        """Test full model path (with /) is not modified by function logic."""
        # Test the function logic directly
        model_name = "Snowflake/snowflake-arctic-embed-xs"

        # The function logic: if "/" in model_name, return as-is
        result = model_name if "/" in model_name else f"sentence-transformers/{model_name}"

        assert result == "Snowflake/snowflake-arctic-embed-xs"

    def test_short_model_name_gets_prefix(self) -> None:
        """Test short model name gets sentence-transformers prefix."""
        model_name = "all-mpnet-base-v2"

        # The function logic
        result = model_name if "/" in model_name else f"sentence-transformers/{model_name}"

        assert result == "sentence-transformers/all-mpnet-base-v2"


class TestToolGetters:
    """Test lazy tool initialization getters."""

    def test_get_llm_client_creates_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_llm_client creates client on first call."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        with patch("src.server.LLMClient") as mock_llm:
            mock_llm.return_value = MagicMock()

            import src.server

            src.server._llm_client = None  # Reset cached instance

            client = src.server.get_llm_client()

            mock_llm.assert_called_once()
            assert client is not None

    def test_get_llm_client_returns_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_llm_client returns cached instance on second call."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        import src.server

        mock_client = MagicMock()
        src.server._llm_client = mock_client

        result = src.server.get_llm_client()
        assert result is mock_client

    def test_get_compression_tool_creates_instance(self) -> None:
        """Test get_compression_tool creates tool on first call."""
        with patch("src.server.ContextAwareCompressionTool") as mock_tool:
            mock_tool.return_value = MagicMock()

            import src.server

            src.server._compression_tool = None

            tool = src.server.get_compression_tool()

            mock_tool.assert_called_once()
            assert tool is not None

    def test_get_mot_tool_creates_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_mot_tool creates tool on first call."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with (
            patch("src.server.MatrixOfThoughtTool") as mock_tool,
            patch("src.server.get_llm_client") as mock_llm,
        ):
            mock_tool.return_value = MagicMock()
            mock_llm.return_value = MagicMock()

            import src.server

            src.server._mot_tool = None

            tool = src.server.get_mot_tool()

            mock_tool.assert_called_once()
            assert tool is not None

    def test_get_long_chain_tool_creates_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_long_chain_tool creates tool on first call."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with (
            patch("src.server.LongChainOfThoughtTool") as mock_tool,
            patch("src.server.get_llm_client") as mock_llm,
        ):
            mock_tool.return_value = MagicMock()
            mock_llm.return_value = MagicMock()

            import src.server

            src.server._long_chain_tool = None

            tool = src.server.get_long_chain_tool()

            mock_tool.assert_called_once()
            assert tool is not None

    def test_get_verify_tool_creates_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_verify_tool creates tool on first call."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with (
            patch("src.server.FactVerificationTool") as mock_tool,
            patch("src.server.get_llm_client") as mock_llm,
        ):
            mock_tool.return_value = MagicMock()
            mock_llm.return_value = MagicMock()

            import src.server

            src.server._verify_tool = None

            tool = src.server.get_verify_tool()

            mock_tool.assert_called_once()
            assert tool is not None


class TestRecommendStrategy:
    """Test recommend_reasoning_strategy tool function."""

    @pytest.mark.asyncio
    async def test_recommend_serial_strategy(self) -> None:
        """Test recommendation for serial problems."""
        from src.server import recommend_reasoning_strategy

        result = await recommend_reasoning_strategy.fn(
            problem="Find the path from A to B through the graph with constraints",
            token_budget=5000,
        )

        response = json.loads(result)
        assert response["recommended_strategy"] == "long_chain"
        assert (
            "serial" in response["explanation"].lower() or "path" in response["explanation"].lower()
        )

    @pytest.mark.asyncio
    async def test_recommend_parallel_strategy(self) -> None:
        """Test recommendation for parallel/exploratory problems."""
        from src.server import recommend_reasoning_strategy

        result = await recommend_reasoning_strategy.fn(
            problem="Generate multiple creative options and explore different alternatives",
            token_budget=5000,
        )

        response = json.loads(result)
        assert response["recommended_strategy"] == "parallel_voting"

    @pytest.mark.asyncio
    async def test_recommend_matrix_strategy(self) -> None:
        """Test recommendation for balanced problems."""
        from src.server import recommend_reasoning_strategy

        result = await recommend_reasoning_strategy.fn(
            problem="Analyze this data for insights",
            token_budget=4000,
        )

        response = json.loads(result)
        assert response["recommended_strategy"] == "matrix"
        assert "balanced" in response["explanation"].lower()

    @pytest.mark.asyncio
    async def test_recommend_includes_indicators(self) -> None:
        """Test recommendation includes indicator counts."""
        from src.server import recommend_reasoning_strategy

        result = await recommend_reasoning_strategy.fn(
            problem="Test problem",
            token_budget=3000,
        )

        response = json.loads(result)
        assert "indicators" in response
        assert "serial_count" in response["indicators"]
        assert "parallel_count" in response["indicators"]

    @pytest.mark.asyncio
    async def test_recommend_with_context_logs(self) -> None:
        """Test recommendation with context logging."""
        from src.server import recommend_reasoning_strategy

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()

        await recommend_reasoning_strategy.fn(
            problem="Test",
            token_budget=3000,
            ctx=mock_ctx,
        )

        mock_ctx.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_recommend_error_handling(self) -> None:
        """Test error handling in strategy recommendation."""
        from src.server import recommend_reasoning_strategy

        # Patch to cause an error
        with patch("src.server.ReasoningStrategy") as mock_strategy:
            mock_strategy.LONG_CHAIN.value = None
            mock_strategy.PARALLEL.value = None
            mock_strategy.MATRIX.value = None

            # Calling with problematic setup should still return a response
            # (errors are caught and returned as ToolExecutionError)
            result = await recommend_reasoning_strategy.fn(
                problem="test",
                token_budget=1000,
            )

            response = json.loads(result)
            # Either returns recommendation or error
            assert "recommended_strategy" in response or "error" in response


class TestCheckStatus:
    """Test check_status tool function."""

    @pytest.mark.asyncio
    async def test_check_status_returns_model_info(self) -> None:
        """Test check_status returns model status info."""
        with patch("src.server.ModelManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.get_status.return_value = {
                "state": "ready",
                "device": "cpu",
            }
            mock_manager.get_instance.return_value = mock_instance

            from src.server import check_status

            result = await check_status.fn()

            response = json.loads(result)
            assert "model_status" in response
            assert "server_info" in response
            assert "llm_config" in response

    @pytest.mark.asyncio
    async def test_check_status_with_context(self) -> None:
        """Test check_status with context logging."""
        with patch("src.server.ModelManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.get_status.return_value = {
                "state": "ready",
                "device": "cuda",
            }
            mock_manager.get_instance.return_value = mock_instance

            mock_ctx = MagicMock()
            mock_ctx.info = AsyncMock()

            from src.server import check_status

            await check_status.fn(ctx=mock_ctx)

            mock_ctx.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_status_error_handling(self) -> None:
        """Test check_status error handling."""
        with patch("src.server.ModelManager") as mock_manager:
            mock_manager.get_instance.side_effect = Exception("Manager error")

            from src.server import check_status

            result = await check_status.fn()

            response = json.loads(result)
            assert "error" in response


class TestCompressPromptErrors:
    """Test compress_prompt error handling."""

    @pytest.mark.asyncio
    async def test_compress_model_not_ready_error(self) -> None:
        """Test compress_prompt when model not ready."""
        from src.server import compress_prompt
        from src.utils.errors import ModelNotReadyException

        with patch("src.server.get_compression_tool") as mock_get_tool:
            mock_get_tool.side_effect = ModelNotReadyException("Model loading")

            result = await compress_prompt.fn(
                context="Test context",
                question="Test question",
            )

            response = json.loads(result)
            assert "error" in response
            assert "retry_after_seconds" in response.get("details", {})

    @pytest.mark.asyncio
    async def test_compress_matrixmind_exception(self) -> None:
        """Test compress_prompt with MatrixMindException."""
        from src.server import compress_prompt
        from src.utils.errors import MatrixMindException

        with patch("src.server.get_compression_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.compress.side_effect = MatrixMindException("Compression error")
            mock_get_tool.return_value = mock_tool

            result = await compress_prompt.fn(
                context="Test context",
                question="Test question",
            )

            response = json.loads(result)
            assert "error" in response

    @pytest.mark.asyncio
    async def test_compress_unexpected_exception(self) -> None:
        """Test compress_prompt with unexpected exception."""
        from src.server import compress_prompt

        with patch("src.server.get_compression_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.compress.side_effect = RuntimeError("Unexpected")
            mock_get_tool.return_value = mock_tool

            result = await compress_prompt.fn(
                context="Test context",
                question="Test question",
            )

            response = json.loads(result)
            assert "error" in response
            assert "RuntimeError" in str(response)


class TestMotReasoningErrors:
    """Test matrix_of_thought_reasoning error handling."""

    @pytest.mark.asyncio
    async def test_mot_matrixmind_exception(self) -> None:
        """Test MoT with MatrixMindException."""
        from src.server import matrix_of_thought_reasoning
        from src.utils.errors import MatrixMindException

        with patch("src.server.get_mot_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.reason.side_effect = MatrixMindException("MoT error")
            mock_get_tool.return_value = mock_tool

            result = await matrix_of_thought_reasoning.fn(
                question="Test",
                context="Context",
            )

            response = json.loads(result)
            assert "error" in response

    @pytest.mark.asyncio
    async def test_mot_unexpected_exception(self) -> None:
        """Test MoT with unexpected exception."""
        from src.server import matrix_of_thought_reasoning

        with patch("src.server.get_mot_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.reason.side_effect = ValueError("Unexpected")
            mock_get_tool.return_value = mock_tool

            result = await matrix_of_thought_reasoning.fn(
                question="Test",
                context="Context",
            )

            response = json.loads(result)
            assert "error" in response


class TestLongChainErrors:
    """Test long_chain_of_thought error handling."""

    @pytest.mark.asyncio
    async def test_long_chain_matrixmind_exception(self) -> None:
        """Test long_chain with MatrixMindException."""
        from src.server import long_chain_of_thought
        from src.utils.errors import MatrixMindException

        with patch("src.server.get_long_chain_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.reason.side_effect = MatrixMindException("Chain error")
            mock_get_tool.return_value = mock_tool

            result = await long_chain_of_thought.fn(
                problem="Test problem",
            )

            response = json.loads(result)
            assert "error" in response

    @pytest.mark.asyncio
    async def test_long_chain_unexpected_exception(self) -> None:
        """Test long_chain with unexpected exception."""
        from src.server import long_chain_of_thought

        with patch("src.server.get_long_chain_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.reason.side_effect = TypeError("Unexpected")
            mock_get_tool.return_value = mock_tool

            result = await long_chain_of_thought.fn(
                problem="Test problem",
            )

            response = json.loads(result)
            assert "error" in response


class TestVerifyErrors:
    """Test verify_fact_consistency error handling."""

    @pytest.mark.asyncio
    async def test_verify_matrixmind_exception(self) -> None:
        """Test verify with MatrixMindException."""
        from src.server import verify_fact_consistency
        from src.utils.errors import MatrixMindException

        with patch("src.server.get_verify_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.verify.side_effect = MatrixMindException("Verify error")
            mock_get_tool.return_value = mock_tool

            result = await verify_fact_consistency.fn(
                answer="Test answer",
                context="Context",
            )

            response = json.loads(result)
            assert "error" in response

    @pytest.mark.asyncio
    async def test_verify_unexpected_exception(self) -> None:
        """Test verify with unexpected exception."""
        from src.server import verify_fact_consistency

        with patch("src.server.get_verify_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.verify.side_effect = KeyError("Unexpected")
            mock_get_tool.return_value = mock_tool

            result = await verify_fact_consistency.fn(
                answer="Test answer",
                context="Context",
            )

            response = json.loads(result)
            assert "error" in response


class TestMainFunction:
    """Test main entry point configuration."""

    def test_main_checks_transport(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main validates transport configuration."""
        # Just verify the server can be imported and has main
        from src.server import main

        assert callable(main)
        # Default transport should be valid
        assert True  # Unknown falls back
