"""Tests for compress module.

Tests context-aware semantic compression tool.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.context_encoder import EncoderException
from src.models.model_manager import ModelManager
from src.tools.compress import ContextAwareCompressionTool
from src.utils.errors import CompressionException, ModelNotReadyException


class TestContextAwareCompressionTool:
    """Test ContextAwareCompressionTool class."""

    @pytest.fixture(autouse=True)
    def setup_mock_manager(self) -> Generator[None, None, None]:
        """Set up mock ModelManager for all tests."""
        ModelManager.reset_instance()

        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 768)
        mock_model.return_value = mock_output
        mock_model.__call__ = MagicMock(return_value=mock_output)

        # Create proper mock batch encoding
        mock_batch_encoding = MagicMock()
        mock_batch_encoding.__getitem__ = lambda self, key: (
            torch.randint(0, 1000, (1, 10)) if key == "input_ids" else torch.ones(1, 10)
        )
        mock_batch_encoding.to = MagicMock(return_value=mock_batch_encoding)

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = mock_batch_encoding
        mock_tokenizer.encode.return_value = list(range(10))

        with patch.object(ModelManager, "get_instance") as mock_get_instance:
            mock_manager = MagicMock()
            mock_manager.device = "cpu"
            mock_manager.is_ready.return_value = True
            mock_manager.get_model.return_value = (mock_model, mock_tokenizer)
            mock_manager.get_hidden_size.return_value = 768
            mock_get_instance.return_value = mock_manager

            yield

        ModelManager.reset_instance()

    def test_init_success(self) -> None:
        """Test successful initialization."""
        tool = ContextAwareCompressionTool()
        assert tool.model_name == "sentence-transformers/all-mpnet-base-v2"

    def test_init_custom_model(self) -> None:
        """Test initialization with custom model name."""
        tool = ContextAwareCompressionTool(model_name="custom-model")
        assert tool.model_name == "custom-model"

    def test_device_property(self) -> None:
        """Test device property returns manager device."""
        tool = ContextAwareCompressionTool()
        assert tool.device == "cpu"

    def test_compress_empty_context_raises(self) -> None:
        """Test compress raises on empty context."""
        tool = ContextAwareCompressionTool()
        with pytest.raises(CompressionException, match="Context cannot be empty"):
            tool.compress(context="", question="What is this?")

    def test_compress_whitespace_context_raises(self) -> None:
        """Test compress raises on whitespace-only context."""
        tool = ContextAwareCompressionTool()
        with pytest.raises(CompressionException, match="Context cannot be empty"):
            tool.compress(context="   ", question="What is this?")

    def test_compress_empty_question_raises(self) -> None:
        """Test compress raises on empty question."""
        tool = ContextAwareCompressionTool()
        with pytest.raises(CompressionException, match="Question cannot be empty"):
            tool.compress(context="Some context here.", question="")

    def test_compress_whitespace_question_raises(self) -> None:
        """Test compress raises on whitespace-only question."""
        tool = ContextAwareCompressionTool()
        with pytest.raises(CompressionException, match="Question cannot be empty"):
            tool.compress(context="Some context here.", question="   ")

    def test_compress_ratio_too_low_raises(self) -> None:
        """Test compress raises on ratio below 0.1."""
        tool = ContextAwareCompressionTool()
        with pytest.raises(CompressionException, match="Compression ratio must be between"):
            tool.compress(
                context="Some context here.", question="What is this?", compression_ratio=0.05
            )

    def test_compress_ratio_too_high_raises(self) -> None:
        """Test compress raises on ratio above 1.0."""
        tool = ContextAwareCompressionTool()
        with pytest.raises(CompressionException, match="Compression ratio must be between"):
            tool.compress(
                context="Some context here.", question="What is this?", compression_ratio=1.5
            )

    def test_compress_no_valid_sentences_raises(self) -> None:
        """Test compress raises when no valid sentences found."""
        tool = ContextAwareCompressionTool()
        # Very short text that won't split into valid sentences
        with pytest.raises(CompressionException, match="Could not split context"):
            tool.compress(context="Hi", question="What is this?")


class TestCompressionToolSentenceSplitting:
    """Test sentence splitting functionality."""

    @pytest.fixture(autouse=True)
    def setup_tool(self) -> Generator[ContextAwareCompressionTool, None, None]:
        """Set up tool with mocked manager."""
        ModelManager.reset_instance()

        with patch.object(ModelManager, "get_instance") as mock_get_instance:
            mock_manager = MagicMock()
            mock_manager.device = "cpu"
            mock_manager.is_ready.return_value = True
            mock_manager.get_model.return_value = (MagicMock(), MagicMock())
            mock_manager.get_hidden_size.return_value = 768
            mock_get_instance.return_value = mock_manager

            tool = ContextAwareCompressionTool()
            yield tool

        ModelManager.reset_instance()

    def test_split_basic_sentences(self, setup_tool: ContextAwareCompressionTool) -> None:
        """Test basic sentence splitting."""
        tool = setup_tool
        text = "This is the first sentence. This is the second sentence. This is third one here."
        sentences = tool._split_sentences(text, min_words=3)
        assert len(sentences) == 3

    def test_split_handles_abbreviations(self, setup_tool: ContextAwareCompressionTool) -> None:
        """Test sentence splitting handles abbreviations."""
        tool = setup_tool
        text = "Dr. Smith went to work. He saw Mrs. Jones there."
        sentences = tool._split_sentences(text, min_words=3)
        assert len(sentences) == 2
        assert "Dr." in sentences[0]
        assert "Mrs." in sentences[1]

    def test_split_filters_short_sentences(self, setup_tool: ContextAwareCompressionTool) -> None:
        """Test sentence splitting filters short sentences."""
        tool = setup_tool
        text = "Hi. This is a longer sentence here. Ok."
        sentences = tool._split_sentences(text, min_words=3)
        assert len(sentences) == 1
        assert "longer sentence" in sentences[0]

    def test_split_with_exclamation_and_question(
        self, setup_tool: ContextAwareCompressionTool
    ) -> None:
        """Test sentence splitting with various endings."""
        tool = setup_tool
        text = "What is happening here? This is great! Another normal sentence here."
        sentences = tool._split_sentences(text, min_words=3)
        assert len(sentences) == 3


class TestCompressionToolExceptionHandling:
    """Test exception handling in compression tool."""

    @pytest.fixture(autouse=True)
    def setup_mock_manager(self) -> Generator[None, None, None]:
        """Set up mock ModelManager for all tests."""
        ModelManager.reset_instance()
        yield
        ModelManager.reset_instance()

    def test_model_not_ready_propagates(self) -> None:
        """Test ModelNotReadyException is re-raised."""
        with patch.object(ModelManager, "get_instance") as mock_get_instance:
            mock_manager = MagicMock()
            mock_manager.device = "cpu"
            mock_manager.is_ready.return_value = True
            mock_manager.get_model.side_effect = ModelNotReadyException("Not ready")
            mock_get_instance.return_value = mock_manager

            tool = ContextAwareCompressionTool()
            tool._encoder = None  # Force re-creation

            with pytest.raises(ModelNotReadyException):
                tool.compress(
                    context="This is a valid sentence here. Another valid sentence here.",
                    question="What is this?",
                )

    def test_encoder_exception_wrapped(self) -> None:
        """Test EncoderException is eventually raised after retry exhaustion."""
        with patch.object(ModelManager, "get_instance") as mock_get_instance:
            mock_manager = MagicMock()
            mock_manager.device = "cpu"
            mock_manager.is_ready.return_value = True

            mock_manager.get_model.return_value = (MagicMock(), MagicMock())
            mock_manager.get_hidden_size.return_value = 768
            mock_get_instance.return_value = mock_manager

            tool = ContextAwareCompressionTool()

            # Mock _get_encoder to return an encoder that raises
            mock_encoder = MagicMock()
            mock_encoder.encode.side_effect = EncoderException("Encoding failed")

            with (
                patch.object(tool, "_get_encoder", return_value=mock_encoder),
                pytest.raises(CompressionException, match="Encoding failed"),
            ):
                # The retry decorator will retry 3 times, then raise CompressionException
                # Need 4+ sentences with 3+ words each to trigger encoding
                long_context = (
                    "This is a valid sentence here. "
                    "Another valid sentence here. "
                    "Third sentence is here now. "
                    "Fourth sentence is also here."
                )
                tool.compress(context=long_context, question="What is this?")

    def test_general_exception_wrapped(self) -> None:
        """Test general exceptions eventually propagate after retry exhaustion."""
        with patch.object(ModelManager, "get_instance") as mock_get_instance:
            mock_manager = MagicMock()
            mock_manager.device = "cpu"
            mock_manager.is_ready.return_value = True
            mock_manager.get_model.return_value = (MagicMock(), MagicMock())
            mock_manager.get_hidden_size.return_value = 768
            mock_get_instance.return_value = mock_manager

            tool = ContextAwareCompressionTool()

            # Mock _get_encoder to return an encoder that raises a general exception
            mock_encoder = MagicMock()
            mock_encoder.encode.side_effect = RuntimeError("Unexpected error")

            with (
                patch.object(tool, "_get_encoder", return_value=mock_encoder),
                pytest.raises(CompressionException, match="Compression failed"),
            ):
                # The retry decorator will retry 3 times, then raise CompressionException
                long_context = (
                    "This is a valid sentence here. "
                    "Another valid sentence here. "
                    "Third sentence is here now. "
                    "Fourth sentence is also here."
                )
                tool.compress(context=long_context, question="What is this?")


class TestCompressionToolEdgeCases:
    """Test edge cases in compression."""

    @pytest.fixture(autouse=True)
    def setup_mock_manager(self) -> Generator[None, None, None]:
        """Set up mock ModelManager for all tests."""
        ModelManager.reset_instance()
        yield
        ModelManager.reset_instance()

    def test_few_sentences_returns_unchanged(self) -> None:
        """Test that few sentences (<=3) return unchanged with actual relevance scores."""
        with patch.object(ModelManager, "get_instance") as mock_get_instance:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768

            # Mock model output for embedding computation
            mock_output = MagicMock()
            # 4 embeddings: 1 for question + 3 for sentences (batch encode calls)
            mock_output.last_hidden_state = torch.randn(4, 10, 768)
            mock_model.return_value = mock_output
            mock_model.__call__ = MagicMock(return_value=mock_output)

            # Mock batch encoding with proper tensor returns
            mock_batch_encoding = MagicMock()
            mock_batch_encoding.__getitem__ = lambda self, key: (
                torch.randint(0, 1000, (4, 10)) if key == "input_ids" else torch.ones(4, 10)
            )
            mock_batch_encoding.to = MagicMock(return_value=mock_batch_encoding)

            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = mock_batch_encoding
            mock_tokenizer.encode.return_value = list(range(10))

            mock_manager = MagicMock()
            mock_manager.device = "cpu"
            mock_manager.is_ready.return_value = True
            mock_manager.get_model.return_value = (mock_model, mock_tokenizer)
            mock_manager.get_hidden_size.return_value = 768
            mock_get_instance.return_value = mock_manager

            tool = ContextAwareCompressionTool()

            # Context with exactly 3 sentences
            context = "First sentence here. Second sentence here. Third sentence here."
            result = tool.compress(context=context, question="What is this?")

            assert result.compression_ratio == 1.0
            assert result.sentences_kept == 3
            assert result.sentences_removed == 0
            # Should have actual relevance scores now (not all 1.0)
            assert len(result.relevance_scores) == 3

    def test_ensures_at_least_one_sentence_selected(self) -> None:
        """Test that at least one sentence is always selected."""
        with patch.object(ModelManager, "get_instance") as mock_get_instance:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768

            mock_output = MagicMock()
            # Return embeddings for question + 4 sentences
            mock_output.last_hidden_state = torch.randn(5, 10, 768)
            mock_model.return_value = mock_output
            mock_model.__call__ = MagicMock(return_value=mock_output)

            mock_batch_encoding = MagicMock()
            mock_batch_encoding.__getitem__ = lambda self, key: (
                torch.randint(0, 1000, (5, 10)) if key == "input_ids" else torch.ones(5, 10)
            )
            mock_batch_encoding.to = MagicMock(return_value=mock_batch_encoding)

            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = mock_batch_encoding
            mock_tokenizer.encode.return_value = list(range(100))  # Long tokens

            mock_manager = MagicMock()
            mock_manager.device = "cpu"
            mock_manager.is_ready.return_value = True
            mock_manager.get_model.return_value = (mock_model, mock_tokenizer)
            mock_manager.get_hidden_size.return_value = 768
            mock_get_instance.return_value = mock_manager

            tool = ContextAwareCompressionTool()

            # Very long sentences with very low compression ratio
            context = (
                "This is a very long sentence that takes up a lot of space. "
                "Another very long sentence that is quite lengthy. "
                "Yet another sentence that is also pretty long. "
                "And one more sentence to make it four total."
            )

            result = tool.compress(context=context, question="What is this?", compression_ratio=0.1)

            # Should have at least 1 sentence
            assert result.sentences_kept >= 1


class TestModelManagerInitialization:
    """Test ModelManager initialization scenarios."""

    @pytest.fixture(autouse=True)
    def reset_manager(self) -> Generator[None, None, None]:
        """Reset ModelManager before each test."""
        ModelManager.reset_instance()
        yield
        ModelManager.reset_instance()

    def test_init_calls_initialize_when_not_ready(self) -> None:
        """Test that initialize is called when manager not ready."""
        with patch.object(ModelManager, "get_instance") as mock_get_instance:
            mock_manager = MagicMock()
            mock_manager.device = "cpu"
            mock_manager.is_ready.return_value = False
            mock_manager.initialize = MagicMock()
            mock_get_instance.return_value = mock_manager

            ContextAwareCompressionTool(model_name="test-model")

            mock_manager.initialize.assert_called_once_with("test-model", blocking=True)

    def test_init_skips_initialize_when_ready(self) -> None:
        """Test that initialize is skipped when manager already ready."""
        with patch.object(ModelManager, "get_instance") as mock_get_instance:
            mock_manager = MagicMock()
            mock_manager.device = "cpu"
            mock_manager.is_ready.return_value = True
            mock_manager.initialize = MagicMock()
            mock_get_instance.return_value = mock_manager

            ContextAwareCompressionTool()

            mock_manager.initialize.assert_not_called()
