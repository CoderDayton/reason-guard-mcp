"""Tests for context_encoder module.

Tests semantic embedding encoding, caching, and pooling strategies.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.context_encoder import (
    ContextEncoder,
    EncoderConfig,
    EncoderException,
    EncodingResult,
    LRUCache,
    PoolingStrategy,
)
from src.models.model_manager import ModelManager
from src.utils.errors import ModelNotReadyException


class TestPoolingStrategy:
    """Test PoolingStrategy enum."""

    def test_pooling_values(self) -> None:
        """Test all pooling strategy values."""
        assert PoolingStrategy.MEAN.value == "mean"
        assert PoolingStrategy.CLS.value == "cls"
        assert PoolingStrategy.MAX.value == "max"
        assert PoolingStrategy.MEAN_SQRT_LEN.value == "mean_sqrt_len"


class TestEncodingResult:
    """Test EncodingResult dataclass."""

    def test_to_numpy(self) -> None:
        """Test conversion to numpy array."""
        embeddings = torch.randn(768)
        result = EncodingResult(
            embeddings=embeddings,
            tokens_per_text=[10],
            pooling_strategy=PoolingStrategy.MEAN,
            model_name="test-model",
        )

        numpy_arr = result.to_numpy()
        assert numpy_arr.shape == (768,)

    def test_similarity(self) -> None:
        """Test similarity computation between results."""
        # Create two similar embeddings
        emb1 = torch.tensor([1.0, 0.0, 0.0])
        emb2 = torch.tensor([1.0, 0.0, 0.0])

        result1 = EncodingResult(
            embeddings=emb1,
            tokens_per_text=[5],
            pooling_strategy=PoolingStrategy.MEAN,
            model_name="test",
        )
        result2 = EncodingResult(
            embeddings=emb2,
            tokens_per_text=[5],
            pooling_strategy=PoolingStrategy.MEAN,
            model_name="test",
        )

        similarity = result1.similarity(result2)
        assert similarity.item() == pytest.approx(1.0, abs=0.01)

    def test_similarity_orthogonal(self) -> None:
        """Test similarity for orthogonal vectors."""
        emb1 = torch.tensor([1.0, 0.0, 0.0])
        emb2 = torch.tensor([0.0, 1.0, 0.0])

        result1 = EncodingResult(
            embeddings=emb1,
            tokens_per_text=[5],
            pooling_strategy=PoolingStrategy.MEAN,
            model_name="test",
        )
        result2 = EncodingResult(
            embeddings=emb2,
            tokens_per_text=[5],
            pooling_strategy=PoolingStrategy.MEAN,
            model_name="test",
        )

        similarity = result1.similarity(result2)
        assert similarity.item() == pytest.approx(0.0, abs=0.01)


class TestEncoderConfig:
    """Test EncoderConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EncoderConfig()
        assert config.model_name == "sentence-transformers/all-mpnet-base-v2"
        assert config.pooling_strategy == PoolingStrategy.MEAN
        assert config.max_length == 512
        assert config.normalize is True
        assert config.cache_size == 1000
        assert config.batch_size == 32

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = EncoderConfig(
            model_name="custom-model",
            pooling_strategy=PoolingStrategy.CLS,
            max_length=256,
            normalize=False,
            cache_size=500,
            batch_size=16,
        )
        assert config.model_name == "custom-model"
        assert config.pooling_strategy == PoolingStrategy.CLS
        assert config.max_length == 256
        assert config.normalize is False


class TestLRUCache:
    """Test LRUCache class."""

    def test_put_and_get(self) -> None:
        """Test basic put and get operations."""
        cache = LRUCache(max_size=10)
        tensor = torch.randn(768)

        cache.put("key1", tensor)
        result = cache.get("key1")

        assert result is not None
        assert torch.equal(result, tensor)

    def test_get_miss(self) -> None:
        """Test cache miss returns None."""
        cache = LRUCache(max_size=10)
        result = cache.get("nonexistent")
        assert result is None

    def test_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size=2)

        cache.put("key1", torch.tensor([1.0]))
        cache.put("key2", torch.tensor([2.0]))
        cache.put("key3", torch.tensor([3.0]))  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_lru_order_update_on_get(self) -> None:
        """Test that get updates LRU order."""
        cache = LRUCache(max_size=2)

        cache.put("key1", torch.tensor([1.0]))
        cache.put("key2", torch.tensor([2.0]))

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key3, should evict key2 (least recently used)
        cache.put("key3", torch.tensor([3.0]))

        assert cache.get("key1") is not None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None

    def test_lru_order_update_on_put(self) -> None:
        """Test that put updates LRU order for existing keys."""
        cache = LRUCache(max_size=2)

        cache.put("key1", torch.tensor([1.0]))
        cache.put("key2", torch.tensor([2.0]))

        # Update key1 to make it recently used
        cache.put("key1", torch.tensor([1.5]))

        # Add key3, should evict key2
        cache.put("key3", torch.tensor([3.0]))

        assert cache.get("key1") is not None
        assert cache.get("key2") is None

    def test_clear(self) -> None:
        """Test cache clearing."""
        cache = LRUCache(max_size=10)
        cache.put("key1", torch.tensor([1.0]))
        cache.put("key2", torch.tensor([2.0]))

        cache.clear()

        # After clear, gets will be misses but stats were reset
        initial_misses = cache.misses
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        # Misses increment after clear because gets return None
        assert cache.misses == initial_misses + 2
        assert cache.hits == 0

    def test_hit_rate(self) -> None:
        """Test hit rate calculation."""
        cache = LRUCache(max_size=10)
        cache.put("key1", torch.tensor([1.0]))

        # 2 hits
        cache.get("key1")
        cache.get("key1")

        # 2 misses
        cache.get("key2")
        cache.get("key3")

        assert cache.hit_rate == 0.5

    def test_hit_rate_empty(self) -> None:
        """Test hit rate with no accesses."""
        cache = LRUCache(max_size=10)
        assert cache.hit_rate == 0.0


class TestContextEncoder:
    """Test ContextEncoder class."""

    @pytest.fixture(autouse=True)
    def setup_mock_manager(self) -> Generator[None, None, None]:
        """Set up mock ModelManager for all tests."""
        ModelManager.reset_instance()

        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768

        # Mock forward pass output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 768)
        mock_model.return_value = mock_output
        mock_model.__call__ = MagicMock(return_value=mock_output)

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }
        mock_tokenizer.encode.return_value = list(range(10))

        # Patch ModelManager methods
        with (
            patch.object(ModelManager, "get_model", return_value=(mock_model, mock_tokenizer)),
            patch.object(ModelManager, "get_hidden_size", return_value=768),
            patch.object(ModelManager, "get_instance") as mock_get_instance,
        ):
            mock_manager = MagicMock()
            mock_manager.device = "cpu"
            mock_manager.get_model.return_value = (mock_model, mock_tokenizer)
            mock_manager.get_hidden_size.return_value = 768
            mock_get_instance.return_value = mock_manager

            yield

        ModelManager.reset_instance()

    def test_init_success(self) -> None:
        """Test successful initialization."""
        encoder = ContextEncoder()
        assert encoder.hidden_size == 768
        assert encoder.device == "cpu"

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = EncoderConfig(
            model_name="custom-model",
            pooling_strategy=PoolingStrategy.CLS,
        )
        encoder = ContextEncoder(config=config)
        assert encoder.config.model_name == "custom-model"
        assert encoder.config.pooling_strategy == PoolingStrategy.CLS

    def test_init_with_model_name_override(self) -> None:
        """Test model_name parameter overrides config."""
        config = EncoderConfig(model_name="config-model")
        encoder = ContextEncoder(config=config, model_name="override-model")
        assert encoder.config.model_name == "override-model"

    def test_init_model_not_ready_raises(self) -> None:
        """Test initialization raises when model not ready."""
        ModelManager.reset_instance()

        with patch.object(ModelManager, "get_instance") as mock_get_instance:
            mock_manager = MagicMock()
            mock_manager.get_model.side_effect = ModelNotReadyException("Not ready")
            mock_get_instance.return_value = mock_manager

            with pytest.raises(ModelNotReadyException):
                ContextEncoder()

    def test_init_general_exception(self) -> None:
        """Test initialization wraps general exceptions."""
        ModelManager.reset_instance()

        with patch.object(ModelManager, "get_instance") as mock_get_instance:
            mock_manager = MagicMock()
            mock_manager.get_model.side_effect = RuntimeError("Unexpected error")
            mock_get_instance.return_value = mock_manager

            with pytest.raises(EncoderException, match="Failed to load encoder"):
                ContextEncoder()

    def test_encode_empty_list_raises(self) -> None:
        """Test encoding empty list raises exception."""
        encoder = ContextEncoder()
        with pytest.raises(EncoderException, match="Cannot encode empty"):
            encoder.encode_batch([])

    def test_encode_empty_text_raises(self) -> None:
        """Test encoding empty text raises exception."""
        encoder = ContextEncoder()
        with pytest.raises(EncoderException, match="Empty text at index"):
            encoder.encode_batch(["valid", ""])

    def test_encode_whitespace_only_raises(self) -> None:
        """Test encoding whitespace-only text raises exception."""
        encoder = ContextEncoder()
        with pytest.raises(EncoderException, match="Empty text at index"):
            encoder.encode_batch(["valid", "   "])

    def test_cache_key_generation(self) -> None:
        """Test cache key is consistent."""
        encoder = ContextEncoder()
        key1 = encoder._cache_key("test text", PoolingStrategy.MEAN)
        key2 = encoder._cache_key("test text", PoolingStrategy.MEAN)
        key3 = encoder._cache_key("test text", PoolingStrategy.CLS)

        assert key1 == key2
        assert key1 != key3

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        encoder = ContextEncoder()
        encoder._cache.put("test", torch.randn(768))
        encoder.clear_cache()
        assert len(encoder._cache.cache) == 0

    def test_cache_stats(self) -> None:
        """Test cache statistics."""
        encoder = ContextEncoder()
        stats = encoder.cache_stats

        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


class TestPoolingStrategies:
    """Test different pooling strategies."""

    def test_cls_pooling(self) -> None:
        """Test CLS token pooling."""
        encoder = ContextEncoder.__new__(ContextEncoder)
        encoder.config = EncoderConfig()

        # Create mock embeddings (batch=2, seq=10, hidden=8)
        token_emb = torch.randn(2, 10, 8)
        attention_mask = torch.ones(2, 10)

        result = encoder._pool_embeddings(token_emb, attention_mask, PoolingStrategy.CLS)

        # CLS pooling should return first token
        assert result.shape == (2, 8)
        assert torch.equal(result, token_emb[:, 0, :])

    def test_max_pooling(self) -> None:
        """Test max pooling."""
        encoder = ContextEncoder.__new__(ContextEncoder)
        encoder.config = EncoderConfig()

        # Create simple embeddings
        token_emb = torch.tensor([[[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]]])  # (1, 3, 2)
        attention_mask = torch.ones(1, 3)

        result = encoder._pool_embeddings(token_emb, attention_mask, PoolingStrategy.MAX)

        assert result.shape == (1, 2)
        # Max should be [3.0, 4.0]
        assert result[0, 0].item() == pytest.approx(3.0)
        assert result[0, 1].item() == pytest.approx(4.0)

    def test_max_pooling_with_padding(self) -> None:
        """Test max pooling ignores padding."""
        encoder = ContextEncoder.__new__(ContextEncoder)
        encoder.config = EncoderConfig()

        # Last position is padding with high values
        token_emb = torch.tensor([[[1.0, 2.0], [3.0, 1.0], [100.0, 100.0]]])
        attention_mask = torch.tensor([[1.0, 1.0, 0.0]])  # Third position is padding

        result = encoder._pool_embeddings(token_emb, attention_mask, PoolingStrategy.MAX)

        # Should ignore padding position
        assert result[0, 0].item() == pytest.approx(3.0)
        assert result[0, 1].item() == pytest.approx(2.0)

    def test_mean_sqrt_len_pooling(self) -> None:
        """Test mean sqrt length pooling."""
        encoder = ContextEncoder.__new__(ContextEncoder)
        encoder.config = EncoderConfig()

        # Create simple embeddings
        token_emb = torch.tensor([[[2.0, 4.0], [2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]])  # (1, 4, 2)
        attention_mask = torch.ones(1, 4)

        result = encoder._pool_embeddings(token_emb, attention_mask, PoolingStrategy.MEAN_SQRT_LEN)

        assert result.shape == (1, 2)
        # Sum = [8, 16], sqrt(4) = 2, result = [4, 8]
        assert result[0, 0].item() == pytest.approx(4.0)
        assert result[0, 1].item() == pytest.approx(8.0)

    def test_mean_pooling_default(self) -> None:
        """Test mean pooling (default)."""
        encoder = ContextEncoder.__new__(ContextEncoder)
        encoder.config = EncoderConfig()

        # Create simple embeddings
        token_emb = torch.tensor([[[2.0, 4.0], [4.0, 8.0]]])  # (1, 2, 2)
        attention_mask = torch.ones(1, 2)

        result = encoder._pool_embeddings(token_emb, attention_mask, PoolingStrategy.MEAN)

        assert result.shape == (1, 2)
        # Mean should be [3.0, 6.0]
        assert result[0, 0].item() == pytest.approx(3.0)
        assert result[0, 1].item() == pytest.approx(6.0)


class TestEncodeFunctions:
    """Test encode_text convenience function."""

    def test_encode_text_function(self) -> None:
        """Test encode_text convenience function."""
        ModelManager.reset_instance()

        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 768)
        mock_model.return_value = mock_output
        mock_model.__call__ = MagicMock(return_value=mock_output)

        # Create a proper mock for tokenizer that supports .to()
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
            mock_manager.get_model.return_value = (mock_model, mock_tokenizer)
            mock_manager.get_hidden_size.return_value = 768
            mock_get_instance.return_value = mock_manager

            from src.models.context_encoder import encode_text

            result = encode_text("test text")
            assert result.shape == (768,)

        ModelManager.reset_instance()
