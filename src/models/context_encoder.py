"""Context Encoder wrapper for semantic embeddings.

Provides a unified interface for encoding text into dense vector representations
using sentence transformer models. Supports batched encoding, caching, and
multiple pooling strategies.

This module enhances compression and reasoning tools by providing high-quality
semantic embeddings for similarity comparisons and clustering.

Note: Uses ModelManager to share the embedding model instance with other tools
(e.g., compress.py), avoiding duplicate model loads and memory waste.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from concurrent.futures import Executor
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger

from src.models.model_manager import ModelManager
from src.utils.errors import ModelNotReadyException, ReasonGuardException


class PoolingStrategy(str, Enum):
    """Supported pooling strategies for token embeddings."""

    MEAN = "mean"  # Average of all token embeddings
    CLS = "cls"  # [CLS] token embedding only
    MAX = "max"  # Element-wise maximum
    MEAN_SQRT_LEN = "mean_sqrt_len"  # Mean divided by sqrt(sequence_length)


class EncoderException(ReasonGuardException):
    """Raised during encoding failures."""

    pass


@dataclass
class EncodingResult:
    """Result from text encoding operation.

    Attributes:
        embeddings: Tensor of shape (batch_size, hidden_size) or (hidden_size,).
        tokens_per_text: Number of tokens in each input text.
        pooling_strategy: Strategy used for pooling.
        model_name: Name of the encoder model.

    """

    embeddings: torch.Tensor
    tokens_per_text: list[int]
    pooling_strategy: PoolingStrategy
    model_name: str
    cached: bool = False

    def to_numpy(self) -> Any:
        """Convert embeddings to numpy array."""
        return self.embeddings.cpu().numpy()

    def similarity(self, other: EncodingResult) -> torch.Tensor:
        """Compute cosine similarity with another encoding result.

        Args:
            other: Another EncodingResult to compare against.

        Returns:
            Tensor of similarity scores.

        """
        return F.cosine_similarity(self.embeddings, other.embeddings, dim=-1)


@dataclass
class EncoderConfig:
    """Configuration for ContextEncoder.

    Attributes:
        model_name: HuggingFace model name for embeddings.
        pooling_strategy: How to pool token embeddings.
        max_length: Maximum sequence length.
        normalize: Whether to L2-normalize embeddings.
        cache_size: Maximum number of cached embeddings.
        batch_size: Default batch size for encoding.

    """

    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    max_length: int = 512
    normalize: bool = True
    cache_size: int = 1000
    batch_size: int = 32


class LRUCache:
    """Memory-aware LRU cache for embedding results.

    P1 Optimization: Tracks memory usage and evicts based on both
    item count and total memory consumed. Tensors are moved to CPU
    to free GPU memory.
    """

    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0) -> None:
        """Initialize LRU cache with memory limits.

        Args:
            max_size: Maximum number of items to cache.
            max_memory_mb: Maximum memory in megabytes to use.

        """
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._current_memory = 0
        self.hits = 0
        self.misses = 0

    def _tensor_memory(self, t: torch.Tensor) -> int:
        """Estimate tensor memory in bytes."""
        return int(t.element_size() * t.nelement())

    def get(self, key: str) -> torch.Tensor | None:
        """Get item from cache, updating LRU order.

        Args:
            key: Cache key.

        Returns:
            Cached tensor or None if not found.

        """
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: torch.Tensor) -> None:
        """Put item in cache, evicting oldest if necessary.

        P1 Optimization: Evicts based on memory limit in addition to
        item count. Moves tensors to CPU for caching to free GPU memory.

        Args:
            key: Cache key.
            value: Tensor to cache.

        """
        tensor_size = self._tensor_memory(value)

        # Evict until we have space (memory and count limits)
        while (
            self._current_memory + tensor_size > self.max_memory_bytes
            or len(self.cache) >= self.max_size
        ) and self.cache:
            _, old_tensor = self.cache.popitem(last=False)
            self._current_memory -= self._tensor_memory(old_tensor)

        # Handle update of existing key
        if key in self.cache:
            old = self.cache.pop(key)
            self._current_memory -= self._tensor_memory(old)

        # Move tensor to CPU for caching (frees GPU memory)
        value = value.detach().cpu()
        self.cache[key] = value
        self._current_memory += tensor_size

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self._current_memory = 0
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def memory_usage_mb(self) -> float:
        """Get current memory usage in megabytes."""
        return self._current_memory / (1024 * 1024)


class ContextEncoder:
    """Context encoder for semantic embeddings.

    Provides a unified interface for encoding text into dense vectors
    using pretrained sentence transformer models.

    Example:
        >>> encoder = ContextEncoder()
        >>> result = encoder.encode("What is the meaning of life?")
        >>> print(result.embeddings.shape)  # torch.Size([768])

        >>> # Batch encoding
        >>> texts = ["First sentence.", "Second sentence."]
        >>> result = encoder.encode_batch(texts)
        >>> print(result.embeddings.shape)  # torch.Size([2, 768])

        >>> # Async batch encoding
        >>> result = await encoder.encode_batch_async(texts)

    """

    def __init__(
        self,
        config: EncoderConfig | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize context encoder.

        Uses ModelManager to get the shared model instance, avoiding duplicate
        model loads when both ContextEncoder and compress tool are used.

        Args:
            config: Encoder configuration. If None, uses defaults.
            model_name: Override model name from config.

        Raises:
            EncoderException: If model loading fails.
            ModelNotReadyException: If model is still downloading/loading.

        """
        self.config = config or EncoderConfig()
        if model_name:
            self.config.model_name = model_name

        # Get shared model instance from ModelManager
        manager = ModelManager.get_instance()
        try:
            self.model, self.tokenizer = manager.get_model()
            self.device = manager.device
            self._inference_lock = manager.inference_lock()
            logger.debug(f"ContextEncoder using shared model on {self.device}")
        except ModelNotReadyException:
            # Re-raise as-is for caller to handle
            raise
        except Exception as e:
            logger.error(f"Failed to get encoder model: {e}")
            raise EncoderException(f"Failed to load encoder: {e}") from e

        # Initialize embedding cache (separate from model cache)
        self._cache = LRUCache(max_size=self.config.cache_size)

        # Get hidden size for reference
        self.hidden_size = manager.get_hidden_size()

    def encode(
        self,
        text: str,
        pooling: PoolingStrategy | None = None,
        use_cache: bool = True,
    ) -> EncodingResult:
        """Encode single text to embedding.

        Args:
            text: Text to encode.
            pooling: Override pooling strategy.
            use_cache: Whether to use embedding cache.

        Returns:
            EncodingResult with embedding tensor.

        Raises:
            EncoderException: If encoding fails.

        """
        result = self.encode_batch(
            [text],
            pooling=pooling,
            use_cache=use_cache,
        )
        return EncodingResult(
            embeddings=result.embeddings[0],
            tokens_per_text=result.tokens_per_text,
            pooling_strategy=result.pooling_strategy,
            model_name=result.model_name,
            cached=result.cached,
        )

    async def encode_async(
        self,
        text: str,
        pooling: PoolingStrategy | None = None,
        use_cache: bool = True,
        executor: Executor | None = None,
    ) -> EncodingResult:
        """Encode single text asynchronously.

        Args:
            text: Text to encode.
            pooling: Override pooling strategy.
            use_cache: Whether to use embedding cache.
            executor: Optional executor to run in.

        Returns:
            EncodingResult with embedding tensor.

        """
        return await self.encode_batch_async(
            [text], pooling=pooling, use_cache=use_cache, executor=executor
        )

    async def encode_batch_async(
        self,
        texts: list[str],
        pooling: PoolingStrategy | None = None,
        use_cache: bool = True,
        executor: Executor | None = None,
    ) -> EncodingResult:
        """Encode batch of texts asynchronously (non-blocking).

        Offloads the CPU-intensive encoding to a separate thread.

        Args:
            texts: List of texts to encode.
            pooling: Override pooling strategy.
            use_cache: Whether to use embedding cache.
            executor: Optional executor to run in (default: loop's default).

        Returns:
            EncodingResult with stacked embedding tensors.

        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            lambda: self.encode_batch(texts, pooling=pooling, use_cache=use_cache),
        )

    def encode_batch(
        self,
        texts: list[str],
        pooling: PoolingStrategy | None = None,
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> EncodingResult:
        """Encode batch of texts to embeddings.

        Args:
            texts: List of texts to encode.
            pooling: Override pooling strategy.
            use_cache: Whether to use embedding cache.
            show_progress: Whether to show progress (for large batches).

        Returns:
            EncodingResult with stacked embedding tensors.

        Raises:
            EncoderException: If encoding fails.

        """
        if not texts:
            raise EncoderException("Cannot encode empty text list")

        pooling = pooling or self.config.pooling_strategy
        all_embeddings: list[tuple[int, torch.Tensor]] = []
        tokens_per_text: list[int] = []
        texts_to_encode: list[tuple[int, str]] = []
        cached_count = 0

        # Check cache for each text
        for idx, text in enumerate(texts):
            if not text or not text.strip():
                # Handle empty strings gracefully - return zero vector or skip
                # For now, we'll raise to maintain strictness, but could be relaxed
                raise EncoderException(f"Empty text at index {idx}")

            cache_key = self._cache_key(text, pooling)
            if use_cache:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    all_embeddings.append((idx, cached))
                    cached_count += 1
                    tokens_per_text.append(-1)  # Unknown for cached
                    continue

            texts_to_encode.append((idx, text))

        # Encode non-cached texts in batches
        if texts_to_encode:
            try:
                for batch_start in range(0, len(texts_to_encode), self.config.batch_size):
                    batch = texts_to_encode[batch_start : batch_start + self.config.batch_size]
                    batch_texts = [t for _, t in batch]
                    batch_indices = [i for i, _ in batch]

                    # Use inference lock to protect tokenizer (not thread-safe)
                    with self._inference_lock:
                        # Tokenize
                        inputs = self.tokenizer(
                            batch_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_length,
                        ).to(self.device)

                        # Get token counts from attention_mask (avoids double tokenization)
                        # Count non-padding, non-special tokens per sequence
                        attention_mask = inputs["attention_mask"]
                        for i in range(attention_mask.shape[0]):
                            # Subtract 2 for [CLS] and [SEP] special tokens
                            token_count = int(attention_mask[i].sum().item()) - 2
                            tokens_per_text.append(max(0, token_count))

                        # Forward pass
                        with torch.no_grad():
                            outputs = self.model(**inputs)

                        # Pool embeddings
                        embeddings = self._pool_embeddings(
                            outputs.last_hidden_state,
                            inputs["attention_mask"],
                            pooling,
                        )

                    # Normalize if configured (outside lock - pure tensor ops)
                    if self.config.normalize:
                        embeddings = F.normalize(embeddings, p=2, dim=1)

                    # Cache and collect results
                    for i, idx in enumerate(batch_indices):
                        emb = embeddings[i]
                        if use_cache:
                            cache_key = self._cache_key(batch_texts[i], pooling)
                            self._cache.put(cache_key, emb)
                        all_embeddings.append((idx, emb))

            except Exception as e:
                logger.error(f"Batch encoding failed: {e}")
                raise EncoderException(f"Encoding failed: {e}") from e

        # Sort by original index and stack
        all_embeddings.sort(key=lambda x: x[0])
        stacked = torch.stack([emb for _, emb in all_embeddings])

        return EncodingResult(
            embeddings=stacked,
            tokens_per_text=tokens_per_text,
            pooling_strategy=pooling,
            model_name=self.config.model_name,
            cached=cached_count == len(texts),
        )

    def _pool_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        strategy: PoolingStrategy,
    ) -> torch.Tensor:
        """Pool token embeddings according to strategy.

        Args:
            token_embeddings: Shape (batch, seq_len, hidden).
            attention_mask: Shape (batch, seq_len).
            strategy: Pooling strategy to use.

        Returns:
            Pooled embeddings of shape (batch, hidden).

        """
        if strategy == PoolingStrategy.CLS:
            return token_embeddings[:, 0, :]

        elif strategy == PoolingStrategy.MAX:
            # Set padding tokens to large negative value
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            token_embeddings[mask_expanded == 0] = -1e9
            return torch.max(token_embeddings, dim=1)[0]

        elif strategy == PoolingStrategy.MEAN_SQRT_LEN:
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            seq_lengths = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1)
            return sum_embeddings / torch.sqrt(seq_lengths.float())

        else:  # MEAN (default)
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

    def _cache_key(self, text: str, pooling: PoolingStrategy) -> str:
        """Generate cache key for text and pooling combination."""
        content = f"{self.config.model_name}:{pooling.value}:{text}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

    def similarity_matrix(
        self,
        texts_a: list[str],
        texts_b: list[str] | None = None,
    ) -> torch.Tensor:
        """Compute pairwise similarity matrix between two text lists.

        Args:
            texts_a: First list of texts.
            texts_b: Second list of texts. If None, computes self-similarity.

        Returns:
            Similarity matrix of shape (len(texts_a), len(texts_b)).

        """
        emb_a = self.encode_batch(texts_a).embeddings
        emb_b = emb_a if texts_b is None else self.encode_batch(texts_b).embeddings

        # Compute cosine similarity matrix
        return F.cosine_similarity(
            emb_a.unsqueeze(1),  # (n, 1, d)
            emb_b.unsqueeze(0),  # (1, m, d)
            dim=2,
        )  # (n, m)

    def find_most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, str, float]]:
        """Find most similar candidates to a query.

        Args:
            query: Query text.
            candidates: List of candidate texts.
            top_k: Number of top results to return.

        Returns:
            List of (index, text, similarity_score) tuples, sorted by score.

        """
        query_emb = self.encode(query).embeddings.unsqueeze(0)
        cand_embs = self.encode_batch(candidates).embeddings

        similarities = F.cosine_similarity(query_emb, cand_embs, dim=1)

        # Get top-k indices
        top_k = min(top_k, len(candidates))
        top_scores, top_indices = torch.topk(similarities, top_k)

        return [
            (int(idx.item()), candidates[int(idx.item())], float(score.item()))
            for idx, score in zip(top_indices, top_scores, strict=False)
        ]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics including memory usage."""
        return {
            "size": len(self._cache.cache),
            "max_size": self._cache.max_size,
            "hits": self._cache.hits,
            "misses": self._cache.misses,
            "hit_rate": self._cache.hit_rate,
            "memory_usage_mb": self._cache.memory_usage_mb,
            "max_memory_mb": self._cache.max_memory_bytes / (1024 * 1024),
        }


# Convenience function for one-off encoding
def encode_text(
    text: str,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> torch.Tensor:
    """Encode single text to embedding (convenience function).

    Note: Creates a new encoder instance each time. For repeated use,
    create a ContextEncoder instance instead.

    Args:
        text: Text to encode.
        model_name: Model to use for encoding.

    Returns:
        Normalized embedding tensor.

    """
    encoder = ContextEncoder(model_name=model_name)
    return encoder.encode(text).embeddings
