"""Context-aware semantic prompt compression tool (CPC).

Implements semantic sentence-level compression that preserves
relevance to a given question while reducing token count.

Based on paper 2409.01227v3 - achieves 10.93× speedup over
token-level methods with minimal quality loss (-0.3 F1).
"""

from __future__ import annotations

import hashlib
import re

import torch.nn.functional as F
from loguru import logger

from src.models.context_encoder import ContextEncoder, EncoderConfig, EncoderException
from src.models.model_manager import ModelManager
from src.utils.errors import CompressionException, ModelNotReadyException
from src.utils.retry import retry_with_backoff
from src.utils.schema import CompressionResult


class ContextAwareCompressionTool:
    """Context-aware semantic prompt compression.

    Uses ContextEncoder for sentence embeddings to score relevance to a question,
    then selects most relevant sentences to achieve target compression.

    Attributes:
        model_name: Name of the sentence transformer model.
        _encoder: Lazy-loaded ContextEncoder instance.

    """

    # P2 Optimization: Class-level abbreviation list (compiled once)
    _ABBREVIATIONS = [
        "Mr.",
        "Mrs.",
        "Dr.",
        "Prof.",
        "Inc.",
        "Ltd.",
        "Jr.",
        "Sr.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
    ]

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        """Initialize compression tool with encoder model.

        Args:
            model_name: HuggingFace model name for sentence embeddings.
                       Default is all-mpnet-base-v2 for good quality/speed balance.

        Note:
            The encoder is lazy-loaded on first use via ModelManager which handles
            caching and download state. If the model is not yet ready, compress()
            will raise ModelNotReadyException with a helpful message.

        """
        self.model_name = model_name
        self._encoder: ContextEncoder | None = None
        self._model_manager = ModelManager.get_instance()

        # P1 Optimization: Token counting cache (text_hash -> token_count)
        self._token_cache: dict[str, int] = {}
        self._token_cache_max_size = 1000

        # P2 Optimization: Sentence splitting cache (text_hash -> sentences)
        self._sentence_cache: dict[str, tuple[str, ...]] = {}
        self._sentence_cache_max_size = 100

        # Ensure model manager is initialized (blocking to ensure model is ready)
        if not self._model_manager.is_ready():
            self._model_manager.initialize(model_name, blocking=True)

    @property
    def device(self) -> str:
        """Get device from model manager."""
        return self._model_manager.device

    def _get_encoder(self) -> ContextEncoder:
        """Get or create ContextEncoder instance.

        Returns:
            ContextEncoder instance.

        Raises:
            ModelNotReadyException: If model is not ready (downloading/loading/failed).

        """
        if self._encoder is None:
            config = EncoderConfig(
                model_name=self.model_name,
                cache_size=500,  # Cache embeddings for repeated sentences
            )
            self._encoder = ContextEncoder(config=config)
        return self._encoder

    @retry_with_backoff(max_attempts=3, base_delay=0.5)
    def compress(
        self,
        context: str,
        question: str,
        compression_ratio: float = 0.3,
        preserve_order: bool = True,
        min_sentence_length: int = 3,
    ) -> CompressionResult:
        """Compress context while preserving semantic relevance to question.

        Algorithm:
        1. Split context into sentences
        2. Use ContextEncoder to find most similar sentences to question
        3. Select top sentences until target ratio reached
        4. Optionally restore original order

        Args:
            context: Full text context to compress.
            question: Query/question to determine relevance.
            compression_ratio: Target ratio (0.3 = keep 30% = 3× compression).
            preserve_order: Whether to restore original sentence order.
            min_sentence_length: Minimum words per sentence to keep.

        Returns:
            CompressionResult with compressed context and metrics.

        Raises:
            CompressionException: If compression fails due to invalid input
                                 or processing error.

        Example:
            >>> tool = ContextAwareCompressionTool()
            >>> result = tool.compress(
            ...     context="Long document text...",
            ...     question="What is the main topic?",
            ...     compression_ratio=0.3
            ... )
            >>> print(result.tokens_saved)  # e.g., 2500

        """
        # Validate inputs
        if not context or not context.strip():
            raise CompressionException("Context cannot be empty")

        if not question or not question.strip():
            raise CompressionException("Question cannot be empty")

        if not 0.1 <= compression_ratio <= 1.0:
            raise CompressionException(
                f"Compression ratio must be between 0.1 and 1.0, got {compression_ratio}"
            )

        try:
            # Get encoder (may raise ModelNotReadyException)
            encoder = self._get_encoder()

            # Split into sentences
            sentences = self._split_sentences(context, min_sentence_length)

            if not sentences:
                raise CompressionException("Could not split context into valid sentences")

            original_tokens = self._count_tokens(context)

            # Always compute relevance scores, even for small contexts
            encoder = self._get_encoder()
            query_emb = encoder.encode(question).embeddings.unsqueeze(0)
            sentence_results = encoder.encode_batch(sentences)
            sentence_embs = sentence_results.embeddings
            similarities = F.cosine_similarity(query_emb, sentence_embs, dim=1)
            relevance_scores = [
                (sent, float(similarities[idx].item())) for idx, sent in enumerate(sentences)
            ]

            # If only a few sentences, return as-is with actual relevance scores
            if len(sentences) <= 3:
                return CompressionResult(
                    compressed_context=context,
                    compression_ratio=1.0,
                    original_tokens=original_tokens,
                    compressed_tokens=original_tokens,
                    sentences_kept=len(sentences),
                    sentences_removed=0,
                    relevance_scores=sorted(relevance_scores, key=lambda x: x[1], reverse=True),
                )

            # Use ContextEncoder to score sentences by similarity to question
            # Build scored list: (index, sentence, score)
            scored_sentences: list[tuple[int, str, float]] = [
                (idx, sent, score) for idx, (sent, score) in enumerate(relevance_scores)
            ]

            # Sort by relevance (descending)
            scored_sentences.sort(key=lambda x: x[2], reverse=True)

            # Select sentences until budget exhausted
            max_chars = int(len(context) * compression_ratio)
            selected_indices: set[int] = set()
            char_count = 0

            for idx, sent, _score in scored_sentences:
                if char_count + len(sent) <= max_chars:
                    selected_indices.add(idx)
                    char_count += len(sent) + 1  # +1 for space

            # Ensure at least one sentence selected
            if not selected_indices and scored_sentences:
                selected_indices.add(scored_sentences[0][0])

            # Build compressed context
            if preserve_order:
                compressed_sents = [
                    sent for idx, sent in enumerate(sentences) if idx in selected_indices
                ]
            else:
                compressed_sents = [
                    sent for idx, sent, _ in scored_sentences if idx in selected_indices
                ]

            compressed_context = " ".join(compressed_sents)
            compressed_tokens = self._count_tokens(compressed_context)

            # Build relevance scores for output
            relevance_scores = [
                (sent, score) for idx, sent, score in scored_sentences if idx in selected_indices
            ]

            return CompressionResult(
                compressed_context=compressed_context,
                compression_ratio=compressed_tokens / original_tokens
                if original_tokens > 0
                else 1.0,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                sentences_kept=len(selected_indices),
                sentences_removed=len(sentences) - len(selected_indices),
                relevance_scores=relevance_scores,
            )

        except CompressionException:
            raise
        except ModelNotReadyException:
            # Re-raise model not ready errors as-is for proper user feedback
            raise
        except EncoderException as e:
            logger.error(f"Encoder error during compression: {e}")
            raise CompressionException(f"Encoding failed: {e!s}") from e
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise CompressionException(f"Compression failed: {e!s}") from e

    def _split_sentences(self, text: str, min_words: int = 3) -> list[str]:
        """Split text into sentences robustly with caching.

        P2 Optimization: Caches sentence splits for repeated compressions
        of the same text (e.g., different questions on same context).

        Handles common abbreviations and edge cases.

        Args:
            text: Text to split.
            min_words: Minimum words per sentence.

        Returns:
            List of sentence strings.

        """
        # Generate hash for cache key
        cache_key = hashlib.md5(f"{text}:{min_words}".encode(), usedforsecurity=False).hexdigest()

        if cache_key in self._sentence_cache:
            return list(self._sentence_cache[cache_key])

        # Perform actual splitting
        sentences = self._split_sentences_impl(text, min_words)

        # Cache result (as tuple for immutability)
        if len(self._sentence_cache) >= self._sentence_cache_max_size:
            # Remove first half of entries
            keys_to_remove = list(self._sentence_cache.keys())[: self._sentence_cache_max_size // 2]
            for key in keys_to_remove:
                del self._sentence_cache[key]

        self._sentence_cache[cache_key] = tuple(sentences)
        return sentences

    def _split_sentences_impl(self, text: str, min_words: int = 3) -> list[str]:
        """Internal sentence splitting implementation.

        Args:
            text: Text to split.
            min_words: Minimum words per sentence.

        Returns:
            List of sentence strings.

        """
        # Handle common abbreviations to avoid false splits
        for abbrev in self._ABBREVIATIONS:
            text = text.replace(abbrev, abbrev.replace(".", "<DOT>"))

        # Split on sentence endings
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text)

        # Restore abbreviations
        sentences = [s.replace("<DOT>", ".").strip() for s in sentences]

        # Filter short/empty sentences
        return [s for s in sentences if s and len(s.split()) >= min_words]

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer with caching.

        P1 Optimization: Uses hash-based memoization to avoid re-tokenizing
        the same text multiple times. Saves 10-20ms per compression.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.

        Raises:
            ModelNotReadyException: If model is not ready.

        """
        # Generate hash for cache key to limit memory usage
        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()

        if text_hash in self._token_cache:
            return self._token_cache[text_hash]

        _, tokenizer = self._model_manager.get_model()
        # Use inference lock - HuggingFace tokenizer's Rust backend is not thread-safe
        with self._model_manager.inference_lock():
            count = len(tokenizer.encode(text, add_special_tokens=False))

        # Evict oldest entries if cache is full
        if len(self._token_cache) >= self._token_cache_max_size:
            # Remove first half of entries (simple eviction)
            keys_to_remove = list(self._token_cache.keys())[: self._token_cache_max_size // 2]
            for key in keys_to_remove:
                del self._token_cache[key]

        self._token_cache[text_hash] = count
        return count
