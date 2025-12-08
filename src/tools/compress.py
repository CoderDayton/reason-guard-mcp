"""Context-aware semantic prompt compression tool (CPC).

Implements semantic sentence-level compression that preserves
relevance to a given question while reducing token count.

Based on paper 2409.01227v3 - achieves 10.93× speedup over
token-level methods with minimal quality loss (-0.3 F1).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from src.utils.errors import CompressionException
from src.utils.retry import retry_with_backoff
from src.utils.schema import CompressionResult

if TYPE_CHECKING:
    pass


class ContextAwareCompressionTool:
    """Context-aware semantic prompt compression.

    Uses sentence embeddings to score relevance to a question,
    then selects most relevant sentences to achieve target compression.

    Attributes:
        model_name: Name of the sentence transformer model.
        device: Device to run model on (cuda/cpu).
        tokenizer: Tokenizer for the embedding model.
        model: The embedding model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        """Initialize compression tool with encoder model.

        Args:
            model_name: HuggingFace model name for sentence embeddings.
                       Default is all-mpnet-base-v2 for good quality/speed balance.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading compression encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Compression encoder loaded on {self.device}")

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
        2. Encode question and all sentences
        3. Score sentences by cosine similarity to question
        4. Select top sentences until target ratio reached
        5. Optionally restore original order

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
            # Split into sentences
            sentences = self._split_sentences(context, min_sentence_length)

            if not sentences:
                raise CompressionException("Could not split context into valid sentences")

            # If only a few sentences, return as-is
            if len(sentences) <= 3:
                original_tokens = self._count_tokens(context)
                return CompressionResult(
                    compressed_context=context,
                    compression_ratio=1.0,
                    original_tokens=original_tokens,
                    compressed_tokens=original_tokens,
                    sentences_kept=len(sentences),
                    sentences_removed=0,
                    relevance_scores=[(s, 1.0) for s in sentences],
                )

            original_tokens = self._count_tokens(context)

            # Encode question
            question_emb = self._encode_text(question)

            # Score each sentence by relevance
            scored_sentences: list[tuple[int, str, float]] = []
            for idx, sent in enumerate(sentences):
                sent_emb = self._encode_text(sent)
                relevance = F.cosine_similarity(
                    question_emb.unsqueeze(0),
                    sent_emb.unsqueeze(0),
                    dim=1,
                ).item()
                scored_sentences.append((idx, sent, float(relevance)))

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
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise CompressionException(f"Compression failed: {e!s}") from e

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to normalized embedding vector.

        Args:
            text: Text to encode.

        Returns:
            Normalized embedding tensor of shape (hidden_size,).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling over token embeddings
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
                input_mask_expanded.sum(dim=1), min=1e-9
            )

        return F.normalize(embeddings, p=2, dim=1)[0]

    def _split_sentences(self, text: str, min_words: int = 3) -> list[str]:
        """Split text into sentences robustly.

        Handles common abbreviations and edge cases.

        Args:
            text: Text to split.
            min_words: Minimum words per sentence.

        Returns:
            List of sentence strings.
        """
        # Handle common abbreviations to avoid false splits
        abbrevs = [
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
        for abbrev in abbrevs:
            text = text.replace(abbrev, abbrev.replace(".", "<DOT>"))

        # Split on sentence endings
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text)

        # Restore abbreviations
        sentences = [s.replace("<DOT>", ".").strip() for s in sentences]

        # Filter short/empty sentences
        return [s for s in sentences if s and len(s.split()) >= min_words]

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))
