"""Semantic scoring functions for MPPA survival probability.

This module provides semantic-aware scoring for reasoning steps,
replacing word-overlap heuristics with embedding-based similarity
and knowledge graph alignment.

Supports three modes of operation:
1. Word-overlap fallback (no dependencies)
2. Embedding-based via ContextEncoder (sync)
3. Embedding-based via AsyncVectorStore (async)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.context_encoder import ContextEncoder
    from src.models.knowledge_graph import KnowledgeGraph
    from src.models.vector_store import AsyncVectorStore


# Weights for combining score components
WEIGHTS = {
    "semantic_similarity": 0.35,
    "kg_alignment": 0.15,
    "specificity": 0.15,
    "structure": 0.15,
    "length": 0.10,
    "position": 0.10,
}

# Phrases indicating uncertainty/vagueness
VAGUE_PHRASES = frozenset(
    [
        "something",
        "somehow",
        "maybe",
        "probably",
        "i guess",
        "not sure",
        "might be",
        "could be",
        "perhaps",
    ]
)

# Logical connectors indicating structured reasoning
LOGICAL_CONNECTORS = frozenset(
    [
        "therefore",
        "because",
        "since",
        "thus",
        "so",
        "given that",
        "based on",
        "consequently",
        "hence",
        "as a result",
        "it follows",
    ]
)

# Strategy-specific keywords for MoT reasoning
STRATEGY_KEYWORDS: dict[str, frozenset[str]] = {
    "direct_factual": frozenset(
        ["fact", "stated", "according to", "explicitly", "evidence", "shows", "demonstrates"]
    ),
    "logical_inference": frozenset(
        ["therefore", "implies", "because", "since", "if", "then", "follows", "conclude"]
    ),
    "analogical": frozenset(
        ["similar", "like", "compared", "analogy", "reminds", "parallel", "resembles"]
    ),
    "causal": frozenset(
        ["cause", "effect", "result", "leads to", "because of", "consequence", "triggers"]
    ),
    "counterfactual": frozenset(
        ["if", "would", "could have", "alternatively", "suppose", "imagine", "hypothetically"]
    ),
}


def _word_overlap_score(thought: str, context: str) -> float:
    """Fallback word-overlap scoring when encoder unavailable.

    Args:
        thought: Candidate reasoning step.
        context: Problem context.

    Returns:
        Overlap score between 0.0 and 1.0.

    """
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "for"}
    context_words = set(context.lower().split()) - stopwords
    thought_words = set(thought.lower().split()) - stopwords

    if not context_words:
        return 0.5

    overlap = len(context_words & thought_words) / len(context_words)
    return min(overlap, 1.0)


def _specificity_score(thought: str) -> float:
    """Score thought specificity (concrete details vs vagueness).

    Args:
        thought: Candidate reasoning step.

    Returns:
        Specificity score between 0.0 and 1.0.

    """
    score = 0.5
    thought_lower = thought.lower()

    # Penalize vague phrases
    vague_count = sum(1 for phrase in VAGUE_PHRASES if phrase in thought_lower)
    score -= vague_count * 0.15

    # Reward concrete details
    has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?\b", thought))
    has_quotes = '"' in thought or "'" in thought
    has_proper_nouns = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", thought))

    if has_numbers:
        score += 0.2
    if has_quotes:
        score += 0.1
    if has_proper_nouns:
        score += 0.1

    return max(0.0, min(1.0, score))


def _structure_score(thought: str, strategy: str | None = None) -> float:
    """Score logical structure and reasoning quality.

    Args:
        thought: Candidate reasoning step.
        strategy: Optional reasoning strategy (for MoT).

    Returns:
        Structure score between 0.0 and 1.0.

    """
    score = 0.5
    thought_lower = thought.lower()

    # Reward logical connectors
    connector_count = sum(1 for conn in LOGICAL_CONNECTORS if conn in thought_lower)
    score += min(connector_count * 0.1, 0.3)

    # Strategy-specific bonus for MoT
    if strategy and strategy in STRATEGY_KEYWORDS:
        keywords = STRATEGY_KEYWORDS[strategy]
        keyword_hits = sum(1 for kw in keywords if kw in thought_lower)
        score += min(keyword_hits * 0.08, 0.2)

    return max(0.0, min(1.0, score))


def _length_score(thought: str) -> float:
    """Score thought length (penalize too short or too long).

    Args:
        thought: Candidate reasoning step.

    Returns:
        Length score between 0.0 and 1.0.

    """
    word_count = len(thought.split())

    if word_count < 10:
        return 0.3  # Too short - probably incomplete
    elif word_count < 20:
        return 0.6  # Marginal
    elif word_count <= 100:
        return 1.0  # Ideal range
    elif word_count <= 150:
        return 0.8  # Slightly verbose
    else:
        return 0.6  # Rambling


def _position_score(position: int, max_position: int = 10) -> float:
    """Score based on position in reasoning chain.

    Early positions get boost to encourage exploration.

    Args:
        position: Current step/column number (0-indexed or 1-indexed).
        max_position: Maximum expected position.

    Returns:
        Position score between 0.0 and 1.0.

    """
    # Normalize position to 0-1 range
    normalized = min(position, max_position) / max_position

    # Early positions (0-0.3) get boost, late positions neutral
    if normalized <= 0.3:
        return 1.0
    elif normalized <= 0.5:
        return 0.8
    else:
        return 0.6


def _kg_alignment_score(
    thought: str,
    kg: KnowledgeGraph | None,
) -> float:
    """Score alignment with knowledge graph facts.

    Args:
        thought: Candidate reasoning step.
        kg: Knowledge graph instance (optional).

    Returns:
        KG alignment score between 0.0 and 1.0.

    """
    if kg is None:
        return 0.5  # Neutral when no KG available

    supporting_facts = kg.get_supporting_facts(thought)

    if not supporting_facts:
        return 0.4  # Slight penalty for no supporting facts

    # Score based on number and confidence of supporting facts
    avg_confidence = sum(r.confidence for r in supporting_facts) / len(supporting_facts)
    fact_bonus = min(len(supporting_facts) * 0.1, 0.3)

    return min(0.5 + avg_confidence * 0.3 + fact_bonus, 1.0)


def semantic_survival_score(
    thought: str,
    context: str,
    position: int = 0,
    *,
    encoder: ContextEncoder | None = None,
    kg: KnowledgeGraph | None = None,
    strategy: str | None = None,
) -> float:
    """Calculate semantic survival probability for a candidate thought.

    Combines multiple scoring dimensions:
    - Semantic similarity (embedding-based when encoder available)
    - Knowledge graph alignment (fact support)
    - Specificity (concrete details vs vagueness)
    - Structure (logical connectors, strategy keywords)
    - Length (ideal range penalizes extremes)
    - Position (early exploration boost)

    Args:
        thought: Candidate reasoning step.
        context: Problem context and previous steps.
        position: Current step/column number.
        encoder: Optional ContextEncoder for semantic similarity.
        kg: Optional KnowledgeGraph for fact alignment.
        strategy: Optional reasoning strategy (for MoT).

    Returns:
        Score between 0.0 and 1.0 (higher = more likely to succeed).

    """
    # Handle empty input
    if not thought or not thought.strip():
        return 0.0
    if not context or not context.strip():
        context = ""

    # 1. Semantic similarity (or word overlap fallback)
    if encoder is not None:
        try:
            # Use embedding similarity
            similarity_matrix = encoder.similarity_matrix([thought], [context])
            semantic_sim = float(similarity_matrix[0, 0])
            # Normalize from [-1, 1] to [0, 1]
            semantic_sim = (semantic_sim + 1) / 2
        except Exception:
            # Fallback on any encoder error
            semantic_sim = _word_overlap_score(thought, context)
    else:
        semantic_sim = _word_overlap_score(thought, context)

    # 2. Knowledge graph alignment
    kg_score = _kg_alignment_score(thought, kg)

    # 3. Specificity
    specificity = _specificity_score(thought)

    # 4. Structure
    structure = _structure_score(thought, strategy)

    # 5. Length
    length = _length_score(thought)

    # 6. Position
    pos_score = _position_score(position)

    # Weighted combination
    final_score = (
        WEIGHTS["semantic_similarity"] * semantic_sim
        + WEIGHTS["kg_alignment"] * kg_score
        + WEIGHTS["specificity"] * specificity
        + WEIGHTS["structure"] * structure
        + WEIGHTS["length"] * length
        + WEIGHTS["position"] * pos_score
    )

    return max(0.0, min(1.0, final_score))


def calculate_survival_score(
    thought: str,
    context: str,
    step_number: int,
    *,
    encoder: ContextEncoder | None = None,
    kg: KnowledgeGraph | None = None,
) -> float:
    """Calculate survival score for long-chain reasoning.

    Wrapper around semantic_survival_score for backward compatibility
    with the long_chain module signature.

    Args:
        thought: Candidate reasoning step.
        context: Problem context and previous steps.
        step_number: Current position in the chain (1-indexed).
        encoder: Optional ContextEncoder for semantic similarity.
        kg: Optional KnowledgeGraph for fact alignment.

    Returns:
        Score between 0.0 and 1.0.

    """
    return semantic_survival_score(
        thought=thought,
        context=context,
        position=step_number,
        encoder=encoder,
        kg=kg,
        strategy=None,
    )


def calculate_cell_survival_score(
    thought: str,
    strategy: str,
    context: str,
    col: int,
    *,
    encoder: ContextEncoder | None = None,
    kg: KnowledgeGraph | None = None,
) -> float:
    """Calculate survival score for MoT matrix cells.

    Wrapper around semantic_survival_score for backward compatibility
    with the mot_reasoning module signature.

    Args:
        thought: Candidate reasoning step.
        strategy: Reasoning strategy being used.
        context: Problem context.
        col: Column index (0 = first iteration).
        encoder: Optional ContextEncoder for semantic similarity.
        kg: Optional KnowledgeGraph for fact alignment.

    Returns:
        Score between 0.0 and 1.0.

    """
    return semantic_survival_score(
        thought=thought,
        context=context,
        position=col,
        encoder=encoder,
        kg=kg,
        strategy=strategy,
    )


# =============================================================================
# Async Scoring with Vector Store
# =============================================================================


async def async_semantic_survival_score(
    thought: str,
    context: str,
    position: int = 0,
    *,
    vector_store: AsyncVectorStore | None = None,
    kg: KnowledgeGraph | None = None,
    strategy: str | None = None,
    session_id: str | None = None,
) -> float:
    """Calculate semantic survival probability using async vector store.

    Async version of semantic_survival_score that uses AsyncVectorStore
    for embedding-based similarity instead of the sync ContextEncoder.

    Args:
        thought: Candidate reasoning step.
        context: Problem context and previous steps.
        position: Current step/column number.
        vector_store: Optional AsyncVectorStore for semantic similarity.
        kg: Optional KnowledgeGraph for fact alignment.
        strategy: Optional reasoning strategy (for MoT).
        session_id: Optional session ID for scoped search.

    Returns:
        Score between 0.0 and 1.0 (higher = more likely to succeed).

    """
    # Handle empty input
    if not thought or not thought.strip():
        return 0.0
    if not context or not context.strip():
        context = ""

    # 1. Semantic similarity via vector store
    if vector_store is not None:
        try:
            # Search for similar thoughts in vector store
            results = await vector_store.search(thought, k=3)
            if results:
                # Use average similarity of top results
                avg_score = sum(r.score for r in results) / len(results)
                # Normalize from cosine similarity [0, 1] to score
                semantic_sim = max(0.0, min(1.0, avg_score))
            else:
                semantic_sim = _word_overlap_score(thought, context)
        except Exception:
            # Fallback on any error
            semantic_sim = _word_overlap_score(thought, context)
    else:
        semantic_sim = _word_overlap_score(thought, context)

    # 2. Knowledge graph alignment
    kg_score = _kg_alignment_score(thought, kg)

    # 3. Specificity
    specificity = _specificity_score(thought)

    # 4. Structure
    structure = _structure_score(thought, strategy)

    # 5. Length
    length = _length_score(thought)

    # 6. Position
    pos_score = _position_score(position)

    # Weighted combination
    final_score = (
        WEIGHTS["semantic_similarity"] * semantic_sim
        + WEIGHTS["kg_alignment"] * kg_score
        + WEIGHTS["specificity"] * specificity
        + WEIGHTS["structure"] * structure
        + WEIGHTS["length"] * length
        + WEIGHTS["position"] * pos_score
    )

    return max(0.0, min(1.0, final_score))


async def async_calculate_survival_score(
    thought: str,
    context: str,
    step_number: int,
    *,
    vector_store: AsyncVectorStore | None = None,
    kg: KnowledgeGraph | None = None,
) -> float:
    """Async wrapper for long-chain reasoning backward compatibility.

    Args:
        thought: Candidate reasoning step.
        context: Problem context and previous steps.
        step_number: Current position in the chain (1-indexed).
        vector_store: Optional AsyncVectorStore for semantic similarity.
        kg: Optional KnowledgeGraph for fact alignment.

    Returns:
        Score between 0.0 and 1.0.

    """
    return await async_semantic_survival_score(
        thought=thought,
        context=context,
        position=step_number,
        vector_store=vector_store,
        kg=kg,
        strategy=None,
    )
