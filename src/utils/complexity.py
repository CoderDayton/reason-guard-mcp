"""Complexity detection for adaptive reasoning strategies.

Analyzes question/context to determine optimal matrix dimensions.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

# LRU cache for complexity results
_complexity_cache: OrderedDict[int, ComplexityResult] = OrderedDict()
_COMPLEXITY_CACHE_MAX_SIZE = 100

ComplexityLevel = Literal["low", "medium", "medium-high", "high"]


@dataclass(frozen=True, slots=True)
class ComplexityResult:
    """Result of complexity analysis for a problem."""

    complexity_score: float
    complexity_level: ComplexityLevel
    recommended_rows: int
    recommended_cols: int
    signals: tuple[str, ...]
    word_count: int
    cached: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "complexity_score": self.complexity_score,
            "complexity_level": self.complexity_level,
            "recommended_rows": self.recommended_rows,
            "recommended_cols": self.recommended_cols,
            "signals": list(self.signals),
            "word_count": self.word_count,
            "cached": self.cached,
        }

    @classmethod
    def merge(cls, *results: ComplexityResult) -> ComplexityResult:
        """Merge multiple complexity results into aggregate analysis.

        Useful for multi-document reasoning where you want combined complexity.
        Takes the maximum recommended dimensions and unions all signals.

        Args:
            *results: ComplexityResult instances to merge.

        Returns:
            New ComplexityResult with aggregated values.

        Raises:
            ValueError: If no results provided.

        Example:
            >>> r1 = detect_complexity("What is 2+2?")
            >>> r2 = detect_complexity("Explain quantum entanglement")
            >>> merged = ComplexityResult.merge(r1, r2)
            >>> merged.recommended_rows  # max of both

        """
        if not results:
            raise ValueError("At least one ComplexityResult required for merge")

        if len(results) == 1:
            return results[0]

        # Aggregate: max score, max dimensions, union signals, sum word count
        max_score = max(r.complexity_score for r in results)
        max_rows = max(r.recommended_rows for r in results)
        max_cols = max(r.recommended_cols for r in results)
        total_words = sum(r.word_count for r in results)

        # Union all signals (deduplicated, sorted for consistency)
        all_signals: set[str] = set()
        for r in results:
            all_signals.update(r.signals)

        # Determine level from max score
        if max_score >= 0.6:
            level: ComplexityLevel = "high"
        elif max_score >= 0.4:
            level = "medium-high"
        elif max_score >= 0.2:
            level = "medium"
        else:
            level = "low"

        return cls(
            complexity_score=round(max_score, 3),
            complexity_level=level,
            recommended_rows=max_rows,
            recommended_cols=max_cols,
            signals=tuple(sorted(all_signals)),
            word_count=total_words,
            cached=False,  # Merged results are never cached
        )


def _get_cache_key(question: str, context: str) -> int:
    """Generate cache key from question and context."""
    return hash((question, context))


def clear_complexity_cache() -> int:
    """Clear the complexity cache. Returns number of items cleared."""
    count = len(_complexity_cache)
    _complexity_cache.clear()
    return count


def get_cache_stats() -> dict[str, int]:
    """Get complexity cache statistics."""
    return {
        "size": len(_complexity_cache),
        "max_size": _COMPLEXITY_CACHE_MAX_SIZE,
    }


def detect_complexity(
    question: str,
    context: str = "",
    *,
    use_cache: bool = True,
) -> ComplexityResult:
    """Detect problem complexity to determine optimal matrix dimensions.

    Analyzes question and context for complexity indicators:
    - Length and structure
    - Multi-hop reasoning signals
    - Mathematical/logical complexity
    - Entity tracking requirements

    Args:
        question: The question to analyze.
        context: Optional context.
        use_cache: Whether to use cached results (default True).

    Returns:
        ComplexityResult with score, level, and recommendations.

    Example:
        >>> result = detect_complexity("What is 2+2?")
        >>> result.complexity_level
        'low'
        >>> result.recommended_rows
        3

    """
    cache_key = _get_cache_key(question, context)

    # Check cache first (LRU: move to end on access)
    if use_cache and cache_key in _complexity_cache:
        _complexity_cache.move_to_end(cache_key)
        cached = _complexity_cache[cache_key]
        # Return cached result with cached=True flag
        return ComplexityResult(
            complexity_score=cached.complexity_score,
            complexity_level=cached.complexity_level,
            recommended_rows=cached.recommended_rows,
            recommended_cols=cached.recommended_cols,
            signals=cached.signals,
            word_count=cached.word_count,
            cached=True,
        )

    text = (question + " " + context).lower()
    word_count = len(text.split())

    score = 0.0
    signals: list[str] = []

    # 1. Length-based complexity
    if word_count > 200:
        score += 0.3
        signals.append("long_context")
    elif word_count > 100:
        score += 0.15
        signals.append("medium_context")

    # 2. Multi-hop indicators (requires chaining facts)
    multi_hop_phrases = [
        "therefore",
        "because",
        "since",
        "implies",
        "if.*then",
        "based on",
        "according to",
        "given that",
        "assuming",
        "first.*then",
        "after.*before",
        "leads to",
        "results in",
    ]
    multi_hop_count = sum(1 for p in multi_hop_phrases if re.search(p, text))
    if multi_hop_count >= 3:
        score += 0.3
        signals.append("multi_hop")
    elif multi_hop_count >= 1:
        score += 0.15
        signals.append("some_chaining")

    # 3. Mathematical complexity
    math_indicators = [
        r"\d+\s*[\+\-\*\/\=]\s*\d+",  # arithmetic
        r"equation|formula|calculate|solve|sum|product",
        r"percent|ratio|fraction|proportion",
        r"x\s*=|y\s*=|n\s*=",  # variables
    ]
    math_count = sum(1 for p in math_indicators if re.search(p, text))
    if math_count >= 2:
        score += 0.25
        signals.append("mathematical")

    # 4. Logical complexity
    logic_indicators = [
        r"\ball\b.*\bare\b|\bsome\b.*\bare\b",  # quantifiers
        r"\bif\b.*\bthen\b|\bonly if\b",  # conditionals
        r"\bnot\b.*\band\b|\bor\b.*\bnot\b",  # negation
        r"\bmust\b|\bcannot\b|\bimpossible\b",  # modals
        r"contradict|paradox|dilemma",
    ]
    logic_count = sum(1 for p in logic_indicators if re.search(p, text))
    if logic_count >= 2:
        score += 0.25
        signals.append("logical")

    # 5. Multiple entities/factors to track
    entity_markers = [
        r"\b(alice|bob|charlie|person|company|city)\b",
        r"\b(factor|aspect|dimension|criterion|variable)\b",
        r"\b(compare|contrast|versus|vs\.?|differ)\b",
    ]
    entity_count = sum(1 for p in entity_markers if re.search(p, text))
    if entity_count >= 2:
        score += 0.2
        signals.append("multi_entity")

    # 6. Question complexity markers
    complex_questions = [
        r"what.*and.*why",
        r"how.*affect.*",
        r"explain.*relationship",
        r"analyze|evaluate|assess|critique",
        r"pros.*cons|advantages.*disadvantages",
    ]
    if any(re.search(p, text) for p in complex_questions):
        score += 0.2
        signals.append("complex_question")

    # Determine recommended rows and cols based on score
    score = min(1.0, score)
    if score >= 0.6:
        recommended_rows = 5  # Max strategies for complex problems
        recommended_cols = 5  # More refinement iterations
        complexity_level: ComplexityLevel = "high"
    elif score >= 0.4:
        recommended_rows = 4
        recommended_cols = 4
        complexity_level = "medium-high"
    elif score >= 0.2:
        recommended_rows = 3  # Default for most problems
        recommended_cols = 3
        complexity_level = "medium"
    else:
        recommended_rows = 3  # Minimum 3 rows for adequate coverage
        recommended_cols = 2  # Fewer iterations for simple problems
        complexity_level = "low"

    result = ComplexityResult(
        complexity_score=round(score, 3),
        complexity_level=complexity_level,
        recommended_rows=recommended_rows,
        recommended_cols=recommended_cols,
        signals=tuple(signals),
        word_count=word_count,
        cached=False,
    )

    # Cache the result (LRU: evict least recently used from front)
    if use_cache:
        if len(_complexity_cache) >= _COMPLEXITY_CACHE_MAX_SIZE:
            _complexity_cache.popitem(last=False)  # Remove oldest (front)
        _complexity_cache[cache_key] = result

    return result
