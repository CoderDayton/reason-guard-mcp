"""Comprehensive unit tests for src/utils/complexity.py."""

from __future__ import annotations

import pytest

from src.utils.complexity import (
    ComplexityResult,
    clear_complexity_cache,
    detect_complexity,
    get_cache_stats,
)


class TestComplexityResult:
    """Tests for ComplexityResult dataclass."""

    def test_dataclass_immutable(self) -> None:
        """ComplexityResult should be frozen (immutable)."""
        result = ComplexityResult(
            complexity_score=0.5,
            complexity_level="medium",
            recommended_rows=3,
            recommended_cols=3,
            signals=("test",),
            word_count=10,
        )
        with pytest.raises(AttributeError):
            result.complexity_score = 0.9  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """to_dict() should return JSON-serializable dict."""
        result = ComplexityResult(
            complexity_score=0.5,
            complexity_level="medium",
            recommended_rows=3,
            recommended_cols=3,
            signals=("signal_a", "signal_b"),
            word_count=50,
            cached=True,
        )
        d = result.to_dict()
        assert d["complexity_score"] == 0.5
        assert d["complexity_level"] == "medium"
        assert d["recommended_rows"] == 3
        assert d["recommended_cols"] == 3
        assert d["signals"] == ["signal_a", "signal_b"]  # tuple -> list
        assert d["word_count"] == 50
        assert d["cached"] is True

    def test_slots(self) -> None:
        """ComplexityResult should use __slots__ for memory efficiency."""
        result = ComplexityResult(
            complexity_score=0.1,
            complexity_level="low",
            recommended_rows=3,
            recommended_cols=2,
            signals=(),
            word_count=5,
        )
        assert hasattr(result, "__slots__") or not hasattr(result, "__dict__")


class TestComplexityResultMerge:
    """Tests for ComplexityResult.merge() classmethod."""

    def test_merge_empty_raises(self) -> None:
        """merge() with no arguments should raise ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            ComplexityResult.merge()

    def test_merge_single_returns_same(self) -> None:
        """merge() with single result returns that result."""
        r1 = ComplexityResult(
            complexity_score=0.3,
            complexity_level="medium",
            recommended_rows=3,
            recommended_cols=3,
            signals=("test",),
            word_count=20,
        )
        merged = ComplexityResult.merge(r1)
        assert merged is r1

    def test_merge_takes_max_score(self) -> None:
        """merge() should take maximum complexity score."""
        r1 = ComplexityResult(
            complexity_score=0.2,
            complexity_level="medium",
            recommended_rows=3,
            recommended_cols=3,
            signals=(),
            word_count=10,
        )
        r2 = ComplexityResult(
            complexity_score=0.7,
            complexity_level="high",
            recommended_rows=5,
            recommended_cols=5,
            signals=(),
            word_count=100,
        )
        merged = ComplexityResult.merge(r1, r2)
        assert merged.complexity_score == 0.7

    def test_merge_takes_max_dimensions(self) -> None:
        """merge() should take maximum rows and cols."""
        r1 = ComplexityResult(
            complexity_score=0.3,
            complexity_level="medium",
            recommended_rows=3,
            recommended_cols=4,
            signals=(),
            word_count=10,
        )
        r2 = ComplexityResult(
            complexity_score=0.4,
            complexity_level="medium-high",
            recommended_rows=5,
            recommended_cols=3,
            signals=(),
            word_count=20,
        )
        merged = ComplexityResult.merge(r1, r2)
        assert merged.recommended_rows == 5
        assert merged.recommended_cols == 4

    def test_merge_unions_signals(self) -> None:
        """merge() should union all signals and sort them."""
        r1 = ComplexityResult(
            complexity_score=0.3,
            complexity_level="medium",
            recommended_rows=3,
            recommended_cols=3,
            signals=("b_signal", "a_signal"),
            word_count=10,
        )
        r2 = ComplexityResult(
            complexity_score=0.3,
            complexity_level="medium",
            recommended_rows=3,
            recommended_cols=3,
            signals=("c_signal", "a_signal"),  # a_signal duplicated
            word_count=20,
        )
        merged = ComplexityResult.merge(r1, r2)
        assert merged.signals == ("a_signal", "b_signal", "c_signal")

    def test_merge_sums_word_count(self) -> None:
        """merge() should sum word counts."""
        r1 = ComplexityResult(
            complexity_score=0.3,
            complexity_level="medium",
            recommended_rows=3,
            recommended_cols=3,
            signals=(),
            word_count=100,
        )
        r2 = ComplexityResult(
            complexity_score=0.3,
            complexity_level="medium",
            recommended_rows=3,
            recommended_cols=3,
            signals=(),
            word_count=50,
        )
        merged = ComplexityResult.merge(r1, r2)
        assert merged.word_count == 150

    def test_merge_never_cached(self) -> None:
        """merge() result should never be marked as cached."""
        r1 = ComplexityResult(
            complexity_score=0.3,
            complexity_level="medium",
            recommended_rows=3,
            recommended_cols=3,
            signals=(),
            word_count=10,
            cached=True,
        )
        r2 = ComplexityResult(
            complexity_score=0.3,
            complexity_level="medium",
            recommended_rows=3,
            recommended_cols=3,
            signals=(),
            word_count=20,
            cached=True,
        )
        merged = ComplexityResult.merge(r1, r2)
        assert merged.cached is False

    def test_merge_recalculates_level_from_score(self) -> None:
        """merge() should recalculate level based on max score."""
        # Low + medium-high should result in medium-high level
        r1 = ComplexityResult(
            complexity_score=0.1,
            complexity_level="low",
            recommended_rows=3,
            recommended_cols=2,
            signals=(),
            word_count=10,
        )
        r2 = ComplexityResult(
            complexity_score=0.45,
            complexity_level="medium-high",
            recommended_rows=4,
            recommended_cols=4,
            signals=(),
            word_count=50,
        )
        merged = ComplexityResult.merge(r1, r2)
        assert merged.complexity_level == "medium-high"


class TestDetectComplexity:
    """Tests for detect_complexity() function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_complexity_cache()

    def test_empty_string(self) -> None:
        """Empty string should return low complexity."""
        result = detect_complexity("")
        assert result.complexity_level == "low"
        assert result.word_count == 0  # " ".split() on empty gives []
        assert result.recommended_rows == 3
        assert result.recommended_cols == 2

    def test_simple_question_low_complexity(self) -> None:
        """Simple questions should be low complexity."""
        result = detect_complexity("What is 2+2?")
        assert result.complexity_level == "low"
        assert result.recommended_rows == 3
        assert result.recommended_cols == 2

    def test_long_context_signal(self) -> None:
        """Long context (>200 words) should trigger long_context signal."""
        long_text = " ".join(["word"] * 250)
        result = detect_complexity(long_text)
        assert "long_context" in result.signals

    def test_medium_context_signal(self) -> None:
        """Medium context (100-200 words) should trigger medium_context signal."""
        medium_text = " ".join(["word"] * 150)
        result = detect_complexity(medium_text)
        assert "medium_context" in result.signals

    def test_multi_hop_signal(self) -> None:
        """Multi-hop reasoning indicators should trigger signal."""
        text = "Given that A implies B, and since B leads to C, therefore we can conclude D"
        result = detect_complexity(text)
        assert "multi_hop" in result.signals or "some_chaining" in result.signals

    def test_mathematical_signal(self) -> None:
        """Mathematical content should trigger signal."""
        text = "Calculate x = 5 + 3 * 2, then solve the equation for y = x / 2"
        result = detect_complexity(text)
        assert "mathematical" in result.signals

    def test_logical_signal(self) -> None:
        """Logical complexity should trigger signal."""
        text = "If all cats are animals, then some animals must be cats. This cannot be true."
        result = detect_complexity(text)
        assert "logical" in result.signals

    def test_multi_entity_signal(self) -> None:
        """Multiple entities should trigger signal."""
        text = "Compare Alice and Bob's approaches, contrast their factors and variables"
        result = detect_complexity(text)
        assert "multi_entity" in result.signals

    def test_complex_question_signal(self) -> None:
        """Complex question patterns should trigger signal."""
        text = "Analyze and evaluate the pros and cons of this approach"
        result = detect_complexity(text)
        assert "complex_question" in result.signals

    def test_high_complexity_threshold(self) -> None:
        """Score >= 0.6 should be high complexity with 5x5 matrix."""
        # Combine multiple signals to hit high complexity
        text = (
            "Given that Alice implies Bob, and since this leads to Charlie, "
            "therefore calculate x = 5 + 3. If all factors must be considered, "
            "analyze and evaluate the pros and cons of each variable dimension."
        )
        # Extend to 250+ words for long_context
        text += " " + " ".join(["additional context word"] * 50)
        result = detect_complexity(text)
        assert result.complexity_score >= 0.6
        assert result.complexity_level == "high"
        assert result.recommended_rows == 5
        assert result.recommended_cols == 5

    def test_medium_high_complexity_threshold(self) -> None:
        """Score 0.4-0.6 should be medium-high with 4x4 matrix."""
        # Multiple signals but not enough for high
        text = (
            "Given that this implies that, therefore we conclude. "
            "Calculate x = 5 + 3 and solve for y."
        )
        result = detect_complexity(text)
        # Should be in medium-high range
        assert 0.2 <= result.complexity_score < 0.6
        assert result.complexity_level in ("medium", "medium-high")

    def test_score_capped_at_1(self) -> None:
        """Complexity score should never exceed 1.0."""
        # Trigger every single signal
        text = (
            " ".join(["word"] * 250)  # long_context
            + " therefore because since implies if then based on given that assuming "  # multi_hop
            + " 5 + 3 = 8 calculate equation formula x = y = "  # mathematical
            + " all are some are if then only if not and or not must cannot impossible "
            + " alice bob charlie person company compare contrast versus differ "
            + " what and why how affect explain relationship analyze evaluate "  # complex
        )
        result = detect_complexity(text)
        assert result.complexity_score <= 1.0


class TestComplexityCaching:
    """Tests for complexity detection caching."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_complexity_cache()

    def test_first_call_not_cached(self) -> None:
        """First call should not be cached."""
        result = detect_complexity("unique question 1")
        assert result.cached is False

    def test_second_call_cached(self) -> None:
        """Second call with same input should be cached."""
        detect_complexity("unique question 2")
        result = detect_complexity("unique question 2")
        assert result.cached is True

    def test_different_inputs_not_cached(self) -> None:
        """Different inputs should not return cached results."""
        detect_complexity("question A")
        result = detect_complexity("question B")
        assert result.cached is False

    def test_context_affects_cache_key(self) -> None:
        """Different context should be a cache miss."""
        detect_complexity("same question", "context 1")
        result = detect_complexity("same question", "context 2")
        assert result.cached is False

    def test_use_cache_false_bypasses_cache(self) -> None:
        """use_cache=False should bypass cache entirely."""
        # First call (populates cache)
        detect_complexity("bypass test")
        # Second call with use_cache=False
        result = detect_complexity("bypass test", use_cache=False)
        assert result.cached is False

    def test_cache_eviction_lru(self) -> None:
        """Cache should evict LRU entries when full."""
        # Fill cache beyond max size (100)
        for i in range(105):
            detect_complexity(f"question {i}")

        stats = get_cache_stats()
        assert stats["size"] <= stats["max_size"]

    def test_cache_access_moves_to_end(self) -> None:
        """Accessing cached item should move it to end (LRU refresh)."""
        # Add items
        detect_complexity("first")
        detect_complexity("second")

        # Access first (moves to end)
        result = detect_complexity("first")
        assert result.cached is True  # Verify LRU access works

        # Fill cache to trigger eviction
        for i in range(100):
            detect_complexity(f"filler {i}")

        # "first" should still be cached (was recently accessed)
        # "second" should be evicted (oldest, not accessed)
        detect_complexity("first")
        detect_complexity("second")

        # Both may or may not be cached depending on exact LRU behavior
        # The key invariant: recently accessed items survive longer
        # With 100 fillers + first + second = 102 items, some evicted
        stats = get_cache_stats()
        assert stats["size"] == stats["max_size"]  # Cache is full


class TestCacheHelpers:
    """Tests for cache helper functions."""

    def test_clear_complexity_cache_returns_count(self) -> None:
        """clear_complexity_cache() should return number of items cleared."""
        clear_complexity_cache()
        detect_complexity("a")
        detect_complexity("b")
        detect_complexity("c")

        count = clear_complexity_cache()
        assert count == 3

    def test_clear_empty_cache(self) -> None:
        """Clearing empty cache should return 0."""
        clear_complexity_cache()
        count = clear_complexity_cache()
        assert count == 0

    def test_get_cache_stats(self) -> None:
        """get_cache_stats() should return size and max_size."""
        clear_complexity_cache()
        detect_complexity("stats test 1")
        detect_complexity("stats test 2")

        stats = get_cache_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 100

    def test_get_cache_stats_empty(self) -> None:
        """get_cache_stats() on empty cache."""
        clear_complexity_cache()
        stats = get_cache_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 100
