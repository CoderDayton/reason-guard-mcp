"""Tests for CISC confidence scoring module."""

from __future__ import annotations

import pytest

from src.utils.confidence import (
    DEFAULT_TEMPERATURES,
    CalibrationTracker,
    CISCSelectionResult,
    ConfidenceMethod,
    ConfidenceResult,
    cisc_select,
    combine_confidences,
    get_calibration_tracker,
    softmax_normalize,
    within_question_discrimination,
)


class TestSoftmaxNormalize:
    """Tests for softmax normalization function."""

    def test_basic_normalization(self) -> None:
        """Test that softmax produces valid probabilities."""
        scores = [0.8, 0.6, 0.4]
        result = softmax_normalize(scores, temperature=1.0)

        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-6  # Sums to 1
        assert all(0 <= p <= 1 for p in result)  # Valid probabilities
        assert result[0] > result[1] > result[2]  # Order preserved

    def test_low_temperature_sharpens(self) -> None:
        """Test that low temperature creates sharper distribution."""
        scores = [0.8, 0.6, 0.4]

        soft = softmax_normalize(scores, temperature=2.0)
        sharp = softmax_normalize(scores, temperature=0.1)

        # Sharp should have winner take more
        assert sharp[0] > soft[0]
        # Sharp should have losers take less
        assert sharp[2] < soft[2]

    def test_high_temperature_flattens(self) -> None:
        """Test that high temperature creates flatter distribution."""
        scores = [0.9, 0.1]

        normal = softmax_normalize(scores, temperature=1.0)
        flat = softmax_normalize(scores, temperature=10.0)

        # Flat distribution should be closer to 50/50
        assert abs(flat[0] - flat[1]) < abs(normal[0] - normal[1])

    def test_empty_input(self) -> None:
        """Test empty input returns empty list."""
        assert softmax_normalize([]) == []

    def test_single_score(self) -> None:
        """Test single score returns [1.0]."""
        result = softmax_normalize([0.5])
        assert result == [1.0]

    def test_identical_scores(self) -> None:
        """Test identical scores produce uniform distribution."""
        result = softmax_normalize([0.5, 0.5, 0.5])
        expected = 1.0 / 3

        for p in result:
            assert abs(p - expected) < 1e-6

    def test_numerical_stability_large_scores(self) -> None:
        """Test numerical stability with large score differences."""
        scores = [100.0, 1.0, 0.0]
        result = softmax_normalize(scores, temperature=1.0)

        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-6
        assert result[0] > 0.99  # First should dominate

    def test_numerical_stability_negative_scores(self) -> None:
        """Test handling of negative scores."""
        scores = [-0.5, -0.3, -0.1]
        result = softmax_normalize(scores, temperature=1.0)

        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-6
        assert result[2] > result[1] > result[0]  # Least negative wins


class TestWithinQuestionDiscrimination:
    """Tests for WQD metric calculation."""

    def test_perfect_discrimination(self) -> None:
        """Test WQD=1.0 when all correct > all incorrect."""
        observations = [
            (0.9, True),
            (0.8, True),
            (0.3, False),
            (0.2, False),
        ]
        wqd = within_question_discrimination(observations)
        assert wqd == 1.0

    def test_no_discrimination(self) -> None:
        """Test WQD=0.5 when scores are inverted."""
        observations = [
            (0.2, True),
            (0.3, True),
            (0.8, False),
            (0.9, False),
        ]
        wqd = within_question_discrimination(observations)
        assert wqd == 0.0

    def test_random_discrimination(self) -> None:
        """Test WQD~0.5 when scores overlap."""
        observations = [
            (0.5, True),
            (0.5, False),
        ]
        wqd = within_question_discrimination(observations)
        assert wqd == 0.5  # Tie

    def test_partial_discrimination(self) -> None:
        """Test WQD between 0 and 1 for mixed results."""
        observations = [
            (0.9, True),  # Correct, high
            (0.4, True),  # Correct, low
            (0.6, False),  # Wrong, medium
            (0.3, False),  # Wrong, low
        ]
        # 0.9 > 0.6 (win), 0.9 > 0.3 (win)
        # 0.4 < 0.6 (loss), 0.4 > 0.3 (win)
        # 3 wins out of 4 comparisons = 0.75
        wqd = within_question_discrimination(observations)
        assert wqd == 0.75

    def test_no_correct_answers(self) -> None:
        """Test WQD=0.5 when no correct answers."""
        observations = [
            (0.8, False),
            (0.6, False),
        ]
        wqd = within_question_discrimination(observations)
        assert wqd == 0.5

    def test_no_incorrect_answers(self) -> None:
        """Test WQD=0.5 when no incorrect answers."""
        observations = [
            (0.8, True),
            (0.6, True),
        ]
        wqd = within_question_discrimination(observations)
        assert wqd == 0.5


class TestCISCSelect:
    """Tests for CISC-style candidate selection."""

    def test_selects_highest_score(self) -> None:
        """Test that selection returns highest-scored candidate."""
        candidates = [
            ("Answer A", 0.6),
            ("Answer B", 0.9),
            ("Answer C", 0.3),
        ]
        result = cisc_select(candidates)

        assert result.selected_index == 1
        assert result.selected_content == "Answer B"

    def test_returns_normalized_weights(self) -> None:
        """Test that normalized weights sum to 1."""
        candidates = [
            ("A", 0.5),
            ("B", 0.7),
            ("C", 0.3),
        ]
        result = cisc_select(candidates)

        assert abs(sum(result.normalized_weights) - 1.0) < 1e-6

    def test_respects_temperature(self) -> None:
        """Test that temperature affects weight distribution."""
        candidates = [
            ("A", 0.8),
            ("B", 0.6),
        ]

        sharp = cisc_select(candidates, temperature=0.1)
        soft = cisc_select(candidates, temperature=2.0)

        # Sharp should have higher max weight
        assert max(sharp.normalized_weights) > max(soft.normalized_weights)

    def test_method_specific_temperature(self) -> None:
        """Test that method determines default temperature."""
        candidates = [("A", 0.7), ("B", 0.5)]

        # P_TRUE uses very low temperature by default
        p_true_result = cisc_select(candidates, method=ConfidenceMethod.P_TRUE)
        # SURVIVAL_HEURISTIC uses higher temperature
        heuristic_result = cisc_select(candidates, method=ConfidenceMethod.SURVIVAL_HEURISTIC)

        assert p_true_result.temperature == DEFAULT_TEMPERATURES[ConfidenceMethod.P_TRUE]
        assert (
            heuristic_result.temperature
            == DEFAULT_TEMPERATURES[ConfidenceMethod.SURVIVAL_HEURISTIC]
        )

    def test_empty_candidates_raises(self) -> None:
        """Test that empty candidates list raises error."""
        with pytest.raises(ValueError, match="empty candidates"):
            cisc_select([])

    def test_single_candidate(self) -> None:
        """Test selection with single candidate."""
        result = cisc_select([("Only option", 0.5)])

        assert result.selected_index == 0
        assert result.selected_content == "Only option"
        assert result.normalized_weights == [1.0]


class TestCombineConfidences:
    """Tests for hybrid confidence combination."""

    def test_default_weights(self) -> None:
        """Test default 0.7/0.3 weighting."""
        combined = combine_confidences(llm_confidence=1.0, heuristic_score=0.0)
        assert abs(combined - 0.7) < 1e-9  # 1.0 * 0.7 + 0.0 * 0.3

        combined = combine_confidences(llm_confidence=0.0, heuristic_score=1.0)
        assert abs(combined - 0.3) < 1e-9  # 0.0 * 0.7 + 1.0 * 0.3

    def test_custom_weights(self) -> None:
        """Test custom weight distribution."""
        combined = combine_confidences(
            llm_confidence=1.0,
            heuristic_score=0.0,
            llm_weight=0.5,
        )
        assert combined == 0.5  # 1.0 * 0.5 + 0.0 * 0.5

    def test_equal_scores(self) -> None:
        """Test that equal scores produce equal result regardless of weights."""
        combined = combine_confidences(
            llm_confidence=0.6,
            heuristic_score=0.6,
            llm_weight=0.9,
        )
        assert combined == 0.6


class TestCalibrationTracker:
    """Tests for calibration tracking and analysis."""

    def test_add_and_retrieve_observations(self) -> None:
        """Test adding and retrieving observations."""
        tracker = CalibrationTracker()
        tracker.add(ConfidenceMethod.VERBAL_0_100, 0.8, True)
        tracker.add(ConfidenceMethod.VERBAL_0_100, 0.6, False)

        stats = tracker.get_stats(ConfidenceMethod.VERBAL_0_100)
        assert stats["observations"] == 2
        assert stats["correct_count"] == 1
        assert stats["incorrect_count"] == 1

    def test_wqd_calculation(self) -> None:
        """Test WQD calculation through tracker."""
        tracker = CalibrationTracker()
        # Perfect discrimination
        tracker.add(ConfidenceMethod.VERBAL_BINARY, 0.9, True)
        tracker.add(ConfidenceMethod.VERBAL_BINARY, 0.3, False)

        wqd = tracker.wqd_score(ConfidenceMethod.VERBAL_BINARY)
        assert wqd == 1.0

    def test_optimal_temperature_insufficient_data(self) -> None:
        """Test optimal temperature returns default with insufficient data."""
        tracker = CalibrationTracker()
        tracker.add(ConfidenceMethod.P_TRUE, 0.8, True)

        opt_t = tracker.find_optimal_temperature(ConfidenceMethod.P_TRUE)
        assert opt_t == DEFAULT_TEMPERATURES[ConfidenceMethod.P_TRUE]

    def test_reset_single_method(self) -> None:
        """Test resetting single method's observations."""
        tracker = CalibrationTracker()
        tracker.add(ConfidenceMethod.VERBAL_0_100, 0.8, True)
        tracker.add(ConfidenceMethod.VERBAL_BINARY, 0.9, True)

        tracker.reset(ConfidenceMethod.VERBAL_0_100)

        assert tracker.get_stats(ConfidenceMethod.VERBAL_0_100)["observations"] == 0
        assert tracker.get_stats(ConfidenceMethod.VERBAL_BINARY)["observations"] == 1

    def test_reset_all(self) -> None:
        """Test resetting all observations."""
        tracker = CalibrationTracker()
        tracker.add(ConfidenceMethod.VERBAL_0_100, 0.8, True)
        tracker.add(ConfidenceMethod.VERBAL_BINARY, 0.9, True)

        tracker.reset()

        all_stats = tracker.get_all_stats()
        assert all_stats["total_observations"] == 0

    def test_get_all_stats_sorted_by_wqd(self) -> None:
        """Test that get_all_stats sorts by WQD descending."""
        tracker = CalibrationTracker()

        # Method A: poor discrimination
        tracker.add(ConfidenceMethod.VERBAL_BINARY, 0.3, True)
        tracker.add(ConfidenceMethod.VERBAL_BINARY, 0.8, False)

        # Method B: good discrimination
        tracker.add(ConfidenceMethod.VERBAL_0_100, 0.9, True)
        tracker.add(ConfidenceMethod.VERBAL_0_100, 0.2, False)

        all_stats = tracker.get_all_stats()
        methods = all_stats["methods"]

        # VERBAL_0_100 should be first (better WQD)
        assert methods[0]["method"] == "verbal_0_100"
        assert methods[1]["method"] == "verbal_binary"


class TestGlobalCalibrationTracker:
    """Tests for global calibration tracker singleton."""

    def test_get_calibration_tracker_returns_singleton(self) -> None:
        """Test that get_calibration_tracker returns same instance."""
        tracker1 = get_calibration_tracker()
        tracker2 = get_calibration_tracker()

        assert tracker1 is tracker2


class TestConfidenceResult:
    """Tests for ConfidenceResult dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = ConfidenceResult(
            score=0.85,
            method=ConfidenceMethod.HYBRID,
            raw_score=0.82,
        )
        d = result.to_dict()

        assert d["confidence"] == 0.85
        assert d["method"] == "hybrid"
        assert d["raw"] == 0.82


class TestCISCSelectionResult:
    """Tests for CISCSelectionResult dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = CISCSelectionResult(
            selected_index=1,
            selected_content="Best answer",
            selected_confidence=0.75,
            normalized_weights=[0.25, 0.75],
            candidates=[("A", 0.4), ("B", 0.8)],
            temperature=0.5,
            method=ConfidenceMethod.HYBRID,
        )
        d = result.to_dict()

        assert d["selected_index"] == 1
        assert d["selected_confidence"] == 0.75
        assert d["normalized_weights"] == [0.25, 0.75]
        assert d["temperature"] == 0.5
        assert d["method"] == "hybrid"
        assert d["candidates_evaluated"] == 2
