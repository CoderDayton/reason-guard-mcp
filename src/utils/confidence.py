"""Confidence-Informed Self-Consistency (CISC) primitives.

Implements confidence scoring techniques from the paper:
"Confidence Improves Self-Consistency in LLMs" (arXiv:2502.06233v2)

Key techniques:
- Softmax temperature normalization for amplifying score differences
- Within-Question Discrimination (WQD) metric for calibration
- Multiple confidence extraction methods (verbal, P(True), hybrid)

The core insight: LLMs can judge their own output quality, and using
softmax-normalized confidence scores for weighted voting reduces
computational cost while maintaining or improving accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConfidenceMethod(Enum):
    """Methods for extracting confidence scores from LLM outputs."""

    # Heuristic scoring based on text features (no LLM call)
    SURVIVAL_HEURISTIC = "survival_heuristic"

    # Ask LLM for binary 0/1 confidence (best discrimination per paper)
    VERBAL_BINARY = "verbal_binary"

    # Ask LLM for 0-100 confidence score (more granular)
    VERBAL_0_100 = "verbal_0_100"

    # Extract probability of "True"/"1" token (requires logit access)
    P_TRUE = "p_true"

    # Combine LLM confidence with heuristic (hybrid approach)
    HYBRID = "hybrid"


# Default temperatures by method (from paper experiments)
# Lower T → sharper distribution (emphasizes highest scores)
# Higher T → uniform distribution (approaches vanilla majority vote)
DEFAULT_TEMPERATURES: dict[ConfidenceMethod, float] = {
    ConfidenceMethod.P_TRUE: 0.001,  # P(True) needs very low T
    ConfidenceMethod.VERBAL_BINARY: 0.5,
    ConfidenceMethod.VERBAL_0_100: 1.0,
    ConfidenceMethod.SURVIVAL_HEURISTIC: 2.0,  # Heuristics need higher T
    ConfidenceMethod.HYBRID: 0.5,
}


@dataclass
class ConfidenceResult:
    """Result of confidence estimation for a single candidate."""

    score: float  # 0.0-1.0 normalized confidence
    method: ConfidenceMethod
    raw_score: float  # Original score before normalization

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "confidence": round(self.score, 3),
            "method": self.method.value,
            "raw": round(self.raw_score, 3),
        }


@dataclass
class CISCSelectionResult:
    """Result of CISC-style weighted selection from candidates."""

    selected_index: int
    selected_content: str
    selected_confidence: float
    normalized_weights: list[float]
    candidates: list[tuple[str, float]]  # (content, raw_score)
    temperature: float
    method: ConfidenceMethod

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "selected_index": self.selected_index,
            "selected_confidence": round(self.selected_confidence, 3),
            "normalized_weights": [round(w, 4) for w in self.normalized_weights],
            "temperature": self.temperature,
            "method": self.method.value,
            "candidates_evaluated": len(self.candidates),
        }


def softmax_normalize(
    scores: list[float],
    temperature: float = 1.0,
) -> list[float]:
    """Apply softmax normalization with temperature scaling.

    This is the key technique from CISC: temperature controls how much
    we emphasize differences between candidates.

    Args:
        scores: Raw confidence scores (any range).
        temperature: Scaling factor. Lower T → sharper distribution.
            - T=0.001: Winner-take-all (best for P(True))
            - T=1.0: Standard softmax
            - T=10.0: Near-uniform (best for sequence probability)

    Returns:
        Normalized probabilities summing to 1.0.

    Example:
        >>> softmax_normalize([0.8, 0.6, 0.5], temperature=1.0)
        [0.42, 0.33, 0.25]  # Proportional to exp(score)

        >>> softmax_normalize([0.8, 0.6, 0.5], temperature=0.1)
        [0.88, 0.10, 0.02]  # Winner emphasized

    """
    if not scores:
        return []

    if temperature <= 0:
        temperature = 1e-10  # Avoid division by zero

    # Scale scores by temperature
    scaled = [s / temperature for s in scores]

    # Numerical stability: subtract max before exp
    max_scaled = max(scaled)
    exp_scaled = [math.exp(s - max_scaled) for s in scaled]

    # Normalize to sum to 1
    total = sum(exp_scaled)
    if total == 0:
        # All scores were -inf or numerical issues
        return [1.0 / len(scores)] * len(scores)

    return [e / total for e in exp_scaled]


def within_question_discrimination(
    observations: list[tuple[float, bool]],
) -> float:
    """Calculate Within-Question Discrimination (WQD) score.

    WQD measures a model's ability to distinguish correct from incorrect
    answers *to the same question*. This is more predictive of CISC
    effectiveness than traditional calibration metrics like ECE.

    Args:
        observations: List of (confidence_score, is_correct) pairs
            for candidates answering the SAME question.

    Returns:
        WQD score between 0.0 and 1.0:
        - 1.0 = Perfect discrimination (all correct > all incorrect)
        - 0.5 = Random (no discrimination ability)
        - 0.0 = Inverse (incorrect consistently higher than correct)

    Example:
        >>> wqd = within_question_discrimination([
        ...     (0.9, True),   # Correct answer, high confidence
        ...     (0.7, True),   # Correct answer, medium confidence
        ...     (0.4, False),  # Wrong answer, low confidence
        ...     (0.3, False),  # Wrong answer, low confidence
        ... ])
        >>> wqd  # Should be 1.0 (all correct > all incorrect)

    """
    correct_scores = [score for score, is_correct in observations if is_correct]
    incorrect_scores = [score for score, is_correct in observations if not is_correct]

    # Need both correct and incorrect to compute discrimination
    if not correct_scores or not incorrect_scores:
        return 0.5  # Cannot compute, return neutral

    # Count wins: how often correct answer has higher confidence than incorrect
    wins = 0.0
    total = 0

    for c_score in correct_scores:
        for i_score in incorrect_scores:
            total += 1
            if c_score > i_score:
                wins += 1.0
            elif c_score == i_score:
                wins += 0.5  # Tie counts as half win

    return wins / total if total > 0 else 0.5


def cisc_select(
    candidates: list[tuple[str, float]],
    method: ConfidenceMethod = ConfidenceMethod.SURVIVAL_HEURISTIC,
    temperature: float | None = None,
) -> CISCSelectionResult:
    """Select best candidate using CISC weighted approach.

    Instead of simple argmax, applies softmax normalization to create
    a weighted distribution, then selects the highest-weighted candidate.

    For standard selection this is equivalent to argmax, but the weights
    are useful for:
    - Aggregating across multiple questions (weighted voting)
    - Detecting low-confidence selections (flat distribution)
    - Calibration analysis

    Args:
        candidates: List of (content, confidence_score) pairs.
        method: Confidence extraction method used.
        temperature: Override default temperature for the method.

    Returns:
        CISCSelectionResult with selected candidate and weights.

    """
    if not candidates:
        raise ValueError("Cannot select from empty candidates list")

    # Use method-specific temperature if not overridden
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURES.get(method, 1.0)

    # Extract scores and normalize
    scores = [score for _, score in candidates]
    normalized = softmax_normalize(scores, temperature=temp)

    # Select highest weight (equivalent to argmax but with weights available)
    best_idx = normalized.index(max(normalized))
    best_content, best_raw_score = candidates[best_idx]

    return CISCSelectionResult(
        selected_index=best_idx,
        selected_content=best_content,
        selected_confidence=normalized[best_idx],
        normalized_weights=normalized,
        candidates=candidates,
        temperature=temp,
        method=method,
    )


def combine_confidences(
    llm_confidence: float,
    heuristic_score: float,
    llm_weight: float = 0.7,
) -> float:
    """Combine LLM confidence with heuristic score (hybrid approach).

    Paper findings suggest LLM confidence (especially P(True)) has better
    discrimination than heuristics, so we weight it higher by default.

    Args:
        llm_confidence: Confidence from LLM self-assessment (0-1).
        heuristic_score: Score from survival/structure heuristics (0-1).
        llm_weight: Weight for LLM confidence (default 0.7 per paper).

    Returns:
        Combined confidence score (0-1).

    """
    heuristic_weight = 1.0 - llm_weight
    return llm_confidence * llm_weight + heuristic_score * heuristic_weight


@dataclass
class CalibrationTracker:
    """Tracks confidence vs correctness for calibration analysis.

    Accumulates observations across reasoning sessions to:
    - Compute WQD scores per confidence method
    - Find optimal temperature through grid search
    - Detect poorly calibrated methods

    Usage:
        tracker = CalibrationTracker()

        # During reasoning
        tracker.add(method=ConfidenceMethod.VERBAL_0_100, confidence=0.8, is_correct=True)
        tracker.add(method=ConfidenceMethod.VERBAL_0_100, confidence=0.6, is_correct=False)

        # After accumulating observations
        stats = tracker.get_stats(ConfidenceMethod.VERBAL_0_100)
        print(f"WQD: {stats['wqd']}, Optimal T: {stats['optimal_temperature']}")

    """

    observations: dict[ConfidenceMethod, list[tuple[float, bool]]] = field(
        default_factory=lambda: {m: [] for m in ConfidenceMethod}
    )

    def add(
        self,
        method: ConfidenceMethod,
        confidence: float,
        is_correct: bool,
    ) -> None:
        """Add a calibration observation.

        Args:
            method: Confidence extraction method used.
            confidence: Confidence score assigned.
            is_correct: Whether the answer was actually correct.

        """
        self.observations[method].append((confidence, is_correct))

    def wqd_score(self, method: ConfidenceMethod) -> float:
        """Get WQD score for a specific method.

        Args:
            method: Confidence method to evaluate.

        Returns:
            WQD score (0.5 if insufficient data).

        """
        obs = self.observations.get(method, [])
        if len(obs) < 2:
            return 0.5
        return within_question_discrimination(obs)

    def find_optimal_temperature(
        self,
        method: ConfidenceMethod,
        temperature_grid: list[float] | None = None,
    ) -> float:
        """Find temperature that maximizes WQD for a method.

        Uses grid search over common temperature values.

        Args:
            method: Confidence method to optimize.
            temperature_grid: Custom temperatures to try.

        Returns:
            Optimal temperature (default if insufficient data).

        """
        obs = self.observations.get(method, [])
        if len(obs) < 10:
            return DEFAULT_TEMPERATURES.get(method, 1.0)

        grid = temperature_grid or [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        best_temp = 1.0
        best_wqd = 0.0

        for temp in grid:
            # Normalize scores with this temperature
            scores = [c for c, _ in obs]
            normalized = softmax_normalize(scores, temp)

            # Compute WQD with normalized scores
            normalized_obs = [
                (n, correct) for n, (_, correct) in zip(normalized, obs, strict=False)
            ]
            wqd = within_question_discrimination(normalized_obs)

            if wqd > best_wqd:
                best_wqd = wqd
                best_temp = temp

        return best_temp

    def get_stats(self, method: ConfidenceMethod) -> dict[str, Any]:
        """Get calibration statistics for a method.

        Args:
            method: Confidence method to analyze.

        Returns:
            Dictionary with WQD, optimal temperature, observation count.

        """
        obs = self.observations.get(method, [])
        return {
            "method": method.value,
            "observations": len(obs),
            "wqd": round(self.wqd_score(method), 3),
            "optimal_temperature": self.find_optimal_temperature(method),
            "correct_count": sum(1 for _, c in obs if c),
            "incorrect_count": sum(1 for _, c in obs if not c),
        }

    def get_all_stats(self) -> dict[str, Any]:
        """Get calibration statistics for all methods with data.

        Returns:
            Dictionary of method stats, sorted by WQD.

        """
        stats = []
        for method in ConfidenceMethod:
            if self.observations.get(method):
                stats.append(self.get_stats(method))

        # Sort by WQD descending
        stats.sort(key=lambda s: s["wqd"], reverse=True)

        return {
            "methods": stats,
            "best_method": stats[0]["method"] if stats else None,
            "total_observations": sum(s["observations"] for s in stats),
        }

    def reset(self, method: ConfidenceMethod | None = None) -> None:
        """Reset calibration data.

        Args:
            method: Specific method to reset, or None for all.

        """
        if method is None:
            for m in ConfidenceMethod:
                self.observations[m] = []
        else:
            self.observations[method] = []


# Global calibration tracker for cross-session analysis
_calibration_tracker = CalibrationTracker()


def get_calibration_tracker() -> CalibrationTracker:
    """Get the global calibration tracker instance."""
    return _calibration_tracker


# =============================================================================
# True CISC: Answer-Level Weighted Majority Voting
# =============================================================================


@dataclass
class CISCVoteResult:
    """Result of CISC weighted majority voting across multiple answers.

    This implements the paper's actual CISC method:
    â = argmax_a Σ(i=1 to m) 1[a_i = a] · c̃_i

    Where c̃_i are softmax-normalized confidence scores.
    """

    winner: str  # The winning answer
    winner_weight: float  # Total weight for winning answer
    answer_weights: dict[str, float]  # Weight per unique answer
    num_samples: int  # Total samples aggregated
    num_unique_answers: int  # Answer diversity
    temperature: float  # Temperature used for softmax
    normalized_confidences: list[float]  # Per-sample normalized confidences
    raw_confidences: list[float]  # Original confidence scores

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "winner": self.winner,
            "winner_weight": round(self.winner_weight, 4),
            "answer_weights": {k: round(v, 4) for k, v in self.answer_weights.items()},
            "num_samples": self.num_samples,
            "num_unique_answers": self.num_unique_answers,
            "answer_diversity": round(self.num_unique_answers / max(self.num_samples, 1), 3),
            "temperature": self.temperature,
            "confidence_stats": {
                "mean": round(sum(self.raw_confidences) / len(self.raw_confidences), 3)
                if self.raw_confidences
                else 0,
                "min": round(min(self.raw_confidences), 3) if self.raw_confidences else 0,
                "max": round(max(self.raw_confidences), 3) if self.raw_confidences else 0,
            },
        }


def cisc_vote(
    answers: list[str],
    confidences: list[float],
    temperature: float = 1.0,
    normalize_answers: bool = True,
) -> CISCVoteResult:
    """Perform CISC weighted majority voting on multiple answers.

    This is the core CISC algorithm from the paper:
    1. Normalize confidences using softmax with temperature
    2. Group answers and sum their normalized weights
    3. Return answer with highest total weight

    Unlike simple majority voting, CISC weights each answer by its
    confidence score, allowing high-confidence correct answers to
    outweigh multiple low-confidence incorrect answers.

    Args:
        answers: List of answers from N reasoning paths.
        confidences: Confidence score (0-1) for each answer.
        temperature: Softmax temperature for normalization.
            - Lower T (0.1): Emphasizes high-confidence answers
            - Higher T (2.0): Approaches uniform weighting
        normalize_answers: If True, strip whitespace and lowercase for comparison.

    Returns:
        CISCVoteResult with winning answer and voting statistics.

    Example:
        >>> # 3 answers: two say "42" (conf 0.8, 0.6), one says "41" (conf 0.9)
        >>> result = cisc_vote(
        ...     answers=["42", "41", "42"],
        ...     confidences=[0.8, 0.9, 0.6],
        ...     temperature=1.0
        ... )
        >>> # With T=1.0, "42" wins due to combined weight
        >>> result.winner
        "42"

        >>> # With very low T, highest single confidence wins
        >>> result = cisc_vote(answers=["42", "41", "42"], confidences=[0.8, 0.9, 0.6], temperature=0.1)
        >>> result.winner
        "41"

    """
    if not answers:
        raise ValueError("Cannot vote on empty answers list")
    if len(answers) != len(confidences):
        raise ValueError(
            f"answers ({len(answers)}) and confidences ({len(confidences)}) must match"
        )

    # Normalize confidences using softmax
    normalized = softmax_normalize(confidences, temperature=temperature)

    # Normalize answer strings for comparison
    def norm(s: str) -> str:
        return s.strip().lower() if normalize_answers else s

    # Group by answer and sum weights
    answer_weights: dict[str, float] = {}
    answer_canonical: dict[str, str] = {}  # normalized -> original

    for ans, weight in zip(answers, normalized, strict=False):
        key = norm(ans)
        if key not in answer_canonical:
            answer_canonical[key] = ans  # Keep first occurrence as canonical
        answer_weights[key] = answer_weights.get(key, 0.0) + weight

    # Find winner
    winner_key = max(answer_weights, key=lambda k: answer_weights[k])
    winner = answer_canonical[winner_key]

    # Build result with original answer forms
    final_weights = {answer_canonical[k]: v for k, v in answer_weights.items()}

    return CISCVoteResult(
        winner=winner,
        winner_weight=answer_weights[winner_key],
        answer_weights=final_weights,
        num_samples=len(answers),
        num_unique_answers=len(answer_weights),
        temperature=temperature,
        normalized_confidences=normalized,
        raw_confidences=confidences,
    )


def cisc_vote_with_reasoning(
    samples: list[tuple[str, str, float]],
    temperature: float = 1.0,
) -> tuple[CISCVoteResult, str]:
    """CISC voting that also returns the best reasoning path.

    Args:
        samples: List of (reasoning_path, answer, confidence) tuples.
        temperature: Softmax temperature.

    Returns:
        Tuple of (CISCVoteResult, best_reasoning_path).
        The reasoning path is from the highest-confidence sample
        that produced the winning answer.

    """
    if not samples:
        raise ValueError("Cannot vote on empty samples")

    answers = [ans for _, ans, _ in samples]
    confidences = [conf for _, _, conf in samples]

    result = cisc_vote(answers, confidences, temperature)

    # Find best reasoning for the winning answer
    def norm(s: str) -> str:
        return s.strip().lower()

    winner_norm = norm(result.winner)
    best_reasoning = ""
    best_conf = -1.0

    for reasoning, ans, conf in samples:
        if norm(ans) == winner_norm and conf > best_conf:
            best_conf = conf
            best_reasoning = reasoning

    return result, best_reasoning


def estimate_cisc_benefit(
    answers: list[str],
    confidences: list[float],
    correct_answer: str | None = None,
) -> dict[str, Any]:
    """Estimate potential CISC benefit for a set of samples.

    Compares CISC result to simple majority voting to measure
    how much confidence weighting changes the outcome.

    Args:
        answers: List of answers from N samples.
        confidences: Confidence scores for each answer.
        correct_answer: If provided, evaluates accuracy.

    Returns:
        Dictionary with comparison metrics.

    """
    if not answers:
        return {"error": "no answers provided"}

    # CISC voting
    cisc_result = cisc_vote(answers, confidences, temperature=1.0)

    # Simple majority voting (uniform weights)
    from collections import Counter

    def norm(s: str) -> str:
        return s.strip().lower()

    answer_counts = Counter(norm(a) for a in answers)
    majority_winner_norm = answer_counts.most_common(1)[0][0]
    # Find original form
    majority_winner = next(a for a in answers if norm(a) == majority_winner_norm)

    # Compare
    same_winner = norm(cisc_result.winner) == norm(majority_winner)

    result = {
        "cisc_winner": cisc_result.winner,
        "majority_winner": majority_winner,
        "same_winner": same_winner,
        "cisc_weight": round(cisc_result.winner_weight, 4),
        "majority_count": answer_counts[majority_winner_norm],
        "num_samples": len(answers),
        "answer_diversity": cisc_result.num_unique_answers,
    }

    if correct_answer is not None:
        correct_norm = norm(correct_answer)
        result["cisc_correct"] = norm(cisc_result.winner) == correct_norm
        result["majority_correct"] = majority_winner_norm == correct_norm
        result["cisc_improves"] = result["cisc_correct"] and not result["majority_correct"]
        result["cisc_hurts"] = not result["cisc_correct"] and result["majority_correct"]

    return result
