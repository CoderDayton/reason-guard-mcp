"""Unit tests for routing rules.

Tests all 5 routing rules (A-E) with various session states.
Each rule is tested independently with focused test cases.
"""

from __future__ import annotations

from src.tools.router_types import (
    COMPLEXITY_CONFIGS,
    Complexity,
    RouterSession,
    RouterStatus,
    RouterStep,
    StepType,
)
from src.tools.routing_rules import (
    can_synthesize,
    check_confidence_threshold,
    check_maximum_steps,
    check_minimum_depth,
    check_valid_transition,
    check_verification_required,
    evaluate_step,
    get_valid_next_steps,
    resolve_complexity,
    steps_remaining,
    steps_until_synthesis_allowed,
)

# --- Test Fixtures ---


def create_session(
    complexity: Complexity = Complexity.MEDIUM,
    steps: list[StepType] | None = None,
    verified: bool = False,
) -> RouterSession:
    """Create a test session with given state."""
    config = COMPLEXITY_CONFIGS[complexity]

    session = RouterSession(
        id="test-session",
        problem="Test problem",
        complexity=complexity,
        min_steps=config.min_steps,
        max_steps=config.max_steps,
        confidence_threshold=config.confidence_threshold,
    )

    # Add steps if provided
    if steps:
        for i, step_type in enumerate(steps):
            session.steps.append(
                RouterStep(
                    id=f"step-{i}",
                    step_type=step_type,
                    content=f"Step {i} content",
                    confidence=0.8,
                )
            )

    # Mark as verified if requested
    if verified:
        session.verified_claims.append("Test claim verified")

    return session


# --- Rule A: Minimum Depth Tests ---


class TestRuleAMinimumDepth:
    """Tests for Rule A: Reject synthesis if insufficient steps."""

    def test_rejects_early_synthesis_low_complexity(self) -> None:
        """LOW complexity: min_steps=2, should reject with 1 step."""
        session = create_session(
            complexity=Complexity.LOW,
            steps=[StepType.PREMISE],
        )
        result = check_minimum_depth(session, StepType.SYNTHESIS)
        assert result == RouterStatus.REJECTED

    def test_rejects_early_synthesis_medium_complexity(self) -> None:
        """MEDIUM complexity: min_steps=4, should reject with 2 steps."""
        session = create_session(
            complexity=Complexity.MEDIUM,
            steps=[StepType.PREMISE, StepType.HYPOTHESIS],
        )
        result = check_minimum_depth(session, StepType.SYNTHESIS)
        assert result == RouterStatus.REJECTED

    def test_rejects_early_synthesis_high_complexity(self) -> None:
        """HIGH complexity: min_steps=6, should reject with 4 steps."""
        session = create_session(
            complexity=Complexity.HIGH,
            steps=[
                StepType.PREMISE,
                StepType.PREMISE,
                StepType.HYPOTHESIS,
                StepType.VERIFICATION,
            ],
        )
        result = check_minimum_depth(session, StepType.SYNTHESIS)
        assert result == RouterStatus.REJECTED

    def test_accepts_synthesis_at_min_steps(self) -> None:
        """Should accept synthesis when exactly at min_steps."""
        session = create_session(
            complexity=Complexity.LOW,  # min_steps=2
            steps=[StepType.PREMISE, StepType.HYPOTHESIS],
        )
        result = check_minimum_depth(session, StepType.SYNTHESIS)
        assert result is None

    def test_accepts_synthesis_above_min_steps(self) -> None:
        """Should accept synthesis when above min_steps."""
        session = create_session(
            complexity=Complexity.LOW,  # min_steps=2
            steps=[
                StepType.PREMISE,
                StepType.PREMISE,
                StepType.HYPOTHESIS,
                StepType.VERIFICATION,
            ],
        )
        result = check_minimum_depth(session, StepType.SYNTHESIS)
        assert result is None

    def test_ignores_non_synthesis_steps(self) -> None:
        """Should return None for non-synthesis steps."""
        session = create_session(complexity=Complexity.HIGH, steps=[])
        assert check_minimum_depth(session, StepType.PREMISE) is None
        assert check_minimum_depth(session, StepType.HYPOTHESIS) is None
        assert check_minimum_depth(session, StepType.VERIFICATION) is None


# --- Rule B: Confidence Branching Tests ---


class TestRuleBConfidenceBranching:
    """Tests for Rule B: Force branching on low confidence."""

    def test_requires_branching_low_confidence_hypothesis(self) -> None:
        """Should require branching when hypothesis confidence is below threshold."""
        session = create_session(complexity=Complexity.MEDIUM)  # threshold=0.70
        result = check_confidence_threshold(session, confidence=0.65, step_type=StepType.HYPOTHESIS)
        assert result == RouterStatus.BRANCH_REQUIRED

    def test_requires_branching_low_confidence_verification(self) -> None:
        """Should require branching when verification confidence is below threshold."""
        session = create_session(complexity=Complexity.MEDIUM)  # threshold=0.70
        result = check_confidence_threshold(
            session, confidence=0.50, step_type=StepType.VERIFICATION
        )
        assert result == RouterStatus.BRANCH_REQUIRED

    def test_accepts_high_confidence(self) -> None:
        """Should accept when confidence is at or above threshold."""
        session = create_session(complexity=Complexity.MEDIUM)  # threshold=0.70
        result = check_confidence_threshold(session, confidence=0.70, step_type=StepType.HYPOTHESIS)
        assert result is None

        result = check_confidence_threshold(session, confidence=0.85, step_type=StepType.HYPOTHESIS)
        assert result is None

    def test_ignores_premise_steps(self) -> None:
        """Should not check confidence for premise steps."""
        session = create_session(complexity=Complexity.HIGH)  # threshold=0.75
        result = check_confidence_threshold(session, confidence=0.30, step_type=StepType.PREMISE)
        assert result is None

    def test_ignores_synthesis_steps(self) -> None:
        """Should not check confidence for synthesis steps."""
        session = create_session(complexity=Complexity.HIGH)
        result = check_confidence_threshold(session, confidence=0.30, step_type=StepType.SYNTHESIS)
        assert result is None

    def test_edge_case_exactly_at_threshold(self) -> None:
        """Confidence exactly at threshold should be accepted."""
        session = create_session(complexity=Complexity.LOW)  # threshold=0.60
        result = check_confidence_threshold(session, confidence=0.60, step_type=StepType.HYPOTHESIS)
        assert result is None


# --- Rule C: Verification Required Tests ---


class TestRuleCVerificationRequired:
    """Tests for Rule C: Require verification before synthesis."""

    def test_requires_verification_before_synthesis(self) -> None:
        """Should require verification when attempting synthesis without it."""
        session = create_session(
            complexity=Complexity.LOW,
            steps=[StepType.PREMISE, StepType.HYPOTHESIS],
        )
        result = check_verification_required(session, StepType.SYNTHESIS)
        assert result == RouterStatus.VERIFICATION_REQUIRED

    def test_allows_synthesis_after_verification(self) -> None:
        """Should allow synthesis when verification step exists."""
        session = create_session(
            complexity=Complexity.LOW,
            steps=[StepType.PREMISE, StepType.HYPOTHESIS, StepType.VERIFICATION],
        )
        result = check_verification_required(session, StepType.SYNTHESIS)
        assert result is None

    def test_ignores_non_synthesis_steps(self) -> None:
        """Should return None for non-synthesis steps."""
        session = create_session(complexity=Complexity.LOW, steps=[])
        assert check_verification_required(session, StepType.PREMISE) is None
        assert check_verification_required(session, StepType.HYPOTHESIS) is None
        assert check_verification_required(session, StepType.VERIFICATION) is None


# --- Rule D: State Machine Transition Tests ---


class TestRuleDStateMachine:
    """Tests for Rule D: Enforce valid state machine transitions."""

    def test_first_step_must_be_premise(self) -> None:
        """First step must be premise, not hypothesis/verification/synthesis."""
        session = create_session(steps=[])
        assert check_valid_transition(session, StepType.PREMISE) is None
        assert check_valid_transition(session, StepType.HYPOTHESIS) == RouterStatus.REJECTED
        assert check_valid_transition(session, StepType.VERIFICATION) == RouterStatus.REJECTED
        assert check_valid_transition(session, StepType.SYNTHESIS) == RouterStatus.REJECTED

    def test_premise_can_go_to_premise_or_hypothesis(self) -> None:
        """From premise, can go to premise or hypothesis."""
        session = create_session(steps=[StepType.PREMISE])
        assert check_valid_transition(session, StepType.PREMISE) is None
        assert check_valid_transition(session, StepType.HYPOTHESIS) is None
        assert check_valid_transition(session, StepType.VERIFICATION) == RouterStatus.REJECTED
        assert check_valid_transition(session, StepType.SYNTHESIS) == RouterStatus.REJECTED

    def test_hypothesis_can_go_to_hypothesis_or_verification(self) -> None:
        """From hypothesis, can go to hypothesis or verification."""
        session = create_session(steps=[StepType.PREMISE, StepType.HYPOTHESIS])
        assert check_valid_transition(session, StepType.PREMISE) == RouterStatus.REJECTED
        assert check_valid_transition(session, StepType.HYPOTHESIS) is None
        assert check_valid_transition(session, StepType.VERIFICATION) is None
        assert check_valid_transition(session, StepType.SYNTHESIS) == RouterStatus.REJECTED

    def test_verification_can_go_to_hypothesis_or_synthesis(self) -> None:
        """From verification, can go to hypothesis or synthesis."""
        session = create_session(
            steps=[StepType.PREMISE, StepType.HYPOTHESIS, StepType.VERIFICATION]
        )
        assert check_valid_transition(session, StepType.PREMISE) == RouterStatus.REJECTED
        assert check_valid_transition(session, StepType.HYPOTHESIS) is None
        assert check_valid_transition(session, StepType.VERIFICATION) == RouterStatus.REJECTED
        assert check_valid_transition(session, StepType.SYNTHESIS) is None

    def test_synthesis_is_terminal(self) -> None:
        """From synthesis, no transitions allowed."""
        session = create_session(
            steps=[
                StepType.PREMISE,
                StepType.HYPOTHESIS,
                StepType.VERIFICATION,
                StepType.SYNTHESIS,
            ]
        )
        assert check_valid_transition(session, StepType.PREMISE) == RouterStatus.REJECTED
        assert check_valid_transition(session, StepType.HYPOTHESIS) == RouterStatus.REJECTED
        assert check_valid_transition(session, StepType.VERIFICATION) == RouterStatus.REJECTED
        assert check_valid_transition(session, StepType.SYNTHESIS) == RouterStatus.REJECTED


# --- Rule E: Maximum Steps Tests ---


class TestRuleEMaximumSteps:
    """Tests for Rule E: Check if maximum steps reached."""

    def test_detects_max_steps_reached(self) -> None:
        """Should return True when at max steps."""
        session = create_session(
            complexity=Complexity.LOW,  # max_steps=5
            steps=[
                StepType.PREMISE,
                StepType.PREMISE,
                StepType.HYPOTHESIS,
                StepType.HYPOTHESIS,
                StepType.VERIFICATION,
            ],
        )
        assert check_maximum_steps(session) is True

    def test_detects_below_max_steps(self) -> None:
        """Should return False when below max steps."""
        session = create_session(
            complexity=Complexity.LOW,  # max_steps=5
            steps=[StepType.PREMISE, StepType.HYPOTHESIS],
        )
        assert check_maximum_steps(session) is False

    def test_synthesis_not_counted(self) -> None:
        """Synthesis steps should not count toward max."""
        session = create_session(
            complexity=Complexity.LOW,  # max_steps=5
            steps=[
                StepType.PREMISE,
                StepType.HYPOTHESIS,
                StepType.VERIFICATION,
                StepType.SYNTHESIS,
            ],
        )
        # Only 3 non-synthesis steps, max is 5
        assert check_maximum_steps(session) is False


# --- Combined Rule Evaluation Tests ---


class TestEvaluateStep:
    """Tests for combined rule evaluation."""

    def test_accepts_valid_first_premise(self) -> None:
        """Should accept a valid first premise step."""
        session = create_session(steps=[])
        status, reason = evaluate_step(session, StepType.PREMISE, confidence=0.9)
        assert status == RouterStatus.ACCEPTED
        assert reason is None

    def test_rejects_invalid_first_step(self) -> None:
        """Should reject non-premise first step."""
        session = create_session(steps=[])
        status, reason = evaluate_step(session, StepType.HYPOTHESIS, confidence=0.9)
        assert status == RouterStatus.REJECTED
        assert "Invalid first step" in reason

    def test_requires_branching_on_low_confidence(self) -> None:
        """Should require branching when confidence below threshold."""
        session = create_session(
            complexity=Complexity.MEDIUM,
            steps=[StepType.PREMISE],  # threshold=0.70
        )
        status, reason = evaluate_step(session, StepType.HYPOTHESIS, confidence=0.50)
        assert status == RouterStatus.BRANCH_REQUIRED
        assert "0.50" in reason

    def test_rejects_synthesis_from_hypothesis_state(self) -> None:
        """Should reject synthesis from hypothesis state (state machine rule)."""
        session = create_session(
            complexity=Complexity.LOW,  # min_steps=2
            steps=[StepType.PREMISE, StepType.HYPOTHESIS],
        )
        # State machine (Rule D) blocks this - can't go HYPOTHESIS -> SYNTHESIS
        status, reason = evaluate_step(session, StepType.SYNTHESIS, confidence=0.9)
        assert status == RouterStatus.REJECTED
        assert "Invalid transition" in reason

    def test_rejects_synthesis_below_min_steps(self) -> None:
        """Should reject synthesis when below minimum steps."""
        session = create_session(
            complexity=Complexity.MEDIUM,  # min_steps=4
            steps=[StepType.PREMISE, StepType.HYPOTHESIS, StepType.VERIFICATION],
        )
        status, reason = evaluate_step(session, StepType.SYNTHESIS, confidence=0.9)
        assert status == RouterStatus.REJECTED
        assert "more reasoning step" in reason

    def test_accepts_valid_synthesis(self) -> None:
        """Should accept synthesis when all conditions met."""
        session = create_session(
            complexity=Complexity.LOW,  # min_steps=2
            steps=[StepType.PREMISE, StepType.HYPOTHESIS, StepType.VERIFICATION],
        )
        status, reason = evaluate_step(session, StepType.SYNTHESIS, confidence=0.9)
        assert status == RouterStatus.ACCEPTED
        assert reason is None


# --- Helper Function Tests ---


class TestGetValidNextSteps:
    """Tests for get_valid_next_steps helper."""

    def test_empty_session_returns_premise(self) -> None:
        """Empty session should only allow premise."""
        session = create_session(steps=[])
        valid = get_valid_next_steps(session)
        assert valid == [StepType.PREMISE]

    def test_after_premise(self) -> None:
        """After premise, can add premise or hypothesis."""
        session = create_session(steps=[StepType.PREMISE])
        valid = get_valid_next_steps(session)
        assert set(valid) == {StepType.PREMISE, StepType.HYPOTHESIS}

    def test_after_hypothesis(self) -> None:
        """After hypothesis, can add hypothesis or verification."""
        session = create_session(steps=[StepType.PREMISE, StepType.HYPOTHESIS])
        valid = get_valid_next_steps(session)
        assert set(valid) == {StepType.HYPOTHESIS, StepType.VERIFICATION}

    def test_after_verification_without_min_steps(self) -> None:
        """After verification but before min_steps, synthesis not allowed."""
        session = create_session(
            complexity=Complexity.MEDIUM,  # min_steps=4
            steps=[StepType.PREMISE, StepType.HYPOTHESIS, StepType.VERIFICATION],
        )
        valid = get_valid_next_steps(session)
        # Synthesis blocked by min_steps rule
        assert StepType.SYNTHESIS not in valid
        assert StepType.HYPOTHESIS in valid


class TestCanSynthesize:
    """Tests for can_synthesize helper."""

    def test_cannot_synthesize_without_steps(self) -> None:
        """Empty session cannot synthesize."""
        session = create_session(steps=[])
        can, blockers = can_synthesize(session)
        assert can is False
        assert len(blockers) >= 1

    def test_cannot_synthesize_without_verification(self) -> None:
        """Cannot synthesize without verification step."""
        session = create_session(
            complexity=Complexity.LOW,
            steps=[StepType.PREMISE, StepType.HYPOTHESIS],
        )
        can, blockers = can_synthesize(session)
        assert can is False
        assert any("verification" in b.lower() for b in blockers)

    def test_cannot_synthesize_below_min_steps(self) -> None:
        """Cannot synthesize below min_steps."""
        session = create_session(
            complexity=Complexity.MEDIUM,  # min_steps=4
            steps=[StepType.PREMISE, StepType.HYPOTHESIS, StepType.VERIFICATION],
        )
        can, blockers = can_synthesize(session)
        assert can is False
        assert any("more reasoning" in b.lower() for b in blockers)

    def test_can_synthesize_when_ready(self) -> None:
        """Can synthesize when all conditions met."""
        session = create_session(
            complexity=Complexity.LOW,  # min_steps=2
            steps=[StepType.PREMISE, StepType.HYPOTHESIS, StepType.VERIFICATION],
        )
        can, blockers = can_synthesize(session)
        assert can is True
        assert blockers == []


class TestResolveComplexity:
    """Tests for complexity resolution."""

    def test_explicit_complexity(self) -> None:
        """Explicit complexity should be returned as-is."""
        resolved, config = resolve_complexity(Complexity.HIGH)
        assert resolved == Complexity.HIGH
        assert config.min_steps == 6

    def test_auto_detects_high_from_keywords(self) -> None:
        """Should detect HIGH complexity from keywords."""
        resolved, _ = resolve_complexity(Complexity.AUTO, "Explain the Monty Hall paradox")
        assert resolved == Complexity.HIGH

        resolved, _ = resolve_complexity(Complexity.AUTO, "What is Simpson's paradox?")
        assert resolved == Complexity.HIGH

    def test_auto_detects_medium_from_length(self) -> None:
        """Should detect MEDIUM complexity from long problem."""
        long_problem = "Consider the following scenario " * 50
        resolved, _ = resolve_complexity(Complexity.AUTO, long_problem)
        assert resolved == Complexity.MEDIUM

    def test_auto_defaults_to_low(self) -> None:
        """Should default to LOW for simple problems."""
        resolved, _ = resolve_complexity(Complexity.AUTO, "What is 2+2?")
        assert resolved == Complexity.LOW


class TestStepsRemaining:
    """Tests for step counting helpers."""

    def test_steps_remaining(self) -> None:
        """Should calculate remaining steps correctly."""
        session = create_session(
            complexity=Complexity.LOW,  # max_steps=5
            steps=[StepType.PREMISE, StepType.HYPOTHESIS],
        )
        assert steps_remaining(session) == 3

    def test_steps_until_synthesis(self) -> None:
        """Should calculate steps until synthesis correctly."""
        session = create_session(
            complexity=Complexity.MEDIUM,  # min_steps=4
            steps=[StepType.PREMISE],
        )
        assert steps_until_synthesis_allowed(session) == 3


# --- Complexity Configuration Tests ---


class TestComplexityConfigs:
    """Tests for complexity configuration values."""

    def test_low_config(self) -> None:
        """LOW complexity config should have lenient values."""
        config = COMPLEXITY_CONFIGS[Complexity.LOW]
        assert config.min_steps == 2
        assert config.max_steps == 5
        assert config.confidence_threshold == 0.60

    def test_medium_config(self) -> None:
        """MEDIUM complexity config should have moderate values."""
        config = COMPLEXITY_CONFIGS[Complexity.MEDIUM]
        assert config.min_steps == 4
        assert config.max_steps == 8
        assert config.confidence_threshold == 0.70

    def test_high_config(self) -> None:
        """HIGH complexity config should have strict values."""
        config = COMPLEXITY_CONFIGS[Complexity.HIGH]
        assert config.min_steps == 6
        assert config.max_steps == 12
        assert config.confidence_threshold == 0.75

    def test_configs_are_ordered(self) -> None:
        """Configs should increase in strictness from LOW to HIGH."""
        low = COMPLEXITY_CONFIGS[Complexity.LOW]
        med = COMPLEXITY_CONFIGS[Complexity.MEDIUM]
        high = COMPLEXITY_CONFIGS[Complexity.HIGH]

        assert low.min_steps < med.min_steps < high.min_steps
        assert low.max_steps < med.max_steps < high.max_steps
        assert low.confidence_threshold < med.confidence_threshold < high.confidence_threshold
