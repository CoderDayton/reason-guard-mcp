"""Pure routing rule functions for Atomic Reasoning Router.

This module contains stateless, pure functions that implement the 5 routing rules.
All functions take a session and return a status or None (no action needed).

Rules:
    A: Minimum Depth - Reject synthesis if insufficient steps
    B: Confidence Branching - Force branching on low confidence
    C: Verification Required - Require verification before synthesis
    D: State Machine - Enforce valid transitions
    E: Maximum Steps - Force synthesis at max steps
"""

from __future__ import annotations

from .router_types import (
    COMPLEXITY_CONFIGS,
    VALID_TRANSITIONS,
    Complexity,
    ComplexityConfig,
    RouterSession,
    RouterStatus,
    StepType,
)

# --- Rule A: Minimum Depth ---


def check_minimum_depth(session: RouterSession, step_type: StepType) -> RouterStatus | None:
    """Rule A: Reject synthesis if insufficient reasoning steps.

    Args:
        session: Current session state
        step_type: Proposed step type

    Returns:
        REJECTED if synthesis attempted too early, None otherwise

    """
    if step_type != StepType.SYNTHESIS:
        return None

    if session.step_count < session.min_steps:
        return RouterStatus.REJECTED

    return None


def get_minimum_depth_reason(session: RouterSession) -> str:
    """Get human-readable reason for minimum depth rejection."""
    needed = session.min_steps - session.step_count
    return (
        f"Synthesis rejected: {needed} more reasoning step(s) required "
        f"(have {session.step_count}, need {session.min_steps})"
    )


# --- Rule B: Confidence Branching ---


def check_confidence_threshold(
    session: RouterSession, confidence: float, step_type: StepType
) -> RouterStatus | None:
    """Rule B: Force branching when confidence is below threshold.

    Args:
        session: Current session state
        confidence: Confidence of proposed step
        step_type: Type of proposed step

    Returns:
        BRANCH_REQUIRED if confidence too low for hypothesis/verification, None otherwise

    """
    # Only check confidence for hypothesis and verification steps
    if step_type not in (StepType.HYPOTHESIS, StepType.VERIFICATION):
        return None

    if confidence < session.confidence_threshold:
        return RouterStatus.BRANCH_REQUIRED

    return None


def get_confidence_reason(session: RouterSession, confidence: float) -> str:
    """Get human-readable reason for confidence branching requirement."""
    return (
        f"Branching required: confidence {confidence:.2f} is below threshold "
        f"{session.confidence_threshold:.2f}. Create 2-4 alternative hypotheses."
    )


# --- Rule C: Verification Required ---


def check_verification_required(session: RouterSession, step_type: StepType) -> RouterStatus | None:
    """Rule C: Require at least one verification step before synthesis.

    Args:
        session: Current session state
        step_type: Proposed step type

    Returns:
        VERIFICATION_REQUIRED if synthesis attempted without verification, None otherwise

    """
    if step_type != StepType.SYNTHESIS:
        return None

    if not session.has_verification:
        return RouterStatus.VERIFICATION_REQUIRED

    return None


def get_verification_reason() -> str:
    """Get human-readable reason for verification requirement."""
    return (
        "Synthesis blocked: at least one verification step is required. "
        "Submit a verification step with evidence before concluding."
    )


# --- Rule D: State Machine Transitions ---


def check_valid_transition(session: RouterSession, step_type: StepType) -> RouterStatus | None:
    """Rule D: Enforce valid state machine transitions.

    State machine:
        None -> PREMISE (first step must be premise)
        PREMISE -> PREMISE | HYPOTHESIS
        HYPOTHESIS -> HYPOTHESIS | VERIFICATION
        VERIFICATION -> HYPOTHESIS | SYNTHESIS (if other rules pass)
        SYNTHESIS -> (terminal)

    Args:
        session: Current session state
        step_type: Proposed step type

    Returns:
        REJECTED if transition is invalid, None otherwise

    """
    current_state = session.last_step_type
    valid_next = VALID_TRANSITIONS.get(current_state, set())

    if step_type not in valid_next:
        return RouterStatus.REJECTED

    return None


def get_transition_reason(session: RouterSession, step_type: StepType) -> str:
    """Get human-readable reason for invalid transition."""
    current = session.last_step_type
    valid = VALID_TRANSITIONS.get(current, set())
    valid_names = [s.value for s in valid] if valid else ["(none - terminal)"]

    if current is None:
        return f"Invalid first step: must start with 'premise', got '{step_type.value}'"

    return (
        f"Invalid transition: cannot go from '{current.value}' to '{step_type.value}'. "
        f"Valid next steps: {', '.join(valid_names)}"
    )


# --- Rule E: Maximum Steps ---


def check_maximum_steps(session: RouterSession) -> bool:
    """Rule E: Check if maximum steps reached (forces synthesis).

    Args:
        session: Current session state

    Returns:
        True if max steps reached, False otherwise

    """
    return session.step_count >= session.max_steps


def get_max_steps_guidance(session: RouterSession) -> str:
    """Get guidance when max steps reached."""
    return (
        f"Maximum steps ({session.max_steps}) reached. "
        "You must now submit a synthesis step to conclude your reasoning."
    )


# --- Combined Rule Checking ---


def evaluate_step(
    session: RouterSession,
    step_type: StepType,
    confidence: float,
) -> tuple[RouterStatus, str | None]:
    """Evaluate a proposed step against all routing rules.

    Rules are checked in order: D (transition) -> A (depth) -> C (verification) -> B (confidence)

    Args:
        session: Current session state
        step_type: Proposed step type
        confidence: Confidence of proposed step

    Returns:
        Tuple of (status, rejection_reason or None)

    """
    # Rule D: Check valid transition first (fundamental)
    if (status := check_valid_transition(session, step_type)) is not None:
        return status, get_transition_reason(session, step_type)

    # Rule A: Check minimum depth for synthesis
    if (status := check_minimum_depth(session, step_type)) is not None:
        return status, get_minimum_depth_reason(session)

    # Rule C: Check verification required for synthesis
    if (status := check_verification_required(session, step_type)) is not None:
        return status, get_verification_reason()

    # Rule B: Check confidence threshold (non-blocking, suggests branching)
    if (status := check_confidence_threshold(session, confidence, step_type)) is not None:
        return status, get_confidence_reason(session, confidence)

    return RouterStatus.ACCEPTED, None


# --- Helper Functions ---


def get_valid_next_steps(session: RouterSession) -> list[StepType]:
    """Get list of valid next step types for current session state.

    Args:
        session: Current session state

    Returns:
        List of valid StepType values

    """
    current = session.last_step_type
    valid = set(VALID_TRANSITIONS.get(current, set()))

    # Filter out synthesis if verification required or depth insufficient
    if StepType.SYNTHESIS in valid:
        can_synth, _ = can_synthesize(session)
        if not can_synth:
            valid.discard(StepType.SYNTHESIS)

    return sorted(valid, key=lambda x: x.value)


def can_synthesize(session: RouterSession) -> tuple[bool, list[str]]:
    """Check if session can proceed to synthesis.

    Args:
        session: Current session state

    Returns:
        Tuple of (can_synthesize: bool, blockers: list of reasons why not)

    """
    blockers: list[str] = []

    # Check minimum depth
    if session.step_count < session.min_steps:
        needed = session.min_steps - session.step_count
        blockers.append(f"Need {needed} more reasoning step(s)")

    # Check verification
    if not session.has_verification:
        blockers.append("Verification step required")

    # Check we're in valid state for synthesis
    if session.last_step_type is not None and session.last_step_type != StepType.VERIFICATION:
        blockers.append(f"Cannot synthesize from '{session.last_step_type.value}' state")

    return len(blockers) == 0, blockers


def resolve_complexity(
    complexity: Complexity | str, problem: str | None = None
) -> tuple[Complexity, ComplexityConfig]:
    """Resolve complexity level and return config.

    Args:
        complexity: Requested complexity level or "auto"
        problem: Problem text (used for auto-detection)

    Returns:
        Tuple of (resolved Complexity, ComplexityConfig)

    """
    if isinstance(complexity, str):
        complexity = Complexity(complexity)

    if complexity == Complexity.AUTO:
        # Simple heuristic: length and keyword-based
        resolved = _auto_detect_complexity(problem or "")
    else:
        resolved = complexity

    config = COMPLEXITY_CONFIGS[resolved]
    return resolved, config


def _auto_detect_complexity(problem: str) -> Complexity:
    """Auto-detect complexity from problem text.

    Simple heuristic based on:
    - Length
    - Presence of complexity keywords

    Args:
        problem: Problem text

    Returns:
        Detected Complexity level

    """
    problem_lower = problem.lower()

    # High complexity indicators
    high_keywords = [
        "paradox",
        "contradiction",
        "prove",
        "counterintuitive",
        "probability",
        "conditional",
        "bayesian",
        "causal",
        "correlation",
        "fallacy",
        "bias",
        "selection",
        "simpson",
        "monty hall",
        "base rate",
    ]
    if any(kw in problem_lower for kw in high_keywords):
        return Complexity.HIGH

    # Medium complexity indicators
    medium_keywords = [
        "compare",
        "analyze",
        "evaluate",
        "trade-off",
        "consider",
        "multiple",
        "factors",
        "depends",
    ]
    if any(kw in problem_lower for kw in medium_keywords) or len(problem) > 500:
        return Complexity.MEDIUM

    return Complexity.LOW


def steps_remaining(session: RouterSession) -> int:
    """Calculate steps remaining before max reached.

    Args:
        session: Current session state

    Returns:
        Number of steps remaining (minimum 0)

    """
    return max(0, session.max_steps - session.step_count)


def steps_until_synthesis_allowed(session: RouterSession) -> int:
    """Calculate steps needed before synthesis is allowed.

    Args:
        session: Current session state

    Returns:
        Number of steps still required (minimum 0)

    """
    return max(0, session.min_steps - session.step_count)
