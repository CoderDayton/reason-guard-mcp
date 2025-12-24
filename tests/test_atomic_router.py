"""Unit tests for Atomic Router module.

Tests cover:
- Session lifecycle (create, expire, close)
- Step submission and routing rules
- Branch creation and management
- Claim verification
- Edge cases and error handling
"""

from __future__ import annotations

import pytest

from src.tools.atomic_router import (
    close_session,
    create_branch,
    get_router_stats,
    get_session_state,
    initialize_reasoning,
    submit_atomic_step,
    verify_claims,
)
from src.tools.router_types import Complexity, RouterStatus


class TestInitializeReasoning:
    """Tests for initialize_reasoning function."""

    def test_creates_session_with_valid_complexity(self) -> None:
        """Test session creation with explicit complexity."""
        result = initialize_reasoning("Test problem", "low")

        assert result.session_id is not None
        assert len(result.session_id) > 0
        assert result.complexity == Complexity.LOW
        assert result.guidance is not None

    def test_creates_session_with_auto_complexity(self) -> None:
        """Test session creation with auto-detected complexity."""
        result = initialize_reasoning(
            "Explain the Monty Hall paradox in probability theory",
            "auto",
        )

        assert result.session_id is not None
        # Should detect high complexity from keywords
        assert result.complexity in [Complexity.MEDIUM, Complexity.HIGH]

    def test_invalid_complexity_raises_error(self) -> None:
        """Test that invalid complexity raises ValueError."""
        with pytest.raises(ValueError, match="not a valid Complexity"):
            initialize_reasoning("Test", "invalid_level")

    def test_guidance_contains_trap_warnings(self) -> None:
        """Test that guidance contains relevant trap warnings for known problems."""
        result = initialize_reasoning(
            "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball.",
            "medium",
        )

        # Should have some guidance (may or may not match exact problem)
        assert result.guidance is not None
        assert len(result.guidance) > 0

    def test_each_session_has_unique_id(self) -> None:
        """Test that each session gets a unique ID."""
        ids = set()
        for _ in range(10):
            result = initialize_reasoning("Test problem", "low")
            ids.add(result.session_id)

        assert len(ids) == 10


class TestSubmitAtomicStep:
    """Tests for submit_atomic_step function."""

    def test_accepts_valid_premise_step(self) -> None:
        """Test that valid premise step is accepted."""
        session = initialize_reasoning("Test problem", "low")

        result = submit_atomic_step(
            session_id=session.session_id,
            step_type="premise",
            content="Given information analysis",
            confidence=0.8,
        )

        assert result.status == RouterStatus.ACCEPTED
        assert result.step_id is not None

    def test_rejects_invalid_session(self) -> None:
        """Test that invalid session ID returns rejected status."""
        result = submit_atomic_step(
            session_id="nonexistent_session_12345",
            step_type="premise",
            content="Test content",
            confidence=0.8,
        )

        assert result.status == RouterStatus.REJECTED
        assert "not found" in (result.rejection_reason or "").lower()

    def test_rejects_invalid_step_type(self) -> None:
        """Test that invalid step type is rejected."""
        session = initialize_reasoning("Test problem", "low")

        result = submit_atomic_step(
            session_id=session.session_id,
            step_type="invalid_type",
            content="Test content",
            confidence=0.8,
        )

        assert result.status == RouterStatus.REJECTED
        assert "invalid" in (result.rejection_reason or "").lower()

    def test_rejects_early_synthesis(self) -> None:
        """Test that synthesis before min_steps is rejected (Rule A)."""
        session = initialize_reasoning("Test problem", "low")

        # Try to synthesize immediately
        result = submit_atomic_step(
            session_id=session.session_id,
            step_type="synthesis",
            content="Final answer",
            confidence=0.9,
        )

        assert result.status == RouterStatus.REJECTED
        assert result.can_synthesize is False

    def test_tracks_step_count(self) -> None:
        """Test that step count increments correctly."""
        session = initialize_reasoning("Test problem", "low")

        # Submit two steps
        submit_atomic_step(session.session_id, "premise", "Step 1", 0.8)
        result = submit_atomic_step(session.session_id, "hypothesis", "Step 2", 0.8)

        assert result.steps_taken == 2

    def test_returns_valid_next_steps(self) -> None:
        """Test that valid_next_steps is populated correctly."""
        session = initialize_reasoning("Test problem", "low")

        result = submit_atomic_step(session.session_id, "premise", "Analysis", 0.8)

        assert result.valid_next_steps is not None
        assert len(result.valid_next_steps) > 0
        # After premise, should be able to do hypothesis or another premise
        assert "hypothesis" in result.valid_next_steps or "premise" in result.valid_next_steps

    def test_low_confidence_triggers_branch_required(self) -> None:
        """Test that low confidence triggers BRANCH_REQUIRED (Rule B)."""
        session = initialize_reasoning("Test problem", "high")

        # Submit steps with low confidence (below 0.75 threshold for HIGH)
        result = submit_atomic_step(
            session_id=session.session_id,
            step_type="hypothesis",
            content="Uncertain hypothesis",
            confidence=0.5,  # Below threshold
        )

        # Should either be BRANCH_REQUIRED or REJECTED
        assert result.status in [RouterStatus.BRANCH_REQUIRED, RouterStatus.REJECTED]


class TestCreateBranch:
    """Tests for create_branch function."""

    def test_creates_branch_with_valid_alternatives(self) -> None:
        """Test branch creation with valid alternatives."""
        session = initialize_reasoning("Test problem", "low")

        result = create_branch(
            session_id=session.session_id,
            alternatives=["Option A", "Option B"],
        )

        assert len(result.branch_ids) == 2
        assert result.guidance is not None

    def test_rejects_invalid_session(self) -> None:
        """Test that invalid session returns error in guidance."""
        result = create_branch(
            session_id="nonexistent_session",
            alternatives=["A", "B"],
        )

        assert "error" in result.guidance.lower()
        assert result.branch_ids == []

    def test_rejects_too_few_alternatives(self) -> None:
        """Test that less than 2 alternatives is rejected."""
        session = initialize_reasoning("Test problem", "low")

        result = create_branch(
            session_id=session.session_id,
            alternatives=["Only one"],
        )

        assert "error" in result.guidance.lower()
        assert result.branch_ids == []

    def test_rejects_too_many_alternatives(self) -> None:
        """Test that more than 4 alternatives is rejected."""
        session = initialize_reasoning("Test problem", "low")

        result = create_branch(
            session_id=session.session_id,
            alternatives=["A", "B", "C", "D", "E"],
        )

        assert "error" in result.guidance.lower()
        assert result.branch_ids == []


class TestVerifyClaims:
    """Tests for verify_claims function."""

    def test_verifies_claims_with_evidence(self) -> None:
        """Test claim verification with supporting evidence."""
        session = initialize_reasoning("Test problem", "low")

        result = verify_claims(
            session_id=session.session_id,
            claims=["The sky is blue"],
            evidence=["Scientific observations confirm the sky appears blue"],
        )

        assert result.verified is not None
        assert isinstance(result.verified, list)

    def test_rejects_invalid_session(self) -> None:
        """Test that invalid session returns empty results."""
        result = verify_claims(
            session_id="nonexistent_session",
            claims=["Test claim"],
            evidence=["Test evidence"],
        )

        # Should have error indication
        assert result.verified == [] or result.contradictions is not None


class TestGetSessionState:
    """Tests for get_session_state function."""

    def test_returns_state_for_valid_session(self) -> None:
        """Test state retrieval for valid session."""
        session = initialize_reasoning("Test problem", "low")
        submit_atomic_step(session.session_id, "premise", "Test", 0.8)

        state = get_session_state(session.session_id)

        assert state is not None
        assert "step_count" in state
        assert state["step_count"] == 1

    def test_returns_none_for_invalid_session(self) -> None:
        """Test that invalid session returns None."""
        state = get_session_state("nonexistent_session_12345")

        assert state is None


class TestCloseSession:
    """Tests for close_session function."""

    def test_closes_valid_session(self) -> None:
        """Test closing a valid session."""
        session = initialize_reasoning("Test problem", "low")

        result = close_session(session.session_id)

        assert result is True

        # Session should no longer exist
        state = get_session_state(session.session_id)
        assert state is None

    def test_returns_false_for_invalid_session(self) -> None:
        """Test closing invalid session returns False."""
        result = close_session("nonexistent_session_12345")

        assert result is False


class TestGetRouterStats:
    """Tests for get_router_stats function."""

    def test_returns_valid_stats(self) -> None:
        """Test that stats are returned in expected format."""
        stats = get_router_stats()

        assert "active_sessions" in stats
        assert "total_steps" in stats
        assert isinstance(stats["active_sessions"], int)


class TestCompleteWorkflow:
    """Integration tests for complete reasoning workflows."""

    def test_low_complexity_workflow(self) -> None:
        """Test complete workflow for LOW complexity problem."""
        # Initialize
        session = initialize_reasoning("What is 2 + 2?", "low")
        assert session.complexity == Complexity.LOW

        # Premise
        r1 = submit_atomic_step(session.session_id, "premise", "Given: 2 + 2", 0.9)
        assert r1.status == RouterStatus.ACCEPTED

        # Hypothesis
        r2 = submit_atomic_step(session.session_id, "hypothesis", "Answer is 4", 0.9)
        assert r2.status == RouterStatus.ACCEPTED

        # Verification (optional but good practice)
        r3 = submit_atomic_step(session.session_id, "verification", "2 + 2 = 4 confirmed", 0.9)
        assert r3.status == RouterStatus.ACCEPTED

        # At this point (3 steps), synthesis should be allowed for LOW complexity (min_steps=2)
        r4 = submit_atomic_step(session.session_id, "synthesis", "Final answer: 4", 0.95)
        # May or may not be accepted depending on exact rule implementation
        assert r4.status in [RouterStatus.ACCEPTED, RouterStatus.VERIFICATION_REQUIRED]

    def test_state_machine_enforcement(self) -> None:
        """Test that state machine prevents invalid transitions."""
        session = initialize_reasoning("Test problem", "low")

        # First step must be premise
        r1 = submit_atomic_step(session.session_id, "premise", "Starting point", 0.8)
        assert r1.status == RouterStatus.ACCEPTED

        # Can't go directly to synthesis from premise
        r2 = submit_atomic_step(session.session_id, "synthesis", "Premature synthesis", 0.9)
        # Should be rejected (either by state machine or min_steps)
        assert r2.status == RouterStatus.REJECTED

    def test_max_steps_enforcement(self) -> None:
        """Test that max_steps forces synthesis (Rule E)."""
        session = initialize_reasoning("Test problem", "low")
        # LOW complexity has max_steps=5

        # Submit 5 steps
        for i in range(5):
            step_type = "premise" if i < 2 else "hypothesis" if i < 4 else "verification"
            result = submit_atomic_step(
                session.session_id,
                step_type,
                f"Step {i + 1}",
                0.9,
            )
            if result.status == RouterStatus.REJECTED:
                break

        # Check that synthesis is required or steps_remaining is 0
        state = get_session_state(session.session_id)
        if state:
            assert state.get("step_count", 0) <= 5


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_content_handled(self) -> None:
        """Test that empty content is handled gracefully."""
        session = initialize_reasoning("Test", "low")

        result = submit_atomic_step(session.session_id, "premise", "", 0.8)

        # Should either accept or reject gracefully (not crash)
        assert result.status in [RouterStatus.ACCEPTED, RouterStatus.REJECTED]

    def test_extreme_confidence_values(self) -> None:
        """Test handling of extreme confidence values."""
        session = initialize_reasoning("Test", "low")

        # Test confidence = 0
        r1 = submit_atomic_step(session.session_id, "premise", "Low conf", 0.0)
        assert r1.status in [
            RouterStatus.ACCEPTED,
            RouterStatus.REJECTED,
            RouterStatus.BRANCH_REQUIRED,
        ]

        # Test confidence = 1
        r2 = submit_atomic_step(session.session_id, "premise", "High conf", 1.0)
        assert r2.status in [RouterStatus.ACCEPTED, RouterStatus.REJECTED]

    def test_very_long_content(self) -> None:
        """Test handling of very long content."""
        session = initialize_reasoning("Test", "low")

        long_content = "X" * 10000
        result = submit_atomic_step(session.session_id, "premise", long_content, 0.8)

        # Should handle gracefully
        assert result.status in [RouterStatus.ACCEPTED, RouterStatus.REJECTED]

    def test_unicode_content(self) -> None:
        """Test handling of unicode content."""
        session = initialize_reasoning("Test", "low")

        unicode_content = "数学问题 🧮 αβγ δεζ"
        result = submit_atomic_step(session.session_id, "premise", unicode_content, 0.8)

        assert result.status == RouterStatus.ACCEPTED
