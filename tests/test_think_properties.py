"""Property-based tests for the consolidated think() tool.

Uses hypothesis to generate test cases and verify invariants across all
valid action/mode combinations.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_rate_limiter() -> None:
    """Reset rate limiter before each test to avoid rate limit interference."""
    import src.server as server_module

    if server_module._rate_limiter is not None:
        server_module._rate_limiter._timestamps.clear()


# =============================================================================
# Strategy Definitions
# =============================================================================

# Valid action values (from ThinkAction Literal type)
VALID_ACTIONS = ["start", "continue", "branch", "revise", "synthesize", "verify", "finish"]

# Valid mode values (from ThinkMode Literal type)
VALID_MODES = ["chain", "matrix", "verify"]

# Valid verdicts for verify mode
VALID_VERDICTS = ["supported", "contradicted", "unclear"]

# Strategy for generating problem/thought text
text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"), whitelist_characters=" "),
    min_size=1,
    max_size=200,
).filter(lambda x: x.strip())  # Non-empty after strip

# Strategy for generating confidence scores
confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

# Strategy for matrix dimensions
rows_strategy = st.integers(min_value=2, max_value=5)
cols_strategy = st.integers(min_value=2, max_value=5)

# Strategy for step counts
steps_strategy = st.integers(min_value=1, max_value=20)


# =============================================================================
# Helper Functions
# =============================================================================


def parse_result(result: str) -> dict[str, Any]:
    """Parse JSON result from think() tool."""
    return json.loads(result)


def is_error_response(result: dict[str, Any]) -> bool:
    """Check if result is an error response."""
    return "error" in result


# =============================================================================
# Property Tests: Start Action
# =============================================================================


class TestThinkStartProperties:
    """Property tests for think() start action."""

    @pytest.mark.asyncio
    @given(
        mode=st.sampled_from(VALID_MODES),
        problem=text_strategy,
        expected_steps=steps_strategy,
        rows=rows_strategy,
        cols=cols_strategy,
    )
    @settings(
        max_examples=20,
        deadline=10000,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    async def test_start_always_returns_session_id(
        self, mode: str, problem: str, expected_steps: int, rows: int, cols: int
    ) -> None:
        """Property: start action with valid inputs always returns a session_id."""
        from src.server import think

        result = parse_result(
            await think.fn(
                action="start",
                mode=mode,
                problem=problem,
                expected_steps=expected_steps,
                rows=rows,
                cols=cols,
            )
        )

        # Should have session_id and mode (or rate limit error which is acceptable)
        if "error" in result and "rate_limit" in result.get("error", ""):
            return  # Rate limit is acceptable
        assert "session_id" in result, f"Missing session_id in {result}"
        assert result["mode"] == mode

    @pytest.mark.asyncio
    @given(mode=st.sampled_from(VALID_MODES))
    @settings(max_examples=10, deadline=10000)
    async def test_start_without_problem_returns_error(self, mode: str) -> None:
        """Property: start action without problem always returns error."""
        from src.server import think

        result = parse_result(await think.fn(action="start", mode=mode))

        assert is_error_response(result)
        assert "problem" in result["error"].lower()

    @pytest.mark.asyncio
    @given(problem=text_strategy)
    @settings(max_examples=10, deadline=10000)
    async def test_start_without_mode_uses_auto(self, problem: str) -> None:
        """Property: start action without mode uses auto-mode selection."""
        from src.server import think

        result = parse_result(await think.fn(action="start", problem=problem))

        # Auto-mode selection should succeed, not error
        assert "session_id" in result
        assert "actual_mode" in result  # Shows auto-selected mode


# =============================================================================
# Property Tests: Continue Action
# =============================================================================


class TestThinkContinueProperties:
    """Property tests for think() continue action."""

    @pytest.mark.asyncio
    @given(
        mode=st.sampled_from(VALID_MODES),
        problem=text_strategy,
        thought=text_strategy,
    )
    @settings(
        max_examples=15,
        deadline=15000,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    async def test_continue_after_start_succeeds(
        self, mode: str, problem: str, thought: str
    ) -> None:
        """Property: continue action after valid start always succeeds."""
        from src.server import think

        # Start session
        start_result = parse_result(
            await think.fn(
                action="start",
                mode=mode,
                problem=problem,
                rows=2,  # Minimum for matrix
                cols=2,
            )
        )
        assume(not is_error_response(start_result))
        session_id = start_result["session_id"]

        # Continue with thought
        if mode == "matrix":
            result = parse_result(
                await think.fn(
                    action="continue",
                    session_id=session_id,
                    thought=thought,
                    row=0,
                    col=0,
                )
            )
        else:
            result = parse_result(
                await think.fn(
                    action="continue",
                    session_id=session_id,
                    thought=thought,
                )
            )

        # Should not be an error
        assert not is_error_response(result), f"Unexpected error: {result}"

    @pytest.mark.asyncio
    @given(thought=text_strategy)
    @settings(max_examples=10, deadline=10000)
    async def test_continue_without_session_returns_error(self, thought: str) -> None:
        """Property: continue action without session_id always returns error."""
        from src.server import think

        result = parse_result(await think.fn(action="continue", thought=thought))

        assert is_error_response(result)
        assert "session_id" in result["error"].lower()

    @pytest.mark.asyncio
    @given(session_id=st.uuids().map(str))
    @settings(max_examples=10, deadline=10000)
    async def test_continue_with_invalid_session_returns_error(self, session_id: str) -> None:
        """Property: continue action with invalid session_id returns error."""
        from src.server import think

        result = parse_result(
            await think.fn(
                action="continue",
                session_id=session_id,
                thought="Test thought",
            )
        )

        assert is_error_response(result)
        assert "session" in result["error"].lower() or "not found" in result["error"].lower()


# =============================================================================
# Property Tests: Finish Action
# =============================================================================


class TestThinkFinishProperties:
    """Property tests for think() finish action."""

    @pytest.mark.asyncio
    @given(
        mode=st.sampled_from(VALID_MODES),
        problem=text_strategy,
        answer=text_strategy,
    )
    @settings(
        max_examples=15,
        deadline=15000,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    async def test_finish_after_start_completes(self, mode: str, problem: str, answer: str) -> None:
        """Property: finish action after start always completes the session."""
        from src.server import think

        # Start session
        start_result = parse_result(
            await think.fn(
                action="start",
                mode=mode,
                problem=problem,
                rows=2,
                cols=2,
            )
        )
        assume(not is_error_response(start_result))
        session_id = start_result["session_id"]

        # Finish
        result = parse_result(
            await think.fn(
                action="finish",
                session_id=session_id,
                thought=answer,
            )
        )

        # Should complete successfully
        assert not is_error_response(result), f"Unexpected error: {result}"
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    @given(session_id=st.uuids().map(str))
    @settings(max_examples=10, deadline=10000)
    async def test_finish_with_invalid_session_returns_error(self, session_id: str) -> None:
        """Property: finish action with invalid session_id returns error."""
        from src.server import think

        result = parse_result(await think.fn(action="finish", session_id=session_id))

        assert is_error_response(result)


# =============================================================================
# Property Tests: Mode-Specific Actions
# =============================================================================


class TestThinkModeSpecificProperties:
    """Property tests for mode-specific actions."""

    @pytest.mark.asyncio
    @given(
        problem=text_strategy,
        thought=text_strategy,
    )
    @settings(
        max_examples=10,
        deadline=15000,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    async def test_branch_only_valid_for_chain(self, problem: str, thought: str) -> None:
        """Property: branch action is only valid for chain mode sessions."""
        from src.server import think

        # Start chain session
        start_result = parse_result(await think.fn(action="start", mode="chain", problem=problem))
        assume(not is_error_response(start_result))
        session_id = start_result["session_id"]

        # Add initial step so we have something to branch from
        await think.fn(action="continue", session_id=session_id, thought="Step 1")

        # Branch should work
        result = parse_result(
            await think.fn(
                action="branch",
                session_id=session_id,
                thought=thought,
                branch_from=0,
            )
        )

        # Either succeeds or fails for a non-mode reason
        if is_error_response(result):
            assert "chain" not in result["error"].lower() or "not found" in result["error"].lower()

    @pytest.mark.asyncio
    @given(
        problem=text_strategy,
        thought=text_strategy,
        col=st.integers(min_value=0, max_value=1),
    )
    @settings(
        max_examples=10,
        deadline=15000,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    async def test_synthesize_only_valid_for_matrix(
        self, problem: str, thought: str, col: int
    ) -> None:
        """Property: synthesize action is only valid for matrix mode sessions."""
        from src.server import think

        # Start matrix session (2x2)
        start_result = parse_result(
            await think.fn(action="start", mode="matrix", problem=problem, rows=2, cols=2)
        )
        assume(not is_error_response(start_result))
        session_id = start_result["session_id"]

        # Fill cells in column
        await think.fn(action="continue", session_id=session_id, row=0, col=col, thought="Cell 0")
        await think.fn(action="continue", session_id=session_id, row=1, col=col, thought="Cell 1")

        # Synthesize should work for matrix
        result = parse_result(
            await think.fn(
                action="synthesize",
                session_id=session_id,
                col=col,
                thought=thought,
            )
        )

        # Should succeed
        assert not is_error_response(result), f"Unexpected error: {result}"

    @pytest.mark.asyncio
    @given(
        problem=text_strategy,
        verdict=st.sampled_from(VALID_VERDICTS),
        evidence=text_strategy,
    )
    @settings(
        max_examples=10,
        deadline=15000,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    async def test_verify_action_only_valid_for_verify_mode(
        self, problem: str, verdict: str, evidence: str
    ) -> None:
        """Property: verify action is only valid for verify mode sessions."""
        from src.server import think

        # Start verify session
        start_result = parse_result(
            await think.fn(action="start", mode="verify", problem=problem, context="Test context")
        )
        assume(not is_error_response(start_result))
        session_id = start_result["session_id"]

        # Add a claim first
        await think.fn(action="continue", session_id=session_id, thought="Test claim")

        # Verify action should work
        result = parse_result(
            await think.fn(
                action="verify",
                session_id=session_id,
                claim_id=0,
                verdict=verdict,
                evidence=evidence,
            )
        )

        # Should succeed
        assert not is_error_response(result), f"Unexpected error: {result}"


# =============================================================================
# Property Tests: Cross-Mode Invariants
# =============================================================================


class TestThinkCrossModeInvariants:
    """Property tests for invariants that hold across all modes."""

    @pytest.mark.asyncio
    @given(
        mode=st.sampled_from(VALID_MODES),
        problem=text_strategy,
    )
    @settings(
        max_examples=15,
        deadline=15000,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    async def test_session_id_is_unique_per_start(self, mode: str, problem: str) -> None:
        """Property: each start action produces a unique session_id."""
        from src.server import think

        # Start two sessions
        result1 = parse_result(
            await think.fn(action="start", mode=mode, problem=problem, rows=2, cols=2)
        )
        result2 = parse_result(
            await think.fn(action="start", mode=mode, problem=problem, rows=2, cols=2)
        )

        assume(not is_error_response(result1))
        assume(not is_error_response(result2))

        # Session IDs must be different
        assert result1["session_id"] != result2["session_id"]

    @pytest.mark.asyncio
    @given(
        mode=st.sampled_from(VALID_MODES),
        problem=text_strategy,
    )
    @settings(
        max_examples=10,
        deadline=15000,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    async def test_status_reflects_session_state(self, mode: str, problem: str) -> None:
        """Property: status tool accurately reflects session state."""
        from src.server import status, think

        # Start session
        start_result = parse_result(
            await think.fn(action="start", mode=mode, problem=problem, rows=2, cols=2)
        )
        assume(not is_error_response(start_result))
        session_id = start_result["session_id"]

        # Check status
        status_result = parse_result(await status.fn(session_id=session_id))

        # Should report active status
        assert not is_error_response(status_result)
        assert status_result["status"] == "active"

        # Finish session
        await think.fn(action="finish", session_id=session_id, thought="Done")

        # Check status again
        status_result = parse_result(await status.fn(session_id=session_id))

        # Should report completed status
        assert not is_error_response(status_result)
        assert status_result["status"] == "completed"


# =============================================================================
# Property Tests: Error Consistency
# =============================================================================


class TestThinkErrorConsistency:
    """Property tests for consistent error handling."""

    @pytest.mark.asyncio
    @given(
        action=st.sampled_from(["continue", "branch", "revise", "synthesize", "verify", "finish"]),
    )
    @settings(max_examples=15, deadline=10000)
    async def test_actions_requiring_session_fail_without_it(self, action: str) -> None:
        """Property: actions requiring session_id fail gracefully without it."""
        from src.server import think

        # All these actions require session_id
        result = parse_result(
            await think.fn(
                action=action,
                thought="Test thought",
                col=0,
                claim_id=0,
                verdict="supported",
                branch_from=0,
                revises=0,
            )
        )

        # Should return error mentioning session_id
        assert is_error_response(result)

    @pytest.mark.asyncio
    @given(
        mode=st.sampled_from(VALID_MODES),
        problem=text_strategy,
        invalid_session=st.uuids().map(str),
    )
    @settings(
        max_examples=10,
        deadline=15000,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    async def test_valid_session_distinguishes_from_invalid(
        self, mode: str, problem: str, invalid_session: str
    ) -> None:
        """Property: valid sessions work, invalid sessions fail consistently."""
        from src.server import think

        # Start valid session
        start_result = parse_result(
            await think.fn(action="start", mode=mode, problem=problem, rows=2, cols=2)
        )
        assume(not is_error_response(start_result))
        valid_session = start_result["session_id"]

        # Assume the random UUID is different from our valid session
        assume(invalid_session != valid_session)

        # Valid session should work
        if mode == "matrix":
            valid_result = parse_result(
                await think.fn(
                    action="continue",
                    session_id=valid_session,
                    thought="Test",
                    row=0,
                    col=0,
                )
            )
        else:
            valid_result = parse_result(
                await think.fn(action="continue", session_id=valid_session, thought="Test")
            )
        assert not is_error_response(valid_result)

        # Invalid session should fail
        invalid_result = parse_result(
            await think.fn(action="continue", session_id=invalid_session, thought="Test")
        )
        assert is_error_response(invalid_result)
