"""Long Chain-of-Thought state manager.

Implements a state management tool for tracking sequential reasoning chains.
The calling LLM does all reasoning; this tool tracks steps, validates structure,
supports branching/revision, and provides heuristic consistency checks.

Based on: "The Surprising Effectiveness of Sequential Thinking" (arXiv:2505.21825)

Architecture:
    - Tool receives reasoning steps FROM the calling LLM
    - Tracks chain state, branches, revisions
    - Provides heuristic verification (no internal LLM)
    - Returns state for next iteration
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger


class ChainStatus(Enum):
    """Status of a reasoning chain."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class StepType(Enum):
    """Type of reasoning step."""

    INITIAL = "initial"
    CONTINUATION = "continuation"
    REVISION = "revision"
    BRANCH = "branch"
    SYNTHESIS = "synthesis"
    FINAL = "final"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    step_number: int
    content: str
    step_type: StepType = StepType.CONTINUATION
    timestamp: datetime = field(default_factory=datetime.now)
    branch_id: str | None = None
    revises_step: int | None = None
    confidence: float | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "content": self.content,
            "step_type": self.step_type.value,
            "timestamp": self.timestamp.isoformat(),
            "branch_id": self.branch_id,
            "revises_step": self.revises_step,
            "confidence": self.confidence,
            "tags": self.tags,
        }


@dataclass
class ChainState:
    """State of a reasoning chain session."""

    session_id: str
    problem: str
    steps: list[ReasoningStep] = field(default_factory=list)
    branches: dict[str, list[ReasoningStep]] = field(default_factory=dict)
    status: ChainStatus = ChainStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expected_steps: int = 10
    final_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "problem": self.problem,
            "steps": [s.to_dict() for s in self.steps],
            "branches": {bid: [s.to_dict() for s in steps] for bid, steps in self.branches.items()},
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expected_steps": self.expected_steps,
            "final_answer": self.final_answer,
            "current_step": len(self.steps),
            "progress": len(self.steps) / max(self.expected_steps, 1),
        }


class LongChainManager:
    """Manages long chain-of-thought reasoning sessions.

    This is a STATE MANAGER, not a reasoner. The calling LLM provides
    reasoning content; this tool tracks and organizes the chain.

    Example usage flow:
        1. Agent calls: start_chain(problem="...")
        2. Agent reasons, then calls: add_step(thought="Step 1: ...")
        3. Agent continues: add_step(thought="Step 2: ...")
        4. Agent can branch: add_step(thought="Alternative...", branch_from=2)
        5. Agent can revise: add_step(thought="Actually...", revises=3)
        6. Agent finalizes: finalize(answer="The answer is...")

    """

    def __init__(self) -> None:
        """Initialize the chain manager."""
        self._sessions: dict[str, ChainState] = {}

    def start_chain(
        self,
        problem: str,
        expected_steps: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a new reasoning chain.

        Args:
            problem: The problem to reason about.
            expected_steps: Expected number of reasoning steps.
            metadata: Optional metadata (context, constraints, etc.)

        Returns:
            Session info with ID and initial state.

        """
        session_id = str(uuid.uuid4())[:8]
        state = ChainState(
            session_id=session_id,
            problem=problem,
            expected_steps=expected_steps,
            metadata=metadata or {},
        )
        self._sessions[session_id] = state

        logger.debug(f"Started chain session {session_id} for problem: {problem[:50]}...")

        return {
            "session_id": session_id,
            "status": "started",
            "problem": problem,
            "expected_steps": expected_steps,
            "next_step": 1,
            "instruction": (
                f"Begin reasoning about this problem. "
                f"Call add_step with your first reasoning step. "
                f"Aim for {expected_steps} steps of thorough analysis."
            ),
        }

    def add_step(
        self,
        session_id: str,
        thought: str,
        step_type: str = "continuation",
        branch_from: int | None = None,
        revises: int | None = None,
        confidence: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add a reasoning step to the chain.

        Args:
            session_id: Session to add step to.
            thought: The reasoning content (from calling LLM).
            step_type: Type of step (continuation, revision, branch, synthesis).
            branch_from: If branching, which step to branch from.
            revises: If revising, which step this revises.
            confidence: Optional confidence score for this step.
            tags: Optional tags for categorization.

        Returns:
            Updated state and guidance for next step.

        """
        state = self._get_session(session_id)
        if state.status != ChainStatus.ACTIVE:
            return {
                "error": f"Session {session_id} is {state.status.value}",
                "status": state.status.value,
            }

        # Determine step number
        if branch_from is not None:
            branch_id = f"branch_{len(state.branches) + 1}"
            step_num = 1
            stype = StepType.BRANCH
        elif revises is not None:
            step_num = len(state.steps) + 1
            stype = StepType.REVISION
            branch_id = None
        else:
            step_num = len(state.steps) + 1
            try:
                stype = StepType(step_type)
            except ValueError:
                stype = StepType.CONTINUATION
            branch_id = None

        # Create step
        step = ReasoningStep(
            step_number=step_num,
            content=thought,
            step_type=stype,
            branch_id=branch_id,
            revises_step=revises,
            confidence=confidence,
            tags=tags or [],
        )

        # Add to appropriate location
        if branch_id:
            state.branches[branch_id] = [step]
        else:
            state.steps.append(step)

        state.updated_at = datetime.now()

        # Run heuristic checks
        issues = self._check_step_consistency(state, step)

        # Determine next action
        progress = len(state.steps) / max(state.expected_steps, 1)
        needs_more = len(state.steps) < state.expected_steps

        response = {
            "session_id": session_id,
            "step_added": step_num,
            "step_type": stype.value,
            "total_steps": len(state.steps),
            "expected_steps": state.expected_steps,
            "progress": round(progress, 2),
            "branches": list(state.branches.keys()),
            "needs_more_steps": needs_more,
        }

        if issues:
            response["consistency_warnings"] = issues

        if needs_more:
            response["instruction"] = (
                f"Continue reasoning. You're at step {len(state.steps)}/{state.expected_steps}. "
                f"Add your next reasoning step, or call finalize() if you've reached a conclusion."
            )
        else:
            response["instruction"] = (
                "You've completed the expected steps. "
                "Call finalize() with your final answer, or add more steps if needed."
            )

        logger.debug(f"Added step {step_num} to session {session_id}")

        return response

    def get_chain(self, session_id: str) -> dict[str, Any]:
        """Get the current state of a reasoning chain.

        Args:
            session_id: Session to retrieve.

        Returns:
            Full chain state including all steps.

        """
        state = self._get_session(session_id)
        return state.to_dict()

    def finalize(
        self,
        session_id: str,
        answer: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Finalize the reasoning chain with an answer.

        Args:
            session_id: Session to finalize.
            answer: The final answer.
            confidence: Optional confidence in the answer.

        Returns:
            Final chain state with summary.

        """
        state = self._get_session(session_id)

        # Add final step
        final_step = ReasoningStep(
            step_number=len(state.steps) + 1,
            content=answer,
            step_type=StepType.FINAL,
            confidence=confidence,
        )
        state.steps.append(final_step)
        state.final_answer = answer
        state.status = ChainStatus.COMPLETED
        state.updated_at = datetime.now()

        # Generate summary
        summary = self._generate_summary(state)

        logger.info(f"Finalized session {session_id} with {len(state.steps)} steps")

        return {
            "session_id": session_id,
            "status": "completed",
            "total_steps": len(state.steps),
            "branches_explored": len(state.branches),
            "final_answer": answer,
            "confidence": confidence,
            "summary": summary,
            "chain": state.to_dict(),
        }

    def adjust_expected_steps(
        self,
        session_id: str,
        new_expected: int,
    ) -> dict[str, Any]:
        """Adjust the expected number of steps.

        Args:
            session_id: Session to adjust.
            new_expected: New expected step count.

        Returns:
            Updated state.

        """
        state = self._get_session(session_id)
        old_expected = state.expected_steps
        state.expected_steps = new_expected
        state.updated_at = datetime.now()

        return {
            "session_id": session_id,
            "old_expected": old_expected,
            "new_expected": new_expected,
            "current_steps": len(state.steps),
            "progress": round(len(state.steps) / max(new_expected, 1), 2),
        }

    def abandon(self, session_id: str, reason: str = "") -> dict[str, Any]:
        """Abandon a reasoning chain.

        Args:
            session_id: Session to abandon.
            reason: Optional reason for abandoning.

        Returns:
            Final state.

        """
        state = self._get_session(session_id)
        state.status = ChainStatus.ABANDONED
        state.metadata["abandon_reason"] = reason
        state.updated_at = datetime.now()

        return {
            "session_id": session_id,
            "status": "abandoned",
            "reason": reason,
            "steps_completed": len(state.steps),
        }

    def list_sessions(self) -> dict[str, Any]:
        """List all active sessions.

        Returns:
            List of session summaries.

        """
        sessions = []
        for sid, state in self._sessions.items():
            sessions.append(
                {
                    "session_id": sid,
                    "problem": state.problem[:50] + "..."
                    if len(state.problem) > 50
                    else state.problem,
                    "status": state.status.value,
                    "steps": len(state.steps),
                    "expected": state.expected_steps,
                    "created": state.created_at.isoformat(),
                }
            )
        return {"sessions": sessions, "total": len(sessions)}

    def _get_session(self, session_id: str) -> ChainState:
        """Get session or raise error."""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")
        return self._sessions[session_id]

    def _check_step_consistency(
        self,
        state: ChainState,
        step: ReasoningStep,
    ) -> list[str]:
        """Check step for consistency issues (heuristic, no LLM).

        Args:
            state: Current chain state.
            step: New step to check.

        Returns:
            List of warning messages.

        """
        issues = []
        content = step.content.lower()

        # Check for contradiction indicators
        contradiction_phrases = [
            "actually, that's wrong",
            "i was mistaken",
            "that contradicts",
            "this is incorrect",
            "wait, no",
        ]
        for phrase in contradiction_phrases:
            if phrase in content:
                issues.append(
                    f"Potential self-contradiction detected: '{phrase}'. "
                    "Consider using revises parameter to properly track the revision."
                )
                break

        # Check for very short steps
        if len(step.content.split()) < 10:
            issues.append("Step seems brief. Consider adding more detailed reasoning.")

        # Check for repetition with previous step
        if len(state.steps) > 0:
            prev_content = state.steps[-1].content.lower()
            # Simple word overlap check
            prev_words = set(prev_content.split())
            curr_words = set(content.split())
            if len(prev_words) > 5 and len(curr_words) > 5:
                overlap = len(prev_words & curr_words) / len(prev_words | curr_words)
                if overlap > 0.7:
                    issues.append(
                        "High similarity with previous step. "
                        "Ensure you're making progress rather than repeating."
                    )

        # Check for conclusion without enough steps
        conclusion_phrases = ["therefore", "in conclusion", "the answer is", "finally"]
        if (
            any(phrase in content for phrase in conclusion_phrases)
            and len(state.steps) < state.expected_steps * 0.5
        ):
            issues.append(
                f"Conclusion reached at step {len(state.steps)}/{state.expected_steps}. "
                "Consider more thorough analysis before concluding."
            )

        # Check for numeric consistency
        numbers_in_step = re.findall(r"\b\d+(?:\.\d+)?\b", step.content)
        if len(state.steps) > 0 and numbers_in_step:
            # Check if numbers appear in previous steps
            all_prev_numbers = set()
            for prev_step in state.steps[-3:]:  # Check last 3 steps
                all_prev_numbers.update(re.findall(r"\b\d+(?:\.\d+)?\b", prev_step.content))

            new_numbers = set(numbers_in_step) - all_prev_numbers
            if new_numbers and len(new_numbers) > 3:
                issues.append(
                    f"Multiple new numbers introduced ({len(new_numbers)}). "
                    "Verify calculations are building on previous steps."
                )

        return issues

    def _generate_summary(self, state: ChainState) -> dict[str, Any]:
        """Generate a summary of the reasoning chain.

        Args:
            state: Completed chain state.

        Returns:
            Summary statistics.

        """
        step_types: dict[str, int] = {}
        for step in state.steps:
            step_types[step.step_type.value] = step_types.get(step.step_type.value, 0) + 1

        total_words = sum(len(s.content.split()) for s in state.steps)

        return {
            "total_steps": len(state.steps),
            "step_types": step_types,
            "total_words": total_words,
            "avg_words_per_step": round(total_words / max(len(state.steps), 1), 1),
            "branches_created": len(state.branches),
            "revisions_made": step_types.get("revision", 0),
            "duration_seconds": (state.updated_at - state.created_at).total_seconds(),
        }


# Global instance for session persistence across tool calls
_chain_manager = LongChainManager()


def get_chain_manager() -> LongChainManager:
    """Get the global chain manager instance."""
    return _chain_manager
