"""Long Chain-of-Thought state manager with Multi-Path Plan Aggregation (MPPA).

Implements a state management tool for tracking sequential reasoning chains.
The calling LLM does all reasoning; this tool tracks steps, validates structure,
supports branching/revision, and provides heuristic consistency checks.

Enhanced with MPPA concepts from:
"Enhancing Long Chain-of-Thought Reasoning through Multi-Path Plan Aggregation"
(arXiv:2510.11620v2)

Key MPPA Features:
    - Planning step detection: Identifies decision points using indicator phrases
    - Multi-path exploration: At planning steps, evaluates multiple candidate paths
    - Survival probability estimation: Scores candidates by likely success
    - Variable interval scheduling: Explores more frequently early, less later

Architecture:
    - Tool receives reasoning steps FROM the calling LLM
    - Tracks chain state, branches, revisions
    - Detects planning vs execution steps
"""

from __future__ import annotations

import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from src.models.knowledge_graph import KnowledgeGraphExtractor
from src.utils.confidence import (
    CISCSelectionResult,
    ConfidenceMethod,
    cisc_select,
    combine_confidences,
)
from src.utils.scoring import calculate_survival_score
from src.utils.session import SessionManager


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
    PLANNING = "planning"  # MPPA: Decision point step
    EXECUTION = "execution"  # MPPA: Following a chosen plan


# =============================================================================
# MPPA: Planning Step Detection
# =============================================================================

# Indicator phrases that signal a planning/decision step (from MPPA paper)
PLANNING_INDICATORS: tuple[str, ...] = (
    "let me",
    "let's",
    "i'll",
    "i will",
    "i should",
    "i need to",
    "first,",
    "wait,",
    "alternatively",
    "maybe",
    "perhaps",
    "one approach",
    "another way",
    "we could",
    "we can",
    "the strategy",
    "my plan",
    "to solve this",
    "i think",
    "considering",
)


def is_planning_step(thought: str) -> bool:
    """Detect if a thought is a planning step (decision point).

    Planning steps are where errors compound most severely.
    MPPA focuses multi-path exploration on these critical junctures.

    Args:
        thought: The reasoning content to analyze.

    Returns:
        True if this appears to be a planning/decision step.

    """
    thought_lower = thought.lower().strip()
    return any(thought_lower.startswith(indicator) for indicator in PLANNING_INDICATORS)


def should_explore_alternatives(step_number: int, last_explore_step: int) -> bool:
    """Determine if we should explore alternatives at this step.

    Uses variable interval scheduling from MPPA paper:
    - Explore more frequently in early steps (where errors compound most)
    - Reduce frequency as chain progresses

    Args:
        step_number: Current step position.
        last_explore_step: Step number of last exploration.

    Returns:
        True if alternatives should be explored.

    """
    if step_number <= 3:
        interval = 1  # Explore every step early on
    elif step_number <= 7:
        interval = 2  # Every other step mid-chain
    else:
        interval = 3  # Less frequent late-chain

    return (step_number - last_explore_step) >= interval


@dataclass
class CandidateThought:
    """A candidate reasoning path for multi-path exploration.

    Enhanced with CISC confidence scoring:
    - survival_score: Heuristic score from text analysis
    - llm_confidence: Self-assessed confidence from calling LLM (0-1)
    - combined_score: Hybrid of LLM confidence + heuristic (weighted)
    """

    content: str
    survival_score: float
    selected: bool = False
    llm_confidence: float | None = None  # CISC: LLM self-assessed confidence
    combined_score: float | None = None  # CISC: Hybrid score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "survival_score": round(self.survival_score, 3),
            "selected": self.selected,
        }
        if self.llm_confidence is not None:
            result["llm_confidence"] = round(self.llm_confidence, 3)
        if self.combined_score is not None:
            result["combined_score"] = round(self.combined_score, 3)
        return result


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
    is_planning: bool = False  # MPPA: Was this detected as a planning step?
    alternatives_considered: list[CandidateThought] = field(default_factory=list)  # MPPA
    survival_score: float | None = None  # MPPA: Score if selected from alternatives

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "step_number": self.step_number,
            "content": self.content,
            "step_type": self.step_type.value,
            "timestamp": self.timestamp.isoformat(),
            "branch_id": self.branch_id,
            "revises_step": self.revises_step,
            "confidence": self.confidence,
            "tags": self.tags,
            "is_planning": self.is_planning,
        }
        if self.alternatives_considered:
            result["alternatives_considered"] = len(self.alternatives_considered)
        if self.survival_score is not None:
            result["survival_score"] = round(self.survival_score, 3)
        return result


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
    # MPPA fields
    last_explore_step: int = 0  # Last step where alternatives were explored
    total_explorations: int = 0  # Count of multi-path explorations
    planning_steps_detected: int = 0  # Count of planning steps found

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
            # MPPA stats
            "mppa_stats": {
                "planning_steps_detected": self.planning_steps_detected,
                "total_explorations": self.total_explorations,
                "last_explore_step": self.last_explore_step,
            },
        }


class LongChainManager(SessionManager[ChainState]):
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

    def __init__(
        self,
        encoder: Any | None = None,
        kg: Any | None = None,
    ) -> None:
        """Initialize the chain manager.

        Args:
            encoder: Optional ContextEncoder for semantic scoring.
            kg: Optional KnowledgeGraph for fact alignment scoring.

        """
        super().__init__()
        self._encoder = encoder
        self._kg = kg

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
        self._register_session(session_id, state)

        # Auto-extract entities from problem into shared KG
        if self._kg is not None:
            try:
                extractor = KnowledgeGraphExtractor()
                extractor.extract(problem, existing_graph=self._kg)
                # Also extract from context if provided in metadata
                context = (metadata or {}).get("context", "")
                if context:
                    extractor.extract(context, existing_graph=self._kg)
                logger.debug(f"Extracted {self._kg.stats()['entity_count']} entities from problem")
            except Exception as e:
                logger.warning(f"KG extraction failed (non-fatal): {e}")

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
        alternatives: list[str] | None = None,  # MPPA: Alternative candidate thoughts
        alternative_confidences: list[float]
        | None = None,  # CISC: LLM confidences for alternatives
    ) -> dict[str, Any]:
        """Add a reasoning step to the chain with CISC-enhanced multi-path exploration.

        MPPA Enhancement: When `alternatives` are provided, this method:
        1. Detects if current step is a planning step
        2. Scores all candidates (thought + alternatives) by survival probability
        3. Selects the highest-scoring candidate
        4. Records the exploration for analysis

        CISC Enhancement: When `alternative_confidences` are also provided:
        1. Combines LLM self-assessed confidence with heuristic scores
        2. Uses softmax normalization to amplify score differences
        3. Selects using weighted hybrid scoring (paper shows 11%+ improvement)

        Args:
            session_id: Session to add step to.
            thought: The reasoning content (from calling LLM).
            step_type: Type of step (continuation, revision, branch, synthesis).
            branch_from: If branching, which step to branch from.
            revises: If revising, which step this revises.
            confidence: Optional confidence score for this step.
            tags: Optional tags for categorization.
            alternatives: MPPA - Alternative reasoning paths to consider.
            alternative_confidences: CISC - LLM self-assessed confidence (0-1) for
                each alternative. If provided, uses hybrid CISC selection.

        Returns:
            Updated state and guidance for next step.

        """
        with self.session(session_id) as state:
            if state.status != ChainStatus.ACTIVE:
                return {
                    "error": f"Session {session_id} is {state.status.value}",
                    "status": state.status.value,
                }

            step_num = len(state.steps) + 1

        # MPPA: Detect if this is a planning step
        detected_planning = is_planning_step(thought)
        if detected_planning:
            state.planning_steps_detected += 1

        # MPPA: Multi-path exploration with CISC enhancement
        selected_thought = thought
        survival_score: float | None = None
        candidates: list[CandidateThought] = []
        exploration_performed = False
        cisc_result: CISCSelectionResult | None = None

        # Build context for scoring
        context = state.problem
        if state.steps:
            context += "\n\nPrevious steps:\n" + "\n".join(
                f"Step {s.step_number}: {s.content[:200]}" for s in state.steps[-3:]
            )

        if alternatives and len(alternatives) > 0:
            # Score all candidates including the primary thought
            all_thoughts = [thought] + alternatives

            # Get LLM confidences if provided
            all_llm_confidences: list[float | None] = [confidence]
            if alternative_confidences:
                all_llm_confidences.extend(alternative_confidences)
            else:
                all_llm_confidences.extend([None] * len(alternatives))

            # Pad if needed
            while len(all_llm_confidences) < len(all_thoughts):
                all_llm_confidences.append(None)

            # Build candidates with both heuristic and LLM scores
            for idx, t in enumerate(all_thoughts):
                heuristic_score = calculate_survival_score(
                    t, context, step_num, encoder=self._encoder, kg=self._kg
                )
                llm_conf = all_llm_confidences[idx]

                # CISC: Combine LLM confidence with heuristic if available
                if llm_conf is not None:
                    combined = combine_confidences(llm_conf, heuristic_score, llm_weight=0.7)
                else:
                    combined = heuristic_score

                candidates.append(
                    CandidateThought(
                        content=t,
                        survival_score=heuristic_score,
                        llm_confidence=llm_conf,
                        combined_score=combined,
                    )
                )

            # CISC selection: use combined scores with softmax normalization
            use_cisc = any(c.llm_confidence is not None for c in candidates)
            if use_cisc:
                # Use CISC weighted selection
                cisc_candidates = [
                    (c.content, c.combined_score or c.survival_score) for c in candidates
                ]
                cisc_result = cisc_select(
                    cisc_candidates,
                    method=ConfidenceMethod.HYBRID,
                    temperature=0.5,  # Per paper recommendations
                )
                best_idx = cisc_result.selected_index
                candidates[best_idx].selected = True
                selected_thought = candidates[best_idx].content
                survival_score = candidates[best_idx].combined_score

                logger.debug(
                    f"CISC selection at step {step_num}: "
                    f"selected idx={best_idx}, score={survival_score:.3f}, "
                    f"weights={[round(w, 3) for w in cisc_result.normalized_weights]}"
                )
            else:
                # Fallback to MPPA heuristic selection
                candidates.sort(
                    key=lambda c: (c.survival_score, random.random()),  # nosec B311
                    reverse=True,
                )
                best = candidates[0]
                best.selected = True
                selected_thought = best.content
                survival_score = best.survival_score

                logger.debug(
                    f"MPPA exploration at step {step_num}: "
                    f"selected score={survival_score:.3f} from {len(candidates)} candidates"
                )

            # Track exploration
            state.total_explorations += 1
            state.last_explore_step = step_num
            exploration_performed = True

        elif detected_planning and should_explore_alternatives(step_num, state.last_explore_step):
            # Planning step detected but no alternatives provided - suggest exploration
            survival_score = calculate_survival_score(
                thought, context, step_num, encoder=self._encoder, kg=self._kg
            )

        # Determine step number and type
        if branch_from is not None:
            branch_id = f"branch_{len(state.branches) + 1}"
            step_num = 1
            stype = StepType.BRANCH
        elif revises is not None:
            stype = StepType.REVISION
            branch_id = None
        else:
            try:
                stype = StepType(step_type)
            except ValueError:
                stype = StepType.PLANNING if detected_planning else StepType.CONTINUATION
            branch_id = None

        # Create step
        step = ReasoningStep(
            step_number=step_num,
            content=selected_thought,
            step_type=stype,
            branch_id=branch_id,
            revises_step=revises,
            confidence=confidence,
            tags=tags or [],
            is_planning=detected_planning,
            alternatives_considered=candidates if exploration_performed else [],
            survival_score=survival_score,
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

        response: dict[str, Any] = {
            "session_id": session_id,
            "step_added": step_num,
            "step_type": stype.value,
            "total_steps": len(state.steps),
            "expected_steps": state.expected_steps,
            "progress": round(progress, 2),
            "branches": list(state.branches.keys()),
            "needs_more_steps": needs_more,
        }

        # MPPA info
        if detected_planning:
            response["is_planning_step"] = True
        if exploration_performed:
            mppa_info: dict[str, Any] = {
                "candidates_evaluated": len(candidates),
                "selected_score": round(survival_score, 3) if survival_score else None,
                "alternatives_scores": [
                    round(c.survival_score, 3) for c in candidates if not c.selected
                ],
            }
            # CISC enhancement info
            if cisc_result is not None:
                mppa_info["cisc_enabled"] = True
                mppa_info["cisc_weights"] = [round(w, 3) for w in cisc_result.normalized_weights]
                mppa_info["cisc_temperature"] = cisc_result.temperature
                mppa_info["llm_confidences"] = [
                    round(c.llm_confidence, 3) if c.llm_confidence is not None else None
                    for c in candidates
                ]
            response["mppa_exploration"] = mppa_info
        elif detected_planning and alternatives is None:
            # Suggest providing alternatives for planning steps
            response["mppa_suggestion"] = (
                "Planning step detected. Consider providing 'alternatives' parameter "
                "with 1-3 alternative reasoning paths for better exploration."
            )
            if survival_score is not None:
                response["current_survival_score"] = round(survival_score, 3)

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
        with self.session(session_id) as state:
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
        with self.session(session_id) as state:
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

        # Generate summary (read-only, outside lock)
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
        with self.session(session_id) as state:
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
        with self.session(session_id) as state:
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
        with self.locked() as sessions:
            result = []
            for sid, state in sessions.items():
                result.append(
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
            return {"sessions": result, "total": len(result)}

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
