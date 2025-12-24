"""Reasoning types and data structures.

Extracted from unified_reasoner.py to reduce module complexity and improve testability.
This module contains all enums, dataclasses, and type definitions used in reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

# =============================================================================
# Enums
# =============================================================================


class ReasoningMode(str, Enum):
    """Available reasoning modes."""

    CHAIN = "chain"  # Sequential chain-of-thought
    MATRIX = "matrix"  # Multi-perspective matrix
    HYBRID = "hybrid"  # Adaptive chain -> matrix escalation
    AUTO = "auto"  # Auto-select based on complexity


class SessionStatus(str, Enum):
    """Status of a reasoning session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"  # Chain escalated to matrix


class ThoughtType(str, Enum):
    """Type of thought/step in the reasoning process."""

    INITIAL = "initial"
    CONTINUATION = "continuation"
    REVISION = "revision"
    BRANCH = "branch"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    FINAL = "final"
    PLANNING = "planning"  # MPPA: Decision point
    EXECUTION = "execution"  # MPPA: Following a plan
    BLIND_SPOT = "blind_spot"  # Detected gap in reasoning


class DomainType(str, Enum):
    """Problem domain types for specialized handling."""

    GENERAL = "general"
    MATH = "math"
    CODE = "code"
    LOGIC = "logic"
    FACTUAL = "factual"


class ResponseVerbosity(str, Enum):
    """Response verbosity levels for token optimization.

    Research shows LLMs don't need verbose metadata for effective reasoning.
    Minimal responses reduce tokens by ~50% without quality loss.
    """

    MINIMAL = "minimal"  # Essential fields only: session_id, thought_id, step, survival_score
    NORMAL = "normal"  # Default: includes guidance and detected issues
    FULL = "full"  # All fields including debugging info


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Thought:
    """A single thought/step in the reasoning process."""

    id: str
    content: str
    thought_type: ThoughtType
    step_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float | None = None
    survival_score: float | None = None
    parent_id: str | None = None  # For graph structure
    branch_id: str | None = None
    revises_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Graph edges
    supports: list[str] = field(default_factory=list)  # IDs this thought supports
    contradicts: list[str] = field(default_factory=list)  # IDs this contradicts
    related_to: list[str] = field(default_factory=list)  # General relations
    # MPPA fields
    is_planning: bool = False
    alternatives_considered: int = 0
    # Vector store ID for RAG
    vector_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "type": self.thought_type.value,
            "step": self.step_number,
            "confidence": self.confidence,
            "survival_score": round(self.survival_score, 3) if self.survival_score else None,
            "parent_id": self.parent_id,
            "branch_id": self.branch_id,
            "is_planning": self.is_planning,
            "graph": {
                "supports": len(self.supports),
                "contradicts": len(self.contradicts),
                "related": len(self.related_to),
            },
        }


@dataclass
class BlindSpot:
    """A detected gap or blind spot in reasoning."""

    id: str
    description: str
    detected_at_step: int
    severity: Literal["low", "medium", "high"]
    suggested_action: str
    addressed: bool = False
    addressed_by: str | None = None  # Thought ID that addressed it

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "severity": self.severity,
            "suggested_action": self.suggested_action,
            "addressed": self.addressed,
            "addressed_by": self.addressed_by,
        }


@dataclass
class RewardSignal:
    """RLVR-style reward signal for learning."""

    step_id: str
    reward_type: Literal["consistency", "coherence", "correctness", "efficiency"]
    value: float  # -1 to 1
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SuggestionRecord:
    """Record of a suggestion made and its outcome.

    Used for learning which suggestions are accepted/rejected.
    """

    suggestion_id: str
    action: str
    urgency: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    outcome: Literal["pending", "accepted", "rejected", "auto_executed"] = "pending"
    outcome_timestamp: datetime | None = None
    # What action the user actually took (if different)
    actual_action: str | None = None


@dataclass
class SuggestionWeights:
    """Learned weights for suggestion prioritization.

    Weights are adjusted based on acceptance/rejection patterns.
    Higher weight = higher priority for that action type.
    """

    resolve: float = 1.0
    continue_blind_spot: float = 0.9
    verify: float = 0.8
    continue_depth: float = 0.7
    synthesize: float = 0.6
    finish: float = 0.5
    continue_default: float = 0.4

    def adjust(self, action: str, accepted: bool, learning_rate: float = 0.1) -> None:
        """Adjust weight based on acceptance/rejection."""
        attr_map = {
            "resolve": "resolve",
            "verify": "verify",
            "synthesize": "synthesize",
            "finish": "finish",
            "continue": "continue_default",
        }
        attr = attr_map.get(action)
        if attr and hasattr(self, attr):
            current = getattr(self, attr)
            delta = learning_rate if accepted else -learning_rate
            # Clamp between 0.1 and 2.0
            new_value = max(0.1, min(2.0, current + delta))
            setattr(self, attr, new_value)

    def get_weight(self, action: str, context: str = "") -> float:
        """Get weight for an action type."""
        if action == "resolve":
            return self.resolve
        elif action == "verify":
            return self.verify
        elif action == "synthesize":
            return self.synthesize
        elif action == "finish":
            return self.finish
        elif action == "continue":
            if "blind" in context.lower():
                return self.continue_blind_spot
            elif "depth" in context.lower():
                return self.continue_depth
            return self.continue_default
        return 0.5

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "resolve": round(self.resolve, 2),
            "continue_blind_spot": round(self.continue_blind_spot, 2),
            "verify": round(self.verify, 2),
            "continue_depth": round(self.continue_depth, 2),
            "synthesize": round(self.synthesize, 2),
            "finish": round(self.finish, 2),
            "continue_default": round(self.continue_default, 2),
        }


@dataclass
class ReasoningSession:
    """State of a unified reasoning session."""

    session_id: str
    problem: str
    context: str
    mode: ReasoningMode
    actual_mode: ReasoningMode  # May differ if AUTO selected
    status: SessionStatus
    domain: DomainType
    complexity: Any  # ComplexityResult - use Any to avoid circular import
    # Thoughts and structure
    thoughts: dict[str, Thought] = field(default_factory=dict)
    thought_order: list[str] = field(default_factory=list)  # Ordered list of thought IDs
    branches: dict[str, list[str]] = field(default_factory=dict)  # branch_id -> thought_ids
    # Matrix structure (if applicable)
    matrix_rows: int = 0
    matrix_cols: int = 0
    matrix_cells: dict[tuple[int, int], str] = field(
        default_factory=dict
    )  # (row,col) -> thought_id
    syntheses: dict[int, str] = field(default_factory=dict)  # col -> thought_id
    # Blind spots and rewards
    blind_spots: list[BlindSpot] = field(default_factory=list)
    rewards: list[RewardSignal] = field(default_factory=list)
    # Final state
    final_answer: str | None = None
    final_confidence: float | None = None
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    # RAG integration
    rag_enabled: bool = False
    rag_retrievals: int = 0
    # MPPA stats
    mppa_explorations: int = 0
    planning_steps: int = 0
    # Thought graph for relationship tracking (Any to avoid circular import)
    thought_graph: Any = field(default=None, repr=False)
    # Domain validation results
    domain_validations: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # thought_id -> validation
    # Graph analysis cache
    graph_contradictions: list[tuple[str, str]] = field(default_factory=list)
    graph_cycles: list[list[str]] = field(default_factory=list)
    # Suggestion history for learning (S2)
    suggestion_history: list[SuggestionRecord] = field(default_factory=list)
    suggestion_weights: SuggestionWeights = field(default_factory=SuggestionWeights)
    # Auto-action stats (S3)
    auto_actions_executed: int = 0
    auto_action_enabled: bool = False

    @property
    def current_step(self) -> int:
        """Get current step number."""
        return len(self.thought_order)

    @property
    def main_chain(self) -> list[Thought]:
        """Get main reasoning chain (excluding branches)."""
        return [
            self.thoughts[tid] for tid in self.thought_order if self.thoughts[tid].branch_id is None
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "session_id": self.session_id,
            "problem": self.problem[:100] + "..." if len(self.problem) > 100 else self.problem,
            "mode": self.mode.value,
            "actual_mode": self.actual_mode.value,
            "status": self.status.value,
            "domain": self.domain.value,
            "complexity": self.complexity.to_dict(),
            "thoughts": {tid: t.to_dict() for tid, t in self.thoughts.items()},
            "thought_count": len(self.thoughts),
            "current_step": self.current_step,
            "branches": {bid: len(tids) for bid, tids in self.branches.items()},
            "blind_spots": [bs.to_dict() for bs in self.blind_spots],
            "blind_spots_unaddressed": sum(1 for bs in self.blind_spots if not bs.addressed),
            "rewards_total": sum(r.value for r in self.rewards),
            "final_answer": self.final_answer,
            "final_confidence": self.final_confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "rag_enabled": self.rag_enabled,
            "rag_retrievals": self.rag_retrievals,
            "mppa_stats": {
                "explorations": self.mppa_explorations,
                "planning_steps": self.planning_steps,
            },
        }

        # Add graph analysis if available
        if self.thought_graph is not None:
            result["graph_analysis"] = {
                "node_count": len(self.thought_graph._nodes)
                if hasattr(self.thought_graph, "_nodes")
                else 0,
                "contradictions_found": len(self.graph_contradictions),
                "cycles_found": len(self.graph_cycles),
            }

        # Add domain validation summary
        if self.domain_validations:
            valid_count = sum(
                1 for v in self.domain_validations.values() if v.get("result") == "valid"
            )
            result["domain_validation_summary"] = {
                "validated_thoughts": len(self.domain_validations),
                "valid_count": valid_count,
                "validation_rate": round(valid_count / len(self.domain_validations), 2)
                if self.domain_validations
                else 0,
            }

        return result


@dataclass
class SessionAnalytics:
    """Consolidated analysis metrics for a reasoning session.

    This provides computed insights without duplicating raw session data.
    Use this for mid-session analysis or post-session review.
    """

    session_id: str
    # Progress metrics
    total_thoughts: int
    main_chain_length: int
    branch_count: int
    average_confidence: float | None
    average_survival_score: float | None
    # Quality metrics
    coherence_score: float  # 0-1: How well thoughts connect
    coverage_score: float  # 0-1: Problem space coverage estimate
    depth_score: float  # 0-1: Depth of reasoning (based on chain length vs complexity)
    # Issue tracking
    contradictions: list[tuple[str, str]]  # (thought_id1, thought_id2)
    unresolved_contradictions: int
    blind_spots_detected: int
    blind_spots_unaddressed: int
    cycles_detected: int
    # Domain validation
    validation_rate: float | None  # % of validated thoughts that passed
    invalid_thoughts: list[str]  # IDs of thoughts that failed validation
    # Efficiency metrics
    planning_ratio: float  # planning_steps / total_steps
    revision_count: int
    branch_utilization: float  # branches with synthesis / total branches
    # Recommendations
    recommendations: list[str]
    # Risk indicators
    risk_level: Literal["low", "medium", "high"]
    risk_factors: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "progress": {
                "total_thoughts": self.total_thoughts,
                "main_chain_length": self.main_chain_length,
                "branch_count": self.branch_count,
                "average_confidence": (
                    round(self.average_confidence, 3) if self.average_confidence else None
                ),
                "average_survival_score": (
                    round(self.average_survival_score, 3) if self.average_survival_score else None
                ),
            },
            "quality": {
                "coherence_score": round(self.coherence_score, 3),
                "coverage_score": round(self.coverage_score, 3),
                "depth_score": round(self.depth_score, 3),
                "overall": round(
                    (self.coherence_score + self.coverage_score + self.depth_score) / 3, 3
                ),
            },
            "issues": {
                "contradictions": len(self.contradictions),
                "unresolved_contradictions": self.unresolved_contradictions,
                "blind_spots_detected": self.blind_spots_detected,
                "blind_spots_unaddressed": self.blind_spots_unaddressed,
                "cycles_detected": self.cycles_detected,
            },
            "validation": {
                "rate": round(self.validation_rate, 3)
                if self.validation_rate is not None
                else None,
                "invalid_thought_count": len(self.invalid_thoughts),
            },
            "efficiency": {
                "planning_ratio": round(self.planning_ratio, 3),
                "revision_count": self.revision_count,
                "branch_utilization": round(self.branch_utilization, 3),
            },
            "recommendations": self.recommendations,
            "risk": {
                "level": self.risk_level,
                "factors": self.risk_factors,
            },
        }
