"""Type definitions for Atomic Reasoning Router.

This module contains all Pydantic models and enums used by the routing system.
Separated from logic for clean imports and testability.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """Types of reasoning steps in the state machine."""

    PREMISE = "premise"
    HYPOTHESIS = "hypothesis"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"


class RouterStatus(str, Enum):
    """Status returned after submitting a step."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    BRANCH_REQUIRED = "BRANCH_REQUIRED"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"


class Complexity(str, Enum):
    """Problem complexity levels with associated constraints."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class ComplexityConfig(BaseModel):
    """Configuration for a complexity level."""

    min_steps: int = Field(ge=1, description="Minimum steps before synthesis allowed")
    max_steps: int = Field(ge=1, description="Maximum steps before forced synthesis")
    confidence_threshold: float = Field(
        ge=0.0, le=1.0, description="Minimum confidence to avoid branching"
    )


# Complexity level configurations
COMPLEXITY_CONFIGS: dict[Complexity, ComplexityConfig] = {
    Complexity.LOW: ComplexityConfig(min_steps=2, max_steps=5, confidence_threshold=0.60),
    Complexity.MEDIUM: ComplexityConfig(min_steps=4, max_steps=8, confidence_threshold=0.70),
    Complexity.HIGH: ComplexityConfig(min_steps=6, max_steps=12, confidence_threshold=0.75),
}


class RouterStep(BaseModel):
    """A single reasoning step in a session."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    step_type: StepType
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    branch_id: str | None = None
    created_at: float = Field(default_factory=time.time)


class Branch(BaseModel):
    """An alternative reasoning branch."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    hypothesis: str
    steps: list[RouterStep] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    active: bool = True


class RouterSession(BaseModel):
    """A complete reasoning session with state tracking."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    problem: str
    complexity: Complexity
    min_steps: int
    max_steps: int
    confidence_threshold: float
    steps: list[RouterStep] = Field(default_factory=list)
    branches: dict[str, Branch] = Field(default_factory=dict)
    verified_claims: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    guidance: str = ""  # Trap warnings from RAG

    @property
    def step_count(self) -> int:
        """Count of non-synthesis steps."""
        return len([s for s in self.steps if s.step_type != StepType.SYNTHESIS])

    @property
    def has_verification(self) -> bool:
        """Whether session has at least one verification step."""
        return any(s.step_type == StepType.VERIFICATION for s in self.steps)

    @property
    def last_step_type(self) -> StepType | None:
        """Type of the last step, if any."""
        return self.steps[-1].step_type if self.steps else None


class Contradiction(BaseModel):
    """A detected contradiction between claims."""

    claim1: str
    claim2: str
    explanation: str


# --- Request/Response Models ---


class InitializeRequest(BaseModel):
    """Request to initialize a reasoning session."""

    problem: str = Field(min_length=1, description="The problem to reason about")
    complexity: Literal["low", "medium", "high", "auto"] = Field(
        default="auto", description="Problem complexity level"
    )


class InitializeResponse(BaseModel):
    """Response from initializing a session."""

    session_id: str
    complexity: str
    min_steps: int
    max_steps: int
    confidence_threshold: float
    guidance: str  # Trap warnings, reasoning hints


class StepRequest(BaseModel):
    """Request to submit a reasoning step."""

    session_id: str
    step_type: Literal["premise", "hypothesis", "verification", "synthesis"]
    content: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class StepResponse(BaseModel):
    """Response from submitting a step."""

    status: RouterStatus
    step_id: str | None = None
    rejection_reason: str | None = None
    valid_next_steps: list[str]
    steps_taken: int
    steps_remaining: int
    can_synthesize: bool
    synthesis_blockers: list[str] = Field(default_factory=list)


class BranchRequest(BaseModel):
    """Request to create alternative branches."""

    session_id: str
    alternatives: list[str] = Field(
        min_length=2, max_length=4, description="Alternative hypotheses to explore"
    )


class BranchResponse(BaseModel):
    """Response from creating branches."""

    branch_ids: list[str]
    guidance: str


class VerifyRequest(BaseModel):
    """Request to verify claims."""

    session_id: str
    claims: list[str] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)


class VerifyResponse(BaseModel):
    """Response from verifying claims."""

    verified: list[str]
    contradictions: list[Contradiction]
    missing_evidence: list[str]
    can_synthesize: bool
    synthesis_blockers: list[str] = Field(default_factory=list)


# --- State Machine Transitions ---

VALID_TRANSITIONS: dict[StepType | None, set[StepType]] = {
    None: {StepType.PREMISE},  # First step must be premise
    StepType.PREMISE: {StepType.PREMISE, StepType.HYPOTHESIS},
    StepType.HYPOTHESIS: {StepType.HYPOTHESIS, StepType.VERIFICATION},
    StepType.VERIFICATION: {StepType.HYPOTHESIS, StepType.SYNTHESIS},
    StepType.SYNTHESIS: set(),  # Terminal state
}
