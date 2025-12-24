# Atomic Reasoning Router - Implementation Plan

## Problem Statement

Reason Guard's current "guidance" approach doesn't help frontier models—they already reason well. The architecture suggests/guides but never **enforces** or **rejects**. Strong models need **constraints**, not suggestions.

**Current Results:**
- GSM8K: 93.3% baseline, 93.3% Reason Guard (no improvement)
- Reasoning traps: 60% baseline, 65% Reason Guard (marginal)

**Target Results:**
- Reasoning traps: 75-85% (significant improvement on problems where baseline fails)
- Rejection rate: 20-40% of steps rejected on first attempt (proves enforcement works)

---

## Architecture Overview

### Philosophy Shift

| Old (Guidance)                    | New (Enforcement)                        |
| --------------------------------- | ---------------------------------------- |
| Suggest reasoning strategies      | Reject premature conclusions             |
| Provide templates                 | Enforce minimum reasoning depth          |
| Score confidence passively        | Force branching on low confidence        |
| Allow any reasoning path          | State machine controls valid transitions |

### State Machine

```
PREMISE ──────► HYPOTHESIS ──────► VERIFICATION ──────► SYNTHESIS
    │               │                    │
    └───────────────┴────────────────────┘
         (can loop back for more depth)
```

**Valid Transitions:**
- `PREMISE → PREMISE` (add more premises)
- `PREMISE → HYPOTHESIS` (form hypothesis from premises)
- `HYPOTHESIS → HYPOTHESIS` (refine hypothesis)
- `HYPOTHESIS → VERIFICATION` (test hypothesis)
- `VERIFICATION → HYPOTHESIS` (hypothesis failed, try another)
- `VERIFICATION → SYNTHESIS` (verified, can conclude)
- `SYNTHESIS` is terminal

### 4 New Tools

#### 1. `initialize_reasoning`
```python
def initialize_reasoning(
    problem: str,
    complexity: Literal["low", "medium", "high", "auto"] = "auto"
) -> InitializeResult:
    """
    Start a new reasoning session with enforced constraints.

    Returns:
        session_id: str
        complexity: str (resolved if auto)
        min_steps: int
        max_steps: int
        confidence_threshold: float
        guidance: str (trap warnings from RAG)
    """
```

#### 2. `submit_atomic_step`
```python
def submit_atomic_step(
    session_id: str,
    step_type: Literal["premise", "hypothesis", "verification", "synthesis"],
    content: str,
    confidence: float,  # 0.0-1.0, required
    evidence: list[str] | None = None
) -> StepResult:
    """
    Submit a single reasoning step. May be REJECTED.

    Returns:
        status: Literal["ACCEPTED", "REJECTED", "BRANCH_REQUIRED", "VERIFICATION_REQUIRED"]
        step_id: str | None (if accepted)
        rejection_reason: str | None (if rejected)
        valid_next_steps: list[str]
        session_state: SessionState
    """
```

#### 3. `create_branch`
```python
def create_branch(
    session_id: str,
    alternatives: list[str]  # 2-4 alternative hypotheses
) -> BranchResult:
    """
    Create alternative reasoning branches (required when confidence < threshold).

    Returns:
        branch_ids: list[str]
        guidance: str (how to evaluate branches)
    """
```

#### 4. `verify_claims`
```python
def verify_claims(
    session_id: str,
    claims: list[str],
    evidence: list[str]
) -> VerifyResult:
    """
    Verify claims against evidence and check for contradictions.

    Returns:
        verified: list[str]
        contradictions: list[Contradiction]
        missing_evidence: list[str]
        can_synthesize: bool
    """
```

### 5 Routing Rules

| Rule | Name                  | Trigger                                      | Action                        |
| ---- | --------------------- | -------------------------------------------- | ----------------------------- |
| A    | Minimum Depth         | `step_type == "synthesis" and steps < min`   | REJECTED                      |
| B    | Confidence Branching  | `confidence < threshold`                     | BRANCH_REQUIRED               |
| C    | Verification Required | `step_type == "synthesis" and not verified`  | VERIFICATION_REQUIRED         |
| D    | State Machine         | Invalid transition                           | REJECTED                      |
| E    | Maximum Steps         | `steps >= max_steps`                         | Force SYNTHESIS (no rejection)|

### Complexity Thresholds

| Level  | min_steps | max_steps | confidence_threshold |
| ------ | --------- | --------- | -------------------- |
| low    | 2         | 5         | 0.60                 |
| medium | 4         | 8         | 0.70                 |
| high   | 6         | 12        | 0.75                 |

---

## File Structure

### New Files

```
src/tools/
├── routing_rules.py      # Pure functions for Rules A-E (no side effects)
├── atomic_router.py      # Tool implementations + session management
└── router_types.py       # Pydantic models for router

tests/
├── test_routing_rules.py # Unit tests for each rule
└── test_atomic_router.py # Integration tests for tools

examples/
└── benchmark_router.py   # New benchmark using router tools
```

### Modified Files

```
src/server.py             # Register 4 new tools, deprecate `think`
src/utils/schema.py       # Add router-specific types (or use router_types.py)
```

### Kept (Reused)

```
src/models/knowledge_graph.py  # Contradiction detection in verify_claims
src/models/vector_store.py     # RAG for trap warnings in initialize_reasoning
src/tools/compress.py          # Unchanged
```

### Deprecated

```
src/tools/unified_reasoner.py  # Replaced by atomic_router.py
```

---

## Implementation Schedule

### Week 1: Core Router

#### Day 1-2: Types & Rules
- [ ] Create `src/tools/router_types.py` with Pydantic models
- [ ] Create `src/tools/routing_rules.py` with pure functions
- [ ] Create `tests/test_routing_rules.py` with unit tests
- [ ] Target: 100% test coverage on routing rules

#### Day 3-4: State Machine & Session Management
- [ ] Implement state machine logic in `routing_rules.py`
- [ ] Add session management (in-memory dict with TTL)
- [ ] Create `src/tools/atomic_router.py` with tool stubs
- [ ] Integration tests for session lifecycle

#### Day 5: Tool Implementation
- [ ] Implement `initialize_reasoning` with RAG integration
- [ ] Implement `submit_atomic_step` with all 5 rules
- [ ] Implement `create_branch`
- [ ] Implement `verify_claims` with knowledge graph

### Week 2: Integration & Benchmarking

#### Day 1-2: Server Integration
- [ ] Register new tools in `src/server.py`
- [ ] Add deprecation warning to `think` tool
- [ ] Update `__init__.py` exports
- [ ] E2E tests via MCP protocol

#### Day 3-4: Benchmarking
- [ ] Create `examples/benchmark_router.py`
- [ ] Expand `examples/benchmark_problems.yaml` to 15 problems
- [ ] Run benchmarks, collect rejection rates
- [ ] Tune thresholds based on results

#### Day 5: Documentation & Cleanup
- [ ] Update README.md with new tools
- [ ] Update BENCHMARKS.md with results
- [ ] Create migration guide from `think` to new tools
- [ ] Remove deprecated code if benchmarks pass

---

## Detailed Specifications

### router_types.py

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal

class StepType(str, Enum):
    PREMISE = "premise"
    HYPOTHESIS = "hypothesis"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"

class RouterStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    BRANCH_REQUIRED = "BRANCH_REQUIRED"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"

class Complexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"

class RouterStep(BaseModel):
    id: str
    step_type: StepType
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    branch_id: str | None = None

class RouterSession(BaseModel):
    id: str
    problem: str
    complexity: Complexity
    min_steps: int
    max_steps: int
    confidence_threshold: float
    steps: list[RouterStep] = Field(default_factory=list)
    branches: dict[str, list[RouterStep]] = Field(default_factory=dict)
    verified_claims: list[str] = Field(default_factory=list)
    created_at: float

class ComplexityConfig(BaseModel):
    min_steps: int
    max_steps: int
    confidence_threshold: float

COMPLEXITY_CONFIGS: dict[Complexity, ComplexityConfig] = {
    Complexity.LOW: ComplexityConfig(min_steps=2, max_steps=5, confidence_threshold=0.60),
    Complexity.MEDIUM: ComplexityConfig(min_steps=4, max_steps=8, confidence_threshold=0.70),
    Complexity.HIGH: ComplexityConfig(min_steps=6, max_steps=12, confidence_threshold=0.75),
}
```

### routing_rules.py (Pure Functions)

```python
def check_minimum_depth(session: RouterSession, step_type: StepType) -> RouterStatus | None:
    """Rule A: Reject synthesis if insufficient steps."""
    if step_type == StepType.SYNTHESIS:
        non_synthesis_steps = [s for s in session.steps if s.step_type != StepType.SYNTHESIS]
        if len(non_synthesis_steps) < session.min_steps:
            return RouterStatus.REJECTED
    return None

def check_confidence_threshold(session: RouterSession, confidence: float) -> RouterStatus | None:
    """Rule B: Force branching on low confidence."""
    if confidence < session.confidence_threshold:
        return RouterStatus.BRANCH_REQUIRED
    return None

def check_verification_required(session: RouterSession, step_type: StepType) -> RouterStatus | None:
    """Rule C: Require verification before synthesis."""
    if step_type == StepType.SYNTHESIS:
        has_verification = any(s.step_type == StepType.VERIFICATION for s in session.steps)
        if not has_verification:
            return RouterStatus.VERIFICATION_REQUIRED
    return None

def check_valid_transition(session: RouterSession, step_type: StepType) -> RouterStatus | None:
    """Rule D: Enforce state machine transitions."""
    if not session.steps:
        # First step must be premise
        if step_type != StepType.PREMISE:
            return RouterStatus.REJECTED
        return None

    last_step = session.steps[-1]
    valid_transitions = get_valid_transitions(last_step.step_type)
    if step_type not in valid_transitions:
        return RouterStatus.REJECTED
    return None

def check_maximum_steps(session: RouterSession) -> bool:
    """Rule E: Check if max steps reached (forces synthesis)."""
    return len(session.steps) >= session.max_steps

def get_valid_transitions(current: StepType) -> set[StepType]:
    """Return valid next step types from current state."""
    transitions = {
        StepType.PREMISE: {StepType.PREMISE, StepType.HYPOTHESIS},
        StepType.HYPOTHESIS: {StepType.HYPOTHESIS, StepType.VERIFICATION},
        StepType.VERIFICATION: {StepType.HYPOTHESIS, StepType.SYNTHESIS},
        StepType.SYNTHESIS: set(),  # Terminal
    }
    return transitions[current]

def get_valid_next_steps(session: RouterSession) -> list[StepType]:
    """Get list of valid next step types for current session state."""
    if not session.steps:
        return [StepType.PREMISE]

    last_step = session.steps[-1]
    valid = get_valid_transitions(last_step.step_type)

    # Filter out synthesis if verification required
    has_verification = any(s.step_type == StepType.VERIFICATION for s in session.steps)
    if StepType.SYNTHESIS in valid and not has_verification:
        valid = valid - {StepType.SYNTHESIS}

    return list(valid)

def can_synthesize(session: RouterSession) -> tuple[bool, str | None]:
    """Check if session can proceed to synthesis."""
    non_synthesis_steps = [s for s in session.steps if s.step_type != StepType.SYNTHESIS]

    if len(non_synthesis_steps) < session.min_steps:
        return False, f"Need {session.min_steps - len(non_synthesis_steps)} more steps"

    has_verification = any(s.step_type == StepType.VERIFICATION for s in session.steps)
    if not has_verification:
        return False, "Verification step required before synthesis"

    return True, None
```

### Test Cases (test_routing_rules.py)

```python
# Rule A: Minimum Depth
def test_rule_a_rejects_early_synthesis():
    session = create_session(complexity="medium", steps=2)  # min=4
    result = check_minimum_depth(session, StepType.SYNTHESIS)
    assert result == RouterStatus.REJECTED

def test_rule_a_accepts_synthesis_after_min_steps():
    session = create_session(complexity="medium", steps=4)
    result = check_minimum_depth(session, StepType.SYNTHESIS)
    assert result is None

# Rule B: Confidence Branching
def test_rule_b_requires_branching_on_low_confidence():
    session = create_session(complexity="medium")  # threshold=0.70
    result = check_confidence_threshold(session, confidence=0.65)
    assert result == RouterStatus.BRANCH_REQUIRED

def test_rule_b_accepts_high_confidence():
    session = create_session(complexity="medium")
    result = check_confidence_threshold(session, confidence=0.75)
    assert result is None

# Rule C: Verification Required
def test_rule_c_requires_verification_before_synthesis():
    session = create_session(steps=[StepType.PREMISE, StepType.HYPOTHESIS])
    result = check_verification_required(session, StepType.SYNTHESIS)
    assert result == RouterStatus.VERIFICATION_REQUIRED

def test_rule_c_allows_synthesis_after_verification():
    session = create_session(steps=[StepType.PREMISE, StepType.HYPOTHESIS, StepType.VERIFICATION])
    result = check_verification_required(session, StepType.SYNTHESIS)
    assert result is None

# Rule D: State Machine
def test_rule_d_requires_premise_first():
    session = create_session(steps=[])
    result = check_valid_transition(session, StepType.HYPOTHESIS)
    assert result == RouterStatus.REJECTED

def test_rule_d_allows_premise_first():
    session = create_session(steps=[])
    result = check_valid_transition(session, StepType.PREMISE)
    assert result is None

def test_rule_d_rejects_invalid_transition():
    session = create_session(steps=[StepType.PREMISE])
    result = check_valid_transition(session, StepType.SYNTHESIS)  # Can't skip hypothesis
    assert result == RouterStatus.REJECTED

# Rule E: Maximum Steps
def test_rule_e_detects_max_steps():
    session = create_session(complexity="low", steps=5)  # max=5
    assert check_maximum_steps(session) is True

def test_rule_e_allows_more_steps():
    session = create_session(complexity="low", steps=3)
    assert check_maximum_steps(session) is False
```

---

## Benchmark Problems (Expanded)

### Current (5 problems)
1. Monty Hall
2. Base rate neglect
3. Simpson's paradox
4. Conjunction fallacy
5. Survivorship bias

### New (10 additional)
6. **Gambler's fallacy** - "After 10 heads, what's P(tails)?"
7. **Regression to mean** - "Tall parents, shorter children - genetic?"
8. **Selection bias** - "Hospital patients sicker than population"
9. **Anchoring** - "Is population > 50M? Exact guess?"
10. **Sunk cost** - "Already invested $1M, should we continue?"
11. **Availability heuristic** - "Plane vs car danger after crash news"
12. **Confirmation bias** - "Testing hypothesis by seeking only confirming evidence"
13. **Hot hand fallacy** - "Player made 5 shots, next one more likely?"
14. **Prosecutor's fallacy** - "1 in million DNA match = 1 in million innocent?"
15. **Ecological fallacy** - "Rich states vote Democrat, rich individuals?"

---

## Success Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Reasoning trap accuracy | 60-65% | 75-85% | 15 problems |
| Rejection rate | 0% | 20-40% | Proves enforcement works |
| Steps per problem | ~1 | 4-6 | Deeper reasoning |
| Branch creation rate | 0% | 10-20% | On low-confidence steps |
| Verification rate | 0% | 80-100% | Before synthesis |

---

## Open Questions

1. **Backwards compatibility:** Keep deprecated `think` tool with warning, or remove?
   - Recommendation: Keep with deprecation warning for 1 version

2. **Auto-complexity detection:** Keep or require explicit?
   - Recommendation: Keep auto-detect, validated by RAG trap matching

3. **Session storage:** In-memory vs Redis?
   - Recommendation: Start in-memory with TTL, add Redis later if needed

4. **Branch evaluation:** How does model choose winning branch?
   - Recommendation: Model submits all branches, highest confidence wins

---

## Next Steps (Immediate)

1. Create `src/tools/router_types.py` with Pydantic models
2. Create `src/tools/routing_rules.py` with pure functions
3. Create `tests/test_routing_rules.py` with unit tests
4. Run tests to validate rule logic

Start with: `router_types.py` → `routing_rules.py` → `test_routing_rules.py`
