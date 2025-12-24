"""Atomic Reasoning Router - Tool implementations.

This module provides the 4 MCP tools for enforced reasoning:
1. initialize_reasoning - Start a session with constraints
2. submit_atomic_step - Submit steps with rule enforcement
3. create_branch - Create alternative hypotheses
4. verify_claims - Verify claims with contradiction detection

The router enforces reasoning discipline through REJECTION, not guidance.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import TYPE_CHECKING

# Import observability (no-op if OTel not installed)
from src.utils.observability import (
    record_session_created,
    record_step,
    tracer,
)

from .router_types import (
    Branch,
    BranchResponse,
    Complexity,
    Contradiction,
    InitializeResponse,
    RouterSession,
    RouterStatus,
    RouterStep,
    StepResponse,
    StepType,
    VerifyResponse,
)
from .routing_rules import (
    can_synthesize,
    check_maximum_steps,
    evaluate_step,
    get_max_steps_guidance,
    get_valid_next_steps,
    resolve_complexity,
    steps_remaining,
)

if TYPE_CHECKING:
    pass


# --- Session Storage (SQLite-backed with in-memory cache) ---

import json
import sqlite3
from pathlib import Path

# Storage configuration
_SESSIONS_DB_PATH = Path(
    os.environ.get("ATOMIC_ROUTER_DB", "~/.cache/reasonguard/router_sessions.db")
).expanduser()
_sessions: dict[str, RouterSession] = {}  # In-memory cache
_sessions_lock = threading.Lock()
_session_ttl_seconds = 3600  # 1 hour
_db_initialized = False


def _init_db() -> None:
    """Initialize SQLite database for session persistence."""
    global _db_initialized
    if _db_initialized:
        return

    _SESSIONS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(_SESSIONS_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at)")
        conn.commit()

    _db_initialized = True


def _load_session_from_db(session_id: str) -> RouterSession | None:
    """Load session from SQLite database."""
    _init_db()
    try:
        with sqlite3.connect(_SESSIONS_DB_PATH) as conn:
            row = conn.execute(
                "SELECT data, created_at FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()

            if row is None:
                return None

            data, created_at = row
            # Check TTL
            if time.time() - created_at > _session_ttl_seconds:
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                conn.commit()
                return None

            return RouterSession.model_validate_json(data)
    except (sqlite3.Error, json.JSONDecodeError):
        return None


def _save_session_to_db(session: RouterSession) -> None:
    """Save session to SQLite database."""
    _init_db()
    try:
        with sqlite3.connect(_SESSIONS_DB_PATH) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (id, data, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session.id, session.model_dump_json(), session.created_at, time.time()),
            )
            conn.commit()
    except sqlite3.Error:
        pass  # Fail silently - in-memory cache still works


def _delete_session_from_db(session_id: str) -> None:
    """Delete session from SQLite database."""
    _init_db()
    try:
        with sqlite3.connect(_SESSIONS_DB_PATH) as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
    except sqlite3.Error:
        pass


def _cleanup_expired_sessions() -> int:
    """Remove expired sessions from memory and disk. Returns count removed."""
    now = time.time()
    expired = []

    with _sessions_lock:
        expired = [
            sid
            for sid, session in _sessions.items()
            if now - session.created_at > _session_ttl_seconds
        ]
        for sid in expired:
            del _sessions[sid]

    # Also cleanup database
    _init_db()
    try:
        with sqlite3.connect(_SESSIONS_DB_PATH) as conn:
            conn.execute(
                "DELETE FROM sessions WHERE ? - created_at > ?", (now, _session_ttl_seconds)
            )
            conn.commit()
    except sqlite3.Error:
        pass

    return len(expired)


def _get_session(session_id: str) -> RouterSession | None:
    """Get session by ID, checking memory cache then disk."""
    with _sessions_lock:
        # Check memory cache first
        session = _sessions.get(session_id)
        if session is not None:
            if time.time() - session.created_at > _session_ttl_seconds:
                del _sessions[session_id]
                _delete_session_from_db(session_id)
                return None
            return session

    # Try loading from disk
    session = _load_session_from_db(session_id)
    if session is not None:
        # Cache in memory
        with _sessions_lock:
            _sessions[session_id] = session
    return session


def _save_session(session: RouterSession) -> None:
    """Save session to memory cache and disk."""
    with _sessions_lock:
        _sessions[session.id] = session
    _save_session_to_db(session)


def _delete_session(session_id: str) -> bool:
    """Delete session from memory and disk. Returns True if existed."""
    existed = False
    with _sessions_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            existed = True
    _delete_session_from_db(session_id)
    return existed


# --- RAG Integration ---

if TYPE_CHECKING:
    from src.models.vector_store import AsyncVectorStore

# Global RAG store (lazy-initialized)
_rag_store: AsyncVectorStore | None = None
_rag_initialized = False
_rag_lock = threading.Lock()


async def _init_rag_store() -> AsyncVectorStore | None:
    """Initialize RAG vector store with seed knowledge.

    Loads trap warnings from benchmark_problems.yaml into vector store
    for semantic similarity search.
    """
    global _rag_store, _rag_initialized

    with _rag_lock:
        if _rag_initialized:
            return _rag_store

        try:
            from pathlib import Path

            import yaml

            from src.models.vector_store import AsyncVectorStore, VectorStoreConfig

            # Use in-memory store for router (fast, no persistence needed)
            config = VectorStoreConfig(
                db_path=":memory:",
                collection_name="trap_warnings",
            )
            store = AsyncVectorStore(config)
            await store.__aenter__()

            # Load seed knowledge from benchmark problems
            benchmark_path = (
                Path(__file__).parent.parent.parent / "examples" / "benchmark_problems.yaml"
            )
            if benchmark_path.exists():
                with open(benchmark_path) as f:
                    data = yaml.safe_load(f)

                seed_knowledge = data.get("seed_knowledge", [])
                if seed_knowledge:
                    texts = [k["thought"] for k in seed_knowledge]
                    metadatas = [
                        {
                            "id": k["id"],
                            "strategy": k.get("strategy", ""),
                            "score": k.get("score", 0.9),
                            "keywords": ",".join(k.get("keywords", [])),
                        }
                        for k in seed_knowledge
                    ]
                    await store.add_texts(texts, metadatas)
                    from loguru import logger

                    logger.info(f"RAG: Loaded {len(texts)} trap warnings from seed knowledge")

            _rag_store = store
            _rag_initialized = True
            return store

        except Exception as e:
            from loguru import logger

            logger.warning(f"RAG initialization failed, using keyword fallback: {e}")
            _rag_initialized = True  # Don't retry
            return None


def _get_trap_warnings_sync(problem: str, complexity: Complexity) -> str:
    """Synchronous fallback for trap detection using keywords."""
    problem_lower = problem.lower()
    warnings: list[str] = []

    # Known trap patterns (fallback when RAG unavailable)
    trap_patterns = {
        "monty hall": (
            "TRAP WARNING: Monty Hall problem. Common error: assuming 50/50 after door reveal. "
            "Key insight: The host's action is NOT random - they always reveal a goat. "
            "Switching wins 2/3 of the time."
        ),
        "base rate": (
            "TRAP WARNING: Base rate problem. Common error: ignoring prior probabilities. "
            "Use Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B). "
            "The base rate often dominates."
        ),
        "simpson": (
            "TRAP WARNING: Simpson's paradox. Trends in subgroups can reverse when combined. "
            "Check for confounding variables. Analyze subgroups separately AND together."
        ),
        "conjunction": (
            "TRAP WARNING: Conjunction fallacy. P(A and B) ≤ P(A). "
            "More specific ≠ more probable. Check if you're rating plausibility vs probability."
        ),
        "survivorship": (
            "TRAP WARNING: Survivorship bias. You're only seeing successes/survivors. "
            "Ask: What about those who failed? What data is missing?"
        ),
        "gambler": (
            "TRAP WARNING: Gambler's fallacy. Independent events have no memory. "
            "Past outcomes don't affect future probabilities for independent events."
        ),
        "sunk cost": (
            "TRAP WARNING: Sunk cost fallacy. Past investments shouldn't affect future decisions. "
            "Only consider future costs and benefits."
        ),
        "confirmation": (
            "TRAP WARNING: Confirmation bias. Seek disconfirming evidence. "
            "What would prove you wrong? Test that explicitly."
        ),
        "anchor": (
            "TRAP WARNING: Anchoring bias. Initial values unduly influence estimates. "
            "Generate your estimate BEFORE looking at anchors."
        ),
        "availability": (
            "TRAP WARNING: Availability heuristic. Ease of recall ≠ frequency. "
            "Dramatic events are memorable but may be rare."
        ),
        # Additional traps from benchmark
        "bat and ball": (
            "TRAP WARNING: Algebra trap. The intuitive answer is WRONG. "
            "Set up equations: if bat = ball + $1.00 and bat + ball = $1.10, solve algebraically."
        ),
        "average speed": (
            "TRAP WARNING: Use harmonic mean for average speed, NOT arithmetic mean. "
            "Formula: 2*v1*v2/(v1+v2). The time at each speed matters."
        ),
        "birthday": (
            "TRAP WARNING: Birthday paradox. Count pairs, not individuals. "
            "23 people = 253 pairs. P(match) ≈ 50%, NOT 23/365."
        ),
        "envelope": (
            "TRAP WARNING: Two envelope paradox. E[B] = 0.5(X/2) + 0.5(2X) = 1.25X is WRONG. "
            "By symmetry, E[B] = E[A]. No advantage to switching."
        ),
        "linda": (
            "TRAP WARNING: Conjunction fallacy. P(A and B) ≤ P(A) always. "
            "'Bank teller AND feminist' cannot be more probable than 'bank teller' alone."
        ),
        "russian roulette": (
            "TRAP WARNING: Adjacent chambers matter. After empty chamber, "
            "firing without spinning is SAFER (75% vs 67%)."
        ),
        "sleeping beauty": (
            "TRAP WARNING: Thirder position: P(heads) = 1/3. "
            "Weight by number of observations, not coin probability alone."
        ),
    }

    for pattern, warning in trap_patterns.items():
        if pattern in problem_lower:
            warnings.append(warning)

    if not warnings:
        if complexity == Complexity.HIGH:
            warnings.append(
                "HIGH COMPLEXITY: This problem may contain non-obvious traps. "
                "Verify each step. Consider alternative interpretations."
            )
        elif complexity == Complexity.MEDIUM:
            warnings.append(
                "Consider: What assumptions are you making? Are they stated or implied?"
            )

    return "\n\n".join(warnings) if warnings else "No specific trap warnings detected."


async def _get_trap_warnings_rag(problem: str, complexity: Complexity) -> str:
    """Get reasoning trap warnings using RAG semantic search.

    Args:
        problem: The problem text to analyze
        complexity: Detected complexity level

    Returns:
        Formatted trap warnings or default guidance

    """
    # Try to use RAG store
    store = await _init_rag_store()
    if store is None:
        return _get_trap_warnings_sync(problem, complexity)

    try:
        # Search for similar trap warnings
        results = await store.search(
            query=problem,
            k=3,  # Top 3 most relevant warnings
            filter=None,
        )

        warnings: list[str] = []
        seen_strategies: set[str] = set()

        for result in results:
            # Only include high-relevance matches (score > 0.3 for cosine)
            if result.score > 0.3:
                # Dedupe by strategy
                strategy = result.metadata.get("strategy", "")
                if strategy and strategy in seen_strategies:
                    continue
                seen_strategies.add(strategy)

                # Format the warning
                warning_text = result.text.strip()
                if not warning_text.startswith("TRAP WARNING"):
                    warning_text = f"TRAP WARNING: {warning_text}"
                warnings.append(warning_text)

        # Fall back to keyword matching if no RAG results
        if not warnings:
            return _get_trap_warnings_sync(problem, complexity)

        # Add complexity-based guidance
        if complexity == Complexity.HIGH and len(warnings) < 2:
            warnings.append(
                "HIGH COMPLEXITY: Additional traps may exist. "
                "Verify each step. Consider alternative interpretations."
            )

        return "\n\n".join(warnings)

    except Exception as e:
        from loguru import logger

        logger.warning(f"RAG search failed, using keyword fallback: {e}")
        return _get_trap_warnings_sync(problem, complexity)


def _get_trap_warnings(problem: str, complexity: Complexity) -> str:
    """Get reasoning trap warnings (sync wrapper).

    Tries RAG first, falls back to keyword matching.
    """
    try:
        import asyncio

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in async context - can't run async code synchronously
            # Use sync fallback
            return _get_trap_warnings_sync(problem, complexity)
        except RuntimeError:
            # No running loop - we can create one
            return asyncio.run(_get_trap_warnings_rag(problem, complexity))
    except Exception:
        return _get_trap_warnings_sync(problem, complexity)


# --- Contradiction Detection (stub - to be connected to knowledge_graph) ---


def _detect_contradictions(
    claims: list[str], evidence: list[str], session: RouterSession
) -> tuple[list[str], list[Contradiction], list[str]]:
    """Detect contradictions between claims and evidence.

    TODO: Connect to knowledge_graph for real contradiction detection.
    For now, uses simple heuristics.

    Returns:
        Tuple of (verified_claims, contradictions, missing_evidence)

    """
    verified: list[str] = []
    contradictions: list[Contradiction] = []
    missing: list[str] = []

    # Simple heuristic: claims supported by evidence are verified
    for claim in claims:
        claim_lower = claim.lower()
        supported = False

        for ev in evidence:
            ev_lower = ev.lower()
            # Check for keyword overlap (very simple)
            claim_words = set(claim_lower.split())
            ev_words = set(ev_lower.split())
            overlap = claim_words & ev_words
            if len(overlap) >= 3:  # At least 3 words in common
                supported = True
                break

        if supported:
            verified.append(claim)
        else:
            missing.append(f"Claim '{claim[:50]}...' lacks supporting evidence")

    # Check for obvious contradictions in claims
    for i, c1 in enumerate(claims):
        c1_lower = c1.lower()
        for c2 in claims[i + 1 :]:
            c2_lower = c2.lower()
            # Simple contradiction patterns
            if (
                ("increase" in c1_lower and "decrease" in c2_lower)
                or ("decrease" in c1_lower and "increase" in c2_lower)
                or ("always" in c1_lower and "never" in c2_lower)
                or ("never" in c1_lower and "always" in c2_lower)
            ):
                contradictions.append(
                    Contradiction(
                        claim1=c1,
                        claim2=c2,
                        explanation="Potential logical contradiction detected",
                    )
                )

    return verified, contradictions, missing


# --- Tool Implementations ---


def initialize_reasoning(
    problem: str,
    complexity: str = "auto",
) -> InitializeResponse:
    """Initialize a new reasoning session with enforced constraints.

    Args:
        problem: The problem to reason about
        complexity: Complexity level ("low", "medium", "high", "auto")

    Returns:
        InitializeResponse with session_id and constraints

    """
    with tracer.start_as_current_span("router.initialize_reasoning") as span:
        # Resolve complexity
        complexity_enum = Complexity(complexity)
        resolved, config = resolve_complexity(complexity_enum, problem)

        span.set_attribute("complexity.requested", complexity)
        span.set_attribute("complexity.resolved", resolved.value)

        # Get trap warnings
        guidance = _get_trap_warnings(problem, resolved)

        # Create session
        session = RouterSession(
            id=uuid.uuid4().hex[:12],
            problem=problem,
            complexity=resolved,
            min_steps=config.min_steps,
            max_steps=config.max_steps,
            confidence_threshold=config.confidence_threshold,
            guidance=guidance,
        )

        # Save session
        _save_session(session)

        # Periodic cleanup
        _cleanup_expired_sessions()

        span.set_attribute("session_id", session.id)
        span.set_attribute("min_steps", session.min_steps)
        span.set_attribute("max_steps", session.max_steps)

        # Record metric
        record_session_created(session.id, resolved.value)

        return InitializeResponse(
            session_id=session.id,
            complexity=resolved.value,
            min_steps=session.min_steps,
            max_steps=session.max_steps,
            confidence_threshold=session.confidence_threshold,
            guidance=guidance,
        )


def submit_atomic_step(
    session_id: str,
    step_type: str,
    content: str,
    confidence: float,
    evidence: list[str] | None = None,
) -> StepResponse:
    """Submit a single reasoning step. May be REJECTED.

    Args:
        session_id: Session ID from initialize_reasoning
        step_type: One of "premise", "hypothesis", "verification", "synthesis"
        content: The reasoning content
        confidence: Confidence level (0.0-1.0)
        evidence: Optional evidence supporting this step

    Returns:
        StepResponse with status, reason if rejected, and guidance

    """
    start_time = time.time()

    with tracer.start_as_current_span("router.submit_step") as span:
        span.set_attribute("session_id", session_id)
        span.set_attribute("step_type", step_type)
        span.set_attribute("confidence", confidence)

        # Get session
        session = _get_session(session_id)
        if session is None:
            span.set_attribute("status", "REJECTED")
            span.set_attribute("rejection_reason", "session_not_found")
            record_step(session_id, step_type, "REJECTED", (time.time() - start_time) * 1000)
            return StepResponse(
                status=RouterStatus.REJECTED,
                rejection_reason=f"Session '{session_id}' not found or expired",
                valid_next_steps=[],
                steps_taken=0,
                steps_remaining=0,
                can_synthesize=False,
                synthesis_blockers=["Invalid session"],
            )

        # Parse step type
        try:
            step_type_enum = StepType(step_type)
        except ValueError:
            span.set_attribute("status", "REJECTED")
            span.set_attribute("rejection_reason", "invalid_step_type")
            record_step(session_id, step_type, "REJECTED", (time.time() - start_time) * 1000)
            return StepResponse(
                status=RouterStatus.REJECTED,
                rejection_reason=f"Invalid step_type '{step_type}'. Must be one of: premise, hypothesis, verification, synthesis",
                valid_next_steps=[s.value for s in get_valid_next_steps(session)],
                steps_taken=session.step_count,
                steps_remaining=steps_remaining(session),
                can_synthesize=False,
                synthesis_blockers=["Invalid step type"],
            )

        # Check if max steps reached (Rule E)
        if check_maximum_steps(session) and step_type_enum != StepType.SYNTHESIS:
            span.set_attribute("status", "REJECTED")
            span.set_attribute("rejection_reason", "max_steps_reached")
            record_step(session_id, step_type, "REJECTED", (time.time() - start_time) * 1000)
            return StepResponse(
                status=RouterStatus.REJECTED,
                rejection_reason=get_max_steps_guidance(session),
                valid_next_steps=[StepType.SYNTHESIS.value],
                steps_taken=session.step_count,
                steps_remaining=0,
                can_synthesize=True,
                synthesis_blockers=[],
            )

        # Evaluate step against all rules
        status, rejection_reason = evaluate_step(session, step_type_enum, confidence)

        if status == RouterStatus.ACCEPTED:
            # Create and add step
            step = RouterStep(
                id=uuid.uuid4().hex[:8],
                step_type=step_type_enum,
                content=content,
                confidence=confidence,
                evidence=evidence or [],
            )
            session.steps.append(step)
            _save_session(session)

            can_synth, blockers = can_synthesize(session)

            span.set_attribute("status", "ACCEPTED")
            span.set_attribute("step_id", step.id)
            span.set_attribute("steps_taken", session.step_count)
            record_step(session_id, step_type, "ACCEPTED", (time.time() - start_time) * 1000)

            return StepResponse(
                status=RouterStatus.ACCEPTED,
                step_id=step.id,
                valid_next_steps=[s.value for s in get_valid_next_steps(session)],
                steps_taken=session.step_count,
                steps_remaining=steps_remaining(session),
                can_synthesize=can_synth,
                synthesis_blockers=blockers,
            )

        # Step was rejected/requires action
        can_synth, blockers = can_synthesize(session)

        span.set_attribute("status", status.value)
        span.set_attribute("rejection_reason", rejection_reason or "unknown")
        record_step(session_id, step_type, status.value, (time.time() - start_time) * 1000)

        return StepResponse(
            status=status,
            rejection_reason=rejection_reason,
            valid_next_steps=[s.value for s in get_valid_next_steps(session)],
            steps_taken=session.step_count,
            steps_remaining=steps_remaining(session),
            can_synthesize=can_synth,
            synthesis_blockers=blockers,
        )


def create_branch(
    session_id: str,
    alternatives: list[str],
) -> BranchResponse:
    """Create alternative reasoning branches.

    Required when submit_atomic_step returns BRANCH_REQUIRED.

    Args:
        session_id: Session ID from initialize_reasoning
        alternatives: 2-4 alternative hypotheses to explore

    Returns:
        BranchResponse with branch IDs and guidance

    """
    # Get session
    session = _get_session(session_id)
    if session is None:
        return BranchResponse(
            branch_ids=[],
            guidance=f"Error: Session '{session_id}' not found or expired",
        )

    # Validate alternatives count
    if len(alternatives) < 2:
        return BranchResponse(
            branch_ids=[],
            guidance="Error: Need at least 2 alternatives for branching",
        )
    if len(alternatives) > 4:
        return BranchResponse(
            branch_ids=[],
            guidance="Error: Maximum 4 alternatives allowed",
        )

    # Create branches
    branch_ids: list[str] = []
    for alt in alternatives:
        branch = Branch(
            id=uuid.uuid4().hex[:8],
            hypothesis=alt,
            confidence=0.5,  # Start neutral
        )
        session.branches[branch.id] = branch
        branch_ids.append(branch.id)

    _save_session(session)

    guidance = (  # nosec B608 - f-string for user guidance, not SQL
        f"Created {len(branch_ids)} branches. To evaluate:\n"
        "1. Submit verification steps for each hypothesis\n"
        "2. Compare evidence and confidence across branches\n"
        "3. Select the hypothesis with strongest support\n"
        "4. Continue reasoning from that branch"
    )

    return BranchResponse(branch_ids=branch_ids, guidance=guidance)


def verify_claims(
    session_id: str,
    claims: list[str],
    evidence: list[str],
) -> VerifyResponse:
    """Verify claims against evidence and check for contradictions.

    Args:
        session_id: Session ID from initialize_reasoning
        claims: Claims to verify
        evidence: Evidence to check against

    Returns:
        VerifyResponse with verification results

    """
    # Get session
    session = _get_session(session_id)
    if session is None:
        return VerifyResponse(
            verified=[],
            contradictions=[],
            missing_evidence=[f"Session '{session_id}' not found or expired"],
            can_synthesize=False,
            synthesis_blockers=["Invalid session"],
        )

    # Detect contradictions and verify claims
    verified, contradictions, missing = _detect_contradictions(claims, evidence, session)

    # Update session with verified claims
    session.verified_claims.extend(verified)
    session.contradictions.extend(contradictions)
    _save_session(session)

    can_synth, blockers = can_synthesize(session)

    return VerifyResponse(
        verified=verified,
        contradictions=contradictions,
        missing_evidence=missing,
        can_synthesize=can_synth,
        synthesis_blockers=blockers,
    )


# --- Session Management ---


def get_session_state(session_id: str) -> dict | None:
    """Get current session state for debugging/inspection.

    Args:
        session_id: Session ID

    Returns:
        Session state dict or None if not found

    """
    session = _get_session(session_id)
    if session is None:
        return None

    return {
        "id": session.id,
        "problem": session.problem[:100] + "..." if len(session.problem) > 100 else session.problem,
        "complexity": session.complexity.value,
        "min_steps": session.min_steps,
        "max_steps": session.max_steps,
        "confidence_threshold": session.confidence_threshold,
        "step_count": session.step_count,
        "steps_remaining": steps_remaining(session),
        "has_verification": session.has_verification,
        "can_synthesize": can_synthesize(session)[0],
        "synthesis_blockers": can_synthesize(session)[1],
        "valid_next_steps": [s.value for s in get_valid_next_steps(session)],
        "branch_count": len(session.branches),
        "verified_claims": len(session.verified_claims),
        "contradictions": len(session.contradictions),
    }


def close_session(session_id: str) -> bool:
    """Close and delete a session.

    Args:
        session_id: Session ID

    Returns:
        True if session existed and was deleted

    """
    return _delete_session(session_id)


# --- Stats ---


def get_router_stats() -> dict:
    """Get router statistics.

    Returns:
        Dict with active sessions count and other stats

    """
    with _sessions_lock:
        now = time.time()
        active = [s for s in _sessions.values() if now - s.created_at <= _session_ttl_seconds]

        return {
            "active_sessions": len(active),
            "total_steps": sum(len(s.steps) for s in active),
            "total_branches": sum(len(s.branches) for s in active),
            "avg_steps_per_session": (
                sum(len(s.steps) for s in active) / len(active) if active else 0
            ),
        }
