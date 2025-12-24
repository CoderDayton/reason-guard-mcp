"""Unified Dynamic Reasoner with auto-mode selection.

Consolidates all reasoning strategies (chain, matrix, verify) into a single
intelligent tool that auto-selects the optimal approach based on problem
characteristics.

Key Features:
    - Auto-mode selection based on complexity detection
    - Blind spot detection across reasoning paths
    - Sparse thought graph for relationship tracking
    - RLVR-style reward tracking for learning
    - RAG integration via AsyncVectorStore
    - Domain-specific handlers (math, code, logic)
    - MPPA, CISC, FOBAR integration

Architecture:
    - Problem arrives -> complexity detection
    - Auto-select mode: chain (simple) vs matrix (complex) vs hybrid
    - Build thought graph as reasoning progresses
    - Apply CISC confidence scoring
    - Detect blind spots
    - Track for RLVR rewards
    - Store thoughts in vector store for RAG
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

# Import types from the new reasoning_types module
from src.tools.reasoning_types import (
    BlindSpot,
    DomainType,
    ReasoningMode,
    ReasoningSession,
    ResponseVerbosity,
    RewardSignal,
    SessionAnalytics,
    SessionStatus,
    SuggestionRecord,
    SuggestionWeights,
    Thought,
    ThoughtType,
)
from src.utils.complexity import ComplexityResult, detect_complexity
from src.utils.confidence import (
    CISCSelectionResult,
    cisc_select,
)
from src.utils.scoring import semantic_survival_score
from src.utils.session import AsyncSessionManager

if TYPE_CHECKING:
    from src.models.context_encoder import ContextEncoder
    from src.models.vector_store import AsyncVectorStore

from src.models.knowledge_graph import KnowledgeGraph, KnowledgeGraphExtractor

# Re-export types for backwards compatibility
__all__ = [
    "BlindSpot",
    "DomainType",
    "ReasoningMode",
    "ReasoningSession",
    "ResponseVerbosity",
    "RewardSignal",
    "SessionAnalytics",
    "SessionStatus",
    "SuggestionRecord",
    "SuggestionWeights",
    "Thought",
    "ThoughtType",
    "UnifiedReasonerManager",
    "detect_blind_spots",
    "detect_domain",
    "get_unified_manager",
    "init_unified_manager",
    "is_planning_step",
    "select_mode",
]

# =============================================================================
# Planning Step Detection (from MPPA)
# =============================================================================

# Precompiled regex for planning detection (LATENCY: avoids lower() on full text)
_PLANNING_PATTERN = re.compile(
    r"^\s*(?:"
    r"let me|let's|i'll|i will|i should|i need to|"
    r"first,|wait,|alternatively|maybe|perhaps|"
    r"one approach|another way|we could|we can|"
    r"the strategy|my plan|to solve this|i think|considering"
    r")",
    re.IGNORECASE,
)


def is_planning_step(thought: str) -> bool:
    """Detect if a thought is a planning step (decision point)."""
    return bool(_PLANNING_PATTERN.match(thought))


# =============================================================================
# Domain Detection
# =============================================================================

DOMAIN_PATTERNS: dict[DomainType, list[re.Pattern[str]]] = {
    DomainType.MATH: [
        re.compile(r"\d+\s*[\+\-\*\/\=]\s*\d+", re.IGNORECASE),
        re.compile(
            r"equation|formula|calculate|solve|sum|product|integral|derivative", re.IGNORECASE
        ),
        re.compile(r"percent|ratio|fraction|proportion|average|mean|median", re.IGNORECASE),
        re.compile(r"x\s*=|y\s*=|n\s*=", re.IGNORECASE),
        re.compile(r"triangle|circle|square|rectangle|angle|area|volume", re.IGNORECASE),
    ],
    DomainType.CODE: [
        re.compile(r"function|def |class |import |from .* import", re.IGNORECASE),
        re.compile(r"if\s*\(|for\s*\(|while\s*\(|return ", re.IGNORECASE),
        re.compile(r"print\(|console\.log|System\.out", re.IGNORECASE),
        re.compile(r"\.py|\.js|\.java|\.cpp|\.rs", re.IGNORECASE),
        re.compile(r"algorithm|complexity|O\(n\)|runtime|memory", re.IGNORECASE),
        re.compile(r"bug|error|exception|debug|compile", re.IGNORECASE),
    ],
    DomainType.LOGIC: [
        re.compile(r"\ball\b.*\bare\b|\bsome\b.*\bare\b", re.IGNORECASE),
        re.compile(r"\bif\b.*\bthen\b|\bonly if\b", re.IGNORECASE),
        re.compile(r"\bnot\b.*\band\b|\bor\b.*\bnot\b", re.IGNORECASE),
        re.compile(r"\bmust\b|\bcannot\b|\bimpossible\b", re.IGNORECASE),
        re.compile(r"contradict|paradox|dilemma|syllogism|premise|conclusion", re.IGNORECASE),
        re.compile(r"valid|invalid|sound|unsound|fallacy", re.IGNORECASE),
    ],
    DomainType.FACTUAL: [
        re.compile(r"who is|what is|when did|where is|how many", re.IGNORECASE),
        re.compile(r"according to|based on|states that|mentions that", re.IGNORECASE),
        re.compile(r"true or false|fact check|verify", re.IGNORECASE),
    ],
}


def detect_domain(text: str) -> DomainType:
    """Detect the problem domain from text."""
    scores: dict[DomainType, int] = dict.fromkeys(DomainType, 0)

    for domain, patterns in DOMAIN_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                scores[domain] += 1

    # Return domain with highest score, or GENERAL if no matches
    max_score = max(scores.values())
    if max_score == 0:
        return DomainType.GENERAL

    return max(scores, key=lambda d: scores[d])


# =============================================================================
# Blind Spot Detection & Speculative Criticism
# =============================================================================

# Precompiled regex patterns for blind spot detection (LATENCY: avoids re.compile on every call)
_BLIND_SPOT_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # (pattern, description, suggested_action)
    # --- Original blind spot patterns ---
    (
        re.compile(r"assume|assuming|assumption", re.IGNORECASE),
        "Unstated assumption detected",
        "Explicitly state and validate assumption",
    ),
    (
        re.compile(r"probably|likely|maybe|perhaps|might", re.IGNORECASE),
        "Uncertain claim without confidence quantification",
        "Provide confidence level or supporting evidence",
    ),
    (
        re.compile(r"etc\.?|and so on|and others", re.IGNORECASE),
        "Incomplete enumeration",
        "List all relevant items or explain why truncated",
    ),
    (
        re.compile(r"obviously|clearly|of course|everyone knows", re.IGNORECASE),
        "Unsupported assertion",
        "Provide reasoning or evidence for claim",
    ),
    (
        re.compile(r"always|never|all|none", re.IGNORECASE),
        "Absolute claim that may have exceptions",
        "Consider edge cases or counterexamples",
    ),
    (
        re.compile(r"but|however|although|despite", re.IGNORECASE),
        "Potential contradiction or tension",
        "Resolve apparent conflict or explain nuance",
    ),
    # --- Speculative Criticism: Logical Fallacies (zero-latency quality boost) ---
    (
        re.compile(r"because .+ therefore|since .+ thus|if .+ then .+ so", re.IGNORECASE),
        "Circular reasoning detected",
        "Break circular dependency - find independent evidence",
    ),
    (
        re.compile(r"(experts?|scientists?|studies?) (say|show|prove|believe)", re.IGNORECASE),
        "Appeal to authority without citation",
        "Name the specific source or provide evidence directly",
    ),
    (
        re.compile(
            r"(this|that|it) (proves|shows|means|implies) (that )?(\w+ ){0,3}(true|correct|right|wrong|false)",
            re.IGNORECASE,
        ),
        "Hasty conclusion from limited evidence",
        "State what evidence is still needed before concluding",
    ),
    (
        re.compile(
            r"(first|1\.|step 1).{0,50}(therefore|thus|so|hence) .{0,50}(answer|result|solution)",
            re.IGNORECASE,
        ),
        "Skipped reasoning steps",
        "Show intermediate steps between premise and conclusion",
    ),
    (
        re.compile(r"(only|just|simply|merely) (need|have|require|must)", re.IGNORECASE),
        "Oversimplification detected",
        "Identify complications or prerequisites being glossed over",
    ),
    (
        re.compile(r"(same|similar|like|analogous).{0,30}(therefore|so|thus|hence)", re.IGNORECASE),
        "Weak analogy as primary evidence",
        "Explain why analogy holds or provide direct evidence",
    ),
    (
        re.compile(
            r"(no|without|lacking) (evidence|proof|data|support).{0,20}(therefore|so|means)",
            re.IGNORECASE,
        ),
        "Argument from ignorance",
        "Absence of evidence is not evidence - seek positive confirmation",
    ),
]

# Backwards compatibility alias
BLIND_SPOT_PATTERNS: list[tuple[re.Pattern[str], str, str]] = _BLIND_SPOT_PATTERNS


def detect_blind_spots(
    thought: str,
    existing_blind_spots: list[BlindSpot],
    step_number: int,
) -> list[BlindSpot]:
    """Detect potential blind spots in reasoning."""
    blind_spots: list[BlindSpot] = []
    thought_lower = thought.lower()

    for pattern, description, action in BLIND_SPOT_PATTERNS:
        if pattern.search(thought_lower):
            # Check if this blind spot type was already detected
            already_exists = any(
                bs.description == description and not bs.addressed for bs in existing_blind_spots
            )
            if not already_exists:
                blind_spots.append(
                    BlindSpot(
                        id=f"bs_{uuid.uuid4().hex[:8]}",
                        description=description,
                        detected_at_step=step_number,
                        severity="medium",  # Could be refined with more analysis
                        suggested_action=action,
                    )
                )

    return blind_spots


# =============================================================================
# Reward Computation (RLVR-style)
# =============================================================================


def compute_step_rewards(
    thought: Thought,
    session: ReasoningSession,
) -> list[RewardSignal]:
    """Compute RLVR-style rewards for a reasoning step."""
    rewards: list[RewardSignal] = []

    # 1. Consistency reward: Does this step align with previous steps?
    if thought.contradicts:
        rewards.append(
            RewardSignal(
                step_id=thought.id,
                reward_type="consistency",
                value=-0.5,
                reason=f"Contradicts {len(thought.contradicts)} previous thought(s)",
            )
        )
    elif thought.supports:
        rewards.append(
            RewardSignal(
                step_id=thought.id,
                reward_type="consistency",
                value=0.3,
                reason=f"Supports {len(thought.supports)} previous thought(s)",
            )
        )

    # 2. Coherence reward: Is the reasoning clear and well-structured?
    if thought.survival_score is not None:
        if thought.survival_score > 0.7:
            rewards.append(
                RewardSignal(
                    step_id=thought.id,
                    reward_type="coherence",
                    value=0.4,
                    reason=f"High survival score: {thought.survival_score:.2f}",
                )
            )
        elif thought.survival_score < 0.3:
            rewards.append(
                RewardSignal(
                    step_id=thought.id,
                    reward_type="coherence",
                    value=-0.3,
                    reason=f"Low survival score: {thought.survival_score:.2f}",
                )
            )

    # 3. Efficiency reward: Are we making progress?
    if thought.thought_type == ThoughtType.REVISION:
        # Revisions are necessary but costly
        rewards.append(
            RewardSignal(
                step_id=thought.id,
                reward_type="efficiency",
                value=-0.1,
                reason="Revision indicates earlier reasoning was incomplete",
            )
        )
    elif thought.is_planning:
        # Planning steps are valuable early, less so later
        if thought.step_number <= 3:
            rewards.append(
                RewardSignal(
                    step_id=thought.id,
                    reward_type="efficiency",
                    value=0.2,
                    reason="Early planning step",
                )
            )
        elif thought.step_number > 7:
            rewards.append(
                RewardSignal(
                    step_id=thought.id,
                    reward_type="efficiency",
                    value=-0.2,
                    reason="Late planning may indicate lack of direction",
                )
            )

    return rewards


# =============================================================================
# Mode Selection Logic
# =============================================================================


def select_mode(
    complexity: ComplexityResult,
    domain: DomainType,
    context_length: int,
) -> ReasoningMode:
    """Auto-select the best reasoning mode based on problem characteristics."""
    # High complexity -> Matrix for multi-perspective analysis
    if complexity.complexity_level == "high":
        return ReasoningMode.MATRIX

    # Medium-high with math/logic -> Matrix for structured exploration
    if complexity.complexity_level == "medium-high" and domain in (
        DomainType.MATH,
        DomainType.LOGIC,
    ):
        return ReasoningMode.MATRIX

    # Long context -> Hybrid (start chain, escalate if needed)
    if context_length > 2000:
        return ReasoningMode.HYBRID

    # Low complexity -> Simple chain
    if complexity.complexity_level == "low":
        return ReasoningMode.CHAIN

    # Default to chain for medium complexity
    return ReasoningMode.CHAIN


# =============================================================================
# Unified Reasoner Manager
# =============================================================================


class UnifiedReasonerManager(AsyncSessionManager[ReasoningSession]):
    """Manages unified reasoning sessions with auto-mode selection.

    This is a STATE MANAGER, not a reasoner. The calling LLM provides
    reasoning content; this tool tracks, organizes, and enhances the process.

    Uses AsyncSessionManager with uvloop for high-performance async operations.
    All session access is non-blocking with asyncio.Lock.
    """

    # Session limits (CWE-770: Allocation of Resources Without Limits)
    MAX_THOUGHTS_PER_SESSION: int = 1000

    def __init__(
        self,
        vector_store: AsyncVectorStore | None = None,
        encoder: ContextEncoder | None = None,
        enable_rag: bool = True,
        enable_blind_spots: bool = True,
        enable_rewards: bool = True,
        enable_graph: bool = True,
        enable_domain_validation: bool = True,
        weight_store: Any = None,
        enable_weight_persistence: bool = True,
        enable_semantic_scoring: bool = True,
    ) -> None:
        """Initialize the unified reasoner manager.

        Args:
            vector_store: Optional AsyncVectorStore for RAG integration.
            encoder: Optional ContextEncoder for semantic similarity scoring.
            enable_rag: Whether to use RAG for thought retrieval.
            enable_blind_spots: Whether to detect blind spots.
            enable_rewards: Whether to compute RLVR rewards.
            enable_graph: Whether to build thought relationship graph.
            enable_domain_validation: Whether to validate thoughts with domain handlers.
            weight_store: Optional WeightStore for persistent weight learning.
            enable_weight_persistence: Whether to persist weights across sessions.
            enable_semantic_scoring: Whether to use embedding-based scoring (vs word overlap).

        """
        super().__init__()
        self._vector_store = vector_store
        self._encoder = encoder
        self._enable_rag = enable_rag and vector_store is not None
        self._enable_semantic_scoring = enable_semantic_scoring and encoder is not None
        self._enable_blind_spots = enable_blind_spots
        self._enable_rewards = enable_rewards
        self._enable_graph = enable_graph
        self._enable_domain_validation = enable_domain_validation
        self._weight_store = weight_store
        self._enable_weight_persistence = enable_weight_persistence

        # Per-session knowledge graphs for fact tracking
        self._session_kgs: dict[str, KnowledgeGraph] = {}

        # OPT3: Cache extractor instance - avoid repeated instantiation (~8ms)
        self._kg_extractor: KnowledgeGraphExtractor | None = None

    async def remove_session(self, session_id: str) -> ReasoningSession | None:
        """Remove a session and clean up associated resources.

        Overrides base class to also clean up knowledge graph memory.

        Args:
            session_id: Session identifier.

        Returns:
            Removed session state, or None if not found.

        """
        # Clean up KG before removing session
        self._session_kgs.pop(session_id, None)
        return await super().remove_session(session_id)

    async def cleanup_stale(
        self,
        max_age: timedelta,
        *,
        now: datetime | None = None,
        predicate: Callable[[ReasoningSession], bool] | None = None,
    ) -> list[str]:
        """Remove stale sessions and clean up associated resources.

        Overrides base class to also clean up knowledge graph memory.

        Args:
            max_age: Maximum age for sessions.
            now: Reference time (defaults to datetime.now()).
            predicate: Optional additional filter.

        Returns:
            List of removed session IDs.

        """
        removed = await super().cleanup_stale(max_age, now=now, predicate=predicate)
        # Clean up KGs for removed sessions
        for session_id in removed:
            self._session_kgs.pop(session_id, None)
        return removed

    def _score_thought(
        self,
        thought: str,
        context: str,
        position: int,
        session_id: str | None = None,
        strategy: str | None = None,
    ) -> float:
        """Score a thought using semantic embeddings when available.

        This is the central scoring method that uses:
        - Embedding-based similarity when encoder is available
        - Knowledge graph alignment when session has a KG
        - Falls back to word overlap when no encoder

        Args:
            thought: The reasoning step to score.
            context: Problem context and previous steps.
            position: Step position in the chain.
            session_id: Optional session ID for KG lookup.
            strategy: Optional reasoning strategy hint.

        Returns:
            Score between 0.0 and 1.0.

        """
        kg = self._session_kgs.get(session_id) if session_id else None
        return semantic_survival_score(
            thought=thought,
            context=context,
            position=position,
            encoder=self._encoder,
            kg=kg,
            strategy=strategy,
        )

    async def _score_thought_async(
        self,
        thought: str,
        context: str,
        position: int,
        session_id: str | None = None,
        strategy: str | None = None,
    ) -> float:
        """Async version of _score_thought for parallel scoring.

        P0-PERF: Runs embedding computation in executor for parallelism.
        When scoring multiple alternatives, use asyncio.gather with this method
        for 4x speedup (from sequential to parallel).

        Args:
            thought: The reasoning step to score.
            context: Problem context and previous steps.
            position: Step position in the chain.
            session_id: Optional session ID for KG lookup.
            strategy: Optional reasoning strategy hint.

        Returns:
            Score between 0.0 and 1.0.

        """
        loop = asyncio.get_running_loop()
        kg = self._session_kgs.get(session_id) if session_id else None

        # Run CPU-bound embedding in executor
        return await loop.run_in_executor(
            None,  # Use default executor
            lambda: semantic_survival_score(
                thought=thought,
                context=context,
                position=position,
                encoder=self._encoder,
                kg=kg,
                strategy=strategy,
            ),
        )

    async def start_session(
        self,
        problem: str,
        context: str = "",
        mode: ReasoningMode = ReasoningMode.AUTO,
        expected_steps: int = 10,
        rows: int | None = None,
        cols: int | None = None,
    ) -> dict[str, Any]:
        """Start a new reasoning session.

        Args:
            problem: The problem to reason about.
            context: Optional background context.
            mode: Reasoning mode (auto by default).
            expected_steps: Expected number of steps for chain mode.
            rows: Matrix rows (auto-detected if None).
            cols: Matrix columns (auto-detected if None).

        Returns:
            Session info with ID, selected mode, and initial guidance.

        """
        # Use full UUID for security (CWE-330: insufficient randomness)
        session_id = str(uuid.uuid4())

        # Analyze problem
        complexity = detect_complexity(problem, context)
        domain = detect_domain(problem + " " + context)

        # Auto-select mode if needed
        actual_mode = mode
        if mode == ReasoningMode.AUTO:
            actual_mode = select_mode(complexity, domain, len(context))
            logger.info(
                f"Auto-selected mode: {actual_mode.value} for complexity={complexity.complexity_level}"
            )

        # Determine matrix dimensions if applicable
        if actual_mode in (ReasoningMode.MATRIX, ReasoningMode.HYBRID):
            rows = rows or complexity.recommended_rows
            cols = cols or complexity.recommended_cols

        # Create session
        session = ReasoningSession(
            session_id=session_id,
            problem=problem,
            context=context,
            mode=mode,
            actual_mode=actual_mode,
            status=SessionStatus.ACTIVE,
            domain=domain,
            complexity=complexity,
            matrix_rows=rows or 0,
            matrix_cols=cols or 0,
            rag_enabled=self._enable_rag,
        )

        # Initialize thought graph
        if self._enable_graph:
            from src.tools.thought_graph import ThoughtGraph

            session.thought_graph = ThoughtGraph()

        # Load persistent weights if enabled
        if self._enable_weight_persistence and self._weight_store is not None:
            try:
                domain_weights = self._weight_store.load_weights(domain.value)
                # Apply loaded weights to session
                session.suggestion_weights.resolve = domain_weights.resolve
                session.suggestion_weights.continue_blind_spot = domain_weights.continue_blind_spot
                session.suggestion_weights.verify = domain_weights.verify
                session.suggestion_weights.continue_depth = domain_weights.continue_depth
                session.suggestion_weights.synthesize = domain_weights.synthesize
                session.suggestion_weights.finish = domain_weights.finish
                session.suggestion_weights.continue_default = domain_weights.continue_default
                logger.debug(f"Loaded persistent weights for domain {domain.value}")
            except Exception as e:
                logger.warning(f"Failed to load persistent weights: {e}")

        await self.register_session(session_id, session)

        # Initialize Knowledge Graph for this session and extract initial entities
        kg = KnowledgeGraph()
        extractor = KnowledgeGraphExtractor()
        try:
            # Extract entities from problem and context
            initial_text = f"{problem}\n{context}" if context else problem
            extractor.extract(initial_text, existing_graph=kg)
            self._session_kgs[session_id] = kg
            logger.debug(
                f"Initialized KG for session {session_id} with {len(kg.entities)} entities"
            )
        except Exception as e:
            # Non-fatal: KG is optional enhancement
            logger.warning(f"KG extraction failed for session start: {e}")
            self._session_kgs[session_id] = kg  # Store empty KG

        logger.info(
            f"Started session {session_id} with mode={actual_mode.value}, domain={domain.value}"
        )

        # Build response with guidance
        response: dict[str, Any] = {
            "session_id": session_id,
            "mode": mode.value,
            "actual_mode": actual_mode.value,
            "domain": domain.value,
            "complexity": complexity.to_dict(),
            "status": "active",
        }

        # Mode-specific guidance
        if actual_mode == ReasoningMode.CHAIN:
            response["guidance"] = {
                "expected_steps": expected_steps,
                "instruction": (
                    "Begin chain-of-thought reasoning. Add steps sequentially using "
                    "action='continue'. Branch or revise as needed."
                ),
            }
        elif actual_mode == ReasoningMode.MATRIX:
            response["guidance"] = {
                "rows": rows,
                "cols": cols,
                "strategies": self._get_strategies_for_domain(domain, rows or 3),
                "instruction": (
                    f"Fill the {rows}x{cols} matrix. Start with row=0 (planning row), "
                    "then fill cells with different strategy perspectives."
                ),
            }
        elif actual_mode == ReasoningMode.HYBRID:
            response["guidance"] = {
                "instruction": (
                    "Start with chain reasoning. If complexity increases, "
                    "the session will escalate to matrix mode automatically."
                ),
                "escalation_threshold": 5,  # Steps before considering escalation
            }

        return response

    async def add_thought(
        self,
        session_id: str,
        content: str,
        thought_type: str | ThoughtType = ThoughtType.CONTINUATION,
        row: int | None = None,
        col: int | None = None,
        branch_from: str | None = None,
        revises: str | None = None,
        confidence: float | None = None,
        alternatives: list[str] | None = None,
        alternative_confidences: list[float] | None = None,
        verbosity: str | ResponseVerbosity = ResponseVerbosity.MINIMAL,
    ) -> dict[str, Any]:
        """Add a thought to the reasoning session.

        Args:
            session_id: Session ID.
            content: The thought content.
            thought_type: Type of thought (continuation, revision, etc.).
            row: Matrix row (for matrix mode).
            col: Matrix column (for matrix mode).
            branch_from: Thought ID to branch from.
            revises: Thought ID being revised.
            confidence: LLM confidence in this thought.
            alternatives: Alternative thoughts considered (MPPA).
            alternative_confidences: Confidence for alternatives (CISC).
            verbosity: Response verbosity level (minimal/normal/full). Default minimal
                for ~50% token reduction without quality loss.

        Returns:
            Result with thought ID, guidance, and any detected issues.

        Raises:
            ValueError: If session thought limit is exceeded.

        """
        # OPT4: Parse verbosity early
        if isinstance(verbosity, str):
            try:
                verbosity = ResponseVerbosity(verbosity)
            except ValueError:
                verbosity = ResponseVerbosity.MINIMAL
        async with self.session(session_id) as session:
            # Check session thought limit (CWE-770 prevention)
            if len(session.thoughts) >= self.MAX_THOUGHTS_PER_SESSION:
                raise ValueError(
                    f"Session thought limit ({self.MAX_THOUGHTS_PER_SESSION}) exceeded. "
                    "Please finalize this session and start a new one."
                )
            # Parse thought type
            if isinstance(thought_type, str):
                try:
                    thought_type = ThoughtType(thought_type)
                except ValueError:
                    thought_type = ThoughtType.CONTINUATION

            # Detect if this is a planning step
            is_planning = is_planning_step(content)
            if is_planning:
                thought_type = ThoughtType.PLANNING
                session.planning_steps += 1

            # RAG: Retrieve similar past thoughts and facts to surface to LLM
            # OPT1: Parallelize both searches with asyncio.gather for ~50ms latency reduction
            rag_context = ""
            rag_info: dict[str, Any] = {}
            if self._enable_rag and self._vector_store is not None:
                try:
                    # Launch both searches concurrently
                    similar_task = self._vector_store.search(content, k=3)
                    kg_facts_task = self._vector_store.search_kg_facts_with_decay(
                        query=content,
                        k=5,  # Get more facts to check for contradictions
                        domain=session.domain.value,
                        min_quality=0.3,
                    )
                    results = await asyncio.gather(
                        similar_task, kg_facts_task, return_exceptions=True
                    )

                    # Extract results with proper type narrowing
                    similar: list[Any] = results[0] if isinstance(results[0], list) else []
                    kg_facts_with_quality: list[tuple[Any, float]] = (
                        results[1] if isinstance(results[1], list) else []
                    )

                    # Log any exceptions that occurred
                    if isinstance(results[0], BaseException):
                        logger.warning(f"Similar thoughts search failed: {results[0]}")
                    if isinstance(results[1], BaseException):
                        logger.warning(f"KG facts search failed: {results[1]}")

                    # Process similar thoughts
                    if similar:
                        relevant_thoughts = [
                            {"text": r.text[:200], "relevance": round(r.score, 2)}
                            for r in similar
                            if r.score > 0.5
                        ]
                        if relevant_thoughts:
                            rag_context = "\nSimilar past reasoning:\n" + "\n".join(
                                f"- {t['text']}" for t in relevant_thoughts
                            )
                            rag_info["similar_thoughts"] = relevant_thoughts
                            logger.debug(f"RAG retrieved {len(relevant_thoughts)} similar thoughts")

                    # Process KG facts
                    if kg_facts_with_quality:
                        relevant_facts = []
                        for r, quality in kg_facts_with_quality:
                            if r.score > 0.5 and quality > 0.3:
                                relevant_facts.append(
                                    {
                                        "fact": r.text,
                                        "relevance": round(r.score, 2),
                                        "quality": round(quality, 2),
                                    }
                                )
                        if relevant_facts:
                            rag_context += "\nRelevant facts:\n" + "\n".join(
                                f"- {f['fact']}" for f in relevant_facts
                            )
                            rag_info["relevant_facts"] = relevant_facts
                            logger.debug(f"RAG retrieved {len(relevant_facts)} KG facts")
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")

            # Calculate survival score (now with RAG context)
            context_for_scoring = (
                session.context + " ".join(t.content for t in session.main_chain[-5:]) + rag_context
            )
            survival_score = self._score_thought(
                thought=content,
                context=context_for_scoring,
                position=session.current_step,
                session_id=session_id,
            )

            # MPPA: Handle alternatives
            cisc_result: CISCSelectionResult | None = None
            if alternatives and len(alternatives) > 0:
                all_candidates = [content] + alternatives
                all_confidences = [confidence or 0.5]
                if alternative_confidences:
                    all_confidences.extend(alternative_confidences)
                else:
                    all_confidences.extend([0.5] * len(alternatives))

                # Use CISC to select best candidate
                # P0-PERF: Parallelize alternative scoring with asyncio.gather
                recent_context = " ".join(t.content for t in session.main_chain[-3:])
                score_tasks = [
                    self._score_thought_async(
                        c,
                        context=recent_context,
                        position=session.current_step,
                        session_id=session_id,
                    )
                    for c in all_candidates
                ]
                scores = await asyncio.gather(*score_tasks)
                # Create candidates as (content, score) tuples for cisc_select
                candidates_with_scores = list(zip(all_candidates, scores, strict=False))
                cisc_result = cisc_select(candidates_with_scores)
                selected_idx = cisc_result.selected_index

                if selected_idx > 0:
                    # An alternative was selected
                    content = alternatives[selected_idx - 1]
                    survival_score = scores[selected_idx]
                    if alternative_confidences and selected_idx <= len(alternative_confidences):
                        confidence = alternative_confidences[selected_idx - 1]

                session.mppa_explorations += 1

            # Create thought
            thought_id = f"t_{uuid.uuid4().hex[:8]}"
            step_number = session.current_step + 1

            thought = Thought(
                id=thought_id,
                content=content,
                thought_type=thought_type,
                step_number=step_number,
                confidence=confidence,
                survival_score=survival_score,
                parent_id=session.thought_order[-1] if session.thought_order else None,
                branch_id=f"b_{uuid.uuid4().hex[:6]}" if branch_from else None,
                revises_id=revises,
                is_planning=is_planning,
                alternatives_considered=len(alternatives) if alternatives else 0,
            )

            # Add to session
            session.thoughts[thought_id] = thought
            session.thought_order.append(thought_id)

            if thought.branch_id:
                if thought.branch_id not in session.branches:
                    session.branches[thought.branch_id] = []
                session.branches[thought.branch_id].append(thought_id)

            # Matrix mode: track cell
            if session.actual_mode == ReasoningMode.MATRIX and row is not None and col is not None:
                session.matrix_cells[(row, col)] = thought_id

            session.updated_at = datetime.now()

            # OPT6: Early exit path for high-confidence thoughts (>0.9)
            # Skip expensive validations when LLM is already confident
            high_confidence = confidence is not None and confidence > 0.9

            # Blind spot detection (skip for high confidence)
            blind_spots_found: list[BlindSpot] = []
            if self._enable_blind_spots and not high_confidence:
                blind_spots_found = detect_blind_spots(
                    content,
                    session.blind_spots,
                    step_number,
                )
                session.blind_spots.extend(blind_spots_found)

            # Reward computation (skip for high confidence - minimal learning value)
            rewards_earned: list[RewardSignal] = []
            if self._enable_rewards and not high_confidence:
                rewards_earned = compute_step_rewards(thought, session)
                session.rewards.extend(rewards_earned)

            # Graph building
            graph_info: dict[str, Any] = {}
            if self._enable_graph and session.thought_graph is not None:
                try:
                    from src.tools.thought_graph import EdgeType

                    # Add node to graph
                    session.thought_graph.add_node(
                        node_id=thought_id,
                        content=content,
                        step_number=step_number,
                        thought_type=thought_type.value,
                        confidence=confidence,
                    )

                    # Add edges based on relationships
                    if thought.parent_id and thought.parent_id in session.thoughts:
                        session.thought_graph.add_edge(
                            thought.parent_id, thought_id, EdgeType.PARENT
                        )

                    if branch_from and branch_from in session.thoughts:
                        session.thought_graph.add_edge(branch_from, thought_id, EdgeType.BRANCH)

                    if revises and revises in session.thoughts:
                        session.thought_graph.add_edge(thought_id, revises, EdgeType.REVISES)

                    # Detect contradictions with recent thoughts
                    for recent_id in session.thought_order[-5:-1]:  # Last 5 except current
                        if recent_id in session.thoughts:
                            recent = session.thoughts[recent_id]
                            if self._detect_contradiction(content, recent.content):
                                session.thought_graph.add_edge(
                                    thought_id, recent_id, EdgeType.CONTRADICTS
                                )
                                thought.contradicts.append(recent_id)
                                session.graph_contradictions.append((thought_id, recent_id))

                    graph_info = {
                        "node_added": thought_id,
                        "edges_added": len(session.thought_graph.get_edges(source=thought_id)),
                        "contradictions": len(thought.contradicts),
                    }
                except Exception as e:
                    logger.warning(f"Failed to update thought graph: {e}")

            # Domain validation
            domain_validation: dict[str, Any] | None = None
            if self._enable_domain_validation:
                try:
                    from src.tools.domain_handlers import validate_thought

                    validation = validate_thought(
                        domain=session.domain.value,
                        thought=content,
                        context=session.context,
                    )
                    if validation is not None:
                        validation_dict = validation.to_dict()
                        domain_validation = validation_dict
                        session.domain_validations[thought_id] = validation_dict

                        # Adjust confidence based on validation
                        if confidence is not None and validation.confidence_adjustment != 0:
                            adjusted_confidence = max(
                                0.0, min(1.0, confidence + validation.confidence_adjustment)
                            )
                            thought.confidence = adjusted_confidence
                except Exception as e:
                    logger.warning(f"Domain validation failed: {e}")

            # RAG: Store thought in vector store
            # OPT2: Fire-and-forget - don't await the write, just schedule it
            # Vector ID is not critical for reasoning flow, so we defer persistence
            if self._enable_rag and self._vector_store is not None:
                vector_store = self._vector_store  # Capture for closure
                # Capture immutable data for background task (avoid race on thought object)
                thought_content = content
                thought_session = session_id
                thought_step = step_number
                thought_strategy = thought_type.value
                thought_score = survival_score

                async def _store_thought_background() -> None:
                    try:
                        await vector_store.add_thought(
                            thought=thought_content,
                            session_id=thought_session,
                            step=thought_step,
                            strategy=thought_strategy,
                            score=thought_score,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store thought in vector store: {e}")

                def _handle_task_exception(task: asyncio.Task[None]) -> None:
                    """Log exceptions from background tasks to prevent silent failures."""
                    if task.cancelled():
                        return
                    if exc := task.exception():
                        logger.warning(f"Background vector store task failed: {exc}")

                # Schedule without awaiting - continues immediately
                task = asyncio.create_task(_store_thought_background())
                task.add_done_callback(_handle_task_exception)

            # Knowledge Graph: Extract entities/relations from thought
            # OPT3: Use cached extractor instance + OPT5: Skip short thoughts
            kg_info: dict[str, Any] = {}
            if session_id in self._session_kgs and len(content) >= 50:
                try:
                    kg = self._session_kgs[session_id]
                    entities_before = len(kg.entities)
                    relations_before = len(kg.relations)

                    # Lazy-initialize cached extractor
                    if self._kg_extractor is None:
                        self._kg_extractor = KnowledgeGraphExtractor()
                    self._kg_extractor.extract(content, existing_graph=kg)

                    # Track what was newly extracted
                    new_entities = len(kg.entities) - entities_before
                    new_relations = len(kg.relations) - relations_before

                    if new_entities > 0 or new_relations > 0:
                        # Surface newly learned facts to LLM
                        kg_info["new_entities"] = new_entities
                        kg_info["new_relations"] = new_relations
                        kg_info["total_entities"] = len(kg.entities)
                        kg_info["total_relations"] = len(kg.relations)

                        # Get the most recent entities (what we just learned)
                        # kg.entities is already a list
                        recent_entities = kg.entities[-new_entities:] if new_entities > 0 else []
                        if recent_entities:
                            kg_info["learned"] = [
                                {"name": e.name, "type": e.entity_type.value}
                                for e in recent_entities[:5]  # Cap at 5 to avoid bloat
                            ]

                    # Check for facts that support current thought
                    supporting = kg.get_supporting_facts(content)
                    if supporting:
                        kg_info["supporting_facts"] = [
                            f"{r.subject.name} {r.predicate_value} {r.object_entity.name}"
                            for r in supporting[:3]
                        ]

                    # Check for facts that CONTRADICT the current thought
                    # Get newly extracted relations to check against existing KG
                    new_rels = kg.relations[-new_relations:] if new_relations > 0 else []
                    contradictions = kg.get_contradicting_facts(content, new_rels)
                    if contradictions:
                        kg_info["conflicting_facts"] = []
                        for existing_rel, new_rel, reason in contradictions[:3]:
                            # Get confidence-weighted resolution
                            new_conf = new_rel.confidence if new_rel != existing_rel else 0.5
                            resolution = kg.resolve_contradiction(
                                existing_rel, new_rel if new_rel != existing_rel else None, new_conf
                            )

                            conflict_entry = {
                                "existing": f"{existing_rel.subject.name} {existing_rel.predicate_value} {existing_rel.object_entity.name}",
                                "reason": reason,
                                "resolution": resolution,
                            }
                            kg_info["conflicting_facts"].append(conflict_entry)

                        logger.warning(
                            f"KG contradiction detected in session {session_id}: {len(contradictions)} conflicts"
                        )

                    logger.debug(
                        f"KG updated for session {session_id}: {len(kg.entities)} entities, "
                        f"{len(kg.relations)} relations (+{new_entities} entities, +{new_relations} relations)"
                    )
                except Exception as e:
                    logger.warning(f"KG extraction failed for thought: {e}")

        # Build response with OPT4 verbosity optimization
        # MINIMAL: Essential fields only (~50% token reduction)
        # NORMAL: Add guidance and detected issues
        # FULL: All fields including debugging info
        response: dict[str, Any] = {
            "session_id": session_id,
            "thought_id": thought_id,
            "step": step_number,
            "survival_score": round(survival_score, 3),
        }

        # Always include confidence if provided
        if confidence is not None:
            response["confidence"] = round(confidence, 3)

        # CRITICAL: Always surface detected issues regardless of verbosity
        # These are actionable and prevent reasoning errors
        if blind_spots_found:
            response["blind_spots"] = [bs.description for bs in blind_spots_found]

        if kg_info.get("conflicting_facts"):
            # Compact contradiction format
            response["conflicts"] = [f["reason"] for f in kg_info["conflicting_facts"]]

        # NORMAL and FULL: Add guidance and RAG context
        if verbosity != ResponseVerbosity.MINIMAL:
            response["type"] = thought_type.value
            response["is_planning"] = is_planning

            # RAG context that helps reasoning (proven +78% improvement)
            # Include scores for confidence-weighted prompt injection (S2)
            if rag_info.get("similar_thoughts"):
                response["similar"] = [
                    {"text": t["text"][:500], "score": t["relevance"]}
                    for t in rag_info["similar_thoughts"][:2]
                ]
            if rag_info.get("relevant_facts"):
                response["facts"] = [
                    {"text": f["fact"][:500], "score": f["relevance"]}
                    for f in rag_info["relevant_facts"][:2]
                ]

            # Compact guidance
            response["guidance"] = self._get_next_step_guidance(session, thought)

        # FULL: Include all debugging and tracking info
        if verbosity == ResponseVerbosity.FULL:
            if cisc_result:
                response["mppa"] = {
                    "alternatives_evaluated": len(alternatives) + 1 if alternatives else 1,
                    "selected_index": cisc_result.selected_index,
                    "cisc_weights": [round(w, 3) for w in cisc_result.normalized_weights],
                }

            if rewards_earned:
                response["rewards"] = [
                    {"type": r.reward_type, "value": round(r.value, 2), "reason": r.reason}
                    for r in rewards_earned
                ]
                response["total_reward"] = round(sum(r.value for r in rewards_earned), 2)

            if graph_info:
                response["graph"] = graph_info
                if graph_info.get("contradictions", 0) > 0:
                    response["contradiction_resolution"] = self._get_contradiction_guidance(
                        session, thought, thought.contradicts
                    )

            if domain_validation:
                response["domain_validation"] = domain_validation

            # Full RAG info
            if rag_info:
                response["rag"] = rag_info

            # Full KG info
            if kg_info:
                response["knowledge_graph"] = kg_info

        return response

    async def synthesize(
        self,
        session_id: str,
        col: int,
        content: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Synthesize a matrix column.

        Args:
            session_id: Session ID.
            col: Column to synthesize.
            content: Synthesis content.
            confidence: Confidence in synthesis.

        Returns:
            Synthesis result.

        """
        async with self.session(session_id) as session:
            if session.actual_mode not in (ReasoningMode.MATRIX, ReasoningMode.HYBRID):
                raise ValueError("Synthesis only available in matrix/hybrid mode")

            thought_id = f"syn_{uuid.uuid4().hex[:8]}"
            step_number = session.current_step + 1

            thought = Thought(
                id=thought_id,
                content=content,
                thought_type=ThoughtType.SYNTHESIS,
                step_number=step_number,
                confidence=confidence,
                survival_score=self._score_thought(
                    content, context=session.context, position=col, session_id=session_id
                ),
            )

            session.thoughts[thought_id] = thought
            session.thought_order.append(thought_id)
            session.syntheses[col] = thought_id
            session.updated_at = datetime.now()

        return {
            "session_id": session_id,
            "thought_id": thought_id,
            "column_synthesized": col,
            "syntheses_complete": len(session.syntheses),
            "total_columns": session.matrix_cols,
        }

    async def finalize(
        self,
        session_id: str,
        answer: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Finalize the reasoning session with an answer.

        Args:
            session_id: Session ID.
            answer: The final answer.
            confidence: Confidence in the answer.

        Returns:
            Final session state with summary.

        """
        async with self.session(session_id) as session:
            # Add final thought
            thought_id = f"final_{uuid.uuid4().hex[:8]}"
            step_number = session.current_step + 1

            thought = Thought(
                id=thought_id,
                content=answer,
                thought_type=ThoughtType.FINAL,
                step_number=step_number,
                confidence=confidence,
                survival_score=self._score_thought(
                    answer, context=session.context, position=step_number, session_id=session_id
                ),
            )

            session.thoughts[thought_id] = thought
            session.thought_order.append(thought_id)
            session.final_answer = answer
            session.final_confidence = confidence
            session.status = SessionStatus.COMPLETED
            session.updated_at = datetime.now()

        # Compute final metrics
        total_reward = sum(r.value for r in session.rewards)
        unaddressed_blind_spots = [bs for bs in session.blind_spots if not bs.addressed]

        result: dict[str, Any] = {
            "session_id": session_id,
            "status": "completed",
            "final_answer": answer,
            "confidence": confidence,
            "total_steps": session.current_step,
            "mode_used": session.actual_mode.value,
            "domain": session.domain.value,
            "complexity": session.complexity.complexity_level,
            "summary": {
                "thoughts": len(session.thoughts),
                "branches": len(session.branches),
                "planning_steps": session.planning_steps,
                "mppa_explorations": session.mppa_explorations,
                "rag_retrievals": session.rag_retrievals,
            },
        }

        if self._enable_rewards:
            result["total_reward"] = round(total_reward, 2)

        if self._enable_blind_spots and unaddressed_blind_spots:
            result["unaddressed_blind_spots"] = [bs.to_dict() for bs in unaddressed_blind_spots]
            result["warning"] = f"{len(unaddressed_blind_spots)} blind spot(s) were not addressed"

        # Add graph analysis summary
        if self._enable_graph and session.thought_graph is not None:
            result["graph_analysis"] = {
                "contradictions_found": len(session.graph_contradictions),
                "cycles_found": len(session.graph_cycles),
            }
            if session.graph_contradictions:
                result["contradictions"] = [
                    {"thought1": c[0], "thought2": c[1]}
                    for c in session.graph_contradictions[:5]  # Limit to 5
                ]

        # Add domain validation summary
        if self._enable_domain_validation and session.domain_validations:
            valid_count = sum(
                1 for v in session.domain_validations.values() if v.get("result") == "valid"
            )
            result["domain_validation_summary"] = {
                "validated_thoughts": len(session.domain_validations),
                "valid_count": valid_count,
                "invalid_count": len(session.domain_validations) - valid_count,
            }

        # Add Knowledge Graph summary and clean up
        if session_id in self._session_kgs:
            kg = self._session_kgs[session_id]
            result["knowledge_graph_summary"] = {
                "entities_extracted": len(kg.entities),
                "relations_found": len(kg.relations),
            }

            # Persist KG facts to vector store for cross-session learning
            # Only persist if session was successful (confidence >= 0.6)
            if (
                self._enable_rag
                and self._vector_store is not None
                and (confidence is None or confidence >= 0.6)
                and (len(kg.entities) > 0 or len(kg.relations) > 0)
            ):
                try:
                    entities_data = [e.to_dict() for e in kg.entities]
                    relations_data = [r.to_dict() for r in kg.relations]
                    stored_ids = await self._vector_store.add_kg_facts(
                        session_id=session_id,
                        entities=entities_data,
                        relations=relations_data,
                        domain=session.domain.value,
                        confidence=confidence,
                    )
                    result["knowledge_graph_summary"]["facts_persisted"] = len(stored_ids)
                    logger.debug(f"Persisted {len(stored_ids)} KG facts from session {session_id}")
                except Exception as e:
                    logger.warning(f"Failed to persist KG facts: {e}")

            # Clean up to prevent memory leak
            del self._session_kgs[session_id]
            logger.debug(f"Cleaned up KG for session {session_id}")

        return result

    async def get_status(self, session_id: str) -> dict[str, Any]:
        """Get current session status.

        Args:
            session_id: Session ID.

        Returns:
            Current session state.

        """
        async with self.session(session_id) as session:
            return session.to_dict()

    async def analyze_session(self, session_id: str) -> SessionAnalytics:
        """Analyze a session and return consolidated metrics.

        This provides computed insights and recommendations without
        duplicating raw session data from get_status()/to_dict().

        Args:
            session_id: Session ID to analyze.

        Returns:
            SessionAnalytics with computed metrics and recommendations.

        """
        async with self.session(session_id) as session:
            return self._analyze_session_unlocked(session)

    def _analyze_session_unlocked(self, session: ReasoningSession) -> SessionAnalytics:
        """Analyze session without acquiring lock (caller must hold lock).

        Internal method for use when the caller already holds the session lock.
        This prevents deadlocks when methods need to call analyze_session
        while already holding the lock.

        Args:
            session: The ReasoningSession object (caller must hold lock).

        Returns:
            SessionAnalytics with computed metrics and recommendations.

        """
        session_id = session.session_id
        # Calculate progress metrics
        total_thoughts = len(session.thoughts)
        main_chain = session.main_chain
        main_chain_length = len(main_chain)
        branch_count = len(session.branches)

        # Calculate average confidence and survival score
        confidences = [t.confidence for t in session.thoughts.values() if t.confidence is not None]
        survival_scores = [
            t.survival_score for t in session.thoughts.values() if t.survival_score is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        avg_survival = sum(survival_scores) / len(survival_scores) if survival_scores else None

        # Calculate coherence score (based on graph connectivity)
        coherence = self._calculate_coherence(session)

        # Calculate coverage score (problem space coverage estimate)
        coverage = self._calculate_coverage(session)

        # Calculate depth score (reasoning depth vs problem complexity)
        depth = self._calculate_depth(session)

        # Gather contradictions
        contradictions = list(session.graph_contradictions)

        # Count unresolved contradictions (thoughts with contradicts that aren't marked resolved)
        unresolved = sum(
            1
            for t in session.thoughts.values()
            if t.contradicts and t.thought_type not in (ThoughtType.REVISION, ThoughtType.SYNTHESIS)
        )

        # Blind spot tracking
        blind_spots_detected = len(session.blind_spots)
        blind_spots_unaddressed = sum(1 for bs in session.blind_spots if not bs.addressed)

        # Cycles detected
        cycles = len(session.graph_cycles)

        # Domain validation stats
        validation_rate: float | None = None
        invalid_thoughts: list[str] = []
        if session.domain_validations:
            valid_count = sum(
                1 for v in session.domain_validations.values() if v.get("result") == "valid"
            )
            validation_rate = valid_count / len(session.domain_validations)
            invalid_thoughts = [
                tid for tid, v in session.domain_validations.items() if v.get("result") != "valid"
            ]

        # Efficiency metrics
        planning_steps = session.planning_steps
        planning_ratio = planning_steps / total_thoughts if total_thoughts > 0 else 0.0
        revision_count = sum(
            1 for t in session.thoughts.values() if t.thought_type == ThoughtType.REVISION
        )
        syntheses_count = sum(
            1 for t in session.thoughts.values() if t.thought_type == ThoughtType.SYNTHESIS
        )
        branch_utilization = syntheses_count / branch_count if branch_count > 0 else 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            session=session,
            coherence=coherence,
            coverage=coverage,
            depth=depth,
            blind_spots_unaddressed=blind_spots_unaddressed,
            unresolved_contradictions=unresolved,
            validation_rate=validation_rate,
        )

        # Assess risk level
        risk_level, risk_factors = self._assess_risk(
            session=session,
            coherence=coherence,
            blind_spots_unaddressed=blind_spots_unaddressed,
            unresolved_contradictions=unresolved,
            invalid_thoughts=invalid_thoughts,
        )

        return SessionAnalytics(
            session_id=session_id,
            total_thoughts=total_thoughts,
            main_chain_length=main_chain_length,
            branch_count=branch_count,
            average_confidence=avg_confidence,
            average_survival_score=avg_survival,
            coherence_score=coherence,
            coverage_score=coverage,
            depth_score=depth,
            contradictions=contradictions,
            unresolved_contradictions=unresolved,
            blind_spots_detected=blind_spots_detected,
            blind_spots_unaddressed=blind_spots_unaddressed,
            cycles_detected=cycles,
            validation_rate=validation_rate,
            invalid_thoughts=invalid_thoughts,
            planning_ratio=planning_ratio,
            revision_count=revision_count,
            branch_utilization=branch_utilization,
            recommendations=recommendations,
            risk_level=risk_level,
            risk_factors=risk_factors,
        )

    def _calculate_coherence(self, session: ReasoningSession) -> float:
        """Calculate coherence score based on thought connectivity.

        Higher score means thoughts are well-connected with explicit relationships.
        """
        if not session.thoughts:
            return 0.0

        total_thoughts = len(session.thoughts)
        connected_thoughts = 0

        for thought in session.thoughts.values():
            # A thought is "connected" if it has parent, supports, or is supported
            has_connection = (
                thought.parent_id is not None
                or len(thought.supports) > 0
                or len(thought.related_to) > 0
                or any(thought.id in t.supports for t in session.thoughts.values())
            )
            if has_connection:
                connected_thoughts += 1

        # Base connectivity score
        connectivity = connected_thoughts / total_thoughts if total_thoughts > 0 else 0.0

        # Penalty for contradictions
        contradiction_penalty = min(0.3, len(session.graph_contradictions) * 0.1)

        # Penalty for cycles (circular reasoning)
        cycle_penalty = min(0.2, len(session.graph_cycles) * 0.1)

        return max(0.0, connectivity - contradiction_penalty - cycle_penalty)

    def _calculate_coverage(self, session: ReasoningSession) -> float:
        """Estimate problem space coverage.

        Based on:
        - Number of branches explored
        - Variety of thought types used
        - Planning steps taken (for complex problems)
        """
        if not session.thoughts:
            return 0.0

        # Branch diversity (more branches = more coverage for complex problems)
        complexity_level = session.complexity.complexity_level
        expected_branches = {
            "trivial": 0,
            "simple": 0,
            "moderate": 1,
            "complex": 2,
            "expert": 3,
        }.get(complexity_level, 1)
        branch_score = (
            min(1.0, len(session.branches) / expected_branches) if expected_branches > 0 else 1.0
        )

        # Thought type diversity
        thought_types_used = {t.thought_type for t in session.thoughts.values()}
        core_types = {ThoughtType.INITIAL, ThoughtType.CONTINUATION, ThoughtType.FINAL}
        advanced_types = {
            ThoughtType.REVISION,
            ThoughtType.BRANCH,
            ThoughtType.SYNTHESIS,
            ThoughtType.VERIFICATION,
            ThoughtType.PLANNING,
        }
        core_coverage = len(thought_types_used & core_types) / len(core_types)
        advanced_coverage = len(thought_types_used & advanced_types) / len(advanced_types)
        type_score = 0.6 * core_coverage + 0.4 * advanced_coverage

        # Planning adequacy for complex problems
        planning_score = 1.0
        if complexity_level in ("complex", "expert") and session.planning_steps == 0:
            planning_score = 0.6  # Penalty for no planning on complex problems

        return (branch_score + type_score + planning_score) / 3

    def _calculate_depth(self, session: ReasoningSession) -> float:
        """Calculate reasoning depth score.

        Compares actual reasoning depth to expected depth based on complexity.
        """
        if not session.thoughts:
            return 0.0

        main_chain_length = len(session.main_chain)
        complexity_level = session.complexity.complexity_level

        # Expected minimum chain length by complexity
        expected_depth = {
            "trivial": 2,
            "simple": 3,
            "moderate": 5,
            "complex": 8,
            "expert": 12,
        }.get(complexity_level, 5)

        # Score is 1.0 if we meet or exceed expected depth
        depth_ratio = main_chain_length / expected_depth if expected_depth > 0 else 1.0

        # Don't penalize for exceeding, but cap at 1.0
        # Slight penalty if way under expected depth
        if depth_ratio >= 1.0:
            return 1.0
        elif depth_ratio >= 0.7:
            return 0.8 + (depth_ratio - 0.7) * (0.2 / 0.3)
        else:
            return depth_ratio * (0.8 / 0.7)

    def _generate_recommendations(
        self,
        session: ReasoningSession,
        coherence: float,
        coverage: float,
        depth: float,
        blind_spots_unaddressed: int,
        unresolved_contradictions: int,
        validation_rate: float | None,
    ) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations: list[str] = []

        # Coherence recommendations
        if coherence < 0.5:
            recommendations.append(
                "Add explicit thought connections using 'supports' or 'related_to' to improve coherence"
            )

        # Contradiction recommendations
        if unresolved_contradictions > 0:
            recommendations.append(
                f"Resolve {unresolved_contradictions} contradiction(s) using 'resolve' action with 'revise' or 'reconcile' strategy"
            )

        # Blind spot recommendations
        if blind_spots_unaddressed > 0:
            recommendations.append(
                f"Address {blind_spots_unaddressed} detected blind spot(s) before finalizing"
            )

        # Coverage recommendations
        if coverage < 0.5 and session.complexity.complexity_level in ("complex", "expert"):
            recommendations.append(
                "Consider exploring alternative branches for better problem coverage"
            )

        # Depth recommendations
        if depth < 0.7:
            recommendations.append(
                "Reasoning chain may be shallow for problem complexity; consider deeper analysis"
            )

        # Planning recommendations
        if (
            session.complexity.complexity_level in ("complex", "expert")
            and session.planning_steps == 0
        ):
            recommendations.append(
                "Add planning step(s) for complex problems to structure the approach"
            )

        # Validation recommendations
        if validation_rate is not None and validation_rate < 0.8:
            recommendations.append(
                f"Domain validation rate is {validation_rate:.0%}; review invalid thoughts for errors"
            )

        # Branch utilization
        if len(session.branches) > 0 and not session.syntheses:
            recommendations.append(
                "Consider adding synthesis thoughts to integrate branch explorations"
            )

        # If all is well
        if not recommendations:
            recommendations.append("Session analysis shows good reasoning quality")

        return recommendations

    def _assess_risk(
        self,
        session: ReasoningSession,
        coherence: float,
        blind_spots_unaddressed: int,
        unresolved_contradictions: int,
        invalid_thoughts: list[str],
    ) -> tuple[Literal["low", "medium", "high"], list[str]]:
        """Assess overall risk level of the reasoning session."""
        risk_factors: list[str] = []
        risk_score = 0

        # Unresolved contradictions are high risk
        if unresolved_contradictions > 0:
            risk_factors.append(f"{unresolved_contradictions} unresolved contradiction(s)")
            risk_score += unresolved_contradictions * 2

        # Unaddressed blind spots
        if blind_spots_unaddressed > 0:
            risk_factors.append(f"{blind_spots_unaddressed} unaddressed blind spot(s)")
            risk_score += blind_spots_unaddressed

        # Low coherence
        if coherence < 0.4:
            risk_factors.append(f"Low coherence score ({coherence:.2f})")
            risk_score += 2

        # Invalid domain validations
        if invalid_thoughts:
            risk_factors.append(f"{len(invalid_thoughts)} thought(s) failed domain validation")
            risk_score += len(invalid_thoughts)

        # Low confidence on final answer
        if session.final_confidence is not None and session.final_confidence < 0.5:
            risk_factors.append(f"Low final confidence ({session.final_confidence:.2f})")
            risk_score += 2

        # Cycles in reasoning
        if session.graph_cycles:
            risk_factors.append(f"{len(session.graph_cycles)} circular reasoning pattern(s)")
            risk_score += len(session.graph_cycles)

        # Determine risk level
        if risk_score >= 5:
            return "high", risk_factors
        elif risk_score >= 2:
            return "medium", risk_factors
        else:
            return "low", risk_factors

    async def suggest_next_action_stream(
        self,
        session_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream suggestions as they are computed (S1).

        Yields suggestions progressively, allowing the LLM to start acting
        on high-priority items immediately.

        Args:
            session_id: Session ID to analyze.

        Yields:
            Suggestion dictionaries in priority order.

        """
        async with self.session(session_id) as session:
            # Yield immediate high-priority checks first
            # 1. Check for unresolved contradictions (fast check)
            unresolved_count = sum(
                1
                for t in session.thoughts.values()
                if t.contradicts
                and t.thought_type not in (ThoughtType.REVISION, ThoughtType.SYNTHESIS)
            )

            if unresolved_count > 0:
                contradicting_id = None
                for tid in reversed(session.thought_order):
                    t = session.thoughts[tid]
                    if t.contradicts:
                        contradicting_id = tid
                        break

                yield {
                    "type": "suggestion",
                    "action": "resolve",
                    "priority": 1,
                    "urgency": "high",
                    "reason": f"Found {unresolved_count} unresolved contradiction(s)",
                    "parameters": {
                        "resolve_strategy": "revise",
                        "contradicting_thought_id": contradicting_id,
                    },
                    "guidance": "Use 'revise' to correct, 'reconcile' to synthesize, or 'branch' to explore both",
                    "is_final": False,
                }

            # 2. Check blind spots (fast check)
            unaddressed_blind_spots = [bs for bs in session.blind_spots if not bs.addressed]
            if unaddressed_blind_spots:
                bs = unaddressed_blind_spots[0]
                yield {
                    "type": "suggestion",
                    "action": "continue",
                    "priority": 2,
                    "urgency": "medium",
                    "reason": f"Found {len(unaddressed_blind_spots)} unaddressed blind spot(s)",
                    "parameters": {"thought_focus": bs.description},
                    "guidance": f"Address: {bs.suggested_action}",
                    "is_final": False,
                }

            # 3. Now do the full analysis (use unlocked version since we hold the lock)
            yield {"type": "status", "message": "Computing full analysis..."}
            analytics = self._analyze_session_unlocked(session)

            # 3. Low-confidence thoughts
            low_confidence_thoughts = [
                t
                for t in session.thoughts.values()
                if t.confidence is not None
                and t.confidence < 0.6
                and t.thought_type not in (ThoughtType.VERIFICATION, ThoughtType.FINAL)
            ]
            if low_confidence_thoughts and not any(
                t.thought_type == ThoughtType.VERIFICATION for t in session.thoughts.values()
            ):
                weakest = min(low_confidence_thoughts, key=lambda t: t.confidence or 0)
                yield {
                    "type": "suggestion",
                    "action": "verify",
                    "priority": 3,
                    "urgency": "medium",
                    "reason": f"Found {len(low_confidence_thoughts)} low-confidence thought(s)",
                    "parameters": {
                        "target_thought_id": weakest.id,
                        "current_confidence": weakest.confidence,
                    },
                    "guidance": f"Verify: '{weakest.content[:50]}...'",
                    "is_final": False,
                }

            # 4. Depth check
            if analytics.depth_score < 0.6:
                expected_steps_map = {"low": 3, "medium": 5, "medium-high": 7, "high": 10}
                expected_steps = expected_steps_map.get(session.complexity.complexity_level, 5)
                expected_more = max(1, int(expected_steps * 0.7) - analytics.main_chain_length)
                yield {
                    "type": "suggestion",
                    "action": "continue",
                    "priority": 4,
                    "urgency": "low",
                    "reason": f"Reasoning depth ({analytics.depth_score:.2f}) below recommended",
                    "parameters": {"recommended_additional_steps": expected_more},
                    "guidance": f"Consider {expected_more} more step(s)",
                    "is_final": False,
                }

            # 5. Branch synthesis
            if analytics.branch_count > 0 and analytics.branch_utilization < 0.5:
                yield {
                    "type": "suggestion",
                    "action": "synthesize",
                    "priority": 5,
                    "urgency": "low",
                    "reason": f"{analytics.branch_count} branch(es) without synthesis",
                    "parameters": {"branches_to_synthesize": list(session.branches.keys())},
                    "guidance": "Synthesize insights from branches",
                    "is_final": False,
                }

            # 6. Ready to finish
            is_ready = (
                analytics.unresolved_contradictions == 0
                and analytics.blind_spots_unaddressed == 0
                and analytics.depth_score >= 0.7
                and analytics.risk_level != "high"
                and session.status == SessionStatus.ACTIVE
            )
            if is_ready:
                avg_conf = analytics.average_confidence or 0.5
                yield {
                    "type": "suggestion",
                    "action": "finish",
                    "priority": 6,
                    "urgency": "low",
                    "reason": "Session ready for conclusion",
                    "parameters": {"recommended_confidence": round(avg_conf, 2)},
                    "guidance": f"Provide final answer with confidence ~{avg_conf:.2f}",
                    "is_final": False,
                }

            # Final summary
            yield {
                "type": "complete",
                "session_summary": {
                    "thoughts": analytics.total_thoughts,
                    "quality": round(
                        (
                            analytics.coherence_score
                            + analytics.coverage_score
                            + analytics.depth_score
                        )
                        / 3,
                        2,
                    ),
                    "risk": analytics.risk_level,
                    "status": session.status.value,
                },
                "is_final": True,
            }

    async def suggest_next_action(self, session_id: str, record: bool = True) -> dict[str, Any]:
        """Suggest the next best action based on session analysis.

        Analyzes current session state and returns a prioritized suggestion
        for what the LLM should do next, reducing cognitive load.

        Args:
            session_id: Session ID to analyze.
            record: Whether to record this suggestion in history (S2).

        Returns:
            Dictionary with suggested action, parameters, and reasoning.

        """
        async with self.session(session_id) as session:
            # Get analytics using unlocked version (we already hold the lock)
            analytics = self._analyze_session_unlocked(session)
            return self._suggest_next_action_unlocked(session, record, analytics)

    def _suggest_next_action_unlocked(
        self,
        session: ReasoningSession,
        record: bool,
        analytics: SessionAnalytics,
    ) -> dict[str, Any]:
        """Suggest next action without acquiring lock (caller must hold lock).

        Internal method for use when the caller already holds the session lock.
        This prevents deadlocks when methods need to call suggest_next_action
        while already holding the lock.

        Args:
            session: The ReasoningSession object (caller must hold lock).
            record: Whether to record this suggestion in history.
            analytics: Pre-computed SessionAnalytics.

        Returns:
            Dictionary with suggested action, parameters, and reasoning.

        """
        session_id = session.session_id
        # Get learned weights for prioritization (S2)
        weights = session.suggestion_weights

        # Priority order for suggestions:
        # 1. Resolve contradictions (highest priority - correctness)
        # 2. Address blind spots (gaps in reasoning)
        # 3. Add verification for low-confidence thoughts
        # 4. Deepen reasoning if shallow
        # 5. Add synthesis if branches exist without synthesis
        # 6. Finish if ready
        # 7. Continue reasoning

        suggestions: list[dict[str, Any]] = []

        # 1. Check for unresolved contradictions
        if analytics.unresolved_contradictions > 0:
            # Find the most recent contradicting thought
            contradicting_id = None
            for tid in reversed(session.thought_order):
                t = session.thoughts[tid]
                if t.contradicts:
                    contradicting_id = tid
                    break

            base_priority = 1
            adjusted_priority = base_priority / weights.get_weight("resolve")
            suggestions.append(
                {
                    "action": "resolve",
                    "priority": adjusted_priority,
                    "base_priority": base_priority,
                    "urgency": "high",
                    "reason": f"Found {analytics.unresolved_contradictions} unresolved contradiction(s)",
                    "parameters": {
                        "resolve_strategy": "revise",  # Default to revise
                        "contradicting_thought_id": contradicting_id,
                    },
                    "guidance": "Use 'revise' to correct, 'reconcile' to synthesize, or 'branch' to explore both possibilities",
                }
            )

        # 2. Check for unaddressed blind spots
        if analytics.blind_spots_unaddressed > 0:
            # Find the first unaddressed blind spot
            unaddressed = next((bs for bs in session.blind_spots if not bs.addressed), None)
            base_priority = 2
            adjusted_priority = base_priority / weights.get_weight("continue", "blind spot")
            suggestions.append(
                {
                    "action": "continue",
                    "priority": adjusted_priority,
                    "base_priority": base_priority,
                    "urgency": "medium",
                    "reason": f"Found {analytics.blind_spots_unaddressed} unaddressed blind spot(s)",
                    "parameters": {
                        "thought_focus": unaddressed.description if unaddressed else "address gaps",
                    },
                    "guidance": f"Address: {unaddressed.suggested_action}"
                    if unaddressed
                    else "Explore gaps in reasoning",
                }
            )

        # 3. Check for low-confidence thoughts needing verification
        low_confidence_thoughts = [
            t
            for t in session.thoughts.values()
            if t.confidence is not None
            and t.confidence < 0.6
            and t.thought_type not in (ThoughtType.VERIFICATION, ThoughtType.FINAL)
        ]
        if low_confidence_thoughts and not any(
            t.thought_type == ThoughtType.VERIFICATION for t in session.thoughts.values()
        ):
            weakest = min(low_confidence_thoughts, key=lambda t: t.confidence or 0)
            base_priority = 3
            adjusted_priority = base_priority / weights.get_weight("verify")
            suggestions.append(
                {
                    "action": "verify",
                    "priority": adjusted_priority,
                    "base_priority": base_priority,
                    "urgency": "medium",
                    "reason": f"Found {len(low_confidence_thoughts)} low-confidence thought(s)",
                    "parameters": {
                        "target_thought_id": weakest.id,
                        "current_confidence": weakest.confidence,
                    },
                    "guidance": f"Verify thought: '{weakest.content[:50]}...' (confidence: {weakest.confidence:.2f})",
                }
            )

        # 4. Check if reasoning is too shallow for complexity
        if analytics.depth_score < 0.6:
            # Estimate expected steps based on complexity level
            expected_steps_map = {
                "low": 3,
                "medium": 5,
                "medium-high": 7,
                "high": 10,
            }
            expected_steps = expected_steps_map.get(session.complexity.complexity_level, 5)
            expected_more = max(1, int(expected_steps * 0.7) - analytics.main_chain_length)
            base_priority = 4
            adjusted_priority = base_priority / weights.get_weight("continue", "depth")
            suggestions.append(
                {
                    "action": "continue",
                    "priority": adjusted_priority,
                    "base_priority": base_priority,
                    "urgency": "low",
                    "reason": f"Reasoning depth ({analytics.depth_score:.2f}) below recommended for {session.complexity.complexity_level} problem",
                    "parameters": {
                        "recommended_additional_steps": expected_more,
                    },
                    "guidance": f"Consider {expected_more} more reasoning step(s) before concluding",
                }
            )

        # 5. Check for branches without synthesis
        if analytics.branch_count > 0 and analytics.branch_utilization < 0.5:
            base_priority = 5
            adjusted_priority = base_priority / weights.get_weight("synthesize")
            suggestions.append(
                {
                    "action": "synthesize",
                    "priority": adjusted_priority,
                    "base_priority": base_priority,
                    "urgency": "low",
                    "reason": f"{analytics.branch_count} branch(es) exist without synthesis",
                    "parameters": {
                        "branches_to_synthesize": list(session.branches.keys()),
                    },
                    "guidance": "Synthesize insights from explored branches",
                }
            )

        # 6. Check if ready to finish
        is_ready_to_finish = (
            analytics.unresolved_contradictions == 0
            and analytics.blind_spots_unaddressed == 0
            and analytics.depth_score >= 0.7
            and analytics.risk_level != "high"
            and session.status == SessionStatus.ACTIVE
        )

        if is_ready_to_finish:
            avg_conf = analytics.average_confidence or 0.5
            base_priority = 6
            adjusted_priority = base_priority / weights.get_weight("finish")
            suggestions.append(
                {
                    "action": "finish",
                    "priority": adjusted_priority,
                    "base_priority": base_priority,
                    "urgency": "low",
                    "reason": "Session appears ready for conclusion",
                    "parameters": {
                        "recommended_confidence": round(avg_conf, 2),
                    },
                    "guidance": f"Provide final answer with confidence ~{avg_conf:.2f}",
                }
            )

        # 7. Default: continue reasoning
        if not suggestions or (len(suggestions) == 1 and suggestions[0]["action"] == "finish"):
            # Get the last thought for context
            last_thought = (
                session.thoughts[session.thought_order[-1]] if session.thought_order else None
            )
            base_priority = 7
            adjusted_priority = base_priority / weights.get_weight("continue")
            suggestions.append(
                {
                    "action": "continue",
                    "priority": adjusted_priority,
                    "base_priority": base_priority,
                    "urgency": "low",
                    "reason": "Continue building reasoning chain",
                    "parameters": {},
                    "guidance": f"Build on: '{last_thought.content[:50]}...'"
                    if last_thought
                    else "Add first reasoning step",
                }
            )

        # Sort by adjusted priority and return top suggestion with alternatives
        suggestions.sort(key=lambda s: s["priority"])
        top_suggestion = suggestions[0]
        alternatives = suggestions[1:4]  # Up to 3 alternatives

        # Record suggestion in history (S2)
        if record:
            suggestion_id = f"sug_{uuid.uuid4().hex[:8]}"
            record_entry = SuggestionRecord(
                suggestion_id=suggestion_id,
                action=top_suggestion["action"],
                urgency=top_suggestion["urgency"],
                reason=top_suggestion["reason"],
            )
            session.suggestion_history.append(record_entry)
        else:
            suggestion_id = None

        return {
            "session_id": session_id,
            "suggestion_id": suggestion_id,
            "suggested_action": top_suggestion["action"],
            "urgency": top_suggestion["urgency"],
            "reason": top_suggestion["reason"],
            "parameters": top_suggestion["parameters"],
            "guidance": top_suggestion["guidance"],
            "alternatives": [
                {
                    "action": alt["action"],
                    "reason": alt["reason"],
                    "urgency": alt["urgency"],
                }
                for alt in alternatives
            ],
            "session_summary": {
                "thoughts": analytics.total_thoughts,
                "quality": round(
                    (analytics.coherence_score + analytics.coverage_score + analytics.depth_score)
                    / 3,
                    2,
                ),
                "risk": analytics.risk_level,
                "status": session.status.value,
            },
            "learning": {
                "weights_applied": weights.to_dict(),
                "history_size": len(session.suggestion_history),
            },
        }

    async def record_suggestion_outcome(
        self,
        session_id: str,
        suggestion_id: str,
        outcome: Literal["accepted", "rejected"],
        actual_action: str | None = None,
    ) -> dict[str, Any]:
        """Record the outcome of a suggestion for learning (S2).

        Args:
            session_id: Session ID.
            suggestion_id: ID of the suggestion to update.
            outcome: Whether the suggestion was accepted or rejected.
            actual_action: What action the user actually took (if different).

        Returns:
            Updated suggestion record and new weights.

        """
        async with self.session(session_id) as session:
            # Find the suggestion record
            record = None
            for rec in session.suggestion_history:
                if rec.suggestion_id == suggestion_id:
                    record = rec
                    break

            if record is None:
                raise ValueError(f"Suggestion not found: {suggestion_id}")

            # Update record
            record.outcome = outcome
            record.outcome_timestamp = datetime.now()
            record.actual_action = actual_action

            # Adjust weights based on outcome
            accepted = outcome == "accepted"
            session.suggestion_weights.adjust(record.action, accepted)

            # If rejected and user took a different action, boost that action
            if not accepted and actual_action and actual_action != record.action:
                session.suggestion_weights.adjust(actual_action, True, learning_rate=0.05)

            # Persist weights if enabled
            if self._enable_weight_persistence and self._weight_store is not None:
                try:
                    self._weight_store.record_feedback(
                        domain=session.domain.value,
                        session_id=session_id,
                        suggestion_id=suggestion_id,
                        suggested_action=record.action,
                        outcome=outcome,
                        actual_action=actual_action,
                    )
                    logger.debug(f"Persisted feedback for domain {session.domain.value}")
                except Exception as e:
                    logger.warning(f"Failed to persist feedback: {e}")

            return {
                "suggestion_id": suggestion_id,
                "outcome": outcome,
                "action": record.action,
                "actual_action": actual_action,
                "weights_updated": session.suggestion_weights.to_dict(),
            }

    async def auto_execute_suggestion(
        self,
        session_id: str,
        thought_generator: Callable[[str, dict[str, Any]], Awaitable[str]] | None = None,
        max_auto_steps: int = 5,
        stop_on_high_risk: bool = True,
    ) -> dict[str, Any]:
        """Automatically execute the top suggestion (S3).

        This enables autonomous reasoning loops with human-in-the-loop
        checkpoints at high-risk moments.

        Args:
            session_id: Session ID.
            thought_generator: Async function to generate thought content.
                              Takes (guidance, parameters) and returns thought text.
                              If None, returns suggestion without executing.
            max_auto_steps: Maximum number of auto-executions before stopping.
            stop_on_high_risk: Whether to stop and return when risk is high.

        Returns:
            Result of the executed action or checkpoint info if stopped.

        """
        async with self.session(session_id) as session:
            # Check if we've exceeded auto steps
            if session.auto_actions_executed >= max_auto_steps:
                # Use unlocked versions since we already hold the lock
                analytics = self._analyze_session_unlocked(session)
                suggestion = self._suggest_next_action_unlocked(
                    session, record=False, analytics=analytics
                )
                return {
                    "status": "checkpoint",
                    "reason": f"Reached max auto steps ({max_auto_steps})",
                    "auto_actions_executed": session.auto_actions_executed,
                    "suggestion": suggestion,
                }

            # Get suggestion using unlocked versions (we already hold the lock)
            analytics = self._analyze_session_unlocked(session)
            suggestion = self._suggest_next_action_unlocked(
                session, record=True, analytics=analytics
            )

            # Check for high-risk checkpoint
            if stop_on_high_risk and suggestion["session_summary"]["risk"] == "high":
                return {
                    "status": "checkpoint",
                    "reason": "High risk detected - human review recommended",
                    "risk_level": "high",
                    "suggestion": suggestion,
                }

            # If no thought generator, return suggestion for manual execution
            if thought_generator is None:
                return {
                    "status": "suggestion_only",
                    "reason": "No thought generator provided",
                    "suggestion": suggestion,
                }

            # Execute the suggested action
            action = suggestion["suggested_action"]
            params = suggestion["parameters"]
            guidance = suggestion["guidance"]

            # Mark suggestion as auto-executed
            if suggestion.get("suggestion_id"):
                for rec in session.suggestion_history:
                    if rec.suggestion_id == suggestion["suggestion_id"]:
                        rec.outcome = "auto_executed"
                        rec.outcome_timestamp = datetime.now()
                        break

            session.auto_actions_executed += 1
            session.auto_action_enabled = True
            # Store for use outside the lock
            auto_actions_count = session.auto_actions_executed

        # Execute action based on type
        try:
            if action == "continue":
                # Generate thought content
                thought_content = await thought_generator(guidance, params)
                result = await self.add_thought(
                    session_id=session_id,
                    content=thought_content,
                    thought_type=ThoughtType.CONTINUATION,
                    confidence=params.get("confidence", 0.7),
                )
                return {
                    "status": "executed",
                    "action": "continue",
                    "result": result,
                    "auto_actions_executed": auto_actions_count,
                }

            elif action == "verify":
                thought_content = await thought_generator(guidance, params)
                result = await self.add_thought(
                    session_id=session_id,
                    content=thought_content,
                    thought_type=ThoughtType.VERIFICATION,
                    confidence=params.get("confidence", 0.8),
                )
                return {
                    "status": "executed",
                    "action": "verify",
                    "result": result,
                    "auto_actions_executed": auto_actions_count,
                }

            elif action == "resolve":
                thought_content = await thought_generator(guidance, params)
                result = await self.resolve_contradiction(
                    session_id=session_id,
                    strategy=params.get("resolve_strategy", "revise"),
                    resolution_content=thought_content,
                    contradicting_thought_id=params.get("contradicting_thought_id"),
                )
                return {
                    "status": "executed",
                    "action": "resolve",
                    "result": result,
                    "auto_actions_executed": auto_actions_count,
                }

            elif action == "synthesize":
                thought_content = await thought_generator(guidance, params)
                # Synthesize requires col parameter for matrix mode
                # For chain mode, just add as synthesis thought
                result = await self.add_thought(
                    session_id=session_id,
                    content=thought_content,
                    thought_type=ThoughtType.SYNTHESIS,
                    confidence=params.get("confidence", 0.8),
                )
                return {
                    "status": "executed",
                    "action": "synthesize",
                    "result": result,
                    "auto_actions_executed": auto_actions_count,
                }

            elif action == "finish":
                thought_content = await thought_generator(guidance, params)
                result = await self.finalize(
                    session_id=session_id,
                    answer=thought_content,
                    confidence=params.get("recommended_confidence", 0.7),
                )
                return {
                    "status": "executed",
                    "action": "finish",
                    "result": result,
                    "auto_actions_executed": auto_actions_count,
                }

            else:
                return {
                    "status": "unsupported",
                    "reason": f"Action '{action}' not supported for auto-execution",
                    "suggestion": suggestion,
                }

        except Exception as e:
            logger.error(f"Auto-execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "action": action,
                "suggestion": suggestion,
            }

    async def retrieve_similar_thoughts(
        self,
        session_id: str,
        query: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve similar thoughts from vector store (RAG).

        Args:
            session_id: Session ID.
            query: Query text.
            k: Number of results.

        Returns:
            List of similar thoughts.

        """
        if not self._enable_rag or self._vector_store is None:
            return []

        async with self.session(session_id) as session:
            session.rag_retrievals += 1

        results = await self._vector_store.search_by_session(query, session_id, k=k)
        return [
            {
                "text": r.text,
                "score": round(r.score, 3),
                "metadata": r.metadata,
            }
            for r in results
        ]

    async def resolve_contradiction(
        self,
        session_id: str,
        strategy: str,
        resolution_content: str,
        contradicting_thought_id: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Resolve a detected contradiction using the specified strategy.

        Strategies:
            revise: Add a revised thought that corrects the contradiction
            branch: Create a new reasoning branch exploring the alternative
            reconcile: Add a synthesis thought that explains how both can be true
            backtrack: Mark the contradicting thought as abandoned and continue

        Args:
            session_id: Session ID.
            strategy: Resolution strategy (revise/branch/reconcile/backtrack).
            resolution_content: The thought content for the resolution.
            contradicting_thought_id: ID of the specific thought to address (optional).
            confidence: Confidence in the resolution (0-1).

        Returns:
            Resolution result with updated session state.

        """
        valid_strategies = {"revise", "branch", "reconcile", "backtrack"}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Valid: {valid_strategies}")

        async with self.session(session_id) as session:
            # Find the thought(s) involved in the contradiction
            if contradicting_thought_id:
                if contradicting_thought_id not in session.thoughts:
                    raise ValueError(f"Thought not found: {contradicting_thought_id}")
                target_thought = session.thoughts[contradicting_thought_id]
            else:
                # Find the most recent thought with contradictions
                target_thought = None
                for tid in reversed(session.thought_order):
                    t = session.thoughts[tid]
                    if t.contradicts:
                        target_thought = t
                        break

            if target_thought is None:
                # No contradictions to resolve - just add as normal thought
                return await self.add_thought(
                    session_id=session_id,
                    content=resolution_content,
                    confidence=confidence,
                )

            thought_id = f"resolve_{uuid.uuid4().hex[:8]}"
            step_number = session.current_step + 1

            # Apply strategy-specific logic
            if strategy == "revise":
                # Create a revision thought that supersedes the contradicting one
                thought = Thought(
                    id=thought_id,
                    content=resolution_content,
                    thought_type=ThoughtType.REVISION,
                    step_number=step_number,
                    confidence=confidence,
                    survival_score=self._score_thought(
                        resolution_content,
                        context=session.context,
                        position=step_number,
                        session_id=session_id,
                    ),
                    revises_id=target_thought.id,
                    metadata={"resolution_strategy": "revise"},
                )
                # Mark original as superseded
                target_thought.metadata["superseded_by"] = thought_id
                target_thought.metadata["superseded_reason"] = "contradiction_resolved"

            elif strategy == "branch":
                # Create a new branch exploring this interpretation
                branch_id = f"branch_{uuid.uuid4().hex[:8]}"
                thought = Thought(
                    id=thought_id,
                    content=resolution_content,
                    thought_type=ThoughtType.BRANCH,
                    step_number=step_number,
                    confidence=confidence,
                    survival_score=self._score_thought(
                        resolution_content,
                        context=session.context,
                        position=step_number,
                        session_id=session_id,
                    ),
                    branch_id=branch_id,
                    parent_id=target_thought.id,
                    metadata={"resolution_strategy": "branch", "explores": "alternative"},
                )
                session.branches[branch_id] = [thought_id]

            elif strategy == "reconcile":
                # Create a synthesis thought that reconciles the contradiction
                reconciled_ids = [target_thought.id] + target_thought.contradicts
                thought = Thought(
                    id=thought_id,
                    content=resolution_content,
                    thought_type=ThoughtType.SYNTHESIS,
                    step_number=step_number,
                    confidence=confidence,
                    survival_score=self._score_thought(
                        resolution_content,
                        context=session.context,
                        position=step_number,
                        session_id=session_id,
                    ),
                    related_to=reconciled_ids,
                    metadata={
                        "resolution_strategy": "reconcile",
                        "reconciles": reconciled_ids,
                    },
                )

            elif strategy == "backtrack":
                # Mark the target thought as abandoned
                thought = Thought(
                    id=thought_id,
                    content=resolution_content,
                    thought_type=ThoughtType.CONTINUATION,
                    step_number=step_number,
                    confidence=confidence,
                    survival_score=self._score_thought(
                        resolution_content,
                        context=session.context,
                        position=step_number,
                        session_id=session_id,
                    ),
                    metadata={
                        "resolution_strategy": "backtrack",
                        "abandons": target_thought.id,
                    },
                )
                # Mark target as abandoned
                target_thought.metadata["abandoned"] = True
                target_thought.metadata["abandoned_reason"] = "contradiction_backtrack"
                target_thought.metadata["replaced_by"] = thought_id

            else:
                # Should not reach here due to validation, but satisfy type checker
                raise ValueError(f"Unhandled strategy: {strategy}")

            # Store the resolution thought
            session.thoughts[thought_id] = thought
            session.thought_order.append(thought_id)
            session.updated_at = datetime.now()

            # Clear the contradiction from the target
            resolved_contradictions = list(target_thought.contradicts)
            target_thought.contradicts.clear()

            # Update graph if enabled
            if self._enable_graph and session.thought_graph is not None:
                try:
                    from src.tools.thought_graph import EdgeType, GraphNode

                    # Add resolution node
                    node = GraphNode(
                        id=thought_id,
                        label=f"Resolution ({strategy})",
                        content=resolution_content,
                        step_number=step_number,
                        metadata=thought.metadata,
                    )
                    session.thought_graph.add_node(node)

                    # Add resolution edge
                    session.thought_graph.add_edge(thought_id, target_thought.id, EdgeType.SUPPORTS)
                except Exception as e:
                    logger.warning(f"Failed to update graph for resolution: {e}")

            # Count remaining contradictions
            remaining = sum(len(t.contradicts) for t in session.thoughts.values())

            survival = thought.survival_score if thought.survival_score is not None else 0.0
            return {
                "session_id": session_id,
                "resolution_id": thought_id,
                "strategy_applied": strategy,
                "target_thought": target_thought.id,
                "resolved_contradictions": resolved_contradictions,
                "remaining_contradictions": remaining,
                "step": step_number,
                "survival_score": round(survival, 3),
                "guidance": {
                    "action": "continue" if remaining == 0 else "review",
                    "message": (
                        "Contradiction resolved. Continue reasoning."
                        if remaining == 0
                        else f"{remaining} contradiction(s) remain. Consider resolving them."
                    ),
                },
            }

    def _get_strategies_for_domain(self, domain: DomainType, num_rows: int) -> list[str]:
        """Get reasoning strategies appropriate for the domain."""
        base_strategies = ["direct_analysis", "logical_inference", "verification"]

        domain_strategies: dict[DomainType, list[str]] = {
            DomainType.MATH: [
                "algebraic",
                "geometric",
                "numerical_estimation",
                "proof_by_contradiction",
            ],
            DomainType.CODE: [
                "trace_execution",
                "edge_cases",
                "complexity_analysis",
                "refactoring",
            ],
            DomainType.LOGIC: [
                "forward_chaining",
                "backward_chaining",
                "proof_by_cases",
                "reductio",
            ],
            DomainType.FACTUAL: ["source_verification", "cross_reference", "temporal_analysis"],
            DomainType.GENERAL: ["analogical", "causal", "counterfactual"],
        }

        strategies = base_strategies + domain_strategies.get(domain, [])
        return strategies[:num_rows]

    def _detect_contradiction(self, thought1: str, thought2: str) -> bool:
        """Detect if two thoughts contradict each other using pattern matching.

        Fast heuristic-based contradiction detection. For semantic detection,
        use _detect_contradiction_semantic() when vector store is available.

        Args:
            thought1: First thought content.
            thought2: Second thought content.

        Returns:
            True if contradiction detected via patterns.

        """
        return self._detect_contradiction_patterns(thought1, thought2)

    def _detect_contradiction_patterns(self, thought1: str, thought2: str) -> bool:
        """Pattern-based contradiction detection (fast, no model required).

        Args:
            thought1: First thought content.
            thought2: Second thought content.

        Returns:
            True if contradiction detected.

        """
        t1_lower = thought1.lower()
        t2_lower = thought2.lower()

        # Pattern pairs that indicate contradiction
        contradiction_patterns = [
            ("is true", "is false"),
            ("is false", "is true"),
            ("always", "never"),
            ("never", "always"),
            ("all", "none"),
            ("none", "all"),
            ("yes", "no"),
            ("correct", "incorrect"),
            ("valid", "invalid"),
            ("possible", "impossible"),
            ("can", "cannot"),
            ("will", "won't"),
            ("must", "must not"),
            ("should", "should not"),
            ("agree", "disagree"),
            ("true", "false"),
            ("increase", "decrease"),
            ("positive", "negative"),
            ("success", "failure"),
            ("accept", "reject"),
        ]

        for p1, p2 in contradiction_patterns:
            if (p1 in t1_lower and p2 in t2_lower) or (p2 in t1_lower and p1 in t2_lower):
                return True

        # Check for negation of same claim
        import re

        # Extract key phrases after "is" (skip negation words)
        # Pattern captures: "is <word>" or "is not <word>"
        phrases1_pos = set(re.findall(r"\bis\s+(\w+)", t1_lower)) - {"not", "no", "never"}
        phrases1_neg = set(re.findall(r"\bis\s+not\s+(\w+)", t1_lower))

        phrases2_pos = set(re.findall(r"\bis\s+(\w+)", t2_lower)) - {"not", "no", "never"}
        phrases2_neg = set(re.findall(r"\bis\s+not\s+(\w+)", t2_lower))

        # Contradiction: positive in one, negative in other for same phrase
        # e.g., "is correct" in t1 and "is not correct" in t2
        return bool(phrases1_pos & phrases2_neg or phrases1_neg & phrases2_pos)

    async def _detect_contradiction_semantic(
        self,
        thought1: str,
        thought2: str,
        similarity_threshold: float = 0.7,
        negation_boost: float = 0.15,
    ) -> tuple[bool, float]:
        """Semantic contradiction detection using vector similarity.

        Detects contradictions by combining:
        1. High semantic similarity (similar topics/subjects)
        2. Presence of negation markers in one thought but not the other

        Args:
            thought1: First thought content.
            thought2: Second thought content.
            similarity_threshold: Minimum similarity to consider (default 0.7).
            negation_boost: Extra weight for negation presence (default 0.15).

        Returns:
            Tuple of (is_contradiction, confidence_score).
            confidence_score is 0-1 where 1 = definite contradiction.

        """
        # First check patterns (fast path)
        if self._detect_contradiction_patterns(thought1, thought2):
            return True, 0.9

        # If no vector store, can't do semantic detection
        if self._vector_store is None:
            return False, 0.0

        try:
            # Compute semantic similarity
            similarity = await self._vector_store.compute_similarity(thought1, thought2)

            # If very low similarity, these aren't about the same thing
            if similarity < similarity_threshold:
                return False, 0.0

            # Check for opposing sentiment/negation markers
            t1_lower = thought1.lower()
            t2_lower = thought2.lower()

            negation_markers = [
                "not",
                "no",
                "never",
                "none",
                "neither",
                "nobody",
                "nothing",
                "nowhere",
                "isn't",
                "aren't",
                "wasn't",
                "weren't",
                "won't",
                "wouldn't",
                "shouldn't",
                "couldn't",
                "can't",
                "don't",
                "doesn't",
                "didn't",
                "hasn't",
                "haven't",
                "hadn't",
                "false",
                "incorrect",
                "wrong",
                "invalid",
                "impossible",
                "unlikely",
                "disagree",
                "reject",
            ]

            # Count negation markers in each
            neg_count1 = sum(1 for marker in negation_markers if marker in t1_lower)
            neg_count2 = sum(1 for marker in negation_markers if marker in t2_lower)

            # Contradiction signal: similar content but different negation polarity
            negation_diff = abs(neg_count1 - neg_count2)

            if negation_diff > 0:
                # High similarity + different negation = likely contradiction
                confidence = min(1.0, similarity + (negation_boost * negation_diff))

                # Consider it a contradiction if confidence exceeds threshold
                is_contradiction = confidence > 0.75
                return is_contradiction, confidence

            # High similarity, same polarity = not a contradiction (agreement)
            return False, 0.0

        except Exception as e:
            logger.warning(f"Semantic contradiction detection failed: {e}")
            return False, 0.0

    async def detect_contradictions_in_session(
        self,
        session_id: str,
        thought_id: str | None = None,
        use_semantic: bool = True,
    ) -> list[dict[str, Any]]:
        """Detect all contradictions in a session.

        Args:
            session_id: Session to analyze.
            thought_id: If provided, only check contradictions with this thought.
            use_semantic: Whether to use semantic detection (requires vector store).

        Returns:
            List of contradiction records with thought pairs and confidence.

        """
        async with self.session(session_id) as session:
            contradictions: list[dict[str, Any]] = []
            thoughts = list(session.thoughts.values())

            # If specific thought, only compare against it
            if thought_id and thought_id in session.thoughts:
                target = session.thoughts[thought_id]
                others = [t for t in thoughts if t.id != thought_id]

                for other in others:
                    if use_semantic and self._vector_store is not None:
                        is_contra, conf = await self._detect_contradiction_semantic(
                            target.content, other.content
                        )
                    else:
                        is_contra = self._detect_contradiction_patterns(
                            target.content, other.content
                        )
                        conf = 0.85 if is_contra else 0.0

                    if is_contra:
                        contradictions.append(
                            {
                                "thought_a": thought_id,
                                "thought_b": other.id,
                                "confidence": round(conf, 3),
                                "method": "semantic" if use_semantic else "pattern",
                            }
                        )
            else:
                # Compare all pairs
                for i, t1 in enumerate(thoughts):
                    for t2 in thoughts[i + 1 :]:
                        if use_semantic and self._vector_store is not None:
                            is_contra, conf = await self._detect_contradiction_semantic(
                                t1.content, t2.content
                            )
                        else:
                            is_contra = self._detect_contradiction_patterns(t1.content, t2.content)
                            conf = 0.85 if is_contra else 0.0

                        if is_contra:
                            contradictions.append(
                                {
                                    "thought_a": t1.id,
                                    "thought_b": t2.id,
                                    "confidence": round(conf, 3),
                                    "method": "semantic" if use_semantic else "pattern",
                                }
                            )

            return contradictions

    def _get_contradiction_guidance(
        self,
        session: ReasoningSession,
        current_thought: Thought,
        contradicting_ids: list[str],
    ) -> dict[str, Any]:
        """Generate guidance for resolving detected contradictions.

        Provides actionable strategies for addressing logical inconsistencies
        in the reasoning chain.

        Args:
            session: Current reasoning session.
            current_thought: The thought that introduced contradictions.
            contradicting_ids: IDs of thoughts this one contradicts.

        Returns:
            Guidance dictionary with resolution strategies and recommendations.

        """
        if not contradicting_ids:
            return {}

        # Gather contradicting thoughts
        contradicting_thoughts: list[dict[str, Any]] = []
        for tid in contradicting_ids:
            if tid in session.thoughts:
                t = session.thoughts[tid]
                contradicting_thoughts.append(
                    {
                        "id": tid,
                        "step": t.step_number,
                        "summary": t.content[:100] + "..." if len(t.content) > 100 else t.content,
                        "type": t.thought_type.value,
                    }
                )

        # Determine resolution strategies based on context
        strategies: list[dict[str, str]] = []

        # Strategy 1: Revise - Update the current thought
        strategies.append(
            {
                "name": "revise",
                "action": "Modify the current thought to resolve the contradiction",
                "when_to_use": "When the current thought contains an error or oversight",
                "how": "Call add_thought with revised content that accounts for the contradicted claims",
            }
        )

        # Strategy 2: Branch - Explore both possibilities
        strategies.append(
            {
                "name": "branch",
                "action": "Create separate reasoning branches for each possibility",
                "when_to_use": "When both positions might be valid under different assumptions",
                "how": "Use branch_thought to explore alternative interpretations",
            }
        )

        # Strategy 3: Reconcile - Find synthesis
        strategies.append(
            {
                "name": "reconcile",
                "action": "Find a higher-level synthesis that resolves the apparent contradiction",
                "when_to_use": "When the contradiction stems from different perspectives or scopes",
                "how": "Add a thought that explains how both claims can coexist",
            }
        )

        # Strategy 4: Backtrack - Reject current thought
        strategies.append(
            {
                "name": "backtrack",
                "action": "Abandon the current line of reasoning",
                "when_to_use": "When earlier thoughts are more firmly established",
                "how": "Continue reasoning from a previous thought, ignoring the contradicting path",
            }
        )

        # Provide specific recommendation based on confidence
        current_conf = current_thought.confidence or 0.5
        contradicting_confs = [
            session.thoughts[tid].confidence or 0.5
            for tid in contradicting_ids
            if tid in session.thoughts
        ]
        avg_contradicting_conf = (
            sum(contradicting_confs) / len(contradicting_confs) if contradicting_confs else 0.5
        )

        if current_conf < avg_contradicting_conf - 0.2:
            recommendation = "revise"
            reason = "Current thought has lower confidence than contradicting thoughts"
        elif current_conf > avg_contradicting_conf + 0.2:
            recommendation = "backtrack"
            reason = "Current thought has higher confidence; earlier reasoning may need revision"
        else:
            recommendation = "reconcile"
            reason = "Similar confidence levels suggest both perspectives may have merit"

        return {
            "contradicting_thoughts": contradicting_thoughts,
            "strategies": strategies,
            "recommendation": {
                "strategy": recommendation,
                "reason": reason,
            },
            "action_required": True,
            "message": (
                f"Contradiction detected with {len(contradicting_ids)} previous thought(s). "
                "Consider using one of the resolution strategies before continuing."
            ),
        }

    def _get_next_step_guidance(
        self, session: ReasoningSession, last_thought: Thought
    ) -> dict[str, Any]:
        """Generate compact guidance for the next reasoning step.

        OPT8: Streamlined output - only essential fields for LLM guidance.
        """
        guidance: dict[str, Any] = {}

        if session.actual_mode == ReasoningMode.CHAIN:
            guidance["next"] = "continue"
            guidance["step"] = session.current_step + 1

        elif session.actual_mode == ReasoningMode.MATRIX:
            # Find next unfilled cell
            for col in range(session.matrix_cols):
                for row in range(session.matrix_rows):
                    if (row, col) not in session.matrix_cells:
                        guidance["next"] = "fill"
                        guidance["cell"] = [row, col]
                        return guidance

            # All cells filled, suggest synthesis or finalize
            if len(session.syntheses) < session.matrix_cols:
                next_col = min(set(range(session.matrix_cols)) - set(session.syntheses.keys()))
                guidance["next"] = "synthesize"
                guidance["col"] = next_col
            else:
                guidance["next"] = "finalize"

        # Add blind spot count only (not full dict)
        unaddressed = sum(1 for bs in session.blind_spots if not bs.addressed)
        if unaddressed > 0:
            guidance["gaps"] = unaddressed

        return guidance


# =============================================================================
# Module-level Singleton
# =============================================================================

_unified_manager: UnifiedReasonerManager | None = None


def get_unified_manager() -> UnifiedReasonerManager:
    """Get or create the unified reasoner manager instance."""
    global _unified_manager
    if _unified_manager is None:
        # Initialize with default weight store
        from src.utils.weight_store import get_weight_store

        _unified_manager = UnifiedReasonerManager(
            weight_store=get_weight_store(),
            enable_weight_persistence=True,
        )
    return _unified_manager


async def init_unified_manager(
    vector_store: AsyncVectorStore | None = None,
    encoder: ContextEncoder | None = None,
    enable_graph: bool = True,
    enable_domain_validation: bool = True,
    enable_weight_persistence: bool = True,
    enable_semantic_scoring: bool = True,
    weight_store: Any = None,
) -> UnifiedReasonerManager:
    """Initialize the unified reasoner manager with optional vector store and encoder.

    Args:
        vector_store: Optional AsyncVectorStore for RAG and thought retrieval.
        encoder: Optional ContextEncoder for semantic similarity scoring.
        enable_graph: Whether to build thought relationship graph.
        enable_domain_validation: Whether to validate thoughts with domain handlers.
        enable_weight_persistence: Whether to persist suggestion weights.
        enable_semantic_scoring: Whether to use embedding-based scoring.
        weight_store: Optional WeightStore (uses default if None and persistence enabled).

    Returns:
        Initialized UnifiedReasonerManager.

    """
    global _unified_manager

    # Initialize weight store if persistence enabled
    if enable_weight_persistence and weight_store is None:
        from src.utils.weight_store import get_weight_store

        weight_store = get_weight_store()

    _unified_manager = UnifiedReasonerManager(
        vector_store=vector_store,
        encoder=encoder,
        enable_rag=vector_store is not None,
        enable_blind_spots=True,
        enable_rewards=True,
        enable_graph=enable_graph,
        enable_domain_validation=enable_domain_validation,
        weight_store=weight_store,
        enable_weight_persistence=enable_weight_persistence,
        enable_semantic_scoring=enable_semantic_scoring,
    )
    return _unified_manager
