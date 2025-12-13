"""MatrixMind MCP Server.

FastMCP 2.0 implementation providing reasoning state management tools.
The calling LLM does all reasoning; these tools track and organize the process.

Tools:
1. think - Unified reasoning tool (auto-selects chain/matrix/hybrid mode)
2. compress - Semantic context compression
3. status - Server/session status

Run with: uvx matrixmind-mcp
Or: python -m src.server
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial
from typing import Any, Literal

from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from loguru import logger

from src.models.model_manager import ModelManager
from src.tools.compress import ContextAwareCompressionTool
from src.tools.unified_reasoner import (
    ReasoningMode,
    SessionStatus,
    ThoughtType,
    UnifiedReasonerManager,
    get_unified_manager,
    init_unified_manager,
)
from src.utils.errors import MatrixMindException, ModelNotReadyException, ToolExecutionError

# Load environment variables from .env file (for local development)
load_dotenv()


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable, treating empty string as unset."""
    value = os.getenv(key, default)
    return value if value else default


def _get_env_int(key: str, default: int) -> int:
    """Get environment variable as integer."""
    value = os.getenv(key)
    if value:
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer for {key}: {value}, using default {default}")
    return default


# =============================================================================
# Configuration from Environment Variables
# =============================================================================

# Model Configuration
EMBEDDING_MODEL = _get_env("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-xs")

# Server Configuration
SERVER_NAME = _get_env("SERVER_NAME", "MatrixMind-MCP")
SERVER_TRANSPORT = _get_env("SERVER_TRANSPORT", "stdio")
# Bind to localhost by default for security (CWE-306: Missing Authentication)
# Use 0.0.0.0 only in production with proper authentication/firewall
SERVER_HOST = _get_env("SERVER_HOST", "127.0.0.1")
SERVER_PORT = _get_env_int("SERVER_PORT", 8000)

# Rate Limiting Configuration
RATE_LIMIT_MAX_SESSIONS = _get_env_int("RATE_LIMIT_MAX_SESSIONS", 100)
RATE_LIMIT_WINDOW_SECONDS = _get_env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
MAX_TOTAL_SESSIONS = _get_env_int("MAX_TOTAL_SESSIONS", 500)

# Session Cleanup Configuration
SESSION_MAX_AGE_MINUTES = _get_env_int("SESSION_MAX_AGE_MINUTES", 30)
CLEANUP_INTERVAL_SECONDS = _get_env_int("CLEANUP_INTERVAL_SECONDS", 60)

# =============================================================================
# Input Size Limits (CWE-400: Uncontrolled Resource Consumption)
# =============================================================================
MAX_PROBLEM_SIZE = _get_env_int("MAX_PROBLEM_SIZE", 50000)  # 50KB
MAX_THOUGHT_SIZE = _get_env_int("MAX_THOUGHT_SIZE", 10000)  # 10KB
MAX_CONTEXT_SIZE = _get_env_int("MAX_CONTEXT_SIZE", 100000)  # 100KB
MAX_ALTERNATIVES = _get_env_int("MAX_ALTERNATIVES", 10)  # Max alternatives for MPPA
MAX_THOUGHTS_PER_SESSION = _get_env_int("MAX_THOUGHTS_PER_SESSION", 1000)  # Memory exhaustion guard

# RAG Configuration
ENABLE_RAG = _get_env("ENABLE_RAG", "false").lower() == "true"
VECTOR_DB_PATH = _get_env("VECTOR_DB_PATH", ":memory:")


def _validate_vector_db_path() -> str:
    """Validate VECTOR_DB_PATH to prevent path traversal (CWE-22)."""
    from src.utils.weight_store import validate_db_path

    if VECTOR_DB_PATH == ":memory:":
        return VECTOR_DB_PATH

    try:
        validated = validate_db_path(VECTOR_DB_PATH)
        return str(validated)
    except ValueError as e:
        logger.error(f"Invalid VECTOR_DB_PATH: {e}")
        logger.warning("Falling back to in-memory vector store")
        return ":memory:"


# =============================================================================
# Initialize Embedding Model Manager
# =============================================================================


def _get_embedding_model_name() -> str:
    """Get full embedding model name."""
    model_name = EMBEDDING_MODEL
    if "/" not in model_name:
        model_name = f"sentence-transformers/{model_name}"
    return model_name


def _init_model_manager() -> None:
    """Initialize the model manager with embedding model."""
    model_manager = ModelManager.get_instance()
    model_name = _get_embedding_model_name()
    logger.info(f"Preloading embedding model: {model_name}")
    model_manager.initialize(model_name, blocking=True)
    logger.info("Embedding model ready")


# =============================================================================
# Thread Pool for CPU-bound Operations
# =============================================================================
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="matrixmind-worker")


# =============================================================================
# Rate Limiting and Session Management
# =============================================================================


class SessionRateLimiter:
    """Sliding window rate limiter for session creation.

    Prevents resource exhaustion by limiting the rate of new session creation.
    Uses a sliding window algorithm for accurate rate limiting.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        """Initialize rate limiter.

        Args:
            max_requests: Maximum sessions allowed in the time window
            window_seconds: Time window in seconds

        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _cleanup_old_timestamps(self) -> None:
        """Remove timestamps outside the current window."""
        cutoff = time.monotonic() - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    async def check_rate_limit(self) -> tuple[bool, dict[str, Any]]:
        """Check if a new session can be created.

        Returns:
            Tuple of (allowed, info_dict).
            If allowed is False, info_dict contains error details.

        """
        async with self._lock:
            self._cleanup_old_timestamps()

            if len(self._timestamps) >= self._max_requests:
                # Calculate when the oldest request will expire
                oldest = self._timestamps[0]
                retry_after = self._window_seconds - (time.monotonic() - oldest)
                return False, {
                    "error": "rate_limit_exceeded",
                    "message": (
                        f"Too many sessions created. "
                        f"Max {self._max_requests} per {self._window_seconds}s."
                    ),
                    "retry_after_seconds": max(1, int(retry_after)),
                    "current_count": len(self._timestamps),
                    "max_allowed": self._max_requests,
                }

            return True, {"remaining": self._max_requests - len(self._timestamps)}

    async def record_request(self) -> None:
        """Record a successful session creation."""
        async with self._lock:
            self._timestamps.append(time.monotonic())

    def get_stats(self) -> dict[str, Any]:
        """Get current rate limiter statistics."""
        self._cleanup_old_timestamps()
        return {
            "current_count": len(self._timestamps),
            "max_allowed": self._max_requests,
            "window_seconds": self._window_seconds,
            "remaining": max(0, self._max_requests - len(self._timestamps)),
        }


# Global rate limiter instance
_rate_limiter: SessionRateLimiter | None = None


def get_rate_limiter() -> SessionRateLimiter:
    """Get or create rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = SessionRateLimiter(
            max_requests=RATE_LIMIT_MAX_SESSIONS,
            window_seconds=RATE_LIMIT_WINDOW_SECONDS,
        )
    return _rate_limiter


# =============================================================================
# Automatic Session Cleanup
# =============================================================================

_cleanup_task: asyncio.Task[None] | None = None


async def _cleanup_stale_sessions() -> None:
    """Background task to clean up stale sessions."""
    max_age = timedelta(minutes=SESSION_MAX_AGE_MINUTES)
    logger.info(
        f"Session cleanup task started (max_age={SESSION_MAX_AGE_MINUTES}m, "
        f"interval={CLEANUP_INTERVAL_SECONDS}s)"
    )

    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)

            manager = get_unified_manager()
            removed = manager.cleanup_stale(max_age)

            if removed:
                logger.info(f"Cleaned up {len(removed)} stale sessions: {removed}")

        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            # Continue running despite errors


def _start_cleanup_task() -> None:
    """Start the background cleanup task if not already running."""
    global _cleanup_task
    try:
        loop = asyncio.get_running_loop()
        if _cleanup_task is None or _cleanup_task.done():
            _cleanup_task = loop.create_task(_cleanup_stale_sessions())
            logger.debug("Cleanup task scheduled")
    except RuntimeError:
        # No running event loop - will be started when server runs
        logger.debug("No event loop available, cleanup task will start with server")


def _stop_cleanup_task() -> None:
    """Stop the background cleanup task."""
    global _cleanup_task
    if _cleanup_task is not None and not _cleanup_task.done():
        _cleanup_task.cancel()
        _cleanup_task = None
        logger.debug("Cleanup task stopped")


# =============================================================================
# Initialize FastMCP Server
# =============================================================================

mcp = FastMCP(
    name=SERVER_NAME,
    instructions="""MatrixMind reasoning state manager with unified reasoning tool.

ARCHITECTURE: You (the calling LLM) do ALL reasoning. These tools TRACK and ORGANIZE your thoughts.

TOOLS:

1. think(action, ...) - Unified reasoning with auto-mode selection
   Actions:
   - "start": Begin reasoning (mode: auto/chain/matrix/hybrid)
   - "continue": Add reasoning step
   - "branch": Branch from a step (chain/hybrid only)
   - "revise": Revise a step (chain/hybrid only)
   - "synthesize": Synthesize column (matrix/hybrid only)
   - "finish": Complete reasoning

2. compress(context, query, ratio) - Reduce context size semantically

3. status(session_id?) - Get server or session status

AUTO-MODE SELECTION:
By default, mode="auto" analyzes your problem and selects:
- CHAIN: Simple, sequential problems (low complexity)
- MATRIX: Complex, multi-perspective problems (high complexity)
- HYBRID: Long context that may need escalation

FEATURES:
- Blind spot detection: Catches unstated assumptions, uncertain claims
- RLVR rewards: Tracks reasoning quality (consistency, coherence, efficiency)
- MPPA integration: Pass alternatives to explore multiple paths
- CISC integration: Weighted candidate selection with confidences
- Domain detection: Math, code, logic, factual handlers

WORKFLOW EXAMPLE (Auto mode):
1. think(action="start", problem="Solve X")
   -> Server analyzes complexity, selects optimal mode
2. think(action="continue", session_id=ID, thought="Step 1: ...")
   -> Receives guidance, blind spot warnings, reward signals
3. think(action="continue", session_id=ID, thought="Step 2: ...", alternatives=["alt1", "alt2"])
   -> MPPA: Server evaluates alternatives via CISC
4. think(action="finish", session_id=ID, thought="The answer is...")
   -> Final summary with total rewards, unaddressed blind spots

WORKFLOW EXAMPLE (Matrix mode):
1. think(action="start", mode="matrix", problem="Analyze Y", rows=3, cols=2)
2. think(action="continue", session_id=ID, row=0, col=0, thought="...")
3. think(action="synthesize", session_id=ID, col=0, thought="Column synthesis")
4. think(action="finish", session_id=ID, thought="Final analysis...")
""",
)

# =============================================================================
# Tool Instances
# =============================================================================

_compression_tool: ContextAwareCompressionTool | None = None


def get_compression_tool() -> ContextAwareCompressionTool:
    """Get or create compression tool instance."""
    global _compression_tool
    if _compression_tool is None:
        model_name = _get_embedding_model_name()
        _compression_tool = ContextAwareCompressionTool(model_name=model_name)
    return _compression_tool


# =============================================================================
# Type Definitions
# =============================================================================

ThinkAction = Literal[
    "start",
    "continue",
    "branch",
    "revise",
    "synthesize",
    "verify",
    "finish",
    "resolve",
    "analyze",
    "suggest",
    "feedback",  # S2: Record suggestion outcome
    "auto",  # S3: Auto-execute suggestion
]
ThinkModeStr = Literal["auto", "chain", "matrix", "hybrid", "verify"]
ResolveStrategy = Literal["revise", "branch", "reconcile", "backtrack"]
SuggestionOutcome = Literal["accepted", "rejected"]


# =============================================================================
# Input Validation Helpers (CWE-400 Prevention)
# =============================================================================


def _validate_input_sizes(
    problem: str | None = None,
    thought: str | None = None,
    context: str | None = None,
    alternatives: list[str] | None = None,
) -> dict[str, Any] | None:
    """Validate input sizes to prevent resource exhaustion.

    Returns:
        None if valid, or dict with error details if invalid.

    """
    if problem and len(problem) > MAX_PROBLEM_SIZE:
        return {
            "error": "input_too_large",
            "field": "problem",
            "max_size": MAX_PROBLEM_SIZE,
            "actual_size": len(problem),
            "message": f"Problem exceeds maximum size ({MAX_PROBLEM_SIZE:,} chars)",
        }

    if thought and len(thought) > MAX_THOUGHT_SIZE:
        return {
            "error": "input_too_large",
            "field": "thought",
            "max_size": MAX_THOUGHT_SIZE,
            "actual_size": len(thought),
            "message": f"Thought exceeds maximum size ({MAX_THOUGHT_SIZE:,} chars)",
        }

    if context and len(context) > MAX_CONTEXT_SIZE:
        return {
            "error": "input_too_large",
            "field": "context",
            "max_size": MAX_CONTEXT_SIZE,
            "actual_size": len(context),
            "message": f"Context exceeds maximum size ({MAX_CONTEXT_SIZE:,} chars)",
        }

    if alternatives:
        if len(alternatives) > MAX_ALTERNATIVES:
            return {
                "error": "too_many_alternatives",
                "max_alternatives": MAX_ALTERNATIVES,
                "actual_count": len(alternatives),
                "message": f"Too many alternatives ({len(alternatives)} > {MAX_ALTERNATIVES})",
            }
        # Also validate each alternative's size
        for i, alt in enumerate(alternatives):
            if len(alt) > MAX_THOUGHT_SIZE:
                return {
                    "error": "input_too_large",
                    "field": f"alternatives[{i}]",
                    "max_size": MAX_THOUGHT_SIZE,
                    "actual_size": len(alt),
                    "message": f"Alternative {i} exceeds maximum size ({MAX_THOUGHT_SIZE:,} chars)",
                }

    return None


# =============================================================================
# TOOL 1: THINK (Unified Reasoning)
# =============================================================================


@mcp.tool
async def think(
    action: ThinkAction,
    mode: ThinkModeStr | None = None,
    session_id: str | None = None,
    problem: str | None = None,
    context: str | None = None,
    thought: str | None = None,
    expected_steps: int = 10,
    rows: int | None = None,
    cols: int | None = None,
    row: int | None = None,
    col: int | None = None,
    branch_from: str | None = None,
    revises: str | None = None,
    confidence: float | None = None,
    alternatives: list[str] | None = None,
    alternative_confidences: list[float] | None = None,
    # Contradiction resolution parameters
    resolve_strategy: ResolveStrategy | None = None,
    contradicting_thought_id: str | None = None,
    # Backwards compatibility for verify mode
    claim_id: int | None = None,
    verdict: str | None = None,
    evidence: str | None = None,
    # S2: Feedback parameters for recording suggestion outcomes
    suggestion_id: str | None = None,
    suggestion_outcome: SuggestionOutcome | None = None,
    actual_action: str | None = None,
    # S3: Auto-execute parameters
    max_auto_steps: int = 5,
    stop_on_high_risk: bool = True,
    ctx: Context | None = None,
) -> str:
    """Unified reasoning tool with auto-mode selection.

    Actions:
        start: Begin a new reasoning session (requires problem, mode defaults to auto)
        continue: Add a reasoning step (requires session_id and thought)
        branch: Branch from a thought (chain/hybrid, requires branch_from)
        revise: Revise a thought (chain/hybrid, requires revises)
        synthesize: Synthesize a column (matrix/hybrid, requires col and thought)
        resolve: Resolve a detected contradiction (requires resolve_strategy and thought)
        analyze: Get mid-session analysis with metrics and recommendations (requires session_id)
        suggest: Get AI-recommended next action based on session state (requires session_id)
        feedback: Record outcome of a suggestion for weight learning (requires suggestion_id, suggestion_outcome)
        auto: Auto-execute the top suggested action (requires session_id)
        finish: Complete reasoning (requires session_id)

    Modes:
        auto: Auto-select based on problem complexity (default)
        chain: Sequential chain-of-thought reasoning
        matrix: Multi-perspective matrix reasoning (rows x cols)
        hybrid: Adaptive chain -> matrix escalation

    Args:
        action: The action to perform (required)
        mode: Reasoning mode for "start" action (default: auto)
        session_id: Session ID for continuing/finishing
        problem: Problem statement for "start" action
        context: Background context (optional)
        thought: Your reasoning content
        expected_steps: Expected steps for chain mode (default 10)
        rows: Matrix rows (auto-detected if None)
        cols: Matrix columns (auto-detected if None)
        row: Matrix row index (0-based)
        col: Matrix column index (0-based)
        branch_from: Thought ID to branch from
        revises: Thought ID to revise
        confidence: Your confidence in this thought (0-1)
        alternatives: MPPA - Alternative reasoning paths to evaluate
        alternative_confidences: CISC - Your confidence scores for alternatives
        resolve_strategy: Strategy for resolving contradiction (revise/branch/reconcile/backtrack)
        contradicting_thought_id: ID of the thought to address in resolution
        suggestion_id: ID of suggestion to record feedback for (feedback action)
        suggestion_outcome: Whether suggestion was accepted or rejected (feedback action)
        actual_action: What action was actually taken, if different from suggestion (feedback action)
        max_auto_steps: Maximum steps for auto-execute (default 5, auto action)
        stop_on_high_risk: Stop auto-execution at high-risk checkpoints (default True, auto action)

    Returns:
        JSON with action result, session state, guidance, and any warnings

    """
    try:
        # Validate input sizes (CWE-400 prevention)
        validation_error = _validate_input_sizes(
            problem=problem,
            thought=thought,
            context=context,
            alternatives=alternatives,
        )
        if validation_error:
            return json.dumps(validation_error)

        manager = get_unified_manager()

        if action == "start":
            return await _think_start(
                manager=manager,
                mode=mode,
                problem=problem,
                context=context,
                expected_steps=expected_steps,
                rows=rows,
                cols=cols,
                ctx=ctx,
            )
        elif action == "continue":
            return await _think_continue(
                manager=manager,
                session_id=session_id,
                thought=thought,
                row=row,
                col=col,
                confidence=confidence,
                alternatives=alternatives,
                alternative_confidences=alternative_confidences,
                ctx=ctx,
            )
        elif action == "branch":
            return await _think_branch(
                manager=manager,
                session_id=session_id,
                thought=thought,
                branch_from=branch_from,
                ctx=ctx,
            )
        elif action == "revise":
            return await _think_revise(
                manager=manager,
                session_id=session_id,
                thought=thought,
                revises=revises,
                ctx=ctx,
            )
        elif action == "synthesize":
            return await _think_synthesize(
                manager=manager,
                session_id=session_id,
                col=col,
                thought=thought,
                confidence=confidence,
                ctx=ctx,
            )
        elif action == "verify":
            # Backwards compatibility: verify action adds a verification thought
            return await _think_verify(
                manager=manager,
                session_id=session_id,
                thought=thought or evidence or "",
                confidence=confidence,
                ctx=ctx,
            )
        elif action == "finish":
            return await _think_finish(
                manager=manager,
                session_id=session_id,
                thought=thought,
                confidence=confidence,
                ctx=ctx,
            )
        elif action == "resolve":
            return await _think_resolve(
                manager=manager,
                session_id=session_id,
                thought=thought,
                resolve_strategy=resolve_strategy,
                contradicting_thought_id=contradicting_thought_id,
                confidence=confidence,
                ctx=ctx,
            )
        elif action == "analyze":
            return await _think_analyze(
                manager=manager,
                session_id=session_id,
                ctx=ctx,
            )
        elif action == "suggest":
            return await _think_suggest(
                manager=manager,
                session_id=session_id,
                ctx=ctx,
            )
        elif action == "feedback":
            return await _think_feedback(
                manager=manager,
                session_id=session_id,
                suggestion_id=suggestion_id,
                suggestion_outcome=suggestion_outcome,
                actual_action=actual_action,
                ctx=ctx,
            )
        elif action == "auto":
            return await _think_auto(
                manager=manager,
                session_id=session_id,
                max_auto_steps=max_auto_steps,
                stop_on_high_risk=stop_on_high_risk,
                ctx=ctx,
            )
        else:
            return json.dumps({"error": f"Unknown action: {action}"})

    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("think", str(e), {"action": action})
        logger.error(f"Think action '{action}' failed: {e}")
        return json.dumps(error.to_dict())


async def _think_start(
    manager: UnifiedReasonerManager,
    mode: ThinkModeStr | None,
    problem: str | None,
    context: str | None,
    expected_steps: int,
    rows: int | None,
    cols: int | None,
    ctx: Context | None,
) -> str:
    """Handle think start action."""
    if not problem:
        return json.dumps({"error": "problem is required for start action"})

    # Check rate limit
    rate_limiter = get_rate_limiter()
    allowed, info = await rate_limiter.check_rate_limit()
    if not allowed:
        if ctx:
            await ctx.warning(f"Rate limit exceeded: {info['message']}")
        return json.dumps(info)

    # Check total session count
    total_sessions = len(manager._sessions)
    if total_sessions >= MAX_TOTAL_SESSIONS:
        error_info = {
            "error": "max_sessions_exceeded",
            "message": (
                f"Maximum total sessions ({MAX_TOTAL_SESSIONS}) reached. "
                "Wait for cleanup or finish existing sessions."
            ),
            "current_sessions": total_sessions,
            "max_sessions": MAX_TOTAL_SESSIONS,
        }
        if ctx:
            await ctx.warning(f"Max sessions exceeded: {total_sessions}/{MAX_TOTAL_SESSIONS}")
        return json.dumps(error_info)

    # Record the request for rate limiting
    await rate_limiter.record_request()

    # Convert mode string to enum
    # Note: "verify" mode is mapped to "chain" for backwards compatibility
    reasoning_mode = ReasoningMode.AUTO
    if mode:
        if mode == "verify":
            # Backwards compatibility: verify mode uses chain reasoning
            reasoning_mode = ReasoningMode.CHAIN
        else:
            try:
                reasoning_mode = ReasoningMode(mode)
            except ValueError:
                return json.dumps({"error": f"Unknown mode: {mode}"})

    # Start session
    result = await manager.start_session(
        problem=problem,
        context=context or "",
        mode=reasoning_mode,
        expected_steps=expected_steps,
        rows=rows,
        cols=cols,
    )

    # Backwards compatibility: if user requested "verify", show that in response
    if mode == "verify":
        result["mode"] = "verify"

    if ctx:
        actual_mode = result.get("actual_mode", "unknown")
        domain = result.get("domain", "unknown")
        await ctx.info(
            f"Started session {result['session_id']} (mode={actual_mode}, domain={domain})"
        )

    return json.dumps(result, indent=2)


async def _think_continue(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    row: int | None,
    col: int | None,
    confidence: float | None,
    alternatives: list[str] | None,
    alternative_confidences: list[float] | None,
    ctx: Context | None,
) -> str:
    """Handle think continue action."""
    if not session_id:
        return json.dumps({"error": "session_id is required for continue action"})
    if not thought:
        return json.dumps({"error": "thought is required for continue action"})

    if session_id not in manager._sessions:
        return json.dumps({"error": "Invalid or expired session"})

    result = await manager.add_thought(
        session_id=session_id,
        content=thought,
        thought_type=ThoughtType.CONTINUATION,
        row=row,
        col=col,
        confidence=confidence,
        alternatives=alternatives,
        alternative_confidences=alternative_confidences,
    )

    if ctx:
        step = result.get("step", "?")
        score = result.get("survival_score", 0)
        await ctx.info(f"Added step {step} (score={score:.2f})")

        # Warn about blind spots
        if "blind_spots_detected" in result:
            count = len(result["blind_spots_detected"])
            await ctx.warning(f"Detected {count} blind spot(s) in this step")

    return json.dumps(result, indent=2)


async def _think_branch(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    branch_from: str | None,
    ctx: Context | None,
) -> str:
    """Handle think branch action."""
    if not session_id:
        return json.dumps({"error": "session_id is required for branch action"})
    if not thought:
        return json.dumps({"error": "thought is required for branch action"})
    if not branch_from:
        return json.dumps({"error": "branch_from is required for branch action"})

    if session_id not in manager._sessions:
        return json.dumps({"error": "Invalid or expired session"})

    result = await manager.add_thought(
        session_id=session_id,
        content=thought,
        thought_type=ThoughtType.BRANCH,
        branch_from=branch_from,
    )

    if ctx:
        await ctx.info(f"Branched from thought {branch_from}")

    return json.dumps(result, indent=2)


async def _think_revise(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    revises: str | None,
    ctx: Context | None,
) -> str:
    """Handle think revise action."""
    if not session_id:
        return json.dumps({"error": "session_id is required for revise action"})
    if not thought:
        return json.dumps({"error": "thought is required for revise action"})
    if not revises:
        return json.dumps({"error": "revises is required for revise action"})

    if session_id not in manager._sessions:
        return json.dumps({"error": "Invalid or expired session"})

    result = await manager.add_thought(
        session_id=session_id,
        content=thought,
        thought_type=ThoughtType.REVISION,
        revises=revises,
    )

    if ctx:
        await ctx.info(f"Revised thought {revises}")

    return json.dumps(result, indent=2)


async def _think_synthesize(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    col: int | None,
    thought: str | None,
    confidence: float | None,
    ctx: Context | None,
) -> str:
    """Handle think synthesize action (matrix/hybrid only)."""
    if not session_id:
        return json.dumps({"error": "session_id is required for synthesize action"})
    if col is None:
        return json.dumps({"error": "col is required for synthesize action"})
    if not thought:
        return json.dumps({"error": "thought is required for synthesize action"})

    if session_id not in manager._sessions:
        return json.dumps({"error": "Invalid or expired session"})

    result = await manager.synthesize(
        session_id=session_id,
        col=col,
        content=thought,
        confidence=confidence,
    )

    if ctx:
        await ctx.info(f"Synthesized column {col}")

    return json.dumps(result, indent=2)


async def _think_verify(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    confidence: float | None,
    ctx: Context | None,
) -> str:
    """Handle think verify action (backwards compatibility).

    In the unified reasoner, verify is treated as a verification thought type.
    """
    if not session_id:
        return json.dumps({"error": "session_id is required for verify action"})
    if not thought:
        return json.dumps({"error": "thought/evidence is required for verify action"})

    if session_id not in manager._sessions:
        return json.dumps({"error": "Invalid or expired session"})

    result = await manager.add_thought(
        session_id=session_id,
        content=thought,
        thought_type=ThoughtType.VERIFICATION,
        confidence=confidence,
    )

    if ctx:
        step = result.get("step", "?")
        await ctx.info(f"Added verification step {step}")

    return json.dumps(result, indent=2)


async def _think_finish(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    confidence: float | None,
    ctx: Context | None,
) -> str:
    """Handle think finish action."""
    if not session_id:
        return json.dumps({"error": "session_id is required for finish action"})

    if session_id not in manager._sessions:
        return json.dumps({"error": "Invalid or expired session"})

    result = await manager.finalize(
        session_id=session_id,
        answer=thought or "",
        confidence=confidence,
    )

    if ctx:
        total_steps = result.get("total_steps", 0)
        mode = result.get("mode_used", "unknown")
        total_reward = result.get("total_reward", 0)
        await ctx.info(f"Finalized: {total_steps} steps, mode={mode}, reward={total_reward:.2f}")

        # Warn about unaddressed blind spots
        if "unaddressed_blind_spots" in result:
            count = len(result["unaddressed_blind_spots"])
            await ctx.warning(f"Warning: {count} blind spot(s) were not addressed")

    return json.dumps(result, indent=2, default=str)


async def _think_resolve(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    resolve_strategy: ResolveStrategy | None,
    contradicting_thought_id: str | None,
    confidence: float | None,
    ctx: Context | None,
) -> str:
    """Handle think resolve action for contradiction resolution.

    Strategies:
        revise: Modify the current thought to resolve the contradiction
        branch: Create separate reasoning branches for each possibility
        reconcile: Find a higher-level synthesis resolving the contradiction
        backtrack: Abandon the contradicting line of reasoning

    """
    if not session_id:
        return json.dumps({"error": "session_id is required for resolve action"})

    if not resolve_strategy:
        return json.dumps(
            {
                "error": "resolve_strategy is required for resolve action",
                "valid_strategies": ["revise", "branch", "reconcile", "backtrack"],
            }
        )

    if not thought:
        return json.dumps({"error": "thought is required for resolve action"})

    if session_id not in manager._sessions:
        return json.dumps({"error": "Invalid or expired session"})

    result = await manager.resolve_contradiction(
        session_id=session_id,
        strategy=resolve_strategy,
        resolution_content=thought,
        contradicting_thought_id=contradicting_thought_id,
        confidence=confidence,
    )

    if ctx:
        strategy = result.get("strategy_applied", resolve_strategy)
        await ctx.info(f"Resolved contradiction using '{strategy}' strategy")

        if result.get("remaining_contradictions", 0) > 0:
            count = result["remaining_contradictions"]
            await ctx.warning(f"Note: {count} contradiction(s) still remain in the session")

    return json.dumps(result, indent=2, default=str)


async def _think_analyze(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    ctx: Context | None,
) -> str:
    """Handle think analyze action for mid-session analysis.

    Returns consolidated metrics, quality scores, and actionable recommendations
    without duplicating raw session data from status/get_status.
    """
    if not session_id:
        return json.dumps({"error": "session_id is required for analyze action"})

    if session_id not in manager._sessions:
        return json.dumps({"error": "Invalid or expired session"})

    analytics = manager.analyze_session(session_id)
    result = analytics.to_dict()

    if ctx:
        # Summarize key findings
        quality = result["quality"]["overall"]
        risk = result["risk"]["level"]
        issues = result["issues"]

        await ctx.info(f"Analysis complete: quality={quality:.2f}, risk={risk}")

        if issues["unresolved_contradictions"] > 0:
            await ctx.warning(
                f"Found {issues['unresolved_contradictions']} unresolved contradiction(s)"
            )

        if issues["blind_spots_unaddressed"] > 0:
            await ctx.warning(
                f"Found {issues['blind_spots_unaddressed']} unaddressed blind spot(s)"
            )

        # Show top recommendation
        if result["recommendations"]:
            await ctx.info(f"Top recommendation: {result['recommendations'][0]}")

    return json.dumps(result, indent=2, default=str)


async def _think_suggest(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    ctx: Context | None,
) -> str:
    """Handle think suggest action for AI-recommended next action.

    Analyzes session state and returns the recommended next action
    with parameters and reasoning, reducing LLM cognitive load.
    """
    if not session_id:
        return json.dumps({"error": "session_id is required for suggest action"})

    if session_id not in manager._sessions:
        return json.dumps({"error": "Invalid or expired session"})

    suggestion = manager.suggest_next_action(session_id)

    if ctx:
        action = suggestion["suggested_action"]
        urgency = suggestion["urgency"]
        reason = suggestion["reason"]

        if urgency == "high":
            await ctx.warning(f"Suggested: {action} (urgent) - {reason}")
        else:
            await ctx.info(f"Suggested: {action} - {reason}")

        # Show guidance
        await ctx.info(f"Guidance: {suggestion['guidance']}")

        # Show alternatives if any
        if suggestion["alternatives"]:
            alt_actions = ", ".join(a["action"] for a in suggestion["alternatives"])
            await ctx.info(f"Alternatives: {alt_actions}")

    return json.dumps(suggestion, indent=2, default=str)


async def _think_feedback(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    suggestion_id: str | None,
    suggestion_outcome: SuggestionOutcome | None,
    actual_action: str | None,
    ctx: Context | None,
) -> str:
    """Handle think feedback action for recording suggestion outcomes.

    Records whether a suggestion was accepted or rejected, and what action
    was actually taken. This data is used to adjust suggestion weights
    for future recommendations.
    """
    if not session_id:
        return json.dumps({"error": "session_id is required for feedback action"})

    if session_id not in manager._sessions:
        return json.dumps({"error": "Invalid or expired session"})

    if not suggestion_id:
        return json.dumps({"error": "suggestion_id is required for feedback action"})

    if not suggestion_outcome:
        return json.dumps({"error": "suggestion_outcome is required for feedback action"})

    # Map string outcome to the expected type
    outcome: Literal["accepted", "rejected"] = suggestion_outcome

    result = manager.record_suggestion_outcome(
        session_id=session_id,
        suggestion_id=suggestion_id,
        outcome=outcome,
        actual_action=actual_action,
    )

    if ctx:
        if result["success"]:
            await ctx.info(
                f"Recorded feedback for suggestion {suggestion_id}: {suggestion_outcome}"
            )
            if actual_action and actual_action != result.get("recorded_action"):
                await ctx.info(f"Actual action taken: {actual_action}")
            await ctx.info(f"Updated weights: {result.get('updated_weights', {})}")
        else:
            await ctx.warning(f"Failed to record feedback: {result.get('error')}")

    return json.dumps(result, indent=2, default=str)


async def _think_auto(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    max_auto_steps: int,
    stop_on_high_risk: bool,
    ctx: Context | None,
) -> str:
    """Handle think auto action for auto-executing suggested actions.

    Automatically executes the top suggested action up to max_auto_steps times,
    stopping at high-risk checkpoints if stop_on_high_risk is True.

    Note: For actions that require thought content (continue, resolve, synthesize),
    auto-execution will generate placeholder content. For full LLM integration,
    use the UnifiedReasonerManager.auto_execute_suggestion() method directly
    with a thought_generator callback.
    """
    if not session_id:
        return json.dumps({"error": "session_id is required for auto action"})

    if session_id not in manager._sessions:
        return json.dumps({"error": "Invalid or expired session"})

    if max_auto_steps < 1:
        return json.dumps({"error": "max_auto_steps must be at least 1"})

    if max_auto_steps > 20:
        return json.dumps({"error": "max_auto_steps cannot exceed 20"})

    # Execute auto steps
    result = await manager.auto_execute_suggestion(
        session_id=session_id,
        max_auto_steps=max_auto_steps,
        stop_on_high_risk=stop_on_high_risk,
        thought_generator=None,  # No LLM integration at server level
    )

    if ctx:
        actions_executed = result.get("actions_executed", [])
        total_executed = len(actions_executed)

        if total_executed > 0:
            await ctx.info(f"Auto-executed {total_executed} action(s)")
            for i, action_info in enumerate(actions_executed, 1):
                action_name = action_info.get("action", "unknown")
                success = action_info.get("success", False)
                status = "✓" if success else "✗"
                await ctx.info(f"  {i}. {action_name} {status}")
        else:
            await ctx.info("No actions auto-executed")

        # Report stop reason
        stop_reason = result.get("stop_reason")
        if stop_reason:
            if stop_reason == "high_risk_checkpoint":
                await ctx.warning("Stopped at high-risk checkpoint (requires human review)")
            elif stop_reason == "max_steps_reached":
                await ctx.info(f"Reached max auto steps ({max_auto_steps})")
            elif stop_reason == "session_finished":
                await ctx.info("Session finished")
            elif stop_reason == "no_suggestion":
                await ctx.info("No more suggestions available")
            else:
                await ctx.info(f"Stopped: {stop_reason}")

        # Show session summary
        session_state = result.get("session_state", {})
        if session_state:
            progress = session_state.get("progress", 0)
            total_thoughts = session_state.get("total_thoughts", 0)
            await ctx.info(f"Progress: {progress:.0%} ({total_thoughts} thoughts)")

    return json.dumps(result, indent=2, default=str)


# =============================================================================
# TOOL 2: COMPRESS
# =============================================================================


@mcp.tool
async def compress(
    context: str,
    query: str,
    ratio: float = 0.3,
    ctx: Context | None = None,
) -> str:
    """Compress long context using semantic-level sentence filtering.

    Reduces token count while preserving semantic relevance to the query.
    Uses sentence embeddings to score and select most relevant content.

    Args:
        context: Long text to compress (required)
        query: Query to determine relevance (required)
        ratio: Target compression ratio 0.1-1.0 (default 0.3)

    Returns:
        JSON with compressed_context, compression_ratio, tokens_saved

    """
    try:
        if not query:
            return json.dumps({"error": "query is required for compression"})

        if ctx:
            await ctx.info(f"Compressing {len(context)} characters...")

        tool = get_compression_tool()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            partial(
                tool.compress,
                context=context,
                question=query,
                compression_ratio=ratio,
                preserve_order=True,
            ),
        )

        if ctx:
            tokens_saved = result.original_tokens - result.compressed_tokens
            await ctx.info(
                f"Compressed to {result.compression_ratio:.1%} ({tokens_saved} tokens saved)"
            )

        # Convert result to dict for JSON serialization
        return json.dumps(
            {
                "compressed": result.compressed_context,
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "compression_ratio": result.compression_ratio,
                "tokens_saved": result.original_tokens - result.compressed_tokens,
                "max_relevance_score": max(
                    (score for _, score in result.relevance_scores), default=0.0
                ),
                "mean_relevance_score": sum(score for _, score in result.relevance_scores)
                / len(result.relevance_scores)
                if result.relevance_scores
                else 0.0,
            },
            indent=2,
        )

    except ModelNotReadyException as e:
        error = ToolExecutionError("compress", str(e), {"retry_after_seconds": 30})
        logger.warning(f"Model not ready: {e}")
        return json.dumps(error.to_dict())
    except MatrixMindException as e:
        error = ToolExecutionError("compress", str(e))
        logger.error(f"Compression failed: {e}")
        return json.dumps(error.to_dict())
    except Exception as e:
        error = ToolExecutionError("compress", str(e), {"type": type(e).__name__})
        logger.error(f"Unexpected error: {e}")
        return json.dumps(error.to_dict())


# =============================================================================
# TOOL 3: STATUS
# =============================================================================


@mcp.tool
async def status(
    session_id: str | None = None,
    ctx: Context | None = None,
) -> str:
    """Get server status or specific session status.

    Args:
        session_id: Optional session ID to get specific session status

    Returns:
        JSON with server info and session counts, or specific session state

    """
    try:
        manager = get_unified_manager()

        # If session_id provided, return that session's status
        if session_id:
            if session_id not in manager._sessions:
                return json.dumps({"error": f"Session not found: {session_id}"})

            result = await manager.get_status(session_id)
            return json.dumps(result, indent=2, default=str)

        # Otherwise return server status
        model_manager = ModelManager.get_instance()
        model_status = model_manager.get_status()
        rate_limiter = get_rate_limiter()

        # Count sessions by status
        active_count = 0
        completed_count = 0
        for session in manager._sessions.values():
            if session.status == SessionStatus.ACTIVE:
                active_count += 1
            elif session.status == SessionStatus.COMPLETED:
                completed_count += 1

        status_result: dict[str, Any] = {
            "server": {
                "name": SERVER_NAME,
                "transport": SERVER_TRANSPORT,
                "tools": ["think", "compress", "status"],
                "version": "2.0.0",  # Unified reasoner version
            },
            "model": model_status,
            "embedding_model": _get_embedding_model_name(),
            "features": {
                "auto_mode_selection": True,
                "blind_spot_detection": True,
                "rlvr_rewards": True,
                "mppa_integration": True,
                "cisc_integration": True,
                "rag_enabled": ENABLE_RAG,
            },
            "sessions": {
                "total": len(manager._sessions),
                "active": active_count,
                "completed": completed_count,
                "max_total": MAX_TOTAL_SESSIONS,
            },
            # Backwards compatibility alias
            "active_sessions": {
                "total": len(manager._sessions),
                "active": active_count,
                "completed": completed_count,
                "max_total": MAX_TOTAL_SESSIONS,
            },
            "rate_limit": rate_limiter.get_stats(),
            "cleanup": {
                "max_age_minutes": SESSION_MAX_AGE_MINUTES,
                "interval_seconds": CLEANUP_INTERVAL_SECONDS,
                "task_running": _cleanup_task is not None and not _cleanup_task.done(),
            },
        }

        if ctx:
            state = model_status.get("state", "unknown")
            await ctx.info(f"Server ready, model state: {state}")

        return json.dumps(status_result, indent=2)

    except Exception as e:
        error = ToolExecutionError("status", str(e))
        logger.error(f"Status check failed: {e}")
        return json.dumps(error.to_dict())


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================


async def _init_unified_manager_async() -> None:
    """Initialize the unified manager with optional vector store."""
    vector_store = None

    if ENABLE_RAG:
        try:
            from src.models.vector_store import AsyncVectorStore, VectorStoreConfig

            # Validate path before use (CWE-22 prevention)
            validated_path = _validate_vector_db_path()
            config = VectorStoreConfig(db_path=validated_path)
            vector_store = AsyncVectorStore(config)
            await vector_store.__aenter__()
            logger.info(f"RAG enabled with vector store at {validated_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize vector store, RAG disabled: {e}")
            vector_store = None

    await init_unified_manager(vector_store=vector_store)
    logger.info("Unified reasoner manager initialized")


def main() -> None:
    """Run the MatrixMind MCP server."""
    logger.info(f"Starting {SERVER_NAME} (transport: {SERVER_TRANSPORT})")

    # Initialize embedding model
    _init_model_manager()

    # Initialize unified manager (sync wrapper for async init)
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_init_unified_manager_async())
        loop.close()
    except Exception as e:
        logger.warning(f"Async init failed, using default manager: {e}")
        # Fallback to sync initialization without vector store
        get_unified_manager()

    # Start background cleanup task
    _start_cleanup_task()

    try:
        if SERVER_TRANSPORT == "stdio":
            mcp.run(transport="stdio")
        elif SERVER_TRANSPORT == "http":
            mcp.run(transport="streamable-http", host=SERVER_HOST, port=SERVER_PORT)
        elif SERVER_TRANSPORT == "sse":
            mcp.run(transport="sse", host=SERVER_HOST, port=SERVER_PORT)
        else:
            logger.warning(f"Unknown transport '{SERVER_TRANSPORT}', falling back to stdio")
            mcp.run(transport="stdio")
    finally:
        _stop_cleanup_task()


if __name__ == "__main__":
    main()
