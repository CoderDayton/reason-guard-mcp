"""MatrixMind MCP Server.

FastMCP 2.0 implementation providing reasoning state management tools.
The calling LLM does all reasoning; these tools track and organize the process.

Tools:
1. think - Unified reasoning tool (chain/matrix/verify modes)
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
from src.tools.long_chain import get_chain_manager
from src.tools.mot_reasoning import get_matrix_manager
from src.tools.verify import get_verification_manager
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
SERVER_HOST = _get_env("SERVER_HOST", "localhost")
SERVER_PORT = _get_env_int("SERVER_PORT", 8000)

# Rate Limiting Configuration
RATE_LIMIT_MAX_SESSIONS = _get_env_int("RATE_LIMIT_MAX_SESSIONS", 100)
RATE_LIMIT_WINDOW_SECONDS = _get_env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
MAX_TOTAL_SESSIONS = _get_env_int("MAX_TOTAL_SESSIONS", 500)

# Session Cleanup Configuration
SESSION_MAX_AGE_MINUTES = _get_env_int("SESSION_MAX_AGE_MINUTES", 30)
CLEANUP_INTERVAL_SECONDS = _get_env_int("CLEANUP_INTERVAL_SECONDS", 60)

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

            chain_mgr = get_chain_manager()
            matrix_mgr = get_matrix_manager()
            verify_mgr = get_verification_manager()

            chain_removed = chain_mgr.cleanup_stale(max_age)
            matrix_removed = matrix_mgr.cleanup_stale(max_age)
            verify_removed = verify_mgr.cleanup_stale(max_age)

            total_removed = len(chain_removed) + len(matrix_removed) + len(verify_removed)
            if total_removed > 0:
                logger.info(
                    f"Cleaned up {total_removed} stale sessions: "
                    f"chain={len(chain_removed)}, matrix={len(matrix_removed)}, "
                    f"verify={len(verify_removed)}"
                )

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
    instructions="""MatrixMind reasoning state manager with 3 unified tools.

ARCHITECTURE: You (the calling LLM) do ALL reasoning. These tools TRACK and ORGANIZE your thoughts.

TOOLS:

1. think(action, ...) - Unified reasoning across all paradigms
   Actions:
   - "start": Begin reasoning (mode: chain/matrix/verify)
   - "continue": Add reasoning step
   - "branch": Branch from a step (chain only)
   - "revise": Revise a step (chain only)
   - "synthesize": Synthesize column (matrix only)
   - "verify": Verify a claim (verify mode only)
   - "finish": Complete reasoning

2. compress(context, query, ratio) - Reduce context size semantically

3. status(session_id?) - Get server or session status

WORKFLOW EXAMPLE (Chain):
1. think(action="start", mode="chain", problem="Solve X", expected_steps=5)
2. think(action="continue", session_id=ID, thought="Step 1: ...")
3. think(action="continue", session_id=ID, thought="Step 2: ...")
4. think(action="finish", session_id=ID, thought="The answer is...")

WORKFLOW EXAMPLE (Matrix):
1. think(action="start", mode="matrix", problem="Analyze Y", rows=3, cols=2)
2. think(action="continue", session_id=ID, row=0, col=0, thought="...")
3. think(action="synthesize", session_id=ID, col=0, thought="Column synthesis")
4. think(action="finish", session_id=ID, thought="Final analysis...")

WORKFLOW EXAMPLE (Verify):
1. think(action="start", mode="verify", problem="Claim to verify", context="...")
2. think(action="continue", session_id=ID, thought="Sub-claim 1")
3. think(action="verify", session_id=ID, claim_id=1, verdict="supported", evidence="...")
4. think(action="finish", session_id=ID)
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

ThinkAction = Literal["start", "continue", "branch", "revise", "synthesize", "verify", "finish"]
ThinkMode = Literal["chain", "matrix", "verify"]
VerifyVerdict = Literal["supported", "contradicted", "unclear"]


# =============================================================================
# TOOL 1: THINK (Unified Reasoning)
# =============================================================================


@mcp.tool
async def think(
    action: ThinkAction,
    mode: ThinkMode | None = None,
    session_id: str | None = None,
    problem: str | None = None,
    context: str | None = None,
    thought: str | None = None,
    expected_steps: int = 10,
    rows: int = 3,
    cols: int = 4,
    row: int | None = None,
    col: int | None = None,
    branch_from: int | None = None,
    revises: int | None = None,
    claim_id: int | None = None,
    verdict: VerifyVerdict | None = None,
    evidence: str | None = None,
    confidence: float | None = None,
    ctx: Context | None = None,
) -> str:
    """Unified reasoning tool supporting chain, matrix, and verification modes.

    Actions:
        start: Begin a new reasoning session (requires mode and problem)
        continue: Add a reasoning step (requires session_id and thought)
        branch: Branch from a step (chain only, requires branch_from)
        revise: Revise a step (chain only, requires revises)
        synthesize: Synthesize a column (matrix only, requires col and thought)
        verify: Verify a claim (verify mode, requires claim_id and verdict)
        finish: Complete reasoning (requires session_id)

    Modes:
        chain: Sequential chain-of-thought reasoning
        matrix: Multi-perspective matrix reasoning (rows x cols)
        verify: Fact verification against context

    Args:
        action: The action to perform (required)
        mode: Reasoning mode for "start" action
        session_id: Session ID for continuing/finishing
        problem: Problem statement for "start" action
        context: Background context (optional)
        thought: Your reasoning content
        expected_steps: Expected steps for chain mode (default 10)
        rows: Matrix rows (default 3)
        cols: Matrix columns (default 4)
        row: Matrix row index (0-based)
        col: Matrix column index (0-based)
        branch_from: Step to branch from (chain only)
        revises: Step to revise (chain only)
        claim_id: Claim ID to verify (verify mode)
        verdict: Verification verdict (supported/contradicted/unclear)
        evidence: Evidence for verification
        confidence: Confidence score (0-1)

    Returns:
        JSON with action result, session state, and next instructions

    """
    try:
        # Route to appropriate handler based on action
        if action == "start":
            return await _think_start(
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
                session_id=session_id,
                thought=thought,
                row=row,
                col=col,
                confidence=confidence,
                ctx=ctx,
            )
        elif action == "branch":
            return await _think_branch(
                session_id=session_id,
                thought=thought,
                branch_from=branch_from,
                ctx=ctx,
            )
        elif action == "revise":
            return await _think_revise(
                session_id=session_id,
                thought=thought,
                revises=revises,
                ctx=ctx,
            )
        elif action == "synthesize":
            return await _think_synthesize(
                session_id=session_id,
                col=col,
                thought=thought,
                ctx=ctx,
            )
        elif action == "verify":
            return await _think_verify(
                session_id=session_id,
                claim_id=claim_id,
                verdict=verdict,
                evidence=evidence,
                confidence=confidence,
                ctx=ctx,
            )
        elif action == "finish":
            return await _think_finish(
                session_id=session_id,
                thought=thought,
                confidence=confidence,
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
    mode: ThinkMode | None,
    problem: str | None,
    context: str | None,
    expected_steps: int,
    rows: int,
    cols: int,
    ctx: Context | None,
) -> str:
    """Handle think start action."""
    if not mode:
        return json.dumps({"error": "mode is required for start action"})
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
    chain_mgr = get_chain_manager()
    matrix_mgr = get_matrix_manager()
    verify_mgr = get_verification_manager()
    total_sessions = (
        len(chain_mgr._sessions) + len(matrix_mgr._sessions) + len(verify_mgr._sessions)
    )

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

    result: dict[str, Any]
    if mode == "chain":
        result = chain_mgr.start_chain(problem=problem, expected_steps=expected_steps)
        result["mode"] = "chain"
        if ctx:
            await ctx.info(f"Started chain session {result['session_id']}")

    elif mode == "matrix":
        result = matrix_mgr.start_matrix(
            question=problem,
            context=context or "",
            rows=rows,
            cols=cols,
        )
        result["mode"] = "matrix"
        if ctx:
            await ctx.info(f"Started {rows}x{cols} matrix session {result['session_id']}")

    elif mode == "verify":
        result = verify_mgr.start_verification(answer=problem, context=context or "")
        result["mode"] = "verify"
        if ctx:
            await ctx.info(f"Started verification session {result['session_id']}")

    else:
        return json.dumps({"error": f"Unknown mode: {mode}"})

    return json.dumps(result, indent=2)


async def _think_continue(
    session_id: str | None,
    thought: str | None,
    row: int | None,
    col: int | None,
    confidence: float | None,
    ctx: Context | None,
) -> str:
    """Handle think continue action."""
    if not session_id:
        return json.dumps({"error": "session_id is required for continue action"})
    if not thought:
        return json.dumps({"error": "thought is required for continue action"})

    # Detect session type and route appropriately
    chain_mgr = get_chain_manager()
    matrix_mgr = get_matrix_manager()
    verify_mgr = get_verification_manager()

    if session_id in chain_mgr._sessions:
        result = chain_mgr.add_step(
            session_id=session_id,
            thought=thought,
            step_type="continuation",
            confidence=confidence,
        )
        if ctx and "step_added" in result:
            await ctx.info(f"Added step {result['step_added']}")

    elif session_id in matrix_mgr._sessions:
        if row is None or col is None:
            return json.dumps({"error": "row and col are required for matrix continue"})
        result = matrix_mgr.set_cell(
            session_id=session_id,
            row=row,
            col=col,
            thought=thought,
            confidence=confidence,
        )
        if ctx and "cell_set" in result:
            await ctx.info(f"Set cell ({row},{col})")

    elif session_id in verify_mgr._sessions:
        result = verify_mgr.add_claim(session_id=session_id, content=thought)
        if ctx and "claim_id" in result:
            await ctx.info(f"Added claim {result['claim_id']}")

    else:
        return json.dumps({"error": f"Session not found: {session_id}"})

    return json.dumps(result, indent=2)


async def _think_branch(
    session_id: str | None,
    thought: str | None,
    branch_from: int | None,
    ctx: Context | None,
) -> str:
    """Handle think branch action (chain only)."""
    if not session_id:
        return json.dumps({"error": "session_id is required for branch action"})
    if not thought:
        return json.dumps({"error": "thought is required for branch action"})
    if branch_from is None:
        return json.dumps({"error": "branch_from is required for branch action"})

    manager = get_chain_manager()
    if session_id not in manager._sessions:
        return json.dumps({"error": f"Chain session not found: {session_id}"})

    result = manager.add_step(
        session_id=session_id,
        thought=thought,
        step_type="branch",
        branch_from=branch_from,
    )

    if ctx:
        await ctx.info(f"Branched from step {branch_from}")

    return json.dumps(result, indent=2)


async def _think_revise(
    session_id: str | None,
    thought: str | None,
    revises: int | None,
    ctx: Context | None,
) -> str:
    """Handle think revise action (chain only)."""
    if not session_id:
        return json.dumps({"error": "session_id is required for revise action"})
    if not thought:
        return json.dumps({"error": "thought is required for revise action"})
    if revises is None:
        return json.dumps({"error": "revises is required for revise action"})

    manager = get_chain_manager()
    if session_id not in manager._sessions:
        return json.dumps({"error": f"Chain session not found: {session_id}"})

    result = manager.add_step(
        session_id=session_id,
        thought=thought,
        step_type="revision",
        revises=revises,
    )

    if ctx:
        await ctx.info(f"Revised step {revises}")

    return json.dumps(result, indent=2)


async def _think_synthesize(
    session_id: str | None,
    col: int | None,
    thought: str | None,
    ctx: Context | None,
) -> str:
    """Handle think synthesize action (matrix only)."""
    if not session_id:
        return json.dumps({"error": "session_id is required for synthesize action"})
    if col is None:
        return json.dumps({"error": "col is required for synthesize action"})
    if not thought:
        return json.dumps({"error": "thought is required for synthesize action"})

    manager = get_matrix_manager()
    if session_id not in manager._sessions:
        return json.dumps({"error": f"Matrix session not found: {session_id}"})

    result = manager.synthesize_column(session_id=session_id, col=col, synthesis=thought)

    if ctx:
        await ctx.info(f"Synthesized column {col}")

    return json.dumps(result, indent=2)


async def _think_verify(
    session_id: str | None,
    claim_id: int | None,
    verdict: VerifyVerdict | None,
    evidence: str | None,
    confidence: float | None,
    ctx: Context | None,
) -> str:
    """Handle think verify action (verify mode only)."""
    if not session_id:
        return json.dumps({"error": "session_id is required for verify action"})
    if claim_id is None:
        return json.dumps({"error": "claim_id is required for verify action"})
    if not verdict:
        return json.dumps({"error": "verdict is required for verify action"})

    manager = get_verification_manager()
    if session_id not in manager._sessions:
        return json.dumps({"error": f"Verification session not found: {session_id}"})

    result = manager.verify_claim(
        session_id=session_id,
        claim_id=claim_id,
        status=verdict,
        evidence=evidence,
        confidence=confidence,
    )

    if ctx:
        await ctx.info(f"Verified claim {claim_id} as {verdict}")

    return json.dumps(result, indent=2)


async def _think_finish(
    session_id: str | None,
    thought: str | None,
    confidence: float | None,
    ctx: Context | None,
) -> str:
    """Handle think finish action."""
    if not session_id:
        return json.dumps({"error": "session_id is required for finish action"})

    chain_mgr = get_chain_manager()
    matrix_mgr = get_matrix_manager()
    verify_mgr = get_verification_manager()

    if session_id in chain_mgr._sessions:
        result = chain_mgr.finalize(
            session_id=session_id,
            answer=thought or "",
            confidence=confidence,
        )
        if ctx:
            await ctx.info(f"Finalized chain with {result.get('total_steps', 0)} steps")

    elif session_id in matrix_mgr._sessions:
        result = matrix_mgr.finalize(
            session_id=session_id,
            answer=thought or "",
            confidence=confidence,
        )
        if ctx:
            await ctx.info("Finalized matrix")

    elif session_id in verify_mgr._sessions:
        result = verify_mgr.finalize(session_id=session_id)
        if ctx:
            verified = result.get("verified", False)
            await ctx.info(f"Verification complete: {'✓' if verified else '✗'}")

    else:
        return json.dumps({"error": f"Session not found: {session_id}"})

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
        chain_mgr = get_chain_manager()
        matrix_mgr = get_matrix_manager()
        verify_mgr = get_verification_manager()

        # If session_id provided, return that session's status
        if session_id:
            if session_id in chain_mgr._sessions:
                result = chain_mgr.get_chain(session_id=session_id)
                result["session_type"] = "chain"
            elif session_id in matrix_mgr._sessions:
                result = matrix_mgr.get_matrix(session_id=session_id)
                result["session_type"] = "matrix"
            elif session_id in verify_mgr._sessions:
                result = verify_mgr.get_status(session_id=session_id)
                result["session_type"] = "verify"
            else:
                return json.dumps({"error": f"Session not found: {session_id}"})

            return json.dumps(result, indent=2, default=str)

        # Otherwise return server status
        model_manager = ModelManager.get_instance()
        model_status = model_manager.get_status()
        rate_limiter = get_rate_limiter()

        status_result: dict[str, Any] = {
            "server": {
                "name": SERVER_NAME,
                "transport": SERVER_TRANSPORT,
                "tools": ["think", "compress", "status"],
            },
            "model": model_status,
            "embedding_model": _get_embedding_model_name(),
            "active_sessions": {
                "chain": len(chain_mgr._sessions),
                "matrix": len(matrix_mgr._sessions),
                "verify": len(verify_mgr._sessions),
                "total": len(chain_mgr._sessions)
                + len(matrix_mgr._sessions)
                + len(verify_mgr._sessions),
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


def main() -> None:
    """Run the MatrixMind MCP server."""
    logger.info(f"Starting {SERVER_NAME} (transport: {SERVER_TRANSPORT})")

    # Initialize embedding model
    _init_model_manager()

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
