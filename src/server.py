"""MatrixMind MCP Server.

FastMCP 2.0 implementation providing reasoning state management tools.
The calling LLM does all reasoning; these tools track and organize the process.

Tools:
1. compress_prompt - Semantic context compression (uses local embeddings)
2. chain_* - Long chain-of-thought state management
3. matrix_* - Matrix of thought state management
4. verify_* - Fact verification state management
5. recommend_reasoning_strategy - Strategy recommendation (heuristics)
6. check_status - Server status

Run with: uvx matrixmind-mcp
Or: python -m src.server
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from loguru import logger

from src.models.model_manager import ModelManager
from src.tools.compress import ContextAwareCompressionTool
from src.tools.long_chain import get_chain_manager
from src.tools.mot_reasoning import get_matrix_manager
from src.tools.verify import get_verification_manager
from src.utils.errors import MatrixMindException, ModelNotReadyException, ToolExecutionError
from src.utils.schema import ReasoningStrategy, safe_json_serialize

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
# Initialize FastMCP Server
# =============================================================================

mcp = FastMCP(
    name=SERVER_NAME,
    instructions="""MatrixMind reasoning state manager with tools for structured thinking.

ARCHITECTURE: You (the calling LLM) do ALL reasoning. These tools TRACK and ORGANIZE your thoughts.

TOOLS:

1. compress_prompt - Reduce context size using semantic compression

2. Chain-of-Thought (sequential reasoning):
   - chain_start(problem, expected_steps) → Start a reasoning chain
   - chain_add_step(session_id, thought) → Add your reasoning step
   - chain_finalize(session_id, answer) → Complete with final answer
   - chain_get(session_id) → Get current chain state

3. Matrix-of-Thought (multi-perspective reasoning):
   - matrix_start(question, context, rows, cols) → Start matrix reasoning
   - matrix_set_cell(session_id, row, col, thought) → Fill a matrix cell
   - matrix_synthesize(session_id, col, synthesis) → Synthesize a column
   - matrix_finalize(session_id, answer) → Complete with final answer

4. Fact Verification:
   - verify_start(answer, context) → Start verification session
   - verify_add_claim(session_id, claim) → Add a claim to verify
   - verify_claim(session_id, claim_id, status, evidence) → Verify a claim
   - verify_finalize(session_id) → Get verification result

5. recommend_reasoning_strategy - Get optimal approach for a problem
6. check_status - Server/model status

WORKFLOW EXAMPLE (Chain):
1. Call chain_start(problem="Solve X", expected_steps=5)
2. Think about step 1, call chain_add_step(session_id, thought="Step 1: ...")
3. Continue for each step
4. Call chain_finalize(session_id, answer="The answer is...")
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


# ============================================================================
# TOOL 1: COMPRESS PROMPT
# ============================================================================


@mcp.tool
async def compress_prompt(
    context: str,
    question: str,
    compression_ratio: float = 0.3,
    preserve_order: bool = True,
    ctx: Context | None = None,
) -> str:
    """Compress long context using semantic-level sentence filtering.

    Reduces token count while preserving semantic relevance to the question.
    Uses sentence embeddings to score and select most relevant content.

    Args:
        context: Long text to compress (required)
        question: Query to determine relevance (required)
        compression_ratio: Target ratio 0.1-1.0 (default 0.3)
        preserve_order: Keep original sentence order (default true)

    Returns:
        JSON with compressed_context, compression_ratio, tokens_saved

    """
    try:
        if ctx:
            await ctx.info(f"Compressing {len(context)} characters...")

        tool = get_compression_tool()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            partial(
                tool.compress,
                context=context,
                question=question,
                compression_ratio=compression_ratio,
                preserve_order=preserve_order,
            ),
        )

        if ctx:
            tokens_saved = result.original_tokens - result.compressed_tokens
            await ctx.info(
                f"Compressed to {result.compression_ratio:.1%} ({tokens_saved} tokens saved)"
            )

        return safe_json_serialize(result)

    except ModelNotReadyException as e:
        error = ToolExecutionError("compress_prompt", str(e), {"retry_after_seconds": 30})
        logger.warning(f"Model not ready: {e}")
        return json.dumps(error.to_dict())
    except MatrixMindException as e:
        error = ToolExecutionError("compress_prompt", str(e))
        logger.error(f"Compression failed: {e}")
        return json.dumps(error.to_dict())
    except Exception as e:
        error = ToolExecutionError("compress_prompt", str(e), {"type": type(e).__name__})
        logger.error(f"Unexpected error: {e}")
        return json.dumps(error.to_dict())


# ============================================================================
# TOOL 2: LONG CHAIN OF THOUGHT (State Manager)
# ============================================================================


@mcp.tool
async def chain_start(
    problem: str,
    expected_steps: int = 10,
    ctx: Context | None = None,
) -> str:
    """Start a new chain-of-thought reasoning session.

    YOU do the reasoning. This tool tracks your progress.

    Args:
        problem: The problem to reason about (required)
        expected_steps: How many reasoning steps you plan (default 10)

    Returns:
        JSON with session_id, status, instruction for next step

    Example:
        chain_start(problem="What is 15 * 17?", expected_steps=5)
        → Returns session_id, then YOU reason and call chain_add_step

    """
    try:
        manager = get_chain_manager()
        result = manager.start_chain(problem=problem, expected_steps=expected_steps)

        if ctx:
            await ctx.info(f"Started chain session {result['session_id']}")

        return json.dumps(result, indent=2)
    except Exception as e:
        error = ToolExecutionError("chain_start", str(e))
        logger.error(f"Chain start failed: {e}")
        return json.dumps(error.to_dict())


@mcp.tool
async def chain_add_step(
    session_id: str,
    thought: str,
    step_type: str = "continuation",
    branch_from: int | None = None,
    revises: int | None = None,
    confidence: float | None = None,
    ctx: Context | None = None,
) -> str:
    """Add a reasoning step to an active chain.

    Call this after YOU have reasoned about the next step.

    Args:
        session_id: Session from chain_start (required)
        thought: Your reasoning for this step (required)
        step_type: Type of step - continuation, revision, branch, synthesis
        branch_from: If branching, which step number to branch from
        revises: If revising, which step number this revises
        confidence: Your confidence in this step (0-1)

    Returns:
        JSON with step_added, progress, needs_more_steps, instruction

    Example:
        chain_add_step(
            session_id="abc123",
            thought="Step 1: First, I'll multiply 15 * 10 = 150"
        )

    """
    try:
        manager = get_chain_manager()
        result = manager.add_step(
            session_id=session_id,
            thought=thought,
            step_type=step_type,
            branch_from=branch_from,
            revises=revises,
            confidence=confidence,
        )

        if ctx and "step_added" in result:
            await ctx.info(
                f"Added step {result['step_added']}, progress: {result.get('progress', 0):.0%}"
            )

        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("chain_add_step", str(e))
        logger.error(f"Chain add step failed: {e}")
        return json.dumps(error.to_dict())


@mcp.tool
async def chain_finalize(
    session_id: str,
    answer: str,
    confidence: float | None = None,
    ctx: Context | None = None,
) -> str:
    """Finalize a reasoning chain with your answer.

    Call this when you've completed your reasoning.

    Args:
        session_id: Session from chain_start (required)
        answer: Your final answer (required)
        confidence: Your confidence in the answer (0-1)

    Returns:
        JSON with status, final_answer, summary of chain

    """
    try:
        manager = get_chain_manager()
        result = manager.finalize(session_id=session_id, answer=answer, confidence=confidence)

        if ctx:
            await ctx.info(f"Finalized chain with {result.get('total_steps', 0)} steps")

        return json.dumps(result, indent=2, default=str)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("chain_finalize", str(e))
        logger.error(f"Chain finalize failed: {e}")
        return json.dumps(error.to_dict())


@mcp.tool
async def chain_get(session_id: str, ctx: Context | None = None) -> str:
    """Get the current state of a reasoning chain.

    Args:
        session_id: Session to retrieve (required)

    Returns:
        JSON with full chain state including all steps

    """
    try:
        manager = get_chain_manager()
        result = manager.get_chain(session_id=session_id)
        return json.dumps(result, indent=2, default=str)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("chain_get", str(e))
        logger.error(f"Chain get failed: {e}")
        return json.dumps(error.to_dict())


# ============================================================================
# TOOL 3: MATRIX OF THOUGHT (State Manager)
# ============================================================================


@mcp.tool
async def matrix_start(
    question: str,
    context: str = "",
    rows: int = 3,
    cols: int = 4,
    strategies: list[str] | None = None,
    ctx: Context | None = None,
) -> str:
    """Start a matrix-of-thought reasoning session.

    Matrix structure:
    - Rows = different reasoning strategies (direct, logical, analogical)
    - Columns = iterative refinement steps
    - YOU fill each cell, then synthesize each column

    Args:
        question: The question to reason about (required)
        context: Background context (optional)
        rows: Number of strategies 2-5 (default 3)
        cols: Number of iterations 2-6 (default 4)
        strategies: Custom strategy names (optional)

    Returns:
        JSON with session_id, matrix_dimensions, strategies, next_cell

    Example:
        matrix_start(
            question="What caused the French Revolution?",
            context="<historical text>",
            rows=3,
            cols=4
        )

    """
    try:
        manager = get_matrix_manager()
        result = manager.start_matrix(
            question=question,
            context=context,
            rows=rows,
            cols=cols,
            strategies=strategies,
        )

        if ctx:
            await ctx.info(f"Started {rows}x{cols} matrix session {result['session_id']}")

        return json.dumps(result, indent=2)
    except Exception as e:
        error = ToolExecutionError("matrix_start", str(e))
        logger.error(f"Matrix start failed: {e}")
        return json.dumps(error.to_dict())


@mcp.tool
async def matrix_set_cell(
    session_id: str,
    row: int,
    col: int,
    thought: str,
    confidence: float | None = None,
    ctx: Context | None = None,
) -> str:
    """Fill a cell in the reasoning matrix.

    Args:
        session_id: Session from matrix_start (required)
        row: Row index 0-based (required)
        col: Column index 0-based (required)
        thought: Your reasoning for this cell (required)
        confidence: Confidence in this reasoning (0-1)

    Returns:
        JSON with cell_set, progress, next_cell or pending_synthesis

    Example:
        matrix_set_cell(
            session_id="abc123",
            row=0, col=0,
            thought="Using direct factual analysis: The text states..."
        )

    """
    try:
        manager = get_matrix_manager()
        result = manager.set_cell(
            session_id=session_id,
            row=row,
            col=col,
            thought=thought,
            confidence=confidence,
        )

        if ctx and "cell_set" in result:
            await ctx.info(f"Set cell ({row},{col}), progress: {result.get('progress', 0):.0%}")

        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("matrix_set_cell", str(e))
        logger.error(f"Matrix set cell failed: {e}")
        return json.dumps(error.to_dict())


@mcp.tool
async def matrix_synthesize(
    session_id: str,
    col: int,
    synthesis: str,
    ctx: Context | None = None,
) -> str:
    """Synthesize a column's perspectives into a unified insight.

    Call this after completing all rows in a column.

    Args:
        session_id: Session from matrix_start (required)
        col: Column to synthesize (required)
        synthesis: Your combined insight from all rows (required)

    Returns:
        JSON with column_synthesized, next_cell or completion instruction

    """
    try:
        manager = get_matrix_manager()
        result = manager.synthesize_column(session_id=session_id, col=col, synthesis=synthesis)

        if ctx:
            await ctx.info(f"Synthesized column {col}")

        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("matrix_synthesize", str(e))
        logger.error(f"Matrix synthesize failed: {e}")
        return json.dumps(error.to_dict())


@mcp.tool
async def matrix_finalize(
    session_id: str,
    answer: str,
    confidence: float | None = None,
    ctx: Context | None = None,
) -> str:
    """Finalize matrix reasoning with your answer.

    Args:
        session_id: Session from matrix_start (required)
        answer: Your final answer (required)
        confidence: Confidence in the answer (0-1)

    Returns:
        JSON with status, final_answer, matrix summary

    """
    try:
        manager = get_matrix_manager()
        result = manager.finalize(session_id=session_id, answer=answer, confidence=confidence)

        if ctx:
            await ctx.info("Finalized matrix with answer")

        return json.dumps(result, indent=2, default=str)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("matrix_finalize", str(e))
        logger.error(f"Matrix finalize failed: {e}")
        return json.dumps(error.to_dict())


@mcp.tool
async def matrix_get(session_id: str, ctx: Context | None = None) -> str:
    """Get the current state of a matrix reasoning session.

    Args:
        session_id: Session to retrieve (required)

    Returns:
        JSON with full matrix state

    """
    try:
        manager = get_matrix_manager()
        result = manager.get_matrix(session_id=session_id)
        return json.dumps(result, indent=2, default=str)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("matrix_get", str(e))
        logger.error(f"Matrix get failed: {e}")
        return json.dumps(error.to_dict())


# ============================================================================
# TOOL 4: FACT VERIFICATION (State Manager)
# ============================================================================


@mcp.tool
async def verify_start(
    answer: str,
    context: str,
    ctx: Context | None = None,
) -> str:
    """Start a fact verification session.

    YOU extract claims from the answer, then verify each against context.

    Args:
        answer: The answer to verify (required)
        context: The factual context to verify against (required)

    Returns:
        JSON with session_id, suggested_claims, instruction

    Example:
        verify_start(
            answer="Einstein published relativity in 1905.",
            context="Albert Einstein published special relativity in 1905..."
        )

    """
    try:
        manager = get_verification_manager()
        result = manager.start_verification(answer=answer, context=context)

        if ctx:
            await ctx.info(f"Started verification session {result['session_id']}")

        return json.dumps(result, indent=2)
    except Exception as e:
        error = ToolExecutionError("verify_start", str(e))
        logger.error(f"Verify start failed: {e}")
        return json.dumps(error.to_dict())


@mcp.tool
async def verify_add_claim(
    session_id: str,
    claim: str,
    ctx: Context | None = None,
) -> str:
    """Add a factual claim to verify.

    Extract claims from the answer and add each one.

    Args:
        session_id: Session from verify_start (required)
        claim: A factual assertion to verify (required)

    Returns:
        JSON with claim_id, instruction for verification

    Example:
        verify_add_claim(
            session_id="abc123",
            claim="Einstein published special relativity in 1905"
        )

    """
    try:
        manager = get_verification_manager()
        result = manager.add_claim(session_id=session_id, content=claim)

        if ctx and "claim_id" in result:
            await ctx.info(f"Added claim {result['claim_id']}")

        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("verify_add_claim", str(e))
        logger.error(f"Verify add claim failed: {e}")
        return json.dumps(error.to_dict())


@mcp.tool
async def verify_claim(
    session_id: str,
    claim_id: int,
    status: str,
    evidence: str | None = None,
    confidence: float | None = None,
    ctx: Context | None = None,
) -> str:
    """Verify a claim against the context.

    Check the context and determine if the claim is supported.

    Args:
        session_id: Session from verify_start (required)
        claim_id: ID of the claim to verify (required)
        status: Verification result - "supported", "contradicted", or "unclear"
        evidence: Quote from context supporting your verdict
        confidence: Confidence in verification (0-1)

    Returns:
        JSON with claim status update, summary, next_claim

    Example:
        verify_claim(
            session_id="abc123",
            claim_id=0,
            status="supported",
            evidence="Einstein published his theory of special relativity in 1905"
        )

    """
    try:
        manager = get_verification_manager()
        result = manager.verify_claim(
            session_id=session_id,
            claim_id=claim_id,
            status=status,
            evidence=evidence,
            confidence=confidence,
        )

        if ctx:
            await ctx.info(f"Verified claim {claim_id} as {status}")

        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("verify_claim", str(e))
        logger.error(f"Verify claim failed: {e}")
        return json.dumps(error.to_dict())


@mcp.tool
async def verify_finalize(session_id: str, ctx: Context | None = None) -> str:
    """Finalize verification and get result.

    Call after verifying all claims.

    Args:
        session_id: Session from verify_start (required)

    Returns:
        JSON with verified (bool), confidence, summary, recommendation

    """
    try:
        manager = get_verification_manager()
        result = manager.finalize(session_id=session_id)

        if ctx:
            verified = result.get("verified", False)
            conf = result.get("confidence", 0)
            await ctx.info(
                f"Verification complete: {'✓' if verified else '✗'} ({conf:.0%} confidence)"
            )

        return json.dumps(result, indent=2, default=str)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("verify_finalize", str(e))
        logger.error(f"Verify finalize failed: {e}")
        return json.dumps(error.to_dict())


@mcp.tool
async def verify_get(session_id: str, ctx: Context | None = None) -> str:
    """Get the current state of a verification session.

    Args:
        session_id: Session to retrieve (required)

    Returns:
        JSON with full verification state

    """
    try:
        manager = get_verification_manager()
        result = manager.get_status(session_id=session_id)
        return json.dumps(result, indent=2, default=str)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        error = ToolExecutionError("verify_get", str(e))
        logger.error(f"Verify get failed: {e}")
        return json.dumps(error.to_dict())


# ============================================================================
# TOOL 5: RECOMMEND REASONING STRATEGY
# ============================================================================


@mcp.tool
async def recommend_reasoning_strategy(
    problem: str,
    token_budget: int = 4000,
    ctx: Context | None = None,
) -> str:
    """Get recommendation for optimal reasoning strategy.

    Analyzes problem structure to recommend the best approach.

    Args:
        problem: Problem description (required)
        token_budget: Available tokens 500-20000 (default 4000)

    Returns:
        JSON with recommended_strategy, explanation

    Strategies:
        - chain: For serial problems (use chain_* tools)
        - matrix: For multi-perspective problems (use matrix_* tools)
        - parallel_voting: For simple exploration

    """
    try:
        problem_lower = problem.lower()

        serial_indicators = [
            "order",
            "sequence",
            "step",
            "then",
            "constraint",
            "path",
            "graph",
            "connect",
            "chain",
            "depend",
        ]
        parallel_indicators = [
            "multiple",
            "different",
            "alternative",
            "creative",
            "generate",
            "brainstorm",
            "explore",
            "options",
        ]

        serial_count = sum(1 for ind in serial_indicators if ind in problem_lower)
        parallel_count = sum(1 for ind in parallel_indicators if ind in problem_lower)

        budget_constrained = token_budget < 1000

        if budget_constrained:
            strategy = ReasoningStrategy.LONG_CHAIN
            depth = min(token_budget // 250, 5)
            explanation = f"Low token budget ({token_budget}) - use simple chain"
        elif serial_count > parallel_count + 1:
            strategy = ReasoningStrategy.LONG_CHAIN
            depth = min(token_budget // 250, 20)
            explanation = "Serial problem detected - use chain_* tools"
        elif parallel_count > serial_count:
            strategy = ReasoningStrategy.PARALLEL
            depth = min(token_budget // 300, 10)
            explanation = "Parallel exploration beneficial - consider multiple approaches"
        else:
            strategy = ReasoningStrategy.MATRIX
            depth = 4
            explanation = "Multi-perspective problem - use matrix_* tools"

        indicator_diff = abs(serial_count - parallel_count)
        confidence = min(0.5 + 0.1 * indicator_diff, 0.9)

        result: dict[str, Any] = {
            "recommended_strategy": strategy.value,
            "estimated_depth_steps": depth,
            "strategy_confidence": round(confidence, 3),
            "explanation": explanation,
            "tool_suggestion": (
                "chain_start, chain_add_step, chain_finalize"
                if strategy == ReasoningStrategy.LONG_CHAIN
                else "matrix_start, matrix_set_cell, matrix_synthesize, matrix_finalize"
            ),
        }

        if ctx:
            await ctx.info(f"Recommended: {strategy.value}")

        return json.dumps(result, indent=2)

    except Exception as e:
        error = ToolExecutionError("recommend_reasoning_strategy", str(e))
        logger.error(f"Strategy recommendation failed: {e}")
        return json.dumps(error.to_dict())


# ============================================================================
# TOOL 6: CHECK STATUS
# ============================================================================


@mcp.tool
async def check_status(ctx: Context | None = None) -> str:
    """Get server and model status.

    Returns:
        JSON with model_status, server_info

    """
    try:
        model_manager = ModelManager.get_instance()
        model_status = model_manager.get_status()

        # Get session counts
        chain_mgr = get_chain_manager()
        matrix_mgr = get_matrix_manager()
        verify_mgr = get_verification_manager()

        result: dict[str, Any] = {
            "model_status": model_status,
            "server_info": {
                "name": SERVER_NAME,
                "transport": SERVER_TRANSPORT,
            },
            "embedding_config": {
                "model": _get_embedding_model_name(),
            },
            "active_sessions": {
                "chain": len(chain_mgr._sessions),
                "matrix": len(matrix_mgr._sessions),
                "verify": len(verify_mgr._sessions),
            },
        }

        if ctx:
            state = model_status.get("state", "unknown")
            await ctx.info(f"Model state: {state}")

        return json.dumps(result, indent=2)

    except Exception as e:
        error = ToolExecutionError("check_status", str(e))
        logger.error(f"Status check failed: {e}")
        return json.dumps(error.to_dict())


# ============================================================================
# SERVER ENTRY POINT
# ============================================================================


def main() -> None:
    """Run the MatrixMind MCP server."""
    logger.info(f"Starting {SERVER_NAME} (transport: {SERVER_TRANSPORT})")

    # Initialize embedding model
    _init_model_manager()

    if SERVER_TRANSPORT == "stdio":
        mcp.run(transport="stdio")
    elif SERVER_TRANSPORT == "http":
        mcp.run(transport="streamable-http", host=SERVER_HOST, port=SERVER_PORT)
    elif SERVER_TRANSPORT == "sse":
        mcp.run(transport="sse", host=SERVER_HOST, port=SERVER_PORT)
    else:
        logger.warning(f"Unknown transport '{SERVER_TRANSPORT}', falling back to stdio")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
