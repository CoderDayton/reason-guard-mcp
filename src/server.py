"""Enhanced Chain-of-Thought MCP Server.

FastMCP 2.0 implementation providing 4 reasoning tools:
1. compress_prompt - Semantic context compression
2. matrix_of_thought_reasoning - Multi-perspective reasoning
3. long_chain_of_thought - Deep sequential reasoning
4. verify_fact_consistency - Answer verification

Run with: fastmcp run src/server.py
Or: python -m src.server
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from loguru import logger

from src.models.llm_client import LLMClient
from src.tools.compress import ContextAwareCompressionTool
from src.tools.long_chain import LongChainOfThoughtTool
from src.tools.mot_reasoning import MatrixOfThoughtTool
from src.tools.verify import FactVerificationTool
from src.utils.errors import EnhancedCoTException, ToolExecutionError
from src.utils.schema import ReasoningStrategy, safe_json_serialize

# Load environment variables
load_dotenv()

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
else:
    logger.warning("config.yaml not found, using defaults")
    config = {
        "server": {"name": "Enhanced-CoT-MCP", "transport": "stdio"},
        "models": {"reasoning_llm": "gpt-4-turbo", "embedding_model": "all-mpnet-base-v2"},
        "llm": {"timeout": 60, "retry_attempts": 3},
    }

# Initialize FastMCP server
mcp = FastMCP(
    name=config["server"]["name"],
    instructions="""Enhanced Chain-of-Thought reasoning server with 4 tools:

1. compress_prompt: Reduce token count for long documents (use first for large inputs)
2. matrix_of_thought_reasoning: Multi-perspective reasoning for complex problems
3. long_chain_of_thought: Deep sequential reasoning for serial problems
4. verify_fact_consistency: Verify answer accuracy against context

Typical workflow: compress → reason (MoT or long_chain) → verify
""",
)

# Lazy-loaded tool instances (initialized on first use)
_llm_client: LLMClient | None = None
_compression_tool: ContextAwareCompressionTool | None = None
_mot_tool: MatrixOfThoughtTool | None = None
_long_chain_tool: LongChainOfThoughtTool | None = None
_verify_tool: FactVerificationTool | None = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client instance."""
    global _llm_client
    if _llm_client is None:
        # Environment variable takes precedence over config
        base_url = os.getenv("OPENAI_BASE_URL") or config.get("llm", {}).get("base_url") or None
        # Empty string from config should be treated as None
        if base_url == "":
            base_url = None

        _llm_client = LLMClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            model=os.getenv("OPENAI_MODEL") or config["models"]["reasoning_llm"],
            timeout=config["llm"]["timeout"],
            max_retries=config["llm"]["retry_attempts"],
        )
    return _llm_client


def get_compression_tool() -> ContextAwareCompressionTool:
    """Get or create compression tool instance."""
    global _compression_tool
    if _compression_tool is None:
        model_name = config["models"].get(
            "embedding_model", "sentence-transformers/all-mpnet-base-v2"
        )
        if not model_name.startswith("sentence-transformers/"):
            model_name = f"sentence-transformers/{model_name}"
        _compression_tool = ContextAwareCompressionTool(model_name=model_name)
    return _compression_tool


def get_mot_tool() -> MatrixOfThoughtTool:
    """Get or create MoT tool instance."""
    global _mot_tool
    if _mot_tool is None:
        _mot_tool = MatrixOfThoughtTool(get_llm_client())
    return _mot_tool


def get_long_chain_tool() -> LongChainOfThoughtTool:
    """Get or create long chain tool instance."""
    global _long_chain_tool
    if _long_chain_tool is None:
        _long_chain_tool = LongChainOfThoughtTool(get_llm_client())
    return _long_chain_tool


def get_verify_tool() -> FactVerificationTool:
    """Get or create verification tool instance."""
    global _verify_tool
    if _verify_tool is None:
        _verify_tool = FactVerificationTool(get_llm_client())
    return _verify_tool


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
        context: Long text to compress (required, can be 5K-50K+ tokens)
        question: Query to determine relevance (required)
        compression_ratio: Target ratio 0.1-1.0 (default 0.3 = 3× compression)
        preserve_order: Keep original sentence order (default true)

    Returns:
        JSON with compressed_context, compression_ratio, tokens_saved,
        sentences_kept, sentences_removed, top_relevance_scores

    Use when:
        - Input documents are very long (>3000 tokens)
        - Need faster reasoning with reduced context
        - Want to preserve semantic meaning while reducing cost

    Performance:
        - 10.93× faster than token-level methods
        - Preserves quality (-0.3 F1 vs -2.8 for baselines)

    Example:
        compress_prompt(
            context="<20 page document>",
            question="What was the main finding?",
            compression_ratio=0.3
        )

    """
    try:
        if ctx:
            await ctx.info(f"Compressing {len(context)} characters...")

        tool = get_compression_tool()
        result = tool.compress(
            context=context,
            question=question,
            compression_ratio=compression_ratio,
            preserve_order=preserve_order,
        )

        if ctx:
            await ctx.info(
                f"Compressed to {result.compression_ratio:.1%} ({result.tokens_saved} tokens saved)"
            )

        return safe_json_serialize(result)

    except EnhancedCoTException as e:
        error = ToolExecutionError("compress_prompt", str(e))
        logger.error(f"Compression failed: {e}")
        return json.dumps(error.to_dict())
    except Exception as e:
        error = ToolExecutionError("compress_prompt", str(e), {"type": type(e).__name__})
        logger.error(f"Unexpected error in compress_prompt: {e}")
        return json.dumps(error.to_dict())


# ============================================================================
# TOOL 2: MATRIX OF THOUGHT REASONING
# ============================================================================


@mcp.tool
async def matrix_of_thought_reasoning(
    question: str,
    context: str,
    matrix_rows: int = 3,
    matrix_cols: int = 4,
    communication_pattern: str = "vert&hor-01",
    ctx: Context | None = None,
) -> str:
    """Multi-dimensional reasoning combining breadth and depth.

    Organizes reasoning in an m×n matrix:
    - Rows = different reasoning strategies (breadth)
    - Columns = iterative refinement steps (depth)
    - Inter-cell communication enables knowledge sharing

    Args:
        question: Problem to solve (required)
        context: Background information (required)
        matrix_rows: Number of strategies 2-5 (default 3)
        matrix_cols: Number of refinements 2-5 (default 4)
        communication_pattern: Weight pattern (default "vert&hor-01")
            - "vert&hor-01": Gradual increase in communication
            - "uniform": Equal communication everywhere
            - "none": Independent cells (like standard ToT)

    Returns:
        JSON with answer, confidence (0-1), reasoning_steps,
        matrix_shape, total_thoughts, num_refinements

    Use when:
        - Multi-hop reasoning needed (3+ logical steps)
        - Multiple perspectives could improve answer
        - Problem has complex constraints
        - Need to explore diverse strategies

    Performance:
        - 7× faster than RATT baseline
        - +4.2% F1 improvement on HotpotQA
        - Generalizes CoT (1×n) and ToT (α=0) as special cases

    Example:
        matrix_of_thought_reasoning(
            question="Who wrote the paper that introduced transformers?",
            context="<research paper summaries>",
            matrix_rows=3,
            matrix_cols=4
        )

    """
    try:
        if ctx:
            await ctx.info(f"Starting MoT reasoning ({matrix_rows}×{matrix_cols} matrix)...")

        tool = get_mot_tool()
        result = tool.reason(
            question=question,
            context=context,
            matrix_rows=matrix_rows,
            matrix_cols=matrix_cols,
            communication_pattern=communication_pattern,
        )

        if ctx:
            await ctx.info(f"Generated answer with {result.confidence:.1%} confidence")

        return safe_json_serialize(result)

    except EnhancedCoTException as e:
        error = ToolExecutionError("matrix_of_thought_reasoning", str(e))
        logger.error(f"MoT reasoning failed: {e}")
        return json.dumps(error.to_dict())
    except Exception as e:
        error = ToolExecutionError("matrix_of_thought_reasoning", str(e))
        logger.error(f"Unexpected error in MoT: {e}")
        return json.dumps(error.to_dict())


# ============================================================================
# TOOL 3: LONG CHAIN OF THOUGHT
# ============================================================================


@mcp.tool
async def long_chain_of_thought(
    problem: str,
    num_steps: int = 15,
    verify_intermediate: bool = True,
    ctx: Context | None = None,
) -> str:
    """Sequential step-by-step reasoning with verification checkpoints.

    Implements deep reasoning where each step builds directly on previous.
    Includes optional intermediate verification to catch errors early.

    Args:
        problem: Problem statement (required)
        num_steps: Number of reasoning steps 1-50 (default 15)
        verify_intermediate: Check consistency every 3 steps (default true)

    Returns:
        JSON with answer, confidence (0.6-0.8), reasoning_steps,
        verification_results, tokens_used

    Use when:
        - Problem has strong serial dependencies
        - Each step fundamentally builds on previous
        - High accuracy more important than speed
        - Problem requires deep logical chain

    When to use over matrix_of_thought:
        - Graph connectivity problems (exponential advantage)
        - Constraint satisfaction (permutations, ordering)
        - Arithmetic with dependencies (iterated operations)
        - Pure serial problems (no parallel decomposition)

    Performance:
        - Exponential advantage over parallel for serial problems
        - +47% improvement on constraint solving (66% vs 36%)
        - Verification catches errors early

    Example:
        long_chain_of_thought(
            problem="Make 24 using the numbers 3, 4, 5, 6",
            num_steps=10,
            verify_intermediate=True
        )

    """
    try:
        if ctx:
            await ctx.info(f"Starting long-chain reasoning ({num_steps} steps)...")

        tool = get_long_chain_tool()
        result = tool.reason(
            problem=problem,
            num_steps=num_steps,
            verify_intermediate=verify_intermediate,
        )

        if ctx:
            verif = result.verification_results or {}
            await ctx.info(
                f"Completed with {verif.get('passed', 0)}/{verif.get('total_verifications', 0)} "
                f"verifications passed"
            )

        return safe_json_serialize(result)

    except EnhancedCoTException as e:
        error = ToolExecutionError("long_chain_of_thought", str(e))
        logger.error(f"Long chain reasoning failed: {e}")
        return json.dumps(error.to_dict())
    except Exception as e:
        error = ToolExecutionError("long_chain_of_thought", str(e))
        logger.error(f"Unexpected error in long_chain: {e}")
        return json.dumps(error.to_dict())


# ============================================================================
# TOOL 4: VERIFY FACT CONSISTENCY
# ============================================================================


@mcp.tool
async def verify_fact_consistency(
    answer: str,
    context: str,
    max_claims: int = 10,
    ctx: Context | None = None,
) -> str:
    """Verify answer claims against knowledge base/context.

    Extracts factual claims from answer and checks each against context.
    Helps prevent hallucinations and ensures answer accuracy.

    Args:
        answer: Answer text to verify (required)
        context: Factual context for verification (required)
        max_claims: Maximum claims to check 1-20 (default 10)

    Returns:
        JSON with verified (bool), confidence (0-1), claims_verified,
        claims_total, reason, recommendation

    Use when:
        - Need to ensure answer factuality
        - QA with external knowledge base
        - Preventing hallucinations
        - Quality assurance on generated answers

    Confidence levels:
        - 0.9-1.0: All claims supported → highly reliable
        - 0.7-0.9: Most claims supported → verified=true
        - 0.5-0.7: Mixed support → use caution
        - <0.5: Few claims supported → verified=false

    Example:
        verify_fact_consistency(
            answer="Einstein published relativity in 1905 and 1915.",
            context="Albert Einstein published special relativity in 1905...",
            max_claims=5
        )

    """
    try:
        if ctx:
            await ctx.info(f"Verifying answer ({len(answer.split())} words)...")

        tool = get_verify_tool()
        result = tool.verify(
            answer=answer,
            context=context,
            max_claims=max_claims,
        )

        if ctx:
            await ctx.info(f"Verification: {result.reason}")

        return safe_json_serialize(result)

    except EnhancedCoTException as e:
        error = ToolExecutionError("verify_fact_consistency", str(e))
        logger.error(f"Verification failed: {e}")
        return json.dumps(error.to_dict())
    except Exception as e:
        error = ToolExecutionError("verify_fact_consistency", str(e))
        logger.error(f"Unexpected error in verification: {e}")
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

    Analyzes problem structure to recommend the best reasoning approach
    based on serial vs parallel nature and resource constraints.

    Args:
        problem: Problem description (required)
        token_budget: Available tokens 500-20000 (default 4000)

    Returns:
        JSON with recommended_strategy, estimated_depth_steps,
        estimated_tokens_needed, expressiveness_guarantee,
        strategy_confidence, explanation

    Strategies:
        - long_chain: For serial problems with strong dependencies
        - matrix: For multi-hop reasoning needing multiple perspectives
        - parallel_voting: For simple problems or exploration

    Decision logic (from research papers):
        - High serial dependency indicators → long_chain
        - Multi-path benefits → matrix
        - Complex exploration with budget → parallel_voting

    Example:
        recommend_reasoning_strategy(
            problem="Find if there's a path from A to D in graph",
            token_budget=5000
        )

    """
    try:
        problem_lower = problem.lower()

        # Heuristic indicators for serial vs parallel
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

        # Determine strategy
        if serial_count > parallel_count + 1:
            strategy = ReasoningStrategy.LONG_CHAIN
            depth = min(token_budget // 250, 20)
            explanation = f"Detected {serial_count} serial indicators (path, sequence, etc.)"
        elif parallel_count > serial_count:
            strategy = ReasoningStrategy.PARALLEL
            depth = min(token_budget // 300, 10)
            explanation = f"Detected {parallel_count} parallel indicators (multiple, explore, etc.)"
        else:
            strategy = ReasoningStrategy.MATRIX
            depth = 4
            explanation = "Balanced problem - matrix combines breadth and depth"

        # Calculate confidence
        indicator_diff = abs(serial_count - parallel_count)
        confidence = min(0.5 + 0.1 * indicator_diff, 0.9)

        result: dict[str, Any] = {
            "recommended_strategy": strategy.value,
            "estimated_depth_steps": depth,
            "estimated_tokens_needed": depth * 250,
            "expressiveness_guarantee": True,
            "strategy_confidence": round(confidence, 3),
            "explanation": explanation,
            "indicators": {
                "serial_count": serial_count,
                "parallel_count": parallel_count,
            },
        }

        if ctx:
            await ctx.info(f"Recommended: {strategy.value} (confidence: {confidence:.0%})")

        return json.dumps(result, indent=2)

    except Exception as e:
        error = ToolExecutionError("recommend_reasoning_strategy", str(e))
        logger.error(f"Strategy recommendation failed: {e}")
        return json.dumps(error.to_dict())


# ============================================================================
# SERVER ENTRY POINT
# ============================================================================


def main() -> None:
    """Run the Enhanced CoT MCP server."""
    transport = config["server"].get("transport", "stdio")

    logger.info(f"Starting Enhanced CoT MCP Server (transport: {transport})")

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "http":
        mcp.run(
            transport="streamable-http",
            host=config["server"].get("host", "localhost"),
            port=config["server"].get("port", 8000),
        )
    elif transport == "sse":
        mcp.run(
            transport="sse",
            host=config["server"].get("host", "localhost"),
            port=config["server"].get("port", 8000),
        )
    else:
        logger.warning(f"Unknown transport '{transport}', falling back to stdio")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
