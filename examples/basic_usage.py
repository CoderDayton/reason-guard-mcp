"""Basic usage examples for MatrixMind MCP Server.

Run after starting the server:
    fastmcp run src/server.py

Then run this script:
    python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv
from mcp.types import TextContent

load_dotenv()


def format_percent(value: float | str | None, default: str = "N/A") -> str:
    """Format a value as percentage, handling non-numeric gracefully."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    try:
        return f"{value:.1%}"
    except (TypeError, ValueError):
        return default


async def main() -> None:
    """Demonstrate basic tool usage."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Please install fastmcp: pip install fastmcp")
        return

    print("=" * 60)
    print("MatrixMind MCP - Basic Usage Examples")
    print("=" * 60)

    # Connect to server
    async with Client("src/server.py") as client:
        # Example 1: Compress a long document
        print("\n📦 Example 1: Compress Prompt")
        print("-" * 40)

        long_text = (
            """
        The Theory of Relativity, developed by Albert Einstein,
        fundamentally changed our understanding of space and time.
        Einstein published two papers on relativity: Special Relativity in 1905
        and General Relativity in 1915. These theories explained phenomena
        like time dilation and gravity as curvature of spacetime.
        The famous equation E=mc² comes from special relativity.
        Einstein received the Nobel Prize in 1921, though not for relativity,
        but for his work on the photoelectric effect.
        He spent his later years at Princeton working on unified field theory.
        Einstein's work laid the foundation for modern physics and cosmology.
        """
            * 3
        )  # Make it longer

        compress_result = await client.call_tool(
            "compress_prompt",
            {
                "context": long_text,
                "question": "When did Einstein publish General Relativity?",
                "compression_ratio": 0.4,
            },
        )

        content = compress_result.content[0]
        result = json.loads(content.text if isinstance(content, TextContent) else "{}")

        if result.get("error"):
            print(f"Error: {result.get('message', 'Unknown error')}")
        else:
            print(f"Original tokens: {result.get('original_tokens', 'N/A')}")
            print(f"Compressed tokens: {result.get('compressed_tokens', 'N/A')}")
            tokens_saved = result.get("original_tokens", 0) - result.get("compressed_tokens", 0)
            print(f"Tokens saved: {tokens_saved}")
            print(f"Compression ratio: {format_percent(result.get('compression_ratio'))}")

        # Example 2: Matrix of Thought reasoning
        print("\n🧠 Example 2: Matrix of Thought Reasoning")
        print("-" * 40)

        mot_result = await client.call_tool(
            "matrix_of_thought_reasoning",
            {
                "question": "What are the key contributions of Einstein to physics?",
                "context": result.get("compressed_context", long_text[:500]),
                "matrix_rows": 3,
                "matrix_cols": 3,
            },
        )

        mot_content = mot_result.content[0]
        mot_data = json.loads(mot_content.text if isinstance(mot_content, TextContent) else "{}")

        if mot_data.get("error"):
            print(f"Error: {mot_data.get('message', 'Unknown error')}")
            mot_answer = ""
        else:
            mot_answer = mot_data.get("answer", "")
            answer_preview = mot_answer[:200] + "..." if len(mot_answer) > 200 else mot_answer
            print(f"Answer: {answer_preview or 'No answer generated'}")
            print(f"Confidence: {format_percent(mot_data.get('confidence'))}")
            print(f"Reasoning steps: {len(mot_data.get('reasoning_steps', []))}")

        # Example 3: Verify the answer
        print("\n✅ Example 3: Verify Fact Consistency")
        print("-" * 40)

        # Use a fallback answer if MoT didn't produce one
        answer_to_verify = mot_answer or "Einstein contributed to physics with relativity theory."

        verify_result = await client.call_tool(
            "verify_fact_consistency",
            {
                "answer": answer_to_verify,
                "context": long_text,
                "max_claims": 5,
            },
        )

        verify_content = verify_result.content[0]
        verify_text = verify_content.text if isinstance(verify_content, TextContent) else "{}"
        verify_data = json.loads(verify_text)

        if verify_data.get("error"):
            print(f"Error: {verify_data.get('message', 'Unknown error')}")
        else:
            print(f"Verified: {verify_data.get('verified', 'N/A')}")
            print(f"Confidence: {format_percent(verify_data.get('confidence'))}")
            claims_verified = verify_data.get("claims_verified", "N/A")
            claims_total = verify_data.get("claims_total", "N/A")
            print(f"Claims verified: {claims_verified}/{claims_total}")
            print(f"Reason: {verify_data.get('reason', 'N/A')}")

        # Example 4: Get strategy recommendation
        print("\n🎯 Example 4: Strategy Recommendation")
        print("-" * 40)

        strategy_result = await client.call_tool(
            "recommend_reasoning_strategy",
            {
                "problem": (
                    "Find if there is a path connecting node A to node D in a directed graph"
                ),
                "token_budget": 3000,
            },
        )

        strategy_content = strategy_result.content[0]
        strategy_text = strategy_content.text if isinstance(strategy_content, TextContent) else "{}"
        strategy_data = json.loads(strategy_text)

        if strategy_data.get("error"):
            print(f"Error: {strategy_data.get('message', 'Unknown error')}")
        else:
            print(f"Recommended: {strategy_data.get('recommended_strategy', 'N/A')}")
            print(f"Estimated steps: {strategy_data.get('estimated_depth_steps', 'N/A')}")
            print(f"Confidence: {format_percent(strategy_data.get('strategy_confidence'))}")
            print(f"Explanation: {strategy_data.get('explanation', 'N/A')}")

    print("\n" + "=" * 60)
    print("Examples completed successfully! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
