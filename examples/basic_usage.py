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


async def main() -> None:
    """Demonstrate basic tool usage."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Please install fastmcp: pip install fastmcp")
        return

    print("=" * 60)
    print("Enhanced Chain-of-Thought MCP - Basic Usage Examples")
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
        print(f"Original tokens: {result.get('original_tokens', 'N/A')}")
        print(f"Compressed tokens: {result.get('compressed_tokens', 'N/A')}")
        print(f"Tokens saved: {result.get('tokens_saved', 'N/A')}")
        print(f"Compression ratio: {result.get('compression_ratio', 'N/A'):.1%}")

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
        print(f"Answer: {mot_data.get('answer', 'N/A')[:200]}...")
        print(f"Confidence: {mot_data.get('confidence', 'N/A'):.1%}")
        print(f"Reasoning steps: {mot_data.get('num_reasoning_steps', 'N/A')}")

        # Example 3: Verify the answer
        print("\n✅ Example 3: Verify Fact Consistency")
        print("-" * 40)

        verify_result = await client.call_tool(
            "verify_fact_consistency",
            {
                "answer": mot_data.get("answer", "Einstein contributed to physics."),
                "context": long_text,
                "max_claims": 5,
            },
        )

        verify_content = verify_result.content[0]
        verify_text = verify_content.text if isinstance(verify_content, TextContent) else "{}"
        verify_data = json.loads(verify_text)
        print(f"Verified: {verify_data.get('verified', 'N/A')}")
        print(f"Confidence: {verify_data.get('confidence', 'N/A'):.1%}")
        claims_verified = verify_data.get("claims_verified", "N/A")
        claims_total = verify_data.get("claims_total", "N/A")
        print(f"Claims verified: {claims_verified}/{claims_total}")
        print(f"Recommendation: {verify_data.get('recommendation', 'N/A')}")

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
        print(f"Recommended: {strategy_data.get('recommended_strategy', 'N/A')}")
        print(f"Estimated steps: {strategy_data.get('estimated_depth_steps', 'N/A')}")
        print(f"Confidence: {strategy_data.get('strategy_confidence', 'N/A'):.1%}")
        print(f"Explanation: {strategy_data.get('explanation', 'N/A')}")

    print("\n" + "=" * 60)
    print("Examples completed successfully! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
