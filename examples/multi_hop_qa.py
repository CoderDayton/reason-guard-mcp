"""Multi-hop Question Answering example for MatrixMind MCP Server.

Demonstrates how to use the Matrix of Thought and Long Chain reasoning
tools for complex multi-hop QA tasks that require connecting multiple
pieces of information.

Example questions:
- "Who wrote the paper that introduced the concept used in X?"
- "What is the capital of the country where Y was born?"
- "When did the founder of company Z receive their degree?"

Run after starting the server:
    fastmcp run src/server.py

Then run this script:
    python examples/multi_hop_qa.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from mcp.types import TextContent

load_dotenv()


@dataclass
class MultiHopExample:
    """A multi-hop QA example with context and question."""

    name: str
    context: str
    question: str
    expected_hops: int
    hint: str = ""


# Example multi-hop questions with their contexts
EXAMPLES = [
    MultiHopExample(
        name="Scientific Discovery Chain",
        context="""
        Marie Curie was a physicist and chemist who conducted pioneering
        research on radioactivity. She was born in Warsaw, Poland in 1867.
        Marie Curie discovered two elements: polonium and radium.
        Polonium was named after her native country, Poland. She won the
        Nobel Prize in Physics in 1903, shared with her husband Pierre Curie
        and Henri Becquerel. In 1911, she won a second Nobel Prize,
        this time in Chemistry, for her discovery of radium and polonium. She was the first woman to
        win a Nobel Prize and the first person to win Nobel Prizes in two different sciences.
        The Pierre and Marie Curie University in Paris was named in honor of both scientists.
        Marie Curie died in 1934 from aplastic anemia, likely caused by
        her long-term exposure to radiation.
        """,
        question="What element did the first female Nobel laureate name after her birth country?",
        expected_hops=3,
        hint=(
            "Requires: (1) identify first female Nobel laureate, "
            "(2) find her birth country, (3) find element named after it"
        ),
    ),
    MultiHopExample(
        name="Historical Figure Chain",
        context="""
        Albert Einstein was born in Ulm, Germany in 1879. He developed the
        theory of special relativity in 1905 while working at the Swiss
        Patent Office in Bern. His famous equation E=mc² emerged
        from this work. In 1915, he published his theory of general relativity. Einstein received
        the Nobel Prize in Physics in 1921, but not for relativity - it was for his explanation of
        the photoelectric effect. He emigrated to the United States in 1933 due to the rise of
        Nazi Germany. Einstein worked at the Institute for Advanced Study in Princeton, New Jersey
        until his death in 1955. The element einsteinium (Es, atomic number 99) was named after him.
        Princeton University, where the Institute is located, was founded in 1746.
        """,
        question=(
            "In what year was the university founded where Einstein worked after leaving Germany?"
        ),
        expected_hops=3,
        hint=(
            "Requires: (1) where Einstein went after Germany, "
            "(2) which university, (3) when founded"
        ),
    ),
    MultiHopExample(
        name="Technology Chain",
        context="""
        Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN in Switzerland.
        He wrote the first web browser called WorldWideWeb (later renamed Nexus). CERN, the
        European Organization for Nuclear Research, was established in 1954. The organization
        is located near Geneva on the Franco-Swiss border. Berners-Lee also created HTML
        (HyperText Markup Language) and HTTP (HyperText Transfer Protocol). He founded the
        World Wide Web Consortium (W3C) in 1994 at MIT. The W3C develops web standards and
        guidelines. Berners-Lee was knighted by Queen Elizabeth II in 2004 for his services
        to the development of the Internet. He currently holds the 3Com Founders Chair at MIT.
        """,
        question=(
            "What organization did the inventor of the World Wide Web establish at MIT, and when?"
        ),
        expected_hops=2,
        hint="Requires: (1) identify WWW inventor, (2) find organization founded at MIT",
    ),
    MultiHopExample(
        name="Geographic Chain",
        context="""
        The Amazon River is the largest river by discharge volume of water in the world.
        It flows through several countries including Brazil, Peru, and Colombia. The river
        is approximately 6,400 kilometers long. The Amazon rainforest, which the river
        flows through, is the world's largest tropical rainforest. Brazil is the largest
        country in South America and the fifth largest in the world. The capital of Brazil
        is Brasília, which was founded in 1960. Before Brasília, Rio de Janeiro was the
        capital from 1763 to 1960. São Paulo is the largest city in Brazil and South America.
        The Amazon basin covers about 40% of South America's landmass.
        """,
        question=(
            "What was the capital before the current capital of the country "
            "the Amazon mostly flows through?"
        ),
        expected_hops=3,
        hint="Requires: (1) main country of Amazon, (2) current capital, (3) previous capital",
    ),
]


async def run_multi_hop_qa() -> None:
    """Run multi-hop QA examples using the MCP server."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Please install fastmcp: pip install fastmcp")
        sys.exit(1)

    print("=" * 70)
    print("Enhanced Chain-of-Thought MCP - Multi-Hop QA Examples")
    print("=" * 70)

    async with Client("src/server.py") as client:
        for i, example in enumerate(EXAMPLES, 1):
            print(f"\n{'=' * 70}")
            print(f"Example {i}: {example.name}")
            print(f"{'=' * 70}")
            print(f"\nQuestion: {example.question}")
            print(f"Expected reasoning hops: {example.expected_hops}")
            print(f"Hint: {example.hint}")

            # Step 1: Compress context to focus on relevant information
            print("\n--- Step 1: Compress Context ---")
            compress_result = await client.call_tool(
                "compress_prompt",
                {
                    "context": example.context,
                    "question": example.question,
                    "compression_ratio": 0.6,  # Keep 60% for multi-hop
                },
            )
            compress_content = compress_result.content[0]
            compress_text = (
                compress_content.text if isinstance(compress_content, TextContent) else "{}"
            )
            compress_data = json.loads(compress_text)

            if "error" in compress_data:
                print(f"Compression error: {compress_data['error']}")
                continue

            print(
                f"Compression: {compress_data.get('original_tokens', '?')} -> "
                f"{compress_data.get('compressed_tokens', '?')} tokens"
            )
            print(f"Tokens saved: {compress_data.get('tokens_saved', '?')}")

            compressed_context = compress_data.get("compressed_context", example.context)

            # Step 2: Use Matrix of Thought for multi-perspective reasoning
            print("\n--- Step 2: Matrix of Thought Reasoning ---")
            mot_result = await client.call_tool(
                "matrix_of_thought_reasoning",
                {
                    "question": example.question,
                    "context": compressed_context,
                    "matrix_rows": min(example.expected_hops, 4),  # Match expected hops
                    "matrix_cols": 4,  # 4 iterations for refinement
                },
            )
            mot_content = mot_result.content[0]
            mot_data = json.loads(
                mot_content.text if isinstance(mot_content, TextContent) else "{}"
            )

            if "error" in mot_data:
                print(f"MoT error: {mot_data['error']}")
                continue

            print(f"Answer: {mot_data.get('answer', 'N/A')}")
            print(f"Confidence: {mot_data.get('confidence', 0):.1%}")
            print(f"Reasoning steps: {mot_data.get('num_reasoning_steps', 0)}")

            # Show first few reasoning steps
            steps = mot_data.get("reasoning_steps", [])
            if steps:
                print("\nKey reasoning steps:")
                for j, step in enumerate(steps[:3], 1):
                    print(f"  {j}. {step[:100]}...")

            # Step 3: Verify the answer against context
            print("\n--- Step 3: Verify Answer ---")
            verify_result = await client.call_tool(
                "verify_fact_consistency",
                {
                    "answer": mot_data.get("answer", ""),
                    "context": example.context,  # Use full context for verification
                    "max_claims": 5,
                },
            )
            verify_content = verify_result.content[0]
            verify_text = verify_content.text if isinstance(verify_content, TextContent) else "{}"
            verify_data = json.loads(verify_text)

            if "error" in verify_data:
                print(f"Verification error: {verify_data['error']}")
                continue

            print(f"Verified: {verify_data.get('verified', False)}")
            claims_verified = verify_data.get("claims_verified", 0)
            claims_total = verify_data.get("claims_total", 0)
            print(f"Claims verified: {claims_verified}/{claims_total}")
            print(f"Recommendation: {verify_data.get('recommendation', 'N/A')}")

            # Step 4: For complex questions, also try Long Chain reasoning
            if example.expected_hops >= 3:
                print("\n--- Step 4: Long Chain Reasoning (for comparison) ---")
                long_chain_result = await client.call_tool(
                    "long_chain_of_thought",
                    {
                        "problem": f"Context: {compressed_context}\n\nQuestion: {example.question}",
                        "num_steps": example.expected_hops * 3,  # More steps for complex chains
                        "verify_intermediate": True,
                    },
                )
                lc_content = long_chain_result.content[0]
                lc_data = json.loads(
                    lc_content.text if isinstance(lc_content, TextContent) else "{}"
                )

                if "error" not in lc_data:
                    print(f"Long Chain Answer: {lc_data.get('answer', 'N/A')}")
                    print(f"Confidence: {lc_data.get('confidence', 0):.1%}")
                    verif = lc_data.get("verification_results", {})
                    passed = verif.get("passed", 0)
                    total = verif.get("total_verifications", 0)
                    print(f"Intermediate verifications passed: {passed}/{total}")

            print("\n" + "-" * 70)

    print("\n" + "=" * 70)
    print("Multi-Hop QA Examples Completed!")
    print("=" * 70)


async def run_strategy_recommendation() -> None:
    """Demonstrate strategy recommendation for different problem types."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Please install fastmcp: pip install fastmcp")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Strategy Recommendation Examples")
    print("=" * 70)

    problems = [
        ("Multi-hop: Who is the author of the paper that introduced transformers?", 4000),
        ("Serial: Calculate 2^10 step by step", 2000),
        ("Creative: Generate multiple ideas for a sustainable energy solution", 5000),
        ("Constraint: Find a path from A to D in a directed graph", 3000),
    ]

    async with Client("src/server.py") as client:
        for problem, budget in problems:
            print(f"\nProblem: {problem}")
            print(f"Token budget: {budget}")

            result = await client.call_tool(
                "recommend_reasoning_strategy",
                {
                    "problem": problem,
                    "token_budget": budget,
                },
            )
            result_content = result.content[0]
            data = json.loads(
                result_content.text if isinstance(result_content, TextContent) else "{}"
            )

            if "error" in data:
                print(f"Error: {data['error']}")
                continue

            print(f"  Recommended: {data.get('recommended_strategy', 'N/A')}")
            print(f"  Estimated steps: {data.get('estimated_depth_steps', 'N/A')}")
            print(f"  Confidence: {data.get('strategy_confidence', 0):.1%}")
            print(f"  Explanation: {data.get('explanation', 'N/A')}")


async def main() -> None:
    """Run all multi-hop QA demonstrations."""
    await run_multi_hop_qa()
    await run_strategy_recommendation()


if __name__ == "__main__":
    asyncio.run(main())
