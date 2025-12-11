"""Multi-hop Question Answering example for MatrixMind MCP Server.

Demonstrates Matrix of Thought and verification for complex multi-hop QA
tasks that require connecting multiple pieces of information.

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


def parse_response(result) -> dict:
    """Parse MCP tool result to dict."""
    content = result.content[0]
    return json.loads(content.text if isinstance(content, TextContent) else "{}")


@dataclass
class MultiHopExample:
    """A multi-hop QA example."""

    name: str
    context: str
    question: str
    expected_hops: int
    matrix_thoughts: dict[tuple[int, int], str]  # Pre-defined thoughts for demo
    claims: list[tuple[str, str, str]]  # (claim, status, evidence)
    answer: str


EXAMPLES = [
    MultiHopExample(
        name="Scientific Discovery Chain",
        context="""
        Marie Curie was a physicist and chemist who conducted pioneering
        research on radioactivity. She was born in Warsaw, Poland in 1867.
        Marie Curie discovered two elements: polonium and radium.
        Polonium was named after her native country, Poland.
        She won the Nobel Prize in Physics in 1903, shared with her husband
        Pierre Curie and Henri Becquerel. In 1911, she won a second Nobel
        Prize in Chemistry. She was the first woman to win a Nobel Prize.
        """,
        question="What element did the first female Nobel laureate name after her birth country?",
        expected_hops=3,
        matrix_thoughts={
            (0, 0): "Hop 1: First female Nobel laureate? Marie Curie, 1903.",
            (0, 1): "Confirming: Marie Curie was the first woman to win a Nobel Prize.",
            (1, 0): "Hop 2: Where was Marie Curie born? Warsaw, Poland.",
            (1, 1): "Confirming: Her birth country was Poland.",
            (2, 0): "Hop 3: Element named after Poland? Polonium.",
            (2, 1): "Confirming: Polonium was named after her native country.",
        },
        claims=[
            (
                "Marie Curie was the first female Nobel laureate",
                "supported",
                "first woman to win a Nobel Prize",
            ),
            ("Marie Curie was born in Poland", "supported", "born in Warsaw, Poland"),
            (
                "Polonium was named after Poland",
                "supported",
                "named after her native country, Poland",
            ),
        ],
        answer="Polonium",
    ),
    MultiHopExample(
        name="Historical Figure Chain",
        context="""
        Albert Einstein was born in Ulm, Germany in 1879. He developed
        the theory of special relativity in 1905 while working at the
        Swiss Patent Office in Bern. In 1915, he published general relativity.
        Einstein received the Nobel Prize in Physics in 1921 for the
        photoelectric effect. He emigrated to the United States in 1933.
        Einstein worked at the Institute for Advanced Study in Princeton,
        New Jersey until his death in 1955. Princeton University was
        founded in 1746.
        """,
        question="When was the university founded where Einstein worked after leaving Germany?",
        expected_hops=3,
        matrix_thoughts={
            (0, 0): "Hop 1: When did Einstein leave Germany? Emigrated to US in 1933.",
            (0, 1): "Confirming: Einstein left Germany in 1933 due to Nazi rise.",
            (1, 0): "Hop 2: Where did Einstein work? Institute for Advanced Study, Princeton.",
            (1, 1): "Confirming: Princeton, New Jersey was his workplace.",
            (2, 0): "Hop 3: When was Princeton University founded? 1746.",
            (2, 1): "Confirming: The text states Princeton University was founded in 1746.",
        },
        claims=[
            (
                "Einstein emigrated to the US in 1933",
                "supported",
                "emigrated to the United States in 1933",
            ),
            (
                "Einstein worked at Princeton",
                "supported",
                "Institute for Advanced Study in Princeton",
            ),
            ("Princeton University was founded in 1746", "supported", "founded in 1746"),
        ],
        answer="1746",
    ),
]


async def solve_multi_hop(client, example: MultiHopExample) -> None:
    """Solve a multi-hop question using Matrix of Thought + Verification."""
    print(f"\n{'=' * 60}")
    print(f"Example: {example.name}")
    print(f"{'=' * 60}")
    print(f"\nQuestion: {example.question}")
    print(f"Expected hops: {example.expected_hops}")

    # Step 1: Compress context
    print("\n► Step 1: Compressing context...")
    result = await client.call_tool(
        "compress_prompt",
        {
            "context": example.context,
            "question": example.question,
            "compression_ratio": 0.7,
        },
    )
    data = parse_response(result)
    compressed = data.get("compressed_context", example.context)
    print(f"  Tokens: {data.get('original_tokens')} → {data.get('compressed_tokens')}")

    # Step 2: Matrix of Thought reasoning
    print("\n► Step 2: Matrix of Thought reasoning...")
    rows = example.expected_hops
    cols = 2  # Initial thought + confirmation

    result = await client.call_tool(
        "matrix_start",
        {"question": example.question, "context": compressed, "rows": rows, "cols": cols},
    )
    data = parse_response(result)

    if data.get("error"):
        print(f"  Error: {data['error']}")
        return

    session_id = data["session_id"]
    print(f"  Session: {session_id[:8]}... ({rows}x{cols} matrix)")

    # Fill matrix with hop-by-hop reasoning
    print("\n  Filling matrix cells:")
    for (row, col), thought in example.matrix_thoughts.items():
        if row < rows and col < cols:
            await client.call_tool(
                "matrix_set_cell",
                {"session_id": session_id, "row": row, "col": col, "thought": thought},
            )
            hop_label = f"Hop {row + 1}" if col == 0 else "Confirm"
            print(f"    [{row},{col}] {hop_label}: {thought[:45]}...")

    # Synthesize columns
    print("\n  Synthesizing columns:")
    await client.call_tool(
        "matrix_synthesize",
        {"session_id": session_id, "col": 0, "synthesis": "Initial reasoning chain established"},
    )
    await client.call_tool(
        "matrix_synthesize",
        {"session_id": session_id, "col": 1, "synthesis": "All hops confirmed from context"},
    )
    print("    Col 0: Initial reasoning")
    print("    Col 1: Confirmations")

    # Finalize matrix
    result = await client.call_tool(
        "matrix_finalize",
        {"session_id": session_id, "answer": example.answer},
    )
    data = parse_response(result)
    print(f"\n  Matrix finalized: {data.get('status')}")
    print(f"  Answer: {example.answer}")

    # Step 3: Verify answer
    print("\n► Step 3: Verifying answer...")
    result = await client.call_tool(
        "verify_start",
        {"answer": f"The answer is {example.answer}", "context": example.context},
    )
    data = parse_response(result)
    verify_session = data["session_id"]

    # Add and verify claims
    print("\n  Verifying claims:")
    for claim_text, status, evidence in example.claims:
        # Add claim
        result = await client.call_tool(
            "verify_add_claim",
            {"session_id": verify_session, "claim": claim_text},
        )
        claim_data = parse_response(result)
        claim_id = claim_data["claim_id"]

        # Verify claim
        await client.call_tool(
            "verify_claim",
            {
                "session_id": verify_session,
                "claim_id": claim_id,
                "status": status,
                "evidence": evidence,
            },
        )
        print(f"    ✓ {claim_text[:50]}...")

    # Finalize verification
    result = await client.call_tool("verify_finalize", {"session_id": verify_session})
    data = parse_response(result)
    summary = data.get("summary", {})
    print(f"\n  Verification: {'✓ PASSED' if data.get('verified') else '✗ FAILED'}")
    supported = summary.get("supported", 0)
    contradicted = summary.get("contradicted", 0)
    print(f"  Claims: {supported} supported, {contradicted} contradicted")


async def demo_strategy_selection(client) -> None:
    """Show strategy recommendations for different question types."""
    print("\n" + "=" * 60)
    print("Strategy Recommendations for Multi-Hop QA")
    print("=" * 60)

    questions = [
        ("Single hop", "What year was Einstein born?"),
        ("Multi-hop (2)", "Where did the person who invented relativity work?"),
        ("Multi-hop (3)", "When was the university founded where the Nobel laureate worked?"),
        ("Comparison", "Compare Einstein and Curie's Nobel achievements"),
    ]

    print("\n{:<15} {:<20} {:<15}".format("Type", "Strategy", "Confidence"))
    print("─" * 50)

    for q_type, question in questions:
        result = await client.call_tool(
            "recommend_reasoning_strategy",
            {"problem": question, "token_budget": 3000},
        )
        data = parse_response(result)
        strategy = data.get("recommended_strategy", "error")
        confidence = data.get("strategy_confidence", 0)
        print(f"{q_type:<15} {strategy:<20} {confidence:.0%}")


async def main() -> None:
    """Run multi-hop QA demonstrations."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Please install fastmcp: pip install fastmcp")
        sys.exit(1)

    print("=" * 60)
    print("MatrixMind MCP - Multi-Hop Question Answering")
    print("Matrix of Thought + Verification Pipeline")
    print("=" * 60)

    async with Client("src/server.py") as client:
        # Run each example
        for example in EXAMPLES:
            await solve_multi_hop(client, example)

        # Show strategy recommendations
        await demo_strategy_selection(client)

    print("\n" + "=" * 60)
    print("Multi-Hop QA Examples Completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
