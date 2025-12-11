"""Basic usage examples for MatrixMind MCP Server.

Demonstrates the multi-call state manager workflow for all tools.

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


def parse_response(result) -> dict:
    """Parse MCP tool result to dict."""
    content = result.content[0]
    return json.loads(content.text if isinstance(content, TextContent) else "{}")


def format_percent(value: float | str | None, default: str = "N/A") -> str:
    """Format a value as percentage."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    try:
        return f"{value:.1%}"
    except (TypeError, ValueError):
        return default


async def demo_compression(client) -> str:
    """Demo 1: Compress a long document."""
    print("\n" + "=" * 60)
    print("Demo 1: Context Compression")
    print("=" * 60)

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
    )

    result = await client.call_tool(
        "compress_prompt",
        {
            "context": long_text,
            "question": "When did Einstein publish General Relativity?",
            "compression_ratio": 0.4,
        },
    )
    data = parse_response(result)

    if data.get("error"):
        print(f"Error: {data.get('message')}")
        return long_text[:500]

    print(f"Original tokens: {data.get('original_tokens', 'N/A')}")
    print(f"Compressed tokens: {data.get('compressed_tokens', 'N/A')}")
    print(f"Tokens saved: {data.get('tokens_saved', 'N/A')}")
    print(f"Compression ratio: {format_percent(data.get('compression_ratio'))}")

    return data.get("compressed_context", long_text[:500])


async def demo_chain_reasoning(client) -> str:
    """Demo 2: Long Chain of Thought reasoning (multi-call workflow)."""
    print("\n" + "=" * 60)
    print("Demo 2: Long Chain of Thought (State Manager)")
    print("=" * 60)

    problem = "What is 15% of 80, then add 25% of that result?"

    # Step 1: Start the chain
    print("\n[1] Starting chain session...")
    result = await client.call_tool("chain_start", {"problem": problem, "expected_steps": 4})
    data = parse_response(result)

    if data.get("error"):
        print(f"Error: {data['error']}")
        return ""

    session_id = data["session_id"]
    print(f"    Session ID: {session_id}")
    print(f"    Status: {data['status']}")

    # Step 2: Add reasoning steps (LLM would generate these)
    thoughts = [
        "First, I need to calculate 15% of 80. 15% = 0.15, so 0.15 × 80 = 12.",
        "The result of the first calculation is 12.",
        "Now I need to calculate 25% of 12. 25% = 0.25, so 0.25 × 12 = 3.",
        "Finally, I add this to get the answer: The result is 3.",
    ]

    print("\n[2] Adding reasoning steps...")
    for i, thought in enumerate(thoughts, 1):
        result = await client.call_tool(
            "chain_add_step", {"session_id": session_id, "thought": thought}
        )
        data = parse_response(result)
        print(f"    Step {i}: {thought[:50]}...")

    # Step 3: Check current state
    print("\n[3] Checking session state...")
    result = await client.call_tool("chain_get", {"session_id": session_id})
    data = parse_response(result)
    print(f"    Current step: {data.get('current_step')}/{data.get('expected_steps')}")
    print(f"    Status: {data.get('status')}")

    # Step 4: Finalize with answer
    print("\n[4] Finalizing chain...")
    result = await client.call_tool(
        "chain_finalize",
        {"session_id": session_id, "answer": "The final result is 3."},
    )
    data = parse_response(result)
    print(f"    Status: {data['status']}")
    print(f"    Total steps: {len(data.get('chain', {}).get('steps', []))}")

    return "The final result is 3."


async def demo_matrix_reasoning(client, context: str) -> str:
    """Demo 3: Matrix of Thought reasoning (multi-call workflow)."""
    print("\n" + "=" * 60)
    print("Demo 3: Matrix of Thought (State Manager)")
    print("=" * 60)

    question = "What are Einstein's key contributions to physics?"

    # Step 1: Start the matrix
    print("\n[1] Starting matrix session (2x2)...")
    result = await client.call_tool(
        "matrix_start",
        {"question": question, "context": context, "rows": 2, "cols": 2},
    )
    data = parse_response(result)

    if data.get("error"):
        print(f"Error: {data['error']}")
        return ""

    session_id = data["session_id"]
    print(f"    Session ID: {session_id}")
    print(f"    Matrix: {data.get('matrix_dimensions')}")

    # Step 2: Fill matrix cells with different perspectives
    # Row 0: Theoretical contributions, Row 1: Practical impact
    # Col 0: Initial thoughts, Col 1: Refined analysis
    cell_thoughts = {
        (0, 0): "Special Relativity (1905): Unified space and time, E=mc²",
        (0, 1): "General Relativity (1915): Gravity as spacetime curvature",
        (1, 0): "Photoelectric effect: Foundation of quantum mechanics",
        (1, 1): "Modern GPS, nuclear energy, and cosmology all stem from his work",
    }

    print("\n[2] Filling matrix cells...")
    for (row, col), thought in cell_thoughts.items():
        result = await client.call_tool(
            "matrix_set_cell",
            {"session_id": session_id, "row": row, "col": col, "thought": thought},
        )
        print(f"    [{row},{col}]: {thought[:40]}...")

    # Step 3: Synthesize columns
    print("\n[3] Synthesizing columns...")
    syntheses = [
        "Einstein's theoretical work revolutionized our understanding of space, time, and energy.",
        "His contributions have practical applications in modern technology and science.",
    ]
    for col, synthesis in enumerate(syntheses):
        result = await client.call_tool(
            "matrix_synthesize", {"session_id": session_id, "col": col, "synthesis": synthesis}
        )
        print(f"    Col {col}: {synthesis[:50]}...")

    # Step 4: Check state
    print("\n[4] Checking matrix state...")
    result = await client.call_tool("matrix_get", {"session_id": session_id})
    data = parse_response(result)
    print(f"    Cells filled: {data.get('cells_filled')}/{data.get('total_cells')}")

    # Step 5: Finalize
    print("\n[5] Finalizing matrix...")
    final_answer = (
        "Einstein's key contributions include Special Relativity (E=mc²), "
        "General Relativity (gravity as spacetime curvature), and the photoelectric effect. "
        "These form the foundation of modern physics."
    )
    result = await client.call_tool(
        "matrix_finalize", {"session_id": session_id, "answer": final_answer}
    )
    data = parse_response(result)
    print(f"    Status: {data['status']}")

    return final_answer


async def demo_verification(client, answer: str, context: str) -> None:
    """Demo 4: Fact verification (multi-call workflow)."""
    print("\n" + "=" * 60)
    print("Demo 4: Fact Verification (State Manager)")
    print("=" * 60)

    # Step 1: Start verification session
    print("\n[1] Starting verification session...")
    result = await client.call_tool("verify_start", {"answer": answer, "context": context})
    data = parse_response(result)

    if data.get("error"):
        print(f"Error: {data['error']}")
        return

    session_id = data["session_id"]
    print(f"    Session ID: {session_id}")

    # Step 2: Add claims to verify
    claims = [
        "Einstein developed Special Relativity",
        "E=mc² comes from Special Relativity",
        "Einstein developed General Relativity",
    ]

    print("\n[2] Adding claims...")
    claim_ids = []
    for claim in claims:
        result = await client.call_tool(
            "verify_add_claim", {"session_id": session_id, "claim": claim}
        )
        data = parse_response(result)
        claim_ids.append(data["claim_id"])
        print(f"    Claim {data['claim_id']}: {claim}")

    # Step 3: Verify each claim (LLM would determine status)
    print("\n[3] Verifying claims...")
    verifications = [
        ("supported", "Context confirms Einstein developed relativity theories"),
        ("supported", "Context states E=mc² comes from special relativity"),
        ("supported", "Context mentions General Relativity published in 1915"),
    ]

    for claim_id, (status, evidence) in zip(claim_ids, verifications, strict=False):
        result = await client.call_tool(
            "verify_claim",
            {
                "session_id": session_id,
                "claim_id": claim_id,
                "status": status,
                "evidence": evidence,
            },
        )
        data = parse_response(result)
        print(f"    Claim {claim_id}: {status}")

    # Step 4: Finalize verification
    print("\n[4] Finalizing verification...")
    result = await client.call_tool("verify_finalize", {"session_id": session_id})
    data = parse_response(result)

    print(f"    Verified: {data.get('verified')}")
    summary = data.get("summary", {})
    supported = summary.get("supported", 0)
    contradicted = summary.get("contradicted", 0)
    print(f"    Claims: {supported} supported, {contradicted} contradicted")


async def demo_strategy_recommendation(client) -> None:
    """Demo 5: Get strategy recommendation."""
    print("\n" + "=" * 60)
    print("Demo 5: Strategy Recommendation")
    print("=" * 60)

    problems = [
        ("Calculate 2^10 step by step", 2000),
        ("Compare three approaches to solving climate change", 4000),
    ]

    for problem, budget in problems:
        print(f"\n    Problem: {problem[:50]}...")
        result = await client.call_tool(
            "recommend_reasoning_strategy",
            {"problem": problem, "token_budget": budget},
        )
        data = parse_response(result)

        if data.get("error"):
            print(f"    Error: {data['error']}")
            continue

        print(f"    Recommended: {data.get('recommended_strategy')}")
        print(f"    Explanation: {data.get('explanation', 'N/A')[:60]}...")


async def main() -> None:
    """Run all basic usage demonstrations."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Please install fastmcp: pip install fastmcp")
        return

    print("=" * 60)
    print("MatrixMind MCP - Basic Usage Examples")
    print("State Manager Architecture Demo")
    print("=" * 60)

    async with Client("src/server.py") as client:
        # Demo 1: Compression (single call)
        compressed = await demo_compression(client)

        # Demo 2: Long Chain reasoning (multi-call)
        await demo_chain_reasoning(client)

        # Demo 3: Matrix of Thought (multi-call)
        answer = await demo_matrix_reasoning(client, compressed)

        # Demo 4: Verification (multi-call)
        await demo_verification(client, answer, compressed)

        # Demo 5: Strategy recommendation (single call)
        await demo_strategy_recommendation(client)

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
