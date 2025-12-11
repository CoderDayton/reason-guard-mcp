"""Constraint Solving example for MatrixMind MCP Server.

Demonstrates Long Chain of Thought reasoning for constraint satisfaction
problems that require serial reasoning steps.

Examples include:
- Game of 24 (make 24 from 4 numbers)
- Logic puzzles
- Path finding with constraints

Run after starting the server:
    fastmcp run src/server.py

Then run this script:
    python examples/constraint_solving.py
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
class ConstraintProblem:
    """A constraint satisfaction problem example."""

    name: str
    problem: str
    problem_type: str
    expected_steps: int
    reasoning_steps: list[str]  # Pre-defined steps (LLM would generate these)
    answer: str


# Example constraint problems with their reasoning chains
PROBLEMS = [
    ConstraintProblem(
        name="Game of 24 - Basic",
        problem="Using 3, 4, 5, 6, make 24. Use +, -, *, / and each number once.",
        problem_type="arithmetic_constraint",
        expected_steps=5,
        reasoning_steps=[
            "I need to make 24 using 3, 4, 5, 6 with basic operations.",
            "Let me try multiplication first: 4 × 6 = 24. Still need 3 and 5.",
            "If 3 and 5 make 1: (5-3-1... no). Try: (5-3) = 2, then 2 × ... no.",
            "Different approach: 6 / (1 - 3/4) = 24. Uses 6, 3, 4. Need 5.",
            "Try: (6 - 3) × (4 + 5 - 5) = 12. No. 4 × (6 - (5-3)) = 16. No.",
        ],
        answer="Requires creative grouping. One path: explore (a+b)*(c-d) forms.",
    ),
    ConstraintProblem(
        name="Logic Puzzle - Houses",
        problem="""Three people (Alice, Bob, Carol) live in three houses (Red, Blue, Green).
- Alice does not live in the red house.
- Bob does not live in the blue house.
- Carol lives in the green house.
Determine who lives in which house.""",
        problem_type="logic_constraint",
        expected_steps=6,
        reasoning_steps=[
            "Given: Alice ≠ Red, Bob ≠ Blue, Carol = Green. Three houses: Red, Blue, Green.",
            "Since Carol lives in Green, that leaves Red and Blue for Alice and Bob.",
            "Alice cannot live in Red (given), so Alice must live in Blue.",
            "That leaves only Red for Bob.",
            "Verify: Bob in Red (allowed, constraint was Bob ≠ Blue). Alice=Blue, Carol=Green.",
            "All constraints satisfied: Alice-Blue, Bob-Red, Carol-Green.",
        ],
        answer="Alice lives in Blue, Bob lives in Red, Carol lives in Green.",
    ),
    ConstraintProblem(
        name="Path Constraint",
        problem="""Find a path from A to F visiting exactly 4 nodes.
Graph: A→B, A→C, B→D, B→E, C→D, C→F, D→F, E→F""",
        problem_type="graph_constraint",
        expected_steps=5,
        reasoning_steps=[
            "Need path A to F with exactly 4 nodes (including A and F).",
            "From A, I can go to B or C.",
            "Path A→C→F is only 3 nodes. Need one more.",
            "Path A→C→D→F is exactly 4 nodes: A, C, D, F. Check edges: A→C ✓, C→D ✓, D→F ✓",
            "Valid path found: A → C → D → F (4 nodes).",
        ],
        answer="A → C → D → F",
    ),
]


async def solve_with_chain(client, problem: ConstraintProblem) -> dict:
    """Solve a constraint problem using Long Chain of Thought."""
    print(f"\n{'─' * 60}")
    print(f"Problem: {problem.name}")
    print(f"Type: {problem.problem_type}")
    print(f"{'─' * 60}")
    print(f"\n{problem.problem}\n")

    # Step 1: Get strategy recommendation
    print("► Getting strategy recommendation...")
    result = await client.call_tool(
        "recommend_reasoning_strategy",
        {"problem": problem.problem, "token_budget": 3000},
    )
    data = parse_response(result)
    if "error" not in data:
        strategy = data.get("recommended_strategy")
        confidence = data.get("strategy_confidence", 0)
        print(f"  Recommended: {strategy} ({confidence:.0%} confidence)")

    # Step 2: Start chain session
    print("\n► Starting chain reasoning session...")
    result = await client.call_tool(
        "chain_start",
        {"problem": problem.problem, "expected_steps": problem.expected_steps},
    )
    data = parse_response(result)

    if data.get("error"):
        print(f"  Error: {data['error']}")
        return {"error": data["error"]}

    session_id = data["session_id"]
    print(f"  Session: {session_id[:8]}...")

    # Step 3: Add reasoning steps
    print("\n► Adding reasoning steps...")
    for i, thought in enumerate(problem.reasoning_steps, 1):
        result = await client.call_tool(
            "chain_add_step",
            {"session_id": session_id, "thought": thought},
        )
        step_data = parse_response(result)

        # Show step with any feedback
        print(f"  Step {i}: {thought[:60]}...")
        if step_data.get("issues"):
            for issue in step_data["issues"]:
                print(f"    ⚠ {issue}")

    # Step 4: Check progress
    result = await client.call_tool("chain_get", {"session_id": session_id})
    data = parse_response(result)
    print(f"\n► Progress: {data.get('current_step')}/{data.get('expected_steps')} steps")

    # Step 5: Finalize
    print("\n► Finalizing with answer...")
    result = await client.call_tool(
        "chain_finalize",
        {"session_id": session_id, "answer": problem.answer},
    )
    data = parse_response(result)

    print(f"  Status: {data.get('status')}")
    print(f"  Answer: {problem.answer[:80]}...")

    return data


async def solve_with_matrix(client, problem: ConstraintProblem) -> dict:
    """Solve using Matrix of Thought for comparison."""
    print("\n► Matrix of Thought (comparison)...")

    # Start matrix
    result = await client.call_tool(
        "matrix_start",
        {"question": problem.problem, "rows": 2, "cols": 2},
    )
    data = parse_response(result)

    if data.get("error"):
        print(f"  Error: {data['error']}")
        return {"error": data["error"]}

    session_id = data["session_id"]

    # Fill with condensed reasoning
    thoughts = [
        (0, 0, "Identify constraints and variables"),
        (0, 1, "Apply elimination based on constraints"),
        (1, 0, "Check remaining possibilities"),
        (1, 1, "Verify solution satisfies all constraints"),
    ]

    for row, col, thought in thoughts:
        await client.call_tool(
            "matrix_set_cell",
            {"session_id": session_id, "row": row, "col": col, "thought": thought},
        )

    # Synthesize
    await client.call_tool(
        "matrix_synthesize",
        {"session_id": session_id, "col": 0, "synthesis": "Constraint analysis complete"},
    )
    await client.call_tool(
        "matrix_synthesize",
        {"session_id": session_id, "col": 1, "synthesis": "Solution verified"},
    )

    # Finalize
    result = await client.call_tool(
        "matrix_finalize",
        {"session_id": session_id, "answer": problem.answer},
    )
    data = parse_response(result)

    print(f"  Matrix status: {data.get('status')}")
    return data


async def run_comparison_benchmark(client) -> None:
    """Compare Chain vs Matrix on different problem types."""
    print("\n" + "=" * 60)
    print("Strategy Comparison")
    print("=" * 60)

    test_cases = [
        ("Serial calculation", "Calculate: 2 × 3 + 4 × 5 - 6 ÷ 2"),
        ("Multi-path", "List 3 ways to get from NYC to LA"),
        ("Constraint", "Arrange A, B, C where A < B and B < C"),
    ]

    print("\n{:<20} {:<20}".format("Problem Type", "Recommended Strategy"))
    print("─" * 40)

    for problem_type, problem in test_cases:
        result = await client.call_tool(
            "recommend_reasoning_strategy",
            {"problem": problem, "token_budget": 2000},
        )
        data = parse_response(result)
        strategy = data.get("recommended_strategy", "error")
        print(f"{problem_type:<20} {strategy:<20}")


async def main() -> None:
    """Run constraint solving demonstrations."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Please install fastmcp: pip install fastmcp")
        sys.exit(1)

    print("=" * 60)
    print("MatrixMind MCP - Constraint Solving Examples")
    print("Long Chain of Thought for Serial Reasoning")
    print("=" * 60)

    async with Client("src/server.py") as client:
        # Solve each problem
        for problem in PROBLEMS:
            await solve_with_chain(client, problem)
            await solve_with_matrix(client, problem)

        # Run comparison
        await run_comparison_benchmark(client)

    print("\n" + "=" * 60)
    print("Constraint Solving Examples Completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
