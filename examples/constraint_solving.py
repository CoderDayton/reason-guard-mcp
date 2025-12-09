"""Constraint Solving example for MatrixMind MCP Server.

Demonstrates how to use Long Chain of Thought reasoning for
constraint satisfaction problems that require serial reasoning.

Examples include:
- Game of 24 (make 24 from 4 numbers)
- Logic puzzles (Einstein's riddle style)
- Path finding with constraints
- Scheduling problems

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


@dataclass
class ConstraintProblem:
    """A constraint satisfaction problem example."""

    name: str
    problem: str
    problem_type: str
    expected_steps: int
    hint: str = ""


# Example constraint problems
PROBLEMS = [
    ConstraintProblem(
        name="Game of 24 - Basic",
        problem="""Using the numbers 3, 4, 5, 6, make the number 24.
You can use +, -, *, / and each number exactly once.
Show your work step by step.""",
        problem_type="arithmetic_constraint",
        expected_steps=6,
        hint=(
            "One solution: (6 - 3) * (4 + 5) = 3 * 8 = 24, "
            "but try (5 - 3) * (6 + 4) = 2 * 10... no. Try 6 / (1 - 3/4) = 24"
        ),
    ),
    ConstraintProblem(
        name="Game of 24 - Medium",
        problem="""Using the numbers 1, 5, 5, 5, make the number 24.
You can use +, -, *, / and each number exactly once.
Show your work step by step.""",
        problem_type="arithmetic_constraint",
        expected_steps=8,
        hint="Think about: 5 * 5 = 25, 25 - 1 = 24. Then 24 * (5/5) = 24",
    ),
    ConstraintProblem(
        name="Logic Puzzle - Houses",
        problem="""Three people (Alice, Bob, Carol) live in three houses (Red, Blue, Green).
- Alice does not live in the red house.
- Bob does not live in the blue house.
- Carol lives in the green house.
- The person in the red house is not Carol.

Determine who lives in which house.""",
        problem_type="logic_constraint",
        expected_steps=10,
        hint="Start with what we know for certain, then eliminate",
    ),
    ConstraintProblem(
        name="Path Constraint",
        problem="""Find a path from A to F in this graph. Visit exactly 4 nodes:

Graph edges:
A -> B, A -> C
B -> D, B -> E
C -> D, C -> F
D -> F
E -> F

Find a valid path of exactly 4 nodes from A to F.""",
        problem_type="graph_constraint",
        expected_steps=8,
        hint="Try paths: A->C->D->F (only 4 nodes including start/end)",
    ),
    ConstraintProblem(
        name="Scheduling Constraint",
        problem="""Schedule 4 meetings (A, B, C, D) into 4 time slots (9am, 10am, 11am, 12pm).
Constraints:
- Meeting A must be before meeting B
- Meeting C cannot be at 9am or 12pm
- Meeting D must be immediately after meeting C
- Meeting B cannot be at 10am

Find a valid schedule.""",
        problem_type="scheduling_constraint",
        expected_steps=12,
        hint="C must be at 10am or 11am. If C at 10am, D at 11am...",
    ),
    ConstraintProblem(
        name="Cryptarithmetic",
        problem="""Solve this cryptarithmetic puzzle:
    SEND
  + MORE
  ------
   MONEY

Each letter represents a unique digit (0-9).
Leading digits (S and M) cannot be 0.
Find the mapping of letters to digits.""",
        problem_type="cryptarithmetic",
        expected_steps=15,
        hint="M must be 1 (carry from addition). S must be 8 or 9...",
    ),
]


async def run_constraint_solving() -> None:
    """Run constraint solving examples using the MCP server."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Please install fastmcp: pip install fastmcp")
        sys.exit(1)

    print("=" * 70)
    print("Enhanced Chain-of-Thought MCP - Constraint Solving Examples")
    print("=" * 70)
    print("\nConstraint satisfaction problems benefit from long-chain serial")
    print("reasoning where each step builds on previous deductions.")

    async with Client("src/server.py") as client:
        for i, problem in enumerate(PROBLEMS, 1):
            print(f"\n{'=' * 70}")
            print(f"Problem {i}: {problem.name}")
            print(f"Type: {problem.problem_type}")
            print(f"{'=' * 70}")
            print(f"\n{problem.problem}")
            print(f"\nExpected reasoning steps: ~{problem.expected_steps}")

            # Step 1: Get strategy recommendation
            print("\n--- Step 1: Strategy Recommendation ---")
            strategy_result = await client.call_tool(
                "recommend_reasoning_strategy",
                {
                    "problem": problem.problem,
                    "token_budget": 4000,
                },
            )
            content = strategy_result.content[0]
            strategy_data = json.loads(content.text if isinstance(content, TextContent) else "{}")

            if "error" not in strategy_data:
                print(f"Recommended: {strategy_data.get('recommended_strategy', 'N/A')}")
                print(f"Confidence: {strategy_data.get('strategy_confidence', 0):.1%}")
                print(f"Explanation: {strategy_data.get('explanation', 'N/A')}")

            # Step 2: Use Long Chain of Thought (preferred for constraints)
            print("\n--- Step 2: Long Chain of Thought Reasoning ---")
            lc_result = await client.call_tool(
                "long_chain_of_thought",
                {
                    "problem": problem.problem,
                    "num_steps": problem.expected_steps + 5,  # Extra steps for safety
                    "verify_intermediate": True,
                },
            )
            lc_content = lc_result.content[0]
            lc_data = json.loads(lc_content.text if isinstance(lc_content, TextContent) else "{}")

            if "error" in lc_data:
                print(f"Error: {lc_data['error']}")
                continue

            print(f"\nFinal Answer: {lc_data.get('answer', 'N/A')}")
            print(f"Confidence: {lc_data.get('confidence', 0):.1%}")

            verif = lc_data.get("verification_results", {})
            passed = verif.get("passed", 0)
            total = verif.get("total_verifications", 0)
            print(f"Intermediate verifications: {passed}/{total} passed")

            # Show reasoning chain
            steps = lc_data.get("reasoning_steps", [])
            if steps:
                print(f"\nReasoning chain ({len(steps)} steps):")
                for j, step in enumerate(steps[:5], 1):
                    step_preview = step[:80] + "..." if len(step) > 80 else step
                    print(f"  Step {j}: {step_preview}")
                if len(steps) > 5:
                    print(f"  ... and {len(steps) - 5} more steps")

            # Step 3: Also try Matrix of Thought for comparison
            print("\n--- Step 3: Matrix of Thought (for comparison) ---")
            mot_result = await client.call_tool(
                "matrix_of_thought_reasoning",
                {
                    "question": problem.problem,
                    "context": f"Problem type: {problem.problem_type}. Solve step by step.",
                    "matrix_rows": 3,
                    "matrix_cols": 3,
                },
            )
            mot_content = mot_result.content[0]
            mot_text = mot_content.text if isinstance(mot_content, TextContent) else "{}"
            mot_data = json.loads(mot_text)

            if "error" not in mot_data:
                print(f"MoT Answer: {mot_data.get('answer', 'N/A')[:100]}...")
                print(f"MoT Confidence: {mot_data.get('confidence', 0):.1%}")

            print("\n--- Comparison Summary ---")
            print(f"Long Chain: {lc_data.get('confidence', 0):.1%} confidence")
            if "error" not in mot_data:
                print(f"Matrix of Thought: {mot_data.get('confidence', 0):.1%} confidence")
            print("Expected: Long Chain should perform better on serial constraint problems")

            print("\n" + "-" * 70)

    print("\n" + "=" * 70)
    print("Constraint Solving Examples Completed!")
    print("=" * 70)


async def run_comparison_benchmark() -> None:
    """Run a simple benchmark comparing strategies."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Please install fastmcp: pip install fastmcp")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Strategy Comparison Benchmark")
    print("=" * 70)

    # Simple test problems for quick comparison
    test_problems = [
        (
            "Serial",
            (
                "Calculate: Start with 2, multiply by 3, add 5, "
                "divide by 11, multiply by 4. What is the result?"
            ),
        ),
        ("Multi-path", "What are three different ways to travel from New York to Los Angeles?"),
        ("Constraint", "I have 3 coins totaling 25 cents. What are they?"),
    ]

    async with Client("src/server.py") as client:
        print("\n{:<12} {:<15} {:<15} {:<15}".format("Type", "Long Chain", "MoT", "Recommended"))
        print("-" * 60)

        for problem_type, problem in test_problems:
            # Get recommendation
            rec_result = await client.call_tool(
                "recommend_reasoning_strategy",
                {"problem": problem, "token_budget": 2000},
            )
            rec_content = rec_result.content[0]
            rec_text = rec_content.text if isinstance(rec_content, TextContent) else "{}"
            rec_data = json.loads(rec_text)
            recommended = rec_data.get("recommended_strategy", "?")

            # Try Long Chain
            lc_result = await client.call_tool(
                "long_chain_of_thought",
                {"problem": problem, "num_steps": 5, "verify_intermediate": False},
            )
            lc_content = lc_result.content[0]
            lc_text = lc_content.text if isinstance(lc_content, TextContent) else "{}"
            lc_data = json.loads(lc_text)
            lc_conf = lc_data.get("confidence", 0) if "error" not in lc_data else 0

            # Try MoT
            mot_result = await client.call_tool(
                "matrix_of_thought_reasoning",
                {
                    "question": problem,
                    "context": "Solve this problem.",
                    "matrix_rows": 2,
                    "matrix_cols": 2,
                },
            )
            mot_content = mot_result.content[0]
            mot_text = mot_content.text if isinstance(mot_content, TextContent) else "{}"
            mot_data = json.loads(mot_text)
            mot_conf = mot_data.get("confidence", 0) if "error" not in mot_data else 0

            print(
                "{:<12} {:<15} {:<15} {:<15}".format(
                    problem_type,
                    f"{lc_conf:.1%}",
                    f"{mot_conf:.1%}",
                    recommended,
                )
            )


async def main() -> None:
    """Run all constraint solving demonstrations."""
    await run_constraint_solving()
    await run_comparison_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
