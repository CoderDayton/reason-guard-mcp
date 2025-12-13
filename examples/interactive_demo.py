#!/usr/bin/env python3
"""Interactive demo for MatrixMind MCP `think` tool.

Walks through a complete reasoning workflow with live output,
demonstrating all 12 actions of the unified `think` function.

Run:
    uv run python examples/interactive_demo.py
    uv run python examples/interactive_demo.py --mode matrix
    uv run python examples/interactive_demo.py --mode verify
    uv run python examples/interactive_demo.py --fast  # Skip pauses
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

DEMO_PROBLEMS = {
    "chain": {
        "problem": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
        "steps": [
            "Let me read the problem carefully: '17 sheep, all but 9 run away'",
            "The phrase 'all but 9' means 9 remain, not 17-9=8",
            "This is a classic trick question - the answer is 9 sheep",
        ],
        "answer": "9 sheep remain (all but 9 ran away means 9 stayed)",
    },
    "matrix": {
        "problem": "Should our startup use PostgreSQL or MongoDB for our new product?",
        "perspectives": ["Technical", "Business"],
        "criteria": ["Pros", "Cons"],
        "cells": [
            ("Technical", "Pros", "PostgreSQL: ACID compliance, complex queries, mature ecosystem"),
            ("Technical", "Cons", "PostgreSQL: Rigid schema, vertical scaling limits"),
            ("Business", "Pros", "PostgreSQL: Free, large talent pool, enterprise adoption"),
            ("Business", "Cons", "PostgreSQL: May need schema migrations as product evolves"),
        ],
        "synthesis": [
            "Pros: Strong technical foundation with business viability",
            "Cons: Schema rigidity is manageable with good practices",
        ],
        "answer": "PostgreSQL - better for data integrity and complex queries typical in startups",
    },
    "verify": {
        "problem": "The Great Wall of China is visible from space.",
        "context": """NASA has confirmed that the Great Wall of China is NOT visible from low
Earth orbit with the naked eye. Astronauts have reported they cannot see the wall
without aid. The wall is only about 15-30 feet wide, which is too narrow to be
seen from orbit. The myth likely originated from speculation before space travel.""",
        "claims": [
            ("Great Wall visible from space", "contradicted", "NASA confirms not visible"),
            ("Wall is very long", "supported", "Implied by being a famous landmark"),
        ],
        "answer": "The claim is FALSE - NASA confirms the wall is not visible from space",
    },
}

# =============================================================================
# Display Utilities
# =============================================================================


def clear_line() -> None:
    """Clear current line."""
    print("\r" + " " * 80 + "\r", end="", flush=True)


def print_header(text: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step: int, action: str, desc: str) -> None:
    """Print a step indicator."""
    print(f'\n[Step {step}] think(action="{action}")')
    print(f"         {desc}")


def print_response(data: dict[str, Any], indent: int = 2) -> None:
    """Print formatted response."""
    print(" " * indent + "→ Response:")
    for key, value in data.items():
        if isinstance(value, dict):
            print(" " * (indent + 4) + f"{key}:")
            for k, v in value.items():
                print(" " * (indent + 8) + f"{k}: {v}")
        elif isinstance(value, list) and len(value) > 3:
            print(" " * (indent + 4) + f"{key}: [{len(value)} items]")
        else:
            val_str = str(value)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            print(" " * (indent + 4) + f"{key}: {val_str}")


def print_typing(text: str, delay: float = 0.02) -> None:
    """Print text with typing effect."""
    for char in text:
        print(char, end="", flush=True)
        if delay > 0:
            time.sleep(delay)
    print()


def wait_for_enter(fast: bool = False) -> None:
    """Wait for user to press Enter."""
    if fast:
        time.sleep(0.3)
    else:
        input("\n    Press Enter to continue...")


# =============================================================================
# MCP Client Utilities
# =============================================================================


def parse_response(result: Any) -> dict[str, Any]:
    """Parse MCP tool response."""
    from mcp.types import TextContent

    if hasattr(result, "content") and result.content:
        content = result.content[0]
        text = content.text if isinstance(content, TextContent) else "{}"
        return json.loads(text)
    return {}


# =============================================================================
# Demo Workflows
# =============================================================================


async def demo_chain(client: Any, fast: bool = False) -> None:
    """Demonstrate chain reasoning workflow."""
    config = DEMO_PROBLEMS["chain"]

    print_header("CHAIN REASONING DEMO")
    print(f"\nProblem: {config['problem']}")
    wait_for_enter(fast)

    # Step 1: Start
    print_step(1, "start", "Initialize a chain reasoning session")
    result = await client.call_tool(
        "think",
        {"action": "start", "mode": "chain", "problem": config["problem"]},
    )
    resp = parse_response(result)
    session_id = resp.get("session_id", "")
    print_response(resp)
    wait_for_enter(fast)

    # Steps 2-4: Continue (add reasoning steps)
    for i, thought in enumerate(config["steps"], 2):
        print_step(i, "continue", "Add a reasoning step")
        print(f'         Thought: "{thought}"')
        result = await client.call_tool(
            "think",
            {"action": "continue", "session_id": session_id, "thought": thought},
        )
        resp = parse_response(result)
        print_response(resp)
        wait_for_enter(fast)

    # Step 5: Analyze
    print_step(len(config["steps"]) + 2, "analyze", "Get quality metrics for the session")
    result = await client.call_tool(
        "think",
        {"action": "analyze", "session_id": session_id},
    )
    resp = parse_response(result)
    print_response(resp)
    wait_for_enter(fast)

    # Step 6: Suggest
    print_step(len(config["steps"]) + 3, "suggest", "Get AI suggestion for next action")
    result = await client.call_tool(
        "think",
        {"action": "suggest", "session_id": session_id},
    )
    resp = parse_response(result)
    print_response(resp)
    wait_for_enter(fast)

    # Step 7: Finish
    print_step(len(config["steps"]) + 4, "finish", "Complete with final answer")
    result = await client.call_tool(
        "think",
        {
            "action": "finish",
            "session_id": session_id,
            "thought": config["answer"],
            "confidence": 0.95,
        },
    )
    resp = parse_response(result)
    print_response(resp)

    print("\n" + "=" * 70)
    print("  CHAIN DEMO COMPLETE!")
    print("=" * 70)


async def demo_matrix(client: Any, fast: bool = False) -> None:
    """Demonstrate matrix of thought workflow."""
    config = DEMO_PROBLEMS["matrix"]

    print_header("MATRIX OF THOUGHT DEMO")
    print(f"\nProblem: {config['problem']}")
    print(f"Perspectives: {config['perspectives']}")
    print(f"Criteria: {config['criteria']}")
    wait_for_enter(fast)

    # Step 1: Start matrix
    print_step(1, "start", "Initialize a matrix reasoning session")
    result = await client.call_tool(
        "think",
        {
            "action": "start",
            "mode": "matrix",
            "problem": config["problem"],
            "rows": len(config["perspectives"]),
            "cols": len(config["criteria"]),
        },
    )
    resp = parse_response(result)
    session_id = resp.get("session_id", "")
    print_response(resp)
    wait_for_enter(fast)

    # Steps 2-5: Fill matrix cells
    step = 2
    for perspective, criterion, thought in config["cells"]:
        row = config["perspectives"].index(perspective)
        col = config["criteria"].index(criterion)
        print_step(step, "continue", f"Fill cell [{row},{col}]: {perspective} × {criterion}")
        result = await client.call_tool(
            "think",
            {
                "action": "continue",
                "session_id": session_id,
                "row": row,
                "col": col,
                "thought": thought,
            },
        )
        resp = parse_response(result)
        print_response(resp)
        wait_for_enter(fast)
        step += 1

    # Steps 6-7: Synthesize columns
    for col, synthesis in enumerate(config["synthesis"]):
        print_step(step, "synthesize", f"Synthesize column {col}: {config['criteria'][col]}")
        result = await client.call_tool(
            "think",
            {
                "action": "synthesize",
                "session_id": session_id,
                "col": col,
                "thought": synthesis,
            },
        )
        resp = parse_response(result)
        print_response(resp)
        wait_for_enter(fast)
        step += 1

    # Final step: Finish
    print_step(step, "finish", "Complete with final answer")
    result = await client.call_tool(
        "think",
        {
            "action": "finish",
            "session_id": session_id,
            "thought": config["answer"],
            "confidence": 0.85,
        },
    )
    resp = parse_response(result)
    print_response(resp)

    print("\n" + "=" * 70)
    print("  MATRIX DEMO COMPLETE!")
    print("=" * 70)


async def demo_verify(client: Any, fast: bool = False) -> None:
    """Demonstrate verification workflow."""
    config = DEMO_PROBLEMS["verify"]

    print_header("VERIFICATION DEMO")
    print(f"\nClaim to verify: {config['problem']}")
    print("\nContext provided:")
    print_typing(config["context"][:200] + "...", delay=0.01 if not fast else 0)
    wait_for_enter(fast)

    # Step 1: Start verify session
    print_step(1, "start", "Initialize a verification session")
    result = await client.call_tool(
        "think",
        {
            "action": "start",
            "mode": "verify",
            "problem": config["problem"],
            "context": config["context"],
        },
    )
    resp = parse_response(result)
    session_id = resp.get("session_id", "")
    print_response(resp)
    wait_for_enter(fast)

    # Steps 2-3: Add claims
    step = 2
    for claim, _, _ in config["claims"]:
        print_step(step, "continue", f'Add claim: "{claim}"')
        result = await client.call_tool(
            "think",
            {"action": "continue", "session_id": session_id, "thought": claim},
        )
        resp = parse_response(result)
        print_response(resp)
        wait_for_enter(fast)
        step += 1

    # Steps 4-5: Verify each claim
    for i, (claim, verdict, evidence) in enumerate(config["claims"]):
        print_step(step, "verify", f"Verify claim {i}: {verdict}")
        result = await client.call_tool(
            "think",
            {
                "action": "verify",
                "session_id": session_id,
                "claim_id": i,
                "verdict": verdict,
                "evidence": evidence,
            },
        )
        resp = parse_response(result)
        print_response(resp)
        wait_for_enter(fast)
        step += 1

    # Final step: Finish
    print_step(step, "finish", "Complete verification")
    result = await client.call_tool(
        "think",
        {
            "action": "finish",
            "session_id": session_id,
            "thought": config["answer"],
        },
    )
    resp = parse_response(result)
    print_response(resp)

    print("\n" + "=" * 70)
    print("  VERIFICATION DEMO COMPLETE!")
    print("=" * 70)


async def demo_suggestion_learning(client: Any, fast: bool = False) -> None:
    """Demonstrate suggestion and feedback loop."""
    print_header("SUGGESTION LEARNING DEMO")
    print("\nThis demo shows the AI suggestion system with feedback learning.")
    wait_for_enter(fast)

    # Start a session
    print_step(1, "start", "Initialize session for suggestion demo")
    result = await client.call_tool(
        "think",
        {"action": "start", "mode": "chain", "problem": "What is 15% of 200?"},
    )
    resp = parse_response(result)
    session_id = resp.get("session_id", "")
    print_response(resp)
    wait_for_enter(fast)

    # Get suggestion
    print_step(2, "suggest", "Ask the system what to do next")
    result = await client.call_tool(
        "think",
        {"action": "suggest", "session_id": session_id},
    )
    resp = parse_response(result)
    print_response(resp)
    wait_for_enter(fast)

    # Auto-execute suggestion
    print_step(3, "auto", "Auto-execute the suggestion")
    result = await client.call_tool(
        "think",
        {"action": "auto", "session_id": session_id},
    )
    resp = parse_response(result)
    print_response(resp)
    wait_for_enter(fast)

    # Provide feedback
    print_step(4, "feedback", "Record that suggestion was helpful")
    result = await client.call_tool(
        "think",
        {"action": "feedback", "session_id": session_id, "outcome": "accepted"},
    )
    resp = parse_response(result)
    print_response(resp)
    wait_for_enter(fast)

    # Finish
    print_step(5, "finish", "Complete the session")
    result = await client.call_tool(
        "think",
        {
            "action": "finish",
            "session_id": session_id,
            "thought": "15% of 200 = 0.15 × 200 = 30",
            "confidence": 0.99,
        },
    )
    resp = parse_response(result)
    print_response(resp)

    print("\n" + "=" * 70)
    print("  SUGGESTION LEARNING DEMO COMPLETE!")
    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


async def main() -> None:
    """Run the interactive demo."""
    parser = argparse.ArgumentParser(
        description="Interactive MatrixMind MCP Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["chain", "matrix", "verify", "suggest", "all"],
        default="all",
        help="Demo mode to run (default: all)",
    )
    parser.add_argument(
        "--fast",
        "-f",
        action="store_true",
        help="Skip pauses and run quickly",
    )
    args = parser.parse_args()

    try:
        from fastmcp import Client
    except ImportError:
        print("Error: fastmcp not installed. Run: pip install fastmcp")
        sys.exit(1)

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " MATRIXMIND MCP INTERACTIVE DEMO ".center(68) + "║")
    print("║" + " Demonstrating the unified `think` function ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    if args.mode == "all":
        print("\nThis demo will walk through all reasoning modes:")
        print("  1. Chain Reasoning - Step-by-step problem solving")
        print("  2. Matrix of Thought - Multi-perspective analysis")
        print("  3. Verification - Fact checking against context")
        print("  4. Suggestion Learning - AI-guided reasoning")
    else:
        print(f"\nRunning {args.mode} demo...")

    if not args.fast:
        input("\nPress Enter to start...")

    async with Client("src/server.py") as client:
        if args.mode in ("chain", "all"):
            await demo_chain(client, fast=args.fast)
            if args.mode == "all":
                wait_for_enter(args.fast)

        if args.mode in ("matrix", "all"):
            await demo_matrix(client, fast=args.fast)
            if args.mode == "all":
                wait_for_enter(args.fast)

        if args.mode in ("verify", "all"):
            await demo_verify(client, fast=args.fast)
            if args.mode == "all":
                wait_for_enter(args.fast)

        if args.mode in ("suggest", "all"):
            await demo_suggestion_learning(client, fast=args.fast)

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " DEMO COMPLETE! ".center(68) + "║")
    print("║" + " ".center(68) + "║")
    print("║" + " Try it yourself: ".center(68) + "║")
    print("║" + "   uv run python -m src.server ".center(68) + "║")
    print("╚" + "═" * 68 + "╝\n")


if __name__ == "__main__":
    asyncio.run(main())
