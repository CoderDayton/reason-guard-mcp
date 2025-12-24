#!/usr/bin/env python3
"""Benchmark: Baseline LLM vs Reason Guard on GSM8K (Grade School Math).

Uses the official GSM8K dataset from OpenAI with guaranteed correct answers.

Run:
    uv run python examples/benchmark_gsm8k.py
    uv run python examples/benchmark_gsm8k.py --n 20  # Test on 20 problems
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass

from datasets import load_dataset
from dotenv import load_dotenv
from fastmcp import Client

from examples.llm_client import LLMClient
from src.models.context_encoder import ContextEncoder
from src.models.model_manager import ModelManager
from src.models.vector_store import AsyncVectorStore, VectorStoreConfig
from src.server import mcp
from src.tools.unified_reasoner import init_unified_manager

load_dotenv()


@dataclass
class Problem:
    id: str
    question: str
    answer: str  # Just the number
    full_solution: str  # Step-by-step reasoning


def extract_answer(answer_text: str) -> str:
    """Extract final numeric answer from GSM8K solution."""
    match = re.search(r"####\s*([\d,]+)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return ""


def load_gsm8k(n: int = 50, seed: int = 42) -> list[Problem]:
    """Load n problems from GSM8K test set."""
    ds = load_dataset("openai/gsm8k", "main", split="test")

    # Shuffle and select n problems
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    problems = []
    for i, ex in enumerate(ds):
        problems.append(
            Problem(
                id=f"gsm8k_{i + 1}",
                question=ex["question"],
                answer=extract_answer(ex["answer"]),
                full_solution=ex["answer"],
            )
        )
    return problems


def extract_number(response: str) -> str:
    """Extract the final numeric answer from LLM response."""
    # Remove think tags if present (Qwen models)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    # Look for common patterns (in priority order)
    patterns = [
        r"Answer:\s*\$?(-?[\d,]+)",  # Our requested format
        r"(?:final answer|answer is|equals?|=)\s*\$?(-?[\d,]+)",
        r"####\s*(-?[\d,]+)",
        r"\*\*(-?[\d,]+)\*\*",
        r"(?:the answer|result|total)\s+(?:is|=|:)?\s*\$?(-?[\d,]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "")

    # Fall back: find last number in response
    numbers = re.findall(r"\b(\d+)\b", response)
    if numbers:
        return numbers[-1]
    return ""


def check_answer(expected: str, got: str) -> bool:
    """Check if answers match (numeric comparison)."""
    try:
        return int(expected) == int(got)
    except (ValueError, TypeError):
        return expected.strip() == got.strip()


async def solve_baseline(problem: Problem, llm: LLMClient) -> tuple[bool, str]:
    """Solve problem with baseline LLM (no Reason Guard)."""
    prompt = f"""Solve this math problem step by step.

Problem: {problem.question}

Show your work, then on the final line write ONLY: "Answer: [number]" where [number] is just the final numeric answer."""

    response = await llm.ask(prompt)
    got = extract_number(response)
    correct = check_answer(problem.answer, got)
    return correct, got


async def solve_with_reason_guard(
    problem: Problem, llm: LLMClient, client: Client, debug: bool = False
) -> tuple[bool, str]:
    """Solve problem with Reason Guard-enhanced reasoning."""
    # Start thinking session
    start_result = await client.call_tool(
        "think",
        {"action": "start", "problem": problem.question},
    )

    # Extract session_id from start result
    session_id = None
    start_text = ""
    if hasattr(start_result, "content"):
        for content in start_result.content:
            if hasattr(content, "text"):
                start_text = content.text
                break
    elif isinstance(start_result, str):
        start_text = start_result

    if start_text:
        try:
            start_data = json.loads(start_text)
            session_id = start_data.get("session_id")
            if debug:
                print(f"    [DEBUG] Session started: {session_id}")
                print(
                    f"    [DEBUG] Mode: {start_data.get('actual_mode')}, Domain: {start_data.get('domain')}"
                )
        except json.JSONDecodeError:
            if debug:
                print(f"    [DEBUG] Failed to parse start result: {start_text[:100]}")

    # Get initial reasoning from LLM
    prompt = f"""Solve this math problem step by step.

Problem: {problem.question}

Show your work, then on the final line write ONLY: "Answer: [number]" where [number] is just the final numeric answer."""

    reasoning = await llm.ask(prompt)
    if debug:
        print(f"    [DEBUG] LLM reasoning ({len(reasoning)} chars): {reasoning[:150]}...")

    # Add thought to Reason Guard with session_id
    continue_params = {"action": "continue", "thought": reasoning}
    if session_id:
        continue_params["session_id"] = session_id

    continue_result = await client.call_tool("think", continue_params)

    # Get feedback
    feedback_text = ""
    if hasattr(continue_result, "content"):
        for content in continue_result.content:
            if hasattr(content, "text"):
                feedback_text = content.text
                break
    elif isinstance(continue_result, str):
        feedback_text = continue_result

    if debug:
        print(f"    [DEBUG] Reason Guard feedback: {feedback_text[:300]}...")

    # Parse guidance from feedback - now properly handles the response structure
    guidance_parts = []
    try:
        data = json.loads(feedback_text)

        # Check for blind spots (key actionable insight)
        blind_spots = data.get("blind_spots", [])
        if blind_spots:
            guidance_parts.append(f"Potential issues: {', '.join(blind_spots)}")

        # Check for conflicts (contradiction detection)
        conflicts = data.get("conflicts", [])
        if conflicts:
            guidance_parts.append(f"Conflicts detected: {', '.join(conflicts)}")

        # Check for guidance dict (from NORMAL/FULL verbosity)
        guidance_obj = data.get("guidance", {})
        if isinstance(guidance_obj, dict):
            action = guidance_obj.get("suggested_action", "")
            reason = guidance_obj.get("reason", "")
            if action or reason:
                guidance_parts.append(f"Suggested: {action}. {reason}")
        elif isinstance(guidance_obj, str) and guidance_obj:
            guidance_parts.append(guidance_obj)

        # Check for similar thoughts from RAG
        similar = data.get("similar", [])
        if similar:
            hints = [s.get("text", "")[:200] for s in similar[:2]]
            guidance_parts.append(f"Similar reasoning patterns: {'; '.join(hints)}")

        # Check survival score as quality indicator
        survival_score = data.get("survival_score", 0)
        if survival_score < 0.5:
            guidance_parts.append(f"Low confidence ({survival_score:.2f}) - verify your reasoning")

    except json.JSONDecodeError:
        if debug:
            print("    [DEBUG] Failed to parse feedback JSON")

    guidance = "\n".join(guidance_parts) if guidance_parts else ""
    if debug:
        print(f"    [DEBUG] Extracted guidance: {guidance[:200] if guidance else '(none)'}")

    # Get refined answer using feedback
    if guidance:
        refine_prompt = f"""Review and verify your solution based on this analysis:
{guidance}

Original problem: {problem.question}

Your previous reasoning: {reasoning}

Please reconsider your solution. If the feedback suggests issues, fix them.
On the final line, write ONLY: "Answer: [number]" where [number] is just the final numeric answer."""
        final_response = await llm.ask(refine_prompt)
        if debug:
            print(f"    [DEBUG] Refined response: {final_response[:150]}...")
    else:
        final_response = reasoning

    # Finalize session
    finish_params = {"action": "finish"}
    if session_id:
        finish_params["session_id"] = session_id
    await client.call_tool("think", finish_params)

    got = extract_number(final_response)
    correct = check_answer(problem.answer, got)
    return correct, got


async def main(n_problems: int = 20, debug: bool = False):
    """Run GSM8K benchmark."""
    print("=" * 60)
    print("GSM8K Benchmark: Baseline vs Reason Guard")
    print("=" * 60)
    if debug:
        print("[DEBUG MODE ENABLED]")

    # Load problems
    print(f"\nLoading {n_problems} problems from GSM8K...")
    problems = load_gsm8k(n=n_problems)
    print(f"Loaded {len(problems)} problems")

    # Initialize components
    print("\nInitializing embedding model...")
    ModelManager.get_instance().initialize("BAAI/bge-small-en-v1.5", blocking=True)
    encoder = ContextEncoder()
    print(f"  Encoder on {encoder.device}")

    print("Initializing vector store...")
    config = VectorStoreConfig(db_path=":memory:", embedding_model="BAAI/bge-small-en-v1.5")
    vector_store = AsyncVectorStore(config, encoder=encoder)
    await vector_store.__aenter__()

    print("Initializing unified reasoner manager...")
    await init_unified_manager(
        vector_store=vector_store,
        encoder=encoder,
        enable_semantic_scoring=True,
    )

    # Seed with math reasoning patterns
    seeds = [
        "When solving word problems, identify: what is given, what is asked, and what operations connect them.",
        "For multi-step problems, solve one step at a time and verify intermediate results.",
        "Check your answer by substituting back into the original problem.",
        "Look for hidden constraints or conditions in the problem statement.",
        "For rate problems, identify the unit rate first (per hour, per item, etc).",
    ]
    for i, seed in enumerate(seeds):
        await vector_store.add_thought(
            thought=seed,
            session_id="seed",
            step=i,
            strategy="math_reasoning",
            score=0.9,
        )
    print(f"  Seeded {len(seeds)} reasoning patterns")

    llm = LLMClient()
    print(f"\nLLM: {llm.model} @ {llm.base_url}")

    # Connect to MCP server
    async with Client(mcp) as client:
        print("Connected to Reason Guard MCP server")

        baseline_correct = 0
        reasonguard_correct = 0

        print(f"\n{'=' * 60}")
        print("Running benchmark...")
        print(f"{'=' * 60}\n")

        for i, problem in enumerate(problems, 1):
            print(f"[{i}/{len(problems)}] {problem.id}")
            print(f"  Q: {problem.question[:60]}...")
            print(f"  Expected: {problem.answer}")

            # Baseline
            b_ok, b_got = await solve_baseline(problem, llm)
            baseline_correct += b_ok
            print(f"  Baseline:    {'✓' if b_ok else '✗'} (got {b_got})")

            # Reason Guard
            m_ok, m_got = await solve_with_reason_guard(problem, llm, client, debug=debug)
            reasonguard_correct += m_ok
            print(f"  ReasonGuard: {'✓' if m_ok else '✗'} (got {m_got})")
            print()

    # Results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    b_pct = 100 * baseline_correct / len(problems)
    m_pct = 100 * reasonguard_correct / len(problems)

    print(f"{'Method':<15} {'Correct':<10} {'Accuracy':<10}")
    print(f"{'Baseline':<15} {baseline_correct}/{len(problems):<10} {b_pct:.1f}%")
    print(f"{'ReasonGuard':<15} {reasonguard_correct}/{len(problems):<10} {m_pct:.1f}%")
    print(f"\nImprovement: {m_pct - b_pct:+.1f}%")

    # Save results
    results = {
        "benchmark": "GSM8K",
        "model": llm.model,
        "n_problems": len(problems),
        "baseline_correct": baseline_correct,
        "baseline_accuracy": b_pct,
        "reasonguard_correct": reasonguard_correct,
        "reasonguard_accuracy": m_pct,
        "improvement": m_pct - b_pct,
    }

    with open("gsm8k_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to gsm8k_results.json")


if __name__ == "__main__":
    n = 20  # Default number of problems
    debug = False

    # Parse args
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--n" and i + 1 < len(args):
            n = int(args[i + 1])
        elif arg == "--debug":
            debug = True

    asyncio.run(main(n_problems=n, debug=debug))
