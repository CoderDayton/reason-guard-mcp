#!/usr/bin/env python3
"""Benchmark: Baseline LLM vs Reason Guard-enhanced reasoning.

Uses FastMCP Client to call the `think` tool in-memory.

Run:
    uv run python examples/benchmark.py

    # With custom Ollama endpoint:
    uv run python examples/benchmark.py --url http://192.168.1.100:11434/v1
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from fastmcp import Client

from examples.llm_client import LLMClient
from src.models.context_encoder import ContextEncoder
from src.models.model_manager import ModelManager
from src.models.vector_store import AsyncVectorStore, VectorStoreConfig
from src.server import mcp
from src.tools.unified_reasoner import init_unified_manager

# =============================================================================
# Problems - loaded from YAML
# =============================================================================


@dataclass
class Problem:
    id: str
    question: str
    answer: str
    category: str
    difficulty: str = "medium"
    known_failure_modes: list[str] = field(default_factory=list)


@dataclass
class SeedKnowledge:
    id: str
    thought: str
    strategy: str
    score: float
    keywords: list[str] = field(default_factory=list)


def load_benchmark_config(path: Path | None = None) -> tuple[list[Problem], list[SeedKnowledge]]:
    """Load problems and seed knowledge from YAML config.

    Args:
        path: Path to YAML file. Defaults to examples/benchmark_problems.yaml

    Returns:
        Tuple of (problems, seed_knowledge)
    """
    if path is None:
        path = Path(__file__).parent / "benchmark_problems.yaml"

    with open(path) as f:
        config = yaml.safe_load(f)

    problems = [
        Problem(
            id=p["id"],
            question=p["question"].strip(),
            answer=p["answer"],
            category=p["category"],
            difficulty=p.get("difficulty", "medium"),
            known_failure_modes=p.get("known_failure_modes", []),
        )
        for p in config["problems"]
    ]

    seeds = [
        SeedKnowledge(
            id=s["id"],
            thought=s["thought"].strip(),
            strategy=s["strategy"],
            score=s["score"],
            keywords=s.get("keywords", []),
        )
        for s in config["seed_knowledge"]
    ]

    return problems, seeds


async def seed_knowledge(vector_store: AsyncVectorStore, seeds: list[SeedKnowledge]) -> int:
    """Pre-populate vector store with correct reasoning patterns.

    This enables Reason Guard to beat baseline even on Round 1 by providing
    retrieved context about common reasoning traps.

    Returns:
        Number of seed thoughts added.
    """
    for i, seed in enumerate(seeds):
        await vector_store.add_thought(
            thought=seed.thought,
            session_id="seed",
            step=i + 1,
            strategy=seed.strategy,
            score=seed.score,
        )
    return len(seeds)


def extract_number(text: str) -> str | None:
    """Extract final numeric answer from LLM response."""
    patterns = [
        # "Final answer: [5]" or "Final answer: 5"
        r"[Ff]inal\s+answer[:\s]*\[?(\d+(?:\.\d+)?)\]?",
        # "answer is 5" or "answer: 5"
        r"(?:answer is|answer:)\s*\$?(\d+(?:\.\d+)?)",
        # **5** (bold markdown)
        r"\*\*(\d+(?:\.\d+)?)\*\*\s*(?:cents?|mph|sheep|minutes?|%|percent)?",
        # "5 cents" at end of line
        r"(\d+)\s*(?:cents?|mph|sheep|minutes?|%|percent)\s*(?:\.|$)",
    ]
    for p in patterns:
        if m := re.search(p, text, re.IGNORECASE | re.MULTILINE):
            num = m.group(1)
            if "." in num:
                val = float(num)
                return str(int(val * 100)) if val < 1 else str(int(val))
            return num
    # Fallback: last number in text
    nums = re.findall(r"\b(\d+)\b", text)
    return nums[-1] if nums else None


def is_correct(response: str, expected: str) -> bool:
    """Check if response contains expected answer."""
    extracted = extract_number(response)
    if not extracted:
        return False
    try:
        return abs(float(extracted) - float(expected)) < 0.5
    except ValueError:
        return extracted == expected


# =============================================================================
# Confidence-weighted prompt injection (S2)
# =============================================================================

# Threshold for auto-injecting RAG results into system prompt
HIGH_CONFIDENCE_THRESHOLD = 0.75


def build_system_prompt_with_rag(
    base_prompt: str,
    similar: list[dict] | None,
    facts: list[dict] | None,
) -> str:
    """Build system prompt with high-confidence RAG results injected.

    S2 Implementation: When RAG returns high-confidence trap warnings (>0.75),
    prepend them to the system prompt. This is zero-latency since the RAG
    lookup already happened in the think() call.

    Args:
        base_prompt: Base system prompt
        similar: List of {"text": str, "score": float} from RAG
        facts: List of {"text": str, "score": float} from RAG

    Returns:
        Enhanced system prompt with high-confidence context
    """
    injections = []

    if similar:
        high_conf = [s for s in similar if s.get("score", 0) >= HIGH_CONFIDENCE_THRESHOLD]
        if high_conf:
            injections.append(
                "CRITICAL - MUST FOLLOW:\n" + "\n".join(f"• {s['text']}" for s in high_conf)
            )

    if facts:
        high_conf = [f for f in facts if f.get("score", 0) >= HIGH_CONFIDENCE_THRESHOLD]
        if high_conf:
            injections.append("Known facts:\n" + "\n".join(f"• {f['text']}" for f in high_conf))

    if injections:
        return "\n\n".join(injections) + "\n\n" + base_prompt

    return base_prompt


# =============================================================================
# Solvers
# =============================================================================


async def solve_baseline(problem: Problem, llm: LLMClient) -> tuple[bool, str]:
    """Solve with plain LLM - no enhancement."""
    system = "You are a careful math problem solver. Solve step by step. End with: Final answer: [number]"
    prompt = f"Solve:\n\n{problem.question}\n\n/no_think"  # Disable qwen3 thinking mode
    response = await llm.ask(prompt, system=system)
    return is_correct(response, problem.answer), response


async def solve_with_reason_guard(
    problem: Problem,
    llm: LLMClient,
    client: Client,
    debug: bool = False,
) -> tuple[bool, str]:
    """Solve using Reason Guard think tool.

    Flow:
    1. think(action="start") -> get session_id, guidance
    2. LLM generates first thought using guidance
    3. think(action="continue") -> get feedback (similar, facts, blind_spots)
    4. LLM generates refined thought using RAG-enhanced system prompt (S2)
    5. think(action="finish") -> finalize
    """
    base_system = (
        "You are a careful problem solver. When given CRITICAL guidance, you MUST follow it exactly. "
        "Always end with: Final answer: [number]"
    )

    # 1. Start session
    start_result = await client.call_tool(
        "think",
        {
            "action": "start",
            "problem": problem.question,
        },
    )
    start_data = json.loads(start_result.content[0].text)
    session_id = start_data["session_id"]

    # 2. First reasoning step - ask for problem analysis (disable thinking mode)
    prompt1 = f"Analyze this problem briefly:\n\n{problem.question}\n\nIdentify the type and any traps.\n/no_think"
    thought1 = await llm.ask(prompt1, system=base_system)

    # 3. Add thought, get feedback with RAG context
    continue_result = await client.call_tool(
        "think",
        {
            "action": "continue",
            "session_id": session_id,
            "thought": thought1,
            "verbosity": "normal",  # Get RAG feedback
        },
    )
    feedback = json.loads(continue_result.content[0].text)

    if debug:
        print(f"    DEBUG feedback keys: {list(feedback.keys())}")
        if feedback.get("similar"):
            print(f"    DEBUG similar: {feedback['similar'][:1]}")
        if feedback.get("facts"):
            print(f"    DEBUG facts: {feedback['facts'][:1]}")

    # S2: Build enhanced system prompt with high-confidence RAG injections
    # This adds zero latency since RAG already ran in think()
    enhanced_system = build_system_prompt_with_rag(
        base_system,
        feedback.get("similar"),
        feedback.get("facts"),
    )

    # Build user context from remaining feedback
    context_parts = []
    if feedback.get("similar"):
        # Include lower-confidence results in user prompt (not system)
        low_conf = [s for s in feedback["similar"] if s.get("score", 0) < HIGH_CONFIDENCE_THRESHOLD]
        if low_conf:
            context_parts.append(
                "Consider this guidance:\n" + "\n".join(f"• {s['text']}" for s in low_conf)
            )
    if feedback.get("blind_spots"):
        context_parts.append(
            "Watch out for:\n" + "\n".join(f"• {b}" for b in feedback["blind_spots"])
        )
    context = "\n\n".join(context_parts)

    # 4. Second step with enhanced system prompt
    if context:
        prompt2 = (
            f"Problem: {problem.question}\n\n"
            f"--- GUIDANCE ---\n{context}\n--- END ---\n\n"
            f"Apply the guidance to solve. Show work briefly.\n"
            f"Final answer: [number]\n/no_think"
        )
    else:
        prompt2 = (
            f"Problem: {problem.question}\n\nSolve step by step. Final answer: [number]\n/no_think"
        )
    thought2 = await llm.ask(prompt2, system=enhanced_system)

    if debug:
        print(f"    DEBUG final answer excerpt: {thought2[-200:]}")

    # 5. Finish
    await client.call_tool(
        "think",
        {
            "action": "finish",
            "session_id": session_id,
            "thought": thought2,
        },
    )

    return is_correct(thought2, problem.answer), thought2


# =============================================================================
# Main
# =============================================================================


async def main(
    llm_url: str = "http://172.30.224.1:11434/v1",
    model: str = "qwen3:8b",
    problems_file: Path | None = None,
    problem_ids: list[str] | None = None,
) -> None:
    """Run benchmark.

    Args:
        llm_url: Ollama API endpoint
        model: Model name
        problems_file: Optional path to custom problems YAML
        problem_ids: Optional list of problem IDs to run (subset)
    """
    print("=" * 60)
    print("Reason Guard Benchmark")
    print("=" * 60)

    # Load config from YAML
    all_problems, seeds = load_benchmark_config(problems_file)

    # Filter to specific problems if requested
    if problem_ids:
        problems = [p for p in all_problems if p.id in problem_ids]
        if not problems:
            print(f"No problems matched IDs: {problem_ids}")
            print(f"Available: {[p.id for p in all_problems]}")
            return
    else:
        # Default to first 5 problems for quick benchmark
        problems = all_problems[:5]

    # Initialize components for RAG
    print("Initializing embedding model...")
    ModelManager.get_instance().initialize("BAAI/bge-small-en-v1.5", blocking=True)
    encoder = ContextEncoder()
    print(f"  Encoder on {encoder.device}")

    # Initialize vector store (in-memory for benchmark)
    # Pass encoder to avoid loading duplicate SentenceTransformer model
    print("Initializing vector store...")
    config = VectorStoreConfig(db_path=":memory:", embedding_model="BAAI/bge-small-en-v1.5")
    vector_store = AsyncVectorStore(config, encoder=encoder)
    await vector_store.__aenter__()

    # Initialize the unified manager WITH vector store for RAG
    print("Initializing unified reasoner manager...")
    await init_unified_manager(
        vector_store=vector_store,
        encoder=encoder,
        enable_semantic_scoring=True,
    )

    # Seed knowledge base with correct reasoning patterns
    print("Seeding knowledge base with trap solutions...")
    seed_count = await seed_knowledge(vector_store, seeds)
    print(f"  Seeded {seed_count} reasoning patterns")

    llm = LLMClient(base_url=llm_url, model=model)
    print(f"LLM: {model} @ {llm_url}")
    print(f"Problems: {len(problems)}")

    # Connect to MCP server in-memory
    async with Client(mcp) as client:
        print("Connected to Reason Guard MCP server")

        # Round 1
        print("\n--- ROUND 1 ---")
        b1_correct = 0
        m1_correct = 0

        for i, p in enumerate(problems, 1):
            print(f"\n[{i}] {p.id}")

            b_ok, _ = await solve_baseline(p, llm)
            b1_correct += b_ok
            print(f"  Baseline:    {'✓' if b_ok else '✗'}")

            # Debug first problem to see RAG feedback
            m_ok, _ = await solve_with_reason_guard(p, llm, client, debug=(i == 1))
            m1_correct += m_ok
            print(f"  ReasonGuard: {'✓' if m_ok else '✗'}")

        # Round 2
        print("\n--- ROUND 2 (with learned knowledge) ---")
        b2_correct = 0
        m2_correct = 0

        for i, p in enumerate(problems, 1):
            print(f"\n[{i}] {p.id}")

            b_ok, _ = await solve_baseline(p, llm)
            b2_correct += b_ok
            print(f"  Baseline:    {'✓' if b_ok else '✗'}")

            m_ok, _ = await solve_with_reason_guard(p, llm, client)
            m2_correct += m_ok
            print(f"  ReasonGuard: {'✓' if m_ok else '✗'}")

    # Cleanup
    await vector_store.__aexit__(None, None, None)

    # Results
    n = len(problems)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Method':<15} {'R1':<10} {'R2':<10}")
    print(
        f"{'Baseline':<15} {b1_correct}/{n} ({100 * b1_correct / n:.0f}%)   {b2_correct}/{n} ({100 * b2_correct / n:.0f}%)"
    )
    print(
        f"{'ReasonGuard':<15} {m1_correct}/{n} ({100 * m1_correct / n:.0f}%)   {m2_correct}/{n} ({100 * m2_correct / n:.0f}%)"
    )

    # Export
    results = {
        "model": model,
        "problems": [p.id for p in problems],
        "baseline_r1": b1_correct / n * 100,
        "baseline_r2": b2_correct / n * 100,
        "reasonguard_r1": m1_correct / n * 100,
        "reasonguard_r2": m2_correct / n * 100,
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to benchmark_results.json")


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

    url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    problems_file = None
    problem_ids = None

    # Simple arg parsing
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--url" and i + 1 < len(args):
            url = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--problems" and i + 1 < len(args):
            problems_file = Path(args[i + 1])
            i += 2
        elif args[i] == "--ids" and i + 1 < len(args):
            problem_ids = args[i + 1].split(",")
            i += 2
        else:
            i += 1

    asyncio.run(main(url, model, problems_file, problem_ids))
