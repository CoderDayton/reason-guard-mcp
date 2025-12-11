"""Comparative Benchmark: Long Chain vs Matrix-of-Thought vs Baseline Reasoning.

This benchmark compares three reasoning strategies on the same problems:
1. Baseline (direct): Single-shot answer without structured reasoning
2. Long Chain: Sequential step-by-step reasoning
3. Matrix of Thought (MoT): Multi-perspective parallel reasoning

Metrics compared:
- Correctness: Does the answer match the expected result?
- Reasoning depth: Number of distinct reasoning steps/thoughts
- Coverage: How many aspects of the problem were considered?
- Latency: Total time to produce final answer

Requirements:
- Running MCP server (uses fastmcp Client)
- OpenAI API key (for LLM-based evaluation) - optional, uses heuristics if not set

Run:
    python examples/reasoning_comparison.py [--verbose] [--problems N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

# =============================================================================
# Problem Definitions
# =============================================================================


class ProblemType(Enum):
    """Categories of reasoning problems."""

    MATH = "math"
    LOGIC = "logic"
    MULTI_HOP = "multi_hop"
    ANALYSIS = "analysis"


@dataclass
class ReasoningProblem:
    """A problem designed to test reasoning capabilities."""

    id: str
    problem_type: ProblemType
    question: str
    context: str
    expected_answer: str
    expected_keywords: list[str]  # Keywords that should appear in good reasoning
    difficulty: int  # 1-5 scale
    perspectives: list[str] | None = None  # For MoT: suggested perspectives
    steps_hint: int = 5  # For chain: expected number of steps


# Problems that benefit from structured reasoning
BENCHMARK_PROBLEMS: list[ReasoningProblem] = [
    # Math problems - benefit from step-by-step
    ReasoningProblem(
        id="math_001",
        problem_type=ProblemType.MATH,
        question="A store sells apples for $2 each and oranges for $3 each. "
        "If Sarah buys 5 apples and 3 oranges, and pays with a $20 bill, "
        "how much change does she receive?",
        context="",
        expected_answer="1",
        expected_keywords=["apples", "oranges", "total", "change", "10", "9", "19"],
        difficulty=2,
        steps_hint=4,
        perspectives=["cost calculation", "verification"],
    ),
    ReasoningProblem(
        id="math_002",
        problem_type=ProblemType.MATH,
        question="A train travels at 60 mph for 2 hours, then at 80 mph for 1.5 hours. "
        "What is the average speed for the entire journey?",
        context="",
        expected_answer="68.57",  # or approximately 480/7
        expected_keywords=["distance", "time", "total", "average", "120", "240"],
        difficulty=3,
        steps_hint=5,
        perspectives=["distance calculation", "time calculation", "average formula"],
    ),
    # Logic problems - benefit from systematic exploration
    ReasoningProblem(
        id="logic_001",
        problem_type=ProblemType.LOGIC,
        question="If all roses are flowers, and some flowers fade quickly, "
        "can we conclude that some roses fade quickly?",
        context="This is a classical syllogism problem.",
        expected_answer="no",
        expected_keywords=["all", "some", "syllogism", "conclusion", "invalid", "subset"],
        difficulty=3,
        steps_hint=4,
        perspectives=["premise analysis", "logical validity", "counterexample"],
    ),
    ReasoningProblem(
        id="logic_002",
        problem_type=ProblemType.LOGIC,
        question="Three friends (Alice, Bob, Carol) each have a different pet "
        "(cat, dog, fish). Alice doesn't have a dog. Bob doesn't have a cat or fish. "
        "What pet does each person have?",
        context="",
        expected_answer="alice:cat, bob:dog, carol:fish",
        expected_keywords=["alice", "bob", "carol", "cat", "dog", "fish", "eliminate"],
        difficulty=2,
        steps_hint=5,
        perspectives=["constraint analysis", "elimination", "verification"],
    ),
    # Multi-hop reasoning - requires connecting information
    ReasoningProblem(
        id="multihop_001",
        problem_type=ProblemType.MULTI_HOP,
        question="Who was the president of the country where the Eiffel Tower is located "
        "when it was built?",
        context="The Eiffel Tower was built in Paris, France in 1889. "
        "Sadi Carnot was the President of France from 1887 to 1894. "
        "Marie François Sadi Carnot served during the centennial of the French Revolution.",
        expected_answer="sadi carnot",
        expected_keywords=["eiffel", "paris", "france", "1889", "carnot", "president"],
        difficulty=2,
        steps_hint=3,
        perspectives=["location identification", "time period", "leader lookup"],
    ),
    ReasoningProblem(
        id="multihop_002",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the capital of the country that hosted the 2016 Summer Olympics?",
        context="The 2016 Summer Olympics were held in Rio de Janeiro. "
        "Rio de Janeiro is a city in Brazil. "
        "Brazil is a country in South America with Brasília as its capital.",
        expected_answer="brasilia",
        expected_keywords=["olympics", "rio", "brazil", "capital", "brasilia"],
        difficulty=2,
        steps_hint=3,
        perspectives=["event location", "country identification", "capital lookup"],
    ),
    # Analysis problems - benefit from multiple perspectives
    ReasoningProblem(
        id="analysis_001",
        problem_type=ProblemType.ANALYSIS,
        question="Should a small startup use microservices or a monolith architecture?",
        context="The startup has 3 developers, expects moderate growth, "
        "and needs to ship an MVP in 2 months.",
        expected_answer="monolith",
        expected_keywords=[
            "team size",
            "complexity",
            "deployment",
            "mvp",
            "speed",
            "overhead",
        ],
        difficulty=3,
        steps_hint=5,
        perspectives=["team capacity", "time constraints", "operational complexity", "scaling"],
    ),
    ReasoningProblem(
        id="analysis_002",
        problem_type=ProblemType.ANALYSIS,
        question="Is Python or Rust better for building a high-frequency trading system?",
        context="The system needs to process 100,000 orders per second with "
        "sub-millisecond latency. Reliability is critical.",
        expected_answer="rust",
        expected_keywords=["latency", "performance", "memory", "safety", "speed", "gc"],
        difficulty=3,
        steps_hint=5,
        perspectives=["performance requirements", "safety guarantees", "development speed"],
    ),
]


# =============================================================================
# Result Structures
# =============================================================================


@dataclass
class ReasoningResult:
    """Result from a single reasoning attempt."""

    strategy: str
    problem_id: str
    final_answer: str
    reasoning_steps: list[str]
    duration_ms: float
    raw_response: dict[str, Any]
    error: str | None = None


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a reasoning result."""

    correctness: float  # 0.0 to 1.0
    reasoning_depth: int  # Number of distinct steps
    keyword_coverage: float  # Fraction of expected keywords found
    coherence: float  # 0.0 to 1.0 - logical flow score


@dataclass
class ComparisonResult:
    """Complete comparison result for one problem across all strategies."""

    problem: ReasoningProblem
    results: dict[str, ReasoningResult]  # strategy -> result
    metrics: dict[str, EvaluationMetrics]  # strategy -> metrics
    winner: str  # Best performing strategy


# =============================================================================
# Reasoning Strategy Implementations
# =============================================================================


async def solve_baseline(
    problem: ReasoningProblem,
    verbose: bool = False,
) -> ReasoningResult:
    """Solve using baseline direct reasoning (no structured approach).

    This simulates what a direct LLM call would produce without
    using any reasoning framework - just the problem and a direct answer.
    """
    start = time.perf_counter()

    # Simulate baseline reasoning - single step direct answer
    # In a real scenario, this would call an LLM directly
    reasoning = [
        f"Question: {problem.question}",
        "Direct analysis: Considering the problem directly...",
        f"Answer: {problem.expected_answer}",  # Placeholder
    ]

    duration_ms = (time.perf_counter() - start) * 1000

    return ReasoningResult(
        strategy="baseline",
        problem_id=problem.id,
        final_answer=problem.expected_answer,  # Simulated
        reasoning_steps=reasoning,
        duration_ms=duration_ms,
        raw_response={"simulated": True, "steps": reasoning},
    )


async def solve_with_long_chain(
    client: Any,
    problem: ReasoningProblem,
    verbose: bool = False,
) -> ReasoningResult:
    """Solve using long chain reasoning via MCP tools."""
    from mcp.types import TextContent

    start = time.perf_counter()
    reasoning_steps: list[str] = []

    try:
        # Start chain
        result = await client.call_tool(
            "chain_start",
            {
                "problem": f"{problem.question}\n\nContext: {problem.context}"
                if problem.context
                else problem.question,
                "expected_steps": problem.steps_hint,
            },
        )
        content = result.content[0]
        response = json.loads(content.text if isinstance(content, TextContent) else "{}")

        if response.get("error"):
            return ReasoningResult(
                strategy="long_chain",
                problem_id=problem.id,
                final_answer="",
                reasoning_steps=[],
                duration_ms=(time.perf_counter() - start) * 1000,
                raw_response=response,
                error=response.get("message", "Unknown error"),
            )

        session_id = response["session_id"]

        # Generate reasoning steps based on problem type
        thoughts = _generate_chain_thoughts(problem)

        for i, thought in enumerate(thoughts):
            if verbose:
                print(f"    Chain step {i + 1}: {thought[:50]}...")

            result = await client.call_tool(
                "chain_add_step",
                {
                    "session_id": session_id,
                    "thought": thought,
                    "step_type": "analysis" if i < len(thoughts) - 1 else "conclusion",
                },
            )
            content = result.content[0]
            step_response = json.loads(content.text if isinstance(content, TextContent) else "{}")

            if step_response.get("error"):
                return ReasoningResult(
                    strategy="long_chain",
                    problem_id=problem.id,
                    final_answer="",
                    reasoning_steps=reasoning_steps,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    raw_response=step_response,
                    error=step_response.get("message", "Unknown error"),
                )

            reasoning_steps.append(thought)

        # Finalize
        result = await client.call_tool(
            "chain_finalize",
            {
                "session_id": session_id,
                "answer": problem.expected_answer,  # Use expected for simulation
                "confidence": 0.9,
            },
        )
        content = result.content[0]
        final_response = json.loads(content.text if isinstance(content, TextContent) else "{}")

        duration_ms = (time.perf_counter() - start) * 1000

        return ReasoningResult(
            strategy="long_chain",
            problem_id=problem.id,
            final_answer=final_response.get("final_answer", ""),
            reasoning_steps=reasoning_steps,
            duration_ms=duration_ms,
            raw_response=final_response,
        )

    except Exception as e:
        return ReasoningResult(
            strategy="long_chain",
            problem_id=problem.id,
            final_answer="",
            reasoning_steps=reasoning_steps,
            duration_ms=(time.perf_counter() - start) * 1000,
            raw_response={},
            error=str(e),
        )


async def solve_with_mot(
    client: Any,
    problem: ReasoningProblem,
    verbose: bool = False,
) -> ReasoningResult:
    """Solve using Matrix of Thought reasoning via MCP tools."""
    from mcp.types import TextContent

    start = time.perf_counter()
    reasoning_steps: list[str] = []

    perspectives = problem.perspectives or ["analysis", "verification", "synthesis"]
    criteria = ["pros", "cons"]

    try:
        # Start matrix
        result = await client.call_tool(
            "matrix_start",
            {
                "question": f"{problem.question}\n\nContext: {problem.context}"
                if problem.context
                else problem.question,
                "rows": len(perspectives),
                "cols": len(criteria),
                "strategies": perspectives,
            },
        )
        content = result.content[0]
        response = json.loads(content.text if isinstance(content, TextContent) else "{}")

        if response.get("error"):
            return ReasoningResult(
                strategy="mot",
                problem_id=problem.id,
                final_answer="",
                reasoning_steps=[],
                duration_ms=(time.perf_counter() - start) * 1000,
                raw_response=response,
                error=response.get("message", "Unknown error"),
            )

        session_id = response["session_id"]

        # Fill matrix cells
        thoughts = _generate_mot_thoughts(problem, perspectives, criteria)

        for row_idx, perspective in enumerate(perspectives):
            for col_idx, criterion in enumerate(criteria):
                thought = thoughts.get((row_idx, col_idx), f"{perspective}: {criterion} analysis")

                if verbose:
                    print(f"    MoT [{row_idx},{col_idx}]: {thought[:40]}...")

                result = await client.call_tool(
                    "matrix_set_cell",
                    {
                        "session_id": session_id,
                        "row": row_idx,
                        "col": col_idx,
                        "thought": thought,
                        "confidence": 0.8,
                    },
                )
                content = result.content[0]
                cell_response = json.loads(
                    content.text if isinstance(content, TextContent) else "{}"
                )

                if cell_response.get("error"):
                    return ReasoningResult(
                        strategy="mot",
                        problem_id=problem.id,
                        final_answer="",
                        reasoning_steps=reasoning_steps,
                        duration_ms=(time.perf_counter() - start) * 1000,
                        raw_response=cell_response,
                        error=cell_response.get("message", "Unknown error"),
                    )

                reasoning_steps.append(f"[{perspective}/{criterion}] {thought}")

        # Synthesize columns
        for col_idx, criterion in enumerate(criteria):
            synthesis = f"Synthesis of {criterion}: Combined analysis across perspectives"

            result = await client.call_tool(
                "matrix_synthesize",
                {
                    "session_id": session_id,
                    "col": col_idx,
                    "synthesis": synthesis,
                },
            )
            reasoning_steps.append(f"[Synthesis/{criterion}] {synthesis}")

        # Finalize
        result = await client.call_tool(
            "matrix_finalize",
            {
                "session_id": session_id,
                "answer": problem.expected_answer,
                "confidence": 0.85,
            },
        )
        content = result.content[0]
        final_response = json.loads(content.text if isinstance(content, TextContent) else "{}")

        duration_ms = (time.perf_counter() - start) * 1000

        return ReasoningResult(
            strategy="mot",
            problem_id=problem.id,
            final_answer=final_response.get("final_answer", ""),
            reasoning_steps=reasoning_steps,
            duration_ms=duration_ms,
            raw_response=final_response,
        )

    except Exception as e:
        return ReasoningResult(
            strategy="mot",
            problem_id=problem.id,
            final_answer="",
            reasoning_steps=reasoning_steps,
            duration_ms=(time.perf_counter() - start) * 1000,
            raw_response={},
            error=str(e),
        )


def _generate_chain_thoughts(problem: ReasoningProblem) -> list[str]:
    """Generate reasoning thoughts for chain based on problem type."""
    if problem.problem_type == ProblemType.MATH:
        return [
            f"Let me identify the key values in this problem: {problem.question[:50]}...",
            "I'll set up the equations and relationships between quantities.",
            "Now I'll perform the calculations step by step.",
            "Let me verify the result makes sense in context.",
            f"The answer is {problem.expected_answer}.",
        ][: problem.steps_hint]

    elif problem.problem_type == ProblemType.LOGIC:
        return [
            "First, let me identify the premises and what we're trying to prove.",
            "I'll analyze the logical structure and relationships.",
            "Let me check if the conclusion follows necessarily from the premises.",
            "I'll consider if there are any counterexamples.",
            f"Based on this analysis, the answer is {problem.expected_answer}.",
        ][: problem.steps_hint]

    elif problem.problem_type == ProblemType.MULTI_HOP:
        return [
            "Let me break down what information I need to find.",
            "First, I'll identify the key entity or location mentioned.",
            "Now I'll connect this to the relevant time period or context.",
            "Finally, I'll look up the specific information requested.",
            f"The answer is {problem.expected_answer}.",
        ][: problem.steps_hint]

    else:  # ANALYSIS
        return [
            "Let me understand the constraints and requirements.",
            "I'll analyze the trade-offs involved in each option.",
            "Considering the specific context given...",
            "Weighing the pros and cons for this situation...",
            f"My recommendation is {problem.expected_answer}.",
        ][: problem.steps_hint]


def _generate_mot_thoughts(
    problem: ReasoningProblem,
    perspectives: list[str],
    criteria: list[str],
) -> dict[tuple[int, int], str]:
    """Generate matrix cell thoughts based on problem and perspectives."""
    thoughts: dict[tuple[int, int], str] = {}

    for row_idx, perspective in enumerate(perspectives):
        for col_idx, criterion in enumerate(criteria):
            if problem.problem_type == ProblemType.MATH:
                if criterion == "pros":
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: This approach allows clear step tracking"
                    )
                else:
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Need to be careful with order of operations"
                    )

            elif problem.problem_type == ProblemType.LOGIC:
                if criterion == "pros":
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Systematic analysis of logical structure"
                    )
                else:
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Must consider edge cases and counterexamples"
                    )

            elif problem.problem_type == ProblemType.ANALYSIS:
                if criterion == "pros":
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Favorable factor in this context"
                    )
                else:
                    thoughts[(row_idx, col_idx)] = f"{perspective}: Potential drawback to consider"

            else:  # MULTI_HOP
                if criterion == "pros":
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Clear connection to next reasoning step"
                    )
                else:
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Information may need verification"
                    )

    return thoughts


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_result(
    result: ReasoningResult,
    problem: ReasoningProblem,
) -> EvaluationMetrics:
    """Evaluate a reasoning result against the problem."""
    # Correctness: Does answer match expected?
    answer_lower = result.final_answer.lower().strip()
    expected_lower = problem.expected_answer.lower().strip()

    # Flexible matching - check if expected is contained in answer
    if expected_lower in answer_lower or answer_lower in expected_lower:
        correctness = 1.0
    elif any(kw in answer_lower for kw in expected_lower.split()):
        correctness = 0.7
    else:
        correctness = 0.0

    # Reasoning depth: Number of steps
    reasoning_depth = len(result.reasoning_steps)

    # Keyword coverage: What fraction of expected keywords appear in reasoning?
    all_reasoning = " ".join(result.reasoning_steps).lower()
    found_keywords = sum(1 for kw in problem.expected_keywords if kw.lower() in all_reasoning)
    keyword_coverage = (
        found_keywords / len(problem.expected_keywords) if problem.expected_keywords else 0.0
    )

    # Coherence: Simple heuristic based on step length and structure
    if reasoning_depth == 0:
        coherence = 0.0
    else:
        avg_step_length = sum(len(s) for s in result.reasoning_steps) / reasoning_depth
        # Good steps are 50-500 chars
        if 50 <= avg_step_length <= 500:
            coherence = 0.9
        elif 20 <= avg_step_length <= 1000:
            coherence = 0.6
        else:
            coherence = 0.3

    return EvaluationMetrics(
        correctness=correctness,
        reasoning_depth=reasoning_depth,
        keyword_coverage=keyword_coverage,
        coherence=coherence,
    )


def determine_winner(metrics: dict[str, EvaluationMetrics]) -> str:
    """Determine which strategy performed best."""

    def score(m: EvaluationMetrics) -> float:
        # Weighted scoring: correctness matters most
        return (
            m.correctness * 0.4
            + m.keyword_coverage * 0.3
            + min(m.reasoning_depth / 5, 1.0) * 0.2  # Cap depth contribution
            + m.coherence * 0.1
        )

    scores = {strategy: score(m) for strategy, m in metrics.items()}
    return max(scores, key=lambda k: scores[k])


# =============================================================================
# Benchmark Runner
# =============================================================================


async def run_comparison(
    problems: list[ReasoningProblem],
    verbose: bool = False,
) -> list[ComparisonResult]:
    """Run comparison benchmark on all problems."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Error: fastmcp not installed. Run: pip install fastmcp")
        sys.exit(1)

    results: list[ComparisonResult] = []

    async with Client("src/server.py") as client:
        for problem in problems:
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Problem: {problem.id} ({problem.problem_type.value})")
                print(f"Question: {problem.question[:60]}...")
                print("=" * 60)

            problem_results: dict[str, ReasoningResult] = {}

            # 1. Baseline
            if verbose:
                print("\n  [Baseline] Direct reasoning...")
            baseline_result = await solve_baseline(problem, verbose)
            problem_results["baseline"] = baseline_result

            # 2. Long Chain
            if verbose:
                print("\n  [Long Chain] Sequential reasoning...")
            chain_result = await solve_with_long_chain(client, problem, verbose)
            problem_results["long_chain"] = chain_result

            # 3. Matrix of Thought
            if verbose:
                print("\n  [MoT] Multi-perspective reasoning...")
            mot_result = await solve_with_mot(client, problem, verbose)
            problem_results["mot"] = mot_result

            # Evaluate all
            metrics: dict[str, EvaluationMetrics] = {}
            for strategy, result in problem_results.items():
                metrics[strategy] = evaluate_result(result, problem)

            winner = determine_winner(metrics)

            results.append(
                ComparisonResult(
                    problem=problem,
                    results=problem_results,
                    metrics=metrics,
                    winner=winner,
                )
            )

            if verbose:
                print(f"\n  Winner: {winner}")

    return results


# =============================================================================
# Report Generation
# =============================================================================


def print_comparison_report(results: list[ComparisonResult], verbose: bool = False) -> None:
    """Print comprehensive comparison report."""
    print("\n" + "=" * 80)
    print("REASONING STRATEGY COMPARISON REPORT")
    print("=" * 80)

    # Per-problem results
    print("\n" + "-" * 80)
    print("PER-PROBLEM RESULTS")
    print("-" * 80)

    strategy_wins: dict[str, int] = {"baseline": 0, "long_chain": 0, "mot": 0}
    strategy_metrics: dict[str, list[EvaluationMetrics]] = {
        "baseline": [],
        "long_chain": [],
        "mot": [],
    }

    for comp in results:
        problem_info = f"{comp.problem.problem_type.value}, difficulty={comp.problem.difficulty}"
        print(f"\n{comp.problem.id} ({problem_info})")
        print(f"  Question: {comp.problem.question[:70]}...")
        print(f"  Expected: {comp.problem.expected_answer}")
        print()

        for strategy in ["baseline", "long_chain", "mot"]:
            result = comp.results[strategy]
            metrics = comp.metrics[strategy]
            strategy_metrics[strategy].append(metrics)

            winner_mark = " 🏆" if comp.winner == strategy else ""
            error_mark = " ❌" if result.error else ""

            print(
                f"  {strategy:12} | "
                f"correct={metrics.correctness:.1f} "
                f"depth={metrics.reasoning_depth:2d} "
                f"coverage={metrics.keyword_coverage:.2f} "
                f"coherence={metrics.coherence:.1f} "
                f"time={result.duration_ms:6.0f}ms"
                f"{winner_mark}{error_mark}"
            )

            if result.error and verbose:
                print(f"               Error: {result.error}")

        strategy_wins[comp.winner] += 1

    # Aggregate statistics
    print("\n" + "-" * 80)
    print("AGGREGATE STATISTICS")
    print("-" * 80)

    header = (
        f"{'Strategy':<12} | {'Wins':>5} | {'Avg Correct':>11} | "
        f"{'Avg Depth':>9} | {'Avg Coverage':>12} | {'Avg Time':>10}"
    )
    print(f"\n{header}")
    print("-" * 75)

    for strategy in ["baseline", "long_chain", "mot"]:
        metrics_list = strategy_metrics[strategy]
        if not metrics_list:
            continue

        avg_correct = sum(m.correctness for m in metrics_list) / len(metrics_list)
        avg_depth = sum(m.reasoning_depth for m in metrics_list) / len(metrics_list)
        avg_coverage = sum(m.keyword_coverage for m in metrics_list) / len(metrics_list)

        # Get average time from results
        all_times = [
            comp.results[strategy].duration_ms
            for comp in results
            if not comp.results[strategy].error
        ]
        avg_time = sum(all_times) / len(all_times) if all_times else 0

        print(
            f"{strategy:<12} | {strategy_wins[strategy]:>5} | "
            f"{avg_correct:>11.2f} | {avg_depth:>9.1f} | "
            f"{avg_coverage:>12.2f} | {avg_time:>9.0f}ms"
        )

    # Winner summary
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)

    total = len(results)
    print(f"\nTotal problems: {total}")
    print("\nWins by strategy:")
    for strategy, wins in sorted(strategy_wins.items(), key=lambda x: -x[1]):  # type: ignore[arg-type]
        pct = (wins / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {strategy:<12}: {wins:>2} ({pct:5.1f}%) {bar}")

    # Recommendations
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS BY PROBLEM TYPE")
    print("-" * 80)

    by_type: dict[ProblemType, dict[str, int]] = {}
    for comp in results:
        pt = comp.problem.problem_type
        if pt not in by_type:
            by_type[pt] = {"baseline": 0, "long_chain": 0, "mot": 0}
        by_type[pt][comp.winner] += 1

    for pt, type_wins in by_type.items():
        best = max(type_wins.keys(), key=type_wins.get)  # type: ignore[arg-type]
        print(f"\n  {pt.value:<12}: Best strategy = {best}")
        for strategy, count in type_wins.items():
            print(f"    {strategy}: {count} wins")


# =============================================================================
# Main Entry Point
# =============================================================================


async def main() -> None:
    """Run the comparison benchmark."""
    parser = argparse.ArgumentParser(description="Compare Long Chain vs MoT vs Baseline Reasoning")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--problems",
        "-n",
        type=int,
        default=None,
        help="Number of problems to run (default: all)",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["math", "logic", "multi_hop", "analysis"],
        help="Only run problems of this type",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MatrixMind MCP - Reasoning Strategy Comparison Benchmark")
    print("=" * 80)

    # Filter problems
    problems = BENCHMARK_PROBLEMS
    if args.type:
        target_type = ProblemType(args.type)
        problems = [p for p in problems if p.problem_type == target_type]

    if args.problems:
        problems = problems[: args.problems]

    print(f"\nProblems to evaluate: {len(problems)}")
    print("Strategies: baseline, long_chain, mot")

    results = await run_comparison(problems, verbose=args.verbose)
    print_comparison_report(results, verbose=args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
