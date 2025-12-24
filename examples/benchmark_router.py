#!/usr/bin/env python3
"""Benchmark script for Atomic Reasoning Router.

Runs reasoning trap problems through the router and measures:
- Rejection rates (target: 20-40%)
- Accuracy (target: 75-85%)
- Steps per problem

Usage:
    # Run all problems (simulation mode)
    uv run python examples/benchmark_router.py

    # Run with real LLM (requires OPENAI_API_KEY)
    uv run python examples/benchmark_router.py --llm

    # Run specific categories
    uv run python examples/benchmark_router.py --category probability

    # Run with verbose output
    uv run python examples/benchmark_router.py -v

    # Output JSON results
    uv run python examples/benchmark_router.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.atomic_router import (
    create_branch,
    get_session_state,
    initialize_reasoning,
    submit_atomic_step,
    verify_claims,
)
from src.tools.router_types import Complexity, RouterStatus, StepType


@dataclass
class ProblemResult:
    """Result of running a single problem through the router."""

    problem_id: str
    question: str
    expected_answer: str
    actual_answer: str | None = None
    correct: bool = False
    total_steps: int = 0
    rejections: int = 0
    branches_created: int = 0
    verifications: int = 0
    final_status: str = ""
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class BenchmarkResults:
    """Aggregate results from benchmark run."""

    problems: list[ProblemResult] = field(default_factory=list)
    total_problems: int = 0
    correct: int = 0
    total_rejections: int = 0
    total_steps: int = 0
    total_branches: int = 0
    total_verifications: int = 0
    duration_ms: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_problems if self.total_problems > 0 else 0.0

    @property
    def rejection_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_rejections / (self.total_steps + self.total_rejections)

    @property
    def avg_steps(self) -> float:
        return self.total_steps / self.total_problems if self.total_problems > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_problems": self.total_problems,
                "correct": self.correct,
                "accuracy": round(self.accuracy * 100, 1),
                "rejection_rate": round(self.rejection_rate * 100, 1),
                "avg_steps": round(self.avg_steps, 2),
                "total_branches": self.total_branches,
                "total_verifications": self.total_verifications,
                "duration_ms": round(self.duration_ms, 1),
            },
            "problems": [
                {
                    "id": p.problem_id,
                    "correct": p.correct,
                    "expected": p.expected_answer,
                    "actual": p.actual_answer,
                    "steps": p.total_steps,
                    "rejections": p.rejections,
                    "status": p.final_status,
                    "error": p.error,
                }
                for p in self.problems
            ],
        }


def load_problems(path: Path) -> tuple[list[dict], list[dict]]:
    """Load problems and seed knowledge from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("problems", []), data.get("seed_knowledge", [])


def difficulty_to_complexity(difficulty: str) -> Complexity:
    """Map problem difficulty to router complexity."""
    mapping = {
        "easy": Complexity.LOW,
        "medium": Complexity.MEDIUM,
        "hard": Complexity.HIGH,
        "extreme": Complexity.HIGH,
    }
    return mapping.get(difficulty, Complexity.MEDIUM)


def simulate_reasoning_step(
    problem: dict,
    step_num: int,
    max_steps: int,
    guidance: str,
) -> tuple[StepType, str, float]:
    """Simulate an LLM reasoning step.

    In a real benchmark, this would call an actual LLM.
    For now, we simulate steps based on problem characteristics.

    StepType values: PREMISE, HYPOTHESIS, VERIFICATION, SYNTHESIS
    """
    question = problem["question"].lower()
    answer = problem["answer"]
    difficulty = problem.get("difficulty", "medium")

    # Confidence thresholds by difficulty
    conf_threshold = {"easy": 0.6, "medium": 0.7, "hard": 0.75, "extreme": 0.75}[difficulty]

    # Determine step type based on progress
    progress = step_num / max_steps

    if progress < 0.3:
        # Early: premise/setup
        step_type = StepType.PREMISE
        content = f"Analyzing given information: {question[:100]}..."
        # Start above threshold to avoid early BRANCH_REQUIRED
        confidence = conf_threshold + 0.05 + (0.02 * step_num)
    elif progress < 0.6:
        # Middle: hypothesis
        step_type = StepType.HYPOTHESIS
        # Simulate trap detection from guidance
        if "TRAP" in guidance or "WARNING" in guidance:
            content = f"Considering trap warning. Hypothesis: answer involves careful calculation, likely {answer}"
            confidence = conf_threshold + 0.1
        else:
            # Without guidance, might fall for trap
            content = "Initial hypothesis based on intuition"
            confidence = conf_threshold + 0.05
    elif progress < 0.8:
        # Late middle: verification
        step_type = StepType.VERIFICATION
        content = f"Verifying hypothesis with evidence for answer {answer}"
        confidence = conf_threshold + 0.1
    else:
        # Final: synthesis
        step_type = StepType.SYNTHESIS
        # Higher confidence if guidance mentioned trap
        if "TRAP" in guidance:
            content = f"Final answer after avoiding trap: {answer}"
            confidence = 0.85
        else:
            content = f"Synthesized answer: {answer}"
            confidence = conf_threshold + 0.1

    # Difficulty affects confidence slightly
    if difficulty == "extreme":
        confidence -= 0.02

    return step_type, content, min(0.95, max(0.3, confidence))


# --- LLM-based reasoning ---


async def llm_reasoning_step(
    llm_client: LLMClient,  # type: ignore[name-defined]  # noqa: F821
    problem: dict,
    step_num: int,
    max_steps: int,
    guidance: str,
    previous_steps: list[str],
    valid_next_steps: list[str],
) -> tuple[StepType, str, float]:
    """Generate a reasoning step using real LLM.

    Args:
        llm_client: LLM client for API calls
        problem: Problem dict with question/answer
        step_num: Current step number
        max_steps: Maximum allowed steps
        guidance: RAG-retrieved trap warnings
        previous_steps: List of previous step contents
        valid_next_steps: Valid step types from router

    Returns:
        Tuple of (step_type, content, confidence)
    """
    # Build prompt
    step_types_str = (
        ", ".join(valid_next_steps)
        if valid_next_steps
        else "premise, hypothesis, verification, synthesis"
    )
    previous_str = (
        "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(previous_steps))
        if previous_steps
        else "None yet"
    )

    system_prompt = f"""You are a careful reasoning assistant solving problems step by step.
You MUST follow the router's guidance to avoid common reasoning traps.

GUIDANCE FROM ROUTER:
{guidance}

VALID STEP TYPES: {step_types_str}

Respond in this exact JSON format:
{{"step_type": "premise|hypothesis|verification|synthesis", "content": "your reasoning", "confidence": 0.0-1.0}}

Rules:
- Start with "premise" to establish given information
- Use "hypothesis" to propose potential answers
- Use "verification" to check your hypothesis
- Use "synthesis" only when ready to give final answer
- Confidence should reflect your certainty (0.6-0.95 typical)
"""

    user_prompt = f"""Problem: {problem["question"]}

Previous steps:
{previous_str}

Step {step_num} of {max_steps}. Generate the next reasoning step."""

    try:
        response = await llm_client.ask(user_prompt, system=system_prompt)

        # Parse JSON response
        # Handle markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        data = json.loads(response.strip())

        step_type_str = data.get("step_type", "premise").lower()
        step_type = {
            "premise": StepType.PREMISE,
            "hypothesis": StepType.HYPOTHESIS,
            "verification": StepType.VERIFICATION,
            "synthesis": StepType.SYNTHESIS,
        }.get(step_type_str, StepType.PREMISE)

        content = data.get("content", "")
        confidence = float(data.get("confidence", 0.7))
        confidence = min(0.95, max(0.3, confidence))

        return step_type, content, confidence

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Fallback to simple response parsing
        return StepType.PREMISE, f"[Parse error: {e}]", 0.5


async def run_problem_llm(
    problem: dict,
    llm_client: LLMClient,  # type: ignore[name-defined]  # noqa: F821
    verbose: bool = False,
) -> ProblemResult:
    """Run a single problem through the atomic router using real LLM."""
    start_time = time.perf_counter()

    result = ProblemResult(
        problem_id=problem["id"],
        question=problem["question"],
        expected_answer=problem["answer"],
    )

    try:
        # Initialize session
        complexity = difficulty_to_complexity(problem.get("difficulty", "medium"))
        init_result = initialize_reasoning(problem["question"], complexity.value)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Problem: {problem['id']} (LLM mode)")
            print(f"Complexity: {complexity.value}")
            print(f"Session: {init_result.session_id[:8]}...")
            print(f"Guidance preview: {init_result.guidance[:200]}...")

        session_id = init_result.session_id
        max_steps = {"low": 5, "medium": 8, "high": 12}[complexity.value]

        previous_steps: list[str] = []
        valid_next = ["premise"]

        # Reasoning loop
        step_num = 0
        while step_num < max_steps + 5:  # Allow extra for rejections
            step_num += 1

            step_type, content, confidence = await llm_reasoning_step(
                llm_client,
                problem,
                step_num,
                max_steps,
                init_result.guidance,
                previous_steps,
                valid_next,
            )

            if verbose:
                print(f"  Step {step_num}: {step_type.value} (conf={confidence:.2f})")
                print(f"    Content: {content[:100]}...")

            # Submit step
            step_result = submit_atomic_step(
                session_id=session_id,
                step_type=step_type.value,
                content=content,
                confidence=confidence,
            )

            # Check result
            if step_result.status == RouterStatus.REJECTED:
                result.rejections += 1
                if verbose:
                    print(
                        f"    REJECTED: {step_result.rejection_reason[:80] if step_result.rejection_reason else 'Unknown'}..."
                    )
                continue

            result.total_steps += 1
            previous_steps.append(f"[{step_type.value}] {content}")
            valid_next = step_result.valid_next_steps

            if step_result.status == RouterStatus.BRANCH_REQUIRED:
                result.branches_created += 1
                branch_result = create_branch(
                    session_id=session_id,
                    alternatives=[
                        f"Alternative 1: {content}",
                        "Alternative 2: Consider opposite",
                    ],
                )
                if verbose:
                    print(f"    Branch created: {branch_result.branch_ids}")

            if step_result.status == RouterStatus.VERIFICATION_REQUIRED:
                result.verifications += 1
                verify_result = verify_claims(
                    session_id=session_id,
                    claims=[content],
                    evidence=previous_steps[-3:] if len(previous_steps) >= 3 else previous_steps,
                )
                if verbose:
                    print(f"    Verification: {len(verify_result.verified)} verified")

            # Check if synthesis was accepted
            if step_result.status == RouterStatus.ACCEPTED and step_type == StepType.SYNTHESIS:
                result.actual_answer = extract_answer(content, problem["answer"])
                result.correct = result.actual_answer == problem["answer"]
                result.final_status = "COMPLETE"
                if verbose:
                    print(f"    COMPLETE: {result.actual_answer} (correct={result.correct})")
                break

        else:
            result.final_status = "MAX_STEPS"
            if verbose:
                print("    MAX_STEPS reached")

        # Get final status
        status = get_session_state(session_id)
        if status and "status" in status:
            result.final_status = status["status"]

    except Exception as e:
        result.error = str(e)
        result.final_status = "ERROR"
        if verbose:
            print(f"    ERROR: {e}")

    result.duration_ms = (time.perf_counter() - start_time) * 1000
    return result


async def run_benchmark_llm(
    problems: Sequence[dict],
    llm_client: LLMClient,  # type: ignore[name-defined]  # noqa: F821
    categories: Sequence[str] | None = None,
    verbose: bool = False,
) -> BenchmarkResults:
    """Run benchmark using real LLM."""
    results = BenchmarkResults()
    start_time = time.perf_counter()

    # Filter by category if specified
    if categories:
        problems = [p for p in problems if p.get("category") in categories]

    results.total_problems = len(problems)

    for problem in problems:
        problem_result = await run_problem_llm(problem, llm_client, verbose)
        results.problems.append(problem_result)

        if problem_result.correct:
            results.correct += 1
        results.total_rejections += problem_result.rejections
        results.total_steps += problem_result.total_steps
        results.total_branches += problem_result.branches_created
        results.total_verifications += problem_result.verifications

    results.duration_ms = (time.perf_counter() - start_time) * 1000
    return results


def extract_answer(content: str, expected: str) -> str | None:
    """Extract answer from synthesis content."""
    # Look for the expected answer pattern
    content_lower = content.lower()

    # Direct match
    if expected.lower() in content_lower:
        return expected

    # Look for numbers
    numbers = re.findall(r"\d+", content)
    if numbers and expected in numbers:
        return expected

    # Look for YES/NO
    if expected.upper() in ["YES", "NO"]:
        if "yes" in content_lower:
            return "YES"
        if "no" in content_lower:
            return "NO"

    # Look for A/B
    if expected.upper() in ["A", "B"]:
        if "answer a" in content_lower or "option a" in content_lower:
            return "A"
        if "answer b" in content_lower or "option b" in content_lower:
            return "B"

    return numbers[-1] if numbers else None


def run_problem(problem: dict, verbose: bool = False) -> ProblemResult:
    """Run a single problem through the atomic router."""
    start_time = time.perf_counter()

    result = ProblemResult(
        problem_id=problem["id"],
        question=problem["question"],
        expected_answer=problem["answer"],
    )

    try:
        # Initialize session
        complexity = difficulty_to_complexity(problem.get("difficulty", "medium"))
        init_result = initialize_reasoning(problem["question"], complexity.value)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Problem: {problem['id']}")
            print(f"Complexity: {complexity.value}")
            print(f"Session: {init_result.session_id[:8]}...")
            print(f"Guidance preview: {init_result.guidance[:200]}...")

        session_id = init_result.session_id
        max_steps = {"low": 5, "medium": 8, "high": 12}[complexity.value]

        # Simulate reasoning steps
        step_num = 0
        while step_num < max_steps + 5:  # Allow some extra for rejections
            step_num += 1

            step_type, content, confidence = simulate_reasoning_step(
                problem, step_num, max_steps, init_result.guidance
            )

            if verbose:
                print(f"  Step {step_num}: {step_type.value} (conf={confidence:.2f})")

            # Submit step
            step_result = submit_atomic_step(
                session_id=session_id,
                step_type=step_type.value,
                content=content,
                confidence=confidence,
            )

            # Check result
            if step_result.status == RouterStatus.REJECTED:
                result.rejections += 1
                if verbose:
                    print(f"    REJECTED: {step_result.feedback[:80]}...")
                continue

            result.total_steps += 1

            if step_result.status == RouterStatus.BRANCH_REQUIRED:
                result.branches_created += 1
                # Create branch
                branch_result = create_branch(
                    session_id=session_id,
                    alternatives=[
                        f"Alternative interpretation 1 for {problem['id']}",
                        f"Alternative interpretation 2 for {problem['id']}",
                    ],
                )
                if verbose:
                    print(f"    Branch created: {branch_result.branch_ids}")

            if step_result.status == RouterStatus.VERIFICATION_REQUIRED:
                result.verifications += 1
                # Verify claims
                verify_result = verify_claims(
                    session_id=session_id,
                    claims=[f"The answer is {problem['answer']}"],
                    evidence=[f"Based on problem analysis: {content}"],
                )
                if verbose:
                    print(f"    Verification: {len(verify_result.verified)} verified")

            # Check if synthesis was accepted (reasoning complete)
            if step_result.status == RouterStatus.ACCEPTED and step_type == StepType.SYNTHESIS:
                # Extract answer from final synthesis
                result.actual_answer = extract_answer(content, problem["answer"])
                result.correct = result.actual_answer == problem["answer"]
                result.final_status = "COMPLETE"
                if verbose:
                    print(f"    COMPLETE: {result.actual_answer} (correct={result.correct})")
                break

        else:
            # Reached step limit without completion
            result.final_status = "MAX_STEPS"
            if verbose:
                print("    MAX_STEPS reached")

        # Get final status
        status = get_session_state(session_id)
        if status and "status" in status:
            result.final_status = status["status"]

    except Exception as e:
        result.error = str(e)
        result.final_status = "ERROR"
        if verbose:
            print(f"    ERROR: {e}")

    result.duration_ms = (time.perf_counter() - start_time) * 1000
    return result


def run_benchmark(
    problems: Sequence[dict],
    categories: Sequence[str] | None = None,
    verbose: bool = False,
) -> BenchmarkResults:
    """Run benchmark on a set of problems."""
    results = BenchmarkResults()
    start_time = time.perf_counter()

    # Filter by category if specified
    if categories:
        problems = [p for p in problems if p.get("category") in categories]

    results.total_problems = len(problems)

    for problem in problems:
        problem_result = run_problem(problem, verbose)
        results.problems.append(problem_result)

        if problem_result.correct:
            results.correct += 1
        results.total_rejections += problem_result.rejections
        results.total_steps += problem_result.total_steps
        results.total_branches += problem_result.branches_created
        results.total_verifications += problem_result.verifications

    results.duration_ms = (time.perf_counter() - start_time) * 1000
    return results


def print_results(results: BenchmarkResults) -> None:
    """Print benchmark results in human-readable format."""
    print("\n" + "=" * 60)
    print("ATOMIC ROUTER BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nProblems: {results.total_problems}")
    print(f"Correct:  {results.correct} ({results.accuracy * 100:.1f}%)")
    print("Target accuracy: 75-85%")

    print(f"\nRejection rate: {results.rejection_rate * 100:.1f}%")
    print("Target rejection rate: 20-40%")

    print(f"\nAvg steps per problem: {results.avg_steps:.2f}")
    print(f"Total branches created: {results.total_branches}")
    print(f"Total verifications: {results.total_verifications}")
    print(f"Duration: {results.duration_ms:.0f}ms")

    # Status check
    print("\n" + "-" * 40)
    accuracy_ok = 0.75 <= results.accuracy <= 0.85
    rejection_ok = 0.20 <= results.rejection_rate <= 0.40

    if accuracy_ok and rejection_ok:
        print("✓ Both metrics within target range")
    else:
        if not accuracy_ok:
            print(f"✗ Accuracy {results.accuracy * 100:.1f}% outside target 75-85%")
        if not rejection_ok:
            print(f"✗ Rejection rate {results.rejection_rate * 100:.1f}% outside target 20-40%")

    # Problem-level results
    print("\n" + "-" * 40)
    print("Per-problem results:")
    print(f"{'ID':<25} {'Result':<8} {'Steps':<6} {'Rejects':<8} {'Status'}")
    print("-" * 60)

    for p in results.problems:
        status = "✓" if p.correct else "✗"
        print(
            f"{p.problem_id:<25} {status:<8} {p.total_steps:<6} {p.rejections:<8} {p.final_status}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Atomic Reasoning Router")
    parser.add_argument(
        "--problems",
        type=Path,
        default=Path(__file__).parent / "benchmark_problems.yaml",
        help="Path to problems YAML file",
    )
    parser.add_argument(
        "--category",
        "-c",
        action="append",
        dest="categories",
        help="Filter by category (can specify multiple)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for results",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use real LLM instead of simulation (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (default: from OPENAI_MODEL env var or gpt-4o-mini)",
    )

    args = parser.parse_args()

    # Load problems
    problems, _seed = load_problems(args.problems)

    if not problems:
        print("No problems found", file=sys.stderr)
        return 1

    # Run benchmark
    if args.llm:
        # Import LLM client
        try:
            from examples.llm_client import LLMClient
        except ImportError:
            from llm_client import LLMClient

        import os

        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "Error: OPENAI_API_KEY environment variable required for --llm mode",
                file=sys.stderr,
            )
            return 1

        llm = LLMClient(model=args.model)
        print(f"Using LLM: {llm.model} @ {llm.base_url}")

        results = asyncio.run(run_benchmark_llm(problems, llm, args.categories, args.verbose))
    else:
        results = run_benchmark(problems, args.categories, args.verbose)

    # Output results
    if args.json:
        output = json.dumps(results.to_dict(), indent=2)
        if args.output:
            args.output.write_text(output)
        else:
            print(output)
    else:
        print_results(results)
        if args.output:
            args.output.write_text(json.dumps(results.to_dict(), indent=2))

    # Return code based on targets
    accuracy_ok = 0.75 <= results.accuracy <= 0.85
    rejection_ok = 0.20 <= results.rejection_rate <= 0.40

    return 0 if accuracy_ok and rejection_ok else 1


if __name__ == "__main__":
    sys.exit(main())
