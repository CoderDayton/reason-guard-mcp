#!/usr/bin/env python3
"""Reasoning Quality Benchmark: Baseline vs Think Tool.

Compares answer correctness between:
1. Baseline: Direct LLM answer (no structured reasoning)
2. Think (Chain): Step-by-step reasoning with `think` tool
3. Think (Matrix): Multi-perspective reasoning with `think` tool

Uses challenging problems across categories:
- Math word problems (GSM8K-style)
- Logic puzzles
- Multi-hop reasoning
- Code debugging

Run:
    uv run python examples/reasoning_benchmark.py --llm
    uv run python examples/reasoning_benchmark.py --llm --problems 20
    uv run python examples/reasoning_benchmark.py --llm --category math
    uv run python examples/reasoning_benchmark.py --llm --export results.json

Requires: Local LLM via Ollama or OpenAI-compatible API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llm_client import (
    add_llm_args,
    close_llm_client,
    init_llm_client,
    is_llm_enabled,
)

# =============================================================================
# Problem Definitions
# =============================================================================


class ProblemCategory(Enum):
    MATH = "math"
    LOGIC = "logic"
    MULTIHOP = "multihop"
    CODE = "code"


@dataclass
class Problem:
    """A benchmark problem with expected answer."""

    id: str
    category: ProblemCategory
    question: str
    expected: str  # Expected answer (for exact or fuzzy matching)
    keywords: list[str] = field(default_factory=list)  # Alternative valid answers/keywords
    context: str = ""  # Optional context
    difficulty: str = "medium"  # easy, medium, hard
    steps_hint: int = 3  # Suggested reasoning steps


# GSM8K-style math problems
MATH_PROBLEMS = [
    Problem(
        id="math_1",
        category=ProblemCategory.MATH,
        question="A bakery sold 125 muffins on Monday and 87 muffins on Tuesday. If each muffin costs $3, how much money did they make in total?",
        expected="636",
        keywords=["636", "$636", "636 dollars"],
        difficulty="easy",
        steps_hint=3,
    ),
    Problem(
        id="math_2",
        category=ProblemCategory.MATH,
        question="A train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours. What is the total distance traveled?",
        expected="270",
        keywords=["270", "270 miles"],
        difficulty="easy",
        steps_hint=3,
    ),
    Problem(
        id="math_3",
        category=ProblemCategory.MATH,
        question="In a class of 40 students, 60% are girls. If 5 more boys join the class, what percentage of the class are now girls?",
        expected="53.3",
        keywords=["53.3", "53%", "24/45", "0.533"],
        difficulty="medium",
        steps_hint=4,
    ),
    Problem(
        id="math_4",
        category=ProblemCategory.MATH,
        question="A store buys shirts for $15 each and sells them with a 40% markup. During a sale, they offer 20% off. What is the sale price?",
        expected="16.8",
        keywords=["16.8", "$16.80", "16.80"],
        difficulty="medium",
        steps_hint=4,
    ),
    Problem(
        id="math_5",
        category=ProblemCategory.MATH,
        question="Tom is twice as old as Jerry. In 10 years, Tom will be 1.5 times as old as Jerry. How old is Jerry now?",
        expected="20",
        keywords=["20", "20 years"],
        difficulty="hard",
        steps_hint=5,
    ),
    Problem(
        id="math_6",
        category=ProblemCategory.MATH,
        question="A rectangular pool is 20m long and 12m wide. A 2m wide walkway surrounds the pool. What is the area of just the walkway?",
        expected="144",
        keywords=["144", "144 m²", "144 square meters"],
        difficulty="hard",
        steps_hint=5,
    ),
    # Additional hard problems for statistical significance
    Problem(
        id="math_7",
        category=ProblemCategory.MATH,
        question="A water tank is 3/4 full. After using 30 liters, it becomes 2/3 full. What is the total capacity of the tank?",
        expected="360",
        keywords=["360", "360 liters"],
        difficulty="hard",
        steps_hint=5,
    ),
    Problem(
        id="math_8",
        category=ProblemCategory.MATH,
        question="Two pipes can fill a tank in 6 hours and 8 hours respectively. A drain can empty it in 12 hours. If all three are open, how many hours to fill the tank?",
        expected="4.8",
        keywords=["4.8", "24/5", "4 hours 48 minutes"],
        difficulty="hard",
        steps_hint=5,
    ),
    Problem(
        id="math_9",
        category=ProblemCategory.MATH,
        question="A man walks from A to B at 4 km/h and returns at 6 km/h. What is his average speed for the entire journey?",
        expected="4.8",
        keywords=["4.8", "24/5"],
        difficulty="hard",
        steps_hint=4,
    ),
    Problem(
        id="math_10",
        category=ProblemCategory.MATH,
        question="In a mixture of 60 liters, the ratio of milk to water is 2:1. How much water must be added to make the ratio 1:2?",
        expected="60",
        keywords=["60", "60 liters"],
        difficulty="hard",
        steps_hint=5,
    ),
    Problem(
        id="math_11",
        category=ProblemCategory.MATH,
        question="A boat goes 30 km upstream in 6 hours and 44 km downstream in 4 hours. Find the speed of the stream.",
        expected="3",
        keywords=["3", "3 km/h", "3 kmph"],
        difficulty="hard",
        steps_hint=5,
    ),
    Problem(
        id="math_12",
        category=ProblemCategory.MATH,
        question="The sum of three consecutive even numbers is 66. What is the product of the largest and smallest numbers?",
        expected="440",
        keywords=["440"],
        difficulty="medium",
        steps_hint=4,
    ),
    Problem(
        id="math_13",
        category=ProblemCategory.MATH,
        question="A shopkeeper marks goods 25% above cost price, then offers a 10% discount. What is his profit percentage?",
        expected="12.5",
        keywords=["12.5", "12.5%"],
        difficulty="hard",
        steps_hint=5,
    ),
    Problem(
        id="math_14",
        category=ProblemCategory.MATH,
        question="The compound interest on $10000 for 2 years at 10% per annum is how much more than the simple interest?",
        expected="100",
        keywords=["100", "$100"],
        difficulty="hard",
        steps_hint=5,
    ),
    Problem(
        id="math_15",
        category=ProblemCategory.MATH,
        question="A cone has radius 6 cm and height 8 cm. What is its slant height?",
        expected="10",
        keywords=["10", "10 cm"],
        difficulty="medium",
        steps_hint=3,
    ),
    Problem(
        id="math_16",
        category=ProblemCategory.MATH,
        question="If 8 workers can build a wall in 10 days, how many days would it take 5 workers to build the same wall?",
        expected="16",
        keywords=["16", "16 days"],
        difficulty="medium",
        steps_hint=3,
    ),
]

# Logic puzzles
LOGIC_PROBLEMS = [
    Problem(
        id="logic_1",
        category=ProblemCategory.LOGIC,
        question="A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
        expected="9",
        keywords=["9", "nine"],
        difficulty="easy",
        steps_hint=2,
    ),
    Problem(
        id="logic_2",
        category=ProblemCategory.LOGIC,
        question="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        expected="5",
        keywords=["5", "5 minutes", "five minutes"],
        difficulty="medium",
        steps_hint=3,
    ),
    Problem(
        id="logic_3",
        category=ProblemCategory.LOGIC,
        question="A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        expected="0.05",
        keywords=["0.05", "$0.05", "5 cents", "five cents"],
        difficulty="medium",
        steps_hint=3,
    ),
    Problem(
        id="logic_4",
        category=ProblemCategory.LOGIC,
        question="Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. All labels are wrong. You pick one fruit from the 'Mixed' box and it's an apple. What's actually in each box?",
        expected="Apples has Oranges, Oranges has Mixed, Mixed has Apples",
        keywords=["oranges", "apples", "mixed"],
        difficulty="hard",
        steps_hint=5,
    ),
    # Additional logic problems
    Problem(
        id="logic_5",
        category=ProblemCategory.LOGIC,
        question="You have two ropes. Each takes exactly 1 hour to burn, but they burn unevenly. How do you measure exactly 45 minutes?",
        expected="45",
        keywords=["light both ends", "one rope", "both ends", "45"],
        difficulty="hard",
        steps_hint=4,
    ),
    Problem(
        id="logic_6",
        category=ProblemCategory.LOGIC,
        question="A lily pad doubles in size every day. If it takes 48 days to cover the entire pond, how many days does it take to cover half the pond?",
        expected="47",
        keywords=["47", "47 days"],
        difficulty="hard",
        steps_hint=3,
    ),
    Problem(
        id="logic_7",
        category=ProblemCategory.LOGIC,
        question="I have two coins that add up to 30 cents. One of them is not a nickel. What are the two coins?",
        expected="quarter and nickel",
        keywords=["quarter", "nickel", "25", "5"],
        difficulty="hard",
        steps_hint=3,
    ),
]

# Multi-hop reasoning
MULTIHOP_PROBLEMS = [
    Problem(
        id="multi_1",
        category=ProblemCategory.MULTIHOP,
        question="What is the birth year of the inventor of the telephone?",
        context="Alexander Graham Bell invented the telephone. He was born in Edinburgh, Scotland in 1847 and died in 1922.",
        expected="1847",
        keywords=["1847"],
        difficulty="easy",
        steps_hint=2,
    ),
    Problem(
        id="multi_2",
        category=ProblemCategory.MULTIHOP,
        question="If Project A depends on Project B, and Project B depends on Project C, and Project C takes 3 weeks, Project B takes 2 weeks after C, and Project A takes 1 week after B, what is the minimum total time?",
        expected="6",
        keywords=["6", "6 weeks", "six weeks"],
        difficulty="medium",
        steps_hint=3,
    ),
    Problem(
        id="multi_3",
        category=ProblemCategory.MULTIHOP,
        question="Given: auth.py imports user.py; user.py imports database.py; database.py imports config.py. If config.py has a bug, which module will NOT be affected?",
        context="payment.py imports stripe.py only.",
        expected="payment",
        keywords=["payment", "payment.py", "stripe"],
        difficulty="medium",
        steps_hint=3,
    ),
    # Additional multi-hop problems
    Problem(
        id="multi_4",
        category=ProblemCategory.MULTIHOP,
        question="Alice is taller than Bob. Carol is shorter than Bob. Dave is taller than Alice. Who is the shortest?",
        expected="Carol",
        keywords=["carol"],
        difficulty="medium",
        steps_hint=3,
    ),
    Problem(
        id="multi_5",
        category=ProblemCategory.MULTIHOP,
        question="Service A calls Service B, which calls Service C. Service C has 99% uptime, B has 99.5% uptime, A has 99.9% uptime. What is the approximate combined uptime?",
        expected="98.4",
        keywords=["98.4", "98%", "0.984"],
        difficulty="hard",
        steps_hint=4,
    ),
]

# Code debugging problems
CODE_PROBLEMS = [
    Problem(
        id="code_1",
        category=ProblemCategory.CODE,
        question="""What is wrong with this code?
```python
def get_user_email(users, user_id):
    user = users.get(user_id)
    return user.get('email')
```""",
        expected="None",
        keywords=["None", "NoneType", "AttributeError", "null", "check"],
        difficulty="easy",
        steps_hint=2,
    ),
    Problem(
        id="code_2",
        category=ProblemCategory.CODE,
        question="""This loop should print 1 to 10 but doesn't work correctly. What's the fix?
```python
for i in range(1, 10):
    print(i)
```""",
        expected="11",
        keywords=["11", "range(1, 11)", "inclusive", "exclusive"],
        difficulty="easy",
        steps_hint=2,
    ),
    Problem(
        id="code_3",
        category=ProblemCategory.CODE,
        question="""What security vulnerability exists in this code?
```python
def search_users(name):
    query = f"SELECT * FROM users WHERE name = '{name}'"
    return db.execute(query)
```""",
        expected="SQL injection",
        keywords=["SQL", "injection", "parameterized", "prepared", "sanitize"],
        difficulty="medium",
        steps_hint=3,
    ),
    # Additional code problems
    Problem(
        id="code_4",
        category=ProblemCategory.CODE,
        question="""What's the time complexity of this function?
```python
def find_pair(arr, target):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i] + arr[j] == target:
                return (i, j)
    return None
```""",
        expected="O(n^2)",
        keywords=["O(n^2)", "O(n²)", "n squared", "quadratic", "n^2"],
        difficulty="easy",
        steps_hint=2,
    ),
    Problem(
        id="code_5",
        category=ProblemCategory.CODE,
        question="""What bug exists in this recursive function?
```python
def factorial(n):
    return n * factorial(n - 1)
```""",
        expected="base case",
        keywords=["base case", "recursion", "infinite", "stack overflow", "n == 0", "n == 1"],
        difficulty="medium",
        steps_hint=3,
    ),
    Problem(
        id="code_6",
        category=ProblemCategory.CODE,
        question="""What's wrong with this thread-safe counter implementation?
```python
class Counter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1
```""",
        expected="race condition",
        keywords=["race condition", "thread", "lock", "atomic", "mutex", "not thread-safe"],
        difficulty="hard",
        steps_hint=3,
    ),
]

ALL_PROBLEMS = MATH_PROBLEMS + LOGIC_PROBLEMS + MULTIHOP_PROBLEMS + CODE_PROBLEMS


# =============================================================================
# Result Structures
# =============================================================================


@dataclass
class MethodResult:
    """Result of running a single method on a problem."""

    answer: str
    is_correct: bool
    latency_ms: float
    reasoning: str = ""
    error: str | None = None


@dataclass
class ProblemResult:
    """Results for all methods on a single problem."""

    problem: Problem
    baseline: MethodResult
    think_chain: MethodResult
    think_matrix: MethodResult | None = None  # Optional for speed


@dataclass
class BenchmarkSummary:
    """Overall benchmark summary."""

    total_problems: int
    baseline_correct: int
    chain_correct: int
    matrix_correct: int
    baseline_accuracy: float
    chain_accuracy: float
    matrix_accuracy: float
    chain_improvement: float  # vs baseline
    matrix_improvement: float  # vs baseline
    avg_baseline_latency_ms: float
    avg_chain_latency_ms: float
    avg_matrix_latency_ms: float
    by_category: dict[str, dict[str, float]]
    by_difficulty: dict[str, dict[str, float]]


# =============================================================================
# Answer Checking
# =============================================================================


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Convert to lowercase
    s = answer.lower().strip()
    # Remove common punctuation and formatting
    s = re.sub(r"[,\$%]", "", s)
    # Remove trailing periods
    s = s.rstrip(".")
    # Normalize whitespace
    s = " ".join(s.split())
    return s


def check_answer(response: str, problem: Problem) -> bool:
    """Check if response contains the correct answer."""
    response_norm = normalize_answer(response)
    expected_norm = normalize_answer(problem.expected)

    # Exact match
    if expected_norm in response_norm:
        return True

    # Check keywords
    for keyword in problem.keywords:
        if normalize_answer(keyword) in response_norm:
            return True

    # For numeric answers, try to extract and compare
    try:
        expected_num = float(re.sub(r"[^\d.-]", "", problem.expected))
        response_nums = re.findall(r"-?\d+\.?\d*", response)
        for num_str in response_nums:
            if abs(float(num_str) - expected_num) < 0.01:
                return True
    except (ValueError, TypeError):
        pass

    return False


# =============================================================================
# LLM Helpers
# =============================================================================


async def get_baseline_answer(problem: Problem, llm_client: Any) -> tuple[str, str]:
    """Get direct LLM answer without structured reasoning."""
    prompt = problem.question
    if problem.context:
        prompt = f"Context: {problem.context}\n\nQuestion: {problem.question}"

    system = (
        "Answer the question directly and concisely. "
        "Give only the final answer with brief explanation if needed."
    )

    answer = await llm_client.complete(prompt, system=system, max_tokens=256)
    return answer, ""  # No separate reasoning


async def run_think_chain(
    problem: Problem,
    client: Any,
    llm_client: Any,
) -> tuple[str, str]:
    """Run chain reasoning using the think tool."""

    # Helper to parse response
    def parse_response(result: Any) -> dict[str, Any]:
        from mcp.types import TextContent

        if hasattr(result, "content") and result.content:
            content = result.content[0]
            text = content.text if isinstance(content, TextContent) else "{}"
            return json.loads(text)
        return {}

    reasoning_steps: list[str] = []

    # Start session
    result = await client.call_tool(
        "think",
        {
            "action": "start",
            "mode": "chain",
            "problem": problem.question,
            "context": problem.context if problem.context else None,
        },
    )
    resp = parse_response(result)
    session_id = resp.get("session_id", "")

    if not session_id:
        return "[Error: Failed to start session]", ""

    # Generate reasoning steps
    num_steps = problem.steps_hint
    previous_steps: list[str] = []

    for step_num in range(1, num_steps + 1):
        # Build step prompt
        if previous_steps:
            history = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(previous_steps))
            step_context = f"Previous reasoning:\n{history}\n\n"
        else:
            step_context = ""

        if step_num == 1:
            instruction = "Start by identifying key information and approach."
        elif step_num == num_steps:
            instruction = "Give your final answer based on the reasoning."
        else:
            instruction = "Continue the reasoning."

        step_prompt = f"""{step_context}Problem: {problem.question}
{f"Context: {problem.context}" if problem.context else ""}

{instruction} (Step {step_num}/{num_steps})"""

        system = "You are reasoning step by step. Be specific and show your work."
        step_content = await llm_client.complete(step_prompt, system=system, max_tokens=200)

        # Add step to think tool
        await client.call_tool(
            "think",
            {"action": "continue", "session_id": session_id, "thought": step_content},
        )

        previous_steps.append(step_content)
        reasoning_steps.append(step_content)

    # Finish and get final answer
    final_prompt = f"""Based on this reasoning:
{chr(10).join(f"Step {i + 1}: {s}" for i, s in enumerate(reasoning_steps))}

Give the final answer for: {problem.question}
Answer concisely with just the answer value."""

    system = "Give only the final answer based on the reasoning."
    final_answer = await llm_client.complete(final_prompt, system=system, max_tokens=64)

    await client.call_tool(
        "think",
        {
            "action": "finish",
            "session_id": session_id,
            "thought": final_answer,
            "confidence": 0.9,
        },
    )

    full_reasoning = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(reasoning_steps))
    return final_answer, full_reasoning


async def run_think_matrix(
    problem: Problem,
    client: Any,
    llm_client: Any,
) -> tuple[str, str]:
    """Run matrix reasoning using the think tool."""

    def parse_response(result: Any) -> dict[str, Any]:
        from mcp.types import TextContent

        if hasattr(result, "content") and result.content:
            content = result.content[0]
            text = content.text if isinstance(content, TextContent) else "{}"
            return json.loads(text)
        return {}

    perspectives = ["analytical", "intuitive"]
    criteria = ["accuracy", "completeness"]

    # Start matrix session
    result = await client.call_tool(
        "think",
        {
            "action": "start",
            "mode": "matrix",
            "problem": problem.question,
            "context": problem.context if problem.context else None,
            "rows": len(perspectives),
            "cols": len(criteria),
        },
    )
    resp = parse_response(result)
    session_id = resp.get("session_id", "")

    if not session_id:
        return "[Error: Failed to start matrix session]", ""

    all_cells: list[str] = []

    # Fill matrix cells
    for row, perspective in enumerate(perspectives):
        for col, criterion in enumerate(criteria):
            cell_prompt = f"""Problem: {problem.question}
{f"Context: {problem.context}" if problem.context else ""}

Analyze from a {perspective} perspective, focusing on {criterion}.
Give a brief analysis (1-2 sentences)."""

            system = f"Analyze as a {perspective} thinker focusing on {criterion}."
            cell_content = await llm_client.complete(cell_prompt, system=system, max_tokens=100)

            await client.call_tool(
                "think",
                {
                    "action": "continue",
                    "session_id": session_id,
                    "row": row,
                    "col": col,
                    "thought": cell_content,
                },
            )
            all_cells.append(f"[{perspective}×{criterion}]: {cell_content}")

    # Synthesize
    for col, criterion in enumerate(criteria):
        synth_prompt = f"Synthesize the {criterion} analysis for: {problem.question}"
        synth = await llm_client.complete(
            synth_prompt, system="Synthesize insights.", max_tokens=80
        )
        await client.call_tool(
            "think",
            {"action": "synthesize", "session_id": session_id, "col": col, "thought": synth},
        )

    # Final answer
    final_prompt = f"""Based on multi-perspective analysis:
{chr(10).join(all_cells)}

Give the final answer for: {problem.question}
Answer concisely with just the answer value."""

    final_answer = await llm_client.complete(
        final_prompt, system="Give final answer.", max_tokens=64
    )

    await client.call_tool(
        "think",
        {"action": "finish", "session_id": session_id, "thought": final_answer, "confidence": 0.85},
    )

    return final_answer, "\n".join(all_cells)


# =============================================================================
# Benchmark Runner
# =============================================================================


async def run_benchmark(
    problems: list[Problem],
    include_matrix: bool = False,
    verbose: bool = False,
) -> tuple[list[ProblemResult], BenchmarkSummary]:
    """Run the full benchmark."""
    from llm_client import get_llm_client

    try:
        from fastmcp import Client
    except ImportError:
        print("Error: fastmcp not installed")
        sys.exit(1)

    llm = get_llm_client()
    if llm is None:
        print("Error: LLM client not initialized")
        sys.exit(1)

    results: list[ProblemResult] = []

    async with Client("src/server.py") as mcp_client:
        for i, problem in enumerate(problems):
            print(f"\n[{i + 1}/{len(problems)}] {problem.id} ({problem.category.value})")
            print(f"    Q: {problem.question[:60]}...")

            # Baseline
            start = time.perf_counter()
            try:
                baseline_answer, _ = await get_baseline_answer(problem, llm)
                baseline_correct = check_answer(baseline_answer, problem)
                baseline_result = MethodResult(
                    answer=baseline_answer,
                    is_correct=baseline_correct,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            except Exception as e:
                baseline_result = MethodResult(
                    answer="",
                    is_correct=False,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    error=str(e),
                )

            # Think Chain
            start = time.perf_counter()
            try:
                chain_answer, chain_reasoning = await run_think_chain(problem, mcp_client, llm)
                chain_correct = check_answer(chain_answer, problem)
                chain_result = MethodResult(
                    answer=chain_answer,
                    is_correct=chain_correct,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    reasoning=chain_reasoning,
                )
            except Exception as e:
                chain_result = MethodResult(
                    answer="",
                    is_correct=False,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    error=str(e),
                )

            # Think Matrix (optional)
            matrix_result = None
            if include_matrix:
                start = time.perf_counter()
                try:
                    matrix_answer, matrix_reasoning = await run_think_matrix(
                        problem, mcp_client, llm
                    )
                    matrix_correct = check_answer(matrix_answer, problem)
                    matrix_result = MethodResult(
                        answer=matrix_answer,
                        is_correct=matrix_correct,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        reasoning=matrix_reasoning,
                    )
                except Exception as e:
                    matrix_result = MethodResult(
                        answer="",
                        is_correct=False,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        error=str(e),
                    )

            results.append(
                ProblemResult(
                    problem=problem,
                    baseline=baseline_result,
                    think_chain=chain_result,
                    think_matrix=matrix_result,
                )
            )

            # Print result
            b_mark = "✓" if baseline_result.is_correct else "✗"
            c_mark = "✓" if chain_result.is_correct else "✗"
            print(f"    Baseline: {b_mark} | Chain: {c_mark}", end="")
            if matrix_result:
                m_mark = "✓" if matrix_result.is_correct else "✗"
                print(f" | Matrix: {m_mark}", end="")
            print(f" | Expected: {problem.expected}")

            if verbose:
                print(f"    Baseline answer: {baseline_result.answer[:50]}...")
                print(f"    Chain answer: {chain_result.answer[:50]}...")

    # Compute summary
    summary = compute_summary(results, include_matrix)
    return results, summary


def compute_summary(results: list[ProblemResult], include_matrix: bool) -> BenchmarkSummary:
    """Compute benchmark summary statistics."""
    total = len(results)
    baseline_correct = sum(1 for r in results if r.baseline.is_correct)
    chain_correct = sum(1 for r in results if r.think_chain.is_correct)
    matrix_correct = sum(1 for r in results if r.think_matrix and r.think_matrix.is_correct)

    baseline_acc = baseline_correct / total * 100 if total > 0 else 0
    chain_acc = chain_correct / total * 100 if total > 0 else 0
    matrix_acc = matrix_correct / total * 100 if total > 0 and include_matrix else 0

    # By category
    by_category: dict[str, dict[str, float]] = {}
    for cat in ProblemCategory:
        cat_results = [r for r in results if r.problem.category == cat]
        if cat_results:
            cat_total = len(cat_results)
            by_category[cat.value] = {
                "total": cat_total,
                "baseline": sum(1 for r in cat_results if r.baseline.is_correct) / cat_total * 100,
                "chain": sum(1 for r in cat_results if r.think_chain.is_correct) / cat_total * 100,
            }

    # By difficulty
    by_difficulty: dict[str, dict[str, float]] = {}
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r.problem.difficulty == diff]
        if diff_results:
            diff_total = len(diff_results)
            by_difficulty[diff] = {
                "total": diff_total,
                "baseline": sum(1 for r in diff_results if r.baseline.is_correct)
                / diff_total
                * 100,
                "chain": sum(1 for r in diff_results if r.think_chain.is_correct)
                / diff_total
                * 100,
            }

    # Latencies
    baseline_latencies = [r.baseline.latency_ms for r in results]
    chain_latencies = [r.think_chain.latency_ms for r in results]
    matrix_latencies = [r.think_matrix.latency_ms for r in results if r.think_matrix]

    return BenchmarkSummary(
        total_problems=total,
        baseline_correct=baseline_correct,
        chain_correct=chain_correct,
        matrix_correct=matrix_correct,
        baseline_accuracy=baseline_acc,
        chain_accuracy=chain_acc,
        matrix_accuracy=matrix_acc,
        chain_improvement=chain_acc - baseline_acc,
        matrix_improvement=matrix_acc - baseline_acc if include_matrix else 0,
        avg_baseline_latency_ms=statistics.mean(baseline_latencies) if baseline_latencies else 0,
        avg_chain_latency_ms=statistics.mean(chain_latencies) if chain_latencies else 0,
        avg_matrix_latency_ms=statistics.mean(matrix_latencies) if matrix_latencies else 0,
        by_category=by_category,
        by_difficulty=by_difficulty,
    )


# =============================================================================
# Report Generation
# =============================================================================


def print_report(summary: BenchmarkSummary, include_matrix: bool) -> None:
    """Print benchmark report."""
    W = 70

    print("\n" + "═" * W)
    print(" REASONING QUALITY BENCHMARK RESULTS ".center(W))
    print("═" * W)

    # Overall accuracy
    print("\n ACCURACY ".center(W, "─"))
    print(f"  {'Method':<20} {'Correct':<12} {'Accuracy':<12} {'vs Baseline':<12}")
    print("  " + "-" * 56)
    print(
        f"  {'Baseline':<20} {summary.baseline_correct}/{summary.total_problems:<10} {summary.baseline_accuracy:>6.1f}%{'':>6} —"
    )
    chain_delta = (
        f"+{summary.chain_improvement:.1f}%"
        if summary.chain_improvement > 0
        else f"{summary.chain_improvement:.1f}%"
    )
    print(
        f"  {'Think (Chain)':<20} {summary.chain_correct}/{summary.total_problems:<10} {summary.chain_accuracy:>6.1f}%{'':>6} {chain_delta}"
    )
    if include_matrix:
        matrix_delta = (
            f"+{summary.matrix_improvement:.1f}%"
            if summary.matrix_improvement > 0
            else f"{summary.matrix_improvement:.1f}%"
        )
        print(
            f"  {'Think (Matrix)':<20} {summary.matrix_correct}/{summary.total_problems:<10} {summary.matrix_accuracy:>6.1f}%{'':>6} {matrix_delta}"
        )

    # By category
    print("\n BY CATEGORY ".center(W, "─"))
    print(f"  {'Category':<15} {'Baseline':<15} {'Chain':<15} {'Improvement':<15}")
    print("  " + "-" * 56)
    for cat, stats in summary.by_category.items():
        imp = stats["chain"] - stats["baseline"]
        imp_str = f"+{imp:.1f}%" if imp > 0 else f"{imp:.1f}%"
        print(
            f"  {cat:<15} {stats['baseline']:>6.1f}%{'':>8} {stats['chain']:>6.1f}%{'':>8} {imp_str}"
        )

    # By difficulty
    print("\n BY DIFFICULTY ".center(W, "─"))
    print(f"  {'Difficulty':<15} {'Baseline':<15} {'Chain':<15} {'Improvement':<15}")
    print("  " + "-" * 56)
    for diff, stats in summary.by_difficulty.items():
        imp = stats["chain"] - stats["baseline"]
        imp_str = f"+{imp:.1f}%" if imp > 0 else f"{imp:.1f}%"
        print(
            f"  {diff:<15} {stats['baseline']:>6.1f}%{'':>8} {stats['chain']:>6.1f}%{'':>8} {imp_str}"
        )

    # Latency
    print("\n LATENCY ".center(W, "─"))
    print(f"  Baseline avg:     {summary.avg_baseline_latency_ms:>8.0f} ms")
    print(f"  Chain avg:        {summary.avg_chain_latency_ms:>8.0f} ms")
    if include_matrix:
        print(f"  Matrix avg:       {summary.avg_matrix_latency_ms:>8.0f} ms")

    # Verdict
    print("\n" + "═" * W)
    if summary.chain_improvement > 0:
        print(f" ✓ Think (Chain) improved accuracy by {summary.chain_improvement:.1f}% ".center(W))
    elif summary.chain_improvement < 0:
        print(
            f" ✗ Think (Chain) decreased accuracy by {abs(summary.chain_improvement):.1f}% ".center(
                W
            )
        )
    else:
        print(" = Think (Chain) had same accuracy as baseline ".center(W))
    print("═" * W + "\n")


def export_results(
    results: list[ProblemResult],
    summary: BenchmarkSummary,
    filepath: str,
) -> None:
    """Export results to JSON."""
    export_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_problems": summary.total_problems,
            "baseline_accuracy": summary.baseline_accuracy,
            "chain_accuracy": summary.chain_accuracy,
            "matrix_accuracy": summary.matrix_accuracy,
            "chain_improvement": summary.chain_improvement,
            "matrix_improvement": summary.matrix_improvement,
            "by_category": summary.by_category,
            "by_difficulty": summary.by_difficulty,
        },
        "problems": [
            {
                "id": r.problem.id,
                "category": r.problem.category.value,
                "difficulty": r.problem.difficulty,
                "question": r.problem.question[:100],
                "expected": r.problem.expected,
                "baseline": {
                    "answer": r.baseline.answer,
                    "correct": r.baseline.is_correct,
                    "latency_ms": r.baseline.latency_ms,
                },
                "chain": {
                    "answer": r.think_chain.answer,
                    "correct": r.think_chain.is_correct,
                    "latency_ms": r.think_chain.latency_ms,
                },
            }
            for r in results
        ],
    }

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"\nResults exported to: {filepath}")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run the reasoning quality benchmark."""
    parser = argparse.ArgumentParser(
        description="Reasoning Quality Benchmark: Baseline vs Think Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--problems",
        "-n",
        type=int,
        default=10,
        help="Number of problems to run (default: 10)",
    )
    parser.add_argument(
        "--category",
        "-c",
        choices=["math", "logic", "multihop", "code", "all"],
        default="all",
        help="Problem category (default: all)",
    )
    parser.add_argument(
        "--matrix",
        "-m",
        action="store_true",
        help="Include Matrix of Thought (slower)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed answers",
    )
    parser.add_argument(
        "--export",
        "-e",
        type=str,
        metavar="FILE",
        help="Export results to JSON",
    )
    add_llm_args(parser)
    args = parser.parse_args()

    if not args.llm:
        print("Error: This benchmark requires an LLM. Use --llm flag.")
        print("Example: uv run python examples/reasoning_benchmark.py --llm")
        sys.exit(1)

    print("═" * 70)
    print(" REASONING QUALITY BENCHMARK ".center(70))
    print(" Baseline vs Think Tool ".center(70))
    print("═" * 70)

    # Initialize LLM
    print(f"\nInitializing LLM: {args.llm_model or 'default'} at {args.llm_url or 'default'}")
    await init_llm_client(base_url=args.llm_url, model=args.llm_model)

    if not is_llm_enabled():
        print("Error: Failed to initialize LLM client")
        sys.exit(1)

    # Select problems
    if args.category == "all":
        problems = ALL_PROBLEMS[: args.problems]
    else:
        cat = ProblemCategory(args.category)
        cat_problems = [p for p in ALL_PROBLEMS if p.category == cat]
        problems = cat_problems[: args.problems]

    print(f"Running {len(problems)} problems (category: {args.category})")
    if args.matrix:
        print("Including Matrix of Thought (this will be slower)")

    try:
        results, summary = await run_benchmark(
            problems,
            include_matrix=args.matrix,
            verbose=args.verbose,
        )

        print_report(summary, include_matrix=args.matrix)

        if args.export:
            export_results(results, summary, args.export)

    finally:
        await close_llm_client()


if __name__ == "__main__":
    asyncio.run(main())
