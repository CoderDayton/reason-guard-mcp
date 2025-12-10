"""Benchmark for MatrixMind MCP Server (State Manager Architecture).

This benchmark tests the MCP tool API contract against a live server.
Tools are now state managers - the calling LLM does all reasoning.

Tools covered:
1. compress_prompt - Context compression with token reduction
2. recommend_reasoning_strategy - Strategy selection (stateless)
3. chain_start/chain_add_step/chain_finalize - Long chain state management
4. matrix_start/matrix_set_cell/matrix_synthesize/matrix_finalize - MoT state management
5. verify_start/verify_add_claim/verify_claim/verify_finalize - Verification state management

Run:
    python examples/benchmark.py [--verbose]

Requires: Running MCP server (tests against live server via fastmcp Client)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkCase:
    """A benchmark test case with verifiable expected outcomes."""

    name: str
    tool: str
    params: dict[str, Any]
    assertions: list[tuple[str, str, Any]]  # (field_path, operator, expected)
    description: str = ""
    timeout_seconds: float = 30.0


@dataclass
class BenchmarkResult:
    """Result of a single benchmark case."""

    case: BenchmarkCase
    passed: bool
    duration_ms: float
    response: dict[str, Any] | None = None
    error: str | None = None
    assertion_results: list[tuple[str, bool, str]] = field(default_factory=list)


# =============================================================================
# Benchmark Test Cases - Single Tool Calls
# =============================================================================

SINGLE_CALL_CASES: list[BenchmarkCase] = [
    # -------------------------------------------------------------------------
    # compress_prompt - Stateless compression
    # -------------------------------------------------------------------------
    BenchmarkCase(
        name="compress_basic",
        tool="compress_prompt",
        params={
            "context": """
            Albert Einstein was born in Ulm, Germany in 1879. He developed the
            theory of special relativity in 1905. His famous equation E=mc²
            emerged from this work. Einstein received the Nobel Prize in 1921.
            He moved to the United States in 1933 and worked at Princeton.
            Einstein died in 1955 in Princeton, New Jersey.
            """
            * 3,
            "question": "When was Einstein born?",
            "compression_ratio": 0.5,
        },
        assertions=[
            ("compressed_context", "contains", "1879"),
            ("compressed_context", "exists", True),
            ("compression_ratio", "<=", 0.7),
            ("original_tokens", ">", 50),
        ],
        description="Compress context while preserving question-relevant info",
    ),
    BenchmarkCase(
        name="compress_preserves_key_facts",
        tool="compress_prompt",
        params={
            "context": """
            The Eiffel Tower is located in Paris, France. It was built in 1889
            for the World's Fair. The tower is 330 meters tall. It was designed
            by Gustave Eiffel. The tower has three levels for visitors.
            Over 7 million people visit the Eiffel Tower each year.
            """
            * 2,
            "question": "How tall is the Eiffel Tower?",
            "compression_ratio": 0.4,
        },
        assertions=[
            ("compressed_context", "contains", "330"),
            ("tokens_saved", ">", 0),
        ],
        description="Compression preserves numeric facts relevant to question",
    ),
    # -------------------------------------------------------------------------
    # recommend_reasoning_strategy - Stateless analysis
    # -------------------------------------------------------------------------
    BenchmarkCase(
        name="recommend_serial_problem",
        tool="recommend_reasoning_strategy",
        params={
            "problem": "Find a path from node A to Z in a directed graph with 50 nodes.",
            "token_budget": 3000,
        },
        assertions=[
            ("recommended_strategy", "==", "long_chain"),
            ("strategy_confidence", ">=", 0.6),
            ("explanation", "exists", True),
        ],
        description="Recommend long_chain for graph traversal",
    ),
    BenchmarkCase(
        name="recommend_parallel_problem",
        tool="recommend_reasoning_strategy",
        params={
            "problem": "What are multiple different approaches to reduce carbon emissions?",
            "token_budget": 5000,
        },
        assertions=[
            ("recommended_strategy", "==", "parallel_voting"),
            ("strategy_confidence", ">=", 0.5),
        ],
        description="Recommend parallel strategy for multi-perspective exploration",
    ),
    BenchmarkCase(
        name="recommend_low_budget",
        tool="recommend_reasoning_strategy",
        params={
            "problem": "Explain quantum entanglement in detail with multiple perspectives.",
            "token_budget": 500,
        },
        assertions=[
            ("estimated_depth_steps", "<=", 10),
            ("explanation", "contains_any", ["budget", "token", "limit", "constraint"]),
        ],
        description="Adapt recommendation to token budget",
    ),
    # -------------------------------------------------------------------------
    # chain_start - Initialize long chain state
    # -------------------------------------------------------------------------
    BenchmarkCase(
        name="chain_start_basic",
        tool="chain_start",
        params={
            "problem": "Calculate: (5 + 3) * 4 - 10",
            "max_steps": 5,
        },
        assertions=[
            ("session_id", "exists", True),
            ("status", "==", "active"),
            ("current_step", "==", 0),
            ("max_steps", "==", 5),
        ],
        description="Initialize chain reasoning session",
    ),
    # -------------------------------------------------------------------------
    # matrix_start - Initialize MoT state
    # -------------------------------------------------------------------------
    BenchmarkCase(
        name="matrix_start_basic",
        tool="matrix_start",
        params={
            "question": "What is the capital of France?",
            "perspectives": ["historical", "cultural", "economic"],
            "criteria": ["accuracy", "relevance"],
        },
        assertions=[
            ("session_id", "exists", True),
            ("status", "==", "active"),
            ("matrix_shape.rows", "==", 3),
            ("matrix_shape.cols", "==", 2),
        ],
        description="Initialize matrix reasoning session",
    ),
    # -------------------------------------------------------------------------
    # verify_start - Initialize verification state
    # -------------------------------------------------------------------------
    BenchmarkCase(
        name="verify_start_basic",
        tool="verify_start",
        params={
            "context": "Albert Einstein was born in 1879 in Ulm, Germany.",
        },
        assertions=[
            ("session_id", "exists", True),
            ("status", "==", "active"),
            ("claims_count", "==", 0),
        ],
        description="Initialize verification session",
    ),
]


# =============================================================================
# Multi-Step Workflow Tests
# =============================================================================


@dataclass
class WorkflowCase:
    """A multi-step workflow test."""

    name: str
    description: str
    steps: list[tuple[str, dict[str, Any]]]  # [(tool, params), ...]
    final_assertions: list[tuple[str, str, Any]]
    timeout_seconds: float = 60.0


WORKFLOW_CASES: list[WorkflowCase] = [
    WorkflowCase(
        name="chain_full_workflow",
        description="Complete long chain reasoning workflow",
        steps=[
            ("chain_start", {"problem": "What is 2 + 2?", "max_steps": 3}),
            # chain_add_step uses session_id from previous
            ("chain_add_step", {"thought": "I need to add 2 and 2", "is_final": False}),
            ("chain_add_step", {"thought": "2 + 2 = 4", "is_final": False}),
            ("chain_finalize", {"final_answer": "4", "confidence": 1.0}),
        ],
        final_assertions=[
            ("status", "==", "completed"),
            ("final_answer", "==", "4"),
            ("total_steps", ">=", 2),
        ],
    ),
    WorkflowCase(
        name="matrix_full_workflow",
        description="Complete matrix of thought workflow",
        steps=[
            (
                "matrix_start",
                {
                    "question": "Is Python good for AI?",
                    "perspectives": ["performance", "ecosystem"],
                    "criteria": ["pros", "cons"],
                },
            ),
            (
                "matrix_set_cell",
                {"row": 0, "col": 0, "content": "Slower than C++ but fast enough for most ML"},
            ),
            (
                "matrix_set_cell",
                {"row": 0, "col": 1, "content": "GIL limits true parallelism"},
            ),
            (
                "matrix_set_cell",
                {"row": 1, "col": 0, "content": "PyTorch, TensorFlow, scikit-learn"},
            ),
            (
                "matrix_set_cell",
                {"row": 1, "col": 1, "content": "Dependency management can be complex"},
            ),
            (
                "matrix_synthesize",
                {"synthesis": "Python is excellent for AI due to ecosystem despite perf tradeoffs"},
            ),
            ("matrix_finalize", {"final_answer": "Yes, Python is great for AI", "confidence": 0.9}),
        ],
        final_assertions=[
            ("status", "==", "completed"),
            ("final_answer", "contains", "Python"),
            ("matrix_complete", "==", True),
        ],
    ),
    WorkflowCase(
        name="verify_full_workflow",
        description="Complete verification workflow",
        steps=[
            ("verify_start", {"context": "The Eiffel Tower is 330 meters tall. Built in 1889."}),
            ("verify_add_claim", {"claim": "The Eiffel Tower is 330 meters tall"}),
            ("verify_add_claim", {"claim": "The Eiffel Tower was built in 1889"}),
            (
                "verify_claim",
                {
                    "claim_index": 0,
                    "verdict": "supported",
                    "evidence": "Context states 330 meters",
                    "confidence": 1.0,
                },
            ),
            (
                "verify_claim",
                {
                    "claim_index": 1,
                    "verdict": "supported",
                    "evidence": "Context states built in 1889",
                    "confidence": 1.0,
                },
            ),
            ("verify_finalize", {}),
        ],
        final_assertions=[
            ("status", "==", "completed"),
            ("summary.total_claims", "==", 2),
            ("summary.supported", "==", 2),
            ("summary.overall_verdict", "==", "fully_supported"),
        ],
    ),
]


# =============================================================================
# Assertion Evaluator
# =============================================================================


def get_nested_value(data: dict[str, Any], path: str) -> Any:
    """Get a nested value from a dict using dot notation."""
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def evaluate_assertion(
    response: dict[str, Any],
    field_path: str,
    operator: str,
    expected: Any,
) -> tuple[bool, str]:
    """Evaluate a single assertion against response data."""
    actual = get_nested_value(response, field_path)

    if operator == "exists":
        passed = (actual is not None) == expected
        return passed, f"{field_path} exists={actual is not None}, expected={expected}"

    if actual is None:
        return False, f"{field_path} is None (not found)"

    if operator == "==":
        passed = actual == expected
        return passed, f"{field_path}={actual}, expected={expected}"

    elif operator == "!=":
        passed = actual != expected
        return passed, f"{field_path}={actual}, expected!={expected}"

    elif operator == ">=":
        passed = actual >= expected
        return passed, f"{field_path}={actual}, expected>={expected}"

    elif operator == "<=":
        passed = actual <= expected
        return passed, f"{field_path}={actual}, expected<={expected}"

    elif operator == ">":
        passed = actual > expected
        return passed, f"{field_path}={actual}, expected>{expected}"

    elif operator == "<":
        passed = actual < expected
        return passed, f"{field_path}={actual}, expected<{expected}"

    elif operator == "contains":
        passed = expected.lower() in str(actual).lower()
        return passed, f"{field_path} contains '{expected}': {passed}"

    elif operator == "contains_any":
        passed = any(e.lower() in str(actual).lower() for e in expected)
        return passed, f"{field_path} contains any of {expected}: {passed}"

    elif operator == "len>=":
        length = len(actual) if hasattr(actual, "__len__") else 0
        passed = length >= expected
        return passed, f"len({field_path})={length}, expected>={expected}"

    elif operator == "len==":
        length = len(actual) if hasattr(actual, "__len__") else 0
        passed = length == expected
        return passed, f"len({field_path})={length}, expected=={expected}"

    else:
        return False, f"Unknown operator: {operator}"


# =============================================================================
# Benchmark Runner
# =============================================================================


async def run_single_case(
    client: Any,
    case: BenchmarkCase,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark case."""
    from mcp.types import TextContent

    if verbose:
        print(f"\n  Running: {case.name}...")

    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            client.call_tool(case.tool, case.params),
            timeout=case.timeout_seconds,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        content = result.content[0]
        response = json.loads(content.text if isinstance(content, TextContent) else "{}")

        if response.get("error"):
            return BenchmarkResult(
                case=case,
                passed=False,
                duration_ms=duration_ms,
                response=response,
                error=response.get("message", "Unknown error"),
            )

        assertion_results: list[tuple[str, bool, str]] = []
        all_passed = True

        for field_path, operator, expected in case.assertions:
            passed, msg = evaluate_assertion(response, field_path, operator, expected)
            assertion_results.append((f"{field_path} {operator} {expected}", passed, msg))
            if not passed:
                all_passed = False

        return BenchmarkResult(
            case=case,
            passed=all_passed,
            duration_ms=duration_ms,
            response=response,
            assertion_results=assertion_results,
        )

    except TimeoutError:
        return BenchmarkResult(
            case=case,
            passed=False,
            duration_ms=case.timeout_seconds * 1000,
            error=f"Timeout after {case.timeout_seconds}s",
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return BenchmarkResult(
            case=case,
            passed=False,
            duration_ms=duration_ms,
            error=str(e),
        )


async def run_workflow(
    client: Any,
    workflow: WorkflowCase,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run a multi-step workflow test."""
    from mcp.types import TextContent

    if verbose:
        print(f"\n  Running workflow: {workflow.name}...")

    start = time.perf_counter()
    session_id: str | None = None
    last_response: dict[str, Any] = {}

    try:
        for i, (tool, params) in enumerate(workflow.steps):
            # Inject session_id for subsequent calls
            if session_id and "session_id" not in params:
                params = {**params, "session_id": session_id}

            if verbose:
                print(f"    Step {i + 1}: {tool}")

            result = await asyncio.wait_for(
                client.call_tool(tool, params),
                timeout=workflow.timeout_seconds / len(workflow.steps),
            )

            content = result.content[0]
            response = json.loads(content.text if isinstance(content, TextContent) else "{}")

            if response.get("error"):
                return BenchmarkResult(
                    case=BenchmarkCase(
                        name=workflow.name,
                        tool=tool,
                        params=params,
                        assertions=[],
                        description=workflow.description,
                    ),
                    passed=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    response=response,
                    error=f"Step {i + 1} ({tool}): {response.get('message', 'Unknown error')}",
                )

            # Capture session_id from first step
            if i == 0 and "session_id" in response:
                session_id = response["session_id"]

            last_response = response

        duration_ms = (time.perf_counter() - start) * 1000

        # Evaluate final assertions
        assertion_results: list[tuple[str, bool, str]] = []
        all_passed = True

        for field_path, operator, expected in workflow.final_assertions:
            passed, msg = evaluate_assertion(last_response, field_path, operator, expected)
            assertion_results.append((f"{field_path} {operator} {expected}", passed, msg))
            if not passed:
                all_passed = False

        return BenchmarkResult(
            case=BenchmarkCase(
                name=workflow.name,
                tool="workflow",
                params={},
                assertions=workflow.final_assertions,
                description=workflow.description,
            ),
            passed=all_passed,
            duration_ms=duration_ms,
            response=last_response,
            assertion_results=assertion_results,
        )

    except TimeoutError:
        return BenchmarkResult(
            case=BenchmarkCase(
                name=workflow.name,
                tool="workflow",
                params={},
                assertions=[],
                description=workflow.description,
            ),
            passed=False,
            duration_ms=workflow.timeout_seconds * 1000,
            error=f"Timeout after {workflow.timeout_seconds}s",
        )
    except Exception as e:
        return BenchmarkResult(
            case=BenchmarkCase(
                name=workflow.name,
                tool="workflow",
                params={},
                assertions=[],
                description=workflow.description,
            ),
            passed=False,
            duration_ms=(time.perf_counter() - start) * 1000,
            error=str(e),
        )


async def run_benchmark(verbose: bool = False) -> list[BenchmarkResult]:
    """Run all benchmark cases against live MCP server."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Error: fastmcp not installed. Run: pip install fastmcp")
        sys.exit(1)

    results: list[BenchmarkResult] = []

    async with Client("src/server.py") as client:
        # Run single-call tests
        print("\nRunning single-call tests...")
        for case in SINGLE_CALL_CASES:
            result = await run_single_case(client, case, verbose)
            results.append(result)

        # Run workflow tests
        print("\nRunning workflow tests...")
        for workflow in WORKFLOW_CASES:
            result = await run_workflow(client, workflow, verbose)
            results.append(result)

    return results


# =============================================================================
# Report Generation
# =============================================================================


def print_report(results: list[BenchmarkResult], verbose: bool = False) -> None:
    """Print benchmark results report."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    by_tool: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        tool = r.case.tool
        if tool not in by_tool:
            by_tool[tool] = []
        by_tool[tool].append(r)

    total_passed = 0
    total_failed = 0
    total_duration = 0.0

    for tool, tool_results in by_tool.items():
        passed = sum(1 for r in tool_results if r.passed)
        failed = len(tool_results) - passed
        total_passed += passed
        total_failed += failed

        print(f"\n{tool}")
        print("-" * 50)

        for r in tool_results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            print(f"  {status} {r.case.name} ({r.duration_ms:.0f}ms)")
            total_duration += r.duration_ms

            if r.error:
                print(f"       Error: {r.error}")

            if verbose or not r.passed:
                for _assertion, passed, msg in r.assertion_results:
                    icon = "✓" if passed else "✗"
                    print(f"       {icon} {msg}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = total_passed + total_failed
    if total > 0:
        print(f"Total:    {total_passed}/{total} passed ({100 * total_passed / total:.1f}%)")
        print(f"Duration: {total_duration:.0f}ms total, {total_duration / total:.0f}ms avg")
    else:
        print("No tests run")

    if total_failed > 0:
        print(f"\n❌ {total_failed} test(s) failed")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")


# =============================================================================
# Main Entry Point
# =============================================================================


async def main() -> None:
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(description="MatrixMind MCP Benchmark")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MatrixMind MCP - State Manager Benchmark")
    print("=" * 70)
    print(f"Single-call tests: {len(SINGLE_CALL_CASES)}")
    print(f"Workflow tests: {len(WORKFLOW_CASES)}")

    results = await run_benchmark(verbose=args.verbose)
    print_report(results, verbose=args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
