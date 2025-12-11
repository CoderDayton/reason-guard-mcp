"""Performance Benchmark for MatrixMind MCP Server.

Provides actionable metrics for production deployment decisions:
- Latency percentiles (p50, p95, p99)
- Throughput (operations/second)
- Memory footprint
- Compression efficiency
- Session management overhead
- Strategy comparison across problem types

Run:
    python examples/benchmark.py                    # Quick benchmark
    python examples/benchmark.py --full             # Full benchmark suite
    python examples/benchmark.py --export results.json  # Export metrics
    python examples/benchmark.py --compare          # Compare strategies

Requires: Running MCP server (tests against live server via fastmcp Client)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_ITERATIONS = 10
WARMUP_ITERATIONS = 2
FULL_ITERATIONS = 50


class BenchmarkCategory(Enum):
    """Categories of benchmarks."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    COMPRESSION = "compression"
    SESSION = "session"
    STRATEGY = "strategy"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class LatencyStats:
    """Latency statistics in milliseconds."""

    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    samples: int

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> LatencyStats:
        """Compute stats from raw samples."""
        if not samples_ms:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)

        sorted_samples = sorted(samples_ms)
        n = len(sorted_samples)

        return cls(
            min_ms=sorted_samples[0],
            max_ms=sorted_samples[-1],
            mean_ms=statistics.mean(sorted_samples),
            median_ms=statistics.median(sorted_samples),
            p95_ms=sorted_samples[int(n * 0.95)] if n >= 20 else sorted_samples[-1],
            p99_ms=sorted_samples[int(n * 0.99)] if n >= 100 else sorted_samples[-1],
            std_dev_ms=statistics.stdev(sorted_samples) if n > 1 else 0,
            samples=n,
        )

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary."""
        return {
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
            "samples": self.samples,
        }


@dataclass
class ThroughputStats:
    """Throughput statistics."""

    ops_per_second: float
    total_ops: int
    total_duration_s: float
    errors: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary."""
        return {
            "ops_per_second": round(self.ops_per_second, 2),
            "total_ops": self.total_ops,
            "total_duration_s": round(self.total_duration_s, 3),
            "errors": self.errors,
            "success_rate": round((self.total_ops - self.errors) / self.total_ops * 100, 1)
            if self.total_ops > 0
            else 0,
        }


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    peak_mb: float
    current_mb: float
    allocated_blocks: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary."""
        return {
            "peak_mb": round(self.peak_mb, 2),
            "current_mb": round(self.current_mb, 2),
            "allocated_blocks": self.allocated_blocks,
        }


@dataclass
class CompressionStats:
    """Compression efficiency statistics."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    tokens_saved: int
    preservation_score: float  # How well key info was preserved

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary."""
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": round(self.compression_ratio, 3),
            "tokens_saved": self.tokens_saved,
            "tokens_saved_pct": round(self.tokens_saved / self.original_tokens * 100, 1)
            if self.original_tokens > 0
            else 0,
            "preservation_score": round(self.preservation_score, 2),
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    name: str
    category: BenchmarkCategory
    latency: LatencyStats | None = None
    throughput: ThroughputStats | None = None
    memory: MemoryStats | None = None
    compression: CompressionStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        result: dict[str, Any] = {
            "name": self.name,
            "category": self.category.value,
            "passed": self.passed,
        }
        if self.latency:
            result["latency"] = self.latency.to_dict()
        if self.throughput:
            result["throughput"] = self.throughput.to_dict()
        if self.memory:
            result["memory"] = self.memory.to_dict()
        if self.compression:
            result["compression"] = self.compression.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        if self.error:
            result["error"] = self.error
        return result


# =============================================================================
# Benchmark Utilities
# =============================================================================


async def measure_latency(
    fn: Callable[[], Any],
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
) -> tuple[LatencyStats, list[Any]]:
    """Measure latency of an async function over multiple iterations."""
    # Warmup
    for _ in range(warmup):
        with contextlib.suppress(Exception):
            await fn()

    samples: list[float] = []
    results: list[Any] = []

    for _ in range(iterations):
        gc.collect()  # Reduce GC noise
        start = time.perf_counter()
        try:
            result = await fn()
            results.append(result)
        except Exception as e:
            results.append(e)
        elapsed_ms = (time.perf_counter() - start) * 1000
        samples.append(elapsed_ms)

    return LatencyStats.from_samples(samples), results


async def measure_throughput(
    fn: Callable[[], Any],
    duration_seconds: float = 5.0,
) -> ThroughputStats:
    """Measure throughput over a fixed duration."""
    start = time.perf_counter()
    ops = 0
    errors = 0

    while (time.perf_counter() - start) < duration_seconds:
        try:
            await fn()
            ops += 1
        except Exception:
            errors += 1
            ops += 1

    elapsed = time.perf_counter() - start
    return ThroughputStats(
        ops_per_second=ops / elapsed if elapsed > 0 else 0,
        total_ops=ops,
        total_duration_s=elapsed,
        errors=errors,
    )


def measure_memory(fn: Callable[[], Any]) -> tuple[MemoryStats, Any]:
    """Measure memory usage of a synchronous function."""
    gc.collect()
    tracemalloc.start()

    result = fn()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return MemoryStats(
        peak_mb=peak / (1024 * 1024),
        current_mb=current / (1024 * 1024),
        allocated_blocks=0,  # tracemalloc doesn't expose this easily
    ), result


# =============================================================================
# Benchmark Test Cases
# =============================================================================


async def bench_compress_latency(client: Any, iterations: int) -> BenchmarkResult:
    """Benchmark compression latency with varying context sizes."""
    from mcp.types import TextContent

    # Medium-sized context (realistic use case)
    context = (
        """
    Albert Einstein was born in Ulm, Germany in 1879. He developed the
    theory of special relativity in 1905, revolutionizing physics.
    His famous equation E=mc² emerged from this groundbreaking work.
    Einstein received the Nobel Prize in Physics in 1921 for his
    explanation of the photoelectric effect. He moved to the United
    States in 1933 to escape Nazi persecution and worked at Princeton
    University's Institute for Advanced Study until his death in 1955.
    """
        * 5
    )  # ~1000 tokens

    async def compress_call() -> dict[str, Any]:
        result = await client.call_tool(
            "compress_prompt",
            {
                "context": context,
                "question": "When did Einstein receive the Nobel Prize?",
                "compression_ratio": 0.5,
            },
        )
        content = result.content[0]
        return json.loads(content.text if isinstance(content, TextContent) else "{}")

    latency, results = await measure_latency(compress_call, iterations)

    # Calculate compression stats from results
    valid_results = [r for r in results if isinstance(r, dict) and "error" not in r]
    if valid_results:
        avg_original = sum(r.get("original_tokens", 0) for r in valid_results) / len(valid_results)
        avg_compressed = sum(r.get("compressed_tokens", 0) for r in valid_results) / len(
            valid_results
        )
        avg_ratio = sum(r.get("compression_ratio", 1) for r in valid_results) / len(valid_results)

        # Check if key fact preserved
        preserved = sum(
            1 for r in valid_results if "1921" in r.get("compressed_context", "")
        ) / len(valid_results)

        compression = CompressionStats(
            original_tokens=int(avg_original),
            compressed_tokens=int(avg_compressed),
            compression_ratio=avg_ratio,
            tokens_saved=int(avg_original - avg_compressed),
            preservation_score=preserved,
        )
    else:
        compression = None

    return BenchmarkResult(
        name="compress_prompt_latency",
        category=BenchmarkCategory.LATENCY,
        latency=latency,
        compression=compression,
        metadata={"context_size": len(context), "target_ratio": 0.5},
    )


async def bench_compress_sizes(client: Any) -> BenchmarkResult:
    """Benchmark compression across different context sizes."""
    from mcp.types import TextContent

    base_text = "The quick brown fox jumps over the lazy dog. " * 10

    sizes = [
        ("small", base_text * 1),  # ~100 tokens
        ("medium", base_text * 10),  # ~1000 tokens
        ("large", base_text * 50),  # ~5000 tokens
    ]

    size_metrics: dict[str, dict[str, Any]] = {}

    for size_name, context in sizes:
        samples: list[float] = []
        for _ in range(5):
            start = time.perf_counter()
            try:
                result = await client.call_tool(
                    "compress_prompt",
                    {
                        "context": context,
                        "question": "What does the fox do?",
                        "compression_ratio": 0.5,
                    },
                )
                content = result.content[0]
                resp = json.loads(content.text if isinstance(content, TextContent) else "{}")
                elapsed_ms = (time.perf_counter() - start) * 1000
                samples.append(elapsed_ms)

                if samples and resp.get("original_tokens"):
                    size_metrics[size_name] = {
                        "tokens": resp["original_tokens"],
                        "latency_ms": statistics.mean(samples),
                        "ms_per_token": statistics.mean(samples) / resp["original_tokens"],
                    }
            except Exception:
                pass

    return BenchmarkResult(
        name="compress_scaling",
        category=BenchmarkCategory.COMPRESSION,
        metadata={"size_metrics": size_metrics},
    )


async def bench_strategy_recommendation(client: Any, iterations: int) -> BenchmarkResult:
    """Benchmark strategy recommendation latency."""
    from mcp.types import TextContent

    problems = [
        ("serial", "Find a path from A to Z in a graph with 100 nodes."),
        ("parallel", "What are 5 different approaches to reduce carbon emissions?"),
        ("matrix", "Compare Python, Rust, and Go for building web services."),
    ]

    all_samples: list[float] = []
    strategy_results: dict[str, list[str]] = {p[0]: [] for p in problems}

    for problem_type, problem in problems:
        for _ in range(iterations // len(problems)):
            start = time.perf_counter()
            try:
                result = await client.call_tool(
                    "recommend_reasoning_strategy",
                    {"problem": problem, "token_budget": 5000},
                )
                content = result.content[0]
                resp = json.loads(content.text if isinstance(content, TextContent) else "{}")
                elapsed_ms = (time.perf_counter() - start) * 1000
                all_samples.append(elapsed_ms)

                if "recommended_strategy" in resp:
                    strategy_results[problem_type].append(resp["recommended_strategy"])
            except Exception:
                pass

    # Analyze strategy consistency
    consistency: dict[str, float] = {}
    for ptype, strategies in strategy_results.items():
        if strategies:
            most_common = max(set(strategies), key=strategies.count)
            consistency[ptype] = strategies.count(most_common) / len(strategies)

    return BenchmarkResult(
        name="recommend_strategy_latency",
        category=BenchmarkCategory.STRATEGY,
        latency=LatencyStats.from_samples(all_samples),
        metadata={
            "strategy_consistency": consistency,
            "problem_types_tested": len(problems),
        },
    )


async def bench_session_lifecycle(client: Any, iterations: int) -> BenchmarkResult:
    """Benchmark session creation and management overhead."""
    from mcp.types import TextContent

    create_samples: list[float] = []
    step_samples: list[float] = []
    finalize_samples: list[float] = []

    for _ in range(iterations):
        # Measure session creation
        start = time.perf_counter()
        result = await client.call_tool(
            "chain_start",
            {"problem": "Test problem", "expected_steps": 3},
        )
        content = result.content[0]
        resp = json.loads(content.text if isinstance(content, TextContent) else "{}")
        create_samples.append((time.perf_counter() - start) * 1000)

        session_id = resp.get("session_id")
        if not session_id:
            continue

        # Measure step addition
        start = time.perf_counter()
        await client.call_tool(
            "chain_add_step",
            {"session_id": session_id, "thought": "Step 1 reasoning"},
        )
        step_samples.append((time.perf_counter() - start) * 1000)

        # Measure finalization
        start = time.perf_counter()
        await client.call_tool(
            "chain_finalize",
            {"session_id": session_id, "answer": "Result", "confidence": 0.9},
        )
        finalize_samples.append((time.perf_counter() - start) * 1000)

    return BenchmarkResult(
        name="session_lifecycle",
        category=BenchmarkCategory.SESSION,
        latency=LatencyStats.from_samples(create_samples + step_samples + finalize_samples),
        metadata={
            "create_latency": LatencyStats.from_samples(create_samples).to_dict(),
            "step_latency": LatencyStats.from_samples(step_samples).to_dict(),
            "finalize_latency": LatencyStats.from_samples(finalize_samples).to_dict(),
        },
    )


async def bench_matrix_workflow(client: Any, iterations: int) -> BenchmarkResult:
    """Benchmark complete matrix workflow."""
    from mcp.types import TextContent

    workflow_samples: list[float] = []
    cell_samples: list[float] = []

    for _ in range(iterations):
        workflow_start = time.perf_counter()

        # Start matrix
        result = await client.call_tool(
            "matrix_start",
            {"question": "Evaluate Python for ML", "rows": 2, "cols": 2},
        )
        content = result.content[0]
        resp = json.loads(content.text if isinstance(content, TextContent) else "{}")
        session_id = resp.get("session_id")

        if not session_id:
            continue

        # Fill cells (2x2 = 4 cells)
        thoughts = [
            "Good libraries like PyTorch",
            "Can be slow for large data",
            "Easy to prototype",
            "Memory management issues",
        ]
        for i, thought in enumerate(thoughts):
            row, col = divmod(i, 2)
            start = time.perf_counter()
            await client.call_tool(
                "matrix_set_cell",
                {"session_id": session_id, "row": row, "col": col, "thought": thought},
            )
            cell_samples.append((time.perf_counter() - start) * 1000)

        # Synthesize columns
        for col in range(2):
            await client.call_tool(
                "matrix_synthesize",
                {"session_id": session_id, "col": col, "synthesis": f"Column {col} summary"},
            )

        # Finalize
        await client.call_tool(
            "matrix_finalize",
            {"session_id": session_id, "answer": "Python is good for ML", "confidence": 0.85},
        )

        workflow_samples.append((time.perf_counter() - workflow_start) * 1000)

    return BenchmarkResult(
        name="matrix_workflow",
        category=BenchmarkCategory.LATENCY,
        latency=LatencyStats.from_samples(workflow_samples),
        metadata={
            "workflow_type": "2x2 matrix",
            "cell_latency": LatencyStats.from_samples(cell_samples).to_dict(),
            "total_operations_per_workflow": 8,  # start + 4 cells + 2 synth + finalize
        },
    )


async def bench_verification_workflow(client: Any, iterations: int) -> BenchmarkResult:
    """Benchmark verification workflow."""
    from mcp.types import TextContent

    workflow_samples: list[float] = []
    claim_samples: list[float] = []

    for _ in range(iterations):
        workflow_start = time.perf_counter()

        # Start verification
        result = await client.call_tool(
            "verify_start",
            {
                "answer": "Paris is the capital of France. It has 2 million people.",
                "context": "Paris is the capital of France with ~2.1 million people.",
            },
        )
        content = result.content[0]
        resp = json.loads(content.text if isinstance(content, TextContent) else "{}")
        session_id = resp.get("session_id")

        if not session_id:
            continue

        # Add claims
        claims = ["Paris is the capital of France", "Paris has 2 million people"]
        for claim in claims:
            await client.call_tool(
                "verify_add_claim",
                {"session_id": session_id, "claim": claim},
            )

        # Verify claims
        for i, status in enumerate(["supported", "supported"]):
            start = time.perf_counter()
            await client.call_tool(
                "verify_claim",
                {
                    "session_id": session_id,
                    "claim_id": str(i),
                    "status": status,
                    "evidence": "Found in context",
                },
            )
            claim_samples.append((time.perf_counter() - start) * 1000)

        # Finalize
        await client.call_tool("verify_finalize", {"session_id": session_id})

        workflow_samples.append((time.perf_counter() - workflow_start) * 1000)

    return BenchmarkResult(
        name="verification_workflow",
        category=BenchmarkCategory.LATENCY,
        latency=LatencyStats.from_samples(workflow_samples),
        metadata={
            "claims_per_workflow": 2,
            "claim_verification_latency": LatencyStats.from_samples(claim_samples).to_dict(),
        },
    )


async def bench_concurrent_sessions(client: Any) -> BenchmarkResult:
    """Benchmark concurrent session handling."""
    from mcp.types import TextContent

    async def create_and_use_session(session_num: int) -> float:
        start = time.perf_counter()

        result = await client.call_tool(
            "chain_start",
            {"problem": f"Problem {session_num}", "expected_steps": 2},
        )
        content = result.content[0]
        resp = json.loads(content.text if isinstance(content, TextContent) else "{}")
        session_id = resp.get("session_id")

        if session_id:
            await client.call_tool(
                "chain_add_step",
                {"session_id": session_id, "thought": f"Thought for {session_num}"},
            )
            await client.call_tool(
                "chain_finalize",
                {"session_id": session_id, "answer": f"Answer {session_num}", "confidence": 0.8},
            )

        return (time.perf_counter() - start) * 1000

    # Test different concurrency levels
    concurrency_results: dict[int, dict[str, float]] = {}

    for concurrency in [1, 5, 10, 20]:
        tasks = [create_and_use_session(i) for i in range(concurrency)]
        start = time.perf_counter()
        durations = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.perf_counter() - start) * 1000

        valid_durations = [d for d in durations if isinstance(d, float)]
        errors = len(durations) - len(valid_durations)

        concurrency_results[concurrency] = {
            "total_time_ms": round(total_time, 2),
            "avg_per_session_ms": round(statistics.mean(valid_durations), 2)
            if valid_durations
            else 0,
            "throughput_ops_s": round(concurrency / (total_time / 1000), 2),
            "errors": errors,
        }

    return BenchmarkResult(
        name="concurrent_sessions",
        category=BenchmarkCategory.THROUGHPUT,
        metadata={"concurrency_scaling": concurrency_results},
    )


async def bench_throughput(client: Any, duration_s: float = 5.0) -> BenchmarkResult:
    """Measure sustained throughput."""
    from mcp.types import TextContent

    async def light_operation() -> None:
        result = await client.call_tool(
            "recommend_reasoning_strategy",
            {"problem": "Simple test", "token_budget": 1000},
        )
        content = result.content[0]
        json.loads(content.text if isinstance(content, TextContent) else "{}")

    throughput = await measure_throughput(light_operation, duration_s)

    return BenchmarkResult(
        name="sustained_throughput",
        category=BenchmarkCategory.THROUGHPUT,
        throughput=throughput,
        metadata={"operation": "recommend_reasoning_strategy", "duration_s": duration_s},
    )


# =============================================================================
# Report Generation
# =============================================================================


def print_report(results: list[BenchmarkResult], verbose: bool = False) -> None:
    """Print human-readable benchmark report."""
    print("\n" + "=" * 78)
    print("                    MATRIXMIND MCP PERFORMANCE REPORT")
    print("=" * 78)

    # Group by category
    by_category: dict[BenchmarkCategory, list[BenchmarkResult]] = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)

    # Print each category
    for category in BenchmarkCategory:
        if category not in by_category:
            continue

        cat_results = by_category[category]
        print(f"\n┌─ {category.value.upper()} {'─' * (73 - len(category.value))}")

        for result in cat_results:
            status = "✅" if result.passed else "❌"
            print(f"│\n│  {status} {result.name}")

            if result.error:
                print(f"│     Error: {result.error}")
                continue

            # Latency stats
            if result.latency and result.latency.samples > 0:
                lat = result.latency
                print(f"│     Latency (n={lat.samples}):")
                print(f"│       p50: {lat.median_ms:>8.2f} ms")
                print(f"│       p95: {lat.p95_ms:>8.2f} ms")
                print(f"│       p99: {lat.p99_ms:>8.2f} ms")
                print(f"│       min: {lat.min_ms:>8.2f} ms  max: {lat.max_ms:.2f} ms")

            # Throughput stats
            if result.throughput:
                tp = result.throughput
                print(f"│     Throughput: {tp.ops_per_second:.1f} ops/sec")
                print(f"│       Total ops: {tp.total_ops}, Errors: {tp.errors}")
                if tp.total_ops > 0:
                    success_rate = (tp.total_ops - tp.errors) / tp.total_ops * 100
                    print(f"│       Success rate: {success_rate:.1f}%")

            # Compression stats
            if result.compression:
                comp = result.compression
                print("│     Compression:")
                print(f"│       Ratio: {comp.compression_ratio:.1%}")
                print(f"│       Tokens: {comp.original_tokens} → {comp.compressed_tokens}")
                saved_pct = comp.tokens_saved / comp.original_tokens * 100
                print(f"│       Saved: {comp.tokens_saved} tokens ({saved_pct:.1f}%)")
                print(f"│       Key info preserved: {comp.preservation_score:.0%}")

            # Metadata highlights
            if result.metadata and verbose:
                print("│     Details:")
                for key, value in result.metadata.items():
                    if isinstance(value, dict):
                        print(f"│       {key}:")
                        for k, v in value.items():
                            print(f"│         {k}: {v}")
                    else:
                        print(f"│       {key}: {value}")

        print("│")

    # Summary
    print("└" + "─" * 77)
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print(f"  Tests:  {passed}/{total} passed ({100 * passed / total:.0f}%)")

    # Key metrics summary
    latency_results = [r for r in results if r.latency and r.latency.samples > 0]
    if latency_results:
        all_p95 = [r.latency.p95_ms for r in latency_results if r.latency]
        print(f"  p95 Latency Range: {min(all_p95):.1f} - {max(all_p95):.1f} ms")

    throughput_results = [r for r in results if r.throughput]
    if throughput_results:
        max_throughput = max(
            r.throughput.ops_per_second for r in throughput_results if r.throughput
        )
        print(f"  Peak Throughput: {max_throughput:.1f} ops/sec")

    if failed > 0:
        print(f"\n  ❌ {failed} benchmark(s) failed")
    else:
        print("\n  ✅ All benchmarks passed")

    print("=" * 78)


def export_results(results: list[BenchmarkResult], filepath: str) -> None:
    """Export results to JSON file."""
    export_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [r.to_dict() for r in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
        },
    }

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"\nResults exported to: {filepath}")


# =============================================================================
# Benchmark Runner
# =============================================================================


async def run_benchmarks(
    full: bool = False,
    compare: bool = False,
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """Run benchmark suite."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Error: fastmcp not installed. Run: pip install fastmcp")
        sys.exit(1)

    iterations = FULL_ITERATIONS if full else DEFAULT_ITERATIONS
    results: list[BenchmarkResult] = []

    print(f"\nRunning {'full' if full else 'quick'} benchmark ({iterations} iterations)...")

    async with Client("src/server.py") as client:
        # Core latency benchmarks
        print("  → Compression latency...")
        results.append(await bench_compress_latency(client, iterations))

        print("  → Strategy recommendation...")
        results.append(await bench_strategy_recommendation(client, iterations))

        print("  → Session lifecycle...")
        results.append(await bench_session_lifecycle(client, iterations))

        # Workflow benchmarks
        print("  → Matrix workflow...")
        results.append(await bench_matrix_workflow(client, iterations // 2))

        print("  → Verification workflow...")
        results.append(await bench_verification_workflow(client, iterations // 2))

        # Scaling benchmarks (only in full mode)
        if full or compare:
            print("  → Compression scaling...")
            results.append(await bench_compress_sizes(client))

            print("  → Concurrent sessions...")
            results.append(await bench_concurrent_sessions(client))

            print("  → Sustained throughput...")
            results.append(await bench_throughput(client, duration_s=10.0 if full else 5.0))

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


async def main() -> None:
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(
        description="MatrixMind MCP Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py              Quick benchmark (10 iterations)
  python benchmark.py --full       Full benchmark (50 iterations)
  python benchmark.py --export r.json  Export results to JSON
  python benchmark.py -v           Verbose output with all details
        """,
    )
    parser.add_argument("--full", "-f", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--compare", "-c", action="store_true", help="Include comparison tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--export", "-e", type=str, metavar="FILE", help="Export results to JSON")
    args = parser.parse_args()

    print("=" * 78)
    print("              MATRIXMIND MCP PERFORMANCE BENCHMARK")
    print("=" * 78)

    results = await run_benchmarks(
        full=args.full,
        compare=args.compare,
        verbose=args.verbose,
    )

    print_report(results, verbose=args.verbose)

    if args.export:
        export_results(results, args.export)

    # Exit with error if any benchmark failed
    if any(not r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
