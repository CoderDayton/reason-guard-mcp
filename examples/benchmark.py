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
    python examples/benchmark.py --llm              # Enable LLM for realistic inputs

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

# Import shared LLM utilities for future LLM-powered benchmarks
from llm_client import (
    CISCSolveResult,
    add_llm_args,
    cisc_solve,
    close_llm_client,
    get_llm_alternatives,
    get_llm_answer,
    get_llm_confidence,
    init_llm_client,
    is_llm_enabled,
)

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


def parse_tool_response(result: Any) -> dict[str, Any]:
    """Parse tool response content to dict."""
    from mcp.types import TextContent

    if hasattr(result, "content") and result.content:
        content = result.content[0]
        text = content.text if isinstance(content, TextContent) else "{}"
        return json.loads(text)
    return {}


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
        allocated_blocks=0,
    ), result


# =============================================================================
# Benchmark Test Cases
# =============================================================================


async def bench_compress_latency(client: Any, iterations: int) -> BenchmarkResult:
    """Benchmark compression latency with varying context sizes."""
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
            "compress",
            {
                "context": context,
                "query": "When did Einstein receive the Nobel Prize?",
                "ratio": 0.5,
            },
        )
        return parse_tool_response(result)

    latency, results = await measure_latency(compress_call, iterations)

    # Calculate compression stats from results
    valid_results = [r for r in results if isinstance(r, dict) and "error" not in r]
    compression: CompressionStats | None = None

    if valid_results:
        avg_original = sum(r.get("original_tokens", 0) for r in valid_results) / len(valid_results)
        avg_compressed = sum(r.get("compressed_tokens", 0) for r in valid_results) / len(
            valid_results
        )
        avg_ratio = sum(r.get("compression_ratio", 1) for r in valid_results) / len(valid_results)

        # Check if key fact preserved
        preserved = sum(1 for r in valid_results if "1921" in r.get("compressed", "")) / len(
            valid_results
        )

        compression = CompressionStats(
            original_tokens=int(avg_original),
            compressed_tokens=int(avg_compressed),
            compression_ratio=avg_ratio,
            tokens_saved=int(avg_original - avg_compressed),
            preservation_score=preserved,
        )

    return BenchmarkResult(
        name="compress_latency",
        category=BenchmarkCategory.COMPRESSION,
        latency=latency,
        compression=compression,
        metadata={"context_size": len(context), "target_ratio": 0.5},
    )


async def bench_compress_sizes(client: Any) -> BenchmarkResult:
    """Benchmark compression across different context sizes."""
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
                    "compress",
                    {"context": context, "query": "What does the fox do?", "ratio": 0.5},
                )
                resp = parse_tool_response(result)
                elapsed_ms = (time.perf_counter() - start) * 1000
                samples.append(elapsed_ms)

                if samples and resp.get("original_tokens"):
                    size_metrics[size_name] = {
                        "tokens": resp["original_tokens"],
                        "latency_ms": round(statistics.mean(samples), 2),
                        "ms_per_token": round(
                            statistics.mean(samples) / resp["original_tokens"], 3
                        ),
                    }
            except Exception:
                pass

    return BenchmarkResult(
        name="compress_scaling",
        category=BenchmarkCategory.COMPRESSION,
        metadata={"size_metrics": size_metrics},
    )


async def bench_chain_workflow(client: Any, iterations: int) -> BenchmarkResult:
    """Benchmark chain reasoning workflow (start → continue → finish)."""
    start_samples: list[float] = []
    continue_samples: list[float] = []
    finish_samples: list[float] = []
    workflow_samples: list[float] = []

    for _ in range(iterations):
        workflow_start = time.perf_counter()

        # Start chain
        start = time.perf_counter()
        result = await client.call_tool(
            "think",
            {"action": "start", "mode": "chain", "problem": "What is 2+2?", "expected_steps": 3},
        )
        resp = parse_tool_response(result)
        start_samples.append((time.perf_counter() - start) * 1000)

        session_id = resp.get("session_id")
        if not session_id:
            continue

        # Continue (add step)
        start = time.perf_counter()
        await client.call_tool(
            "think",
            {"action": "continue", "session_id": session_id, "thought": "Let me add 2 and 2"},
        )
        continue_samples.append((time.perf_counter() - start) * 1000)

        # Finish
        start = time.perf_counter()
        await client.call_tool(
            "think",
            {
                "action": "finish",
                "session_id": session_id,
                "thought": "The answer is 4",
                "confidence": 0.99,
            },
        )
        finish_samples.append((time.perf_counter() - start) * 1000)

        workflow_samples.append((time.perf_counter() - workflow_start) * 1000)

    return BenchmarkResult(
        name="chain_workflow",
        category=BenchmarkCategory.SESSION,
        latency=LatencyStats.from_samples(workflow_samples),
        metadata={
            "start_latency": LatencyStats.from_samples(start_samples).to_dict(),
            "continue_latency": LatencyStats.from_samples(continue_samples).to_dict(),
            "finish_latency": LatencyStats.from_samples(finish_samples).to_dict(),
            "operations_per_workflow": 3,
        },
    )


async def bench_matrix_workflow(client: Any, iterations: int) -> BenchmarkResult:
    """Benchmark complete matrix workflow."""
    workflow_samples: list[float] = []
    cell_samples: list[float] = []

    for _ in range(iterations):
        workflow_start = time.perf_counter()

        # Start matrix (2x2 for speed)
        result = await client.call_tool(
            "think",
            {
                "action": "start",
                "mode": "matrix",
                "problem": "Evaluate Python",
                "rows": 2,
                "cols": 2,
            },
        )
        resp = parse_tool_response(result)
        session_id = resp.get("session_id")

        if not session_id:
            continue

        # Fill cells (2x2 = 4 cells)
        thoughts = ["Good libraries", "Can be slow", "Easy syntax", "Memory issues"]
        for i, thought in enumerate(thoughts):
            row, col = divmod(i, 2)
            start = time.perf_counter()
            await client.call_tool(
                "think",
                {
                    "action": "continue",
                    "session_id": session_id,
                    "row": row,
                    "col": col,
                    "thought": thought,
                },
            )
            cell_samples.append((time.perf_counter() - start) * 1000)

        # Synthesize columns
        for col in range(2):
            await client.call_tool(
                "think",
                {
                    "action": "synthesize",
                    "session_id": session_id,
                    "col": col,
                    "thought": f"Column {col} synthesis",
                },
            )

        # Finish
        await client.call_tool(
            "think",
            {
                "action": "finish",
                "session_id": session_id,
                "thought": "Python is good for ML",
                "confidence": 0.85,
            },
        )

        workflow_samples.append((time.perf_counter() - workflow_start) * 1000)

    return BenchmarkResult(
        name="matrix_workflow",
        category=BenchmarkCategory.LATENCY,
        latency=LatencyStats.from_samples(workflow_samples),
        metadata={
            "matrix_size": "2x2",
            "cell_latency": LatencyStats.from_samples(cell_samples).to_dict(),
            "operations_per_workflow": 8,  # start + 4 cells + 2 synth + finish
        },
    )


async def bench_verify_workflow(client: Any, iterations: int) -> BenchmarkResult:
    """Benchmark verification workflow."""
    workflow_samples: list[float] = []
    verify_samples: list[float] = []

    for _ in range(iterations):
        workflow_start = time.perf_counter()

        # Start verification
        result = await client.call_tool(
            "think",
            {
                "action": "start",
                "mode": "verify",
                "problem": "Paris is the capital of France",
                "context": "Paris is the capital city of France with ~2 million people.",
            },
        )
        resp = parse_tool_response(result)
        session_id = resp.get("session_id")

        if not session_id:
            continue

        # Add a claim via continue
        await client.call_tool(
            "think",
            {"action": "continue", "session_id": session_id, "thought": "Paris is capital"},
        )

        # Verify the claim
        start = time.perf_counter()
        await client.call_tool(
            "think",
            {
                "action": "verify",
                "session_id": session_id,
                "claim_id": 0,
                "verdict": "supported",
                "evidence": "Stated in context",
            },
        )
        verify_samples.append((time.perf_counter() - start) * 1000)

        # Finish
        await client.call_tool("think", {"action": "finish", "session_id": session_id})

        workflow_samples.append((time.perf_counter() - workflow_start) * 1000)

    return BenchmarkResult(
        name="verify_workflow",
        category=BenchmarkCategory.LATENCY,
        latency=LatencyStats.from_samples(workflow_samples),
        metadata={
            "verify_latency": LatencyStats.from_samples(verify_samples).to_dict(),
            "operations_per_workflow": 4,
        },
    )


async def bench_status(client: Any, iterations: int) -> BenchmarkResult:
    """Benchmark status tool (lightweight operation baseline)."""

    async def status_call() -> dict[str, Any]:
        result = await client.call_tool("status", {})
        return parse_tool_response(result)

    latency, _ = await measure_latency(status_call, iterations)

    return BenchmarkResult(
        name="status_latency",
        category=BenchmarkCategory.LATENCY,
        latency=latency,
        metadata={"operation": "status (baseline)"},
    )


async def bench_concurrent_sessions(client: Any) -> BenchmarkResult:
    """Benchmark concurrent session handling."""

    async def create_and_use_session(session_num: int) -> float:
        start = time.perf_counter()

        result = await client.call_tool(
            "think",
            {
                "action": "start",
                "mode": "chain",
                "problem": f"Problem {session_num}",
                "expected_steps": 2,
            },
        )
        resp = parse_tool_response(result)
        session_id = resp.get("session_id")

        if session_id:
            await client.call_tool(
                "think",
                {
                    "action": "continue",
                    "session_id": session_id,
                    "thought": f"Thought for {session_num}",
                },
            )
            await client.call_tool(
                "think",
                {
                    "action": "finish",
                    "session_id": session_id,
                    "thought": f"Answer {session_num}",
                    "confidence": 0.8,
                },
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
    """Measure sustained throughput using status (lightweight op)."""

    async def light_operation() -> None:
        result = await client.call_tool("status", {})
        parse_tool_response(result)

    throughput = await measure_throughput(light_operation, duration_s)

    return BenchmarkResult(
        name="sustained_throughput",
        category=BenchmarkCategory.THROUGHPUT,
        throughput=throughput,
        metadata={"operation": "status", "duration_s": duration_s},
    )


# =============================================================================
# LLM-Powered Reasoning Quality Benchmark
# =============================================================================

# Realistic coding workflow problems that benefit from structured reasoning
# These simulate debugging, code review, and architectural decisions
LLM_BENCHMARK_PROBLEMS = [
    # --- Multi-step debugging scenarios (chain reasoning helps) ---
    {
        "id": "debug_null_pointer",
        "question": """A Python function crashes with 'AttributeError: NoneType has no attribute get'.
The code is:
```python
def get_user_email(users, user_id):
    user = users.get(user_id)
    return user.get('email')
```
What's the bug and how do you fix it?""",
        "expected": "None",
        "keywords": ["None", "check", "if", "default", "get", "or", "is None"],
        "category": "debugging",
    },
    {
        "id": "debug_off_by_one",
        "question": """This loop should print numbers 1 to 10 but prints 1 to 9:
```python
for i in range(1, 10):
    print(i)
```
What's wrong and what's the fix?""",
        "expected": "11",
        "keywords": ["range", "exclusive", "11", "10", "end", "inclusive"],
        "category": "debugging",
    },
    {
        "id": "debug_async_race",
        "question": """This async code sometimes returns stale data:
```python
cache = {}
async def get_data(key):
    if key in cache:
        return cache[key]
    data = await fetch_from_db(key)
    cache[key] = data
    return data
```
What's the race condition and how do you fix it?""",
        "expected": "lock",
        "keywords": ["race", "condition", "lock", "async", "concurrent", "await", "mutex"],
        "category": "debugging",
    },
    # --- Code review scenarios (matrix/multi-perspective helps) ---
    {
        "id": "review_sql_injection",
        "question": """Review this code for security issues:
```python
def search_users(name):
    query = f"SELECT * FROM users WHERE name = '{name}'"
    return db.execute(query)
```
What vulnerabilities exist and how should it be fixed?""",
        "expected": "injection",
        "keywords": [
            "SQL",
            "injection",
            "parameterized",
            "prepared",
            "sanitize",
            "escape",
            "?",
            "%s",
        ],
        "category": "security",
    },
    {
        "id": "review_memory_leak",
        "question": """This class has a memory leak. Find it:
```python
class EventHandler:
    _handlers = []  # class variable

    def __init__(self, callback):
        EventHandler._handlers.append(callback)

    def trigger(self, event):
        for h in self._handlers:
            h(event)
```
What causes the leak and how do you fix it?""",
        "expected": "class variable",
        "keywords": [
            "class",
            "variable",
            "instance",
            "shared",
            "append",
            "grows",
            "never",
            "removed",
            "weakref",
        ],
        "category": "memory",
    },
    # --- Architecture decisions (requires weighing tradeoffs) ---
    {
        "id": "arch_cache_strategy",
        "question": """A web API has 10K requests/sec. Database queries take 50ms.
We need sub-10ms response times. Cache hit rate is 80%.
Should we use: (A) Redis with 1ms latency, (B) In-memory cache with 0.1ms latency, or (C) CDN edge cache?
Consider: consistency, memory limits, deployment complexity.""",
        "expected": "Redis",
        "keywords": [
            "Redis",
            "memory",
            "distributed",
            "consistency",
            "TTL",
            "invalidation",
            "scale",
        ],
        "category": "architecture",
    },
    {
        "id": "arch_microservice_split",
        "question": """A monolith handles: user auth, payments, notifications, and analytics.
Team size: 4 developers. Traffic: 1000 req/sec.
Should we split into microservices? If yes, which services first? If no, why not?""",
        "expected": "no",
        "keywords": ["monolith", "complexity", "team", "small", "overhead", "latency", "deploy"],
        "category": "architecture",
    },
    # --- Multi-hop reasoning (requires connecting multiple facts) ---
    {
        "id": "multihop_dependency",
        "question": """Given these module dependencies:
- auth.py imports user.py and crypto.py
- user.py imports database.py
- payment.py imports user.py and stripe_api.py
- database.py imports config.py

If config.py has a bug, which modules are affected?""",
        "expected": "auth",
        "keywords": ["database", "user", "auth", "payment", "config", "transitive", "all"],
        "category": "dependency",
    },
    {
        "id": "multihop_error_trace",
        "question": """Error trace:
1. API returns 500 error
2. Logs show: "ConnectionError in payment_service.charge()"
3. payment_service calls billing_api.create_invoice()
4. billing_api uses database connection pool
5. Database connection pool exhausted (max_connections=10, active=10)

What is the root cause? What should be fixed first?""",
        "expected": "pool",
        "keywords": [
            "connection",
            "pool",
            "exhausted",
            "max",
            "increase",
            "leak",
            "close",
            "timeout",
        ],
        "category": "debugging",
    },
]

# GSM8K-style math word problems for CISC evaluation
# These have answer diversity (models may get different numeric answers)
# which is essential for CISC weighted majority voting to help
GSM8K_STYLE_PROBLEMS = [
    {
        "id": "gsm_shopping",
        "question": """Janet buys 3 pounds of apples at $2 per pound and 2 pounds of oranges at $3 per pound.
She pays with a $20 bill. How much change does she receive?""",
        "expected": "8",
        "keywords": ["6", "6", "12", "20", "8", "change"],
        "category": "arithmetic",
    },
    {
        "id": "gsm_time",
        "question": """A train leaves at 9:15 AM and arrives at 2:45 PM.
The journey includes a 30-minute stop. How many hours was the train actually moving?""",
        "expected": "5",
        "keywords": ["5", "hours", "30", "minutes", "subtract"],
        "category": "time",
    },
    {
        "id": "gsm_ratio",
        "question": """In a class of 30 students, the ratio of boys to girls is 2:3.
How many girls are in the class?""",
        "expected": "18",
        "keywords": ["18", "girls", "ratio", "2", "3", "30"],
        "category": "ratio",
    },
    {
        "id": "gsm_percentage",
        "question": """A shirt originally costs $80. It's on sale for 25% off.
What is the sale price?""",
        "expected": "60",
        "keywords": ["60", "25", "percent", "20", "off", "discount"],
        "category": "percentage",
    },
    {
        "id": "gsm_work",
        "question": """Alice can paint a room in 6 hours. Bob can paint the same room in 3 hours.
If they work together, how many hours will it take them to paint the room?""",
        "expected": "2",
        "keywords": ["2", "hours", "together", "rate", "combined"],
        "category": "work_rate",
    },
    {
        "id": "gsm_distance",
        "question": """A car travels at 60 mph for 2 hours, then at 40 mph for 3 hours.
What is the average speed for the entire trip?""",
        "expected": "48",
        "keywords": ["48", "mph", "total", "distance", "240", "5"],
        "category": "average",
    },
    {
        "id": "gsm_profit",
        "question": """A store buys items for $15 each and sells them for $24 each.
If they sell 50 items, what is their total profit?""",
        "expected": "450",
        "keywords": ["450", "profit", "9", "50", "difference"],
        "category": "profit",
    },
    {
        "id": "gsm_area",
        "question": """A rectangular garden is 12 meters long and 8 meters wide.
A path 1 meter wide runs around the outside. What is the area of the path?""",
        "expected": "44",
        "keywords": ["44", "area", "path", "outer", "inner", "subtract"],
        "category": "geometry",
    },
    {
        "id": "gsm_age",
        "question": """Tom is 3 times as old as his son. In 12 years, Tom will be twice as old as his son.
How old is Tom now?""",
        "expected": "36",
        "keywords": ["36", "Tom", "12", "son", "years"],
        "category": "algebra",
    },
    {
        "id": "gsm_probability",
        "question": """A bag contains 4 red balls and 6 blue balls.
If you draw 2 balls without replacement, what is the probability both are red?
Express as a simplified fraction.""",
        "expected": "2/15",
        "keywords": ["2/15", "4", "10", "3", "9", "probability"],
        "category": "probability",
    },
]

# Code context for embedding-based retrieval benchmark
CODE_CONTEXT_SAMPLES = [
    {
        "file": "auth/login.py",
        "content": """
class LoginService:
    def __init__(self, user_repo, session_manager):
        self.user_repo = user_repo
        self.session_manager = session_manager

    def authenticate(self, username, password):
        user = self.user_repo.find_by_username(username)
        if user and user.verify_password(password):
            return self.session_manager.create_session(user)
        raise AuthenticationError("Invalid credentials")
""",
    },
    {
        "file": "auth/session.py",
        "content": """
class SessionManager:
    def __init__(self, redis_client, ttl=3600):
        self.redis = redis_client
        self.ttl = ttl

    def create_session(self, user):
        token = generate_secure_token()
        self.redis.setex(f"session:{token}", self.ttl, user.id)
        return token

    def get_user_id(self, token):
        return self.redis.get(f"session:{token}")
""",
    },
    {
        "file": "models/user.py",
        "content": """
class User:
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    def verify_password(self, password):
        return bcrypt.checkpw(password.encode(), self.password_hash)
""",
    },
    {
        "file": "repos/user_repository.py",
        "content": """
class UserRepository:
    def __init__(self, db_connection):
        self.db = db_connection

    def find_by_username(self, username):
        result = self.db.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        return User(*result) if result else None
""",
    },
    {
        "file": "api/endpoints.py",
        "content": """
@app.post("/login")
def login_endpoint(request):
    username = request.json.get("username")
    password = request.json.get("password")
    try:
        token = login_service.authenticate(username, password)
        return {"token": token}
    except AuthenticationError as e:
        return {"error": str(e)}, 401
""",
    },
    # Additional files to increase corpus size for better retrieval testing
    {
        "file": "payment/stripe_client.py",
        "content": """
class StripeClient:
    def __init__(self, api_key):
        self.api_key = api_key
        stripe.api_key = api_key

    def create_charge(self, amount, currency, customer_id):
        return stripe.Charge.create(
            amount=amount,
            currency=currency,
            customer=customer_id
        )

    def refund(self, charge_id):
        return stripe.Refund.create(charge=charge_id)
""",
    },
    {
        "file": "payment/invoice.py",
        "content": """
class InvoiceGenerator:
    def __init__(self, template_engine):
        self.template = template_engine

    def generate(self, order, customer):
        items = [{"name": i.name, "price": i.price} for i in order.items]
        total = sum(i.price for i in order.items)
        return self.template.render("invoice.html", items=items, total=total)
""",
    },
    {
        "file": "cache/redis_cache.py",
        "content": """
class RedisCache:
    def __init__(self, host, port, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)
        self.default_ttl = 3600

    def get(self, key):
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set(self, key, value, ttl=None):
        self.client.setex(key, ttl or self.default_ttl, json.dumps(value))

    def invalidate(self, pattern):
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)
""",
    },
    {
        "file": "utils/logging.py",
        "content": """
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file, level=logging.INFO):
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
""",
    },
    {
        "file": "utils/validators.py",
        "content": """
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password):
    # At least 8 chars, 1 uppercase, 1 lowercase, 1 digit
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'[0-9]', password):
        return False
    return True
""",
    },
]

# Retrieval test queries with expected relevant files
RETRIEVAL_TEST_CASES = [
    {
        "query": "How does user login authentication work?",
        "expected_files": ["auth/login.py", "models/user.py", "api/endpoints.py"],
        "description": "Multi-file authentication flow",
    },
    {
        "query": "Where is the session token stored and managed?",
        "expected_files": ["auth/session.py", "cache/redis_cache.py"],
        "description": "Session and cache management",
    },
    {
        "query": "How do I process a payment refund?",
        "expected_files": ["payment/stripe_client.py"],
        "description": "Payment processing",
    },
    {
        "query": "What validation rules are used for user registration?",
        "expected_files": ["utils/validators.py", "models/user.py"],
        "description": "Input validation",
    },
    {
        "query": "How is the database queried for user data?",
        "expected_files": ["repos/user_repository.py"],
        "description": "Database access layer",
    },
]


def _check_answer(response: str, expected: str, keywords: list[str]) -> tuple[bool, float]:
    """Check if response contains expected answer and keywords.

    Returns (is_correct, keyword_coverage).
    """
    response_lower = response.lower()
    expected_lower = expected.lower()

    # Check for exact answer
    is_correct = expected_lower in response_lower

    # Check keyword coverage
    found = sum(1 for kw in keywords if kw.lower() in response_lower)
    coverage = found / len(keywords) if keywords else 1.0

    return is_correct, coverage


async def bench_code_retrieval(client: Any, iterations: int = 2) -> BenchmarkResult:
    """Benchmark semantic code retrieval using embedding-based compression.

    Tests the compress tool's ability to identify relevant code files for a query.
    Uses CODE_CONTEXT_SAMPLES as the codebase and RETRIEVAL_TEST_CASES as test queries.

    Measures:
    - Precision: % of retrieved files that are relevant
    - Recall: % of relevant files that are retrieved
    - Mean Reciprocal Rank (MRR): How early relevant files appear in results

    This benchmark does NOT require LLM - it uses only the compress tool's
    embedding-based semantic similarity scoring.
    """
    results_per_query: list[dict[str, Any]] = []
    latencies: list[float] = []

    for _ in range(iterations):
        for test_case in RETRIEVAL_TEST_CASES:
            query = test_case["query"]
            expected_files = set(test_case["expected_files"])

            # Score each file against the query using compress tool
            file_scores: list[tuple[str, float, float]] = []  # (filename, score, latency)

            for code_sample in CODE_CONTEXT_SAMPLES:
                filename = code_sample["file"]
                content = code_sample["content"]

                start = time.perf_counter()
                try:
                    result = await client.call_tool(
                        "compress",
                        {
                            "context": content,
                            "query": query,
                            "ratio": 0.9,  # High ratio to get relevance score without much compression
                        },
                    )
                    resp = parse_tool_response(result)
                    latency_ms = (time.perf_counter() - start) * 1000
                    latencies.append(latency_ms)

                    # Use max relevance score from sentence embeddings
                    # This directly measures semantic similarity between query and file content
                    score = resp.get("max_relevance_score", 0.0)
                    if score == 0.0:
                        # Fallback: use mean relevance score if max not available
                        score = resp.get("mean_relevance_score", 0.0)
                    file_scores.append((filename, score, latency_ms))
                except Exception:
                    file_scores.append((filename, 0.0, 0.0))

            # Rank files by score (higher = more relevant)
            ranked_files = sorted(file_scores, key=lambda x: x[1], reverse=True)
            top_k = 3  # Retrieve top 3 files
            retrieved = {f[0] for f in ranked_files[:top_k]}

            # Calculate metrics
            true_positives = len(retrieved & expected_files)
            precision = true_positives / len(retrieved) if retrieved else 0.0
            recall = true_positives / len(expected_files) if expected_files else 0.0

            # MRR: 1/rank of first relevant result
            mrr = 0.0
            for rank, (filename, _, _) in enumerate(ranked_files, 1):
                if filename in expected_files:
                    mrr = 1.0 / rank
                    break

            results_per_query.append(
                {
                    "query": query[:50] + "...",
                    "precision": precision,
                    "recall": recall,
                    "mrr": mrr,
                    "retrieved": list(retrieved),
                    "expected": list(expected_files),
                }
            )

    # Aggregate metrics
    if results_per_query:
        avg_precision = statistics.mean(r["precision"] for r in results_per_query)
        avg_recall = statistics.mean(r["recall"] for r in results_per_query)
        avg_mrr = statistics.mean(r["mrr"] for r in results_per_query)
        f1 = (
            2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0.0
        )
    else:
        avg_precision = avg_recall = avg_mrr = f1 = 0.0

    return BenchmarkResult(
        name="code_retrieval",
        category=BenchmarkCategory.COMPRESSION,
        latency=LatencyStats.from_samples(latencies) if latencies else None,
        passed=avg_precision >= 0.3,  # Passing threshold: 30% precision
        metadata={
            "precision": round(avg_precision * 100, 1),
            "recall": round(avg_recall * 100, 1),
            "f1_score": round(f1 * 100, 1),
            "mrr": round(avg_mrr, 3),
            "num_files_in_corpus": len(CODE_CONTEXT_SAMPLES),
            "num_queries": len(RETRIEVAL_TEST_CASES),
            "iterations": iterations,
            "top_k": 3,
            "query_results": results_per_query[:5],  # First 5 for details
        },
    )


async def bench_llm_reasoning_quality(client: Any, iterations: int = 3) -> BenchmarkResult:
    """Benchmark end-to-end reasoning quality with LLM.

    Compares three strategies on the same problems:
    - Baseline: Direct LLM answer (no MCP)
    - Chain: Long chain reasoning via MCP
    - Matrix: Matrix of Thought via MCP

    Measures correctness, latency, and reasoning depth.
    Requires --llm flag to be set.
    """
    if not is_llm_enabled():
        return BenchmarkResult(
            name="llm_reasoning_quality",
            category=BenchmarkCategory.STRATEGY,
            passed=True,
            metadata={"skipped": True, "reason": "LLM not enabled (use --llm flag)"},
        )

    strategy_results: dict[str, dict[str, Any]] = {
        "baseline": {"correct": 0, "total": 0, "latencies_ms": [], "coverage": []},
        "chain": {"correct": 0, "total": 0, "latencies_ms": [], "coverage": []},
        "matrix": {"correct": 0, "total": 0, "latencies_ms": [], "coverage": []},
    }

    problems = LLM_BENCHMARK_PROBLEMS * iterations

    for problem in problems:
        question = problem["question"]
        expected = problem["expected"]
        keywords = problem["keywords"]

        # 1. Baseline: Direct LLM call (no MCP reasoning)
        start = time.perf_counter()
        baseline_answer = await get_llm_answer(question)
        baseline_ms = (time.perf_counter() - start) * 1000

        is_correct, coverage = _check_answer(baseline_answer, expected, keywords)
        strategy_results["baseline"]["total"] += 1
        strategy_results["baseline"]["correct"] += int(is_correct)
        strategy_results["baseline"]["latencies_ms"].append(baseline_ms)
        strategy_results["baseline"]["coverage"].append(coverage)

        # 2. Chain: Long chain reasoning via MCP with LLM steps
        start = time.perf_counter()
        try:
            # Start chain session
            result = await client.call_tool(
                "think",
                {"action": "start", "mode": "chain", "problem": question, "expected_steps": 3},
            )
            resp = parse_tool_response(result)
            session_id = resp.get("session_id")

            if session_id:
                # Generate and add reasoning steps
                for step_num in range(1, 4):
                    step_prompt = f"Step {step_num} for: {question}"
                    step_content = await get_llm_answer(step_prompt)
                    await client.call_tool(
                        "think",
                        {"action": "continue", "session_id": session_id, "thought": step_content},
                    )

                # Get final answer
                final_prompt = f"Based on reasoning, answer: {question}"
                final_answer = await get_llm_answer(final_prompt)
                await client.call_tool(
                    "think",
                    {
                        "action": "finish",
                        "session_id": session_id,
                        "thought": final_answer,
                        "confidence": 0.9,
                    },
                )
                chain_answer = final_answer
            else:
                chain_answer = ""
        except Exception as e:
            chain_answer = f"Error: {e}"

        chain_ms = (time.perf_counter() - start) * 1000
        is_correct, coverage = _check_answer(chain_answer, expected, keywords)
        strategy_results["chain"]["total"] += 1
        strategy_results["chain"]["correct"] += int(is_correct)
        strategy_results["chain"]["latencies_ms"].append(chain_ms)
        strategy_results["chain"]["coverage"].append(coverage)

        # 3. Matrix: Matrix of Thought via MCP with LLM cells
        start = time.perf_counter()
        try:
            # Start matrix session (2x2 for speed)
            result = await client.call_tool(
                "think",
                {"action": "start", "mode": "matrix", "problem": question, "rows": 2, "cols": 2},
            )
            resp = parse_tool_response(result)
            session_id = resp.get("session_id")

            if session_id:
                perspectives = ["mathematical", "logical"]
                criteria = ["accuracy", "completeness"]

                # Fill cells with LLM-generated content
                for row, perspective in enumerate(perspectives):
                    for col, criterion in enumerate(criteria):
                        cell_prompt = (
                            f"Analyze '{question}' from {perspective} view for {criterion}"
                        )
                        cell_content = await get_llm_answer(cell_prompt)
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

                # Synthesize columns
                for col in range(2):
                    synth_prompt = f"Synthesize {criteria[col]} for: {question}"
                    synth_content = await get_llm_answer(synth_prompt)
                    await client.call_tool(
                        "think",
                        {
                            "action": "synthesize",
                            "session_id": session_id,
                            "col": col,
                            "thought": synth_content,
                        },
                    )

                # Get final answer
                final_prompt = f"Final answer for: {question}"
                final_answer = await get_llm_answer(final_prompt)
                await client.call_tool(
                    "think",
                    {
                        "action": "finish",
                        "session_id": session_id,
                        "thought": final_answer,
                        "confidence": 0.9,
                    },
                )
                matrix_answer = final_answer
            else:
                matrix_answer = ""
        except Exception as e:
            matrix_answer = f"Error: {e}"

        matrix_ms = (time.perf_counter() - start) * 1000
        is_correct, coverage = _check_answer(matrix_answer, expected, keywords)
        strategy_results["matrix"]["total"] += 1
        strategy_results["matrix"]["correct"] += int(is_correct)
        strategy_results["matrix"]["latencies_ms"].append(matrix_ms)
        strategy_results["matrix"]["coverage"].append(coverage)

    # Compute summary statistics
    summary: dict[str, dict[str, Any]] = {}
    for strategy, data in strategy_results.items():
        total = data["total"]
        correct = data["correct"]
        latencies = data["latencies_ms"]
        coverages = data["coverage"]

        summary[strategy] = {
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
            "correct": correct,
            "total": total,
            "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(
                sorted(latencies)[int(len(latencies) * 0.95)]
                if len(latencies) >= 2
                else max(latencies, default=0),
                1,
            ),
            "avg_keyword_coverage": round(statistics.mean(coverages) * 100, 1) if coverages else 0,
        }

    # Determine winner by accuracy, then by latency
    best_strategy = max(
        summary.keys(),
        key=lambda s: (summary[s]["accuracy"], -summary[s]["avg_latency_ms"]),
    )

    return BenchmarkResult(
        name="llm_reasoning_quality",
        category=BenchmarkCategory.STRATEGY,
        passed=True,
        metadata={
            "problems_per_strategy": len(problems),
            "strategies": summary,
            "winner": best_strategy,
            "winner_accuracy": summary[best_strategy]["accuracy"],
        },
    )


async def bench_cisc_confidence_impact(client: Any, iterations: int = 2) -> BenchmarkResult:
    """Benchmark CISC confidence scoring improvement over baseline.

    Compares reasoning with and without LLM confidence scores:
    - chain_baseline: Chain reasoning without confidence scores (MPPA heuristics only)
    - chain_cisc: Chain reasoning with LLM confidence scores (CISC hybrid selection)

    Measures:
    - Accuracy improvement from CISC
    - Calibration quality (WQD score)
    - Latency overhead from confidence extraction

    The CISC paper shows 11%+ improvement for smaller models - this benchmark
    validates that our implementation provides measurable benefit.

    Requires --llm flag to be set.
    """
    if not is_llm_enabled():
        return BenchmarkResult(
            name="cisc_confidence_impact",
            category=BenchmarkCategory.STRATEGY,
            passed=True,
            metadata={"skipped": True, "reason": "LLM not enabled (use --llm flag)"},
        )

    # Track results for both methods
    results: dict[str, dict[str, Any]] = {
        "chain_baseline": {
            "correct": 0,
            "total": 0,
            "latencies_ms": [],
            "coverage": [],
            "confidences": [],  # Track assigned confidences
        },
        "chain_cisc": {
            "correct": 0,
            "total": 0,
            "latencies_ms": [],
            "coverage": [],
            "confidences": [],
            "llm_confidence_time_ms": [],  # Time spent getting LLM confidences
        },
    }

    # Track calibration data: (confidence, is_correct) pairs
    calibration_data: list[tuple[float, bool]] = []

    # Use a subset of problems for this benchmark (it's expensive)
    problems = LLM_BENCHMARK_PROBLEMS[:5] * iterations

    for problem in problems:
        question = problem["question"]
        expected = problem["expected"]
        keywords = problem["keywords"]

        # =================================================================
        # 1. Chain baseline: No LLM confidence scores (MPPA heuristics only)
        # =================================================================
        start = time.perf_counter()
        try:
            result = await client.call_tool(
                "think",
                {"action": "start", "mode": "chain", "problem": question, "expected_steps": 3},
            )
            resp = parse_tool_response(result)
            session_id = resp.get("session_id")

            baseline_answer = ""
            if session_id:
                # Generate steps without confidence scores
                for step_num in range(1, 4):
                    step_prompt = f"Step {step_num} for: {question}"
                    step_content = await get_llm_answer(step_prompt)

                    # No alternatives or confidences - pure sequential chain
                    await client.call_tool(
                        "think",
                        {"action": "continue", "session_id": session_id, "thought": step_content},
                    )

                # Final answer
                final_prompt = f"Based on reasoning, answer: {question}"
                baseline_answer = await get_llm_answer(final_prompt)
                await client.call_tool(
                    "think",
                    {
                        "action": "finish",
                        "session_id": session_id,
                        "thought": baseline_answer,
                        "confidence": 0.8,  # Fixed confidence (no LLM assessment)
                    },
                )
        except Exception as e:
            baseline_answer = f"Error: {e}"

        baseline_ms = (time.perf_counter() - start) * 1000
        is_correct, coverage = _check_answer(baseline_answer, expected, keywords)
        results["chain_baseline"]["total"] += 1
        results["chain_baseline"]["correct"] += int(is_correct)
        results["chain_baseline"]["latencies_ms"].append(baseline_ms)
        results["chain_baseline"]["coverage"].append(coverage)
        results["chain_baseline"]["confidences"].append(0.8)

        # =================================================================
        # 2. Chain CISC: With LLM confidence scores and alternatives
        # =================================================================
        start = time.perf_counter()
        llm_confidence_time = 0.0
        cisc_answer = ""
        final_confidence = 0.5
        try:
            result = await client.call_tool(
                "think",
                {"action": "start", "mode": "chain", "problem": question, "expected_steps": 3},
            )
            resp = parse_tool_response(result)
            session_id = resp.get("session_id")

            if session_id:
                # Generate steps WITH confidence scores and alternatives
                for step_num in range(1, 4):
                    step_prompt = f"Step {step_num} for: {question}"
                    step_content = await get_llm_answer(step_prompt)

                    # Get LLM confidence for this step
                    conf_start = time.perf_counter()
                    step_confidence = await get_llm_confidence(step_content, question)
                    llm_confidence_time += (time.perf_counter() - conf_start) * 1000

                    # Generate alternatives with confidences (MPPA + CISC)
                    alternatives = await get_llm_alternatives(
                        question, step_content, num_alternatives=2
                    )
                    alt_thoughts = [alt[0] for alt in alternatives]
                    alt_confidences = [alt[1] for alt in alternatives]

                    # Pass alternatives and confidences to chain tool
                    await client.call_tool(
                        "think",
                        {
                            "action": "continue",
                            "session_id": session_id,
                            "thought": step_content,
                            "confidence": step_confidence,
                            "alternatives": alt_thoughts if alt_thoughts else None,
                            "alternative_confidences": alt_confidences if alt_confidences else None,
                        },
                    )

                # Final answer with confidence
                final_prompt = f"Based on reasoning, answer: {question}"
                cisc_answer = await get_llm_answer(final_prompt)

                # Get final confidence
                conf_start = time.perf_counter()
                final_confidence = await get_llm_confidence(cisc_answer, question)
                llm_confidence_time += (time.perf_counter() - conf_start) * 1000

                await client.call_tool(
                    "think",
                    {
                        "action": "finish",
                        "session_id": session_id,
                        "thought": cisc_answer,
                        "confidence": final_confidence,
                    },
                )
        except Exception as e:
            cisc_answer = f"Error: {e}"

        cisc_ms = (time.perf_counter() - start) * 1000
        is_correct, coverage = _check_answer(cisc_answer, expected, keywords)
        results["chain_cisc"]["total"] += 1
        results["chain_cisc"]["correct"] += int(is_correct)
        results["chain_cisc"]["latencies_ms"].append(cisc_ms)
        results["chain_cisc"]["coverage"].append(coverage)
        results["chain_cisc"]["confidences"].append(final_confidence)
        results["chain_cisc"]["llm_confidence_time_ms"].append(llm_confidence_time)

        # Track calibration for WQD calculation
        calibration_data.append((final_confidence, is_correct))

    # Compute summary statistics
    summary: dict[str, dict[str, Any]] = {}
    for method, data in results.items():
        total = data["total"]
        correct = data["correct"]
        latencies = data["latencies_ms"]
        coverages = data["coverage"]
        confidences = data["confidences"]

        summary[method] = {
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
            "correct": correct,
            "total": total,
            "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
            "avg_keyword_coverage": round(statistics.mean(coverages) * 100, 1) if coverages else 0,
            "avg_confidence": round(statistics.mean(confidences), 3) if confidences else 0,
        }

        # Add CISC-specific metrics
        if "llm_confidence_time_ms" in data and data["llm_confidence_time_ms"]:
            summary[method]["confidence_overhead_ms"] = round(
                statistics.mean(data["llm_confidence_time_ms"]), 1
            )

    # Calculate Within-Question Discrimination (WQD) from calibration data
    wqd_score = 0.5  # Default neutral
    if len(calibration_data) >= 4:
        correct_confs = [c for c, is_c in calibration_data if is_c]
        incorrect_confs = [c for c, is_c in calibration_data if not is_c]

        if correct_confs and incorrect_confs:
            # WQD: % of times correct answer has higher confidence than incorrect
            wins = 0
            total_pairs = 0
            for c_conf in correct_confs:
                for i_conf in incorrect_confs:
                    total_pairs += 1
                    if c_conf > i_conf:
                        wins += 1
                    elif c_conf == i_conf:
                        wins += 0.5
            wqd_score = wins / total_pairs if total_pairs > 0 else 0.5

    # Calculate improvement
    baseline_acc = summary["chain_baseline"]["accuracy"]
    cisc_acc = summary["chain_cisc"]["accuracy"]
    improvement = cisc_acc - baseline_acc

    return BenchmarkResult(
        name="cisc_confidence_impact",
        category=BenchmarkCategory.STRATEGY,
        passed=True,
        metadata={
            "problems_tested": len(problems),
            "methods": summary,
            "accuracy_improvement_pct": round(improvement, 1),
            "wqd_score": round(wqd_score, 3),
            "wqd_interpretation": (
                "excellent" if wqd_score >= 0.7 else "good" if wqd_score >= 0.6 else "moderate"
            ),
            "calibration_samples": len(calibration_data),
        },
    )


async def bench_true_cisc_voting(client: Any, iterations: int = 1) -> BenchmarkResult:
    """Benchmark TRUE CISC: multiple complete chains with weighted majority voting.

    This implements the actual CISC algorithm from the paper:
    1. Generate N complete reasoning chains per problem
    2. Extract final answer from each chain
    3. Get confidence score for each complete chain
    4. Apply weighted majority voting

    Compares:
    - majority_vote: Simple majority voting (baseline self-consistency)
    - cisc_vote: Confidence-weighted majority voting

    Uses GSM8K-style math problems where answer diversity is expected.

    Requires --llm flag to be set.
    """
    if not is_llm_enabled():
        return BenchmarkResult(
            name="true_cisc_voting",
            category=BenchmarkCategory.STRATEGY,
            passed=True,
            metadata={"skipped": True, "reason": "LLM not enabled (use --llm flag)"},
        )

    num_samples = 5  # Number of reasoning chains per problem (paper uses 5-30)

    results: dict[str, dict[str, Any]] = {
        "majority_vote": {"correct": 0, "total": 0, "latencies_ms": []},
        "cisc_vote": {"correct": 0, "total": 0, "latencies_ms": [], "changed_answer": 0},
    }

    # Use GSM8K-style problems for answer diversity
    problems = GSM8K_STYLE_PROBLEMS[:5] * iterations

    for problem in problems:
        question = problem["question"]
        expected = problem["expected"]
        keywords = problem["keywords"]

        # Generate multiple complete reasoning chains with CISC voting
        start = time.perf_counter()
        cisc_result: CISCSolveResult = await cisc_solve(
            problem=question,
            num_samples=num_samples,
            temperature=1.0,  # Standard softmax temperature
            sampling_temperature=0.7,  # LLM sampling temperature for diversity
        )
        total_ms = (time.perf_counter() - start) * 1000

        # Check majority vote result
        majority_correct, _ = _check_answer(cisc_result.majority_winner, expected, keywords)
        results["majority_vote"]["total"] += 1
        results["majority_vote"]["correct"] += int(majority_correct)
        results["majority_vote"]["latencies_ms"].append(total_ms)

        # Check CISC vote result
        cisc_correct, _ = _check_answer(cisc_result.winner, expected, keywords)
        results["cisc_vote"]["total"] += 1
        results["cisc_vote"]["correct"] += int(cisc_correct)
        results["cisc_vote"]["latencies_ms"].append(total_ms)

        if cisc_result.cisc_changed_answer:
            results["cisc_vote"]["changed_answer"] += 1

    # Compute summary
    summary: dict[str, dict[str, Any]] = {}
    for method, data in results.items():
        total = data["total"]
        correct = data["correct"]
        latencies = data["latencies_ms"]

        summary[method] = {
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
            "correct": correct,
            "total": total,
            "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        }

    # CISC-specific metrics
    summary["cisc_vote"]["answers_changed"] = results["cisc_vote"]["changed_answer"]
    summary["cisc_vote"]["change_rate"] = round(
        results["cisc_vote"]["changed_answer"] / max(results["cisc_vote"]["total"], 1) * 100, 1
    )

    # Calculate improvement
    majority_acc = summary["majority_vote"]["accuracy"]
    cisc_acc = summary["cisc_vote"]["accuracy"]
    improvement = cisc_acc - majority_acc

    return BenchmarkResult(
        name="true_cisc_voting",
        category=BenchmarkCategory.STRATEGY,
        passed=True,
        metadata={
            "problems_tested": len(problems),
            "samples_per_problem": num_samples,
            "methods": summary,
            "accuracy_improvement_pct": round(improvement, 1),
            "cisc_changed_answers": results["cisc_vote"]["changed_answer"],
        },
    )


# =============================================================================
# Report Generation
# =============================================================================


# =============================================================================
# Report Formatting Utilities
# =============================================================================


def _box_line(left: str, fill: str, mid: str, right: str, width: int = 80) -> str:
    """Create a box-drawing line."""
    return left + fill * (width - 2) + right


def _split_line(left: str, fill: str, mid: str, right: str, width: int = 80) -> str:
    """Create a box line with a middle separator."""
    half = (width - 3) // 2
    return left + fill * half + mid + fill * (width - 3 - half) + right


def _format_value(value: float, unit: str = "", precision: int = 1) -> str:
    """Format a numeric value with optional unit."""
    if value >= 1000000:
        return f"{value / 1000000:.{precision}f}M{unit}"
    elif value >= 1000:
        return f"{value / 1000:.{precision}f}K{unit}"
    else:
        return f"{value:.{precision}f}{unit}"


def _bar_graph(
    value: float, max_value: float, width: int = 20, filled: str = "█", empty: str = "░"
) -> str:
    """Create a simple horizontal bar graph."""
    if max_value <= 0:
        return empty * width
    ratio = min(1.0, value / max_value)
    filled_count = int(ratio * width)
    return filled * filled_count + empty * (width - filled_count)


def _delta_str(value: float, positive_good: bool = True) -> str:
    """Format a delta value with color indicator."""
    if value > 0:
        symbol = "▲" if positive_good else "▼"
        sign = "+"
    elif value < 0:
        symbol = "▼" if positive_good else "▲"
        sign = ""
    else:
        return "  —"
    return f"{symbol}{sign}{value:.1f}%"


def _grade(score: float, thresholds: tuple[float, float, float] = (90, 70, 50)) -> str:
    """Return a letter grade based on score."""
    if score >= thresholds[0]:
        return "A"
    elif score >= thresholds[1]:
        return "B"
    elif score >= thresholds[2]:
        return "C"
    else:
        return "D"


def _row(content: str, width: int = 80) -> str:
    """Create a properly padded row for the box."""
    # Account for the box characters
    inner_width = width - 2
    return "║" + content.ljust(inner_width)[:inner_width] + "║"


# =============================================================================
# Main Report Printer
# =============================================================================


def print_report(results: list[BenchmarkResult], verbose: bool = False) -> None:
    """Print a beautiful, user-focused benchmark dashboard.

    Presents metrics that matter for production deployment:
    - Performance: Latency percentiles, throughput, success rate
    - Efficiency: Compression savings, token reduction
    - Quality: Accuracy improvements from CISC, retrieval metrics
    - Reliability: Error rates, consistency
    """
    W = 80  # Total width

    # =========================================================================
    # Header
    # =========================================================================
    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║" + "MATRIXMIND MCP BENCHMARK RESULTS".center(W - 2) + "║")
    print("║" + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()).center(W - 2) + "║")
    print("╠" + "═" * (W - 2) + "╣")

    # =========================================================================
    # Quick Status Bar
    # =========================================================================
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    if failed == 0:
        status_bar = f"{'█' * 40}"
    else:
        pass_width = int(40 * passed / total)
        status_bar = f"{'█' * pass_width}{'░' * (40 - pass_width)}"

    print(f"║  Status: [{status_bar}] {passed}/{total:<5}".ljust(W - 1) + "║")
    print("╠" + "═" * (W - 2) + "╣")

    # =========================================================================
    # Aggregate Key Metrics
    # =========================================================================
    latency_results = [r for r in results if r.latency and r.latency.samples > 0]
    throughput_results = [r for r in results if r.throughput]
    compression_results = [r for r in results if r.compression]
    [
        r
        for r in results
        if r.category == BenchmarkCategory.STRATEGY and not r.metadata.get("skipped")
    ]

    # Calculate aggregate values
    if latency_results:
        all_p50 = [r.latency.median_ms for r in latency_results if r.latency]
        all_p95 = [r.latency.p95_ms for r in latency_results if r.latency]
        avg_p50 = statistics.mean(all_p50)
        avg_p95 = statistics.mean(all_p95)
        min_p50 = min(all_p50)
        max_p95 = max(all_p95)
    else:
        avg_p50 = avg_p95 = min_p50 = max_p95 = 0

    if throughput_results:
        max_throughput = max(
            r.throughput.ops_per_second for r in throughput_results if r.throughput
        )
        total_ops = sum(r.throughput.total_ops for r in throughput_results if r.throughput)
        total_errors = sum(r.throughput.errors for r in throughput_results if r.throughput)
        success_rate = (total_ops - total_errors) / total_ops * 100 if total_ops > 0 else 100
    else:
        max_throughput = total_ops = success_rate = 0

    if compression_results:
        total_original = sum(
            r.compression.original_tokens for r in compression_results if r.compression
        )
        total_compressed = sum(
            r.compression.compressed_tokens for r in compression_results if r.compression
        )
        total_saved = total_original - total_compressed
        avg_ratio = total_compressed / total_original if total_original > 0 else 1.0
        avg_preservation = statistics.mean(
            r.compression.preservation_score for r in compression_results if r.compression
        )
    else:
        total_saved = 0
        avg_ratio = 1.0
        avg_preservation = 0

    # =========================================================================
    # Performance Panel
    # =========================================================================
    print(_row("  PERFORMANCE"))
    print("║" + "─" * (W - 2) + "║")

    if latency_results:
        # Latency grade based on p95 (lower is better)
        lat_grade = (
            _grade(100 - avg_p95, (95, 80, 50))
            if avg_p95 < 100
            else _grade(100 - avg_p95 / 10, (95, 80, 50))
        )
        print(
            _row(
                f"    Latency         p50: {avg_p50:>7.1f} ms    p95: {avg_p95:>7.1f} ms    Grade: {lat_grade}"
            )
        )
        print(_row(f"    (response time) min: {min_p50:>7.1f} ms    max: {max_p95:>7.1f} ms"))
    else:
        print(_row("    Latency         No data available"))

    if throughput_results:
        # Throughput visualization
        tp_bar = _bar_graph(max_throughput, 10000, width=15)
        print(_row(f"    Throughput      {max_throughput:>7.0f} ops/sec   [{tp_bar}]"))
        print(_row(f"    Success Rate    {success_rate:>7.1f}%         {total_ops:>6} total ops"))
    else:
        print(_row("    Throughput      No data available"))

    print(_row(""))

    # =========================================================================
    # Efficiency Panel (Compression)
    # =========================================================================
    print("╠" + "═" * (W - 2) + "╣")
    print(_row("  TOKEN EFFICIENCY"))
    print("║" + "─" * (W - 2) + "║")

    if compression_results:
        savings_pct = (1 - avg_ratio) * 100
        savings_bar = _bar_graph(savings_pct, 100, width=20)
        print(_row(f"    Compression     {savings_pct:>5.1f}% reduction   [{savings_bar}]"))
        print(
            _row(
                f"    Tokens Saved    {_format_value(total_saved):>8} tokens   ~${total_saved * 0.00001:.4f} saved/call"
            )
        )
        print(
            _row(
                f"    Info Preserved  {avg_preservation * 100:>5.1f}%            Quality grade: {_grade(avg_preservation * 100)}"
            )
        )
    else:
        print(_row("    No compression benchmarks run"))

    print(_row(""))

    # =========================================================================
    # Quality Panel (Strategy Comparison / CISC)
    # =========================================================================
    print("╠" + "═" * (W - 2) + "╣")
    print(_row("  REASONING QUALITY"))
    print("║" + "─" * (W - 2) + "║")

    # Find strategy results
    llm_quality_result = next(
        (
            r
            for r in results
            if r.name == "llm_reasoning_quality" and r.metadata and not r.metadata.get("skipped")
        ),
        None,
    )
    cisc_impact_result = next(
        (
            r
            for r in results
            if r.name == "cisc_confidence_impact" and r.metadata and not r.metadata.get("skipped")
        ),
        None,
    )
    true_cisc_result = next(
        (
            r
            for r in results
            if r.name == "true_cisc_voting" and r.metadata and not r.metadata.get("skipped")
        ),
        None,
    )
    retrieval_result = next((r for r in results if r.name == "code_retrieval" and r.metadata), None)

    has_quality_data = any(
        [llm_quality_result, cisc_impact_result, true_cisc_result, retrieval_result]
    )

    if not has_quality_data:
        if any(
            r.metadata.get("skipped") for r in results if r.category == BenchmarkCategory.STRATEGY
        ):
            print(_row("    LLM benchmarks skipped (use --llm flag to enable)"))
        else:
            print(_row("    No quality benchmarks run"))
    else:
        # Strategy comparison
        if llm_quality_result and "strategies" in llm_quality_result.metadata:
            meta = llm_quality_result.metadata
            strategies = meta["strategies"]
            winner = meta.get("winner", "N/A")

            print(
                _row(
                    f"    Strategy Comparison ({meta.get('problems_per_strategy', '?')} problems each)"
                )
            )
            for strat, stats in strategies.items():
                marker = "*" if strat == winner else " "
                acc_bar = _bar_graph(stats["accuracy"], 100, width=10)
                print(
                    _row(
                        f"     {marker} {strat:<10} {stats['accuracy']:>5.1f}% [{acc_bar}] {stats['avg_latency_ms']:>6.0f}ms"
                    )
                )
            print(_row(""))

        # CISC Impact (step-level)
        if cisc_impact_result and "methods" in cisc_impact_result.metadata:
            meta = cisc_impact_result.metadata
            improvement = meta.get("accuracy_improvement_pct", 0)
            wqd = meta.get("wqd_score", 0.5)

            print(_row("    CISC Confidence Impact"))
            methods = meta["methods"]
            baseline_acc = methods.get("chain_baseline", {}).get("accuracy", 0)
            cisc_acc = methods.get("chain_cisc", {}).get("accuracy", 0)

            delta = _delta_str(improvement)
            print(
                _row(
                    f"      Baseline: {baseline_acc:>5.1f}%    CISC-guided: {cisc_acc:>5.1f}%    {delta}"
                )
            )
            interp = meta.get("wqd_interpretation", "unknown")
            print(_row(f"      WQD Score: {wqd:.3f}    Interpretation: {interp}"))
            print(_row(""))

        # True CISC (answer-level voting)
        if true_cisc_result and "methods" in true_cisc_result.metadata:
            meta = true_cisc_result.metadata
            methods = meta["methods"]
            samples = meta.get("samples_per_problem", "?")
            problems = meta.get("problems_tested", "?")
            improvement = meta.get("accuracy_improvement_pct", 0)

            majority_acc = methods.get("majority_vote", {}).get("accuracy", 0)
            cisc_acc = methods.get("cisc_vote", {}).get("accuracy", 0)
            answers_changed = methods.get("cisc_vote", {}).get("answers_changed", 0)

            print(_row(f"    True CISC Voting ({samples} samples x {problems} problems)"))
            print("║" + "─" * (W - 2) + "║")

            # Side-by-side comparison
            maj_bar = _bar_graph(majority_acc, 100, width=12)
            cisc_bar = _bar_graph(cisc_acc, 100, width=12)

            delta = _delta_str(improvement)
            print(_row(f"      Majority Vote:  {majority_acc:>5.1f}% [{maj_bar}]"))
            print(_row(f"      CISC Weighted:  {cisc_acc:>5.1f}% [{cisc_bar}]  {delta}"))
            print(_row(""))

            # Impact analysis
            if answers_changed > 0:
                print(_row(f"      CISC changed {answers_changed} answer(s) from majority vote"))
                if improvement > 0:
                    print(_row(f"      [+] Net improvement: +{improvement:.1f}% accuracy"))
                elif improvement < 0:
                    print(_row(f"      [-] Net regression: {improvement:.1f}% accuracy"))
                else:
                    print(_row("      [=] No net change in accuracy"))
            else:
                print(_row("      CISC agreed with majority on all problems"))
            print(_row(""))

        # Code Retrieval
        if retrieval_result and retrieval_result.metadata:
            meta = retrieval_result.metadata
            precision = meta.get("precision", 0)
            recall = meta.get("recall", 0)
            f1 = meta.get("f1_score", 0)
            mrr = meta.get("mrr", 0)

            print(_row("    Code Retrieval (Embedding-based)"))
            print(
                _row(
                    f"      Precision: {precision:>5.1f}%   Recall: {recall:>5.1f}%   F1: {f1:>5.1f}%   MRR: {mrr:.2f}"
                )
            )

    print(_row(""))

    # =========================================================================
    # Detailed Results (verbose mode or failures)
    # =========================================================================
    failed_results = [r for r in results if not r.passed]

    if failed_results:
        print("╠" + "═" * (W - 2) + "╣")
        print(_row("  FAILURES"))
        print("║" + "─" * (W - 2) + "║")
        for r in failed_results:
            print(_row(f"    [X] {r.name}"))
            if r.error:
                error_msg = r.error[: W - 12]
                print(_row(f"        {error_msg}"))
        print(_row(""))

    if verbose:
        print("╠" + "═" * (W - 2) + "╣")
        print(_row("  DETAILED RESULTS"))
        print("║" + "─" * (W - 2) + "║")

        for r in results:
            status = "[+]" if r.passed else "[X]"
            print(_row(f"    {status} {r.name} ({r.category.value})"))

            if r.latency and r.latency.samples > 0:
                lat = r.latency
                print(
                    _row(
                        f"        Latency: p50={lat.median_ms:.1f}ms p95={lat.p95_ms:.1f}ms (n={lat.samples})"
                    )
                )

            if r.throughput:
                tp = r.throughput
                sr = (tp.total_ops - tp.errors) / tp.total_ops * 100 if tp.total_ops > 0 else 0
                print(
                    _row(f"        Throughput: {tp.ops_per_second:.0f} ops/sec, {sr:.0f}% success")
                )

            if r.compression:
                comp = r.compression
                print(
                    _row(
                        f"        Compression: {comp.compression_ratio:.0%} ratio, {comp.preservation_score:.0%} preserved"
                    )
                )

        print(_row(""))

    # =========================================================================
    # Summary Footer
    # =========================================================================
    print("╠" + "═" * (W - 2) + "╣")
    print(_row("  SUMMARY"))
    print("║" + "─" * (W - 2) + "║")

    # Overall grade
    metrics_count = 0
    metrics_sum = 0

    if latency_results:
        # Lower latency is better, grade based on p95 < 50ms = A
        lat_score = max(0, 100 - avg_p95 * 2)  # p95 of 50ms = 0 score
        metrics_sum += lat_score
        metrics_count += 1

    if throughput_results:
        # Higher throughput is better
        tp_score = min(100, max_throughput / 100)  # 10K ops/sec = 100
        metrics_sum += tp_score
        metrics_count += 1

    if compression_results:
        # Compression efficiency + preservation
        comp_score = (1 - avg_ratio) * 100 * 0.5 + avg_preservation * 100 * 0.5
        metrics_sum += comp_score
        metrics_count += 1

    if success_rate > 0:
        metrics_sum += success_rate
        metrics_count += 1

    overall_score = metrics_sum / metrics_count if metrics_count > 0 else 0
    overall_grade = _grade(overall_score, (85, 70, 50))

    # Summary lines
    summary_items = []
    if latency_results:
        summary_items.append(f"p50: {avg_p50:.1f}ms")
    if throughput_results:
        summary_items.append(f"Throughput: {max_throughput:.0f}/s")
        summary_items.append(f"Success: {success_rate:.0f}%")
    if compression_results:
        summary_items.append(f"Compression: {(1 - avg_ratio) * 100:.0f}%")

    summary_line = " | ".join(summary_items)
    print(_row(f"  {summary_line}"))
    print(_row(""))

    # Final verdict
    if failed == 0:
        verdict = f"[+] ALL {total} BENCHMARKS PASSED"
        grade_display = f"Overall Grade: {overall_grade}"
    else:
        verdict = f"[!] {failed}/{total} BENCHMARKS FAILED"
        grade_display = "Review failures above"

    print(_row(f"  {verdict.center(35)} | {grade_display.center(35)}"))

    print("╚" + "═" * (W - 2) + "╝")
    print()


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
        # Baseline latency
        print("  -> Status (baseline)...")
        results.append(await bench_status(client, iterations))

        # Compression benchmarks
        print("  -> Compression latency...")
        results.append(await bench_compress_latency(client, iterations))

        # Workflow benchmarks
        print("  -> Chain workflow...")
        results.append(await bench_chain_workflow(client, iterations))

        print("  -> Matrix workflow...")
        results.append(await bench_matrix_workflow(client, iterations // 2))

        print("  -> Verify workflow...")
        results.append(await bench_verify_workflow(client, iterations // 2))

        # Scaling benchmarks (only in full mode)
        if full or compare:
            print("  -> Compression scaling...")
            results.append(await bench_compress_sizes(client))

            print("  -> Concurrent sessions...")
            results.append(await bench_concurrent_sessions(client))

            print("  -> Sustained throughput...")
            results.append(await bench_throughput(client, duration_s=10.0 if full else 5.0))

            print("  -> Code retrieval (embedding-based)...")
            retrieval_iterations = 3 if full else 1
            results.append(await bench_code_retrieval(client, iterations=retrieval_iterations))

        # LLM-powered reasoning quality benchmark (requires --llm flag)
        if is_llm_enabled():
            print("  -> LLM reasoning quality (comparing strategies)...")
            llm_iterations = 5 if full else 2
            results.append(await bench_llm_reasoning_quality(client, iterations=llm_iterations))

            print("  -> CISC confidence impact (baseline vs confidence-guided)...")
            cisc_iterations = 3 if full else 1
            results.append(await bench_cisc_confidence_impact(client, iterations=cisc_iterations))

            print("  -> True CISC voting (multiple chains with weighted voting)...")
            true_cisc_iterations = 2 if full else 1
            results.append(await bench_true_cisc_voting(client, iterations=true_cisc_iterations))
        elif compare:
            # Show skipped message in compare mode
            results.append(
                BenchmarkResult(
                    name="llm_reasoning_quality",
                    category=BenchmarkCategory.STRATEGY,
                    passed=True,
                    metadata={"skipped": True, "reason": "Use --llm to enable"},
                )
            )
            results.append(
                BenchmarkResult(
                    name="cisc_confidence_impact",
                    category=BenchmarkCategory.STRATEGY,
                    passed=True,
                    metadata={"skipped": True, "reason": "Use --llm to enable"},
                )
            )

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
  python benchmark.py --llm        Enable LLM for realistic benchmark inputs
        """,
    )
    parser.add_argument("--full", "-f", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--compare", "-c", action="store_true", help="Include comparison tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--export", "-e", type=str, metavar="FILE", help="Export results to JSON")
    add_llm_args(parser)
    args = parser.parse_args()

    print("=" * 78)
    print("              MATRIXMIND MCP PERFORMANCE BENCHMARK")
    print("=" * 78)

    # Initialize LLM client if requested (for future LLM-powered benchmarks)
    if args.llm:
        print(f"\nLLM Mode: Using {args.llm_model or 'default'} at {args.llm_url or 'default'}")
        await init_llm_client(base_url=args.llm_url, model=args.llm_model)
    else:
        print("\nSimulation Mode: Using fixed inputs (use --llm for realistic inputs)")

    try:
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
    finally:
        await close_llm_client()


if __name__ == "__main__":
    asyncio.run(main())
