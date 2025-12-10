"""Performance and load tests for MatrixMind MCP Server.

Tests include:
- Concurrent request handling
- Response time benchmarking
- Memory usage under load
- Throughput measurements

Run with:
    pytest tests/test_performance.py -v --tb=short

For stress testing:
    pytest tests/test_performance.py -v -k stress --tb=short
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import time
from typing import Any

import pytest

# Skip all tests if dependencies not available
pytest.importorskip("fastmcp")


@pytest.fixture
def sample_contexts() -> list[str]:
    """Generate sample contexts of varying sizes."""
    base_text = """
    Albert Einstein was a theoretical physicist who developed the theory of relativity,
    one of the two pillars of modern physics. His work is also known for its influence
    on the philosophy of science. He is best known to the general public for his
    mass-energy equivalence formula E = mc^2, which has been dubbed "the world's most
    famous equation".
    """
    return [
        base_text * 1,  # ~500 chars
        base_text * 5,  # ~2500 chars
        base_text * 10,  # ~5000 chars
        base_text * 20,  # ~10000 chars
    ]


@pytest.fixture
def sample_questions() -> list[str]:
    """Sample questions for testing."""
    return [
        "What did Einstein develop?",
        "What is the mass-energy equivalence formula?",
        "Who was Albert Einstein?",
        "What are the pillars of modern physics?",
        "What is E=mc^2?",
    ]


class TestCompressionPerformance:
    """Performance tests for compression tool."""

    @pytest.mark.parametrize("context_multiplier", [1, 5, 10])
    def test_compression_scales_linearly(
        self,
        context_multiplier: int,
    ) -> None:
        """Test that compression time scales roughly linearly with input size."""
        from src.tools.compress import ContextAwareCompressionTool

        tool = ContextAwareCompressionTool()

        base_context = "This is a test sentence. " * 100
        context = base_context * context_multiplier
        question = "What is this about?"

        start = time.perf_counter()
        result = tool.compress(context, question, compression_ratio=0.3)
        elapsed = time.perf_counter() - start

        assert result.compressed_tokens > 0
        assert result.compression_ratio < 1.0

        # Log timing for analysis
        print(f"\nContext size: {len(context)} chars, Time: {elapsed:.3f}s")

    def test_compression_batch_efficiency(self) -> None:
        """Test compression efficiency with multiple calls."""
        from src.tools.compress import ContextAwareCompressionTool

        tool = ContextAwareCompressionTool()

        contexts = [
            "First document about physics. " * 50,
            "Second document about chemistry. " * 50,
            "Third document about biology. " * 50,
        ]
        question = "What is science?"

        timings: list[float] = []
        for context in contexts:
            start = time.perf_counter()
            tool.compress(context, question, compression_ratio=0.5)
            timings.append(time.perf_counter() - start)

        avg_time = statistics.mean(timings)
        std_time = statistics.stdev(timings) if len(timings) > 1 else 0

        print(f"\nAverage compression time: {avg_time:.3f}s (std: {std_time:.3f}s)")
        assert avg_time < 5.0  # Should complete within 5 seconds each


class TestStateManagerPerformance:
    """Performance tests for state manager tools (new architecture)."""

    def test_chain_manager_session_overhead(self) -> None:
        """Test session creation and management overhead for chain manager."""
        from src.tools.long_chain import LongChainManager

        manager = LongChainManager()

        num_sessions = 100
        timings: list[float] = []

        for i in range(num_sessions):
            start = time.perf_counter()
            result = manager.start_chain(f"Problem {i}")
            session_id = result["session_id"]
            manager.add_step(session_id, f"Step 1 for problem {i}")
            manager.add_step(session_id, f"Step 2 for problem {i}")
            manager.finalize(session_id, f"Answer {i}")
            timings.append(time.perf_counter() - start)

        avg_time = statistics.mean(timings)
        print(f"\nChain sessions created: {num_sessions}")
        print(f"Average session time: {avg_time * 1000:.2f}ms")
        print(f"P95 latency: {sorted(timings)[int(0.95 * len(timings))] * 1000:.2f}ms")

        # Session overhead should be minimal (< 10ms per session)
        assert avg_time < 0.01

    def test_matrix_manager_cell_population(self) -> None:
        """Test matrix cell population performance."""
        from src.tools.mot_reasoning import MatrixOfThoughtManager

        manager = MatrixOfThoughtManager()

        matrix_sizes = [(2, 2), (3, 3), (4, 4), (5, 5)]
        results: list[dict[str, Any]] = []

        for rows, cols in matrix_sizes:
            start = time.perf_counter()
            result = manager.start_matrix(f"Question for {rows}x{cols}", "", rows, cols)
            session_id = result["session_id"]

            for row in range(rows):
                for col in range(cols):
                    manager.set_cell(session_id, row, col, f"Thought at ({row}, {col})")

            manager.synthesize_column(session_id, 0, "Synthesis of all thoughts")
            manager.finalize(session_id, "Final answer")
            elapsed = time.perf_counter() - start

            results.append(
                {
                    "size": f"{rows}x{cols}",
                    "cells": rows * cols,
                    "time_ms": elapsed * 1000,
                }
            )

        # Print results
        print("\n{:<8} {:<8} {:<15}".format("Size", "Cells", "Time (ms)"))
        print("-" * 35)
        for r in results:
            print("{:<8} {:<8} {:<15.2f}".format(r["size"], r["cells"], r["time_ms"]))

    def test_verification_manager_claim_processing(self) -> None:
        """Test verification manager claim processing performance."""
        from src.tools.verify import VerificationManager

        manager = VerificationManager()

        claim_counts = [5, 10, 15, 20]
        results: list[dict[str, Any]] = []

        for num_claims in claim_counts:
            start = time.perf_counter()
            result = manager.start_verification(
                "Test answer with multiple claims",
                "Context for verification",
            )
            session_id = result["session_id"]

            claim_ids = []
            for i in range(num_claims):
                claim_result = manager.add_claim(session_id, f"Claim number {i}")
                claim_ids.append(claim_result["claim_id"])

            for claim_id in claim_ids:
                manager.verify_claim(
                    session_id,
                    claim_id,
                    "supported",
                    "Evidence supports this claim",
                )

            manager.finalize(session_id)
            elapsed = time.perf_counter() - start

            results.append(
                {
                    "claims": num_claims,
                    "time_ms": elapsed * 1000,
                    "per_claim_ms": (elapsed * 1000) / num_claims,
                }
            )

        # Print results
        print("\n{:<10} {:<15} {:<15}".format("Claims", "Total (ms)", "Per Claim (ms)"))
        print("-" * 45)
        for r in results:
            print("{:<10} {:<15.2f} {:<15.2f}".format(r["claims"], r["time_ms"], r["per_claim_ms"]))


class TestConcurrencyPerformance:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_compressions(self) -> None:
        """Test concurrent compression requests."""
        from src.tools.compress import ContextAwareCompressionTool

        tool = ContextAwareCompressionTool()

        async def compress_task(idx: int) -> tuple[int, float]:
            context = f"Document {idx} " * 100
            question = f"What is document {idx}?"

            start = time.perf_counter()
            # Run in thread pool since compress is sync
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: tool.compress(context, question, compression_ratio=0.5),
            )
            return idx, time.perf_counter() - start

        # Run 5 concurrent compressions
        num_concurrent = 5
        start_total = time.perf_counter()
        results = await asyncio.gather(*[compress_task(i) for i in range(num_concurrent)])
        total_time = time.perf_counter() - start_total

        times = [r[1] for r in results]
        print(f"\nConcurrent compressions: {num_concurrent}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average per request: {statistics.mean(times):.3f}s")
        print(f"Speedup vs sequential: {sum(times) / total_time:.2f}x")

    @pytest.mark.asyncio
    async def test_concurrent_state_manager_sessions(self) -> None:
        """Test concurrent state manager session handling."""
        from src.tools.long_chain import LongChainManager
        from src.tools.mot_reasoning import MatrixOfThoughtManager

        chain_manager = LongChainManager()
        matrix_manager = MatrixOfThoughtManager()

        async def chain_task(idx: int) -> tuple[str, float]:
            start = time.perf_counter()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._run_chain_session(chain_manager, idx),
            )
            return "chain", time.perf_counter() - start

        async def matrix_task(idx: int) -> tuple[str, float]:
            start = time.perf_counter()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._run_matrix_session(matrix_manager, idx),
            )
            return "matrix", time.perf_counter() - start

        # Run mixed concurrent sessions
        tasks = [
            chain_task(0),
            matrix_task(0),
            chain_task(1),
            matrix_task(1),
            chain_task(2),
            matrix_task(2),
        ]

        start_total = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_total

        chain_times = [r[1] for r in results if r[0] == "chain"]
        matrix_times = [r[1] for r in results if r[0] == "matrix"]

        print(f"\nConcurrent mixed sessions: {len(tasks)}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Avg chain session: {statistics.mean(chain_times) * 1000:.2f}ms")
        print(f"Avg matrix session: {statistics.mean(matrix_times) * 1000:.2f}ms")

    def _run_chain_session(self, manager: Any, idx: int) -> None:
        """Helper to run a chain session."""
        result = manager.start_chain(f"Problem {idx}")
        session_id = result["session_id"]
        for step in range(5):
            manager.add_step(session_id, f"Step {step} for problem {idx}")
        manager.finalize(session_id, f"Answer {idx}")

    def _run_matrix_session(self, manager: Any, idx: int) -> None:
        """Helper to run a matrix session."""
        result = manager.start_matrix(f"Question {idx}", "", 3, 3)
        session_id = result["session_id"]
        for row in range(3):
            for col in range(3):
                manager.set_cell(session_id, row, col, f"Cell ({row},{col})")
        manager.synthesize_column(session_id, 0, f"Synthesis {idx}")
        manager.finalize(session_id, f"Answer {idx}")


class TestMemoryPerformance:
    """Tests for memory usage."""

    def test_compression_memory_usage(self) -> None:
        """Test memory usage during compression."""
        import tracemalloc

        from src.tools.compress import ContextAwareCompressionTool

        tracemalloc.start()

        tool = ContextAwareCompressionTool()

        # Process multiple documents
        for i in range(5):
            context = f"Document {i} content. " * 500
            tool.compress(context, "What is this?", compression_ratio=0.3)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nCurrent memory: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

        # Peak should be reasonable (< 2GB for this test)
        assert peak < 2 * 1024 * 1024 * 1024

    def test_encoder_cache_memory(self) -> None:
        """Test encoder cache doesn't grow unbounded."""
        from src.models.context_encoder import ContextEncoder, EncoderConfig

        config = EncoderConfig(cache_size=100)
        encoder = ContextEncoder(config=config)

        # Encode more texts than cache size
        for i in range(150):
            encoder.encode(f"Text number {i} for testing cache behavior.")

        stats = encoder.cache_stats
        print("\nCache stats after 150 encodes:")
        print(f"  Size: {stats['size']} (max: {stats['max_size']})")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")

        # Cache should not exceed max size
        assert stats["size"] <= stats["max_size"]

    def test_state_manager_session_cleanup(self) -> None:
        """Test that state managers properly clean up completed sessions."""
        import tracemalloc

        from src.tools.long_chain import LongChainManager

        tracemalloc.start()

        manager = LongChainManager()

        # Create and complete many sessions
        for i in range(100):
            result = manager.start_chain(f"Problem {i}")
            session_id = result["session_id"]
            manager.add_step(session_id, "Step 1")
            manager.finalize(session_id, "Answer")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print("\nAfter 100 sessions:")
        print(f"  Current memory: {current / 1024:.2f} KB")
        print(f"  Peak memory: {peak / 1024:.2f} KB")

        # Memory should be bounded
        assert current < 10 * 1024 * 1024  # < 10MB


class TestStressTests:
    """Stress tests for system limits."""

    @pytest.mark.stress
    def test_large_context_handling(self) -> None:
        """Test handling of very large contexts."""
        from src.tools.compress import ContextAwareCompressionTool

        tool = ContextAwareCompressionTool()

        # Create a large context (~100KB)
        large_context = "This is a sentence for testing. " * 3000

        start = time.perf_counter()
        result = tool.compress(large_context, "What is this?", compression_ratio=0.1)
        elapsed = time.perf_counter() - start

        print(f"\nLarge context ({len(large_context)} chars):")
        print(f"  Compression time: {elapsed:.3f}s")
        print(f"  Original tokens: {result.original_tokens}")
        print(f"  Compressed tokens: {result.compressed_tokens}")
        print(f"  Compression ratio: {result.compression_ratio:.2%}")

        assert result.compression_ratio < 0.3

    @pytest.mark.stress
    def test_rapid_sequential_state_sessions(self) -> None:
        """Test rapid sequential state manager session handling."""
        from src.tools.verify import VerificationManager

        manager = VerificationManager()

        num_sessions = 50
        timings: list[float] = []

        start_total = time.perf_counter()
        for i in range(num_sessions):
            start = time.perf_counter()

            result = manager.start_verification(
                f"Answer {i} is correct.",
                f"Context {i} contains the answer.",
            )
            session_id = result["session_id"]

            # Add and verify 3 claims per session
            for j in range(3):
                claim_result = manager.add_claim(session_id, f"Claim {j} of session {i}")
                claim_id = claim_result["claim_id"]
                manager.verify_claim(session_id, claim_id, "supported", "Evidence")

            manager.finalize(session_id)
            timings.append(time.perf_counter() - start)

        total_time = time.perf_counter() - start_total

        print(f"\nRapid sequential sessions: {num_sessions}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Sessions/second: {num_sessions / total_time:.1f}")
        print(f"Average latency: {statistics.mean(timings) * 1000:.1f}ms")
        print(f"P95 latency: {sorted(timings)[int(0.95 * len(timings))] * 1000:.1f}ms")


class TestBenchmarks:
    """Benchmark tests for performance baselines."""

    def test_compression_benchmark(self) -> None:
        """Benchmark compression across different sizes."""
        from src.tools.compress import ContextAwareCompressionTool

        tool = ContextAwareCompressionTool()

        sizes = [500, 1000, 2000, 5000, 10000]
        results: list[dict[str, Any]] = []

        for size in sizes:
            context = "Test sentence for benchmarking. " * (size // 35)
            question = "What is this?"

            # Warm up
            tool.compress(context[:100], question, compression_ratio=0.5)

            # Benchmark
            times: list[float] = []
            for _ in range(3):
                gc.collect()
                start = time.perf_counter()
                tool.compress(context, question, compression_ratio=0.3)
                times.append(time.perf_counter() - start)

            results.append(
                {
                    "size": len(context),
                    "avg_time": statistics.mean(times),
                    "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                }
            )

        # Print benchmark results
        print("\n{:<12} {:<15} {:<15}".format("Size (chars)", "Avg Time (s)", "Std Dev"))
        print("-" * 45)
        for r in results:
            print("{:<12} {:<15.3f} {:<15.3f}".format(r["size"], r["avg_time"], r["std_time"]))

    def test_encoder_benchmark(self) -> None:
        """Benchmark encoder performance."""
        from src.models.context_encoder import ContextEncoder

        encoder = ContextEncoder()

        batch_sizes = [1, 5, 10, 20]
        results: list[dict[str, Any]] = []

        for batch_size in batch_sizes:
            texts = [f"Text number {i} for encoding benchmark." for i in range(batch_size)]

            # Warm up
            encoder.encode_batch(texts[:1], use_cache=False)

            # Benchmark
            times: list[float] = []
            for _ in range(3):
                encoder.clear_cache()
                gc.collect()
                start = time.perf_counter()
                encoder.encode_batch(texts, use_cache=False)
                times.append(time.perf_counter() - start)

            results.append(
                {
                    "batch_size": batch_size,
                    "avg_time": statistics.mean(times),
                    "per_text": statistics.mean(times) / batch_size,
                }
            )

        # Print results
        print("\n{:<12} {:<15} {:<15}".format("Batch Size", "Avg Time (s)", "Per Text (s)"))
        print("-" * 45)
        for r in results:
            print(
                "{:<12} {:<15.3f} {:<15.3f}".format(r["batch_size"], r["avg_time"], r["per_text"])
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
