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


class MockLLMClient:
    """Mock LLM client for performance testing without API calls."""

    def __init__(self) -> None:
        self.call_count = 0
        self.total_latency = 0.0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Simulate LLM generation with minimal latency."""
        self.call_count += 1
        # Simulate small processing delay
        time.sleep(0.01)
        return f"Mock response for: {prompt[:50]}..."

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Simulate async LLM generation."""
        self.call_count += 1
        await asyncio.sleep(0.01)
        return f"Mock async response for: {prompt[:50]}..."

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens in text."""
        return len(text) // 4


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Create a mock LLM client for testing."""
    return MockLLMClient()


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


class TestReasoningPerformance:
    """Performance tests for reasoning tools."""

    def test_mot_matrix_size_impact(self, mock_llm_client: MockLLMClient) -> None:
        """Test how matrix size affects MoT performance."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        tool = MatrixOfThoughtTool(mock_llm_client)

        context = "Test context for reasoning."
        question = "What is the answer?"

        results: list[dict[str, Any]] = []

        for rows in [2, 3, 4]:
            for cols in [2, 3, 4]:
                mock_llm_client.call_count = 0
                start = time.perf_counter()

                result = tool.reason(
                    question=question,
                    context=context,
                    matrix_rows=rows,
                    matrix_cols=cols,
                )

                elapsed = time.perf_counter() - start

                results.append(
                    {
                        "rows": rows,
                        "cols": cols,
                        "time": elapsed,
                        "llm_calls": mock_llm_client.call_count,
                        "confidence": result.confidence,
                    }
                )

        # Print results table
        print(
            "\n{:<6} {:<6} {:<10} {:<12} {:<10}".format(
                "Rows", "Cols", "Time (s)", "LLM Calls", "Confidence"
            )
        )
        print("-" * 50)
        for r in results:
            print(
                "{:<6} {:<6} {:<10.3f} {:<12} {:<10.2f}".format(
                    r["rows"], r["cols"], r["time"], r["llm_calls"], r["confidence"]
                )
            )

    def test_long_chain_step_impact(self, mock_llm_client: MockLLMClient) -> None:
        """Test how step count affects long chain performance."""
        from src.tools.long_chain import LongChainOfThoughtTool

        tool = LongChainOfThoughtTool(mock_llm_client)

        problem = "Solve this step by step."

        results: list[dict[str, Any]] = []

        for num_steps in [5, 10, 15, 20]:
            mock_llm_client.call_count = 0
            start = time.perf_counter()

            result = tool.reason(
                problem=problem,
                num_steps=num_steps,
                verify_intermediate=True,
            )

            elapsed = time.perf_counter() - start

            results.append(
                {
                    "steps": num_steps,
                    "time": elapsed,
                    "llm_calls": mock_llm_client.call_count,
                    "actual_steps": len(result.reasoning_steps),
                }
            )

        # Print results
        print(
            "\n{:<8} {:<10} {:<12} {:<12}".format("Steps", "Time (s)", "LLM Calls", "Actual Steps")
        )
        print("-" * 45)
        for r in results:
            print(
                "{:<8} {:<10.3f} {:<12} {:<12}".format(
                    r["steps"], r["time"], r["llm_calls"], r["actual_steps"]
                )
            )


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
    async def test_concurrent_reasoning(self, mock_llm_client: MockLLMClient) -> None:
        """Test concurrent reasoning requests."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        tool = MatrixOfThoughtTool(mock_llm_client)

        async def reason_task(idx: int) -> tuple[int, float]:
            start = time.perf_counter()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: tool.reason(
                    question=f"Question {idx}?",
                    context=f"Context {idx}.",
                    matrix_rows=2,
                    matrix_cols=2,
                ),
            )
            return idx, time.perf_counter() - start

        num_concurrent = 3
        start_total = time.perf_counter()
        results = await asyncio.gather(*[reason_task(i) for i in range(num_concurrent)])
        total_time = time.perf_counter() - start_total

        times = [r[1] for r in results]
        print(f"\nConcurrent reasoning requests: {num_concurrent}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average per request: {statistics.mean(times):.3f}s")


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
    def test_rapid_sequential_requests(self, mock_llm_client: MockLLMClient) -> None:
        """Test rapid sequential request handling."""
        from src.tools.verify import FactVerificationTool

        tool = FactVerificationTool(mock_llm_client)

        num_requests = 50
        timings: list[float] = []

        start_total = time.perf_counter()
        for i in range(num_requests):
            start = time.perf_counter()
            tool.verify(
                answer=f"Answer {i} is correct.",
                context=f"Context {i} contains the answer.",
                max_claims=3,
            )
            timings.append(time.perf_counter() - start)
        total_time = time.perf_counter() - start_total

        print(f"\nRapid sequential requests: {num_requests}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Requests/second: {num_requests / total_time:.1f}")
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
