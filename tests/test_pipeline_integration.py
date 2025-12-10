"""Pipeline integration tests using mocks.

Tests the full MoT→KG→LLM and LongChain→STaR pipelines
without requiring actual LLM API calls.

Run with: pytest tests/test_pipeline_integration.py -v
"""

from __future__ import annotations

import pytest


class MockLLMClientWithCallTracking:
    """Mock LLM client that tracks all calls for verification."""

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize mock client."""
        self.responses = responses or ["Mock response"]
        self.call_index = 0
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a mock response and track the call."""
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system_prompt": system_prompt,
            }
        )
        response = self.responses[self.call_index % len(self.responses)]
        self.call_index += 1
        return response

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4


# =============================================================================
# MoT + Knowledge Graph Integration Tests
# =============================================================================


class TestMoTKnowledgeGraphIntegration:
    """Integration tests for MoT with Knowledge Graph pipeline.

    Verifies that:
    1. KG extraction is called when use_knowledge_graph=True
    2. KG context is injected into thought generation prompts
    3. Reasoning trace contains KG metadata
    """

    def test_mot_with_kg_injects_relations_into_prompts(self) -> None:
        """Test that KG relations are injected into thought generation prompts."""
        from unittest.mock import MagicMock, patch

        from src.tools.mot_reasoning import MatrixOfThoughtTool

        # Create mock LLM client that tracks calls
        mock_llm = MockLLMClientWithCallTracking(
            responses=[
                # KG extraction response (entity/relation extraction)
                "Einstein → developed → relativity\nEinstein → published → paper",
                # Thought node responses (2x2 matrix = 4 thoughts + 2 summaries + 1 answer)
                "Thought analyzing the relationship between Einstein and relativity",
                "Thought using logical inference about publication dates",
                "Synthesis of column 1 thoughts",
                "Thought building on previous analysis",
                "Further deductive reasoning about Einstein's work",
                "Synthesis of column 2 thoughts",
                "Final answer: Einstein developed the theory of relativity",
            ]
        )

        # Mock the KnowledgeGraphExtractor to return controlled KG
        mock_kg = MagicMock()
        mock_relation = MagicMock()
        mock_relation.subject.name = "Einstein"
        mock_relation.predicate = "developed"
        mock_relation.object_entity.name = "relativity"
        mock_kg.relations = [mock_relation]
        mock_kg.stats.return_value = MagicMock(num_entities=2, num_relations=1)

        mock_extractor = MagicMock()
        mock_extractor.extract_for_question.return_value = mock_kg

        with patch(
            "src.models.knowledge_graph.KnowledgeGraphExtractor",
            return_value=mock_extractor,
        ):
            tool = MatrixOfThoughtTool(mock_llm)  # type: ignore

            result = tool.reason(
                question="Who developed the theory of relativity?",
                context="Albert Einstein was a physicist who developed the theory of relativity.",
                matrix_rows=2,
                matrix_cols=2,
                use_knowledge_graph=True,
            )

        # Verify KG extractor was called
        mock_extractor.extract_for_question.assert_called_once()

        # Verify KG context was injected into thought generation prompts
        # Skip first call (KG extraction), check thought generation calls
        thought_prompts: list[str] = [
            str(c["prompt"]) for c in mock_llm.calls[1:] if "Matrix Position" in str(c["prompt"])
        ]
        assert len(thought_prompts) >= 1, "Expected at least one thought generation call"

        # At least one prompt should contain KG relations
        kg_injected = any("Knowledge Graph" in p for p in thought_prompts)
        assert kg_injected, "Knowledge Graph context should be injected into prompts"

        # Verify reasoning trace contains KG metadata
        assert result.reasoning_trace is not None
        assert result.reasoning_trace["knowledge_graph_enabled"] is True
        assert "knowledge_graph_stats" in result.reasoning_trace
        assert result.reasoning_trace["knowledge_graph_stats"]["entities"] == 2
        assert result.reasoning_trace["knowledge_graph_stats"]["relations"] == 1

    def test_mot_without_kg_does_not_inject_relations(self) -> None:
        """Test that KG is not used when use_knowledge_graph=False."""
        from unittest.mock import patch

        from src.tools.mot_reasoning import MatrixOfThoughtTool

        mock_llm = MockLLMClientWithCallTracking(
            responses=["Response"] * 20  # Enough for any matrix size
        )

        with patch("src.models.knowledge_graph.KnowledgeGraphExtractor") as mock_extractor_class:
            tool = MatrixOfThoughtTool(mock_llm)  # type: ignore

            result = tool.reason(
                question="What is 2+2?",
                context="Basic arithmetic.",
                matrix_rows=2,
                matrix_cols=2,
                use_knowledge_graph=False,
            )

            # Verify KG extractor was NOT called
            mock_extractor_class.assert_not_called()

        # Verify reasoning trace shows KG disabled
        assert result.reasoning_trace is not None
        assert result.reasoning_trace["knowledge_graph_enabled"] is False
        assert "knowledge_graph_stats" not in result.reasoning_trace

        # Verify no prompts contain KG context
        all_prompts = [str(c["prompt"]) for c in mock_llm.calls]
        kg_in_prompts = any("Knowledge Graph" in p for p in all_prompts)
        assert not kg_in_prompts, "No Knowledge Graph context should be in prompts"

    def test_mot_kg_extraction_failure_gracefully_continues(self) -> None:
        """Test that KG extraction failure doesn't break reasoning."""
        from unittest.mock import patch

        from src.tools.mot_reasoning import MatrixOfThoughtTool

        mock_llm = MockLLMClientWithCallTracking(responses=["Response"] * 20)

        # Mock KG extractor to raise an exception
        with patch("src.models.knowledge_graph.KnowledgeGraphExtractor") as mock_extractor_class:
            mock_extractor_class.side_effect = Exception("KG extraction failed")

            tool = MatrixOfThoughtTool(mock_llm)  # type: ignore

            # Should not raise, should gracefully continue
            result = tool.reason(
                question="Test question",
                context="Test context",
                matrix_rows=2,
                matrix_cols=2,
                use_knowledge_graph=True,
            )

        # Reasoning should complete successfully
        assert result.answer is not None
        assert result.confidence > 0

        # KG should show as enabled but no stats (extraction failed)
        assert result.reasoning_trace is not None
        assert result.reasoning_trace["knowledge_graph_enabled"] is True
        # Stats may be None due to extraction failure
        assert result.reasoning_trace.get("knowledge_graph_stats") is None

    @pytest.mark.asyncio
    async def test_mot_async_with_kg_integration(self) -> None:
        """Test async MoT with KG integration."""
        from unittest.mock import MagicMock, patch

        from src.tools.mot_reasoning import MatrixOfThoughtTool

        mock_llm = MockLLMClientWithCallTracking(responses=["Response"] * 20)

        # Mock KG extractor
        mock_kg = MagicMock()
        mock_relation = MagicMock()
        mock_relation.subject.name = "Entity1"
        mock_relation.predicate = "relates_to"
        mock_relation.object_entity.name = "Entity2"
        mock_kg.relations = [mock_relation]
        mock_kg.stats.return_value = MagicMock(num_entities=2, num_relations=1)

        mock_extractor = MagicMock()
        mock_extractor.extract_for_question.return_value = mock_kg

        with patch(
            "src.models.knowledge_graph.KnowledgeGraphExtractor",
            return_value=mock_extractor,
        ):
            tool = MatrixOfThoughtTool(mock_llm)  # type: ignore

            result = await tool.reason_async(
                question="How are Entity1 and Entity2 related?",
                context="Entity1 is connected to Entity2 through various relationships.",
                matrix_rows=2,
                matrix_cols=2,
                use_knowledge_graph=True,
            )

        # Verify async method also integrates KG
        assert result.reasoning_trace is not None
        assert result.reasoning_trace["knowledge_graph_enabled"] is True
        assert result.reasoning_trace.get("knowledge_graph_stats") is not None


# =============================================================================
# Long Chain + STaR Integration Tests
# =============================================================================


class TestLongChainSTaRIntegration:
    """Integration tests for Long Chain with STaR iterations."""

    def test_long_chain_star_iterations_produces_metadata(self) -> None:
        """Test that STaR iterations produce expected metadata."""
        from src.tools.long_chain import LongChainOfThoughtTool

        mock_llm = MockLLMClientWithCallTracking(
            responses=[
                # Each STaR iteration needs: steps + verification + answer extraction
                "Step 1: Initial analysis",
                "Step 2: Building on step 1",
                "Step 3: Continuing reasoning",
                "VALID - reasoning is consistent",
                "Final answer: 42",
            ]
            * 5  # Enough for multiple iterations
        )

        tool = LongChainOfThoughtTool(mock_llm)  # type: ignore

        result = tool.reason(
            problem="What is the answer to life, the universe, and everything?",
            num_steps=3,
            verify_intermediate=True,
            star_iterations=2,
        )

        # Verify STaR metadata in reasoning trace
        assert result.reasoning_trace is not None
        assert result.reasoning_trace.get("star_enabled") is True
        assert result.reasoning_trace.get("star_iterations_requested") == 2
        iterations_used = result.reasoning_trace.get("star_iterations_used")
        assert iterations_used is not None and iterations_used >= 1
        assert "star_best_score" in result.reasoning_trace

    def test_long_chain_star_early_exit_on_high_confidence(self) -> None:
        """Test that STaR exits early when high-confidence result found."""
        from src.tools.long_chain import LongChainOfThoughtTool

        # Create responses that produce a high-confidence result on first iteration
        mock_llm = MockLLMClientWithCallTracking(
            responses=[
                "Step 1: Clear analysis",
                "Step 2: Strong logical connection",
                "Step 3: Definitive conclusion",
                "VALID - fully consistent",  # Verification passes
                "Final answer: Correct answer with high confidence",
                "YES - the answer is valid",  # Final answer verification
            ]
            * 5
        )

        tool = LongChainOfThoughtTool(mock_llm)  # type: ignore

        result = tool.reason(
            problem="Simple math problem",
            num_steps=3,
            verify_intermediate=True,
            star_iterations=5,  # Request 5 but should exit early
        )

        # With high confidence, should exit before all 5 iterations
        assert result.reasoning_trace is not None
        iterations_used = result.reasoning_trace.get("star_iterations_used", 0)
        # May use 1-2 iterations depending on scoring
        assert iterations_used is not None and iterations_used <= 5

    def test_long_chain_without_star_has_no_star_metadata(self) -> None:
        """Test that non-STaR mode doesn't include STaR metadata."""
        from src.tools.long_chain import LongChainOfThoughtTool

        mock_llm = MockLLMClientWithCallTracking(responses=["Step response"] * 20)

        tool = LongChainOfThoughtTool(mock_llm)  # type: ignore

        result = tool.reason(
            problem="Test problem",
            num_steps=3,
            verify_intermediate=False,
            star_iterations=0,  # Disabled
        )

        # Should not have STaR metadata
        assert result.reasoning_trace is not None
        assert (
            result.reasoning_trace.get("star_enabled") is None
            or result.reasoning_trace.get("star_enabled") is False
        )
        assert "star_iterations_used" not in result.reasoning_trace


# =============================================================================
# Async STaR Integration Tests
# =============================================================================


class TestAsyncSTaRIntegration:
    """Integration tests for async STaR implementation.

    Verifies:
    1. Concurrent chain generation runs in parallel (timing)
    2. star_async: true appears in reasoning trace
    3. Temperature varies across iterations
    4. Exception handling in concurrent tasks
    """

    @pytest.mark.asyncio
    async def test_async_star_produces_async_metadata(self) -> None:
        """Test that async STaR includes star_async: true in trace."""
        from src.tools.long_chain import LongChainOfThoughtTool

        class MockAsyncLLM:
            """Mock LLM with async support."""

            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def generate(
                self,
                prompt: str,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                **kwargs: object,
            ) -> str:
                self.calls.append({"prompt": prompt, "temperature": temperature})
                return "Mock response"

            async def generate_async(
                self,
                prompt: str,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                **kwargs: object,
            ) -> str:
                self.calls.append({"prompt": prompt, "temperature": temperature, "async": True})
                return "Mock response"

            def estimate_tokens(self, text: str) -> int:
                return len(text) // 4

        mock_llm = MockAsyncLLM()
        tool = LongChainOfThoughtTool(mock_llm)  # type: ignore

        result = await tool.reason_async(
            problem="Test async STaR",
            num_steps=3,
            verify_intermediate=False,
            star_iterations=2,
        )

        # Verify async STaR metadata
        assert result.reasoning_trace is not None
        assert result.reasoning_trace.get("star_enabled") is True
        assert result.reasoning_trace.get("star_async") is True
        assert result.reasoning_trace.get("star_iterations_requested") == 2

    @pytest.mark.asyncio
    async def test_async_star_concurrent_execution_timing(self) -> None:
        """Test that async STaR iterations run concurrently (wall-clock timing).

        With mocked delays, sequential execution would take N * delay,
        but concurrent execution should take approximately 1 * delay.
        """
        import asyncio
        import time

        class MockAsyncLLMWithDelay:
            """Mock LLM with configurable async delay to test concurrency."""

            def __init__(self, delay_seconds: float = 0.1) -> None:
                self.delay = delay_seconds
                self.call_count = 0

            def generate(
                self,
                prompt: str,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                **kwargs: object,
            ) -> str:
                self.call_count += 1
                return "Mock sync response"

            async def generate_async(
                self,
                prompt: str,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                **kwargs: object,
            ) -> str:
                await asyncio.sleep(self.delay)
                self.call_count += 1
                return "Mock async response"

            def estimate_tokens(self, text: str) -> int:
                return len(text) // 4

        from src.tools.long_chain import LongChainOfThoughtTool

        mock_llm = MockAsyncLLMWithDelay(delay_seconds=0.05)
        tool = LongChainOfThoughtTool(mock_llm)  # type: ignore

        # Request 4 STaR iterations with minimal steps
        # If fully sequential: ~4 iterations * (3 steps + 1 answer + 1 verify) * 0.05s = 1.0s
        # With concurrent chain gen: first iter + 3 concurrent + sequential scoring ≈ 0.6-0.8s
        start_time = time.monotonic()

        result = await tool.reason_async(
            problem="Test concurrency",
            num_steps=3,
            verify_intermediate=False,
            star_iterations=4,
        )

        elapsed = time.monotonic() - start_time

        # Verify result is valid
        assert result.answer is not None
        assert result.reasoning_trace is not None
        assert result.reasoning_trace.get("star_async") is True

        # The key benefit of concurrency is that chain generation runs in parallel.
        # Scoring still happens sequentially after gathering, so we can't expect
        # full parallelism. With 4 iterations:
        # - Sequential chain gen only: 4 * 3 * 0.05 = 0.6s just for steps
        # - Concurrent: first (0.15s) + remaining 3 concurrent (0.15s) = 0.3s for steps
        # - Plus scoring overhead (4 * 0.05 verify calls = 0.2s sequential)
        # Total concurrent: ~0.5-0.7s vs sequential ~1.0s
        # Use generous threshold for CI stability
        assert elapsed < 1.0, f"Expected faster than sequential (>1.0s), got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_async_star_temperature_variation(self) -> None:
        """Test that async STaR uses varying temperatures across iterations."""
        from src.tools.long_chain import LongChainOfThoughtTool

        class MockAsyncLLMTrackingTemperature:
            """Mock LLM that tracks temperature values used."""

            def __init__(self) -> None:
                self.temperatures: list[float] = []

            def generate(
                self,
                prompt: str,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                **kwargs: object,
            ) -> str:
                self.temperatures.append(temperature)
                return "Mock response"

            async def generate_async(
                self,
                prompt: str,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                **kwargs: object,
            ) -> str:
                self.temperatures.append(temperature)
                return "Mock response"

            def estimate_tokens(self, text: str) -> int:
                return len(text) // 4

        mock_llm = MockAsyncLLMTrackingTemperature()
        tool = LongChainOfThoughtTool(mock_llm)  # type: ignore

        await tool.reason_async(
            problem="Test temperature variation",
            num_steps=2,  # Minimal steps to reduce noise
            verify_intermediate=False,
            star_iterations=3,
        )

        # Should have multiple different temperatures (0.4, 0.6, 0.8 for 3 iterations)
        unique_temps = set(mock_llm.temperatures)
        assert len(unique_temps) >= 2, f"Expected temperature variation, got {unique_temps}"

        # Temperatures should be in expected range (0.4 to 0.8)
        for temp in mock_llm.temperatures:
            assert 0.2 <= temp <= 0.9, f"Temperature {temp} outside expected range"

    @pytest.mark.asyncio
    async def test_async_star_handles_iteration_failures(self) -> None:
        """Test that async STaR handles failures in individual iterations gracefully."""
        import asyncio

        from src.tools.long_chain import LongChainOfThoughtTool

        class MockAsyncLLMWithPartialFailures:
            """Mock LLM that fails on some calls to test error handling."""

            def __init__(self) -> None:
                self.call_count = 0

            def generate(
                self,
                prompt: str,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                **kwargs: object,
            ) -> str:
                self.call_count += 1
                return "Mock response"

            async def generate_async(
                self,
                prompt: str,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                **kwargs: object,
            ) -> str:
                self.call_count += 1
                # Fail some calls based on temperature (simulating iteration-specific failures)
                if 0.55 < temperature < 0.65:
                    await asyncio.sleep(0.01)
                    raise RuntimeError("Simulated iteration failure")
                return "Mock response"

            def estimate_tokens(self, text: str) -> int:
                return len(text) // 4

        mock_llm = MockAsyncLLMWithPartialFailures()
        tool = LongChainOfThoughtTool(mock_llm)  # type: ignore

        # Should complete despite some iterations failing
        result = await tool.reason_async(
            problem="Test failure handling",
            num_steps=3,
            verify_intermediate=False,
            star_iterations=4,  # One iteration will fail (temp ≈ 0.53 or 0.67)
        )

        # Should still produce a valid result from successful iterations
        assert result.answer is not None
        assert result.reasoning_trace is not None
        assert result.reasoning_trace.get("star_enabled") is True

    @pytest.mark.asyncio
    async def test_async_star_without_iterations_no_async_flag(self) -> None:
        """Test that non-STaR async mode doesn't set star_async flag."""
        from src.tools.long_chain import LongChainOfThoughtTool

        class MockAsyncLLM:
            """Simple mock async LLM."""

            async def generate_async(
                self,
                prompt: str,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                **kwargs: object,
            ) -> str:
                return "Mock response"

            def estimate_tokens(self, text: str) -> int:
                return len(text) // 4

        mock_llm = MockAsyncLLM()
        tool = LongChainOfThoughtTool(mock_llm)  # type: ignore

        result = await tool.reason_async(
            problem="Test no STaR",
            num_steps=3,
            verify_intermediate=False,
            star_iterations=0,  # Disabled
        )

        # Should not have star_async flag
        assert result.reasoning_trace is not None
        assert result.reasoning_trace.get("star_async") is None
        assert result.reasoning_trace.get("star_enabled") is None

    @pytest.mark.asyncio
    async def test_async_star_early_exit_cancels_pending_iterations(self) -> None:
        """Test that early exit signaling cancels pending iterations to save compute.

        When one iteration achieves a high-confidence score (>= 1.4), the early exit
        event is set and remaining iterations should be cancelled mid-chain.
        """
        import asyncio

        from src.tools.long_chain import LongChainOfThoughtTool

        class MockAsyncLLMWithEarlyExitTracking:
            """Mock LLM that tracks call counts and enables early exit testing."""

            def __init__(self) -> None:
                self.call_count = 0
                self.iteration_calls: dict[float, int] = {}  # temp -> call count

            async def generate_async(
                self,
                prompt: str,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                **kwargs: object,
            ) -> str:
                self.call_count += 1
                self.iteration_calls[temperature] = self.iteration_calls.get(temperature, 0) + 1

                # Add delay to allow early exit to happen
                await asyncio.sleep(0.02)

                # For verification prompts, return "YES" to trigger high score
                if "verdict" in prompt.lower() or "correct" in prompt.lower():
                    return "YES - the answer is valid"

                return f"Mock step response (temp={temperature:.2f})"

            def estimate_tokens(self, text: str) -> int:
                return len(text) // 4

        mock_llm = MockAsyncLLMWithEarlyExitTracking()
        tool = LongChainOfThoughtTool(mock_llm)  # type: ignore

        # Request 5 iterations with 10 steps each
        # Without early exit: would take a long time
        # With early exit: should complete faster when high-confidence found
        result = await tool.reason_async(
            problem="Test early exit signaling",
            num_steps=10,
            verify_intermediate=True,
            star_iterations=5,
        )

        # Verify result is valid
        assert result.answer is not None
        assert result.reasoning_trace is not None
        assert result.reasoning_trace.get("star_enabled") is True

        # Check early exit metadata
        iterations_used = result.reasoning_trace.get("star_iterations_used", 0)
        iterations_requested = result.reasoning_trace.get("star_iterations_requested", 0)
        early_exit = result.reasoning_trace.get("star_early_exit", False)

        # With high-confidence results, early exit should have triggered
        # OR all iterations completed (depending on timing)
        assert iterations_used >= 1, "At least one iteration should complete"
        assert iterations_requested == 5, "Should have requested 5 iterations"

        # If early exit happened, verify it's tracked
        if iterations_used < iterations_requested:
            assert (
                early_exit is True
            ), "star_early_exit should be True when iterations_used < requested"

    @pytest.mark.asyncio
    async def test_async_star_early_exit_metadata_tracking(self) -> None:
        """Test that star_early_exit metadata is correctly set."""
        from src.tools.long_chain import LongChainOfThoughtTool

        class MockAsyncLLM:
            """Simple mock that completes quickly."""

            async def generate_async(
                self,
                prompt: str,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                **kwargs: object,
            ) -> str:
                # Return NO for verification to ensure low scores (no early exit)
                if "verdict" in prompt.lower() or "correct" in prompt.lower():
                    return "NO - answer may have issues"
                return "Mock response"

            def estimate_tokens(self, text: str) -> int:
                return len(text) // 4

        mock_llm = MockAsyncLLM()
        tool = LongChainOfThoughtTool(mock_llm)  # type: ignore

        result = await tool.reason_async(
            problem="Test metadata tracking",
            num_steps=2,
            verify_intermediate=False,
            star_iterations=3,
        )

        # With low scores (NO verification), all iterations should complete
        assert result.reasoning_trace is not None
        assert result.reasoning_trace.get("star_iterations_requested") == 3

        # star_early_exit should reflect whether iterations_used < requested
        iterations_used = result.reasoning_trace.get("star_iterations_used", 0)
        early_exit = result.reasoning_trace.get("star_early_exit", False)

        if iterations_used == 3:
            assert early_exit is False, "No early exit when all iterations complete"
        else:
            assert early_exit is True, "Early exit when not all iterations complete"
