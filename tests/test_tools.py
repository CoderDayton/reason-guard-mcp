"""Unit tests for src/tools/verify.py, mot_reasoning.py, long_chain.py, compress.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.utils.errors import ReasoningException, VerificationException


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize mock client with optional responses."""
        self.responses = responses or ["Mock response"]
        self.call_index = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a mock response."""
        response = self.responses[self.call_index % len(self.responses)]
        self.call_index += 1
        return response

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4


# =============================================================================
# FactVerificationTool Tests
# =============================================================================


class TestFactVerificationTool:
    """Tests for FactVerificationTool."""

    def test_verify_empty_answer_raises(self) -> None:
        """Test empty answer raises VerificationException."""
        from src.tools.verify import FactVerificationTool

        tool = FactVerificationTool(MockLLMClient())  # type: ignore

        with pytest.raises(VerificationException, match="Answer cannot be empty"):
            tool.verify(answer="", context="Some context")

    def test_verify_empty_context_raises(self) -> None:
        """Test empty context raises VerificationException."""
        from src.tools.verify import FactVerificationTool

        tool = FactVerificationTool(MockLLMClient())  # type: ignore

        with pytest.raises(VerificationException, match="Context cannot be empty"):
            tool.verify(answer="Some answer", context="")

    def test_verify_invalid_max_claims_raises(self) -> None:
        """Test invalid max_claims raises VerificationException."""
        from src.tools.verify import FactVerificationTool

        tool = FactVerificationTool(MockLLMClient())  # type: ignore

        with pytest.raises(VerificationException, match="max_claims must be 1-20"):
            tool.verify(answer="Answer", context="Context", max_claims=0)

        with pytest.raises(VerificationException, match="max_claims must be 1-20"):
            tool.verify(answer="Answer", context="Context", max_claims=21)

    def test_verify_no_claims_found(self) -> None:
        """Test verification when no claims are extracted."""
        from src.tools.verify import FactVerificationTool

        # LLM returns empty response for claim extraction
        tool = FactVerificationTool(MockLLMClient(["", ""]))  # type: ignore

        result = tool.verify(answer="OK", context="Context here")

        assert result.verified is True
        assert result.confidence == 0.5
        assert "No verifiable claims" in result.reason

    def test_verify_claims_supported(self) -> None:
        """Test verification when claims are supported."""
        from src.tools.verify import FactVerificationTool

        responses = [
            "Einstein was a physicist\nHe developed relativity",  # Claims
            "SUPPORTED - The context confirms this",  # Verify claim 1
            "SUPPORTED - Yes this is correct",  # Verify claim 2
        ]
        tool = FactVerificationTool(MockLLMClient(responses))  # type: ignore

        result = tool.verify(
            answer="Einstein was a physicist who developed relativity",
            context="Albert Einstein was a theoretical physicist known for relativity",
        )

        assert result.verified is True
        assert result.confidence >= 0.7

    def test_verify_claims_contradicted(self) -> None:
        """Test verification when claims are contradicted."""
        from src.tools.verify import FactVerificationTool

        responses = [
            "Einstein was born in 1800",  # Claim
            "CONTRADICTED - Einstein was born in 1879",  # Verify
        ]
        tool = FactVerificationTool(MockLLMClient(responses))  # type: ignore

        result = tool.verify(
            answer="Einstein was born in 1800",
            context="Einstein was born in 1879",
        )

        assert result.verified is False
        assert len(result.claim_details) >= 1

    def test_verify_claims_unclear(self) -> None:
        """Test verification when claims are unclear."""
        from src.tools.verify import FactVerificationTool

        responses = [
            "Some random fact",
            "UNCLEAR - Context doesn't mention this",
        ]
        tool = FactVerificationTool(MockLLMClient(responses))  # type: ignore

        result = tool.verify(
            answer="Some random fact",
            context="Unrelated context about weather",
        )

        # Unclear is treated as not supported (0 verified / 1 total = 0.0)
        # The individual claim has confidence=0.5, but result confidence is verified_count/total
        assert result.confidence == 0.0
        assert result.claims_verified == 0
        assert result.claims_total == 1
        # Check individual claim has the 0.5 confidence
        assert result.claim_details[0]["confidence"] == 0.5

    def test_verify_llm_failure_in_extraction_uses_fallback(self) -> None:
        """Test fallback to sentence splitting when LLM fails."""
        from src.tools.verify import FactVerificationTool

        # First call raises exception, subsequent calls work
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            Exception("LLM Error"),  # Extraction fails
            "SUPPORTED",  # Verification works
        ]
        mock_llm.estimate_tokens.return_value = 100

        tool = FactVerificationTool(mock_llm)

        result = tool.verify(
            answer="This is a sentence with more than five words. Another good sentence here.",
            context="Some context",
        )

        # Fallback should have extracted sentences
        assert result is not None

    def test_verify_llm_failure_in_verification(self) -> None:
        """Test handling of LLM failure during claim verification."""
        from src.tools.verify import FactVerificationTool

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            "Claim one\nClaim two",  # Extraction works
            Exception("LLM Error"),  # Verification fails
        ]
        mock_llm.estimate_tokens.return_value = 100

        tool = FactVerificationTool(mock_llm)

        result = tool.verify(answer="Test answer here", context="Context")

        # Should handle error gracefully
        assert result is not None


# =============================================================================
# MatrixOfThoughtTool Tests
# =============================================================================


class TestMatrixOfThoughtTool:
    """Tests for MatrixOfThoughtTool."""

    def test_reason_empty_question_raises(self) -> None:
        """Test empty question raises ReasoningException."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        tool = MatrixOfThoughtTool(MockLLMClient())  # type: ignore

        with pytest.raises(ReasoningException, match="Question cannot be empty"):
            tool.reason(question="", context="Some context")

    def test_reason_empty_context_raises(self) -> None:
        """Test empty context raises ReasoningException."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        tool = MatrixOfThoughtTool(MockLLMClient())  # type: ignore

        with pytest.raises(ReasoningException, match="Context cannot be empty"):
            tool.reason(question="Question", context="")

    def test_reason_invalid_matrix_rows_raises(self) -> None:
        """Test invalid matrix_rows raises ReasoningException."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        tool = MatrixOfThoughtTool(MockLLMClient())  # type: ignore

        with pytest.raises(ReasoningException, match="matrix_rows must be 2-5"):
            tool.reason(question="Q", context="C", matrix_rows=1)

        with pytest.raises(ReasoningException, match="matrix_rows must be 2-5"):
            tool.reason(question="Q", context="C", matrix_rows=6)

    def test_reason_invalid_matrix_cols_raises(self) -> None:
        """Test invalid matrix_cols raises ReasoningException."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        tool = MatrixOfThoughtTool(MockLLMClient())  # type: ignore

        with pytest.raises(ReasoningException, match="matrix_cols must be 2-5"):
            tool.reason(question="Q", context="C", matrix_cols=1)

        with pytest.raises(ReasoningException, match="matrix_cols must be 2-5"):
            tool.reason(question="Q", context="C", matrix_cols=6)

    def test_reason_success(self) -> None:
        """Test successful MoT reasoning."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        responses = [
            "Thought 1",
            "Thought 2",
            "Thought 3",
            "Thought 4",
            "Synthesis 1",
            "Thought 5",
            "Thought 6",
            "Thought 7",
            "Thought 8",
            "Synthesis 2",
            "Final answer: 42",
        ]
        tool = MatrixOfThoughtTool(MockLLMClient(responses))  # type: ignore

        result = tool.reason(
            question="What is the answer?",
            context="Context information here",
            matrix_rows=2,
            matrix_cols=2,
        )

        assert result.answer is not None
        assert result.confidence > 0

    def test_reason_with_uniform_pattern(self) -> None:
        """Test MoT with uniform communication pattern."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        tool = MatrixOfThoughtTool(MockLLMClient(["Response"] * 20))  # type: ignore

        result = tool.reason(
            question="Q",
            context="C",
            matrix_rows=2,
            matrix_cols=2,
            communication_pattern="uniform",
        )

        assert result.reasoning_trace["communication_pattern"] == "uniform"

    def test_reason_with_none_pattern(self) -> None:
        """Test MoT with no communication pattern."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        tool = MatrixOfThoughtTool(MockLLMClient(["Response"] * 20))  # type: ignore

        result = tool.reason(
            question="Q",
            context="C",
            matrix_rows=2,
            matrix_cols=2,
            communication_pattern="none",
        )

        assert result.reasoning_trace["communication_pattern"] == "none"

    def test_reason_with_unknown_pattern(self) -> None:
        """Test MoT with unknown pattern uses default."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        tool = MatrixOfThoughtTool(MockLLMClient(["Response"] * 20))  # type: ignore

        result = tool.reason(
            question="Q",
            context="C",
            matrix_rows=2,
            matrix_cols=2,
            communication_pattern="unknown_pattern",
        )

        assert result is not None

    def test_reason_thought_generation_failure(self) -> None:
        """Test handling of thought generation failure."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        mock_llm = MagicMock()
        mock_llm.generate.return_value = ""  # Empty response
        mock_llm.estimate_tokens.return_value = 0

        tool = MatrixOfThoughtTool(mock_llm)

        result = tool.reason(question="Q", context="C", matrix_rows=2, matrix_cols=2)

        assert result.answer == "Unable to generate answer"

    def test_reason_synthesis_failure(self) -> None:
        """Test handling of synthesis failure."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            "Thought 1",
            "Thought 2",
            Exception("Synthesis failed"),  # Synthesis
            "Thought 3",
            "Thought 4",
            "Final synthesis",
            "Final answer",
        ]
        mock_llm.estimate_tokens.return_value = 100

        tool = MatrixOfThoughtTool(mock_llm)

        result = tool.reason(question="Q", context="C", matrix_rows=2, matrix_cols=2)

        assert result is not None

    def test_reason_answer_extraction_failure(self) -> None:
        """Test fallback when answer extraction fails."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        mock_llm = MagicMock()
        # All thoughts and synthesis work, final extraction fails
        responses = ["Thought"] * 10 + ["Synthesis is here"]
        mock_llm.generate.side_effect = responses + [Exception("Extract failed")]
        mock_llm.estimate_tokens.return_value = 100

        tool = MatrixOfThoughtTool(mock_llm)

        result = tool.reason(question="Q", context="C", matrix_rows=2, matrix_cols=2)

        # Should fallback to summary
        assert result is not None


# =============================================================================
# LongChainOfThoughtTool Tests
# =============================================================================


class TestLongChainOfThoughtTool:
    """Tests for LongChainOfThoughtTool."""

    def test_reason_empty_problem_raises(self) -> None:
        """Test empty problem raises ReasoningException."""
        from src.tools.long_chain import LongChainOfThoughtTool

        tool = LongChainOfThoughtTool(MockLLMClient())  # type: ignore

        with pytest.raises(ReasoningException, match="Problem cannot be empty"):
            tool.reason(problem="")

    def test_reason_invalid_num_steps_raises(self) -> None:
        """Test invalid num_steps raises ReasoningException."""
        from src.tools.long_chain import LongChainOfThoughtTool

        tool = LongChainOfThoughtTool(MockLLMClient())  # type: ignore

        with pytest.raises(ReasoningException, match="num_steps must be 1-50"):
            tool.reason(problem="P", num_steps=0)

        with pytest.raises(ReasoningException, match="num_steps must be 1-50"):
            tool.reason(problem="P", num_steps=51)

    def test_reason_success_with_verification(self) -> None:
        """Test successful reasoning with intermediate verification."""
        from src.tools.long_chain import LongChainOfThoughtTool

        responses = [
            "Step 1 reasoning",
            "Step 2 reasoning",
            "Step 3 reasoning",
            "YES - Step 3 is valid",  # Verification
            "Step 4 reasoning",
            "Step 5 reasoning",
            "Step 6 reasoning",
            "YES - Step 6 is valid",  # Verification
            "Final answer is 42",  # Answer extraction
        ]
        tool = LongChainOfThoughtTool(MockLLMClient(responses))  # type: ignore

        result = tool.reason(
            problem="Solve this problem",
            num_steps=6,
            verify_intermediate=True,
            verification_frequency=3,
        )

        assert result.answer is not None
        assert result.verification_results["total_verifications"] == 2
        assert result.verification_results["passed"] == 2

    def test_reason_without_verification(self) -> None:
        """Test reasoning without intermediate verification."""
        from src.tools.long_chain import LongChainOfThoughtTool

        tool = LongChainOfThoughtTool(MockLLMClient(["Response"] * 10))  # type: ignore

        result = tool.reason(
            problem="Problem",
            num_steps=5,
            verify_intermediate=False,
        )

        assert result.verification_results["total_verifications"] == 0
        assert result.confidence == 0.7  # Default without verification

    def test_reason_verification_failure(self) -> None:
        """Test handling of verification failure."""
        from src.tools.long_chain import LongChainOfThoughtTool

        responses = [
            "Step 1",
            "Step 2",
            "Step 3",
            "NO - This step is invalid",  # Verification fails
            "Step 4",
            "Step 5",
            "Step 6",
            "YES - Valid",  # Verification passes
            "Final answer",
        ]
        tool = LongChainOfThoughtTool(MockLLMClient(responses))  # type: ignore

        result = tool.reason(
            problem="P",
            num_steps=6,
            verify_intermediate=True,
            verification_frequency=3,
        )

        assert result.verification_results["passed"] == 1
        assert result.verification_results["failed"] == 1

    def test_reason_step_generation_failure(self) -> None:
        """Test handling when step generation fails."""
        from src.tools.long_chain import LongChainOfThoughtTool

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            "Step 1",
            "Step 2",
            "",  # Empty response stops chain
        ]
        mock_llm.estimate_tokens.return_value = 100

        tool = LongChainOfThoughtTool(mock_llm)

        result = tool.reason(problem="P", num_steps=5, verify_intermediate=False)

        # Chain should stop early but still produce result
        assert len(result.reasoning_steps) == 2

    def test_reason_answer_extraction_failure(self) -> None:
        """Test fallback when answer extraction fails."""
        from src.tools.long_chain import LongChainOfThoughtTool

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            "Step 1",
            "Step 2",
            "Step 3",
            Exception("Extraction failed"),  # Answer extraction fails
        ]
        mock_llm.estimate_tokens.return_value = 100

        tool = LongChainOfThoughtTool(mock_llm)

        result = tool.reason(problem="P", num_steps=3, verify_intermediate=False)

        # Should fallback to last step
        assert "Step 3" in result.answer

    def test_reason_empty_chain(self) -> None:
        """Test handling of empty reasoning chain."""
        from src.tools.long_chain import LongChainOfThoughtTool

        mock_llm = MagicMock()
        mock_llm.generate.return_value = ""  # All steps fail
        mock_llm.estimate_tokens.return_value = 0

        tool = LongChainOfThoughtTool(mock_llm)

        result = tool.reason(problem="P", num_steps=3, verify_intermediate=False)

        assert result.answer == "Unable to generate answer"

    def test_build_recent_context_empty(self) -> None:
        """Test _build_recent_context with empty chain."""
        from src.tools.long_chain import LongChainOfThoughtTool

        tool = LongChainOfThoughtTool(MockLLMClient())  # type: ignore

        result = tool._build_recent_context([], max_steps=5)
        assert result == ""

    def test_build_recent_context_truncates_long_steps(self) -> None:
        """Test _build_recent_context truncates long steps."""
        from src.tools.long_chain import LongChainOfThoughtTool

        tool = LongChainOfThoughtTool(MockLLMClient())  # type: ignore

        long_step = "A" * 200  # Longer than 150
        result = tool._build_recent_context([long_step], max_steps=5)

        assert "..." in result


# =============================================================================
# Integration Tests for Error Propagation
# =============================================================================


class TestToolErrorPropagation:
    """Test error handling behaviors in tools."""

    def test_verify_uses_fallback_on_extraction_error(self) -> None:
        """Test that verify uses fallback sentence splitting when extraction fails."""
        from src.tools.verify import FactVerificationTool

        mock_llm = MagicMock()
        # First call (extraction) fails, second (verification) succeeds
        mock_llm.generate.side_effect = [
            Exception("Extraction error"),
            "SUPPORTED - this is verified",
        ]

        tool = FactVerificationTool(mock_llm)

        # Should not raise - uses fallback
        result = tool.verify(
            answer="This sentence has more than five words for testing purposes.",
            context="Test context",
        )
        # Fallback splits by sentence and uses sentences with >= 5 words
        assert result is not None

    def test_mot_generates_empty_matrix_on_failure(self) -> None:
        """Test MoT handles generation failures gracefully."""
        from src.tools.mot_reasoning import MatrixOfThoughtTool

        mock_llm = MagicMock()
        # All generations fail, then synthesis still works
        mock_llm.generate.side_effect = [
            Exception("Gen error"),  # (0,0)
            Exception("Gen error"),  # (1,0)
            Exception("Gen error"),  # (0,1)
            Exception("Gen error"),  # (1,1)
            "Final synthesis answer",  # synthesis
            "42",  # answer extraction
        ]
        mock_llm.estimate_tokens.return_value = 0

        tool = MatrixOfThoughtTool(mock_llm)

        # Should not raise - handles gracefully with empty thoughts
        result = tool.reason(question="Q", context="C", matrix_rows=2, matrix_cols=2)
        assert result is not None

    def test_long_chain_stops_on_failure(self) -> None:
        """Test long chain stops when step generation fails."""
        from src.tools.long_chain import LongChainOfThoughtTool

        mock_llm = MagicMock()
        # First step fails
        mock_llm.generate.side_effect = [
            RuntimeError("Step error"),  # step 1 fails
            "Final answer",  # answer extraction still runs
        ]
        mock_llm.estimate_tokens.return_value = 0

        tool = LongChainOfThoughtTool(mock_llm)

        # Should not raise - stops chain early
        result = tool.reason(problem="P", num_steps=3, verify_intermediate=False)
        # Result will have empty chain but still returns
        assert result is not None
        assert len(result.reasoning_steps) == 0  # No successful steps
