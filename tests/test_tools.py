"""Unit tests for state manager tools: verify.py, mot_reasoning.py, long_chain.py.

These tests verify the state management logic of each tool.
The tools no longer contain LLM logic - they track state for the calling LLM.
"""

from __future__ import annotations

import pytest

from src.tools.long_chain import LongChainManager
from src.tools.mot_reasoning import MatrixOfThoughtManager
from src.tools.verify import VerificationManager

# =============================================================================
# LongChainManager Tests
# =============================================================================


class TestLongChainManager:
    """Tests for LongChainManager state management."""

    def test_start_chain_creates_session(self) -> None:
        """Test start_chain() creates a new session with correct state."""
        manager = LongChainManager()
        result = manager.start_chain(problem="Test problem", expected_steps=10)

        assert "session_id" in result
        assert result["status"] == "started"
        assert result["problem"] == "Test problem"
        assert result["expected_steps"] == 10
        assert result["next_step"] == 1

    def test_start_chain_with_metadata(self) -> None:
        """Test start_chain() with optional metadata."""
        manager = LongChainManager()
        result = manager.start_chain(
            problem="Problem",
            expected_steps=5,
            metadata={"source": "test"},
        )

        assert result["status"] == "started"
        # Metadata is stored in state, verify via get_chain
        state = manager.get_chain(result["session_id"])
        assert state["session_id"] == result["session_id"]

    def test_add_step_increments_state(self) -> None:
        """Test add_step() adds thought and increments counter."""
        manager = LongChainManager()
        start = manager.start_chain(problem="P", expected_steps=5)
        session_id = start["session_id"]

        result = manager.add_step(session_id, thought="First thought")

        assert result["step_added"] == 1
        assert result["total_steps"] == 1

    def test_add_step_multiple(self) -> None:
        """Test adding multiple steps."""
        manager = LongChainManager()
        start = manager.start_chain(problem="P", expected_steps=5)
        session_id = start["session_id"]

        manager.add_step(session_id, thought="Step 1")
        manager.add_step(session_id, thought="Step 2")
        result = manager.add_step(session_id, thought="Step 3")

        assert result["step_added"] == 3
        assert result["total_steps"] == 3

    def test_add_step_invalid_session(self) -> None:
        """Test add_step() with invalid session ID."""
        manager = LongChainManager()

        with pytest.raises(ValueError, match="Session .* not found"):
            manager.add_step("invalid-id", thought="Test")

    def test_add_step_progress_tracking(self) -> None:
        """Test add_step() tracks progress correctly."""
        manager = LongChainManager()
        start = manager.start_chain(problem="P", expected_steps=5)
        session_id = start["session_id"]

        result = manager.add_step(session_id, thought="Step 1")
        assert result["progress"] == 0.2  # 1/5

        manager.add_step(session_id, thought="Step 2")
        manager.add_step(session_id, thought="Step 3")
        result = manager.add_step(session_id, thought="Step 4")
        assert result["progress"] == 0.8  # 4/5

    def test_add_step_needs_more_flag(self) -> None:
        """Test add_step() sets needs_more_steps correctly."""
        manager = LongChainManager()
        start = manager.start_chain(problem="P", expected_steps=2)
        session_id = start["session_id"]

        result1 = manager.add_step(session_id, thought="Step 1")
        assert result1["needs_more_steps"] is True

        result2 = manager.add_step(session_id, thought="Step 2")
        assert result2["needs_more_steps"] is False

    def test_finalize_completes_session(self) -> None:
        """Test finalize() completes session with answer."""
        manager = LongChainManager()
        start = manager.start_chain(problem="P", expected_steps=5)
        session_id = start["session_id"]

        manager.add_step(session_id, thought="Step 1")
        result = manager.finalize(session_id, answer="42", confidence=0.9)

        assert result["status"] == "completed"
        assert result["final_answer"] == "42"
        assert result["confidence"] == 0.9

    def test_finalize_invalid_session(self) -> None:
        """Test finalize() with invalid session."""
        manager = LongChainManager()

        with pytest.raises(ValueError, match="Session .* not found"):
            manager.finalize("invalid-id", answer="X", confidence=0.5)

    def test_get_chain_returns_current_state(self) -> None:
        """Test get_chain() returns current session state."""
        manager = LongChainManager()
        start = manager.start_chain(problem="Test", expected_steps=5)
        session_id = start["session_id"]

        manager.add_step(session_id, thought="Step 1")
        state = manager.get_chain(session_id)

        assert state["current_step"] == 1
        assert state["status"] == "active"

    def test_multiple_sessions_independent(self) -> None:
        """Test multiple sessions are independent."""
        manager = LongChainManager()

        s1 = manager.start_chain(problem="Problem 1", expected_steps=5)
        s2 = manager.start_chain(problem="Problem 2", expected_steps=10)

        manager.add_step(s1["session_id"], thought="S1 Step")

        state1 = manager.get_chain(s1["session_id"])
        state2 = manager.get_chain(s2["session_id"])

        assert state1["current_step"] == 1
        assert state2["current_step"] == 0
        assert state1["problem"] == "Problem 1"
        assert state2["problem"] == "Problem 2"

    def test_abandon_marks_session(self) -> None:
        """Test abandon() marks session as abandoned."""
        manager = LongChainManager()
        start = manager.start_chain(problem="P", expected_steps=5)
        session_id = start["session_id"]

        result = manager.abandon(session_id)
        assert result["status"] == "abandoned"

        # Session still exists but is abandoned
        state = manager.get_chain(session_id)
        assert state["status"] == "abandoned"

    def test_list_sessions(self) -> None:
        """Test list_sessions() returns all sessions."""
        manager = LongChainManager()

        manager.start_chain(problem="Problem 1", expected_steps=5)
        manager.start_chain(problem="Problem 2", expected_steps=5)

        sessions = manager.list_sessions()
        assert sessions["total"] == 2
        assert len(sessions["sessions"]) == 2

    def test_add_step_with_branch(self) -> None:
        """Test add_step() with branching."""
        manager = LongChainManager()
        start = manager.start_chain(problem="P", expected_steps=5)
        session_id = start["session_id"]

        manager.add_step(session_id, thought="Step 1")
        manager.add_step(session_id, thought="Step 2")

        # Branch from step 1
        result = manager.add_step(
            session_id,
            thought="Alternative approach",
            branch_from=1,
        )

        assert result["step_type"] == "branch"
        assert "branch_1" in result["branches"]

    def test_add_step_with_revision(self) -> None:
        """Test add_step() with revision."""
        manager = LongChainManager()
        start = manager.start_chain(problem="P", expected_steps=5)
        session_id = start["session_id"]

        manager.add_step(session_id, thought="Step 1")
        manager.add_step(session_id, thought="Step 2 - wrong")

        # Revise step 2
        result = manager.add_step(
            session_id,
            thought="Step 2 - corrected",
            revises=2,
        )

        assert result["step_type"] == "revision"


# =============================================================================
# MatrixOfThoughtManager Tests
# =============================================================================


class TestMatrixOfThoughtManager:
    """Tests for MatrixOfThoughtManager state management."""

    def test_start_matrix_creates_matrix(self) -> None:
        """Test start_matrix() creates matrix with correct dimensions."""
        manager = MatrixOfThoughtManager()
        result = manager.start_matrix(
            question="What is X?",
            rows=3,
            cols=2,
        )

        assert "session_id" in result
        assert result["status"] == "started"
        assert result["question"] == "What is X?"
        assert result["matrix_dimensions"]["rows"] == 3
        assert result["matrix_dimensions"]["cols"] == 2

    def test_start_matrix_with_context(self) -> None:
        """Test start_matrix() with optional context."""
        manager = MatrixOfThoughtManager()
        result = manager.start_matrix(
            question="Q",
            rows=2,
            cols=2,
            context="Some context",
        )

        assert result["status"] == "started"

    def test_start_matrix_with_strategies(self) -> None:
        """Test start_matrix() with custom strategies."""
        manager = MatrixOfThoughtManager()
        result = manager.start_matrix(
            question="Q",
            rows=2,
            cols=2,
            strategies=["analytical", "creative"],
        )

        # Strategies stored in session, verify via get_matrix
        state = manager.get_matrix(result["session_id"])
        assert state["strategies"] == ["analytical", "creative"]

    def test_set_cell_updates_matrix(self) -> None:
        """Test set_cell() updates correct matrix position."""
        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Q", rows=2, cols=2)
        session_id = start["session_id"]

        result = manager.set_cell(session_id, row=0, col=0, thought="Analysis for 0,0")

        assert "progress" in result  # Has progress tracking
        state = manager.get_matrix(session_id)
        assert state["cells_filled"] == 1

    def test_set_cell_validates_bounds(self) -> None:
        """Test set_cell() validates row/col bounds."""
        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Q", rows=2, cols=2)
        session_id = start["session_id"]

        # Out of bounds should return error dict
        result = manager.set_cell(session_id, row=5, col=0, thought="X")
        assert "error" in result

        result = manager.set_cell(session_id, row=0, col=5, thought="X")
        assert "error" in result

    def test_set_cell_invalid_session(self) -> None:
        """Test set_cell() with invalid session."""
        manager = MatrixOfThoughtManager()

        with pytest.raises(ValueError, match="Session .* not found"):
            manager.set_cell("invalid", row=0, col=0, thought="X")

    def test_synthesize_column_stores_synthesis(self) -> None:
        """Test synthesize_column() stores synthesis for a column."""
        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Q", rows=2, cols=2)
        session_id = start["session_id"]

        # Fill entire column 0 first (required for synthesis)
        manager.set_cell(session_id, row=0, col=0, thought="A")
        manager.set_cell(session_id, row=1, col=0, thought="B")

        result = manager.synthesize_column(session_id, col=0, synthesis="Combined: A and B")

        # Should succeed without error
        assert "error" not in result
        assert result["column_synthesized"] == 0

    def test_synthesize_column_invalid_session(self) -> None:
        """Test synthesize_column() with invalid session."""
        manager = MatrixOfThoughtManager()

        with pytest.raises(ValueError, match="Session .* not found"):
            manager.synthesize_column("invalid", col=0, synthesis="X")

    def test_finalize_completes_session(self) -> None:
        """Test finalize() completes session."""
        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Q", rows=2, cols=2)
        session_id = start["session_id"]

        # Fill matrix
        manager.set_cell(session_id, row=0, col=0, thought="A")
        manager.set_cell(session_id, row=0, col=1, thought="B")
        manager.set_cell(session_id, row=1, col=0, thought="C")
        manager.set_cell(session_id, row=1, col=1, thought="D")

        result = manager.finalize(session_id, answer="Answer", confidence=0.85)

        assert result["status"] == "completed"
        assert result["final_answer"] == "Answer"
        assert result["confidence"] == 0.85

    def test_matrix_complete_detection(self) -> None:
        """Test cells_filled tracks progress correctly."""
        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Q", rows=2, cols=2)
        session_id = start["session_id"]

        # Partially filled
        result = manager.set_cell(session_id, row=0, col=0, thought="A")
        assert result["cells_filled"] == 1
        assert result["progress"] == 0.25

        # Fill remaining
        manager.set_cell(session_id, row=0, col=1, thought="B")
        manager.set_cell(session_id, row=1, col=0, thought="C")
        result = manager.set_cell(session_id, row=1, col=1, thought="D")

        assert result["cells_filled"] == 4
        assert result["progress"] == 1.0

    def test_abandon_marks_session(self) -> None:
        """Test abandon() marks session as abandoned."""
        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Q", rows=2, cols=2)
        session_id = start["session_id"]

        result = manager.abandon(session_id)
        assert result["status"] == "abandoned"


# =============================================================================
# VerificationManager Tests
# =============================================================================


class TestVerificationManager:
    """Tests for VerificationManager state management."""

    def test_start_verification_creates_session(self) -> None:
        """Test start_verification() creates verification session."""
        manager = VerificationManager()
        result = manager.start_verification(
            answer="Test answer",
            context="Some context to verify against",
        )

        assert "session_id" in result
        assert result["status"] == "started"
        assert "suggested_claims" in result

    def test_add_claim_stores_claim(self) -> None:
        """Test add_claim() stores claim for verification."""
        manager = VerificationManager()
        start = manager.start_verification(answer="A", context="Test context")
        session_id = start["session_id"]

        result = manager.add_claim(session_id, content="The sky is blue")

        assert result["claim_id"] == 0
        assert result["total_claims"] == 1

    def test_add_claim_multiple(self) -> None:
        """Test adding multiple claims."""
        manager = VerificationManager()
        start = manager.start_verification(answer="A", context="Test context")
        session_id = start["session_id"]

        manager.add_claim(session_id, content="Claim 1")
        manager.add_claim(session_id, content="Claim 2")
        result = manager.add_claim(session_id, content="Claim 3")

        assert result["total_claims"] == 3

    def test_add_claim_invalid_session(self) -> None:
        """Test add_claim() with invalid session."""
        manager = VerificationManager()

        with pytest.raises(ValueError, match="Session .* not found"):
            manager.add_claim("invalid", content="X")

    def test_verify_claim_updates_status(self) -> None:
        """Test verify_claim() updates claim verification status."""
        manager = VerificationManager()
        start = manager.start_verification(answer="A", context="Test context")
        session_id = start["session_id"]

        manager.add_claim(session_id, content="Test claim")

        result = manager.verify_claim(
            session_id,
            claim_id=0,
            status="supported",
            evidence="Found in context",
            confidence=0.9,
        )

        assert result["claim_id"] == 0
        assert result["status"] == "supported"

    def test_verify_claim_validates_claim_id(self) -> None:
        """Test verify_claim() validates claim ID."""
        manager = VerificationManager()
        start = manager.start_verification(answer="A", context="Test context")
        session_id = start["session_id"]

        manager.add_claim(session_id, content="Only claim")

        result = manager.verify_claim(
            session_id,
            claim_id=999,
            status="supported",
            evidence="X",
            confidence=0.5,
        )
        assert "error" in result

    def test_finalize_computes_summary(self) -> None:
        """Test finalize() computes verification summary."""
        manager = VerificationManager()
        start = manager.start_verification(answer="A", context="Test context")
        session_id = start["session_id"]

        manager.add_claim(session_id, content="Claim 1")
        manager.add_claim(session_id, content="Claim 2")
        manager.add_claim(session_id, content="Claim 3")

        manager.verify_claim(session_id, 0, "supported", "E1", 0.9)
        manager.verify_claim(session_id, 1, "contradicted", "E2", 0.8)
        manager.verify_claim(session_id, 2, "supported", "E3", 0.95)

        result = manager.finalize(session_id)

        assert result["status"] == "completed"
        assert result["summary"]["total_claims"] == 3
        assert result["summary"]["supported"] == 2
        assert result["summary"]["contradicted"] == 1

    def test_finalize_overall_verdict_fully_supported(self) -> None:
        """Test verified=True when all claims supported."""
        manager = VerificationManager()
        start = manager.start_verification(answer="A", context="Test context")
        session_id = start["session_id"]

        manager.add_claim(session_id, content="Claim 1")
        manager.add_claim(session_id, content="Claim 2")

        manager.verify_claim(session_id, 0, "supported", "E1", 0.9)
        manager.verify_claim(session_id, 1, "supported", "E2", 0.85)

        result = manager.finalize(session_id)

        assert result["verified"] is True

    def test_finalize_overall_verdict_contradicted(self) -> None:
        """Test verified=False when any claim contradicted."""
        manager = VerificationManager()
        start = manager.start_verification(answer="A", context="Test context")
        session_id = start["session_id"]

        manager.add_claim(session_id, content="Claim 1")
        manager.add_claim(session_id, content="Claim 2")

        manager.verify_claim(session_id, 0, "supported", "E1", 0.9)
        manager.verify_claim(session_id, 1, "contradicted", "E2", 0.95)

        result = manager.finalize(session_id)

        assert result["verified"] is False

    def test_valid_statuses(self) -> None:
        """Test all valid status values."""
        manager = VerificationManager()
        start = manager.start_verification(answer="A", context="Test context")
        session_id = start["session_id"]

        # Valid statuses are: supported, contradicted
        valid_statuses = ["supported", "contradicted"]

        for i, status in enumerate(valid_statuses):
            manager.add_claim(session_id, content=f"Claim for {status}")
            result = manager.verify_claim(session_id, i, status, f"Evidence for {status}", 0.5)
            # verify_claim returns claim_id and status
            assert result["claim_id"] == i
            assert result["status"] == status

    def test_abandon_marks_session(self) -> None:
        """Test abandon() marks session as abandoned."""
        manager = VerificationManager()
        start = manager.start_verification(answer="A", context="Test")
        session_id = start["session_id"]

        result = manager.abandon(session_id)
        assert result["status"] == "abandoned"


# =============================================================================
# Compress Tool Tests (doesn't use sessions)
# =============================================================================


class TestCompress:
    """Tests for compression tool."""

    @pytest.fixture
    def compressor(self):
        """Create compressor with default model."""
        from src.tools.compress import ContextAwareCompressionTool

        return ContextAwareCompressionTool()

    def test_compress_reduces_tokens(self, compressor) -> None:
        """Test compression reduces token count."""
        long_context = "This is a test sentence. " * 50
        question = "What is this?"

        result = compressor.compress(
            context=long_context,
            question=question,
            compression_ratio=0.5,
        )

        assert result.original_tokens > result.compressed_tokens
        assert result.compression_ratio <= 0.7  # Allow some tolerance

    def test_compress_preserves_relevant_content(self, compressor) -> None:
        """Test compression preserves question-relevant content."""
        context = """
        The quick brown fox jumps over the lazy dog.
        Albert Einstein was born in 1879.
        The weather today is sunny with clear skies.
        Einstein developed the theory of relativity.
        Random filler text that is not relevant.
        """
        question = "When was Einstein born?"

        result = compressor.compress(
            context=context,
            question=question,
            compression_ratio=0.5,
        )

        # Should preserve "1879" as it's relevant to the question
        assert "1879" in result.compressed_context
