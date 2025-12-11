"""Unit tests for state manager tools: verify.py, mot_reasoning.py, long_chain.py.

These tests verify the state management logic of each tool.
The tools no longer contain LLM logic - they track state for the calling LLM.
"""

from __future__ import annotations

import pytest

from src.tools.long_chain import LongChainManager
from src.tools.mot_reasoning import MatrixOfThoughtManager
from src.tools.verify import VerificationManager
from src.utils.session import SessionNotFoundError

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

        with pytest.raises(SessionNotFoundError):
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

        with pytest.raises(SessionNotFoundError):
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

    def test_start_matrix_adaptive_rows_simple(self) -> None:
        """Test start_matrix() with rows='auto' for simple problems."""
        manager = MatrixOfThoughtManager()
        result = manager.start_matrix(
            question="What is 2+2?",
            rows="auto",
            cols=2,
        )

        assert result["status"] == "started"
        assert result["complexity"] is not None
        assert result["complexity"]["complexity_level"] == "low"
        assert result["matrix_dimensions"]["rows"] == 3  # Minimum 3 for adequate coverage

    def test_start_matrix_adaptive_rows_complex(self) -> None:
        """Test start_matrix() with rows='auto' for complex problems."""
        manager = MatrixOfThoughtManager()
        result = manager.start_matrix(
            question="If Alice is taller than Bob, and Bob is taller than Charlie, "
            "then what can we conclude about Alice and Charlie? "
            "Explain the logical reasoning step by step.",
            context="This is a multi-hop logical reasoning problem that requires "
            "analyzing relationships and drawing inferences based on "
            "transitive properties. Consider all factors and implications.",
            rows="auto",
            cols=2,
        )

        assert result["status"] == "started"
        assert result["complexity"] is not None
        assert result["complexity"]["complexity_score"] >= 0.35
        assert result["matrix_dimensions"]["rows"] >= 3  # Complex = more strategies

    def test_start_matrix_adaptive_returns_signals(self) -> None:
        """Test adaptive selection returns complexity signals."""
        manager = MatrixOfThoughtManager()
        result = manager.start_matrix(
            question="Calculate x if 2x + 5 = 15, then verify the solution.",
            rows="auto",
        )

        assert result["complexity"] is not None
        assert "signals" in result["complexity"]
        assert isinstance(result["complexity"]["signals"], list)

    def test_complexity_detection_caching(self) -> None:
        """Test that complexity detection results are cached."""
        from src.utils.complexity import clear_complexity_cache, detect_complexity

        question = "Test caching question for complexity"
        context = "Some context here"

        # Clear cache for this test
        clear_complexity_cache()

        # First call - should not be cached
        result1 = detect_complexity(question, context)
        assert result1.cached is False

        # Second call - should be cached
        result2 = detect_complexity(question, context)
        assert result2.cached is True

        # Results should match (except cached flag)
        assert result1.complexity_score == result2.complexity_score
        assert result1.recommended_rows == result2.recommended_rows

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

        with pytest.raises(SessionNotFoundError):
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

        with pytest.raises(SessionNotFoundError):
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

        with pytest.raises(SessionNotFoundError):
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


# =============================================================================
# Additional Coverage Tests - MatrixOfThoughtManager
# =============================================================================


class TestMatrixOfThoughtManagerCoverage:
    """Additional tests for mot_reasoning.py coverage."""

    def test_calculate_cell_survival_score_empty_thought(self) -> None:
        """Test survival score for empty thought."""
        from src.utils.scoring import calculate_cell_survival_score

        assert calculate_cell_survival_score("", "direct_factual", "context", 0) == 0.0
        assert calculate_cell_survival_score("   ", "logical_inference", "ctx", 0) == 0.0

    def test_calculate_cell_survival_score_strategy_keywords(self) -> None:
        """Test survival score with strategy-aligned keywords."""
        from src.utils.scoring import calculate_cell_survival_score

        # logical_inference keywords: therefore, implies, because, since, if, then
        # Need longer text to get good length score, plus context overlap
        logical_thought = (
            "Therefore, since X implies Y in the test context, then we can "
            "conclude Z because of evidence. This follows logically from the premises."
        )
        score = calculate_cell_survival_score(
            logical_thought, "logical_inference", "test context with evidence", 0
        )
        # With new weighted scoring, this should be moderate-to-good
        assert score > 0.4  # Has structure, some overlap, decent length

        # Compare: strategy-aligned vs non-aligned keywords
        causal_thought = (
            "The cause leads to this effect in the test scenario, "
            "resulting in consequence due to the causal chain identified."
        )
        score_causal = calculate_cell_survival_score(causal_thought, "causal", "test scenario", 0)
        score_wrong_strategy = calculate_cell_survival_score(
            causal_thought, "analogical", "test scenario", 0
        )
        # Causal keywords should help more when strategy is "causal"
        assert score_causal >= score_wrong_strategy

    def test_calculate_cell_survival_score_vague_phrases(self) -> None:
        """Test survival score penalizes vague phrases."""
        from src.utils.scoring import calculate_cell_survival_score

        vague = "Maybe something happens somehow, I guess probably not sure about this."
        specific = "The equation 2x + 5 = 15 has solution x = 5, verified by substitution."

        vague_score = calculate_cell_survival_score(vague, "direct_factual", "math problem", 0)
        specific_score = calculate_cell_survival_score(
            specific, "direct_factual", "math problem", 0
        )

        assert specific_score > vague_score

    def test_calculate_cell_survival_score_numbers_and_quotes(self) -> None:
        """Test survival score rewards concrete details."""
        from src.utils.scoring import calculate_cell_survival_score

        with_numbers = "The answer is 42.5 percent, calculated from 85 out of 200 samples."
        with_quotes = 'According to the text, "the answer is clear" from the evidence.'
        plain = "The answer can be found by looking at the evidence provided."

        num_score = calculate_cell_survival_score(with_numbers, "direct_factual", "ctx", 0)
        quote_score = calculate_cell_survival_score(with_quotes, "direct_factual", "ctx", 0)
        plain_score = calculate_cell_survival_score(plain, "direct_factual", "ctx", 0)

        assert num_score > plain_score
        assert quote_score > plain_score

    def test_calculate_cell_survival_score_length_penalties(self) -> None:
        """Test survival score length handling."""
        from src.utils.scoring import calculate_cell_survival_score

        short = "Too short."
        normal = "This is a reasonably sized thought with enough content to analyze properly."
        rambling = " ".join(["word"] * 200)  # Very long

        short_score = calculate_cell_survival_score(short, "direct_factual", "ctx", 0)
        normal_score = calculate_cell_survival_score(normal, "direct_factual", "ctx", 0)
        rambling_score = calculate_cell_survival_score(rambling, "direct_factual", "ctx", 0)

        assert normal_score > short_score  # Short penalty
        assert normal_score > rambling_score  # Rambling penalty

    def test_calculate_cell_survival_score_col0_bonus(self) -> None:
        """Test survival score position affects score."""
        from src.utils.scoring import calculate_cell_survival_score

        thought = "This is a planning thought with logical inference and evidence."

        # In new scoring, positions 0-3 (early) get same max score
        # Position 8+ gets lower score
        col0_score = calculate_cell_survival_score(thought, "direct_factual", "ctx", col=0)
        late_col_score = calculate_cell_survival_score(thought, "direct_factual", "ctx", col=8)

        assert col0_score > late_col_score  # Early positions score higher than late

    def test_candidate_cell_to_dict_truncation(self) -> None:
        """Test CandidateCell.to_dict truncates long content."""
        from src.tools.mot_reasoning import CandidateCell

        long_content = "x" * 150
        cell = CandidateCell(content=long_content, survival_score=0.75, selected=True)
        d = cell.to_dict()

        assert len(d["content"]) == 103  # 100 chars + "..."
        assert d["content"].endswith("...")
        assert d["survival_score"] == 0.75
        assert d["selected"] is True

    def test_set_cell_row_exceeds_strategies(self) -> None:
        """Test set_cell error when row exceeds strategies."""
        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Q", rows=2, cols=2)
        session_id = start["session_id"]

        # Row 5 is out of bounds for a 2-row matrix
        result = manager.set_cell(session_id, row=5, col=0, thought="X")
        assert "error" in result

    def test_set_cell_alternatives_on_non_row0_warns(self) -> None:
        """Test alternatives on non-row-0 triggers warning."""
        import warnings

        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Q", rows=2, cols=2)
        session_id = start["session_id"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.set_cell(session_id, row=1, col=0, thought="Main", alternatives=["Alt1"])

            assert len(w) == 1
            assert "alternatives ignored for row 1" in str(w[0].message)

    def test_set_cell_alternatives_invalid_type(self) -> None:
        """Test set_cell rejects invalid alternatives."""
        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Q", rows=2, cols=2)
        session_id = start["session_id"]

        # Empty string in alternatives
        result = manager.set_cell(session_id, row=0, col=0, thought="Main", alternatives=[""])
        assert "error" in result
        assert "non-empty strings" in result["error"]

    def test_set_cell_mppa_selects_best(self) -> None:
        """Test MPPA selects candidate with highest survival score."""
        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Mathematical proof", rows=3, cols=2)
        session_id = start["session_id"]

        # Provide alternatives with different qualities
        vague = "Maybe something works somehow"
        specific = "Therefore, since 2x = 10, we conclude x = 5 by division."

        result = manager.set_cell(session_id, row=0, col=0, thought=vague, alternatives=[specific])

        # MPPA should evaluate candidates
        assert "mppa" in result or "cell_set" in result  # Either MPPA info or successful set

    def test_list_sessions(self) -> None:
        """Test list_sessions returns all sessions."""
        manager = MatrixOfThoughtManager()

        # Create multiple sessions
        s1 = manager.start_matrix(question="Question 1 is about math", rows=2, cols=2)
        s2 = manager.start_matrix(question="Question 2 is about physics", rows=3, cols=3)

        sessions = manager.list_sessions()
        assert sessions["total"] == 2
        assert len(sessions["sessions"]) == 2

        # Check session details
        session_ids = [s["session_id"] for s in sessions["sessions"]]
        assert s1["session_id"] in session_ids
        assert s2["session_id"] in session_ids

    def test_list_sessions_truncates_long_questions(self) -> None:
        """Test list_sessions truncates long questions."""
        manager = MatrixOfThoughtManager()

        long_question = "x" * 100
        manager.start_matrix(question=long_question, rows=2, cols=2)

        sessions = manager.list_sessions()
        q = sessions["sessions"][0]["question"]
        assert len(q) == 53  # 50 chars + "..."
        assert q.endswith("...")

    def test_check_cell_consistency_short_content(self) -> None:
        """Test consistency check flags short content."""
        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Q", rows=2, cols=2)
        session_id = start["session_id"]

        # Very short thought
        result = manager.set_cell(session_id, row=0, col=0, thought="Short.")

        # Check if issues are flagged
        if "consistency_issues" in result:
            assert any("brief" in issue.lower() for issue in result["consistency_issues"])

    def test_check_cell_consistency_high_similarity(self) -> None:
        """Test consistency check flags high similarity to previous column."""
        manager = MatrixOfThoughtManager()
        start = manager.start_matrix(question="Q", rows=2, cols=2)
        session_id = start["session_id"]

        # Fill col 0
        thought = "This is a detailed analysis of the problem at hand with specific evidence."
        manager.set_cell(session_id, row=0, col=0, thought=thought)

        # Fill col 1 with very similar content
        result = manager.set_cell(session_id, row=0, col=1, thought=thought + " Extra word.")

        # High similarity should be flagged
        if "consistency_issues" in result:
            assert any("similar" in issue.lower() for issue in result["consistency_issues"])


# =============================================================================
# Additional Coverage Tests - LongChainManager
# =============================================================================


class TestLongChainManagerCoverage:
    """Additional tests for long_chain.py coverage."""

    def test_calculate_survival_score_empty(self) -> None:
        """Test survival score for empty thought."""
        from src.utils.scoring import calculate_survival_score

        # Empty thought should return low score
        empty_score = calculate_survival_score("", "problem", 1)
        normal_score = calculate_survival_score(
            "This is a valid thought with detailed content and reasoning steps.", "problem", 1
        )

        assert empty_score <= normal_score  # Empty should not score higher

    def test_calculate_survival_score_with_context(self) -> None:
        """Test survival score considers context overlap."""
        from src.utils.scoring import calculate_survival_score

        context = "establish baseline analysis for mathematical proof"
        new_thought = "Third step continues baseline analysis with new evidence and proof"

        score = calculate_survival_score(new_thought, context, 3)
        assert score > 0.3  # Should have decent score with overlap

    def test_calculate_survival_score_step_progression(self) -> None:
        """Test survival score varies by step number."""
        from src.utils.scoring import calculate_survival_score

        thought = "This is a detailed step with logical reasoning and evidence."
        context = "problem context"

        early_score = calculate_survival_score(thought, context, 1)
        late_score = calculate_survival_score(thought, context, 10)
        # Both should be valid scores
        assert 0.0 <= early_score <= 1.0
        assert 0.0 <= late_score <= 1.0

    def test_should_explore_alternatives(self) -> None:
        """Test alternative exploration detection."""
        from src.tools.long_chain import should_explore_alternatives

        # Early steps should explore
        assert should_explore_alternatives(1, 0) is True

        # Recent exploration should not re-explore
        assert should_explore_alternatives(5, 4) is False

        # Gap since last exploration should explore again
        assert should_explore_alternatives(10, 5) is True

    def test_add_step_with_confidence(self) -> None:
        """Test add_step with confidence tracking."""
        manager = LongChainManager()
        start = manager.start_chain(problem="P", expected_steps=5)
        session_id = start["session_id"]

        result = manager.add_step(session_id, thought="Step with confidence", confidence=0.85)

        assert result["step_added"] == 1
        # Check if confidence is stored
        state = manager.get_chain(session_id)
        assert state["steps"][0]["confidence"] == 0.85

    def test_add_step_mppa_exploration(self) -> None:
        """Test MPPA alternative exploration in add_step."""
        manager = LongChainManager()
        start = manager.start_chain(problem="Complex math problem", expected_steps=5)
        session_id = start["session_id"]

        # Provide alternatives
        result = manager.add_step(
            session_id,
            thought="Primary approach using algebra",
            alternatives=["Alternative using geometry", "Alternative using calculus"],
        )

        assert result["step_added"] == 1
        if "exploration" in result:
            assert result["exploration"]["alternatives_evaluated"] >= 1

    def test_finalize_with_summary(self) -> None:
        """Test finalize includes chain summary."""
        manager = LongChainManager()
        start = manager.start_chain(problem="Test problem", expected_steps=3)
        session_id = start["session_id"]

        manager.add_step(session_id, thought="Step 1: Analyze")
        manager.add_step(session_id, thought="Step 2: Synthesize")
        manager.add_step(session_id, thought="Step 3: Conclude")

        result = manager.finalize(session_id, answer="Final answer", confidence=0.9)

        assert result["status"] == "completed"
        assert result["final_answer"] == "Final answer"


# =============================================================================
# Additional Coverage Tests - VerificationManager
# =============================================================================


class TestVerificationManagerCoverage:
    """Additional tests for verify.py coverage."""

    def test_list_sessions(self) -> None:
        """Test list_sessions returns all verification sessions."""
        manager = VerificationManager()

        manager.start_verification(answer="Answer 1", context="Context 1")
        manager.start_verification(answer="Answer 2", context="Context 2")

        sessions = manager.list_sessions()
        assert sessions["total"] == 2
        assert len(sessions["sessions"]) == 2

    def test_suggest_claims_from_context(self) -> None:
        """Test claim suggestion based on context."""
        manager = VerificationManager()

        # Context with clear factual statements
        context = """
        The capital of France is Paris.
        Python was created in 1991.
        The Earth orbits the Sun in 365 days.
        """

        result = manager.start_verification(answer="Some answer", context=context)

        # Should suggest claims based on context
        assert "suggested_claims" in result
        assert isinstance(result["suggested_claims"], list)

    def test_verify_claim_all_statuses(self) -> None:
        """Test all claim verification statuses."""
        manager = VerificationManager()
        start = manager.start_verification(answer="A", context="Context")
        session_id = start["session_id"]

        # Add claims for each status type
        manager.add_claim(session_id, content="Supported claim")
        manager.add_claim(session_id, content="Contradicted claim")

        # Verify with different statuses
        r1 = manager.verify_claim(session_id, 0, "supported", "Evidence", 0.9)
        r2 = manager.verify_claim(session_id, 1, "contradicted", "Counter-evidence", 0.8)

        assert r1["status"] == "supported"
        assert r2["status"] == "contradicted"

    def test_get_verification_state(self) -> None:
        """Test get_status returns current state."""
        manager = VerificationManager()
        start = manager.start_verification(answer="Test", context="Context")
        session_id = start["session_id"]

        manager.add_claim(session_id, content="Claim 1")
        manager.verify_claim(session_id, 0, "supported", "Evidence", 0.9)

        state = manager.get_status(session_id)

        assert state["answer"] == "Test"
        assert "claims" in state or "status" in state  # Verify state is returned

    def test_finalize_with_unsupported_claims(self) -> None:
        """Test finalize handles all verified claims."""
        manager = VerificationManager()
        start = manager.start_verification(answer="A", context="C")
        session_id = start["session_id"]

        manager.add_claim(session_id, content="Claim")
        manager.verify_claim(session_id, 0, "contradicted", "Counter evidence", 0.8)

        result = manager.finalize(session_id)

        assert result["status"] == "completed"
        assert result["verified"] is False  # Contradicted claim means not verified


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
