"""Integration tests for MatrixMind MCP.

Tests the full workflow of tools working together.
"""

from __future__ import annotations

import pytest


class TestLongChainWorkflow:
    """Integration tests for long chain reasoning workflow."""

    def test_complete_chain_workflow(self) -> None:
        """Test complete chain reasoning from start to finish."""
        from src.tools.long_chain import LongChainManager

        manager = LongChainManager()

        # Start a reasoning chain
        start = manager.start_chain(
            problem="What is the sum of the first 5 prime numbers?",
            expected_steps=5,
        )
        session_id = start["session_id"]
        assert start["status"] == "started"

        # Step 1: Identify what we need
        step1 = manager.add_step(
            session_id,
            thought="I need to find the first 5 prime numbers: 2, 3, 5, 7, 11",
        )
        assert step1["step_added"] == 1

        # Step 2: Set up calculation
        step2 = manager.add_step(
            session_id,
            thought="Now I need to add them: 2 + 3 + 5 + 7 + 11",
        )
        assert step2["step_added"] == 2

        # Step 3: Calculate
        step3 = manager.add_step(
            session_id,
            thought="2 + 3 = 5, 5 + 5 = 10, 10 + 7 = 17, 17 + 11 = 28",
        )
        assert step3["step_added"] == 3

        # Finalize
        result = manager.finalize(session_id, answer="28", confidence=0.95)
        assert result["status"] == "completed"
        assert result["final_answer"] == "28"

    def test_chain_early_termination(self) -> None:
        """Test abandoning a chain before completion."""
        from src.tools.long_chain import LongChainManager

        manager = LongChainManager()

        start = manager.start_chain(problem="Test problem", expected_steps=10)
        session_id = start["session_id"]

        manager.add_step(session_id, thought="First thought")
        manager.add_step(session_id, thought="Second thought")

        # Abandon
        result = manager.abandon(session_id)
        assert result["status"] == "abandoned"

        # Session still exists but is abandoned
        state = manager.get_chain(session_id)
        assert state["status"] == "abandoned"


class TestMatrixOfThoughtWorkflow:
    """Integration tests for matrix of thought workflow."""

    def test_complete_matrix_workflow(self) -> None:
        """Test complete matrix reasoning from start to finish."""
        from src.tools.mot_reasoning import MatrixOfThoughtManager

        manager = MatrixOfThoughtManager()

        # Start a matrix session
        start = manager.start_matrix(
            question="Should we migrate to microservices?",
            rows=3,  # 3 perspectives
            cols=2,  # 2 criteria
            strategies=["technical", "business", "operational"],
        )
        session_id = start["session_id"]
        assert start["status"] == "started"
        assert start["matrix_dimensions"]["rows"] == 3
        assert start["matrix_dimensions"]["cols"] == 2

        # Fill the matrix (perspectives x criteria)
        # Row 0: Technical perspective
        manager.set_cell(session_id, 0, 0, thought="Technical pros: Better scalability")
        manager.set_cell(session_id, 0, 1, thought="Technical cons: Complexity")

        # Row 1: Business perspective
        manager.set_cell(session_id, 1, 0, thought="Business pros: Faster delivery")
        manager.set_cell(session_id, 1, 1, thought="Business cons: Higher cost")

        # Row 2: Operational perspective
        manager.set_cell(session_id, 2, 0, thought="Ops pros: Better fault isolation")
        manager.set_cell(session_id, 2, 1, thought="Ops cons: More infrastructure")

        # Finalize
        result = manager.finalize(
            session_id,
            answer="Yes, migrate for scalability benefits",
            confidence=0.75,
        )
        assert result["status"] == "completed"

    def test_matrix_partial_fill_finalize(self) -> None:
        """Test finalizing matrix with partial fill."""
        from src.tools.mot_reasoning import MatrixOfThoughtManager

        manager = MatrixOfThoughtManager()

        start = manager.start_matrix(question="Test?", rows=2, cols=2)
        session_id = start["session_id"]

        # Only fill some cells
        manager.set_cell(session_id, 0, 0, thought="Only one cell")

        # Should still be able to finalize
        result = manager.finalize(session_id, answer="Partial answer", confidence=0.5)
        assert result["status"] == "completed"


class TestVerificationWorkflow:
    """Integration tests for verification workflow."""

    def test_complete_verification_workflow(self) -> None:
        """Test complete verification from start to finish."""
        from src.tools.verify import VerificationManager

        manager = VerificationManager()

        # Context to verify against
        context = """
        The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.
        It was constructed from 1887 to 1889 and is 330 metres tall.
        It was named after the engineer Gustave Eiffel.
        """

        # Answer to verify
        answer = "The Eiffel Tower is 330 meters tall and was built by Gustave Eiffel."

        # Start verification
        start = manager.start_verification(answer=answer, context=context)
        session_id = start["session_id"]
        assert start["status"] == "started"

        # Add claims from the answer
        manager.add_claim(session_id, content="The Eiffel Tower is 330 meters tall")
        manager.add_claim(session_id, content="It was built by Gustave Eiffel")

        # Verify each claim
        manager.verify_claim(
            session_id,
            claim_id=0,
            status="supported",
            evidence="Context states: 'is 330 metres tall'",
            confidence=0.95,
        )
        manager.verify_claim(
            session_id,
            claim_id=1,
            status="supported",
            evidence="Context states: 'named after the engineer Gustave Eiffel'",
            confidence=0.85,
        )

        # Finalize
        result = manager.finalize(session_id)
        assert result["status"] == "completed"
        assert result["summary"]["total_claims"] == 2
        assert result["summary"]["supported"] == 2
        assert result["verified"] is True

    def test_verification_with_contradictions(self) -> None:
        """Test verification that finds contradictions."""
        from src.tools.verify import VerificationManager

        manager = VerificationManager()

        context = "The Great Wall of China is approximately 21,196 km long."
        answer = "The Great Wall is 5,000 km long."

        start = manager.start_verification(answer=answer, context=context)
        session_id = start["session_id"]

        # Add claim
        manager.add_claim(session_id, content="The Great Wall is 5,000 km long")

        # Verify - contradicts context
        manager.verify_claim(
            session_id,
            claim_id=0,
            status="contradicted",
            evidence="Context says 21,196 km, not 5,000 km",
            confidence=0.99,
        )

        result = manager.finalize(session_id)
        assert result["summary"]["contradicted"] == 1
        assert result["verified"] is False


class TestCrossToolWorkflow:
    """Tests for workflows using multiple tool types."""

    def test_chain_then_verify(self) -> None:
        """Test using chain reasoning followed by verification."""
        from src.tools.long_chain import LongChainManager
        from src.tools.verify import VerificationManager

        chain_manager = LongChainManager()
        verify_manager = VerificationManager()

        # First, use chain reasoning to arrive at an answer
        chain_start = chain_manager.start_chain(
            problem="What year did the French Revolution begin?",
            expected_steps=3,
        )
        chain_id = chain_start["session_id"]

        chain_manager.add_step(
            chain_id, thought="The French Revolution is a major historical event"
        )
        chain_manager.add_step(
            chain_id, thought="It started with the Storming of the Bastille in 1789"
        )

        chain_result = chain_manager.finalize(chain_id, answer="1789", confidence=0.9)
        assert chain_result["final_answer"] == "1789"

        # Now verify the answer
        context = (
            "The French Revolution began in 1789 with the Storming of the Bastille on July 14."
        )
        verify_start = verify_manager.start_verification(
            answer=chain_result["final_answer"],
            context=context,
        )
        verify_id = verify_start["session_id"]

        verify_manager.add_claim(verify_id, content="The French Revolution began in 1789")
        verify_manager.verify_claim(
            verify_id,
            claim_id=0,
            status="supported",
            evidence="Context confirms 1789",
            confidence=0.99,
        )

        verify_result = verify_manager.finalize(verify_id)
        assert verify_result["verified"] is True


class TestSessionIsolation:
    """Tests for session isolation between managers."""

    def test_sessions_are_isolated(self) -> None:
        """Test that sessions don't leak between manager instances."""
        from src.tools.long_chain import LongChainManager

        manager1 = LongChainManager()
        manager2 = LongChainManager()

        # Create session in manager1
        start1 = manager1.start_chain(problem="Problem 1", expected_steps=5)
        session_id = start1["session_id"]

        # Session should exist in manager1
        state1 = manager1.get_chain(session_id)
        assert state1["problem"] == "Problem 1"

        # But not in manager2 (fresh instance)
        with pytest.raises(ValueError, match="Session .* not found"):
            manager2.get_chain(session_id)

    def test_multiple_sessions_same_manager(self) -> None:
        """Test multiple sessions in the same manager are independent."""
        from src.tools.mot_reasoning import MatrixOfThoughtManager

        manager = MatrixOfThoughtManager()

        # Create two sessions
        start1 = manager.start_matrix(question="Question 1?", rows=2, cols=2)
        start2 = manager.start_matrix(question="Question 2?", rows=3, cols=3)

        # Modify first session
        manager.set_cell(start1["session_id"], 0, 0, thought="Session 1 content")

        # Second session should be unaffected
        state2 = manager.get_matrix(start2["session_id"])
        assert state2["cells_filled"] == 0
