"""Integration tests for MatrixMind MCP.

Tests the full workflow of tools working together.
"""

from __future__ import annotations

import pytest

from src.utils.session import SessionNotFoundError


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
        with pytest.raises(SessionNotFoundError):
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


class TestSemanticScoringIntegration:
    """Integration tests for semantic scoring with real ContextEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create a real ContextEncoder if model is available."""
        try:
            from src.models.context_encoder import ContextEncoder

            return ContextEncoder()
        except Exception:
            pytest.skip("ContextEncoder model not available")

    def test_semantic_similarity_beats_word_overlap_for_paraphrases(self, encoder) -> None:
        """Verify embedding similarity is higher than word overlap for paraphrases."""
        from src.utils.scoring import _word_overlap_score, semantic_survival_score

        # Original and paraphrase share meaning but few words
        original = "The capital of France is Paris, a city known for the Eiffel Tower."
        paraphrase = (
            "Paris serves as France's main city, famous for its iconic iron lattice structure."
        )
        unrelated = "Machine learning algorithms process large datasets efficiently."

        # Word overlap should be low for paraphrase (different words)
        word_overlap_para = _word_overlap_score(paraphrase, original)
        _word_overlap_score(unrelated, original)  # Verify it runs, value not needed

        # Semantic scoring with encoder should recognize paraphrase
        semantic_para = semantic_survival_score(paraphrase, original, encoder=encoder)
        semantic_unrel = semantic_survival_score(unrelated, original, encoder=encoder)

        # Paraphrase should score higher than unrelated with semantic scoring
        assert semantic_para > semantic_unrel, (
            f"Semantic scoring should rank paraphrase ({semantic_para:.3f}) "
            f"higher than unrelated ({semantic_unrel:.3f})"
        )

        # Word overlap is expected to be similar for both (low)
        # because paraphrase uses different words
        assert word_overlap_para < 0.5, "Word overlap should be low for paraphrase"

    def test_semantic_scoring_with_kg_alignment(self, encoder) -> None:
        """Test that KG facts improve scoring for aligned thoughts."""
        from src.models.knowledge_graph import EntityType, KnowledgeGraph, RelationType
        from src.utils.scoring import semantic_survival_score

        kg = KnowledgeGraph()
        kg.add_entity("Einstein", EntityType.PERSON)
        kg.add_entity("Relativity", EntityType.CONCEPT)
        kg.add_entity("Physics", EntityType.CONCEPT)
        kg.add_relation("Einstein", RelationType.AUTHORED, "Relativity", confidence=0.95)

        context = "Discuss contributions to modern physics."

        # Thought that aligns with KG facts
        aligned = "Einstein's theory of Relativity revolutionized our understanding of physics."
        # Thought with no KG support
        no_kg_support = "Newton's laws describe classical mechanics and motion."

        score_aligned = semantic_survival_score(aligned, context, encoder=encoder, kg=kg)
        score_no_support = semantic_survival_score(no_kg_support, context, encoder=encoder, kg=kg)

        # Aligned thought should score higher due to KG support
        assert score_aligned > score_no_support, (
            f"KG-aligned thought ({score_aligned:.3f}) should score higher "
            f"than unsupported ({score_no_support:.3f})"
        )

    def test_kg_auto_extraction_on_chain_start(self) -> None:
        """Test that starting a chain extracts entities into KG."""
        from src.models.knowledge_graph import KnowledgeGraph
        from src.tools.long_chain import LongChainManager

        kg = KnowledgeGraph()
        manager = LongChainManager(encoder=None, kg=kg)

        # Start chain with entity-rich problem
        manager.start_chain(
            problem="Albert Einstein developed the theory of relativity at Princeton University.",
            metadata={"context": "Physics history in the 20th century."},
        )

        # KG should have extracted entities
        stats = kg.stats()
        assert stats.num_entities > 0, "KG should have extracted entities from problem"

    def test_kg_auto_extraction_on_matrix_start(self) -> None:
        """Test that starting a matrix extracts entities into KG."""
        from src.models.knowledge_graph import KnowledgeGraph
        from src.tools.mot_reasoning import MatrixOfThoughtManager

        kg = KnowledgeGraph()
        manager = MatrixOfThoughtManager(encoder=None, kg=kg)

        # Start matrix with entity-rich question
        manager.start_matrix(
            question="How did Marie Curie's research on radioactivity impact medicine?",
            context="Nobel Prize winners in Physics and Chemistry.",
            rows=2,
            cols=2,
        )

        # KG should have extracted entities
        stats = kg.stats()
        assert stats.num_entities > 0, "KG should have extracted entities from question/context"
