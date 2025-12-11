"""Tests for src/utils/scoring.py semantic survival scoring."""

from __future__ import annotations

from src.utils.scoring import (
    LOGICAL_CONNECTORS,
    STRATEGY_KEYWORDS,
    VAGUE_PHRASES,
    _kg_alignment_score,
    _length_score,
    _position_score,
    _specificity_score,
    _structure_score,
    _word_overlap_score,
    calculate_cell_survival_score,
    calculate_survival_score,
    semantic_survival_score,
)


class TestWordOverlapScore:
    """Tests for word overlap fallback scoring."""

    def test_identical_text(self) -> None:
        """Test identical text returns high score."""
        score = _word_overlap_score("the cat sat on the mat", "the cat sat on the mat")
        assert score >= 0.8

    def test_no_overlap(self) -> None:
        """Test completely different text returns low score."""
        score = _word_overlap_score("apple banana cherry", "dog elephant fox")
        assert score < 0.3

    def test_partial_overlap(self) -> None:
        """Test partial overlap returns moderate score."""
        # Use words that aren't stopwords and have partial overlap
        score = _word_overlap_score("cat dog elephant", "cat dog fox")
        assert 0.3 < score < 0.9  # 2/3 overlap

    def test_empty_context(self) -> None:
        """Test empty context returns neutral score."""
        score = _word_overlap_score("some thought", "")
        assert score == 0.5

    def test_stopwords_filtered(self) -> None:
        """Test stopwords are filtered out."""
        # Only stopwords in context = empty context = neutral 0.5
        score = _word_overlap_score("the a an is are", "the a an was were")
        assert score == 0.5  # Empty context after filtering returns neutral


class TestSpecificityScore:
    """Tests for specificity scoring."""

    def test_vague_phrases_penalized(self) -> None:
        """Test vague phrases reduce score."""
        vague = "Maybe something could somehow work, I guess"
        specific = "The algorithm processes 1000 items per second"
        assert _specificity_score(vague) < _specificity_score(specific)

    def test_numbers_rewarded(self) -> None:
        """Test numeric content increases score."""
        with_numbers = "The result is 42.5 units"
        without_numbers = "The result is many units"
        assert _specificity_score(with_numbers) > _specificity_score(without_numbers)

    def test_quotes_rewarded(self) -> None:
        """Test quoted content increases score."""
        with_quotes = 'He said "hello world"'
        without_quotes = "He said hello world"
        assert _specificity_score(with_quotes) > _specificity_score(without_quotes)

    def test_proper_nouns_rewarded(self) -> None:
        """Test proper nouns increase score."""
        with_proper = "Einstein developed the theory at Princeton"
        without_proper = "someone developed the theory somewhere"
        assert _specificity_score(with_proper) > _specificity_score(without_proper)

    def test_score_bounded(self) -> None:
        """Test score is bounded 0-1."""
        # Very vague
        vague = " ".join(VAGUE_PHRASES)
        assert 0.0 <= _specificity_score(vague) <= 1.0
        # Very specific
        specific = 'At 3.14159, "John Smith" computed 42 results'
        assert 0.0 <= _specificity_score(specific) <= 1.0


class TestStructureScore:
    """Tests for logical structure scoring."""

    def test_logical_connectors_rewarded(self) -> None:
        """Test logical connectors increase score."""
        with_connectors = "Therefore, since X implies Y, we conclude Z"
        without_connectors = "X leads to Y leads to Z"
        assert _structure_score(with_connectors) > _structure_score(without_connectors)

    def test_strategy_keywords_rewarded(self) -> None:
        """Test strategy-specific keywords boost score."""
        causal = "This causes the effect because of the consequence"
        generic = "This thing relates to that thing"
        assert _structure_score(causal, strategy="causal") > _structure_score(
            generic, strategy="causal"
        )

    def test_unknown_strategy_ignored(self) -> None:
        """Test unknown strategy doesn't crash."""
        score = _structure_score("some thought", strategy="unknown_strategy")
        assert 0.0 <= score <= 1.0

    def test_none_strategy(self) -> None:
        """Test None strategy works."""
        score = _structure_score("therefore this follows", strategy=None)
        assert score > 0.5  # Has connector


class TestLengthScore:
    """Tests for length scoring."""

    def test_too_short_penalized(self) -> None:
        """Test very short thoughts are penalized."""
        short = "Yes"
        # Need 20+ words for ideal score (1.0)
        medium = " ".join(["word"] * 25)
        assert _length_score(short) < _length_score(medium)

    def test_ideal_range(self) -> None:
        """Test ideal length range scores well."""
        ideal = " ".join(["word"] * 50)  # 50 words
        assert _length_score(ideal) == 1.0

    def test_too_long_penalized(self) -> None:
        """Test very long thoughts are slightly penalized."""
        long = " ".join(["word"] * 200)  # 200 words
        ideal = " ".join(["word"] * 50)
        assert _length_score(long) < _length_score(ideal)


class TestPositionScore:
    """Tests for position-based scoring."""

    def test_early_positions_boosted(self) -> None:
        """Test early positions get higher scores."""
        early = _position_score(1)
        late = _position_score(8)
        assert early > late

    def test_first_position_max(self) -> None:
        """Test first position gets maximum score."""
        assert _position_score(0) == 1.0
        assert _position_score(1) == 1.0

    def test_score_bounded(self) -> None:
        """Test score bounded even for extreme positions."""
        assert 0.0 <= _position_score(0) <= 1.0
        assert 0.0 <= _position_score(100) <= 1.0


class TestKGAlignmentScore:
    """Tests for knowledge graph alignment scoring."""

    def test_none_kg_neutral(self) -> None:
        """Test None KG returns neutral score."""
        score = _kg_alignment_score("some thought", None)
        assert score == 0.5

    def test_no_supporting_facts_penalized(self) -> None:
        """Test lack of supporting facts is slightly penalized."""
        from src.models.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        score = _kg_alignment_score("unrelated thought about nothing", kg)
        assert score < 0.5

    def test_supporting_facts_rewarded(self) -> None:
        """Test supporting facts increase score."""
        from src.models.knowledge_graph import EntityType, KnowledgeGraph, RelationType

        kg = KnowledgeGraph()
        kg.add_entity("Einstein", EntityType.PERSON)
        kg.add_entity("Relativity", EntityType.CONCEPT)
        kg.add_relation("Einstein", RelationType.AUTHORED, "Relativity", confidence=0.9)

        score = _kg_alignment_score("Einstein developed Relativity", kg)
        assert score > 0.5


class TestSemanticSurvivalScore:
    """Tests for main semantic_survival_score function."""

    def test_empty_thought_returns_zero(self) -> None:
        """Test empty thought returns 0."""
        assert semantic_survival_score("", "context") == 0.0
        assert semantic_survival_score("   ", "context") == 0.0

    def test_empty_context_handled(self) -> None:
        """Test empty context doesn't crash."""
        score = semantic_survival_score("valid thought here", "")
        assert 0.0 <= score <= 1.0

    def test_score_bounded(self) -> None:
        """Test score always in 0-1 range."""
        score = semantic_survival_score(
            "Therefore, since Einstein's theory of Relativity (1905) shows E=mc^2",
            "Physics and mathematics in the 20th century",
            position=1,
        )
        assert 0.0 <= score <= 1.0

    def test_good_thought_scores_higher(self) -> None:
        """Test well-structured relevant thought scores higher than poor one."""
        context = "What is the capital of France?"
        good = (
            "Therefore, based on the question, Paris is the capital of France "
            "as stated in official records."
        )
        poor = "maybe something idk probably stuff"

        good_score = semantic_survival_score(good, context)
        poor_score = semantic_survival_score(poor, context)
        assert good_score > poor_score

    def test_strategy_affects_score(self) -> None:
        """Test strategy parameter affects scoring."""
        thought = "This causes the effect due to the consequence"
        context = "Analyze the causal relationship"

        causal_score = semantic_survival_score(thought, context, strategy="causal")
        factual_score = semantic_survival_score(thought, context, strategy="direct_factual")
        # Causal keywords should boost score for causal strategy
        assert causal_score >= factual_score


class TestBackwardCompatibility:
    """Tests for backward-compatible wrapper functions."""

    def test_calculate_survival_score_signature(self) -> None:
        """Test calculate_survival_score matches expected signature."""
        score = calculate_survival_score(
            thought="This is a test thought",
            context="Test context",
            step_number=1,
        )
        assert 0.0 <= score <= 1.0

    def test_calculate_survival_score_with_encoder(self) -> None:
        """Test calculate_survival_score accepts encoder parameter."""
        score = calculate_survival_score(
            thought="Test thought",
            context="Test context",
            step_number=1,
            encoder=None,
            kg=None,
        )
        assert 0.0 <= score <= 1.0

    def test_calculate_cell_survival_score_signature(self) -> None:
        """Test calculate_cell_survival_score matches expected signature."""
        score = calculate_cell_survival_score(
            thought="This is a test thought",
            strategy="logical_inference",
            context="Test context",
            col=0,
        )
        assert 0.0 <= score <= 1.0

    def test_calculate_cell_survival_score_with_encoder(self) -> None:
        """Test calculate_cell_survival_score accepts encoder parameter."""
        score = calculate_cell_survival_score(
            thought="Test thought",
            strategy="causal",
            context="Test context",
            col=0,
            encoder=None,
            kg=None,
        )
        assert 0.0 <= score <= 1.0


class TestConstantsExist:
    """Tests that important constants are exported."""

    def test_vague_phrases_nonempty(self) -> None:
        """Test VAGUE_PHRASES is populated."""
        assert len(VAGUE_PHRASES) > 0
        assert "maybe" in VAGUE_PHRASES

    def test_logical_connectors_nonempty(self) -> None:
        """Test LOGICAL_CONNECTORS is populated."""
        assert len(LOGICAL_CONNECTORS) > 0
        assert "therefore" in LOGICAL_CONNECTORS

    def test_strategy_keywords_complete(self) -> None:
        """Test STRATEGY_KEYWORDS has all expected strategies."""
        expected = {"direct_factual", "logical_inference", "analogical", "causal", "counterfactual"}
        assert set(STRATEGY_KEYWORDS.keys()) == expected
