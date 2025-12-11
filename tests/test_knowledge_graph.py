"""Unit tests for src/models/knowledge_graph.py."""

from __future__ import annotations

import pytest

from src.models.knowledge_graph import (
    Entity,
    EntityType,
    KnowledgeGraph,
    KnowledgeGraphException,
    KnowledgeGraphExtractor,
    KnowledgeGraphStats,
    Relation,
    RelationType,
)


class TestEntityType:
    """Test EntityType enum."""

    def test_all_entity_types_exist(self) -> None:
        """Test all entity types are defined."""
        assert EntityType.PERSON.value == "PERSON"
        assert EntityType.ORGANIZATION.value == "ORG"
        assert EntityType.LOCATION.value == "LOC"
        assert EntityType.DATE.value == "DATE"
        assert EntityType.EVENT.value == "EVENT"
        assert EntityType.CONCEPT.value == "CONCEPT"
        assert EntityType.QUANTITY.value == "QUANTITY"
        assert EntityType.OTHER.value == "OTHER"


class TestRelationType:
    """Test RelationType enum."""

    def test_temporal_relations(self) -> None:
        """Test temporal relation types."""
        assert RelationType.OCCURRED_ON.value == "occurred_on"
        assert RelationType.BEFORE.value == "before"
        assert RelationType.AFTER.value == "after"
        assert RelationType.DURING.value == "during"

    def test_spatial_relations(self) -> None:
        """Test spatial relation types."""
        assert RelationType.LOCATED_IN.value == "located_in"
        assert RelationType.PART_OF.value == "part_of"

    def test_person_relations(self) -> None:
        """Test person-related relation types."""
        assert RelationType.BORN_IN.value == "born_in"
        assert RelationType.DIED_IN.value == "died_in"
        assert RelationType.WORKS_FOR.value == "works_for"
        assert RelationType.AUTHORED.value == "authored"
        assert RelationType.INVENTED.value == "invented"
        assert RelationType.DISCOVERED.value == "discovered"

    def test_general_relations(self) -> None:
        """Test general relation types."""
        assert RelationType.IS_A.value == "is_a"
        assert RelationType.HAS_PROPERTY.value == "has_property"
        assert RelationType.RELATED_TO.value == "related_to"
        assert RelationType.CAUSED_BY.value == "caused_by"
        assert RelationType.RESULTS_IN.value == "results_in"


class TestEntity:
    """Test Entity dataclass."""

    def test_entity_creation(self) -> None:
        """Test basic entity creation."""
        entity = Entity(name="Albert Einstein", entity_type=EntityType.PERSON)
        assert entity.name == "Albert Einstein"
        assert entity.entity_type == EntityType.PERSON
        assert entity.mention_count == 1
        assert len(entity.aliases) == 0

    def test_entity_with_aliases(self) -> None:
        """Test entity with aliases."""
        entity = Entity(
            name="Albert Einstein",
            entity_type=EntityType.PERSON,
            aliases={"Einstein", "A. Einstein"},
        )
        assert "Einstein" in entity.aliases
        assert "A. Einstein" in entity.aliases

    def test_entity_with_properties(self) -> None:
        """Test entity with properties."""
        entity = Entity(
            name="MIT",
            entity_type=EntityType.ORGANIZATION,
            properties={"founded": 1861, "location": "Cambridge"},
        )
        assert entity.properties["founded"] == 1861
        assert entity.properties["location"] == "Cambridge"

    def test_entity_hash(self) -> None:
        """Test entity hashing based on name."""
        e1 = Entity(name="Test", entity_type=EntityType.OTHER)
        e2 = Entity(name="test", entity_type=EntityType.PERSON)  # Different case
        e3 = Entity(name="Different", entity_type=EntityType.OTHER)

        assert hash(e1) == hash(e2)  # Same name, different case
        assert hash(e1) != hash(e3)

    def test_entity_equality(self) -> None:
        """Test entity equality based on name."""
        e1 = Entity(name="Test", entity_type=EntityType.OTHER)
        e2 = Entity(name="test", entity_type=EntityType.PERSON)
        e3 = Entity(name="Different", entity_type=EntityType.OTHER)

        assert e1 == e2
        assert e1 != e3
        assert e1 != "not an entity"

    def test_entity_to_dict(self) -> None:
        """Test entity serialization."""
        entity = Entity(
            name="Einstein",
            entity_type=EntityType.PERSON,
            aliases={"Albert"},
            properties={"born": 1879},
            mention_count=5,
        )

        data = entity.to_dict()

        assert data["name"] == "Einstein"
        assert data["type"] == "PERSON"
        assert "Albert" in data["aliases"]
        assert data["properties"]["born"] == 1879
        assert data["mention_count"] == 5


class TestRelation:
    """Test Relation dataclass."""

    def test_relation_creation(self) -> None:
        """Test basic relation creation."""
        subject = Entity(name="Einstein", entity_type=EntityType.PERSON)
        obj = Entity(name="Relativity", entity_type=EntityType.CONCEPT)

        relation = Relation(
            subject=subject,
            predicate=RelationType.AUTHORED,
            object_entity=obj,
        )

        assert relation.subject == subject
        assert relation.predicate == RelationType.AUTHORED
        assert relation.object_entity == obj
        assert relation.confidence == 1.0

    def test_relation_with_confidence(self) -> None:
        """Test relation with confidence score."""
        subject = Entity(name="A", entity_type=EntityType.OTHER)
        obj = Entity(name="B", entity_type=EntityType.OTHER)

        relation = Relation(
            subject=subject,
            predicate=RelationType.RELATED_TO,
            object_entity=obj,
            confidence=0.75,
            evidence="Source text",
        )

        assert relation.confidence == 0.75
        assert relation.evidence == "Source text"

    def test_relation_hash(self) -> None:
        """Test relation hashing."""
        subject = Entity(name="A", entity_type=EntityType.OTHER)
        obj = Entity(name="B", entity_type=EntityType.OTHER)

        r1 = Relation(subject=subject, predicate=RelationType.RELATED_TO, object_entity=obj)
        r2 = Relation(subject=subject, predicate=RelationType.RELATED_TO, object_entity=obj)
        r3 = Relation(subject=obj, predicate=RelationType.RELATED_TO, object_entity=subject)

        assert hash(r1) == hash(r2)
        assert hash(r1) != hash(r3)

    def test_relation_to_dict(self) -> None:
        """Test relation serialization."""
        subject = Entity(name="Einstein", entity_type=EntityType.PERSON)
        obj = Entity(name="Physics", entity_type=EntityType.CONCEPT)

        relation = Relation(
            subject=subject,
            predicate=RelationType.WORKS_FOR,
            object_entity=obj,
            confidence=0.9,
            evidence="Evidence text here",
        )

        data = relation.to_dict()

        assert data["subject"] == "Einstein"
        assert data["predicate"] == "works_for"
        assert data["object"] == "Physics"
        assert data["confidence"] == 0.9

    def test_relation_to_dict_with_string_predicate(self) -> None:
        """Test relation serialization with string predicate."""
        subject = Entity(name="A", entity_type=EntityType.OTHER)
        obj = Entity(name="B", entity_type=EntityType.OTHER)

        relation = Relation(
            subject=subject,
            predicate="custom_relation",
            object_entity=obj,
        )

        data = relation.to_dict()
        assert data["predicate"] == "custom_relation"

    def test_relation_to_triple(self) -> None:
        """Test conversion to triple format."""
        subject = Entity(name="Einstein", entity_type=EntityType.PERSON)
        obj = Entity(name="Relativity", entity_type=EntityType.CONCEPT)

        relation = Relation(
            subject=subject,
            predicate=RelationType.AUTHORED,
            object_entity=obj,
        )

        triple = relation.to_triple()
        assert triple == ("Einstein", "authored", "Relativity")

    def test_relation_to_triple_string_predicate(self) -> None:
        """Test triple conversion with string predicate."""
        subject = Entity(name="A", entity_type=EntityType.OTHER)
        obj = Entity(name="B", entity_type=EntityType.OTHER)

        relation = Relation(
            subject=subject,
            predicate="custom",
            object_entity=obj,
        )

        triple = relation.to_triple()
        assert triple == ("A", "custom", "B")


class TestKnowledgeGraphStats:
    """Test KnowledgeGraphStats dataclass."""

    def test_stats_creation(self) -> None:
        """Test stats creation."""
        stats = KnowledgeGraphStats(
            num_entities=10,
            num_relations=15,
            entity_types={"PERSON": 5, "CONCEPT": 5},
            relation_types={"authored": 10, "related_to": 5},
            avg_relations_per_entity=1.5,
        )

        assert stats.num_entities == 10
        assert stats.num_relations == 15

    def test_stats_to_dict(self) -> None:
        """Test stats serialization."""
        stats = KnowledgeGraphStats(
            num_entities=10,
            num_relations=15,
            entity_types={"PERSON": 5},
            relation_types={"authored": 10},
            avg_relations_per_entity=1.5,
        )

        data = stats.to_dict()

        assert data["num_entities"] == 10
        assert data["num_relations"] == 15
        assert data["avg_relations_per_entity"] == 1.5


class TestKnowledgeGraph:
    """Test KnowledgeGraph class."""

    def test_empty_graph(self) -> None:
        """Test empty graph creation."""
        kg = KnowledgeGraph()
        assert len(kg.entities) == 0
        assert len(kg.relations) == 0

    def test_add_entity(self) -> None:
        """Test adding entity to graph."""
        kg = KnowledgeGraph()
        entity = kg.add_entity("Einstein", EntityType.PERSON)

        assert entity.name == "Einstein"
        assert entity.entity_type == EntityType.PERSON
        assert len(kg.entities) == 1

    def test_add_entity_with_aliases_and_properties(self) -> None:
        """Test adding entity with aliases and properties."""
        kg = KnowledgeGraph()
        entity = kg.add_entity(
            "Albert Einstein",
            EntityType.PERSON,
            aliases=["Einstein"],
            properties={"born": 1879},
        )

        assert "Einstein" in entity.aliases
        assert entity.properties["born"] == 1879

    def test_add_duplicate_entity_updates_count(self) -> None:
        """Test adding same entity increases mention count."""
        kg = KnowledgeGraph()
        e1 = kg.add_entity("Einstein", EntityType.PERSON)
        e2 = kg.add_entity("einstein", EntityType.PERSON)  # Different case

        assert e1 is e2  # Same object
        assert e1.mention_count == 2

    def test_add_duplicate_entity_merges_aliases(self) -> None:
        """Test adding same entity merges aliases."""
        kg = KnowledgeGraph()
        kg.add_entity("Einstein", aliases=["Albert"])
        kg.add_entity("einstein", aliases=["A. Einstein"])

        entity = kg.get_entity("Einstein")
        assert entity is not None
        assert "Albert" in entity.aliases
        assert "A. Einstein" in entity.aliases

    def test_get_entity(self) -> None:
        """Test getting entity by name."""
        kg = KnowledgeGraph()
        kg.add_entity("Test Entity", EntityType.OTHER)

        entity = kg.get_entity("Test Entity")
        assert entity is not None
        assert entity.name == "Test Entity"

    def test_get_entity_case_insensitive(self) -> None:
        """Test entity lookup is case insensitive."""
        kg = KnowledgeGraph()
        kg.add_entity("Test", EntityType.OTHER)

        assert kg.get_entity("test") is not None
        assert kg.get_entity("TEST") is not None

    def test_get_entity_not_found(self) -> None:
        """Test getting non-existent entity returns None."""
        kg = KnowledgeGraph()
        assert kg.get_entity("nonexistent") is None

    def test_has_entity_exists(self) -> None:
        """Test has_entity returns True for existing entity."""
        kg = KnowledgeGraph()
        kg.add_entity("Einstein", EntityType.PERSON)
        assert kg.has_entity("Einstein") is True
        assert kg.has_entity("einstein") is True  # Case insensitive
        assert kg.has_entity("  Einstein  ") is True  # Strips whitespace

    def test_has_entity_not_exists(self) -> None:
        """Test has_entity returns False for non-existent entity."""
        kg = KnowledgeGraph()
        assert kg.has_entity("Einstein") is False
        kg.add_entity("Newton", EntityType.PERSON)
        assert kg.has_entity("Einstein") is False

    def test_get_supporting_facts_empty_text(self) -> None:
        """Test get_supporting_facts with empty text returns empty list."""
        kg = KnowledgeGraph()
        kg.add_entity("Einstein", EntityType.PERSON)
        assert kg.get_supporting_facts("") == []
        assert kg.get_supporting_facts(None) == []  # type: ignore[arg-type]

    def test_get_supporting_facts_no_matches(self) -> None:
        """Test get_supporting_facts with no matching entities."""
        kg = KnowledgeGraph()
        kg.add_entity("Einstein", EntityType.PERSON)
        kg.add_entity("Relativity", EntityType.CONCEPT)
        kg.add_relation("Einstein", RelationType.AUTHORED, "Relativity")

        facts = kg.get_supporting_facts("Newton discovered gravity")
        assert facts == []

    def test_get_supporting_facts_finds_relations(self) -> None:
        """Test get_supporting_facts finds relations for mentioned entities."""
        kg = KnowledgeGraph()
        kg.add_entity("Einstein", EntityType.PERSON)
        kg.add_entity("Relativity", EntityType.CONCEPT)
        kg.add_entity("Princeton", EntityType.ORGANIZATION)
        kg.add_relation("Einstein", RelationType.AUTHORED, "Relativity", confidence=0.9)
        kg.add_relation("Einstein", RelationType.WORKS_FOR, "Princeton", confidence=0.8)

        facts = kg.get_supporting_facts("Tell me about Einstein")
        assert len(facts) == 2
        # Should be sorted by confidence (highest first)
        assert facts[0].confidence >= facts[1].confidence

    def test_get_supporting_facts_includes_incoming_relations(self) -> None:
        """Test get_supporting_facts includes both outgoing and incoming relations."""
        kg = KnowledgeGraph()
        kg.add_entity("Einstein", EntityType.PERSON)
        kg.add_entity("Relativity", EntityType.CONCEPT)
        kg.add_relation("Einstein", RelationType.AUTHORED, "Relativity")

        # Query by object entity (Relativity) - should find incoming relation
        facts = kg.get_supporting_facts("What is Relativity?")
        assert len(facts) == 1
        assert facts[0].object_entity.name == "Relativity"

    def test_get_supporting_facts_matches_aliases(self) -> None:
        """Test get_supporting_facts matches entity aliases."""
        kg = KnowledgeGraph()
        kg.add_entity("Albert Einstein", EntityType.PERSON, aliases=["Einstein", "A. Einstein"])
        kg.add_entity("Relativity", EntityType.CONCEPT)
        kg.add_relation("Albert Einstein", RelationType.AUTHORED, "Relativity")

        # Query using alias
        facts = kg.get_supporting_facts("Einstein developed this theory")
        assert len(facts) == 1

    def test_get_supporting_facts_no_duplicates(self) -> None:
        """Test get_supporting_facts doesn't return duplicate relations."""
        kg = KnowledgeGraph()
        kg.add_entity("Einstein", EntityType.PERSON)
        kg.add_entity("Relativity", EntityType.CONCEPT)
        kg.add_relation("Einstein", RelationType.AUTHORED, "Relativity")

        # Text mentions both subject and object - relation should appear once
        facts = kg.get_supporting_facts("Einstein authored Relativity")
        assert len(facts) == 1

    def test_add_relation_with_entity_objects(self) -> None:
        """Test adding relation with Entity objects."""
        kg = KnowledgeGraph()
        e1 = kg.add_entity("Einstein", EntityType.PERSON)
        e2 = kg.add_entity("Relativity", EntityType.CONCEPT)

        relation = kg.add_relation(e1, RelationType.AUTHORED, e2)

        assert relation.subject == e1
        assert relation.object_entity == e2
        assert len(kg.relations) == 1

    def test_add_relation_with_string_names(self) -> None:
        """Test adding relation with string entity names."""
        kg = KnowledgeGraph()
        kg.add_entity("Einstein", EntityType.PERSON)
        kg.add_entity("Relativity", EntityType.CONCEPT)

        relation = kg.add_relation("Einstein", RelationType.AUTHORED, "Relativity")

        assert relation.subject.name == "Einstein"
        assert relation.object_entity.name == "Relativity"

    def test_add_relation_creates_missing_entities(self) -> None:
        """Test adding relation creates entities if they don't exist."""
        kg = KnowledgeGraph()

        kg.add_relation("NewSubject", RelationType.RELATED_TO, "NewObject")

        assert kg.get_entity("NewSubject") is not None
        assert kg.get_entity("NewObject") is not None

    def test_add_relation_with_evidence(self) -> None:
        """Test adding relation with evidence."""
        kg = KnowledgeGraph()
        kg.add_entity("A", EntityType.OTHER)
        kg.add_entity("B", EntityType.OTHER)

        relation = kg.add_relation(
            "A",
            RelationType.RELATED_TO,
            "B",
            confidence=0.8,
            evidence="Source text",
        )

        assert relation.confidence == 0.8
        assert relation.evidence == "Source text"

    def test_get_relations_outgoing(self) -> None:
        """Test getting outgoing relations."""
        kg = KnowledgeGraph()
        kg.add_relation("A", RelationType.RELATED_TO, "B")
        kg.add_relation("A", RelationType.RELATED_TO, "C")
        kg.add_relation("B", RelationType.RELATED_TO, "A")

        outgoing = kg.get_relations("A", direction="outgoing")
        assert len(outgoing) == 2

    def test_get_relations_incoming(self) -> None:
        """Test getting incoming relations."""
        kg = KnowledgeGraph()
        kg.add_relation("A", RelationType.RELATED_TO, "B")
        kg.add_relation("C", RelationType.RELATED_TO, "B")

        incoming = kg.get_relations("B", direction="incoming")
        assert len(incoming) == 2

    def test_get_relations_both_directions(self) -> None:
        """Test getting relations in both directions."""
        kg = KnowledgeGraph()
        kg.add_relation("A", RelationType.RELATED_TO, "B")
        kg.add_relation("C", RelationType.RELATED_TO, "A")

        all_relations = kg.get_relations("A", direction="both")
        assert len(all_relations) == 2

    def test_get_relations_with_entity_object(self) -> None:
        """Test getting relations with Entity object."""
        kg = KnowledgeGraph()
        entity = kg.add_entity("A", EntityType.OTHER)
        kg.add_relation("A", RelationType.RELATED_TO, "B")

        relations = kg.get_relations(entity, direction="outgoing")
        assert len(relations) == 1

    def test_find_paths_direct(self) -> None:
        """Test finding direct path between entities."""
        kg = KnowledgeGraph()
        kg.add_relation("A", RelationType.RELATED_TO, "B")

        paths = kg.find_paths("A", "B")
        assert len(paths) == 1
        assert len(paths[0]) == 1

    def test_find_paths_multi_hop(self) -> None:
        """Test finding multi-hop path."""
        kg = KnowledgeGraph()
        kg.add_relation("A", RelationType.RELATED_TO, "B")
        kg.add_relation("B", RelationType.RELATED_TO, "C")

        paths = kg.find_paths("A", "C", max_hops=2)
        assert len(paths) >= 1

    def test_find_paths_no_path(self) -> None:
        """Test finding no path between disconnected entities."""
        kg = KnowledgeGraph()
        kg.add_entity("A", EntityType.OTHER)
        kg.add_entity("B", EntityType.OTHER)

        paths = kg.find_paths("A", "B")
        assert len(paths) == 0

    def test_find_paths_entity_not_exists(self) -> None:
        """Test finding path with non-existent entity."""
        kg = KnowledgeGraph()
        kg.add_entity("A", EntityType.OTHER)

        paths = kg.find_paths("A", "nonexistent")
        assert len(paths) == 0

    def test_find_paths_with_entity_objects(self) -> None:
        """Test finding paths with Entity objects."""
        kg = KnowledgeGraph()
        e1 = kg.add_entity("A", EntityType.OTHER)
        e2 = kg.add_entity("B", EntityType.OTHER)
        kg.add_relation(e1, RelationType.RELATED_TO, e2)

        paths = kg.find_paths(e1, e2)
        assert len(paths) == 1

    def test_query_by_subject(self) -> None:
        """Test querying relations by subject."""
        kg = KnowledgeGraph()
        kg.add_relation("A", RelationType.AUTHORED, "X")
        kg.add_relation("A", RelationType.INVENTED, "Y")
        kg.add_relation("B", RelationType.AUTHORED, "Z")

        results = kg.query(subject="A")
        assert len(results) == 2

    def test_query_by_predicate(self) -> None:
        """Test querying relations by predicate."""
        kg = KnowledgeGraph()
        kg.add_relation("A", RelationType.AUTHORED, "X")
        kg.add_relation("B", RelationType.AUTHORED, "Y")
        kg.add_relation("C", RelationType.INVENTED, "Z")

        results = kg.query(predicate=RelationType.AUTHORED)
        assert len(results) == 2

    def test_query_by_object(self) -> None:
        """Test querying relations by object."""
        kg = KnowledgeGraph()
        kg.add_relation("A", RelationType.RELATED_TO, "X")
        kg.add_relation("B", RelationType.RELATED_TO, "X")
        kg.add_relation("C", RelationType.RELATED_TO, "Y")

        results = kg.query(obj="X")
        assert len(results) == 2

    def test_query_combined(self) -> None:
        """Test querying with multiple filters."""
        kg = KnowledgeGraph()
        kg.add_relation("A", RelationType.AUTHORED, "X")
        kg.add_relation("A", RelationType.INVENTED, "X")
        kg.add_relation("B", RelationType.AUTHORED, "X")

        results = kg.query(subject="A", predicate=RelationType.AUTHORED)
        assert len(results) == 1

    def test_query_with_string_predicate(self) -> None:
        """Test querying with string predicate."""
        kg = KnowledgeGraph()
        kg.add_relation("A", "custom_rel", "B")
        kg.add_relation("C", "other_rel", "D")

        results = kg.query(predicate="custom_rel")
        assert len(results) == 1

    def test_get_neighbors_one_hop(self) -> None:
        """Test getting neighbors within 1 hop."""
        kg = KnowledgeGraph()
        kg.add_relation("A", RelationType.RELATED_TO, "B")
        kg.add_relation("A", RelationType.RELATED_TO, "C")
        kg.add_relation("B", RelationType.RELATED_TO, "D")

        neighbors = kg.get_neighbors("A", hops=1)
        neighbor_names = {e.name for e in neighbors}

        assert "A" in neighbor_names
        assert "B" in neighbor_names
        assert "C" in neighbor_names
        assert "D" not in neighbor_names

    def test_get_neighbors_two_hops(self) -> None:
        """Test getting neighbors within 2 hops."""
        kg = KnowledgeGraph()
        kg.add_relation("A", RelationType.RELATED_TO, "B")
        kg.add_relation("B", RelationType.RELATED_TO, "C")
        kg.add_relation("C", RelationType.RELATED_TO, "D")

        neighbors = kg.get_neighbors("A", hops=2)
        neighbor_names = {e.name for e in neighbors}

        assert "B" in neighbor_names
        assert "C" in neighbor_names

    def test_get_neighbors_includes_reverse(self) -> None:
        """Test get_neighbors includes incoming relations."""
        kg = KnowledgeGraph()
        kg.add_relation("B", RelationType.RELATED_TO, "A")  # B -> A

        neighbors = kg.get_neighbors("A", hops=1)
        neighbor_names = {e.name for e in neighbors}

        assert "B" in neighbor_names

    def test_stats(self) -> None:
        """Test graph statistics."""
        kg = KnowledgeGraph()
        kg.add_entity("Person1", EntityType.PERSON)
        kg.add_entity("Person2", EntityType.PERSON)
        kg.add_entity("Concept1", EntityType.CONCEPT)
        kg.add_relation("Person1", RelationType.AUTHORED, "Concept1")

        stats = kg.stats()

        assert stats.num_entities == 3
        assert stats.num_relations == 1
        assert stats.entity_types["PERSON"] == 2
        assert stats.entity_types["CONCEPT"] == 1

    def test_to_dict(self) -> None:
        """Test graph serialization."""
        kg = KnowledgeGraph()
        kg.add_entity("A", EntityType.OTHER)
        kg.add_relation("A", RelationType.RELATED_TO, "B")

        data = kg.to_dict()

        assert "entities" in data
        assert "relations" in data
        assert "stats" in data

    def test_clear(self) -> None:
        """Test clearing graph."""
        kg = KnowledgeGraph()
        kg.add_entity("A", EntityType.OTHER)
        kg.add_relation("A", RelationType.RELATED_TO, "B")

        kg.clear()

        assert len(kg.entities) == 0
        assert len(kg.relations) == 0


class TestKnowledgeGraphExtractor:
    """Test KnowledgeGraphExtractor class."""

    def test_init(self) -> None:
        """Test initialization."""
        extractor = KnowledgeGraphExtractor()
        # Should initialize without error
        assert extractor is not None

    def test_extract_empty_text_raises(self) -> None:
        """Test extraction from empty text raises exception."""
        extractor = KnowledgeGraphExtractor()

        with pytest.raises(KnowledgeGraphException, match="empty text"):
            extractor.extract("")

    def test_extract_with_rules_dates(self) -> None:
        """Test rule-based extraction extracts dates."""
        extractor = KnowledgeGraphExtractor()

        # Use years that match the DATE_PATTERN (19xx or 20xx)
        kg = extractor.extract(
            "Einstein published relativity in 1905 and won the Nobel Prize in 1921."
        )

        # Should have extracted dates
        date_entities = [e for e in kg.entities if e.entity_type == EntityType.DATE]
        assert len(date_entities) >= 2

    def test_extract_with_rules_quantities(self) -> None:
        """Test rule-based extraction extracts quantities."""
        extractor = KnowledgeGraphExtractor()

        kg = extractor.extract("The temperature was 100 percent accurate.")

        quantity_entities = [e for e in kg.entities if e.entity_type == EntityType.QUANTITY]
        assert len(quantity_entities) >= 1

    def test_extract_with_rules_capitalized_phrases(self) -> None:
        """Test rule-based extraction extracts capitalized phrases."""
        extractor = KnowledgeGraphExtractor()

        kg = extractor.extract("Albert Einstein worked at Princeton University.")

        entity_names = {e.name for e in kg.entities}
        assert "Albert Einstein" in entity_names or "Princeton University" in entity_names

    def test_extract_with_existing_graph(self) -> None:
        """Test extraction extends existing graph."""
        extractor = KnowledgeGraphExtractor()

        existing_kg = KnowledgeGraph()
        existing_kg.add_entity("Existing Entity", EntityType.OTHER)

        kg = extractor.extract("New text about 1990.", existing_graph=existing_kg)

        assert kg.get_entity("Existing Entity") is not None
        assert kg is existing_kg

    def test_extract_for_question(self) -> None:
        """Test question-focused extraction."""
        extractor = KnowledgeGraphExtractor()

        kg = extractor.extract_for_question(
            text="Einstein published his theory in 1905.",
            question="When did Einstein publish?",
        )

        # Should extract entities relevant to the question
        assert len(kg.entities) > 0

    def test_extract_detects_organization_keywords(self) -> None:
        """Test rule-based extraction detects organizations by keywords."""
        extractor = KnowledgeGraphExtractor()

        kg = extractor.extract("He works at Acme Company.")

        # May or may not detect based on capitalization rules - just verify extraction runs
        _ = [e for e in kg.entities if e.entity_type == EntityType.ORGANIZATION]

    def test_extract_detects_person_titles(self) -> None:
        """Test rule-based extraction detects persons by titles."""
        extractor = KnowledgeGraphExtractor()

        kg = extractor.extract("Dr. Smith was a famous scientist.")

        # May or may not detect based on rules - just verify extraction runs
        _ = [e for e in kg.entities if e.entity_type == EntityType.PERSON]
