"""Knowledge Graph extraction and reasoning module.

Extracts entities, relations, and facts from text to build a lightweight
knowledge graph for enhanced reasoning. Supports entity linking, relation
extraction, and graph-based inference.

This module enables multi-hop reasoning by providing structured knowledge
that can be traversed and queried.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.models.llm_client import LLMClientProtocol
from src.utils.errors import MatrixMindException

if TYPE_CHECKING:
    pass


class KnowledgeGraphException(MatrixMindException):
    """Raised during knowledge graph operations."""

    pass


class EntityType(str, Enum):
    """Supported entity types."""

    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    QUANTITY = "QUANTITY"
    OTHER = "OTHER"


class RelationType(str, Enum):
    """Common relation types for knowledge graphs."""

    # Temporal relations
    OCCURRED_ON = "occurred_on"
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"

    # Spatial relations
    LOCATED_IN = "located_in"
    PART_OF = "part_of"

    # Person relations
    BORN_IN = "born_in"
    DIED_IN = "died_in"
    WORKS_FOR = "works_for"
    AUTHORED = "authored"
    INVENTED = "invented"
    DISCOVERED = "discovered"

    # Organizational relations
    FOUNDED = "founded"
    HEADQUARTERED_IN = "headquartered_in"
    SUBSIDIARY_OF = "subsidiary_of"

    # General relations
    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    RELATED_TO = "related_to"
    CAUSED_BY = "caused_by"
    RESULTS_IN = "results_in"


@dataclass
class Entity:
    """Represents an entity in the knowledge graph.

    Attributes:
        name: Canonical name of the entity.
        entity_type: Type classification.
        aliases: Alternative names/mentions.
        properties: Additional properties as key-value pairs.
        mention_count: Number of times mentioned in source text.

    """

    name: str
    entity_type: EntityType
    aliases: set[str] = field(default_factory=set)
    properties: dict[str, Any] = field(default_factory=dict)
    mention_count: int = 1

    def __hash__(self) -> int:  # noqa: D105
        return hash(self.name.lower())

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, Entity):
            return False
        return self.name.lower() == other.name.lower()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.entity_type.value,
            "aliases": list(self.aliases),
            "properties": self.properties,
            "mention_count": self.mention_count,
        }


@dataclass
class Relation:
    """Represents a relation between two entities.

    Attributes:
        subject: Source entity.
        predicate: Relation type.
        object_entity: Target entity.
        confidence: Extraction confidence (0-1).
        evidence: Source text supporting this relation.

    """

    subject: Entity
    predicate: RelationType | str
    object_entity: Entity
    confidence: float = 1.0
    evidence: str = ""

    def __hash__(self) -> int:  # noqa: D105
        return hash((self.subject.name, str(self.predicate), self.object_entity.name))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "subject": self.subject.name,
            "predicate": str(
                self.predicate.value if isinstance(self.predicate, RelationType) else self.predicate
            ),
            "object": self.object_entity.name,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence[:200] if self.evidence else "",
        }

    def to_triple(self) -> tuple[str, str, str]:
        """Convert to (subject, predicate, object) triple."""
        pred = (
            self.predicate.value
            if isinstance(self.predicate, RelationType)
            else str(self.predicate)
        )
        return (self.subject.name, pred, self.object_entity.name)


@dataclass
class KnowledgeGraphStats:
    """Statistics about a knowledge graph."""

    num_entities: int
    num_relations: int
    entity_types: dict[str, int]
    relation_types: dict[str, int]
    avg_relations_per_entity: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "entity_types": self.entity_types,
            "relation_types": self.relation_types,
            "avg_relations_per_entity": round(self.avg_relations_per_entity, 2),
        }


class KnowledgeGraph:
    """Lightweight knowledge graph for reasoning.

    Stores entities and relations extracted from text, supporting
    graph traversal and query operations for multi-hop reasoning.

    Example:
        >>> kg = KnowledgeGraph()
        >>> einstein = kg.add_entity("Albert Einstein", EntityType.PERSON)
        >>> relativity = kg.add_entity("Theory of Relativity", EntityType.CONCEPT)
        >>> kg.add_relation(einstein, RelationType.AUTHORED, relativity)
        >>> paths = kg.find_paths(einstein, relativity, max_hops=2)

    """

    def __init__(self) -> None:
        """Initialize empty knowledge graph."""
        self._entities: dict[str, Entity] = {}  # name -> Entity
        self._relations: set[Relation] = set()
        self._adjacency: dict[str, list[Relation]] = defaultdict(
            list
        )  # entity -> outgoing relations
        self._reverse_adjacency: dict[str, list[Relation]] = defaultdict(
            list
        )  # entity -> incoming relations

    def add_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.OTHER,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> Entity:
        """Add or update an entity in the graph.

        Args:
            name: Entity name.
            entity_type: Type classification.
            aliases: Alternative names.
            properties: Additional properties.

        Returns:
            The Entity object (new or existing).

        """
        canonical = name.strip()
        key = canonical.lower()

        if key in self._entities:
            entity = self._entities[key]
            entity.mention_count += 1
            if aliases:
                entity.aliases.update(aliases)
            if properties:
                entity.properties.update(properties)
        else:
            entity = Entity(
                name=canonical,
                entity_type=entity_type,
                aliases=set(aliases) if aliases else set(),
                properties=properties or {},
            )
            self._entities[key] = entity

        return entity

    def get_entity(self, name: str) -> Entity | None:
        """Get entity by name.

        Args:
            name: Entity name to look up.

        Returns:
            Entity if found, None otherwise.

        """
        return self._entities.get(name.lower().strip())

    def add_relation(
        self,
        subject: Entity | str,
        predicate: RelationType | str,
        obj: Entity | str,
        confidence: float = 1.0,
        evidence: str = "",
    ) -> Relation:
        """Add a relation between two entities.

        Args:
            subject: Source entity or entity name.
            predicate: Relation type.
            obj: Target entity or entity name.
            confidence: Extraction confidence.
            evidence: Source text supporting this relation.

        Returns:
            The Relation object.

        Raises:
            KnowledgeGraphException: If entities don't exist.

        """
        # Resolve entity references
        if isinstance(subject, str):
            subject_entity = self.get_entity(subject)
            if not subject_entity:
                subject_entity = self.add_entity(subject)
        else:
            subject_entity = subject

        if isinstance(obj, str):
            obj_entity = self.get_entity(obj)
            if not obj_entity:
                obj_entity = self.add_entity(obj)
        else:
            obj_entity = obj

        relation = Relation(
            subject=subject_entity,
            predicate=predicate,
            object_entity=obj_entity,
            confidence=confidence,
            evidence=evidence,
        )

        self._relations.add(relation)
        self._adjacency[subject_entity.name.lower()].append(relation)
        self._reverse_adjacency[obj_entity.name.lower()].append(relation)

        return relation

    def get_relations(
        self,
        entity: Entity | str,
        direction: str = "outgoing",
    ) -> list[Relation]:
        """Get relations for an entity.

        Args:
            entity: Entity or entity name.
            direction: "outgoing", "incoming", or "both".

        Returns:
            List of relations.

        """
        key = entity.name.lower() if isinstance(entity, Entity) else entity.lower()

        if direction == "outgoing":
            return self._adjacency.get(key, [])
        elif direction == "incoming":
            return self._reverse_adjacency.get(key, [])
        else:
            return self._adjacency.get(key, []) + self._reverse_adjacency.get(key, [])

    def find_paths(
        self,
        start: Entity | str,
        end: Entity | str,
        max_hops: int = 3,
    ) -> list[list[Relation]]:
        """Find all paths between two entities.

        Uses BFS to find paths up to max_hops length.

        Args:
            start: Starting entity.
            end: Target entity.
            max_hops: Maximum path length.

        Returns:
            List of paths, where each path is a list of relations.

        """
        start_key = start.name.lower() if isinstance(start, Entity) else start.lower()
        end_key = end.name.lower() if isinstance(end, Entity) else end.lower()

        if start_key not in self._entities or end_key not in self._entities:
            return []

        paths: list[list[Relation]] = []
        queue: list[tuple[str, list[Relation]]] = [(start_key, [])]
        visited_states: set[tuple[str, ...]] = set()

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            # Create state signature to avoid cycles
            state = tuple(str(r.to_triple()) for r in path)
            if state in visited_states:
                continue
            visited_states.add(state)

            if current == end_key and path:
                paths.append(path)
                continue

            # Explore outgoing relations
            for relation in self._adjacency.get(current, []):
                next_entity = relation.object_entity.name.lower()
                if next_entity not in [r.subject.name.lower() for r in path]:  # Avoid cycles
                    queue.append((next_entity, path + [relation]))

        return paths

    def query(
        self,
        subject: str | None = None,
        predicate: RelationType | str | None = None,
        obj: str | None = None,
    ) -> list[Relation]:
        """Query relations matching pattern.

        Args:
            subject: Subject entity name (or None for wildcard).
            predicate: Relation type (or None for wildcard).
            obj: Object entity name (or None for wildcard).

        Returns:
            List of matching relations.

        """
        results = []

        for relation in self._relations:
            # Check subject match
            if subject and relation.subject.name.lower() != subject.lower():
                continue

            # Check predicate match
            if predicate:
                rel_pred = (
                    relation.predicate.value
                    if isinstance(relation.predicate, RelationType)
                    else str(relation.predicate)
                )
                pred_str = (
                    predicate.value if isinstance(predicate, RelationType) else str(predicate)
                )
                if rel_pred != pred_str:
                    continue

            # Check object match
            if obj and relation.object_entity.name.lower() != obj.lower():
                continue

            results.append(relation)

        return results

    def get_neighbors(
        self,
        entity: Entity | str,
        hops: int = 1,
    ) -> set[Entity]:
        """Get neighboring entities within n hops.

        Args:
            entity: Starting entity.
            hops: Number of hops to traverse.

        Returns:
            Set of reachable entities.

        """
        key = entity.name.lower() if isinstance(entity, Entity) else entity.lower()
        visited: set[str] = {key}
        current_level: set[str] = {key}

        for _ in range(hops):
            next_level: set[str] = set()
            for node in current_level:
                for relation in self._adjacency.get(node, []):
                    next_key = relation.object_entity.name.lower()
                    if next_key not in visited:
                        visited.add(next_key)
                        next_level.add(next_key)
                for relation in self._reverse_adjacency.get(node, []):
                    next_key = relation.subject.name.lower()
                    if next_key not in visited:
                        visited.add(next_key)
                        next_level.add(next_key)
            current_level = next_level

        return {self._entities[k] for k in visited if k in self._entities}

    @property
    def entities(self) -> list[Entity]:
        """Get all entities."""
        return list(self._entities.values())

    @property
    def relations(self) -> list[Relation]:
        """Get all relations."""
        return list(self._relations)

    def stats(self) -> KnowledgeGraphStats:
        """Get graph statistics."""
        entity_types: dict[str, int] = defaultdict(int)
        for entity in self._entities.values():
            entity_types[entity.entity_type.value] += 1

        relation_types: dict[str, int] = defaultdict(int)
        for relation in self._relations:
            pred = (
                relation.predicate.value
                if isinstance(relation.predicate, RelationType)
                else str(relation.predicate)
            )
            relation_types[pred] += 1

        num_entities = len(self._entities)
        avg_relations = len(self._relations) / num_entities if num_entities > 0 else 0

        return KnowledgeGraphStats(
            num_entities=num_entities,
            num_relations=len(self._relations),
            entity_types=dict(entity_types),
            relation_types=dict(relation_types),
            avg_relations_per_entity=avg_relations,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relations": [r.to_dict() for r in self._relations],
            "stats": self.stats().to_dict(),
        }

    def clear(self) -> None:
        """Clear all entities and relations."""
        self._entities.clear()
        self._relations.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()


class KnowledgeGraphExtractor:
    """Extracts knowledge graph from text using LLM.

    Uses an LLM to identify entities, relations, and facts from
    unstructured text, building a structured knowledge graph.

    Example:
        >>> extractor = KnowledgeGraphExtractor(llm_client)
        >>> kg = extractor.extract(
        ...     "Einstein published his theory of relativity in 1905."
        ... )
        >>> print(kg.stats())

    """

    # Patterns for simple entity extraction (fallback)
    DATE_PATTERN = re.compile(
        r"\b((?:19|20)\d{2}|January|February|March|April|May|June|July|August|September|October|November|December)\b",
        re.IGNORECASE,
    )
    QUANTITY_PATTERN = re.compile(
        r"\b\d+(?:\.\d+)?(?:\s*(?:percent|%|million|billion|thousand|kg|km|miles|meters|years|days))?\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        llm_client: LLMClientProtocol | None = None,
        use_llm: bool = True,
    ) -> None:
        """Initialize extractor.

        Args:
            llm_client: LLM client for extraction. If None, uses rule-based only.
            use_llm: Whether to use LLM for extraction.

        """
        self.llm = llm_client
        self.use_llm = use_llm and llm_client is not None

    def extract(
        self,
        text: str,
        existing_graph: KnowledgeGraph | None = None,
    ) -> KnowledgeGraph:
        """Extract knowledge graph from text.

        Args:
            text: Source text to extract from.
            existing_graph: Optional existing graph to extend.

        Returns:
            KnowledgeGraph with extracted entities and relations.

        Raises:
            KnowledgeGraphException: If extraction fails.

        """
        if not text or not text.strip():
            raise KnowledgeGraphException("Cannot extract from empty text")

        kg = existing_graph or KnowledgeGraph()

        try:
            if self.use_llm:
                self._extract_with_llm(text, kg)
            else:
                self._extract_with_rules(text, kg)

        except KnowledgeGraphException:
            raise
        except Exception as e:
            logger.error(f"Knowledge graph extraction failed: {e}")
            raise KnowledgeGraphException(f"Extraction failed: {e}") from e

        return kg

    def _extract_with_llm(self, text: str, kg: KnowledgeGraph) -> None:
        """Extract using LLM."""
        if not self.llm:
            return

        prompt = f"""Extract entities and relations from the following text.

Text:
{text[:2000]}

Output format (one per line):
ENTITY: <name> | <type: PERSON/ORG/LOC/DATE/EVENT/CONCEPT/OTHER>
RELATION: <subject> | <predicate> | <object>

Example:
ENTITY: Albert Einstein | PERSON
ENTITY: Theory of Relativity | CONCEPT
RELATION: Albert Einstein | authored | Theory of Relativity

Extract all meaningful entities and relations:"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,
            )

            # Parse response
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("ENTITY:"):
                    parts = line[7:].split("|")
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        type_str = parts[1].strip().upper()
                        try:
                            entity_type = EntityType(type_str)
                        except ValueError:
                            entity_type = EntityType.OTHER
                        kg.add_entity(name, entity_type)

                elif line.startswith("RELATION:"):
                    parts = line[9:].split("|")
                    if len(parts) >= 3:
                        subject = parts[0].strip()
                        predicate = parts[1].strip()
                        obj = parts[2].strip()

                        # Try to match known relation type
                        try:
                            rel_type: RelationType | str = RelationType(
                                predicate.lower().replace(" ", "_")
                            )
                        except ValueError:
                            rel_type = predicate  # Use as string

                        kg.add_relation(subject, rel_type, obj, evidence=text[:200])

        except Exception as e:
            logger.warning(f"LLM extraction failed, falling back to rules: {e}")
            self._extract_with_rules(text, kg)

    def _extract_with_rules(self, text: str, kg: KnowledgeGraph) -> None:
        """Extract using simple rule-based patterns."""
        # Extract dates
        for match in self.DATE_PATTERN.finditer(text):
            kg.add_entity(match.group(), EntityType.DATE)

        # Extract quantities
        for match in self.QUANTITY_PATTERN.finditer(text):
            kg.add_entity(match.group(), EntityType.QUANTITY)

        # Extract capitalized phrases as potential entities
        # Pattern: 2+ capitalized words in a row
        cap_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
        for match in cap_pattern.finditer(text):
            name = match.group()
            # Simple heuristic for type
            if any(title in name for title in ["Dr.", "Mr.", "Mrs.", "Prof."]):
                kg.add_entity(name, EntityType.PERSON)
            elif any(word in name.lower() for word in ["university", "company", "corp", "inc"]):
                kg.add_entity(name, EntityType.ORGANIZATION)
            else:
                kg.add_entity(name, EntityType.OTHER)

    def extract_for_question(
        self,
        text: str,
        question: str,
    ) -> KnowledgeGraph:
        """Extract knowledge graph focused on answering a question.

        Args:
            text: Source text.
            question: Question to focus extraction on.

        Returns:
            KnowledgeGraph focused on question-relevant entities.

        """
        kg = KnowledgeGraph()

        if not self.llm:
            return self.extract(text, kg)

        prompt = f"""Given the question, extract relevant entities and relations from the text.

Question: {question}

Text:
{text[:2000]}

Output format (one per line):
ENTITY: <name> | <type: PERSON/ORG/LOC/DATE/EVENT/CONCEPT/OTHER>
RELATION: <subject> | <predicate> | <object>

Focus on entities and relations that help answer the question:"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3,
            )

            # Parse response (same as _extract_with_llm)
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("ENTITY:"):
                    parts = line[7:].split("|")
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        type_str = parts[1].strip().upper()
                        try:
                            entity_type = EntityType(type_str)
                        except ValueError:
                            entity_type = EntityType.OTHER
                        kg.add_entity(name, entity_type)

                elif line.startswith("RELATION:"):
                    parts = line[9:].split("|")
                    if len(parts) >= 3:
                        subject = parts[0].strip()
                        predicate = parts[1].strip()
                        obj = parts[2].strip()
                        try:
                            rel_type_q: RelationType | str = RelationType(
                                predicate.lower().replace(" ", "_")
                            )
                        except ValueError:
                            rel_type_q = predicate
                        kg.add_relation(subject, rel_type_q, obj, evidence=text[:200])

        except Exception as e:
            logger.warning(f"Question-focused extraction failed: {e}")
            return self.extract(text, kg)

        return kg
