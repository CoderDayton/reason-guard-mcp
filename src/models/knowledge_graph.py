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
from typing import Any

from loguru import logger

from src.utils.errors import ReasonGuardException


class KnowledgeGraphException(ReasonGuardException):
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


def predicate_str(predicate: RelationType | str) -> str:
    """Convert a predicate (RelationType or str) to its string value.

    Type-safe helper that handles both RelationType enums and plain strings,
    avoiding linter errors from accessing `.value` on union types.

    Args:
        predicate: A RelationType enum or string predicate.

    Returns:
        The string value of the predicate.

    Example:
        >>> predicate_str(RelationType.DISCOVERED)
        'discovered'
        >>> predicate_str("custom_relation")
        'custom_relation'

    """
    return predicate.value if isinstance(predicate, RelationType) else predicate


def entity_type_str(entity_type: EntityType) -> str:
    """Convert an EntityType enum to its string value.

    Type-safe helper for extracting the string value from EntityType enums.

    Args:
        entity_type: An EntityType enum value.

    Returns:
        The string value of the entity type.

    Example:
        >>> entity_type_str(EntityType.PERSON)
        'PERSON'
        >>> entity_type_str(EntityType.ORGANIZATION)
        'ORG'

    """
    return entity_type.value


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


@dataclass(slots=True, eq=False)
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

    @property
    def type_value(self) -> str:
        """Get the string value of the entity type.

        Convenience property for consistent access to the type's string value.

        Returns:
            The string value of the entity type.

        """
        return self.entity_type.value

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
            "type": self.type_value,
            "aliases": list(self.aliases),
            "properties": self.properties,
            "mention_count": self.mention_count,
        }


@dataclass(slots=True, eq=False)
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

    @property
    def predicate_value(self) -> str:
        """Get the string value of the predicate.

        Convenience property that handles both RelationType enums and plain strings.

        Returns:
            The string value of the predicate.

        """
        return predicate_str(self.predicate)

    def __hash__(self) -> int:  # noqa: D105
        return hash((self.subject.name, str(self.predicate), self.object_entity.name))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "subject": self.subject.name,
            "predicate": self.predicate_value,
            "object": self.object_entity.name,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence[:200] if self.evidence else "",
        }

    def to_triple(self) -> tuple[str, str, str]:
        """Convert to (subject, predicate, object) triple."""
        return (self.subject.name, self.predicate_value, self.object_entity.name)


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

    Performance optimization (P0-PERF):
    - Uses inverted index for O(k) entity lookup where k = words in query
    - Previously was O(n × m) where n = entities, m = aliases per entity

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
        # P0-PERF: Inverted index for O(k) entity lookup
        self._entity_index: dict[str, set[str]] = defaultdict(set)  # token -> entity keys
        self._index_dirty: bool = False  # Track if index needs rebuild

    def _index_entity(self, key: str, entity: Entity) -> None:
        """Add entity to the inverted index.

        Indexes entity by:
        - Each word token in the entity key
        - Each word token in aliases

        Args:
            key: The entity key (lowercased name).
            entity: The entity to index.

        """
        # Index by key tokens
        for token in key.split():
            if len(token) >= 2:  # Skip single-char tokens
                self._entity_index[token].add(key)

        # Index by alias tokens
        for alias in entity.aliases:
            alias_lower = alias.lower()
            for token in alias_lower.split():
                if len(token) >= 2:
                    self._entity_index[token].add(key)

    def _rebuild_index(self) -> None:
        """Rebuild the entire entity index.

        Called lazily when index is dirty and a lookup is performed.
        """
        self._entity_index.clear()
        for key, entity in self._entities.items():
            self._index_entity(key, entity)
        self._index_dirty = False

    def add_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.OTHER,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> Entity:
        """Add or update an entity in the graph with automatic disambiguation.

        Performs entity disambiguation by checking:
        1. Exact name match (case-insensitive)
        2. Existing aliases match
        3. Name components match (e.g., "Fleming" matches "Alexander Fleming")

        Args:
            name: Entity name.
            entity_type: Type classification.
            aliases: Alternative names.
            properties: Additional properties.

        Returns:
            The Entity object (new, existing, or merged).

        """
        canonical = name.strip()
        key = canonical.lower()

        # Try exact match first
        if key in self._entities:
            entity = self._entities[key]
            entity.mention_count += 1
            if aliases:
                entity.aliases.update(aliases)
            if properties:
                entity.properties.update(properties)
            return entity

        # Try to find existing entity by alias or partial name match
        existing = self._find_matching_entity(canonical, entity_type)
        if existing:
            # Merge: add this name as an alias to the existing entity
            existing.aliases.add(canonical)
            existing.mention_count += 1
            if aliases:
                existing.aliases.update(aliases)
            if properties:
                existing.properties.update(properties)
            # Also index by the new name for future lookups
            self._entities[key] = existing
            return existing

        # Create new entity
        entity = Entity(
            name=canonical,
            entity_type=entity_type,
            aliases=set(aliases) if aliases else set(),
            properties=properties or {},
        )
        self._entities[key] = entity
        # P0-PERF: Index the new entity
        self._index_entity(key, entity)

        return entity

    def _find_matching_entity(
        self,
        name: str,
        entity_type: EntityType,
    ) -> Entity | None:
        """Find an existing entity that matches the given name.

        Matching strategies:
        1. Name appears in existing entity's aliases
        2. Name is a component of existing entity name (e.g., "Fleming" in "Alexander Fleming")
        3. Existing entity name is a component of this name

        Only matches entities of compatible types.

        Args:
            name: Name to match.
            entity_type: Type of the new entity.

        Returns:
            Matching entity if found, None otherwise.

        """
        name_lower = name.lower()
        name_parts = set(name_lower.split())

        for entity in self._entities.values():
            # Skip incompatible types (unless one is OTHER)
            if (
                entity.entity_type != entity_type
                and entity.entity_type != EntityType.OTHER
                and entity_type != EntityType.OTHER
            ):
                continue

            # Check if name is in aliases
            if any(alias.lower() == name_lower for alias in entity.aliases):
                return entity

            # Check partial name match for PERSON entities (most common case)
            if entity_type == EntityType.PERSON or entity.entity_type == EntityType.PERSON:
                entity_parts = set(entity.name.lower().split())

                # "Fleming" matches "Alexander Fleming" (subset match)
                # Only match if the shorter name is a single word (last name) with 4+ chars
                if len(name_parts) == 1 and len(name_lower) >= 4:
                    if name_lower in entity_parts:
                        return entity

                # "Alexander Fleming" matches existing "Fleming"
                if len(entity_parts) == 1:
                    single_part = next(iter(entity_parts))
                    if len(single_part) >= 4 and single_part in name_parts:
                        # Promote the existing single-name entity to the full name
                        entity.name = name
                        return entity

        return None

    def merge_entities(self, source_name: str, target_name: str) -> Entity | None:
        """Merge two entities, combining their aliases and relations.

        The source entity is merged into the target. All relations involving
        the source are updated to reference the target.

        Args:
            source_name: Name of entity to merge from (will be removed).
            target_name: Name of entity to merge into (will be kept).

        Returns:
            The merged target entity, or None if either entity doesn't exist.

        """
        source_key = source_name.lower().strip()
        target_key = target_name.lower().strip()

        source = self._entities.get(source_key)
        target = self._entities.get(target_key)

        if not source or not target or source == target:
            return None

        # Merge aliases
        target.aliases.add(source.name)
        target.aliases.update(source.aliases)
        target.mention_count += source.mention_count

        # Merge properties (target takes precedence)
        for k, v in source.properties.items():
            if k not in target.properties:
                target.properties[k] = v

        # Update relations: replace source with target
        # Note: This creates new Relation objects since they're immutable-ish
        relations_to_update: list[Relation] = []
        relations_to_update.extend(self._adjacency.get(source_key, []))
        relations_to_update.extend(self._reverse_adjacency.get(source_key, []))

        for rel in relations_to_update:
            self._relations.discard(rel)
            new_subject = target if rel.subject == source else rel.subject
            new_object = target if rel.object_entity == source else rel.object_entity

            # Add updated relation
            self.add_relation(
                new_subject,
                rel.predicate,
                new_object,
                confidence=rel.confidence,
                evidence=rel.evidence,
            )

        # Remove source from all indices
        del self._entities[source_key]
        self._adjacency.pop(source_key, None)
        self._reverse_adjacency.pop(source_key, None)

        # Index target by source name too
        self._entities[source_key] = target

        return target

    def get_entity(self, name: str) -> Entity | None:
        """Get entity by name.

        Args:
            name: Entity name to look up.

        Returns:
            Entity if found, None otherwise.

        """
        return self._entities.get(name.lower().strip())

    def has_entity(self, name: str) -> bool:
        """Check if entity exists in the graph.

        Args:
            name: Entity name to check.

        Returns:
            True if entity exists, False otherwise.

        """
        return name.lower().strip() in self._entities

    def get_supporting_facts(self, text: str) -> list[Relation]:
        """Find relations involving entities mentioned in text.

        P0-PERF: Optimized from O(n × m) to O(k) using inverted index.
        - n = number of entities
        - m = average aliases per entity
        - k = number of words in the query text

        Extracts entity names from the text and returns all relations
        where those entities appear as subject or object.

        Args:
            text: Text to extract entity mentions from.

        Returns:
            List of relations involving mentioned entities.

        """
        if not text:
            return []

        # P0-PERF: Use inverted index for O(k) lookup instead of O(n × m)
        # Rebuild index if dirty (lazy rebuild)
        if self._index_dirty:
            self._rebuild_index()

        text_lower = text.lower()
        mentioned_entities: set[str] = set()

        # Tokenize text and lookup each token in index - O(k) where k = words in text
        # Strip punctuation from tokens to match indexed entities
        import re

        text_tokens = set(re.findall(r"\b\w+\b", text_lower))
        for token in text_tokens:
            if token in self._entity_index:
                mentioned_entities.update(self._entity_index[token])

        # Verify matches by checking if entity key or name actually appears in text
        # This filters out false positives from shared tokens
        verified_entities: set[str] = set()
        for entity_key in mentioned_entities:
            entity = self._entities.get(entity_key)
            if not entity:
                continue
            # Check if the full entity key or name appears (not just a token)
            if entity_key in text_lower or entity.name.lower() in text_lower:
                verified_entities.add(entity_key)
            else:
                # Check aliases
                for alias in entity.aliases:
                    if alias.lower() in text_lower:
                        verified_entities.add(entity_key)
                        break

        # Collect all relations involving these entities
        supporting: list[Relation] = []
        seen: set[int] = set()  # Avoid duplicates via hash

        for entity_key in verified_entities:
            # Outgoing relations
            for rel in self._adjacency.get(entity_key, []):
                rel_hash = hash(rel)
                if rel_hash not in seen:
                    supporting.append(rel)
                    seen.add(rel_hash)
            # Incoming relations
            for rel in self._reverse_adjacency.get(entity_key, []):
                rel_hash = hash(rel)
                if rel_hash not in seen:
                    supporting.append(rel)
                    seen.add(rel_hash)

        # Sort by confidence (highest first)
        supporting.sort(key=lambda r: r.confidence, reverse=True)

        return supporting

    # Predicates where a subject should have at most one object (functional relations)
    _FUNCTIONAL_PREDICATES: set[str] = {
        RelationType.OCCURRED_ON.value,
        RelationType.BORN_IN.value,
        RelationType.DIED_IN.value,
        RelationType.DISCOVERED.value,
        RelationType.INVENTED.value,
        RelationType.FOUNDED.value,
        RelationType.HEADQUARTERED_IN.value,
    }

    def get_contradicting_facts(
        self,
        text: str,
        new_relations: list[Relation] | None = None,
    ) -> list[tuple[Relation, Relation, str]]:
        """Find facts in the KG that contradict claims in the text or new relations.

        Detects contradictions via:
        1. Functional uniqueness: predicates like 'discovered', 'born_in' should have
           unique objects per subject. If KG has "Fleming discovered penicillin" and
           new text claims "Chain discovered penicillin", that's a contradiction.
        2. Date/quantity mismatches: same subject+predicate with different date/number.

        Args:
            text: The text to check for contradictions against stored facts.
            new_relations: Optional list of newly extracted relations to check.

        Returns:
            List of (existing_relation, conflicting_relation, reason) tuples.
            If conflicting_relation is the same as existing, it means text contradicts it.

        """
        if not text and not new_relations:
            return []

        contradictions: list[tuple[Relation, Relation, str]] = []
        text_lower = text.lower() if text else ""

        # Build a map of subject+predicate -> relations for functional predicates
        functional_map: dict[tuple[str, str], list[Relation]] = defaultdict(list)
        for rel in self._relations:
            pred = predicate_str(rel.predicate)
            if pred in self._FUNCTIONAL_PREDICATES:
                key = (rel.subject.name.lower(), pred)
                functional_map[key].append(rel)

        # Check new_relations against existing ones
        if new_relations:
            for new_rel in new_relations:
                pred = predicate_str(new_rel.predicate)
                if pred in self._FUNCTIONAL_PREDICATES:
                    key = (new_rel.subject.name.lower(), pred)
                    for existing in functional_map.get(key, []):
                        # Same subject+predicate but different object = contradiction
                        if (
                            existing.object_entity.name.lower()
                            != new_rel.object_entity.name.lower()
                        ):
                            reason = (
                                f"Conflicting {pred}: KG says '{existing.object_entity.name}' "
                                f"but new claim says '{new_rel.object_entity.name}'"
                            )
                            contradictions.append((existing, new_rel, reason))

        # Check text for date contradictions with stored evidence
        # Look for years in text and compare against evidence text of relevant relations
        if text_lower:
            # Extract all years mentioned in new text
            new_years = set(re.findall(r"\b((?:19|20)\d{2})\b", text_lower))

            if new_years:
                # Find entities mentioned in both text and KG
                # Check both entity keys and their component words (for partial name matches)
                for entity_key, entity in self._entities.items():
                    # Check if entity name or any part of it appears in text
                    # e.g., "Alexander Fleming" matches text containing "Fleming"
                    entity_parts = entity_key.split()
                    entity_mentioned = (
                        entity_key in text_lower
                        or any(part in text_lower for part in entity_parts if len(part) > 3)
                        or any(alias.lower() in text_lower for alias in entity.aliases)
                    )

                    if entity_mentioned:
                        # Check relations involving this entity
                        for rel in self._adjacency.get(entity_key, []):
                            # Look for years in the stored evidence
                            if rel.evidence:
                                stored_years = set(
                                    re.findall(r"\b((?:19|20)\d{2})\b", rel.evidence.lower())
                                )
                                # If evidence has a year and text has a DIFFERENT year
                                conflicting = new_years - stored_years
                                if stored_years and conflicting:
                                    pred = predicate_str(rel.predicate)
                                    reason = (
                                        f"Date inconsistency for {rel.subject.name} {pred}: "
                                        f"evidence mentions {sorted(stored_years)}, but text says {sorted(conflicting)}"
                                    )
                                    contradictions.append((rel, rel, reason))

        return contradictions

    def resolve_contradiction(
        self,
        existing: Relation,
        new_claim: Relation | None,
        new_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Provide confidence-weighted resolution suggestion for a contradiction.

        Uses relation confidence scores and mention counts to suggest which
        claim is more likely correct.

        Args:
            existing: The existing relation in the KG.
            new_claim: The new conflicting relation (if extracted), or None for text-based conflicts.
            new_confidence: Confidence of the new claim (default 0.5 for text-based).

        Returns:
            Resolution suggestion with confidence analysis.

        """
        existing_conf = existing.confidence
        existing_mentions = existing.subject.mention_count + existing.object_entity.mention_count

        # Calculate weighted scores
        # More mentions = more corroboration = higher trust
        mention_boost = min(0.2, existing_mentions * 0.02)  # Up to +0.2 for 10+ mentions
        existing_score = existing_conf + mention_boost

        new_score = new_confidence
        if new_claim:
            new_mentions = new_claim.subject.mention_count + new_claim.object_entity.mention_count
            new_score += min(0.2, new_mentions * 0.02)

        # Determine suggestion
        if existing_score > new_score + 0.15:
            suggestion = "keep_existing"
            confidence_delta = existing_score - new_score
            explanation = (
                f"Existing fact has higher confidence ({existing_conf:.2f}) "
                f"and more corroboration ({existing_mentions} mentions)"
            )
        elif new_score > existing_score + 0.15:
            suggestion = "prefer_new"
            confidence_delta = new_score - existing_score
            explanation = (
                f"New claim appears more confident ({new_confidence:.2f}). "
                "Consider updating the knowledge graph."
            )
        else:
            suggestion = "needs_verification"
            confidence_delta = abs(existing_score - new_score)
            explanation = (
                f"Claims have similar confidence (existing: {existing_conf:.2f}, new: {new_confidence:.2f}). "
                "Recommend external verification."
            )

        return {
            "suggestion": suggestion,
            "existing_confidence": round(existing_conf, 3),
            "new_confidence": round(new_confidence, 3),
            "confidence_delta": round(confidence_delta, 3),
            "explanation": explanation,
            "existing_mentions": existing_mentions,
        }

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
                rel_pred = relation.predicate_value
                pred_str = predicate_str(predicate)
                if rel_pred != pred_str:
                    continue

            # Check object match
            if obj and relation.object_entity.name.lower() != obj.lower():
                continue

            results.append(relation)

        return results

    def query_by_triple(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> list[Relation]:
        """Query relations using string-only triple pattern.

        Convenience method that accepts all string arguments, handling
        predicate conversion internally. Useful when the caller doesn't
        have access to RelationType enum values.

        Args:
            subject: Subject entity name (or None for wildcard).
            predicate: Predicate string value (or None for wildcard).
            obj: Object entity name (or None for wildcard).

        Returns:
            List of matching relations.

        Example:
            >>> kg.query_by_triple("Einstein", "authored", None)
            [Relation(subject=Einstein, predicate=authored, object=Relativity)]
            >>> kg.query_by_triple(None, "discovered", "Penicillin")
            [Relation(subject=Fleming, predicate=discovered, object=Penicillin)]

        """
        return self.query(subject=subject, predicate=predicate, obj=obj)

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
        """Get all unique entities (handles merged entities with multiple keys)."""
        # Use dict to dedupe by object id since merged entities have multiple keys
        seen: dict[int, Entity] = {}
        for entity in self._entities.values():
            seen[id(entity)] = entity
        return list(seen.values())

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
            relation_types[relation.predicate_value] += 1

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
        # P0-PERF: Also clear the inverted index
        self._entity_index.clear()
        self._index_dirty = False


class KnowledgeGraphExtractor:
    """Extracts knowledge graph from text using rule-based patterns.

    Uses regex patterns and heuristics to identify entities, relations,
    and facts from unstructured text, building a structured knowledge graph.

    Example:
        >>> extractor = KnowledgeGraphExtractor()
        >>> kg = extractor.extract(
        ...     "Einstein published his theory of relativity in 1905."
        ... )
        >>> print(kg.stats())

    """

    # Patterns for entity extraction
    DATE_PATTERN = re.compile(
        r"\b((?:19|20)\d{2}|January|February|March|April|May|June|July|August|September|October|November|December)\b",
        re.IGNORECASE,
    )
    QUANTITY_PATTERN = re.compile(
        r"\b\d+(?:\.\d+)?(?:\s*(?:percent|%|million|billion|thousand|kg|km|miles|meters|years|days))?\b",
        re.IGNORECASE,
    )

    # Common titles indicating PERSON entities
    PERSON_TITLES = frozenset(
        {
            "dr",
            "dr.",
            "mr",
            "mr.",
            "mrs",
            "mrs.",
            "ms",
            "ms.",
            "miss",
            "prof",
            "prof.",
            "professor",
            "sir",
            "lord",
            "lady",
            "dame",
            "president",
            "ceo",
            "cfo",
            "cto",
            "king",
            "queen",
            "prince",
            "princess",
            "general",
            "colonel",
            "captain",
            "admiral",
            "senator",
            "governor",
            "saint",
            "st.",
            "rev",
            "rev.",
            "reverend",
            "father",
            "mother",
            "sister",
            "brother",
        }
    )

    # Common first names for PERSON detection (top ~200 English names)
    COMMON_FIRST_NAMES = frozenset(
        {
            # Male names
            "james",
            "john",
            "robert",
            "michael",
            "william",
            "david",
            "richard",
            "joseph",
            "thomas",
            "charles",
            "christopher",
            "daniel",
            "matthew",
            "anthony",
            "mark",
            "donald",
            "steven",
            "paul",
            "andrew",
            "joshua",
            "kenneth",
            "kevin",
            "brian",
            "george",
            "timothy",
            "ronald",
            "edward",
            "jason",
            "jeffrey",
            "ryan",
            "jacob",
            "gary",
            "nicholas",
            "eric",
            "jonathan",
            "stephen",
            "larry",
            "justin",
            "scott",
            "brandon",
            "benjamin",
            "samuel",
            "raymond",
            "gregory",
            "frank",
            "alexander",
            "patrick",
            "jack",
            "dennis",
            "jerry",
            "tyler",
            "aaron",
            "jose",
            "adam",
            "nathan",
            "henry",
            "douglas",
            "zachary",
            "peter",
            "kyle",
            "noah",
            "ethan",
            "jeremy",
            "walter",
            "christian",
            "keith",
            "roger",
            "terry",
            "harry",
            "ralph",
            "sean",
            "jesse",
            "roy",
            "louis",
            "billy",
            "bruce",
            "eugene",
            "carl",
            "arthur",
            "lawrence",
            "albert",
            "isaac",
            "leo",
            "max",
            "oscar",
            "oliver",
            "lucas",
            "liam",
            "mason",
            # Female names
            "mary",
            "patricia",
            "jennifer",
            "linda",
            "barbara",
            "elizabeth",
            "susan",
            "jessica",
            "sarah",
            "karen",
            "lisa",
            "nancy",
            "betty",
            "margaret",
            "sandra",
            "ashley",
            "kimberly",
            "emily",
            "donna",
            "michelle",
            "dorothy",
            "carol",
            "amanda",
            "melissa",
            "deborah",
            "stephanie",
            "rebecca",
            "sharon",
            "laura",
            "cynthia",
            "kathleen",
            "amy",
            "angela",
            "shirley",
            "anna",
            "brenda",
            "pamela",
            "emma",
            "nicole",
            "helen",
            "samantha",
            "katherine",
            "christine",
            "debra",
            "rachel",
            "carolyn",
            "janet",
            "catherine",
            "maria",
            "heather",
            "diane",
            "ruth",
            "julie",
            "olivia",
            "joyce",
            "virginia",
            "victoria",
            "kelly",
            "lauren",
            "christina",
            "joan",
            "evelyn",
            "judith",
            "megan",
            "andrea",
            "cheryl",
            "hannah",
            "jacqueline",
            "martha",
            "gloria",
            "teresa",
            "ann",
            "sara",
            "madison",
            "frances",
            "kathryn",
            "janice",
            "jean",
            "abigail",
            "alice",
            "judy",
            "sophia",
            "grace",
            "denise",
            "amber",
            "doris",
            "marilyn",
            "danielle",
            "beverly",
            "isabella",
            "theresa",
            # Historical/notable names
            "nikola",
            "marie",
            "leonardo",
            "galileo",
            "aristotle",
            "plato",
            "socrates",
            "napoleon",
            "winston",
            "abraham",
            "martin",
            "rosa",
            "mahatma",
            "nelson",
            "barack",
            "vladimir",
        }
    )

    # Surname patterns (common suffixes)
    SURNAME_SUFFIXES = frozenset(
        {
            "son",
            "sen",
            "ski",
            "sky",
            "stein",
            "berg",
            "man",
            "mann",
            "burg",
            "field",
            "ford",
            "wood",
            "worth",
            "ton",
            "ham",
            "ley",
            "land",
        }
    )

    def __init__(self) -> None:
        """Initialize extractor."""
        pass

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
            self._extract_with_rules(text, kg)
        except KnowledgeGraphException:
            raise
        except Exception as e:
            logger.error(f"Knowledge graph extraction failed: {e}")
            raise KnowledgeGraphException(f"Extraction failed: {e}") from e

        return kg

    def _extract_with_rules(self, text: str, kg: KnowledgeGraph) -> None:
        """Extract using rule-based patterns."""
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
            entity_type = self._classify_entity_type(name)
            kg.add_entity(name, entity_type)

        # Extract simple relations using pattern matching
        self._extract_relations_with_rules(text, kg)

    def _classify_entity_type(self, name: str) -> EntityType:
        """Classify entity type based on name patterns.

        Uses multiple heuristics:
        1. Title prefixes (Dr., Prof., Mr., etc.)
        2. Common first names
        3. Surname suffix patterns (-stein, -berg, -son, etc.)
        4. Organization keywords

        Args:
            name: The entity name to classify.

        Returns:
            Best-guess EntityType for the name.

        """
        words = name.split()
        if not words:
            return EntityType.OTHER

        name_lower = name.lower()
        first_word_lower = words[0].lower()

        # Check for title prefix -> PERSON
        if first_word_lower.rstrip(".") in self.PERSON_TITLES:
            return EntityType.PERSON

        # Check for organization keywords
        org_keywords = {
            "university",
            "college",
            "institute",
            "company",
            "corp",
            "corporation",
            "inc",
            "incorporated",
            "ltd",
            "limited",
            "llc",
            "foundation",
            "association",
            "organization",
            "agency",
            "department",
            "ministry",
            "committee",
            "council",
            "bank",
            "group",
            "holdings",
            "partners",
            "systems",
            "technologies",
        }
        if any(kw in name_lower for kw in org_keywords):
            return EntityType.ORGANIZATION

        # Check for location keywords
        loc_keywords = {
            "city",
            "town",
            "village",
            "county",
            "state",
            "province",
            "country",
            "island",
            "mount",
            "mountain",
            "river",
            "lake",
            "ocean",
            "sea",
            "bay",
            "street",
            "avenue",
            "road",
            "boulevard",
            "park",
            "square",
        }
        if any(kw in name_lower for kw in loc_keywords):
            return EntityType.LOCATION

        # Check if first word is a common first name -> PERSON
        if first_word_lower in self.COMMON_FIRST_NAMES:
            return EntityType.PERSON

        # Check for surname suffix patterns (if 2+ words) -> PERSON
        if len(words) >= 2:
            last_word = words[-1].lower()
            for suffix in self.SURNAME_SUFFIXES:
                if last_word.endswith(suffix) and len(last_word) > len(suffix):
                    return EntityType.PERSON

        # Check for "FirstName LastName" pattern with typical name length
        if len(words) == 2:
            # Both words capitalized, reasonable length for names
            if all(3 <= len(w) <= 15 for w in words):
                # Not obviously an organization or location
                return EntityType.PERSON

        return EntityType.OTHER

    # Adverbs and common words that shouldn't be part of entity names
    _ENTITY_STOP_WORDS: frozenset[str] = frozenset(
        {
            "also",
            "then",
            "later",
            "first",
            "next",
            "still",
            "even",
            "just",
            "only",
            "now",
            "once",
            "soon",
            "already",
            "finally",
            "actually",
            "probably",
            "certainly",
            "definitely",
            "recently",
            "previously",
        }
    )

    def _extract_relations_with_rules(self, text: str, kg: KnowledgeGraph) -> None:
        """Extract relations using pattern matching."""
        # Entity pattern: proper nouns (capitalized words)
        entity_pat = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"
        # Concept pattern for active voice objects: "the X", "X of Y"
        # Excludes auxiliary verbs (was, is, were, etc.)
        concept_pat = r"(?:the\s+)?([A-Za-z]+(?:\s+(?:of|and)\s+[A-Za-z]+)*)"
        # Passive subject pattern: noun phrase before "was" - single word or "X of Y"
        # Must NOT capture "was" - uses negative lookahead
        passive_subj_pat = r"(?:The\s+)?([A-Z][a-z]+(?:\s+of\s+[a-z]+)*)"

        # Common relation patterns
        patterns: list[tuple[str, RelationType | str]] = [
            # "X is a Y" / "X is the Y"
            (rf"({entity_pat})\s+is\s+(?:a|the)\s+([A-Za-z]+)", RelationType.IS_A),
            # "X founded Y" / "X created Y"
            (
                rf"({entity_pat})\s+(?:founded|created|established)\s+({entity_pat})",
                RelationType.FOUNDED,
            ),
            # "X located in Y" / "X is in Y"
            (
                rf"({entity_pat})\s+(?:is\s+)?(?:located\s+)?in\s+({entity_pat})",
                RelationType.LOCATED_IN,
            ),
            # "X works for Y" / "X employed by Y"
            (
                rf"({entity_pat})\s+(?:works\s+for|employed\s+by)\s+({entity_pat})",
                RelationType.WORKS_FOR,
            ),
            # "X born in Y"
            (
                rf"({entity_pat})\s+(?:was\s+)?born\s+in\s+({entity_pat}|\d{{4}})",
                RelationType.BORN_IN,
            ),
            # === Scientific/Historical Discovery Patterns ===
            # "X discovered Y" / "X discovered the Y" (Y can be lowercase concept)
            (
                rf"({entity_pat})\s+discovered\s+{concept_pat}",
                RelationType.DISCOVERED,
            ),
            # "Y was discovered by X" (passive voice - subject before "was")
            (
                rf"{passive_subj_pat}\s+was\s+discovered\s+by\s+({entity_pat})",
                "discovered_by",  # Will be reversed to DISCOVERED
            ),
            # "X invented Y" / "X invented the Y" (Y can be lowercase)
            (
                rf"({entity_pat})\s+invented\s+{concept_pat}",
                RelationType.INVENTED,
            ),
            # "Y was invented by X" (passive voice)
            (
                rf"{passive_subj_pat}\s+was\s+invented\s+by\s+({entity_pat})",
                "invented_by",  # Will be reversed to INVENTED
            ),
            # "Y was proposed/formulated by X" (passive voice for theories)
            (
                rf"{passive_subj_pat}\s+was\s+(?:proposed|formulated|introduced)\s+by\s+({entity_pat})",
                "proposed_by",  # Will be reversed to AUTHORED
            ),
            # "X published Y" / "X published his/her Y"
            (
                rf"({entity_pat})\s+published\s+(?:his\s+|her\s+)?{concept_pat}",
                RelationType.AUTHORED,
            ),
            # "X wrote Y" / "X authored Y"
            (
                rf"({entity_pat})\s+(?:wrote|authored)\s+{concept_pat}",
                RelationType.AUTHORED,
            ),
            # "X developed Y" / "X designed Y"
            (
                rf"({entity_pat})\s+(?:developed|designed|built)\s+{concept_pat}",
                RelationType.INVENTED,
            ),
            # "X proposed Y" / "X formulated Y" (for theories)
            (
                rf"({entity_pat})\s+(?:proposed|formulated|introduced)\s+{concept_pat}",
                RelationType.AUTHORED,
            ),
            # === Temporal Patterns ===
            # "X occurred in Y" / "X happened in Y"
            (
                rf"({entity_pat})\s+(?:occurred|happened|took\s+place)\s+in\s+(\d{{4}})",
                RelationType.OCCURRED_ON,
            ),
            # === Causal/Result Patterns ===
            # "X led to Y" / "X resulted in Y"
            (
                rf"{concept_pat}\s+(?:led\s+to|resulted\s+in|caused)\s+{concept_pat}",
                RelationType.RESULTS_IN,
            ),
            # "X was caused by Y"
            (
                rf"{concept_pat}\s+was\s+caused\s+by\s+{concept_pat}",
                RelationType.CAUSED_BY,
            ),
        ]

        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group(1).strip()
                obj = match.group(2).strip()

                if not subject or not obj:
                    continue

                # Clean up "the" prefix from objects
                if obj.lower().startswith("the "):
                    obj = obj[4:].strip()

                # Clean up "the" prefix from subjects too
                if subject.lower().startswith("the "):
                    subject = subject[4:].strip()

                # Remove trailing stop words from subject (e.g., "Fleming also" -> "Fleming")
                subject_words = subject.split()
                while subject_words and subject_words[-1].lower() in self._ENTITY_STOP_WORDS:
                    subject_words.pop()
                subject = " ".join(subject_words)

                # Remove leading stop words from object
                obj_words = obj.split()
                while obj_words and obj_words[0].lower() in self._ENTITY_STOP_WORDS:
                    obj_words.pop(0)
                obj = " ".join(obj_words)

                # Skip if captured auxiliary verbs (bad regex match)
                if obj.lower() in ("was", "is", "were", "are", "by", "the"):
                    continue
                if subject.lower() in ("was", "is", "were", "are", "by", "the"):
                    continue

                # Skip if subject or object ends with "was" (greedy match artifact)
                if subject.lower().endswith(" was") or obj.lower().endswith(" was"):
                    continue

                # Skip if subject or object is too short
                if len(subject) < 2 or len(obj) < 2:
                    continue

                # Handle passive voice patterns (reverse subject/object)
                if rel_type == "discovered_by":
                    subject, obj = obj, subject
                    rel_type = RelationType.DISCOVERED
                elif rel_type == "invented_by":
                    subject, obj = obj, subject
                    rel_type = RelationType.INVENTED
                elif rel_type == "proposed_by":
                    subject, obj = obj, subject
                    rel_type = RelationType.AUTHORED

                # Get surrounding sentence as evidence (capture dates/context)
                match_start = match.start()
                match_end = match.end()
                # Find sentence boundaries (. or start/end of text)
                sent_start = text.rfind(".", 0, match_start)
                sent_start = sent_start + 1 if sent_start != -1 else 0
                sent_end = text.find(".", match_end)
                sent_end = sent_end + 1 if sent_end != -1 else len(text)
                evidence = text[sent_start:sent_end].strip()

                kg.add_relation(subject, rel_type, obj, evidence=evidence)

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

        # Extract normally first
        self.extract(text, kg)

        # Extract entities mentioned in the question for relevance
        question_entities: set[str] = set()
        cap_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
        for match in cap_pattern.finditer(question):
            question_entities.add(match.group().lower())

        # Mark question-relevant entities with higher mention count
        for entity in kg.entities:
            if entity.name.lower() in question_entities:
                entity.mention_count += 5  # Boost relevance

        return kg
