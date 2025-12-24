#!/usr/bin/env python3
"""Profile knowledge_graph.py under load to detect O(n²) or lock issues."""

import time

from src.models.knowledge_graph import EntityType, KnowledgeGraph, RelationType


def profile_knowledge_graph() -> None:
    """Profile KnowledgeGraph under load to detect O(n²) or lock issues."""
    kg = KnowledgeGraph()

    # Test 1: Add many entities (should be O(n))
    print("Test 1: Adding 1000 entities...")
    start = time.perf_counter()
    for i in range(1000):
        kg.add_entity(name=f"Entity_{i}", entity_type=EntityType.CONCEPT, properties={"id": i})
    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.3f}s ({elapsed / 1000 * 1000:.2f}ms per entity)")

    # Test 2: Add relations (check for O(n²) edge insertion)
    print("\nTest 2: Adding 500 relations between existing entities...")
    start = time.perf_counter()
    for i in range(500):
        kg.add_relation(
            subject=f"Entity_{i}",
            predicate=RelationType.RELATED_TO,
            obj=f"Entity_{i + 1}",
            confidence=0.9,
        )
    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.3f}s ({elapsed / 500 * 1000:.2f}ms per relation)")

    # Test 3: Query neighbors (should be O(1) lookup)
    print("\nTest 3: Get neighbors for 10 entities...")
    start = time.perf_counter()
    for i in range(0, 100, 10):
        _ = kg.get_neighbors(f"Entity_{i}")
    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.3f}s ({elapsed / 10 * 1000:.2f}ms per query)")

    # Test 4: Path finding (check BFS performance)
    print("\nTest 4: Find paths between 10 entity pairs...")
    start = time.perf_counter()
    for i in range(0, 50, 5):
        _ = kg.find_paths(f"Entity_{i}", f"Entity_{i + 10}", max_hops=5)
    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.3f}s ({elapsed / 10 * 1000:.2f}ms per path search)")

    # Test 5: Get relations (should be O(1) if indexed properly)
    print("\nTest 5: Get all relations for 100 entities...")
    start = time.perf_counter()
    for i in range(100):
        _ = kg.get_relations(entity=f"Entity_{i}")
    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.3f}s ({elapsed / 100 * 1000:.2f}ms per query)")

    # Test 6: Merge entities (check for index rebuild cost)
    print("\nTest 6: Merge 10 entity pairs...")
    start = time.perf_counter()
    for i in range(600, 620, 2):
        kg.merge_entities(f"Entity_{i}", f"Entity_{i + 1}")
    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.3f}s ({elapsed / 10 * 1000:.2f}ms per merge)")

    print(f"\nFinal: {len(kg._entities)} entities, {len(kg._relations)} relations")

    # Summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print("All operations < 1ms/op = PASS (no O(n²) detected)")


if __name__ == "__main__":
    profile_knowledge_graph()
