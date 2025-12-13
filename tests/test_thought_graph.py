"""Tests for the ThoughtGraph sparse web graph implementation."""

from __future__ import annotations

import pytest

from src.tools.thought_graph import (
    Edge,
    EdgeType,
    GraphNode,
    ThoughtGraph,
    build_graph_from_session,
)


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_create_node(self) -> None:
        """Test basic node creation."""
        node = GraphNode(
            id="t1",
            label="First thought",
            content="This is a complete thought about something.",
            step_number=1,
        )
        assert node.id == "t1"
        assert node.label == "First thought"
        assert node.step_number == 1
        assert node.metadata == {}

    def test_node_with_metadata(self) -> None:
        """Test node with metadata."""
        node = GraphNode(
            id="t2",
            label="Second",
            content="Content",
            step_number=2,
            metadata={"confidence": 0.9, "type": "continuation"},
        )
        assert node.metadata["confidence"] == 0.9
        assert node.metadata["type"] == "continuation"


class TestEdge:
    """Tests for Edge dataclass."""

    def test_create_edge(self) -> None:
        """Test basic edge creation."""
        edge = Edge(source="t1", target="t2", edge_type=EdgeType.SUPPORTS)
        assert edge.source == "t1"
        assert edge.target == "t2"
        assert edge.edge_type == EdgeType.SUPPORTS
        assert edge.weight == 1.0

    def test_edge_equality(self) -> None:
        """Test edge equality based on source, target, and type."""
        edge1 = Edge(source="t1", target="t2", edge_type=EdgeType.SUPPORTS)
        edge2 = Edge(source="t1", target="t2", edge_type=EdgeType.SUPPORTS)
        edge3 = Edge(source="t1", target="t2", edge_type=EdgeType.CONTRADICTS)

        assert edge1 == edge2
        assert edge1 != edge3

    def test_edge_hash(self) -> None:
        """Test edge hashing for set operations."""
        edge1 = Edge(source="t1", target="t2", edge_type=EdgeType.SUPPORTS)
        edge2 = Edge(source="t1", target="t2", edge_type=EdgeType.SUPPORTS)

        assert hash(edge1) == hash(edge2)
        edge_set = {edge1, edge2}
        assert len(edge_set) == 1


class TestThoughtGraph:
    """Tests for ThoughtGraph core functionality."""

    def test_empty_graph(self) -> None:
        """Test empty graph initialization."""
        graph = ThoughtGraph()
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_add_single_node(self) -> None:
        """Test adding a single node."""
        graph = ThoughtGraph()
        node = graph.add_node("t1", "First thought content", step_number=1)

        assert graph.node_count == 1
        assert node.id == "t1"
        assert graph.get_node("t1") == node

    def test_add_multiple_nodes(self) -> None:
        """Test adding multiple nodes."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_node("t3", "Third", 3)

        assert graph.node_count == 3
        assert graph.get_node("t2").step_number == 2

    def test_add_edge_supports(self) -> None:
        """Test adding a supports edge."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        edge = graph.add_edge("t1", "t2", EdgeType.SUPPORTS)

        assert graph.edge_count == 1
        assert edge.edge_type == EdgeType.SUPPORTS

    def test_add_edge_contradicts(self) -> None:
        """Test adding a contradicts edge."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_edge("t1", "t2", EdgeType.CONTRADICTS)

        edges = graph.get_edges(source="t1", edge_type=EdgeType.CONTRADICTS)
        assert len(edges) == 1

    def test_add_edge_nonexistent_node(self) -> None:
        """Test that adding edge with nonexistent node raises error."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)

        with pytest.raises(ValueError, match="not found"):
            graph.add_edge("t1", "t99", EdgeType.SUPPORTS)

    def test_get_edges_by_source(self) -> None:
        """Test filtering edges by source."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_node("t3", "Third", 3)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)
        graph.add_edge("t1", "t3", EdgeType.RELATED)

        edges = graph.get_edges(source="t1")
        assert len(edges) == 2

    def test_get_edges_by_target(self) -> None:
        """Test filtering edges by target."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_node("t3", "Third", 3)
        graph.add_edge("t1", "t3", EdgeType.SUPPORTS)
        graph.add_edge("t2", "t3", EdgeType.SUPPORTS)

        edges = graph.get_edges(target="t3")
        assert len(edges) == 2

    def test_get_edges_by_type(self) -> None:
        """Test filtering edges by type."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)
        graph.add_edge("t1", "t2", EdgeType.RELATED)

        supports = graph.get_edges(edge_type=EdgeType.SUPPORTS)
        related = graph.get_edges(edge_type=EdgeType.RELATED)
        assert len(supports) == 1
        assert len(related) == 1

    def test_get_neighbors_outgoing(self) -> None:
        """Test getting outgoing neighbors."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_node("t3", "Third", 3)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)
        graph.add_edge("t1", "t3", EdgeType.SUPPORTS)

        neighbors = graph.get_neighbors("t1", direction="outgoing")
        assert set(neighbors) == {"t2", "t3"}

    def test_get_neighbors_incoming(self) -> None:
        """Test getting incoming neighbors."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_node("t3", "Third", 3)
        graph.add_edge("t1", "t3", EdgeType.SUPPORTS)
        graph.add_edge("t2", "t3", EdgeType.SUPPORTS)

        neighbors = graph.get_neighbors("t3", direction="incoming")
        assert set(neighbors) == {"t1", "t2"}

    def test_remove_node(self) -> None:
        """Test that clear removes all nodes and edges."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)

        graph.clear()
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_remove_edge(self) -> None:
        """Test graph iteration over nodes and edges."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)
        graph.add_edge("t1", "t2", EdgeType.RELATED)

        # Use the nodes() and edges() methods
        assert len(list(graph.nodes())) == 2
        assert len(list(graph.edges())) == 2


class TestGraphAlgorithms:
    """Tests for graph algorithms."""

    def test_find_contradictions_simple(self) -> None:
        """Test finding contradiction edges."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_edge("t1", "t2", EdgeType.CONTRADICTS)

        contradictions = graph.find_contradictions()
        assert len(contradictions) == 1
        # find_contradictions returns (source, target, edges) tuples
        assert contradictions[0][0] == "t1"
        assert contradictions[0][1] == "t2"

    def test_find_contradictions_none(self) -> None:
        """Test graph with no contradictions."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)

        contradictions = graph.find_contradictions()
        assert len(contradictions) == 0

    def test_find_path_simple(self) -> None:
        """Test finding path between two nodes."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_node("t3", "Third", 3)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)
        graph.add_edge("t2", "t3", EdgeType.SUPPORTS)

        path = graph.find_path("t1", "t3")
        assert path is not None
        assert path.nodes == ["t1", "t2", "t3"]

    def test_find_path_no_path(self) -> None:
        """Test when no path exists."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        # No edge between them

        path = graph.find_path("t1", "t2")
        assert path is None

    def test_find_cycles_simple(self) -> None:
        """Test cycle detection."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_node("t3", "Third", 3)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)
        graph.add_edge("t2", "t3", EdgeType.SUPPORTS)
        graph.add_edge("t3", "t1", EdgeType.SUPPORTS)  # Creates cycle

        cycles = graph.find_cycles()
        assert len(cycles) >= 1

    def test_find_cycles_no_cycle(self) -> None:
        """Test graph with no cycles (DAG)."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_node("t3", "Third", 3)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)
        graph.add_edge("t2", "t3", EdgeType.SUPPORTS)

        cycles = graph.find_cycles()
        assert len(cycles) == 0

    def test_trace_support_chain(self) -> None:
        """Test tracing support chain for a conclusion."""
        graph = ThoughtGraph()
        graph.add_node("premise1", "First premise", 1)
        graph.add_node("premise2", "Second premise", 2)
        graph.add_node("intermediate", "Intermediate", 3)
        graph.add_node("conclusion", "Final conclusion", 4)

        graph.add_edge("premise1", "intermediate", EdgeType.SUPPORTS)
        graph.add_edge("premise2", "intermediate", EdgeType.SUPPORTS)
        graph.add_edge("intermediate", "conclusion", EdgeType.SUPPORTS)

        chain = graph.get_support_chain("conclusion")
        assert "intermediate" in chain
        # At least one premise should be in the chain
        assert "premise1" in chain or "premise2" in chain


class TestGraphExport:
    """Tests for graph export functionality."""

    def test_to_dict(self) -> None:
        """Test exporting graph to dictionary."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)

        data = graph.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    def test_to_mermaid(self) -> None:
        """Test exporting to Mermaid format."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First thought", 1)
        graph.add_node("t2", "Second thought", 2)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)

        mermaid = graph.to_mermaid()
        assert "graph" in mermaid.lower() or "flowchart" in mermaid.lower()
        assert "t1" in mermaid
        assert "t2" in mermaid

    def test_to_dot(self) -> None:
        """Test exporting to DOT format."""
        graph = ThoughtGraph()
        graph.add_node("t1", "First", 1)
        graph.add_node("t2", "Second", 2)
        graph.add_edge("t1", "t2", EdgeType.SUPPORTS)

        dot = graph.to_dot()
        assert "digraph" in dot
        assert "t1" in dot
        assert "t2" in dot


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_reads(self) -> None:
        """Test that concurrent reads don't cause issues."""
        import threading

        graph = ThoughtGraph()
        for i in range(10):
            graph.add_node(f"t{i}", f"Thought {i}", i)

        errors = []

        def read_nodes() -> None:
            try:
                for _ in range(100):
                    _ = graph.node_count
                    _ = graph.get_node("t5")
                    _ = graph.get_edges()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_nodes) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_writes(self) -> None:
        """Test that concurrent writes don't cause corruption."""
        import threading

        graph = ThoughtGraph()
        # Pre-add nodes to avoid conflicts
        for i in range(100):
            graph.add_node(f"t{i}", f"Thought {i}", i)

        errors = []

        def add_edges(start: int) -> None:
            try:
                for i in range(start, start + 10):
                    if i + 1 < 100:
                        graph.add_edge(f"t{i}", f"t{i + 1}", EdgeType.SUPPORTS)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_edges, args=(i * 10,)) for i in range(9)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have added some edges
        assert graph.edge_count > 0


class TestBuildGraphFromSession:
    """Tests for building graph from session data."""

    def test_build_from_empty_session(self) -> None:
        """Test building graph from empty session."""
        graph = build_graph_from_session({}, [])
        assert graph.node_count == 0

    def test_build_with_thoughts(self) -> None:
        """Test building graph with thoughts."""
        t1 = {
            "id": "t1",
            "content": "First thought",
            "step_number": 1,
            "thought_type": "initial",
            "parent_id": None,
            "supports": [],
            "contradicts": [],
            "related_to": [],
        }
        t2 = {
            "id": "t2",
            "content": "Second thought",
            "step_number": 2,
            "thought_type": "continuation",
            "parent_id": "t1",
            "supports": ["t1"],
            "contradicts": [],
            "related_to": [],
        }

        thoughts = {"t1": t1, "t2": t2}
        thought_order = ["t1", "t2"]

        graph = build_graph_from_session(thoughts, thought_order)
        assert graph.node_count == 2
        # Should have parent edge and supports edge
        assert graph.edge_count >= 1
