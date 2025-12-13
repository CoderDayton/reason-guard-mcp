"""Sparse web graph for thought relationships.

Tracks supports/contradicts/related_to relationships between thoughts
in a reasoning session. Enables cycle detection, contradiction finding,
path analysis, and graph export for visualization.

Design Principles:
    - Sparse storage: Only store actual edges, not full adjacency matrix
    - O(1) edge lookup via dict-based adjacency lists
    - Thread-safe operations for concurrent access
    - Export to multiple formats (dict, DOT, mermaid)
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EdgeType(str, Enum):
    """Types of relationships between thoughts."""

    SUPPORTS = "supports"  # This thought provides evidence for another
    CONTRADICTS = "contradicts"  # This thought conflicts with another
    RELATED = "related"  # General association without direction
    PARENT = "parent"  # Sequential parent in chain
    BRANCH = "branch"  # Branched from another thought
    REVISES = "revises"  # Revision of another thought


@dataclass
class Edge:
    """An edge in the thought graph."""

    source: str  # Source thought ID
    target: str  # Target thought ID
    edge_type: EdgeType
    weight: float = 1.0  # Edge strength (for weighted operations)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.edge_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return (
            self.source == other.source
            and self.target == other.target
            and self.edge_type == other.edge_type
        )


@dataclass
class GraphNode:
    """A node in the thought graph with metadata."""

    id: str
    label: str  # Short display label
    content: str  # Full thought content
    step_number: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Cycle:
    """A detected cycle in the graph."""

    nodes: list[str]  # Node IDs in cycle order
    edge_types: list[EdgeType]  # Edge types connecting nodes
    is_contradiction: bool = False  # True if cycle contains contradiction edge


@dataclass
class Path:
    """A path between two nodes."""

    nodes: list[str]  # Node IDs in path order
    edges: list[Edge]  # Edges connecting nodes
    total_weight: float = 0.0


class ThoughtGraph:
    """Sparse web graph for tracking thought relationships.

    Efficiently stores and queries relationships between thoughts
    using adjacency lists. Supports multiple edge types and provides
    algorithms for cycle detection, path finding, and contradiction analysis.

    Thread-safe for concurrent read/write operations.
    """

    def __init__(self) -> None:
        """Initialize empty thought graph."""
        self._nodes: dict[str, GraphNode] = {}
        self._outgoing: dict[str, dict[str, list[Edge]]] = defaultdict(
            lambda: defaultdict(list)
        )  # source -> target -> edges
        self._incoming: dict[str, dict[str, list[Edge]]] = defaultdict(
            lambda: defaultdict(list)
        )  # target -> source -> edges
        self._lock = threading.RLock()

    def add_node(
        self,
        node_id: str,
        content: str,
        step_number: int,
        label: str | None = None,
        **metadata: Any,
    ) -> GraphNode:
        """Add a node to the graph.

        Args:
            node_id: Unique identifier for the node.
            content: Full thought content.
            step_number: Position in reasoning chain.
            label: Optional short label (defaults to truncated content).
            **metadata: Additional metadata for the node.

        Returns:
            The created or updated GraphNode.

        """
        with self._lock:
            if label is None:
                label = content[:50] + "..." if len(content) > 50 else content

            node = GraphNode(
                id=node_id,
                label=label,
                content=content,
                step_number=step_number,
                metadata=metadata,
            )
            self._nodes[node_id] = node
            return node

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        **metadata: Any,
    ) -> Edge:
        """Add an edge between two nodes.

        Args:
            source: Source node ID.
            target: Target node ID.
            edge_type: Type of relationship.
            weight: Edge weight for weighted operations.
            **metadata: Additional metadata for the edge.

        Returns:
            The created Edge.

        Raises:
            ValueError: If source or target node doesn't exist.

        """
        with self._lock:
            if source not in self._nodes:
                raise ValueError(f"Source node '{source}' not found in graph")
            if target not in self._nodes:
                raise ValueError(f"Target node '{target}' not found in graph")

            edge = Edge(
                source=source,
                target=target,
                edge_type=edge_type,
                weight=weight,
                metadata=metadata,
            )

            # Avoid duplicate edges
            existing = self._outgoing[source][target]
            if edge not in existing:
                self._outgoing[source][target].append(edge)
                self._incoming[target][source].append(edge)

            return edge

    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_edges(
        self,
        source: str | None = None,
        target: str | None = None,
        edge_type: EdgeType | None = None,
    ) -> list[Edge]:
        """Get edges matching the given criteria.

        Args:
            source: Filter by source node ID.
            target: Filter by target node ID.
            edge_type: Filter by edge type.

        Returns:
            List of matching edges.

        """
        with self._lock:
            edges: list[Edge] = []

            if source is not None and target is not None:
                # Direct lookup
                edges = list(self._outgoing[source].get(target, []))
            elif source is not None:
                # All edges from source
                for target_edges in self._outgoing[source].values():
                    edges.extend(target_edges)
            elif target is not None:
                # All edges to target
                for source_edges in self._incoming[target].values():
                    edges.extend(source_edges)
            else:
                # All edges
                for source_dict in self._outgoing.values():
                    for target_edges in source_dict.values():
                        edges.extend(target_edges)

            if edge_type is not None:
                edges = [e for e in edges if e.edge_type == edge_type]

            return edges

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        edge_type: EdgeType | None = None,
    ) -> list[str]:
        """Get neighboring nodes.

        Args:
            node_id: Node to find neighbors for.
            direction: "outgoing", "incoming", or "both".
            edge_type: Optional filter by edge type.

        Returns:
            List of neighbor node IDs.

        """
        with self._lock:
            neighbors: set[str] = set()

            if direction in ("outgoing", "both"):
                for target, edges in self._outgoing[node_id].items():
                    if edge_type is None or any(e.edge_type == edge_type for e in edges):
                        neighbors.add(target)

            if direction in ("incoming", "both"):
                for source, edges in self._incoming[node_id].items():
                    if edge_type is None or any(e.edge_type == edge_type for e in edges):
                        neighbors.add(source)

            return list(neighbors)

    def find_cycles(self, max_length: int = 10) -> list[Cycle]:
        """Find all cycles in the graph up to max_length.

        Uses DFS-based cycle detection. Marks cycles as contradictions
        if they contain a CONTRADICTS edge.

        Args:
            max_length: Maximum cycle length to detect.

        Returns:
            List of detected cycles.

        """
        with self._lock:
            cycles: list[Cycle] = []
            visited: set[str] = set()

            def dfs(
                start: str,
                current: str,
                path: list[str],
                edge_path: list[Edge],
                depth: int,
            ) -> None:
                if depth > max_length:
                    return

                for target, edges in self._outgoing[current].items():
                    for edge in edges:
                        if target == start and len(path) > 1:
                            # Found cycle back to start
                            cycle_nodes = path + [target]
                            cycle_edges = edge_path + [edge]
                            has_contradiction = any(
                                e.edge_type == EdgeType.CONTRADICTS for e in cycle_edges
                            )
                            cycles.append(
                                Cycle(
                                    nodes=cycle_nodes,
                                    edge_types=[e.edge_type for e in cycle_edges],
                                    is_contradiction=has_contradiction,
                                )
                            )
                        elif target not in path and target not in visited:
                            dfs(
                                start,
                                target,
                                path + [target],
                                edge_path + [edge],
                                depth + 1,
                            )

            for node_id in self._nodes:
                if node_id not in visited:
                    dfs(node_id, node_id, [node_id], [], 0)
                    visited.add(node_id)

            return cycles

    def find_contradictions(self) -> list[tuple[str, str, list[Edge]]]:
        """Find all contradiction edges and their supporting paths.

        Returns:
            List of (source, target, edges) tuples for contradictions.

        """
        with self._lock:
            contradictions: list[tuple[str, str, list[Edge]]] = []

            for source_dict in self._outgoing.values():
                for target, edges in source_dict.items():
                    contradiction_edges = [e for e in edges if e.edge_type == EdgeType.CONTRADICTS]
                    if contradiction_edges:
                        contradictions.append(
                            (contradiction_edges[0].source, target, contradiction_edges)
                        )

            return contradictions

    def find_path(
        self,
        source: str,
        target: str,
        edge_types: list[EdgeType] | None = None,
    ) -> Path | None:
        """Find shortest path between two nodes using BFS.

        Args:
            source: Start node ID.
            target: End node ID.
            edge_types: Optional filter to only use certain edge types.

        Returns:
            Path object if found, None otherwise.

        """
        with self._lock:
            if source not in self._nodes or target not in self._nodes:
                return None

            if source == target:
                return Path(nodes=[source], edges=[], total_weight=0.0)

            # BFS
            queue: deque[tuple[str, list[str], list[Edge]]] = deque()
            queue.append((source, [source], []))
            visited: set[str] = {source}

            while queue:
                current, path, edge_path = queue.popleft()

                for next_node, edges in self._outgoing[current].items():
                    valid_edges = edges
                    if edge_types is not None:
                        valid_edges = [e for e in edges if e.edge_type in edge_types]

                    if not valid_edges:
                        continue

                    if next_node == target:
                        final_path = path + [next_node]
                        final_edges = edge_path + [valid_edges[0]]
                        return Path(
                            nodes=final_path,
                            edges=final_edges,
                            total_weight=sum(e.weight for e in final_edges),
                        )

                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, path + [next_node], edge_path + [valid_edges[0]]))

            return None

    def get_support_chain(self, node_id: str) -> list[str]:
        """Get the chain of nodes that support this node.

        Traces back through SUPPORTS edges to find all evidence.

        Args:
            node_id: Node to find support chain for.

        Returns:
            List of node IDs that support this node (ordered by distance).

        """
        with self._lock:
            support_chain: list[str] = []
            visited: set[str] = {node_id}
            queue: deque[str] = deque([node_id])

            while queue:
                current = queue.popleft()
                # Find nodes that support current
                for source, edges in self._incoming[current].items():
                    if any(e.edge_type == EdgeType.SUPPORTS for e in edges):
                        if source not in visited:
                            visited.add(source)
                            support_chain.append(source)
                            queue.append(source)

            return support_chain

    def get_contradiction_chain(self, node_id: str) -> list[str]:
        """Get all nodes that contradict this node (directly or transitively).

        Args:
            node_id: Node to find contradictions for.

        Returns:
            List of node IDs that contradict this node.

        """
        with self._lock:
            contradictions: list[str] = []

            # Direct contradictions
            for source, edges in self._incoming[node_id].items():
                if any(e.edge_type == EdgeType.CONTRADICTS for e in edges):
                    contradictions.append(source)

            for target, edges in self._outgoing[node_id].items():
                if any(e.edge_type == EdgeType.CONTRADICTS for e in edges):
                    contradictions.append(target)

            return list(set(contradictions))

    @property
    def node_count(self) -> int:
        """Get number of nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Get number of edges in the graph."""
        count = 0
        for source_dict in self._outgoing.values():
            for edges in source_dict.values():
                count += len(edges)
        return count

    def nodes(self) -> Iterator[GraphNode]:
        """Iterate over all nodes."""
        return iter(self._nodes.values())

    def edges(self) -> Iterator[Edge]:
        """Iterate over all edges."""
        for source_dict in self._outgoing.values():
            for edges in source_dict.values():
                yield from edges

    def to_dict(self) -> dict[str, Any]:
        """Export graph to dictionary format.

        Returns:
            Dictionary with nodes and edges lists.

        """
        with self._lock:
            return {
                "nodes": [
                    {
                        "id": node.id,
                        "label": node.label,
                        "step": node.step_number,
                        "metadata": node.metadata,
                    }
                    for node in self._nodes.values()
                ],
                "edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "type": edge.edge_type.value,
                        "weight": edge.weight,
                    }
                    for edge in self.edges()
                ],
                "stats": {
                    "node_count": self.node_count,
                    "edge_count": self.edge_count,
                    "contradiction_count": len(self.find_contradictions()),
                },
            }

    def to_dot(self, title: str = "Thought Graph") -> str:
        """Export graph to DOT format for Graphviz visualization.

        Args:
            title: Graph title.

        Returns:
            DOT format string.

        """
        with self._lock:
            lines = [f'digraph "{title}" {{', "  rankdir=TB;", "  node [shape=box];"]

            # Add nodes
            for node in self._nodes.values():
                label = node.label.replace('"', '\\"')
                lines.append(f'  "{node.id}" [label="{label}"];')

            # Add edges with style based on type
            edge_styles = {
                EdgeType.SUPPORTS: 'color="green"',
                EdgeType.CONTRADICTS: 'color="red" style="dashed"',
                EdgeType.RELATED: 'color="gray" style="dotted"',
                EdgeType.PARENT: 'color="blue"',
                EdgeType.BRANCH: 'color="purple" style="dashed"',
                EdgeType.REVISES: 'color="orange"',
            }

            for edge in self.edges():
                style = edge_styles.get(edge.edge_type, "")
                lines.append(
                    f'  "{edge.source}" -> "{edge.target}" [{style} label="{edge.edge_type.value}"];'
                )

            lines.append("}")
            return "\n".join(lines)

    def to_mermaid(self, title: str = "Thought Graph") -> str:
        """Export graph to Mermaid format for documentation.

        Args:
            title: Graph title.

        Returns:
            Mermaid format string.

        """
        with self._lock:
            lines = ["```mermaid", "graph TD", f"    %% {title}"]

            # Add nodes
            for node in self._nodes.values():
                label = node.label.replace('"', "'")
                lines.append(f'    {node.id}["{label}"]')

            # Add edges
            arrow_styles = {
                EdgeType.SUPPORTS: "-->|supports|",
                EdgeType.CONTRADICTS: "-.->|contradicts|",
                EdgeType.RELATED: "-.-|related|",
                EdgeType.PARENT: "-->",
                EdgeType.BRANCH: "-.->|branch|",
                EdgeType.REVISES: "-->|revises|",
            }

            for edge in self.edges():
                arrow = arrow_styles.get(edge.edge_type, "-->")
                lines.append(f"    {edge.source} {arrow} {edge.target}")

            lines.append("```")
            return "\n".join(lines)

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        with self._lock:
            self._nodes.clear()
            self._outgoing.clear()
            self._incoming.clear()


def build_graph_from_session(
    thoughts: dict[str, Any],
    thought_order: list[str],
) -> ThoughtGraph:
    """Build a ThoughtGraph from a reasoning session's thoughts.

    Args:
        thoughts: Dictionary mapping thought IDs to Thought objects/dicts.
        thought_order: Ordered list of thought IDs.

    Returns:
        Populated ThoughtGraph.

    """
    graph = ThoughtGraph()

    # Add all nodes first
    for thought_id in thought_order:
        thought = thoughts[thought_id]
        if hasattr(thought, "content"):
            # Thought object
            graph.add_node(
                node_id=thought_id,
                content=thought.content,
                step_number=thought.step_number,
                thought_type=thought.thought_type.value
                if hasattr(thought.thought_type, "value")
                else str(thought.thought_type),
            )
        else:
            # Dict
            graph.add_node(
                node_id=thought_id,
                content=thought.get("content", ""),
                step_number=thought.get("step_number", 0),
            )

    # Add edges
    prev_id = None
    for thought_id in thought_order:
        thought = thoughts[thought_id]

        # Parent edge (sequential)
        if prev_id is not None:
            parent_id = getattr(thought, "parent_id", None) or (
                thought.get("parent_id") if isinstance(thought, dict) else None
            )
            if parent_id and parent_id in graph._nodes:
                graph.add_edge(parent_id, thought_id, EdgeType.PARENT)
            elif prev_id in graph._nodes:
                graph.add_edge(prev_id, thought_id, EdgeType.PARENT)

        # Supports edges
        supports = (
            getattr(thought, "supports", [])
            if hasattr(thought, "supports")
            else thought.get("supports", [])
        )
        for target_id in supports:
            if target_id in graph._nodes:
                graph.add_edge(thought_id, target_id, EdgeType.SUPPORTS)

        # Contradicts edges
        contradicts = (
            getattr(thought, "contradicts", [])
            if hasattr(thought, "contradicts")
            else thought.get("contradicts", [])
        )
        for target_id in contradicts:
            if target_id in graph._nodes:
                graph.add_edge(thought_id, target_id, EdgeType.CONTRADICTS)

        # Related edges
        related = (
            getattr(thought, "related_to", [])
            if hasattr(thought, "related_to")
            else thought.get("related_to", [])
        )
        for target_id in related:
            if target_id in graph._nodes:
                graph.add_edge(thought_id, target_id, EdgeType.RELATED)

        # Branch edge
        branch_id = (
            getattr(thought, "branch_id", None)
            if hasattr(thought, "branch_id")
            else thought.get("branch_id")
        )
        if branch_id:
            # Find the source of the branch
            branch_from = (
                getattr(thought, "parent_id", None)
                if hasattr(thought, "parent_id")
                else thought.get("parent_id")
            )
            if branch_from and branch_from in graph._nodes:
                graph.add_edge(branch_from, thought_id, EdgeType.BRANCH)

        # Revises edge
        revises_id = (
            getattr(thought, "revises_id", None)
            if hasattr(thought, "revises_id")
            else thought.get("revises_id")
        )
        if revises_id and revises_id in graph._nodes:
            graph.add_edge(thought_id, revises_id, EdgeType.REVISES)

        prev_id = thought_id

    return graph
