"""Data lineage graph and blast-radius computation.

Builds a lightweight DAG of ``node -> node`` edges carrying the columns that
flow between them. Given a set of changed columns, computes the downstream
blast radius (which stages/tables may be affected).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LineageEdge:
    source: str
    destination: str
    columns: list[str] = field(default_factory=list)


class LineageGraph:
    def __init__(self, edges: list[LineageEdge] | None = None) -> None:
        self._edges: list[LineageEdge] = []
        self._by_source: dict[str, list[LineageEdge]] = {}
        for e in edges or []:
            self.add_edge(e)

    def add_edge(self, edge: LineageEdge) -> None:
        self._edges.append(edge)
        self._by_source.setdefault(edge.source, []).append(edge)

    def edges_from(self, node: str) -> list[LineageEdge]:
        return self._by_source.get(node, [])

    @property
    def all_edges(self) -> list[LineageEdge]:
        return list(self._edges)

    @property
    def nodes(self) -> list[str]:
        nodes: set[str] = set()
        for e in self._edges:
            nodes.add(e.source)
            nodes.add(e.destination)
        return sorted(nodes)

    def downstream(self, node: str, *, visited: set[str] | None = None) -> list[str]:
        """All nodes reachable from ``node`` (transitive BFS/DFS)."""
        visited = visited or set()
        if node in visited:
            return []
        visited.add(node)
        result: list[str] = []
        for e in self.edges_from(node):
            result.append(e.destination)
            result.extend(self.downstream(e.destination, visited=visited))
        return result

    def blast_radius(self, changed_columns: list[str]) -> dict[str, dict[str, object]]:
        """Return affected downstream nodes + which changed columns reach them.

        A column change to ``source`` propagates to ``destination`` only if the
        changed column is carried by the connecting edge. The blast radius is the
        set of destinations reachable through those carried columns.
        """
        affected: dict[str, set[str]] = {}
        carriers: dict[str, list[str]] = {}

        def propagate(node: str, cols: set[str]) -> None:
            for e in self.edges_from(node):
                carried = set(e.columns) & cols
                if not carried:
                    continue
                carriers.setdefault(e.destination, []).extend(sorted(carried))
                affected.setdefault(e.destination, set()).update(sorted(carried))
                propagate(e.destination, carried)

        for col in changed_columns:
            for node in self.nodes:
                # start propagation from the node that owns this column
                if any(col in e.columns for e in self.edges_from(node)) or self._owns_column(
                    node, col
                ):
                    propagate(node, {col})

        return {
            node: {"changed_columns": sorted(cols), "carried": sorted(set(carriers.get(node, [])))}
            for node, cols in affected.items()
        }

    @staticmethod
    def _owns_column(node: str, col: str) -> bool:
        # Heuristic: a node "owns" a column if it's a table producing it upstream.
        # Exact ownership is resolved by the pipeline schemas; this helper keeps
        # blast-radius conservative by only flagging edges that carry the column.
        return False
