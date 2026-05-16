"""Interoperability with rustworkx."""

import rustworkx as rx

from odme.data.frames import EdgeTable


def edges_to_rustworkx(
    edges: EdgeTable,
    *,
    directed: bool = True,
) -> rx.PyDiGraph | rx.PyGraph:
    """Convert an EdgeTable to a rustworkx graph.

    Args:
        edges: Input edge table.
        directed: Whether to build a directed graph.

    Returns:
        A rustworkx graph with integer node payloads and integer edge weights.
    """
    graph: rx.PyDiGraph | rx.PyGraph
    graph = rx.PyDiGraph(multigraph=True) if directed else rx.PyGraph(multigraph=True)

    if len(edges) == 0:
        return graph

    max_node = int(max(edges.source.max(), edges.target.max()))
    graph.add_nodes_from(range(max_node + 1))
    for s, t, w in zip(edges.source, edges.target, edges.weight, strict=True):
        graph.add_edge(int(s), int(t), int(w))

    return graph


__all__ = ["edges_to_rustworkx"]
