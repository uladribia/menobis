"""Tests for rustworkx interop."""

import numpy as np
import rustworkx as rx

from odme.data.frames import normalize_edges
from odme.interop.rustworkx import edges_to_rustworkx


def test_directed_graph() -> None:
    edges = normalize_edges(np.array([0, 1]), np.array([1, 2]), np.array([3, 4]))
    graph = edges_to_rustworkx(edges, directed=True)
    assert isinstance(graph, rx.PyDiGraph)
    assert graph.num_nodes() == 3
    assert graph.num_edges() == 2


def test_undirected_graph() -> None:
    edges = normalize_edges(np.array([0]), np.array([1]), np.array([3]))
    graph = edges_to_rustworkx(edges, directed=False)
    assert isinstance(graph, rx.PyGraph)
    assert graph.num_nodes() == 2
    assert graph.num_edges() == 1
