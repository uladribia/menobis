"""Tests for clustering coefficients."""

import numpy as np
import pytest

from odme.analysis import clustering_coefficient, weighted_clustering_coefficient
from odme.data.frames import normalize_edges


def test_clustering_triangle() -> None:
    edges = normalize_edges(
        np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([1, 1, 1])
    )
    c = clustering_coefficient(edges)
    for v in c.values:
        assert v == pytest.approx(1.0)


def test_clustering_star() -> None:
    edges = normalize_edges(
        np.array([0, 0, 0]), np.array([1, 2, 3]), np.array([1, 1, 1])
    )
    c = clustering_coefficient(edges)
    assert c.values[0] == pytest.approx(0.0)


def test_weighted_clustering_triangle() -> None:
    edges = normalize_edges(
        np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([2, 3, 4])
    )
    wc = weighted_clustering_coefficient(edges)
    for v in wc.values:
        assert v > 0
