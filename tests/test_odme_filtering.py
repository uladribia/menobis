"""Tests for statistical filtering."""

import numpy as np

from odme.data.frames import EdgeTable
from odme.filtering import filter_fixed_strength_me, filter_strength_edges_me
from odme.models import fit_strength_edges_me


def test_fixed_strength_filter_flags_heavy_edge() -> None:
    edges = EdgeTable(
        source=np.array([0, 0, 1, 1], dtype=np.uint64),
        target=np.array([0, 1, 0, 1], dtype=np.uint64),
        weight=np.array([1, 40, 1, 1], dtype=np.uint64),
    )

    result = filter_fixed_strength_me(edges, alpha=0.95, tail="upper")

    assert len(result.upper.edges.source) >= 1
    assert (0, 1) in set(
        zip(result.upper.edges.source, result.upper.edges.target, strict=True)
    )
    assert result.tail == "upper"


def test_fixed_strength_absent_edges_are_separate() -> None:
    edges = EdgeTable(
        source=np.array([0, 1], dtype=np.uint64),
        target=np.array([0, 1], dtype=np.uint64),
        weight=np.array([20, 20], dtype=np.uint64),
    )

    result = filter_fixed_strength_me(
        edges,
        alpha=0.9,
        detect_absent=True,
        min_occupation=0.5,
        max_absent=10,
    )

    assert len(result.absent_lower.edges.source) > 0
    assert len(result.lower.edges.source) == 0


def test_strength_edges_filter_accepts_fitted_model() -> None:
    s_out = np.array([20.0, 20.0])
    s_in = np.array([20.0, 20.0])
    fit = fit_strength_edges_me(s_out, s_in, target_edges=3.0)
    edges = EdgeTable(
        source=np.array([0, 1], dtype=np.uint64),
        target=np.array([1, 0], dtype=np.uint64),
        weight=np.array([20, 1], dtype=np.uint64),
    )

    result = filter_strength_edges_me(edges, fit, alpha=0.05)

    assert result.upper.edges.num_edges + result.compatible.edges.num_edges >= 1
