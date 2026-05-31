"""Tests for statistical filtering."""

import numpy as np

from menobis.data.frames import EdgeTable, ProbabilityTable
from menobis.filtering.models import (
    _filter_custom_poisson as filter_custom_poisson,
)
from menobis.filtering.models import (
    _filter_strength_edges_poisson as filter_strength_edges_poisson,
)
from menobis.models.fitting import (
    _fit_strength_edges_poisson as fit_strength_edges_poisson,
)
from menobis.routing import Constraint, ModelFamily, filter_model


def test_fixed_strength_filter_flags_heavy_edge() -> None:
    edges = EdgeTable(
        source=np.array([0, 0, 1, 1], dtype=np.uint64),
        target=np.array([0, 1, 0, 1], dtype=np.uint64),
        weight=np.array([1, 40, 1, 1], dtype=np.uint64),
    )

    result = filter_model(
        edges,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        alpha=0.95,
        tail="upper",
    )

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

    result = filter_model(
        edges,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        alpha=0.9,
        detect_absent=True,
        min_occupation=0.5,
        max_absent=10,
    )

    assert len(result.absent_lower.edges.source) > 0
    assert len(result.lower.edges.source) == 0


def test_custom_rates_filter_accepts_occupation_numbers() -> None:
    edges = EdgeTable(
        source=np.array([0], dtype=np.uint64),
        target=np.array([1], dtype=np.uint64),
        weight=np.array([10], dtype=np.uint64),
    )
    rates = ProbabilityTable(
        source=np.array([0, 1], dtype=np.uint64),
        target=np.array([1, 0], dtype=np.uint64),
        probability=np.array([1.0, 5.0], dtype=np.float64),
    )

    result = filter_custom_poisson(
        edges,
        rates,
        detect_absent=True,
        alpha=0.05,
        min_occupation=0.5,
    )

    assert result.upper.edges.num_edges == 1
    assert result.absent_lower.edges.num_edges == 1


def test_strength_edges_filter_accepts_fitted_model() -> None:
    s_out = np.array([20.0, 20.0])
    s_in = np.array([20.0, 20.0])
    fit = fit_strength_edges_poisson(s_out, s_in, target_edges=3.0)
    edges = EdgeTable(
        source=np.array([0, 1], dtype=np.uint64),
        target=np.array([1, 0], dtype=np.uint64),
        weight=np.array([20, 1], dtype=np.uint64),
    )

    result = filter_strength_edges_poisson(edges, fit, alpha=0.05)

    assert result.upper.edges.num_edges + result.compatible.edges.num_edges >= 1
