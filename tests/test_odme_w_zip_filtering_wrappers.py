"""Python wrappers for W zero-inflated filtering families."""

from collections.abc import Callable

import numpy as np
import pytest

from odme.data.frames import EdgeTable
from odme.filtering import (
    filter_degree_events_geometric,
    filter_degree_events_negative_binomial,
    filter_strength_cost_geometric,
    filter_strength_cost_negative_binomial,
    filter_strength_degree_geometric,
    filter_strength_degree_negative_binomial,
    filter_strength_edges_geometric,
    filter_strength_edges_negative_binomial,
)
from odme.filtering_types import FilterResult
from odme.models import FitResult, StrengthCostFit, StrengthDegreeFit, StrengthEdgesFit


def _edges() -> EdgeTable:
    return EdgeTable(
        source=np.array([0, 0, 1], dtype=np.uint64),
        target=np.array([0, 1, 0], dtype=np.uint64),
        weight=np.array([1, 2, 1], dtype=np.uint64),
    )


def _strength_edges_fit() -> StrengthEdgesFit:
    return StrengthEdgesFit(
        node=np.array([0, 1], dtype=np.uint64),
        x=np.array([0.25, 0.35]),
        y=np.array([0.30, 0.40]),
        lam=1.2,
        converged=True,
        iterations=3,
        self_loops=True,
    )


def _strength_degree_fit() -> StrengthDegreeFit:
    return StrengthDegreeFit(
        node=np.array([0, 1], dtype=np.uint64),
        x=np.array([0.25, 0.35]),
        y=np.array([0.30, 0.40]),
        z=np.array([0.7, 0.8]),
        w=np.array([0.6, 0.9]),
        converged=True,
        iterations=3,
        self_loops=True,
    )


def _degree_events_fit() -> FitResult:
    return FitResult(
        node=np.array([0, 1], dtype=np.uint64),
        x=np.array([0.8, 1.1]),
        y=np.array([0.9, 1.0]),
        converged=True,
        iterations=3,
    )


def _strength_cost_fit() -> StrengthCostFit:
    return StrengthCostFit(
        node=np.array([0, 1], dtype=np.uint64),
        x=np.array([0.25, 0.35]),
        y=np.array([0.30, 0.40]),
        gamma=0.1,
        converged=True,
        iterations=3,
        self_loops=True,
    )


def _cost_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([0, 1], dtype=np.uint64),
        np.array([1, 0], dtype=np.uint64),
        np.array([1.0, 2.0], dtype=np.float64),
    )


def _assert_partitions_edges(result: FilterResult) -> None:
    total = (
        result.upper.edges.num_edges
        + result.lower.edges.num_edges
        + result.compatible.edges.num_edges
    )
    assert total == _edges().num_edges
    pvalues = result.compatible.upper_pvalue
    assert np.all((pvalues >= 0.0) & (pvalues <= 1.0))


def test_strength_edges_w_filters_partition_edges() -> None:
    fit = _strength_edges_fit()
    _assert_partitions_edges(filter_strength_edges_geometric(_edges(), fit))
    _assert_partitions_edges(
        filter_strength_edges_negative_binomial(_edges(), fit, layers=3)
    )


@pytest.mark.parametrize(
    "filter_function,layers",
    [
        (filter_strength_degree_geometric, None),
        (filter_strength_degree_negative_binomial, 3),
    ],
)
def test_strength_degree_w_filters_partition_edges(
    filter_function: Callable[..., FilterResult], layers: int | None
) -> None:
    kwargs = {} if layers is None else {"layers": layers}
    result = filter_function(_edges(), _strength_degree_fit(), **kwargs)
    _assert_partitions_edges(result)


def test_degree_events_w_filters_partition_edges() -> None:
    fit = _degree_events_fit()
    _assert_partitions_edges(
        filter_degree_events_geometric(_edges(), fit, positive_weight_rate=0.4)
    )
    _assert_partitions_edges(
        filter_degree_events_negative_binomial(
            _edges(), fit, positive_weight_rate=0.4, layers=3
        )
    )


def test_strength_cost_w_filters_partition_edges_and_absent() -> None:
    fit = _strength_cost_fit()
    cost_sources, cost_targets, cost_values = _cost_arrays()
    _assert_partitions_edges(
        filter_strength_cost_geometric(
            _edges(), fit, cost_sources, cost_targets, cost_values, detect_absent=True
        )
    )
    _assert_partitions_edges(
        filter_strength_cost_negative_binomial(
            _edges(), fit, cost_sources, cost_targets, cost_values, layers=3
        )
    )


def test_w_absent_edge_filtering_works() -> None:
    """W absent-edge filtering for edges/degree/degree-events now works."""
    result = filter_strength_edges_geometric(
        _edges(), _strength_edges_fit(), detect_absent=True
    )
    # absent_lower should have results (may be empty if no absent edges pass)
    assert result.absent_lower is not None
