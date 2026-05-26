"""Public filtering wrappers for MENoBiS null models."""

import math

import numpy as np
from numpy.typing import NDArray

import menobis._menobis as _menobis
from menobis.data.frames import EdgeTable, ProbabilityTable
from menobis.filtering.classify import (
    _classify,
    _classify_native,
    _lower_alpha,
)
from menobis.filtering.types import Correction, FilteredEdges, FilterResult, Tail
from menobis.models.types import (
    DegreeEventsFit,
    StrengthCostFit,
    StrengthDegreeFit,
    StrengthEdgesFit,
    StrengthFit,
)


def filter_strength_poisson(
    edges: EdgeTable,
    fit: "StrengthFit",
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
    self_loops: bool = True,
) -> FilterResult:
    """Filter edges against the independent Poisson fixed-strength ME null."""
    upper, lower, expected, occupation = _menobis.filter_strength_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_poisson(
            fit.x.tolist(),
            fit.y.tolist(),
            edges.source.tolist(),
            edges.target.tolist(),
            self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        np.asarray(occupation, dtype=np.float64),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_custom_poisson(
    edges: EdgeTable,
    rates: ProbabilityTable,
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against user-supplied independent Poisson rates.

    ``rates.probability`` is interpreted as the occupation number/rate
    ``T p_ij`` rather than as a normalized probability.
    """
    upper, lower, expected, occupation = _menobis.filter_custom_poisson(
        rates.source.tolist(),
        rates.target.tolist(),
        rates.probability.tolist(),
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_custom_poisson(
            rates.source.tolist(),
            rates.target.tolist(),
            rates.probability.tolist(),
            edges.source.tolist(),
            edges.target.tolist(),
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        np.asarray(occupation, dtype=np.float64),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_edges_poisson(
    edges: EdgeTable,
    fit: StrengthEdgesFit,
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a fitted strength-edges zero-inflated null model."""
    upper, lower, expected, occupation = _menobis.filter_strength_edges_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.lam,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_edges_poisson(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.lam,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        np.asarray(occupation, dtype=np.float64),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_cost_poisson(
    edges: EdgeTable,
    fit: StrengthCostFit,
    cost_sources: NDArray[np.uint64],
    cost_targets: NDArray[np.uint64],
    cost_values: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a fitted strength-cost Poisson null model."""
    upper, lower, expected, occupation = _menobis.filter_strength_cost_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.gamma,
        cost_sources.tolist(),
        cost_targets.tolist(),
        cost_values.tolist(),
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_cost_poisson(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.gamma,
            cost_sources.tolist(),
            cost_targets.tolist(),
            cost_values.tolist(),
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        np.asarray(occupation, dtype=np.float64),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_degree_poisson(
    edges: EdgeTable,
    fit: StrengthDegreeFit,
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a fitted strength-degree zero-inflated null model."""
    upper, lower, expected, occupation = _menobis.filter_strength_degree_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_degree_poisson(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.z.tolist(),
            fit.w.tolist(),
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        np.asarray(occupation, dtype=np.float64),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_degree_events_poisson(
    edges: EdgeTable,
    fit: DegreeEventsFit,
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a fitted degree-events zero-inflated null model."""
    upper, lower, expected, occupation = _menobis.filter_degree_events_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.q,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_degree_events_poisson(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.q,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        np.asarray(occupation, dtype=np.float64),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_geometric(
    edges: EdgeTable,
    fit: "StrengthFit",
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    self_loops: bool = True,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a geometric null model."""
    upper, lower, expected, occupation = _menobis.filter_strength_geometric(
        fit.x.tolist(),
        fit.y.tolist(),
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_geometric(
            fit.x.tolist(),
            fit.y.tolist(),
            edges.source.tolist(),
            edges.target.tolist(),
            self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper),
        np.asarray(lower),
        np.asarray(expected),
        np.asarray(occupation),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_binomial(
    edges: EdgeTable,
    fit: "StrengthFit",
    *,
    layers: int = 1,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    self_loops: bool = True,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a binomial(M) null model."""
    upper, lower, expected, occupation = _menobis.filter_strength_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        layers,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper),
        np.asarray(lower),
        np.asarray(expected),
        np.asarray(occupation),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_negative_binomial(
    edges: EdgeTable,
    fit: "StrengthFit",
    *,
    layers: int = 1,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    self_loops: bool = True,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a negative binomial(M) null model."""
    upper, lower, expected, occupation = _menobis.filter_strength_negative_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        layers,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_negative_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper),
        np.asarray(lower),
        np.asarray(expected),
        np.asarray(occupation),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_cost_binomial(
    edges: EdgeTable,
    fit: "StrengthCostFit",
    cost_sources: NDArray[np.uint64],
    cost_targets: NDArray[np.uint64],
    cost_values: NDArray[np.float64],
    *,
    layers: int = 1,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a strength-cost binomial null model."""
    upper, lower, expected, occupation = _menobis.filter_strength_cost_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.gamma,
        cost_sources.tolist(),
        cost_targets.tolist(),
        cost_values.tolist(),
        layers,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_cost_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.gamma,
            cost_sources.tolist(),
            cost_targets.tolist(),
            cost_values.tolist(),
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper),
        np.asarray(lower),
        np.asarray(expected),
        np.asarray(occupation),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_edges_binomial(
    edges: EdgeTable,
    fit: "StrengthEdgesFit",
    *,
    layers: int = 1,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a strength-edges binomial zero-inflated null model."""
    upper, lower, expected, occupation = _menobis.filter_strength_edges_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.lam,
        layers,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_edges_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.lam,
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper),
        np.asarray(lower),
        np.asarray(expected),
        np.asarray(occupation),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_degree_binomial(
    edges: EdgeTable,
    fit: "StrengthDegreeFit",
    *,
    layers: int = 1,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a strength-degree binomial zero-inflated null model."""
    upper, lower, expected, occupation = _menobis.filter_strength_degree_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        layers,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_degree_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.z.tolist(),
            fit.w.tolist(),
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper),
        np.asarray(lower),
        np.asarray(expected),
        np.asarray(occupation),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_degree_events_binomial(
    edges: EdgeTable,
    fit: DegreeEventsFit,
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a degree-events binomial zero-inflated null model."""
    layers = fit.layers or 1
    upper, lower, expected, occupation = _menobis.filter_degree_events_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.q,
        layers,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_degree_events_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.q,
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify(
        edges,
        np.asarray(upper),
        np.asarray(lower),
        np.asarray(expected),
        np.asarray(occupation),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_cost_geometric(
    edges: EdgeTable,
    fit: StrengthCostFit,
    cost_sources: NDArray[np.uint64],
    cost_targets: NDArray[np.uint64],
    cost_values: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a strength-cost geometric null model."""
    native_result = _menobis.filter_strength_cost_geometric(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.gamma,
        cost_sources.tolist(),
        cost_targets.tolist(),
        cost_values.tolist(),
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_cost_geometric(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.gamma,
            cost_sources.tolist(),
            cost_targets.tolist(),
            cost_values.tolist(),
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify_native(
        edges,
        native_result,
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_cost_negative_binomial(
    edges: EdgeTable,
    fit: StrengthCostFit,
    cost_sources: NDArray[np.uint64],
    cost_targets: NDArray[np.uint64],
    cost_values: NDArray[np.float64],
    *,
    layers: int = 1,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a strength-cost negative-binomial null model."""
    native_result = _menobis.filter_strength_cost_negative_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.gamma,
        cost_sources.tolist(),
        cost_targets.tolist(),
        cost_values.tolist(),
        layers,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_cost_negative_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.gamma,
            cost_sources.tolist(),
            cost_targets.tolist(),
            cost_values.tolist(),
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            _lower_alpha(alpha, tail),
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify_native(
        edges,
        native_result,
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_edges_geometric(
    edges: EdgeTable,
    fit: StrengthEdgesFit,
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a strength-edges geometric zero-inflated null model."""
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_edges_geometric(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.lam,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            alpha,
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify_native(
        edges,
        _menobis.filter_strength_edges_geometric(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.lam,
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        ),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_edges_negative_binomial(
    edges: EdgeTable,
    fit: StrengthEdgesFit,
    *,
    layers: int = 1,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a strength-edges negative-binomial null model."""
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_edges_negative_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.lam,
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            alpha,
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify_native(
        edges,
        _menobis.filter_strength_edges_negative_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.lam,
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        ),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_degree_geometric(
    edges: EdgeTable,
    fit: StrengthDegreeFit,
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a strength-degree geometric zero-inflated null model."""
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_degree_geometric(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.z.tolist(),
            fit.w.tolist(),
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            alpha,
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify_native(
        edges,
        _menobis.filter_strength_degree_geometric(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.z.tolist(),
            fit.w.tolist(),
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        ),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_strength_degree_negative_binomial(
    edges: EdgeTable,
    fit: StrengthDegreeFit,
    *,
    layers: int = 1,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a strength-degree negative-binomial null model."""
    absent = None
    if detect_absent:
        absent = _menobis.absent_strength_degree_negative_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.z.tolist(),
            fit.w.tolist(),
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            alpha,
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify_native(
        edges,
        _menobis.filter_strength_degree_negative_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.z.tolist(),
            fit.w.tolist(),
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        ),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_degree_events_geometric(
    edges: EdgeTable,
    fit: DegreeEventsFit,
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a degree-events geometric zero-inflated null model."""
    absent = None
    if detect_absent:
        absent = _menobis.absent_degree_events_geometric(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.q,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            alpha,
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify_native(
        edges,
        _menobis.filter_degree_events_geometric(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.q,
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        ),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def filter_degree_events_negative_binomial(
    edges: EdgeTable,
    fit: DegreeEventsFit,
    *,
    alpha: float = 0.05,
    tail: Tail = "two-sided",
    correction: Correction = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a degree-events negative-binomial null model."""
    layers = fit.layers or 1
    absent = None
    if detect_absent:
        absent = _menobis.absent_degree_events_negative_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.q,
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            fit.self_loops,
            alpha,
            min_occupation,
            min_expected,
            max_absent,
        )
    return _classify_native(
        edges,
        _menobis.filter_degree_events_negative_binomial(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.q,
            layers,
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        ),
        absent,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def _node_count(edges: EdgeTable) -> int:
    if len(edges) == 0:
        return 0
    return int(max(edges.source.max(), edges.target.max())) + 1


def _strengths(edges: EdgeTable, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(node_count, dtype=np.uint64)
    incoming = np.zeros(node_count, dtype=np.uint64)
    np.add.at(out, edges.source, edges.weight)
    np.add.at(incoming, edges.target, edges.weight)
    return out, incoming


def _solve_ztp_rate(mean: float) -> float:
    """Solve for the positive-edge Poisson rate given the mean."""
    if mean <= 1.0:
        return 0.0
    low, high = 0.0, max(mean, 1.0)
    while high / (1.0 - math.exp(-high)) < mean:
        high *= 2.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if mid / (1.0 - math.exp(-mid)) < mean:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


__all__ = [
    "FilterResult",
    "FilteredEdges",
    "filter_custom_poisson",
    "filter_degree_events_binomial",
    "filter_degree_events_geometric",
    "filter_degree_events_negative_binomial",
    "filter_degree_events_poisson",
    "filter_strength_binomial",
    "filter_strength_cost_binomial",
    "filter_strength_cost_geometric",
    "filter_strength_cost_negative_binomial",
    "filter_strength_cost_poisson",
    "filter_strength_degree_binomial",
    "filter_strength_degree_geometric",
    "filter_strength_degree_negative_binomial",
    "filter_strength_degree_poisson",
    "filter_strength_edges_binomial",
    "filter_strength_edges_geometric",
    "filter_strength_edges_negative_binomial",
    "filter_strength_edges_poisson",
    "filter_strength_geometric",
    "filter_strength_negative_binomial",
    "filter_strength_poisson",
]
