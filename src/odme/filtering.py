"""Statistical filtering for ODME null models."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.data.frames import EdgeTable, ProbabilityTable
from odme.models.fitting import (
    StrengthCostFit,
    StrengthDegreeFit,
    StrengthEdgesFit,
    fit_strength_poisson,
)

Tail = Literal["upper", "lower", "two-sided"]
Correction = Literal["none", "bonferroni", "fdr"]
Tail = Literal["upper", "lower", "two-sided"]
Correction = Literal["none", "bonferroni", "fdr"]


@dataclass(frozen=True)
class FilteredEdges:
    """Filtered edge table plus p-values and null expectations."""

    edges: EdgeTable
    upper_pvalue: NDArray[np.float64]
    lower_pvalue: NDArray[np.float64]
    expected: NDArray[np.float64]
    occupation: NDArray[np.float64]


@dataclass(frozen=True)
class FilterResult:
    """Statistical filtering result for observed and absent edges."""

    upper: FilteredEdges
    lower: FilteredEdges
    compatible: FilteredEdges
    absent_lower: FilteredEdges
    alpha: float
    tail: Tail
    correction: Correction


def filter_strength_poisson(
    edges: EdgeTable,
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
    node_count = _node_count(edges)
    strengths_out, strengths_in = _strengths(edges, node_count)
    fit = fit_strength_poisson(strengths_out, strengths_in, self_loops=self_loops)
    upper, lower, expected, occupation = _odme.filter_strength_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _odme.absent_strength_poisson(
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
    upper, lower, expected, occupation = _odme.filter_custom_poisson(
        rates.source.tolist(),
        rates.target.tolist(),
        rates.probability.tolist(),
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _odme.absent_custom_poisson(
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
    """Filter edges against a fitted strength-edges ZIP null model."""
    upper, lower, expected, occupation = _odme.filter_strength_edges_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.lam,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _odme.absent_strength_edges_poisson(
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
    upper, lower, expected, occupation = _odme.filter_strength_cost_poisson(
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
        absent = _odme.absent_strength_cost_poisson(
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
    """Filter edges against a fitted strength-degree ZIP null model."""
    upper, lower, expected, occupation = _odme.filter_strength_degree_poisson(
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
        absent = _odme.absent_strength_degree_poisson(
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
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    positive_weight_rate: float,
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
    """Filter edges against a fitted degree-events ZIP null model."""
    upper, lower, expected, occupation = _odme.filter_degree_events_poisson(
        x.tolist(),
        y.tolist(),
        positive_weight_rate,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _odme.absent_degree_events_poisson(
            x.tolist(),
            y.tolist(),
            positive_weight_rate,
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


def filter_strength_geometric(
    edges: EdgeTable,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
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
    upper, lower, expected, occupation = _odme.filter_strength_geometric(
        x.tolist(),
        y.tolist(),
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _odme.absent_strength_geometric(
            x.tolist(),
            y.tolist(),
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
    x: NDArray[np.float64],
    y: NDArray[np.float64],
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
    upper, lower, expected, occupation = _odme.filter_strength_binomial(
        x.tolist(),
        y.tolist(),
        layers,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _odme.absent_strength_binomial(
            x.tolist(),
            y.tolist(),
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


def filter_strength_neg_binomial(
    edges: EdgeTable,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
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
    upper, lower, expected, occupation = _odme.filter_strength_neg_binomial(
        x.tolist(),
        y.tolist(),
        layers,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _odme.absent_strength_neg_binomial(
            x.tolist(),
            y.tolist(),
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
    upper, lower, expected, occupation = _odme.filter_strength_cost_binomial(
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
        absent = _odme.absent_strength_cost_binomial(
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
    """Filter edges against a strength-edges binomial ZIP null model."""
    upper, lower, expected, occupation = _odme.filter_strength_edges_binomial(
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
        absent = _odme.absent_strength_edges_binomial(
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
    """Filter edges against a strength-degree binomial ZIP null model."""
    upper, lower, expected, occupation = _odme.filter_strength_degree_binomial(
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
        absent = _odme.absent_strength_degree_binomial(
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
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    positive_weight_rate: float,
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
    """Filter edges against a degree-events binomial ZIP null model."""
    upper, lower, expected, occupation = _odme.filter_degree_events_binomial(
        x.tolist(),
        y.tolist(),
        positive_weight_rate,
        layers,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _odme.absent_degree_events_binomial(
            x.tolist(),
            y.tolist(),
            positive_weight_rate,
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


def _classify(
    edges: EdgeTable,
    upper: NDArray[np.float64],
    lower: NDArray[np.float64],
    expected: NDArray[np.float64],
    occupation: NDArray[np.float64],
    absent: tuple[list[int], list[int], list[float], list[float], list[float]] | None,
    *,
    alpha: float,
    tail: Tail,
    correction: Correction,
) -> FilterResult:
    upper_mask, lower_mask = _significance_masks(upper, lower, alpha, tail, correction)
    compatible_mask = ~(upper_mask | lower_mask)
    absent_edges = _empty_filtered()
    if absent is not None:
        src, tgt, lower_absent, expected_absent, occupation_absent = absent
        absent_edges = FilteredEdges(
            edges=EdgeTable(
                source=np.asarray(src, dtype=np.uint64),
                target=np.asarray(tgt, dtype=np.uint64),
                weight=np.zeros(len(src), dtype=np.uint64),
            ),
            upper_pvalue=np.ones(len(src), dtype=np.float64),
            lower_pvalue=np.asarray(lower_absent, dtype=np.float64),
            expected=np.asarray(expected_absent, dtype=np.float64),
            occupation=np.asarray(occupation_absent, dtype=np.float64),
        )
    return FilterResult(
        upper=_slice_filtered(edges, upper, lower, expected, occupation, upper_mask),
        lower=_slice_filtered(edges, upper, lower, expected, occupation, lower_mask),
        compatible=_slice_filtered(
            edges, upper, lower, expected, occupation, compatible_mask
        ),
        absent_lower=absent_edges,
        alpha=alpha,
        tail=tail,
        correction=correction,
    )


def _significance_masks(
    upper: NDArray[np.float64],
    lower: NDArray[np.float64],
    alpha: float,
    tail: Tail,
    correction: Correction,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    upper_alpha = _upper_alpha(alpha, tail)
    lower_alpha = _lower_alpha(alpha, tail)
    if correction == "bonferroni":
        m = max(1, len(upper) + len(lower))
        upper_alpha /= m
        lower_alpha /= m
        return upper < upper_alpha, lower < lower_alpha
    if correction == "fdr":
        upper_mask = np.asarray(_odme.benjamini_hochberg(upper.tolist(), upper_alpha))
        lower_mask = np.asarray(_odme.benjamini_hochberg(lower.tolist(), lower_alpha))
        return upper_mask, lower_mask
    if correction != "none":
        msg = f"unknown correction: {correction}"
        raise ValueError(msg)
    return upper < upper_alpha, lower < lower_alpha


def _upper_alpha(alpha: float, tail: Tail) -> float:
    if tail == "upper":
        return alpha
    if tail == "lower":
        return 0.0
    if tail == "two-sided":
        return alpha / 2.0
    msg = f"unknown tail: {tail}"
    raise ValueError(msg)


def _lower_alpha(alpha: float, tail: Tail) -> float:
    if tail == "lower":
        return alpha
    if tail == "upper":
        return 0.0
    if tail == "two-sided":
        return alpha / 2.0
    msg = f"unknown tail: {tail}"
    raise ValueError(msg)


def _slice_filtered(
    edges: EdgeTable,
    upper: NDArray[np.float64],
    lower: NDArray[np.float64],
    expected: NDArray[np.float64],
    occupation: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> FilteredEdges:
    return FilteredEdges(
        edges=EdgeTable(
            source=edges.source[mask],
            target=edges.target[mask],
            weight=edges.weight[mask],
        ),
        upper_pvalue=upper[mask],
        lower_pvalue=lower[mask],
        expected=expected[mask],
        occupation=occupation[mask],
    )


def _empty_filtered() -> FilteredEdges:
    return FilteredEdges(
        edges=EdgeTable(
            source=np.array([], dtype=np.uint64),
            target=np.array([], dtype=np.uint64),
            weight=np.array([], dtype=np.uint64),
        ),
        upper_pvalue=np.array([], dtype=np.float64),
        lower_pvalue=np.array([], dtype=np.float64),
        expected=np.array([], dtype=np.float64),
        occupation=np.array([], dtype=np.float64),
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
    """Solve for the zero-truncated Poisson rate given the mean."""
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
    "filter_degree_events_poisson",
    "filter_strength_binomial",
    "filter_strength_cost_binomial",
    "filter_strength_cost_poisson",
    "filter_strength_degree_binomial",
    "filter_strength_degree_poisson",
    "filter_strength_edges_binomial",
    "filter_strength_edges_poisson",
    "filter_strength_geometric",
    "filter_strength_neg_binomial",
    "filter_strength_poisson",
]
