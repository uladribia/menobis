"""Statistical filtering for ODME null models."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.data.frames import EdgeTable, ProbabilityTable
from odme.models.fitting import StrengthEdgesMEFit, fit_fixed_strength_me

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


def filter_fixed_strength_me(
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
    fit = fit_fixed_strength_me(strengths_out, strengths_in, self_loops=self_loops)
    upper, lower, expected, occupation = _odme.filter_fixed_strength_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _odme.absent_fixed_strength_poisson(
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


def filter_custom_rates_poisson(
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
    upper, lower, expected, occupation = _odme.filter_custom_poisson_rates(
        rates.source.tolist(),
        rates.target.tolist(),
        rates.probability.tolist(),
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _odme.absent_custom_poisson_rates(
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


def filter_strength_edges_me(
    edges: EdgeTable,
    fit: StrengthEdgesMEFit,
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
    upper, lower, expected, occupation = _odme.filter_strength_edges_zip(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.lam,
        edges.source.tolist(),
        edges.target.tolist(),
        edges.weight.tolist(),
    )
    absent = None
    if detect_absent:
        absent = _odme.absent_strength_edges_zip(
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


__all__ = [
    "FilterResult",
    "FilteredEdges",
    "filter_custom_rates_poisson",
    "filter_fixed_strength_me",
    "filter_strength_edges_me",
]
