"""Shared p-value classification helpers for ODME filtering."""

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.data.frames import EdgeTable
from odme.filtering.types import Correction, FilteredEdges, FilterResult, Tail


def _classify_native(
    edges: EdgeTable,
    native_result: tuple[list[float], list[float], list[float], list[float]],
    absent: tuple[list[int], list[int], list[float], list[float], list[float]] | None,
    *,
    alpha: float,
    tail: Tail,
    correction: Correction,
) -> FilterResult:
    """Classify native p-value vectors into filtered edge groups."""
    upper, lower, expected, occupation = native_result
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


def _reject_unsupported_absent_filter(detect_absent: bool, model_name: str) -> None:
    if detect_absent:
        msg = f"absent-edge filtering is not exposed for {model_name} yet"
        raise NotImplementedError(msg)


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


__all__: list[str] = []
