"""Partial-constraint fitting: some p_ij known, rest fitted from ME models."""

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.analysis import directed_strengths
from odme.data.frames import EdgeTable, ProbabilityTable


@dataclass(frozen=True)
class PartialFitResult:
    """Combined rate table from partial-constraint fitting.

    Contains rates for all pairs: known pairs at their fixed values,
    fitted pairs at their model-predicted values. Can be passed directly
    to ``sample_custom_pij_events_poisson`` or ``_multinomial``.
    """

    source: NDArray[np.uint64]
    target: NDArray[np.uint64]
    rate: NDArray[np.float64]

    def as_probability_table(self) -> ProbabilityTable:
        """Convert to ProbabilityTable for sampling (rates as weights)."""
        return ProbabilityTable(
            source=self.source,
            target=self.target,
            probability=self.rate,
        )


def _build_mask(
    n: int,
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
) -> list[bool]:
    mask = [False] * (n * n)
    for s, t in zip(known_source, known_target, strict=True):
        mask[int(s) * n + int(t)] = True
    return mask


def _infer_n(
    strength_out: NDArray[np.float64],
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
) -> int:
    n = len(strength_out)
    if known_source.size > 0:
        n = max(n, int(known_source.max()) + 1, int(known_target.max()) + 1)
    return n


def _compute_excess(
    strength_out: NDArray[np.float64],
    strength_in: NDArray[np.float64],
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
    known_rate: NDArray[np.float64],
    n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    excess_out = strength_out.copy()
    excess_in = strength_in.copy()
    for s, t, r in zip(known_source, known_target, known_rate, strict=True):
        excess_out[int(s)] -= r
        excess_in[int(t)] -= r
    if np.any(excess_out < -1e-6) or np.any(excess_in < -1e-6):
        msg = "known pair rates exceed observed strengths; problem is infeasible"
        raise ValueError(msg)
    excess_out = np.maximum(excess_out, 0.0)
    excess_in = np.maximum(excess_in, 0.0)
    return excess_out, excess_in


def _combine_rates(
    n: int,
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
    known_rate: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    mask: list[bool],
) -> PartialFitResult:
    sources: list[int] = []
    targets: list[int] = []
    rates: list[float] = []
    # Known pairs.
    for s, t, r in zip(known_source, known_target, known_rate, strict=True):
        if r > 0:
            sources.append(int(s))
            targets.append(int(t))
            rates.append(float(r))
    # Fitted pairs.
    for i in range(n):
        for j in range(n):
            if mask[i * n + j]:
                continue
            rate = x[i] * y[j]
            if rate > 0:
                sources.append(i)
                targets.append(j)
                rates.append(rate)
    return PartialFitResult(
        source=np.array(sources, dtype=np.uint64),
        target=np.array(targets, dtype=np.uint64),
        rate=np.array(rates, dtype=np.float64),
    )


def fit_partial_strength_me(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> PartialFitResult:
    """Fit fixed-strength ME with some p_ij pairs known.

    Known pairs contribute their fixed rates to the strength constraints.
    The remaining pairs are fitted using masked IPF so that the full
    strength sequence is recovered.

    Args:
        strength_out: Observed outgoing strength per node.
        strength_in: Observed incoming strength per node.
        known_source: Source node ids of known pairs.
        known_target: Target node ids of known pairs.
        known_rate: Expected event counts for known pairs.
        tolerance: Solver tolerance.
        max_iterations: Maximum solver iterations.

    Returns:
        PartialFitResult with combined rate table.
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)

    n = _infer_n(s_out, k_src, k_tgt)
    if len(s_out) < n:
        s_out = np.pad(s_out, (0, n - len(s_out)))
        s_in = np.pad(s_in, (0, n - len(s_in)))

    excess_out, excess_in = _compute_excess(s_out, s_in, k_src, k_tgt, k_rate, n)
    mask = _build_mask(n, k_src, k_tgt)

    if excess_out.sum() <= 0:
        return _combine_rates(n, k_src, k_tgt, k_rate, np.zeros(n), np.zeros(n), mask)

    # Balance excess so solver doesn't reject.
    diff = excess_out.sum() - excess_in.sum()
    if abs(diff) > 1e-6:
        if diff > 0:
            excess_in[np.argmax(excess_in)] += diff
        else:
            excess_out[np.argmax(excess_out)] -= diff

    x_list, y_list, converged, iters = _odme.fit_masked_strength(
        excess_out.tolist(), excess_in.tolist(), mask, tolerance, max_iterations
    )
    if not converged:
        warnings.warn(
            f"fit_partial_strength_me did not converge after {iters} iterations",
            stacklevel=2,
        )
    x = np.asarray(x_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)
    return _combine_rates(n, k_src, k_tgt, k_rate, x, y, mask)


def fit_from_network_cutoff(
    edges: EdgeTable,
    cutoff: float,
    model: str = "strength",
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> PartialFitResult:
    """Split an observed network by weight cutoff and fit partial constraints.

    Edges with ``weight > cutoff`` are treated as known pairs with fixed rates.
    The remaining structure is fitted using the specified ME model.

    Args:
        edges: Observed weighted edge table.
        cutoff: Weight threshold; edges above this are fixed.
        model: Model variant. Currently ``"strength"`` is supported.
        tolerance: Solver tolerance.
        max_iterations: Maximum solver iterations.

    Returns:
        PartialFitResult with combined rate table.
    """
    s = directed_strengths(edges)

    heavy = edges.weight > cutoff
    known_source = edges.source[heavy]
    known_target = edges.target[heavy]
    known_rate = edges.weight[heavy].astype(np.float64)

    if model == "strength":
        return fit_partial_strength_me(
            s.out.astype(np.float64),
            s.incoming.astype(np.float64),
            known_source,
            known_target,
            known_rate,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    msg = f"unsupported model: {model!r}. Currently only 'strength' is supported."
    raise ValueError(msg)


__all__ = [
    "PartialFitResult",
    "fit_from_network_cutoff",
    "fit_partial_strength_me",
]
