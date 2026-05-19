"""Partial-constraint fitting: some p_ij known, rest fitted from ME models.

All heavy logic (mask building, excess computation, IPF, rate assembly)
runs in Rust. Python only validates inputs and wraps results.
"""

import warnings

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.analysis import directed_degrees, directed_strengths
from odme.data.frames import EdgeTable
from odme.models.types import PartialFitResult


def _warn_if_not_converged(name: str, converged: bool, iters: int) -> None:
    if not converged:
        warnings.warn(
            f"{name} did not converge after {iters} iterations",
            stacklevel=3,
        )


def _validate_known_feasibility(
    out_seq: NDArray[np.float64],
    in_seq: NDArray[np.float64],
    known_src: NDArray[np.uint64],
    known_tgt: NDArray[np.uint64],
    known_contrib: NDArray[np.float64],
) -> None:
    """Raise ValueError if known contributions exceed observed constraints."""
    excess_out = out_seq.copy()
    excess_in = in_seq.copy()
    for s, t, c in zip(known_src, known_tgt, known_contrib, strict=True):
        si, ti = int(s), int(t)
        if si < len(excess_out):
            excess_out[si] -= c
        if ti < len(excess_in):
            excess_in[ti] -= c
    if np.any(excess_out < -1e-6) or np.any(excess_in < -1e-6):
        msg = "known pair contributions exceed observed constraints; infeasible"
        raise ValueError(msg)


def _to_partial_result(
    name: str,
    sources: list[int],
    targets: list[int],
    rates: list[float],
    converged: bool,
    iterations: int,
) -> PartialFitResult:
    """Wrap Rust output into PartialFitResult with convergence warning."""
    _warn_if_not_converged(name, converged, iterations)
    return PartialFitResult(
        source=np.array(sources, dtype=np.uint64),
        target=np.array(targets, dtype=np.uint64),
        rate=np.array(rates, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Public partial fitters
# ---------------------------------------------------------------------------


def fit_partial_strength_poisson(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> PartialFitResult:
    """Fit fixed-strength ME with some p_ij pairs known."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)
    _validate_known_feasibility(s_out, s_in, k_src, k_tgt, k_rate)

    sources, targets, rates, converged, iters = _odme.fit_partial_strength_poisson_full(
        s_out.tolist(),
        s_in.tolist(),
        k_src.tolist(),
        k_tgt.tolist(),
        k_rate.tolist(),
        self_loops,
        tolerance,
        max_iterations,
    )
    return _to_partial_result(
        "fit_partial_strength_poisson",
        sources,
        targets,
        rates,
        converged,
        iters,
    )


def fit_partial_degree_poisson(
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 50000,
) -> PartialFitResult:
    """Fit fixed-degree binary ME with some pairs known."""
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)

    sources, targets, rates, converged, iters = _odme.fit_partial_degree_poisson_full(
        k_out.tolist(),
        k_in.tolist(),
        k_src.tolist(),
        k_tgt.tolist(),
        self_loops,
        tolerance,
        max_iterations,
    )
    return _to_partial_result(
        "fit_partial_degree_poisson",
        sources,
        targets,
        rates,
        converged,
        iters,
    )


def fit_partial_strength_degree_poisson(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 50000,
) -> PartialFitResult:
    """Fit exact ME fixed-strength-degree with some pairs known."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)

    sources, targets, rates, converged, iters = (
        _odme.fit_partial_strength_degree_poisson_full(
            s_out.tolist(),
            s_in.tolist(),
            k_out.tolist(),
            k_in.tolist(),
            k_src.tolist(),
            k_tgt.tolist(),
            k_rate.tolist(),
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    return _to_partial_result(
        "fit_partial_strength_degree_poisson",
        sources,
        targets,
        rates,
        converged,
        iters,
    )


def fit_partial_strength_edges_poisson(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    target_edges: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 50000,
) -> PartialFitResult:
    """Fit fixed-strength + total-edge-count ME with some pairs known."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)

    sources, targets, rates, converged, iters = (
        _odme.fit_partial_strength_edges_poisson_full(
            s_out.tolist(),
            s_in.tolist(),
            k_src.tolist(),
            k_tgt.tolist(),
            k_rate.tolist(),
            target_edges,
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    return _to_partial_result(
        "fit_partial_strength_edges_poisson",
        sources,
        targets,
        rates,
        converged,
        iters,
    )


def fit_partial_strength_cost_poisson(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    cost_sources: NDArray[np.integer],
    cost_targets: NDArray[np.integer],
    cost_values: NDArray[np.floating],
    target_cost: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
) -> PartialFitResult:
    """Fit fixed-strength + fixed-cost ME with some p_ij pairs known."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)
    c_src = np.asarray(cost_sources, dtype=np.int64)
    c_tgt = np.asarray(cost_targets, dtype=np.int64)
    c_val = np.asarray(cost_values, dtype=np.float64)

    sources, targets, rates, converged, iters = (
        _odme.fit_partial_strength_cost_poisson_full(
            s_out.tolist(),
            s_in.tolist(),
            k_src.tolist(),
            k_tgt.tolist(),
            k_rate.tolist(),
            c_src.tolist(),
            c_tgt.tolist(),
            c_val.tolist(),
            target_cost,
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    return _to_partial_result(
        "fit_partial_strength_cost_poisson",
        sources,
        targets,
        rates,
        converged,
        iters,
    )


# ---------------------------------------------------------------------------
# Convenience: split by cutoff
# ---------------------------------------------------------------------------


def fit_from_network_cutoff(
    edges: EdgeTable,
    cutoff: float,
    model: str = "strength",
    *,
    self_loops: bool = True,
    cost_sources: NDArray[np.integer] | None = None,
    cost_targets: NDArray[np.integer] | None = None,
    cost_values: NDArray[np.floating] | None = None,
    target_cost: float | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> PartialFitResult:
    """Split an observed network by weight cutoff and fit partial constraints."""
    s = directed_strengths(edges)
    heavy = edges.weight > cutoff
    known_source = edges.source[heavy]
    known_target = edges.target[heavy]
    known_rate = edges.weight[heavy].astype(np.float64)

    if model == "strength":
        return fit_partial_strength_poisson(
            s.out.astype(np.float64),
            s.incoming.astype(np.float64),
            known_source,
            known_target,
            known_rate,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if model == "degree":
        k = directed_degrees(edges)
        return fit_partial_degree_poisson(
            k.out.astype(np.float64),
            k.incoming.astype(np.float64),
            known_source,
            known_target,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if model == "strength-degree":
        k = directed_degrees(edges)
        return fit_partial_strength_degree_poisson(
            s.out.astype(np.float64),
            s.incoming.astype(np.float64),
            k.out.astype(np.float64),
            k.incoming.astype(np.float64),
            known_source,
            known_target,
            known_rate,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if model == "strength-edges":
        return fit_partial_strength_edges_poisson(
            s.out.astype(np.float64),
            s.incoming.astype(np.float64),
            known_source,
            known_target,
            known_rate,
            float(edges.num_edges),
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if model == "strength-cost":
        if cost_sources is None or cost_targets is None or cost_values is None:
            msg = "strength-cost requires cost_sources, cost_targets, cost_values"
            raise ValueError(msg)
        if target_cost is None:
            cost_map: dict[tuple[int, int], float] = {}
            for cs, ct, cv in zip(
                np.asarray(cost_sources),
                np.asarray(cost_targets),
                np.asarray(cost_values),
                strict=True,
            ):
                cost_map[(int(cs), int(ct))] = float(cv)
            target_cost = sum(
                float(w) * cost_map.get((int(sv), int(tv)), 0.0)
                for sv, tv, w in zip(
                    edges.source, edges.target, edges.weight, strict=True
                )
            )
        return fit_partial_strength_cost_poisson(
            s.out.astype(np.float64),
            s.incoming.astype(np.float64),
            known_source,
            known_target,
            known_rate,
            cost_sources,
            cost_targets,
            cost_values,
            target_cost,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    msg = (
        f"unsupported model: {model!r}. "
        "Use 'strength', 'degree', 'strength-degree', "
        "'strength-edges', or 'strength-cost'."
    )
    raise ValueError(msg)


__all__ = [
    "PartialFitResult",
    "fit_from_network_cutoff",
    "fit_partial_degree_poisson",
    "fit_partial_strength_cost_poisson",
    "fit_partial_strength_degree_poisson",
    "fit_partial_strength_edges_poisson",
    "fit_partial_strength_poisson",
]
