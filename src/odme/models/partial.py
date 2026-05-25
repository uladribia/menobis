"""Partial-constraint fitting: some p_ij known, rest fitted from ME models.

All heavy logic (mask building, excess computation, IPF, rate assembly)
runs in Rust. Python only validates inputs and wraps results.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.analysis import directed_degrees, directed_strengths
from odme.data.frames import EdgeTable
from odme.models.types import OptimizationDiagnostics, PartialFitResult


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
    *,
    constraint: str = "strength",
    self_loops: bool = True,
    family: str = "poisson",
) -> PartialFitResult:
    """Wrap Rust output into PartialFitResult with convergence warning."""
    _warn_if_not_converged(name, converged, iterations)
    return PartialFitResult(
        source=np.array(sources, dtype=np.uint64),
        target=np.array(targets, dtype=np.uint64),
        rate=np.array(rates, dtype=np.float64),
        constraint=constraint,
        family=family,
        self_loops=self_loops,
        converged=converged,
        iterations=iterations,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iterations,
        ),
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
        constraint="strength",
        self_loops=self_loops,
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
        constraint="degree",
        self_loops=self_loops,
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
        constraint="strength_degree",
        self_loops=self_loops,
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
        constraint="strength_edges",
        self_loops=self_loops,
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
        constraint="strength_cost",
        self_loops=self_loops,
    )


def fit_partial_strength_cost_poisson_coordinates(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    target_cost: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
) -> PartialFitResult:
    """Fit partial ME strength-cost from projected Euclidean XY coordinates."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)
    coord_x = np.asarray(x, dtype=np.float64)
    coord_y = np.asarray(y, dtype=np.float64)
    sources, targets, rates, converged, iters = (
        _odme.fit_partial_strength_cost_poisson_coordinates_full(
            s_out.tolist(),
            s_in.tolist(),
            k_src.tolist(),
            k_tgt.tolist(),
            k_rate.tolist(),
            coord_x.tolist(),
            coord_y.tolist(),
            target_cost,
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    return _to_partial_result(
        "fit_partial_strength_cost_poisson_coordinates",
        sources,
        targets,
        rates,
        converged,
        iters,
        constraint="strength_cost",
        self_loops=self_loops,
    )


def _w_rate_func(layers: int) -> Callable[[float, float, float, object], float]:
    """Return W rate function for given M."""
    import math

    def _rate(xi: float, yj: float, f: float, fit: object) -> float:
        # W: E[t_ij] = M * exp(-r) / (1 - exp(-r)), r = -ln(x_i) + (-ln(y_j)) + gamma*d
        # f = exp(-gamma*d), so r = -ln(xi) - ln(yj) - ln(f) = -ln(xi*yj*f)
        z = xi * yj * f
        if z <= 0 or z >= 1:
            return 0.0
        r = -math.log(z)
        if r <= 0:
            return 0.0
        q = math.exp(-r)
        return float(layers) * q / (1.0 - q)

    return _rate


def _partial_family_coordinate(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    target_cost: float,
    *,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
    family: str,
    fit_func: Callable[
        [
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            float,
        ],
        Any,
    ],
    rate_func: Callable[[float, float, float, Any], float],
) -> PartialFitResult:
    """Shared partial coordinate fitting: compute excess, fit family, assemble rates."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)
    coord_x = np.asarray(x, dtype=np.float64)
    coord_y = np.asarray(y, dtype=np.float64)
    n = max(
        len(s_out),
        int(k_src.max()) + 1 if len(k_src) else 0,
        int(k_tgt.max()) + 1 if len(k_tgt) else 0,
    )
    # Pad sequences
    if len(s_out) < n:
        s_out = np.pad(s_out, (0, n - len(s_out)))
    if len(s_in) < n:
        s_in = np.pad(s_in, (0, n - len(s_in)))
    # Build mask
    mask = np.zeros((n, n), dtype=bool)
    for s, t in zip(k_src, k_tgt, strict=True):
        mask[int(s), int(t)] = True
    if not self_loops:
        np.fill_diagonal(mask, True)
    # Compute excess
    excess_out = s_out.copy()
    excess_in = s_in.copy()
    for s, t, r in zip(k_src, k_tgt, k_rate, strict=True):
        excess_out[int(s)] -= r
        excess_in[int(t)] -= r
    excess_out = np.maximum(excess_out, 0.0)
    excess_in = np.maximum(excess_in, 0.0)
    # Balance excess
    diff = excess_out.sum() - excess_in.sum()
    if abs(diff) > 1e-10:
        if diff > 0:
            idx = int(np.argmax(excess_in))
            excess_in[idx] += diff
        else:
            idx = int(np.argmax(excess_out))
            excess_out[idx] -= diff
    # Known cost
    known_cost = sum(
        float(r)
        * float(
            np.hypot(
                coord_x[int(s)] - coord_x[int(t)], coord_y[int(s)] - coord_y[int(t)]
            )
        )
        for s, t, r in zip(k_src, k_tgt, k_rate, strict=True)
    )
    excess_cost = max(0.0, target_cost - known_cost)
    # Fit on excess using family-specific solver
    if excess_out.sum() <= 0:
        # Nothing to fit
        sources_list = k_src.tolist()
        targets_list = k_tgt.tolist()
        rates_list = k_rate.tolist()
    else:
        fit = fit_func(excess_out, excess_in, coord_x, coord_y, excess_cost)
        # Assemble rate table: known + free pairs
        sources_list = []
        targets_list = []
        rates_list = []
        for s, t, r in zip(k_src, k_tgt, k_rate, strict=True):
            sources_list.append(int(s))
            targets_list.append(int(t))
            rates_list.append(float(r))
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    continue
                d = float(np.hypot(coord_x[i] - coord_x[j], coord_y[i] - coord_y[j]))
                f_ij = float(np.exp(-fit.gamma * d))
                rate = rate_func(float(fit.x[i]), float(fit.y[j]), f_ij, fit)
                if rate > 0:
                    sources_list.append(i)
                    targets_list.append(j)
                    rates_list.append(rate)
    _warn_if_not_converged(
        f"fit_partial_strength_cost_{family}_coordinates",
        fit.converged if excess_out.sum() > 0 else True,
        fit.iterations if excess_out.sum() > 0 else 0,
    )
    return PartialFitResult(
        source=np.array(sources_list, dtype=np.uint64),
        target=np.array(targets_list, dtype=np.uint64),
        rate=np.array(rates_list, dtype=np.float64),
        constraint="strength_cost",
        family=family,
        self_loops=self_loops,
        converged=fit.converged if excess_out.sum() > 0 else True,
        iterations=fit.iterations if excess_out.sum() > 0 else 0,
        diagnostics=OptimizationDiagnostics(
            converged=fit.converged if excess_out.sum() > 0 else True,
            status="solved"
            if (excess_out.sum() <= 0 or fit.converged)
            else "inaccurate",
            iterations=fit.iterations if excess_out.sum() > 0 else 0,
        ),
    )


def fit_partial_strength_cost_binomial_coordinates(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    target_cost: float,
    *,
    layers: int = 1,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
) -> PartialFitResult:
    """Fit partial B(M) strength-cost from projected Euclidean XY coordinates.

    Computes excess strengths and cost, fits B(M) on free pairs, assembles
    rate table using E[t_ij] = M * x_i * y_j * f / (1 + x_i * y_j * f).
    """
    from odme.models.fitting import fit_strength_cost_binomial_coordinates as _fit_b

    return _partial_family_coordinate(
        strength_out,
        strength_in,
        known_source,
        known_target,
        known_rate,
        x,
        y,
        target_cost,
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
        family="binomial",
        fit_func=lambda s_o, s_i, cx, cy, tc: _fit_b(
            s_o,
            s_i,
            cx,
            cy,
            tc,
            layers=layers,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
        rate_func=lambda xi, yj, f, fit: (
            float(layers) * xi * yj * f / (1.0 + xi * yj * f)
        ),
    )


def fit_partial_strength_cost_geometric_coordinates(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    target_cost: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
) -> PartialFitResult:
    """Fit partial W geometric strength-cost from projected Euclidean XY coordinates.

    Uses W conic solver on excess, assembles rates with W formula.
    """
    from odme.models.fitting import fit_strength_cost_geometric_coordinates as _fit_wg

    return _partial_family_coordinate(
        strength_out,
        strength_in,
        known_source,
        known_target,
        known_rate,
        x,
        y,
        target_cost,
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
        family="geometric",
        fit_func=lambda s_o, s_i, cx, cy, tc: _fit_wg(
            s_o,
            s_i,
            cx,
            cy,
            tc,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
        rate_func=_w_rate_func(layers=1),
    )


def fit_partial_strength_cost_negative_binomial_coordinates(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    target_cost: float,
    *,
    layers: int = 3,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
) -> PartialFitResult:
    """Fit partial W negative-binomial strength-cost from projected XY.

    Uses W conic solver on excess, assembles rates with W(M) formula.
    """
    from odme.models.fitting import (
        fit_strength_cost_negative_binomial_coordinates as _fit_wnb,
    )

    return _partial_family_coordinate(
        strength_out,
        strength_in,
        known_source,
        known_target,
        known_rate,
        x,
        y,
        target_cost,
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
        family="negative_binomial",
        fit_func=lambda s_o, s_i, cx, cy, tc: _fit_wnb(
            s_o,
            s_i,
            cx,
            cy,
            tc,
            layers=layers,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
        rate_func=_w_rate_func(layers=layers),
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
    "fit_partial_strength_cost_binomial_coordinates",
    "fit_partial_strength_cost_geometric_coordinates",
    "fit_partial_strength_cost_negative_binomial_coordinates",
    "fit_partial_strength_cost_poisson",
    "fit_partial_strength_cost_poisson_coordinates",
    "fit_partial_strength_degree_poisson",
    "fit_partial_strength_edges_poisson",
    "fit_partial_strength_poisson",
]
