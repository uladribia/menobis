"""Partial-constraint fitting: some p_ij known, rest fitted from ME models.

All heavy logic (mask building, excess computation, IPF, rate assembly)
runs in Rust. Python only validates inputs and wraps results.
"""

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

import menobis._menobis as _menobis
from menobis.analysis import directed_degrees, directed_strengths
from menobis.data.frames import EdgeTable
from menobis.models.types import OptimizationDiagnostics, PartialFitResult


def _warn_if_not_converged(name: str, converged: bool, iters: int) -> None:
    if not converged:
        warnings.warn(
            f"{name} did not converge after {iters} iterations",
            stacklevel=3,
        )


def _validate_coordinate_metric(coordinate_metric: str) -> None:
    """Validate supported coordinate distance metric names."""
    if coordinate_metric != "euclidean":
        msg = "coordinate_metric must be 'euclidean'"
        raise ValueError(msg)


def _euclidean_distance(
    coord_x: NDArray[np.float64], coord_y: NDArray[np.float64], source: int, target: int
) -> float:
    """Return projected Euclidean distance between two node coordinates."""
    return float(
        np.hypot(coord_x[source] - coord_x[target], coord_y[source] - coord_y[target])
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

    sources, targets, rates, converged, iters = (
        _menobis.fit_partial_strength_poisson_full(
            s_out.tolist(),
            s_in.tolist(),
            k_src.tolist(),
            k_tgt.tolist(),
            k_rate.tolist(),
            self_loops,
            tolerance,
            max_iterations,
        )
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

    sources, targets, rates, converged, iters = (
        _menobis.fit_partial_degree_poisson_full(
            k_out.tolist(),
            k_in.tolist(),
            k_src.tolist(),
            k_tgt.tolist(),
            self_loops,
            tolerance,
            max_iterations,
        )
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
        _menobis.fit_partial_strength_degree_poisson_full(
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
        _menobis.fit_partial_strength_edges_poisson_full(
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
    coordinate_metric: str = "euclidean",
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
) -> PartialFitResult:
    """Fit partial ME strength-cost from projected coordinates.

    Only ``coordinate_metric="euclidean"`` is currently implemented; the
    argument reserves the API for future metrics such as Manhattan or geodesic
    distance.
    """
    _validate_coordinate_metric(coordinate_metric)
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)
    coord_x = np.asarray(x, dtype=np.float64)
    coord_y = np.asarray(y, dtype=np.float64)
    sources, targets, rates, converged, iters = (
        _menobis.fit_partial_strength_cost_poisson_coordinates_full(
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
    native_func: Callable[..., tuple[list[int], list[int], list[float], bool, int]],
    layers: int,
) -> PartialFitResult:
    """Shared Rust-backed partial coordinate fitting wrapper."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)
    coord_x = np.asarray(x, dtype=np.float64)
    coord_y = np.asarray(y, dtype=np.float64)
    sources_list, targets_list, rates_list, converged, iterations = native_func(
        s_out.tolist(),
        s_in.tolist(),
        k_src.tolist(),
        k_tgt.tolist(),
        k_rate.tolist(),
        coord_x.tolist(),
        coord_y.tolist(),
        float(target_cost),
        layers,
        self_loops,
        tolerance,
        max_iterations,
    )
    _warn_if_not_converged(
        f"fit_partial_strength_cost_{family}_coordinates",
        converged,
        iterations,
    )
    return PartialFitResult(
        source=np.array(sources_list, dtype=np.uint64),
        target=np.array(targets_list, dtype=np.uint64),
        rate=np.array(rates_list, dtype=np.float64),
        constraint="strength_cost",
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
    coordinate_metric: str = "euclidean",
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
) -> PartialFitResult:
    """Fit partial B(M) strength-cost from projected coordinates.

    Computes excess strengths and cost, fits B(M) on free pairs, assembles
    rate table using E[t_ij] = M * x_i * y_j * f / (1 + x_i * y_j * f).
    Only ``coordinate_metric="euclidean"`` is currently implemented.
    """
    _validate_coordinate_metric(coordinate_metric)
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
        native_func=_menobis.fit_partial_strength_cost_binomial_coordinates_full,
        layers=layers,
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
    coordinate_metric: str = "euclidean",
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
) -> PartialFitResult:
    """Fit partial W geometric strength-cost from projected coordinates.

    Uses W conic solver on excess, assembles rates with W formula.
    Only ``coordinate_metric="euclidean"`` is currently implemented.
    """
    _validate_coordinate_metric(coordinate_metric)
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
        native_func=_menobis.fit_partial_strength_cost_w_coordinates_full,
        layers=1,
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
    coordinate_metric: str = "euclidean",
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
) -> PartialFitResult:
    """Fit partial W negative-binomial strength-cost from projected coordinates.

    Uses W conic solver on excess, assembles rates with W(M) formula.
    Only ``coordinate_metric="euclidean"`` is currently implemented.
    """
    _validate_coordinate_metric(coordinate_metric)
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
        native_func=_menobis.fit_partial_strength_cost_w_coordinates_full,
        layers=layers,
    )


# ---------------------------------------------------------------------------
# B (Binomial) partial fits
# ---------------------------------------------------------------------------


def fit_partial_strength_binomial(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    *,
    layers: int = 10,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> PartialFitResult:
    """Partial B(M) strength fit with known pairs."""
    _validate_known_feasibility(
        np.asarray(strength_out, dtype=np.float64),
        np.asarray(strength_in, dtype=np.float64),
        np.asarray(known_source, dtype=np.uint64),
        np.asarray(known_target, dtype=np.uint64),
        np.asarray(known_rate, dtype=np.float64),
    )
    sources, targets, rates, converged, iterations = (
        _menobis.fit_partial_strength_binomial_full(
            np.asarray(strength_out, dtype=np.float64).tolist(),
            np.asarray(strength_in, dtype=np.float64).tolist(),
            np.asarray(known_source, dtype=np.uint64).tolist(),
            np.asarray(known_target, dtype=np.uint64).tolist(),
            np.asarray(known_rate, dtype=np.float64).tolist(),
            int(layers),
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    return _to_partial_result(
        "fit_partial_strength_binomial",
        sources,
        targets,
        rates,
        converged,
        iterations,
        constraint="strength",
        self_loops=self_loops,
        family="binomial",
    )


def fit_partial_strength_edges_binomial(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    target_edges: float,
    *,
    layers: int = 10,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 50000,
) -> PartialFitResult:
    """Partial B(M) strength-edges fit with known pairs."""
    _validate_known_feasibility(
        np.asarray(strength_out, dtype=np.float64),
        np.asarray(strength_in, dtype=np.float64),
        np.asarray(known_source, dtype=np.uint64),
        np.asarray(known_target, dtype=np.uint64),
        np.asarray(known_rate, dtype=np.float64),
    )
    sources, targets, rates, converged, iterations = (
        _menobis.fit_partial_strength_edges_binomial_full(
            np.asarray(strength_out, dtype=np.float64).tolist(),
            np.asarray(strength_in, dtype=np.float64).tolist(),
            np.asarray(known_source, dtype=np.uint64).tolist(),
            np.asarray(known_target, dtype=np.uint64).tolist(),
            np.asarray(known_rate, dtype=np.float64).tolist(),
            float(target_edges),
            int(layers),
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    return _to_partial_result(
        "fit_partial_strength_edges_binomial",
        sources,
        targets,
        rates,
        converged,
        iterations,
        constraint="strength-edges",
        self_loops=self_loops,
        family="binomial",
    )


def fit_partial_strength_degree_binomial(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    *,
    layers: int = 10,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
) -> PartialFitResult:
    """Partial B(M) strength-degree fit with known pairs."""
    _validate_known_feasibility(
        np.asarray(strength_out, dtype=np.float64),
        np.asarray(strength_in, dtype=np.float64),
        np.asarray(known_source, dtype=np.uint64),
        np.asarray(known_target, dtype=np.uint64),
        np.asarray(known_rate, dtype=np.float64),
    )
    sources, targets, rates, converged, iterations = (
        _menobis.fit_partial_strength_degree_binomial_full(
            np.asarray(strength_out, dtype=np.float64).tolist(),
            np.asarray(strength_in, dtype=np.float64).tolist(),
            np.asarray(degree_out, dtype=np.float64).tolist(),
            np.asarray(degree_in, dtype=np.float64).tolist(),
            np.asarray(known_source, dtype=np.uint64).tolist(),
            np.asarray(known_target, dtype=np.uint64).tolist(),
            np.asarray(known_rate, dtype=np.float64).tolist(),
            int(layers),
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    return _to_partial_result(
        "fit_partial_strength_degree_binomial",
        sources,
        targets,
        rates,
        converged,
        iterations,
        constraint="strength-degree",
        self_loops=self_loops,
        family="binomial",
    )


# ---------------------------------------------------------------------------
# W (Geometric) partial fits
# ---------------------------------------------------------------------------


def fit_partial_strength_edges_geometric(
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
    """Partial W (geometric) strength-edges fit with known pairs."""
    _validate_known_feasibility(
        np.asarray(strength_out, dtype=np.float64),
        np.asarray(strength_in, dtype=np.float64),
        np.asarray(known_source, dtype=np.uint64),
        np.asarray(known_target, dtype=np.uint64),
        np.asarray(known_rate, dtype=np.float64),
    )
    sources, targets, rates, converged, iterations = (
        _menobis.fit_partial_strength_edges_w_full(
            np.asarray(strength_out, dtype=np.float64).tolist(),
            np.asarray(strength_in, dtype=np.float64).tolist(),
            np.asarray(known_source, dtype=np.uint64).tolist(),
            np.asarray(known_target, dtype=np.uint64).tolist(),
            np.asarray(known_rate, dtype=np.float64).tolist(),
            float(target_edges),
            1,  # layers=1 for geometric
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    return _to_partial_result(
        "fit_partial_strength_edges_geometric",
        sources,
        targets,
        rates,
        converged,
        iterations,
        constraint="strength-edges",
        self_loops=self_loops,
        family="geometric",
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
    x: NDArray[np.floating] | None = None,
    y: NDArray[np.floating] | None = None,
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
        if x is None or y is None:
            msg = "strength-cost requires projected coordinate arrays x and y"
            raise ValueError(msg)
        coord_x = np.asarray(x, dtype=np.float64)
        coord_y = np.asarray(y, dtype=np.float64)
        if target_cost is None:
            target_cost = float(
                np.sum(
                    edges.weight.astype(np.float64)
                    * np.hypot(
                        coord_x[edges.source.astype(np.int64)]
                        - coord_x[edges.target.astype(np.int64)],
                        coord_y[edges.source.astype(np.int64)]
                        - coord_y[edges.target.astype(np.int64)],
                    )
                )
            )
        return fit_partial_strength_cost_poisson_coordinates(
            s.out.astype(np.float64),
            s.incoming.astype(np.float64),
            known_source,
            known_target,
            known_rate,
            coord_x,
            coord_y,
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
    "fit_partial_strength_binomial",
    "fit_partial_strength_cost_binomial_coordinates",
    "fit_partial_strength_cost_geometric_coordinates",
    "fit_partial_strength_cost_negative_binomial_coordinates",
    "fit_partial_strength_cost_poisson_coordinates",
    "fit_partial_strength_degree_binomial",
    "fit_partial_strength_degree_poisson",
    "fit_partial_strength_edges_binomial",
    "fit_partial_strength_edges_geometric",
    "fit_partial_strength_edges_poisson",
    "fit_partial_strength_poisson",
]
