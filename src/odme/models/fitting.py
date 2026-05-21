"""Lagrange multiplier fitting for ODME multi-edge models."""

import time
import warnings

import numpy as np
from loguru import logger
from numpy.typing import NDArray

import odme._odme as _odme
from odme.models.types import (
    ConicDiagnostics,
    DegreeEventsFit,
    FitResult,
    OptimizationDiagnostics,
    StrengthCostFit,
    StrengthDegreeFit,
    StrengthEdgesFit,
)


def _log_fit_result(
    name: str,
    converged: bool,
    iterations: int,
    elapsed: float,
    verbose: int,
) -> None:
    """Log fitting diagnostics at the requested verbosity level."""
    if verbose >= 2:
        status = "converged" if converged else "NOT converged"
        logger.info(
            "{name} {status} in {iters} iterations ({elapsed:.4f}s)",
            name=name,
            status=status,
            iters=iterations,
            elapsed=elapsed,
        )
    elif verbose >= 1 and not converged:
        logger.warning(
            "{name} did not converge after {iters} iterations ({elapsed:.4f}s)",
            name=name,
            iters=iterations,
            elapsed=elapsed,
        )
    if not converged:
        warnings.warn(
            f"{name} did not converge after {iterations} iterations",
            stacklevel=3,
        )


def _validate_balanced_sequences(
    out_sequence: NDArray[np.float64],
    in_sequence: NDArray[np.float64],
    *,
    name: str,
) -> None:
    if len(out_sequence) != len(in_sequence):
        msg = f"{name}_out and {name}_in must have the same length"
        raise ValueError(msg)
    if np.any(~np.isfinite(out_sequence)) or np.any(~np.isfinite(in_sequence)):
        msg = f"fixed-{name} fitting requires finite sequences"
        raise ValueError(msg)
    if np.any(out_sequence < 0) or np.any(in_sequence < 0):
        msg = f"fixed-{name} fitting requires non-negative sequences"
        raise ValueError(msg)
    if not np.isclose(out_sequence.sum(), in_sequence.sum()):
        msg = f"fixed-{name} fitting requires balanced in/out sequences"
        raise ValueError(msg)


def _candidate_pair_capacity(node_count: int, *, self_loops: bool) -> int:
    """Return candidate directed pair count for a self-loop policy."""
    return (
        node_count * node_count if self_loops else node_count * max(node_count - 1, 0)
    )


def _validate_binomial_feasibility(
    strength_out: NDArray[np.float64],
    strength_in: NDArray[np.float64],
    layers: int,
    *,
    self_loops: bool,
) -> None:
    """Validate that binomial(M) model can achieve the target strengths.

    Maximum achievable strength per node is M * (N or N-1), depending on
    self_loops. Reject inputs that exceed this capacity.
    """
    n = len(strength_out)
    pairs_per_node = n if self_loops else max(n - 1, 0)
    max_achievable = float(layers * pairs_per_node)
    if np.any(strength_out > max_achievable) or np.any(strength_in > max_achievable):
        msg = (
            f"binomial(M={layers}) model cannot achieve strengths exceeding "
            f"M*(N{'' if self_loops else '-1'}) = {max_achievable:.0f}; "
            f"max(s_out)={strength_out.max():.0f}, max(s_in)={strength_in.max():.0f}"
        )
        raise ValueError(msg)


def _validate_strength_edges_constraints(
    strength_out: NDArray[np.float64],
    strength_in: NDArray[np.float64],
    target_edges: float,
    *,
    self_loops: bool,
) -> None:
    """Validate common strength-edges feasibility at the Python boundary."""
    _validate_balanced_sequences(strength_out, strength_in, name="strength")
    if not np.isfinite(target_edges):
        msg = "target_edges must be finite"
        raise ValueError(msg)
    total_strength = float(strength_out.sum())
    capacity = _candidate_pair_capacity(len(strength_out), self_loops=self_loops)
    if target_edges <= 0.0 or target_edges > total_strength:
        msg = "target_edges must be positive and no larger than total strength"
        raise ValueError(msg)
    if target_edges >= capacity:
        msg = "target_edges must be smaller than candidate-pair capacity"
        raise ValueError(msg)


def _validate_cost_entries(
    cost_sources: NDArray[np.integer],
    cost_targets: NDArray[np.integer],
    cost_values: NDArray[np.floating],
    *,
    target_cost: float,
    allow_zero_target: bool = False,
) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.float64]]:
    """Validate shared sparse cost entries for strength-cost fitting."""
    if not np.isfinite(target_cost) or (
        target_cost < 0.0 if allow_zero_target else target_cost <= 0.0
    ):
        qualifier = "non-negative" if allow_zero_target else "positive"
        msg = f"target_cost must be finite and {qualifier}"
        raise ValueError(msg)
    c_src = np.asarray(cost_sources, dtype=np.uint64)
    c_tgt = np.asarray(cost_targets, dtype=np.uint64)
    c_val = np.asarray(cost_values, dtype=np.float64)
    if len(c_src) != len(c_tgt) or len(c_src) != len(c_val):
        msg = "cost_sources, cost_targets, and cost_values must have the same length"
        raise ValueError(msg)
    if np.any(~np.isfinite(c_val)) or np.any(c_val < 0.0):
        msg = "cost_values must be finite and non-negative"
        raise ValueError(msg)
    return c_src, c_tgt, c_val


def _validate_degree_events_constraints(
    degree_out: NDArray[np.float64],
    degree_in: NDArray[np.float64],
    total_events: int,
    *,
    self_loops: bool,
) -> None:
    """Validate shared W degree-events feasibility."""
    _validate_balanced_sequences(degree_out, degree_in, name="degree")
    if not np.isfinite(float(total_events)):
        msg = "total_events must be finite"
        raise ValueError(msg)
    e = float(degree_out.sum())
    t = float(total_events)
    if t < e:
        msg = "total_events must be >= sum(degree_out)"
        raise ValueError(msg)
    capacity = float(len(degree_out) if self_loops else max(len(degree_out) - 1, 0))
    if np.any(degree_out > capacity) or np.any(degree_in > capacity):
        msg = (
            "degree-events fitting received infeasible degree"
            " constraints (above capacity)"
        )
        raise ValueError(msg)


def validate_strength_degree_constraints(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
) -> None:
    """Validate coupled strength-degree model constraints.

    Every node must have strength greater than or equal to degree because ODME
    accepts only positive integer edge weights. A node with degree ``k`` has at
    least ``k`` unit-weight edges and therefore strength at least ``k``.

    Args:
        strength_out: Outgoing strength per node.
        strength_in: Incoming strength per node.
        degree_out: Outgoing binary degree per node.
        degree_in: Incoming binary degree per node.

    Raises:
        ValueError: If arrays are unbalanced, mismatched, negative, or violate
            ``strength >= degree`` for any node.
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)

    _validate_balanced_sequences(s_out, s_in, name="strength")
    _validate_balanced_sequences(k_out, k_in, name="degree")
    if len(s_out) != len(k_out):
        msg = "strength and degree sequences must have the same length"
        raise ValueError(msg)
    if np.any(s_out < k_out) or np.any(s_in < k_in):
        msg = "each node strength must be greater than or equal to its degree"
        raise ValueError(msg)


def fit_strength_cost_poisson(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    cost_sources: NDArray[np.integer],
    cost_targets: NDArray[np.integer],
    cost_values: NDArray[np.floating],
    target_cost: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    verbose: int = 0,
    max_iterations: int = 5000,
) -> StrengthCostFit:
    """Fit the strength-cost ME model: fixed strength + fixed total cost.

    The fitted expectation is ``E[t_ij] = x_i y_j exp(-gamma d_ij)``.

    Args:
        strength_out: Outgoing strength per node.
        strength_in: Incoming strength per node.
        cost_sources: Source node ids for cost entries.
        cost_targets: Target node ids for cost entries.
        cost_values: Cost/distance values for each (source, target) pair.
        target_cost: Observed total cost C = sum t_ij d_ij.
        self_loops: Whether self loops are allowed.
        tolerance: Solver tolerance.
        verbose: Logging level (0=silent, 1=warnings, 2=info).
        max_iterations: Maximum solver iterations.

    Returns:
        StrengthCostFit with x, y multipliers and gamma.
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    _validate_balanced_sequences(s_out, s_in, name="strength")
    c_src_arr, c_tgt_arr, c_val_arr = _validate_cost_entries(
        cost_sources, cost_targets, cost_values, target_cost=target_cost
    )
    if c_val_arr.size > 5_000_000:
        warnings.warn(
            "large sparse cost arrays may use substantial memory; use "
            "fit_strength_cost_poisson_coordinates for complete Euclidean costs",
            stacklevel=2,
        )
    c_src = c_src_arr.tolist()
    c_tgt = c_tgt_arr.tolist()
    c_val = c_val_arr.tolist()

    t0 = time.perf_counter()
    x_list, y_list, gamma, converged, iters = _odme.fit_strength_cost_poisson(
        s_out.tolist(),
        s_in.tolist(),
        c_src,
        c_tgt,
        c_val,
        target_cost,
        self_loops,
        tolerance,
        max_iterations,
    )
    n = len(s_out)
    result = StrengthCostFit(
        node=np.arange(n, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        gamma=gamma,
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="poisson",
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )
    _log_fit_result(
        "fit_strength_cost_poisson", converged, iters, time.perf_counter() - t0, verbose
    )
    return result


def fit_strength_cost_poisson_coordinates(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    target_cost: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    verbose: int = 0,
    max_iterations: int = 5000,
) -> StrengthCostFit:
    """Fit strength-cost ME using projected Euclidean XY coordinates."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    coord_x = np.asarray(x, dtype=np.float64)
    coord_y = np.asarray(y, dtype=np.float64)
    _validate_balanced_sequences(s_out, s_in, name="strength")
    if len(s_out) != len(coord_x) or len(s_out) != len(coord_y):
        msg = "strength and coordinate arrays must have the same length"
        raise ValueError(msg)
    if not np.all(np.isfinite(coord_x)) or not np.all(np.isfinite(coord_y)):
        msg = "coordinates must be finite projected XY values"
        raise ValueError(msg)
    t0 = time.perf_counter()
    x_list, y_list, gamma, converged, iters = (
        _odme.fit_strength_cost_poisson_coordinates(
            s_out.tolist(),
            s_in.tolist(),
            coord_x.tolist(),
            coord_y.tolist(),
            target_cost,
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    result = StrengthCostFit(
        node=np.arange(len(s_out), dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        gamma=gamma,
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="poisson",
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )
    _log_fit_result(
        "fit_strength_cost_poisson_coordinates",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def _coordinate_cost_triples(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.float64]]:
    """Build complete Euclidean cost triples from projected XY coordinates."""
    coord_x = np.asarray(x, dtype=np.float64)
    coord_y = np.asarray(y, dtype=np.float64)
    n = len(coord_x)
    source, target = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    distance = np.hypot(
        coord_x[source] - coord_x[target], coord_y[source] - coord_y[target]
    )
    return (
        source.ravel().astype(np.uint64),
        target.ravel().astype(np.uint64),
        distance.ravel().astype(np.float64),
    )


def fit_strength_cost_binomial_coordinates(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    target_cost: float,
    *,
    layers: int = 1,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    verbose: int = 0,
    max_iterations: int = 5000,
) -> StrengthCostFit:
    """Fit B(M) strength-cost from projected Euclidean XY coordinates.

    Thesis equation: E[t_ij] = M * x_i * y_j * f / (1 + x_i * y_j * f), f=exp(-gamma*d).
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    coord_x = np.asarray(x, dtype=np.float64)
    coord_y = np.asarray(y, dtype=np.float64)
    _validate_balanced_sequences(s_out, s_in, name="strength")
    _validate_binomial_feasibility(s_out, s_in, layers, self_loops=self_loops)
    if len(s_out) != len(coord_x) or len(s_out) != len(coord_y):
        msg = "strength and coordinate arrays must have the same length"
        raise ValueError(msg)
    t0 = time.perf_counter()
    x_list, y_list, gamma, converged, iters = (
        _odme.fit_strength_cost_binomial_coordinates(
            s_out.tolist(),
            s_in.tolist(),
            coord_x.tolist(),
            coord_y.tolist(),
            target_cost,
            layers,
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    result = StrengthCostFit(
        node=np.arange(len(s_out), dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        gamma=gamma,
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="binomial",
        layers=layers,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )
    _log_fit_result(
        "fit_strength_cost_binomial_coordinates",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def _fit_strength_cost_w_coordinates(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    target_cost: float,
    *,
    layers: int,
    self_loops: bool,
    tolerance: float,
    verbose: int,
    max_iterations: int,
    family: str,
) -> StrengthCostFit:
    """Shared W coordinate strength-cost fitting (geometric or negative binomial)."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    coord_x = np.asarray(x, dtype=np.float64)
    coord_y = np.asarray(y, dtype=np.float64)
    _validate_balanced_sequences(s_out, s_in, name="strength")
    if len(s_out) != len(coord_x) or len(s_out) != len(coord_y):
        msg = "strength and coordinate arrays must have the same length"
        raise ValueError(msg)
    t0 = time.perf_counter()
    x_list, y_list, gamma, converged, iters = _odme.fit_strength_cost_w_coordinates(
        s_out.tolist(),
        s_in.tolist(),
        coord_x.tolist(),
        coord_y.tolist(),
        target_cost,
        layers,
        self_loops,
        tolerance,
        max_iterations,
    )
    result = StrengthCostFit(
        node=np.arange(len(s_out), dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        gamma=gamma,
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family=family,
        layers=layers if layers > 1 else None,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )
    _log_fit_result(
        f"fit_strength_cost_{family}_coordinates",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def fit_strength_cost_geometric_coordinates(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    target_cost: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    verbose: int = 0,
    max_iterations: int = 5000,
) -> StrengthCostFit:
    """Fit W geometric strength-cost from projected Euclidean XY coordinates.

    Thesis: E[t_ij] = exp(-r) / (1-exp(-r)), r = a_i + b_j + gamma*d_ij.
    """
    return _fit_strength_cost_w_coordinates(
        strength_out,
        strength_in,
        x,
        y,
        target_cost,
        layers=1,
        self_loops=self_loops,
        tolerance=tolerance,
        verbose=verbose,
        max_iterations=max_iterations,
        family="geometric",
    )


def fit_strength_cost_negative_binomial_coordinates(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    target_cost: float,
    *,
    layers: int = 3,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    verbose: int = 0,
    max_iterations: int = 5000,
) -> StrengthCostFit:
    """Fit W negative-binomial strength-cost from projected Euclidean XY coordinates.

    Thesis: E[t_ij] = M*exp(-r) / (1-exp(-r)), r = a_i + b_j + gamma*d_ij.
    """
    return _fit_strength_cost_w_coordinates(
        strength_out,
        strength_in,
        x,
        y,
        target_cost,
        layers=layers,
        self_loops=self_loops,
        tolerance=tolerance,
        verbose=verbose,
        max_iterations=max_iterations,
        family="negative_binomial",
    )


def fit_strength_edges_poisson(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    target_edges: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    verbose: int = 0,
    max_iterations: int = 50000,
) -> StrengthEdgesFit:
    """Fit exact ME fixed-strength and total-edge-count constraints."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    _validate_strength_edges_constraints(
        s_out, s_in, target_edges, self_loops=self_loops
    )
    t0 = time.perf_counter()
    x_list, y_list, lam, converged, iters = _odme.fit_strength_edges_poisson(
        s_out.tolist(),
        s_in.tolist(),
        target_edges,
        self_loops,
        tolerance,
        max_iterations,
    )
    result = StrengthEdgesFit(
        node=np.arange(len(s_out), dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        lam=float(lam),
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="poisson",
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )
    _log_fit_result(
        "fit_strength_edges_poisson",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def _wrap_w_strength_edges_fit(
    native_result: tuple,
    *,
    node_count: int,
    self_loops: bool,
) -> StrengthEdgesFit:
    (
        x_list,
        y_list,
        lam,
        layers,
        status,
        objective,
        iterations,
        diagnostics,
        metrics,
    ) = native_result
    (
        min_margin,
        max_q,
        max_strength_residual,
        total_strength_residual,
        _edge_residual,
    ) = diagnostics
    (
        variables,
        auxiliary_variables,
        exponential_cones,
        power_cones,
        linear_constraints,
        sparse_nonzeros,
    ) = metrics
    converged = status == "solved"
    return StrengthEdgesFit(
        node=np.arange(node_count, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        lam=float(lam),
        self_loops=self_loops,
        converged=converged,
        iterations=iterations,
        family="geometric" if layers == 1 else "negative_binomial",
        layers=layers,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status=status,
            iterations=iterations,
            objective=objective,
            max_strength_residual=max_strength_residual,
            total_strength_residual=total_strength_residual,
            conic=ConicDiagnostics(
                min_margin=min_margin,
                max_q=max_q,
                variables=variables,
                auxiliary_variables=auxiliary_variables,
                exponential_cones=exponential_cones,
                power_cones=power_cones,
                linear_constraints=linear_constraints,
                sparse_nonzeros=sparse_nonzeros,
            ),
        ),
    )


def fit_strength_edges_geometric(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    target_edges: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 1000,
) -> StrengthEdgesFit:
    """Fit the W fixed-strength-plus-edge-count geometric model."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    _validate_strength_edges_constraints(
        s_out, s_in, target_edges, self_loops=self_loops
    )
    t0 = time.perf_counter()
    result = _wrap_w_strength_edges_fit(
        _odme.fit_strength_edges_geometric(
            s_out.tolist(),
            s_in.tolist(),
            float(target_edges),
            self_loops,
            tolerance,
            max_iterations,
        ),
        node_count=len(s_out),
        self_loops=self_loops,
    )
    _log_fit_result(
        "fit_strength_edges_geometric",
        result.converged,
        result.iterations,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def fit_strength_edges_negative_binomial(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    target_edges: float,
    *,
    layers: int = 3,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 1000,
) -> StrengthEdgesFit:
    """Fit the W fixed-strength-plus-edge-count negative-binomial model."""
    if layers <= 1:
        msg = "negative binomial W fitting requires layers > 1; use geometric for M = 1"
        raise ValueError(msg)
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    _validate_strength_edges_constraints(
        s_out, s_in, target_edges, self_loops=self_loops
    )
    t0 = time.perf_counter()
    result = _wrap_w_strength_edges_fit(
        _odme.fit_strength_edges_negative_binomial(
            s_out.tolist(),
            s_in.tolist(),
            float(target_edges),
            layers,
            self_loops,
            tolerance,
            max_iterations,
        ),
        node_count=len(s_out),
        self_loops=self_loops,
    )
    _log_fit_result(
        "fit_strength_edges_negative_binomial",
        result.converged,
        result.iterations,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def _wrap_w_strength_degree_fit(
    native_result: tuple,
    *,
    node_count: int,
    self_loops: bool,
) -> StrengthDegreeFit:
    (
        x_list,
        y_list,
        z_list,
        w_list,
        layers,
        status,
        objective,
        iterations,
        diagnostics,
        metrics,
    ) = native_result
    (
        min_margin,
        max_q,
        max_strength_residual,
        total_strength_residual,
        _max_degree_residual,
    ) = diagnostics
    (
        variables,
        auxiliary_variables,
        exponential_cones,
        power_cones,
        linear_constraints,
        sparse_nonzeros,
    ) = metrics
    converged = status == "solved"
    return StrengthDegreeFit(
        node=np.arange(node_count, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        z=np.asarray(z_list, dtype=np.float64),
        w=np.asarray(w_list, dtype=np.float64),
        self_loops=self_loops,
        converged=converged,
        iterations=iterations,
        family="geometric" if layers == 1 else "negative_binomial",
        layers=layers,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status=status,
            iterations=iterations,
            objective=objective,
            max_strength_residual=max_strength_residual,
            total_strength_residual=total_strength_residual,
            conic=ConicDiagnostics(
                min_margin=min_margin,
                max_q=max_q,
                variables=variables,
                auxiliary_variables=auxiliary_variables,
                exponential_cones=exponential_cones,
                power_cones=power_cones,
                linear_constraints=linear_constraints,
                sparse_nonzeros=sparse_nonzeros,
            ),
        ),
    )


def fit_strength_degree_geometric(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 1000,
) -> StrengthDegreeFit:
    """Fit the W fixed-strength-degree geometric model."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    validate_strength_degree_constraints(s_out, s_in, k_out, k_in)
    t0 = time.perf_counter()
    result = _wrap_w_strength_degree_fit(
        _odme.fit_strength_degree_geometric(
            s_out.tolist(),
            s_in.tolist(),
            k_out.tolist(),
            k_in.tolist(),
            self_loops,
            tolerance,
            max_iterations,
        ),
        node_count=len(s_out),
        self_loops=self_loops,
    )
    _log_fit_result(
        "fit_strength_degree_geometric",
        result.converged,
        result.iterations,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def fit_strength_degree_negative_binomial(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    *,
    layers: int = 3,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 1000,
) -> StrengthDegreeFit:
    """Fit the W fixed-strength-degree negative-binomial model."""
    if layers <= 1:
        msg = "negative binomial W fitting requires layers > 1; use geometric for M = 1"
        raise ValueError(msg)
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    validate_strength_degree_constraints(s_out, s_in, k_out, k_in)
    t0 = time.perf_counter()
    result = _wrap_w_strength_degree_fit(
        _odme.fit_strength_degree_negative_binomial(
            s_out.tolist(),
            s_in.tolist(),
            k_out.tolist(),
            k_in.tolist(),
            layers,
            self_loops,
            tolerance,
            max_iterations,
        ),
        node_count=len(s_out),
        self_loops=self_loops,
    )
    _log_fit_result(
        "fit_strength_degree_negative_binomial",
        result.converged,
        result.iterations,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def fit_strength_degree_poisson(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    verbose: int = 0,
    max_iterations: int = 50000,
) -> StrengthDegreeFit:
    """Fit exact grand-canonical ME fixed-strength-degree constraints.

    The fitted expectation is the thesis Case 4 ME equation:
    ``E[t_ij] = z_i w_j x_i y_j exp(x_i y_j) /
    (1 + z_i w_j (exp(x_i y_j) - 1))``.

    Args:
        strength_out: Expected outgoing strength per node.
        strength_in: Expected incoming strength per node.
        degree_out: Expected outgoing binary degree per node.
        degree_in: Expected incoming binary degree per node.
        self_loops: Whether self loops are allowed.
        tolerance: Solver tolerance.
        verbose: Logging level (0=silent, 1=warnings, 2=info).
        max_iterations: Maximum solver iterations.

    Returns:
        StrengthDegreeFit with thesis multipliers x, y, z, and w.
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    validate_strength_degree_constraints(s_out, s_in, k_out, k_in)

    t0 = time.perf_counter()
    x_list, y_list, z_list, w_list, converged, iters = (
        _odme.fit_strength_degree_poisson(
            s_out.tolist(),
            s_in.tolist(),
            k_out.tolist(),
            k_in.tolist(),
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    n = len(s_out)
    result = StrengthDegreeFit(
        node=np.arange(n, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        z=np.asarray(z_list, dtype=np.float64),
        w=np.asarray(w_list, dtype=np.float64),
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="poisson",
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )
    _log_fit_result(
        "fit_strength_degree_poisson",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def fit_strength_poisson(
    strength_out: NDArray[np.integer],
    strength_in: NDArray[np.integer],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 10000,
) -> FitResult:
    """Fit Lagrange multipliers for the directed ME fixed-strength model.

    Args:
        strength_out: Outgoing strength per node.
        strength_in: Incoming strength per node.
        self_loops: Whether self loops are allowed.
        tolerance: Convergence tolerance for iterative balancing.
        verbose: Logging level (0=silent, 1=warnings, 2=info).
        max_iterations: Maximum iterations for iterative balancing.

    Returns:
        FitResult with node ids and x, y multipliers.
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)

    _validate_balanced_sequences(s_out, s_in, name="strength")

    n = len(s_out)
    t0 = time.perf_counter()
    x_list, y_list, converged, iters = _odme.fit_strength_poisson(
        s_out.tolist(), s_in.tolist(), self_loops, tolerance, max_iterations
    )
    _log_fit_result(
        "fit_strength_poisson", converged, iters, time.perf_counter() - t0, verbose
    )
    return FitResult(
        node=np.arange(n, dtype=np.uint64),
        x=np.array(x_list),
        y=np.array(y_list),
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="poisson",
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )


def _wrap_w_strength_fit(
    native_result: tuple[
        list[float],
        list[float],
        int,
        str,
        float,
        int,
        float,
        float,
        float,
        float,
        tuple[int, int, int, int, int, int],
    ],
    *,
    node_count: int,
    self_loops: bool = True,
) -> FitResult:
    (
        x_list,
        y_list,
        layers,
        status,
        objective,
        iterations,
        min_margin,
        max_q,
        max_strength_residual,
        total_strength_residual,
        metrics,
    ) = native_result
    (
        variables,
        auxiliary_variables,
        exponential_cones,
        power_cones,
        linear_constraints,
        sparse_nonzeros,
    ) = metrics
    converged = status == "solved"
    return FitResult(
        node=np.arange(node_count, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        self_loops=self_loops,
        converged=converged,
        iterations=iterations,
        family="geometric" if layers == 1 else "negative_binomial",
        layers=layers,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status=status,
            iterations=iterations,
            objective=objective,
            max_strength_residual=max_strength_residual,
            total_strength_residual=total_strength_residual,
            conic=ConicDiagnostics(
                min_margin=min_margin,
                max_q=max_q,
                variables=variables,
                auxiliary_variables=auxiliary_variables,
                exponential_cones=exponential_cones,
                power_cones=power_cones,
                linear_constraints=linear_constraints,
                sparse_nonzeros=sparse_nonzeros,
            ),
        ),
    )


def fit_strength_geometric(
    strength_out: NDArray[np.integer],
    strength_in: NDArray[np.integer],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 1000,
) -> FitResult:
    """Fit the independent W fixed-strength geometric model."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    _validate_balanced_sequences(s_out, s_in, name="strength")

    t0 = time.perf_counter()
    result = _wrap_w_strength_fit(
        _odme.fit_strength_geometric(
            s_out.tolist(), s_in.tolist(), self_loops, tolerance, max_iterations
        ),
        node_count=len(s_out),
        self_loops=self_loops,
    )
    _log_fit_result(
        "fit_strength_geometric",
        result.converged,
        result.iterations,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def fit_strength_negative_binomial(
    strength_out: NDArray[np.integer],
    strength_in: NDArray[np.integer],
    *,
    layers: int = 3,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 1000,
) -> FitResult:
    """Fit the independent W fixed-strength negative-binomial model."""
    if layers <= 1:
        msg = "negative binomial W fitting requires layers > 1; use geometric for M = 1"
        raise ValueError(msg)
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    _validate_balanced_sequences(s_out, s_in, name="strength")

    t0 = time.perf_counter()
    result = _wrap_w_strength_fit(
        _odme.fit_strength_negative_binomial(
            s_out.tolist(),
            s_in.tolist(),
            layers,
            self_loops,
            tolerance,
            max_iterations,
        ),
        node_count=len(s_out),
        self_loops=self_loops,
    )
    _log_fit_result(
        "fit_strength_negative_binomial",
        result.converged,
        result.iterations,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def _wrap_w_strength_cost_fit(
    native_result: tuple,
    *,
    node_count: int,
    self_loops: bool,
) -> StrengthCostFit:
    (
        x_list,
        y_list,
        gamma,
        layers,
        status,
        objective,
        iterations,
        diagnostics,
        metrics,
    ) = native_result
    (
        min_margin,
        max_q,
        max_strength_residual,
        total_strength_residual,
        cost_residual,
    ) = diagnostics
    (
        variables,
        auxiliary_variables,
        exponential_cones,
        power_cones,
        linear_constraints,
        sparse_nonzeros,
    ) = metrics
    converged = status == "solved"
    return StrengthCostFit(
        node=np.arange(node_count, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        gamma=float(gamma),
        self_loops=self_loops,
        converged=converged,
        iterations=iterations,
        family="geometric" if layers == 1 else "negative_binomial",
        layers=layers,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status=status,
            iterations=iterations,
            objective=objective,
            max_strength_residual=max_strength_residual,
            total_strength_residual=total_strength_residual,
            cost_residual=cost_residual,
            conic=ConicDiagnostics(
                min_margin=min_margin,
                max_q=max_q,
                variables=variables,
                auxiliary_variables=auxiliary_variables,
                exponential_cones=exponential_cones,
                power_cones=power_cones,
                linear_constraints=linear_constraints,
                sparse_nonzeros=sparse_nonzeros,
            ),
        ),
    )


def fit_strength_cost_geometric(
    strength_out: NDArray[np.integer],
    strength_in: NDArray[np.integer],
    cost_sources: NDArray[np.integer],
    cost_targets: NDArray[np.integer],
    cost_values: NDArray[np.floating],
    target_cost: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 1000,
) -> StrengthCostFit:
    """Fit the W fixed-strength-plus-cost geometric model."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    _validate_balanced_sequences(s_out, s_in, name="strength")
    c_src, c_tgt, c_val = _validate_cost_entries(
        cost_sources,
        cost_targets,
        cost_values,
        target_cost=target_cost,
        allow_zero_target=True,
    )

    t0 = time.perf_counter()
    result = _wrap_w_strength_cost_fit(
        _odme.fit_strength_cost_geometric(
            s_out.tolist(),
            s_in.tolist(),
            c_src.tolist(),
            c_tgt.tolist(),
            c_val.tolist(),
            float(target_cost),
            self_loops,
            tolerance,
            max_iterations,
        ),
        node_count=len(s_out),
        self_loops=self_loops,
    )
    _log_fit_result(
        "fit_strength_cost_geometric",
        result.converged,
        result.iterations,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def fit_strength_cost_negative_binomial(
    strength_out: NDArray[np.integer],
    strength_in: NDArray[np.integer],
    cost_sources: NDArray[np.integer],
    cost_targets: NDArray[np.integer],
    cost_values: NDArray[np.floating],
    target_cost: float,
    *,
    layers: int = 3,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 1000,
) -> StrengthCostFit:
    """Fit the W fixed-strength-plus-cost negative-binomial model."""
    if layers <= 1:
        msg = "negative binomial W fitting requires layers > 1; use geometric for M = 1"
        raise ValueError(msg)
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    _validate_balanced_sequences(s_out, s_in, name="strength")
    c_src, c_tgt, c_val = _validate_cost_entries(
        cost_sources,
        cost_targets,
        cost_values,
        target_cost=target_cost,
        allow_zero_target=True,
    )

    t0 = time.perf_counter()
    result = _wrap_w_strength_cost_fit(
        _odme.fit_strength_cost_negative_binomial(
            s_out.tolist(),
            s_in.tolist(),
            c_src.tolist(),
            c_tgt.tolist(),
            c_val.tolist(),
            float(target_cost),
            layers,
            self_loops,
            tolerance,
            max_iterations,
        ),
        node_count=len(s_out),
        self_loops=self_loops,
    )
    _log_fit_result(
        "fit_strength_cost_negative_binomial",
        result.converged,
        result.iterations,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def fit_degree_bernoulli(
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    verbose: int = 0,
    max_iterations: int = 50000,
) -> FitResult:
    """Fit multipliers for the directed binary fixed-degree model.

    The model uses ``p_ij = x_i y_j / (1 + x_i y_j)`` and fits expected
    outgoing and incoming binary degrees.

    Args:
        degree_out: Expected outgoing degree per node.
        degree_in: Expected incoming degree per node.
        self_loops: Whether self loops are allowed.
        tolerance: Convergence tolerance for alternating coordinate updates.
        verbose: Logging level (0=silent, 1=warnings, 2=info).
        max_iterations: Maximum iterations.

    Returns:
        FitResult with node ids and x, y multipliers.
    """
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    _validate_balanced_sequences(k_out, k_in, name="degree")

    n = len(k_out)
    max_degree = float(n if self_loops else max(n - 1, 0))
    if np.any(k_out > max_degree) or np.any(k_in > max_degree):
        msg = "fixed-degree fitting received infeasible degree constraints"
        raise ValueError(msg)
    if np.any(k_out == max_degree) or np.any(k_in == max_degree):
        msg = (
            "fixed-degree fitting received boundary degree constraints; "
            "degrees equal to the candidate-pair capacity require divergent "
            "multipliers"
        )
        raise ValueError(msg)

    if k_out.sum() == 0:
        return FitResult(
            node=np.arange(n, dtype=np.uint64),
            x=np.zeros(n),
            y=np.zeros(n),
            self_loops=self_loops,
            family="bernoulli",
            diagnostics=OptimizationDiagnostics(
                converged=True, status="solved", iterations=0
            ),
        )

    t0 = time.perf_counter()
    x_list, y_list, converged, iters = _odme.fit_degree_bernoulli(
        k_out.tolist(), k_in.tolist(), self_loops, tolerance, max_iterations
    )
    result = FitResult(
        node=np.arange(n, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="bernoulli",
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )
    _log_fit_result(
        "fit_degree_bernoulli", converged, iters, time.perf_counter() - t0, verbose
    )
    return result


def fit_strength_binomial(
    strength_out: NDArray[np.integer],
    strength_in: NDArray[np.integer],
    *,
    layers: int = 1,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 10000,
) -> FitResult:
    """Fit Lagrange multipliers for the binomial(M) fixed-strength model.

    E[t_ij] = M * x_i*y_j / (1 + x_i*y_j).

    Args:
        strength_out: Outgoing strength per node.
        strength_in: Incoming strength per node.
        layers: Number of binary layers M.
        self_loops: Whether self loops are allowed.
        tolerance: Convergence tolerance.
        verbose: Logging level.
        max_iterations: Maximum iterations.

    Returns:
        FitResult with node ids and x, y multipliers.
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    _validate_balanced_sequences(s_out, s_in, name="strength")
    _validate_binomial_feasibility(s_out, s_in, layers, self_loops=self_loops)
    n = len(s_out)
    t0 = time.perf_counter()
    x_list, y_list, converged, iters = _odme.fit_strength_binomial(
        s_out.tolist(), s_in.tolist(), layers, self_loops, tolerance, max_iterations
    )
    _log_fit_result(
        "fit_strength_binomial",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return FitResult(
        node=np.arange(n, dtype=np.uint64),
        x=np.array(x_list),
        y=np.array(y_list),
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="binomial",
        layers=layers,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )


def fit_strength_cost_binomial(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    cost_sources: NDArray[np.integer],
    cost_targets: NDArray[np.integer],
    cost_values: NDArray[np.floating],
    target_cost: float,
    *,
    layers: int = 1,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    verbose: int = 0,
    max_iterations: int = 5000,
) -> StrengthCostFit:
    """Fit the strength-cost binomial(M) model."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    _validate_balanced_sequences(s_out, s_in, name="strength")
    c_src_arr, c_tgt_arr, c_val_arr = _validate_cost_entries(
        cost_sources, cost_targets, cost_values, target_cost=target_cost
    )
    c_src = c_src_arr.tolist()
    c_tgt = c_tgt_arr.tolist()
    c_val = c_val_arr.tolist()
    t0 = time.perf_counter()
    x_list, y_list, gamma, converged, iters = _odme.fit_strength_cost_binomial(
        s_out.tolist(),
        s_in.tolist(),
        c_src,
        c_tgt,
        c_val,
        target_cost,
        layers,
        self_loops,
        tolerance,
        max_iterations,
    )
    n = len(s_out)
    result = StrengthCostFit(
        node=np.arange(n, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        gamma=gamma,
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="binomial",
        layers=layers,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )
    _log_fit_result(
        "fit_strength_cost_binomial",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def fit_strength_edges_binomial(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    target_edges: float,
    *,
    layers: int = 1,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    verbose: int = 0,
    max_iterations: int = 50000,
) -> StrengthEdgesFit:
    """Fit the strength-edges binomial(M) model."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    _validate_strength_edges_constraints(
        s_out, s_in, target_edges, self_loops=self_loops
    )
    _validate_binomial_feasibility(s_out, s_in, layers, self_loops=self_loops)
    t0 = time.perf_counter()
    x_list, y_list, lam, converged, iters = _odme.fit_strength_edges_poisson(
        s_out.tolist(),
        s_in.tolist(),
        target_edges,
        self_loops,
        tolerance,
        max_iterations,
    )
    result = StrengthEdgesFit(
        node=np.arange(len(s_out), dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        lam=float(lam),
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="binomial",
        layers=layers,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )
    _log_fit_result(
        "fit_strength_edges_binomial",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def fit_strength_degree_binomial(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    *,
    layers: int = 1,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    verbose: int = 0,
    max_iterations: int = 50000,
) -> StrengthDegreeFit:
    """Fit the strength-degree binomial(M) model."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    validate_strength_degree_constraints(s_out, s_in, k_out, k_in)
    _validate_binomial_feasibility(s_out, s_in, layers, self_loops=self_loops)
    t0 = time.perf_counter()
    x_list, y_list, z_list, w_list, converged, iters = (
        _odme.fit_strength_degree_poisson(
            s_out.tolist(),
            s_in.tolist(),
            k_out.tolist(),
            k_in.tolist(),
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    n = len(s_out)
    result = StrengthDegreeFit(
        node=np.arange(n, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        z=np.asarray(z_list, dtype=np.float64),
        w=np.asarray(w_list, dtype=np.float64),
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="binomial",
        layers=layers,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )
    _log_fit_result(
        "fit_strength_degree_binomial",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return result


def fit_degree_events_poisson(
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    total_events: int,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    verbose: int = 0,
    max_iterations: int = 50000,
) -> DegreeEventsFit:
    """Fit the ME degree-events Poisson model."""
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    _validate_degree_events_constraints(
        k_out, k_in, total_events, self_loops=self_loops
    )
    n = len(k_out)
    e = float(k_out.sum())
    t = float(total_events)
    t0 = time.perf_counter()
    x_list, y_list, converged, iters = _odme.fit_degree_bernoulli(
        k_out.tolist(), k_in.tolist(), self_loops, tolerance, max_iterations
    )
    positive_mean = t / e if e > 0 else 1.0
    q = 1.0 - e / t if t > e else 0.0
    _log_fit_result(
        "fit_degree_events_poisson", converged, iters, time.perf_counter() - t0, verbose
    )
    return DegreeEventsFit(
        node=np.arange(n, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        q=q,
        positive_mean=positive_mean,
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="poisson",
        layers=None,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )


def fit_degree_events_binomial(
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    total_events: int,
    *,
    layers: int = 1,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    verbose: int = 0,
    max_iterations: int = 50000,
) -> DegreeEventsFit:
    """Fit the B degree-events binomial(M) model."""
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    _validate_degree_events_constraints(
        k_out, k_in, total_events, self_loops=self_loops
    )
    n = len(k_out)
    e = float(k_out.sum())
    t = float(total_events)
    t0 = time.perf_counter()
    x_list, y_list, converged, iters = _odme.fit_degree_bernoulli(
        k_out.tolist(), k_in.tolist(), self_loops, tolerance, max_iterations
    )
    positive_mean = t / e if e > 0 else 1.0
    q = 1.0 - e / t if t > e else 0.0
    _log_fit_result(
        "fit_degree_events_binomial",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return DegreeEventsFit(
        node=np.arange(n, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        q=q,
        positive_mean=positive_mean,
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="binomial",
        layers=layers,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )


def fit_degree_events_geometric(
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    total_events: int,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 10000,
) -> DegreeEventsFit:
    """Fit the W degree-events geometric model.

    Decomposes into:
    1. Solve q from positive geometric mean = T/E (analytic: q = 1 - E/T).
    2. Fit occupation via standard Bernoulli degree IPF (in Rust).

    Args:
        degree_out: Outgoing degree per node.
        degree_in: Incoming degree per node.
        total_events: Total weight T.
        self_loops: Whether self-loops are allowed.
        tolerance: Convergence tolerance for IPF.
        max_iterations: Maximum IPF iterations.
        verbose: Logging level.

    Returns:
        DegreeEventsFit with x, y, q, and positive_mean.
    """
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    _validate_degree_events_constraints(
        k_out, k_in, total_events, self_loops=self_loops
    )
    n = len(k_out)
    t0 = time.perf_counter()
    x_list, y_list, q, positive_mean, converged, iters = (
        _odme.fit_degree_events_geometric(
            k_out.tolist(),
            k_in.tolist(),
            int(total_events),
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    _log_fit_result(
        "fit_degree_events_geometric",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return DegreeEventsFit(
        node=np.arange(n, dtype=np.uint64),
        x=np.array(x_list),
        y=np.array(y_list),
        q=q,
        positive_mean=positive_mean,
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="geometric",
        layers=1,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )


def fit_degree_events_negative_binomial(
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    total_events: int,
    *,
    layers: int = 3,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    verbose: int = 0,
    max_iterations: int = 10000,
) -> DegreeEventsFit:
    """Fit the W degree-events negative binomial(M) model.

    Decomposes into:
    1. Solve q from positive negative binomial(M) mean = T/E via bisection (in Rust).
    2. Fit occupation via standard Bernoulli degree IPF (in Rust).

    Args:
        degree_out: Outgoing degree per node.
        degree_in: Incoming degree per node.
        total_events: Total weight T.
        layers: Number of negative binomial layers M.
        self_loops: Whether self-loops are allowed.
        tolerance: Convergence tolerance for IPF.
        max_iterations: Maximum IPF iterations.
        verbose: Logging level.

    Returns:
        DegreeEventsFit with x, y, q, and positive_mean.
    """
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    if layers <= 1:
        msg = "negative binomial W fitting requires layers > 1; use geometric for M = 1"
        raise ValueError(msg)
    _validate_degree_events_constraints(
        k_out, k_in, total_events, self_loops=self_loops
    )
    n = len(k_out)
    t0 = time.perf_counter()
    x_list, y_list, q, positive_mean, converged, iters = (
        _odme.fit_degree_events_negative_binomial(
            k_out.tolist(),
            k_in.tolist(),
            int(total_events),
            layers,
            self_loops,
            tolerance,
            max_iterations,
        )
    )
    _log_fit_result(
        "fit_degree_events_negative_binomial",
        converged,
        iters,
        time.perf_counter() - t0,
        verbose,
    )
    return DegreeEventsFit(
        node=np.arange(n, dtype=np.uint64),
        x=np.array(x_list),
        y=np.array(y_list),
        q=q,
        positive_mean=positive_mean,
        self_loops=self_loops,
        converged=converged,
        iterations=iters,
        family="negative_binomial",
        layers=layers,
        diagnostics=OptimizationDiagnostics(
            converged=converged,
            status="solved" if converged else "inaccurate",
            iterations=iters,
        ),
    )


__all__ = [
    "ConicDiagnostics",
    "DegreeEventsFit",
    "FitResult",
    "OptimizationDiagnostics",
    "StrengthCostFit",
    "StrengthDegreeFit",
    "StrengthEdgesFit",
    "fit_degree_bernoulli",
    "fit_degree_events_binomial",
    "fit_degree_events_geometric",
    "fit_degree_events_negative_binomial",
    "fit_degree_events_poisson",
    "fit_strength_binomial",
    "fit_strength_cost_binomial",
    "fit_strength_cost_binomial_coordinates",
    "fit_strength_cost_geometric",
    "fit_strength_cost_geometric_coordinates",
    "fit_strength_cost_negative_binomial",
    "fit_strength_cost_negative_binomial_coordinates",
    "fit_strength_cost_poisson",
    "fit_strength_cost_poisson_coordinates",
    "fit_strength_degree_binomial",
    "fit_strength_degree_geometric",
    "fit_strength_degree_negative_binomial",
    "fit_strength_degree_poisson",
    "fit_strength_edges_binomial",
    "fit_strength_edges_geometric",
    "fit_strength_edges_negative_binomial",
    "fit_strength_edges_poisson",
    "fit_strength_geometric",
    "fit_strength_negative_binomial",
    "fit_strength_poisson",
    "validate_strength_degree_constraints",
]
