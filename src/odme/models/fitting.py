"""Lagrange multiplier fitting for ODME multi-edge models."""

import time
import warnings
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

import odme._odme as _odme


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


@dataclass(frozen=True)
class StrengthCostFit:
    """Fitted strength-cost ME model: E[t_ij] = x_i y_j exp(-gamma d_ij)."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    gamma: float
    self_loops: bool
    converged: bool
    iterations: int


@dataclass(frozen=True)
class StrengthEdgesFit:
    """Fitted exact ME fixed-strength-and-edge-count ME model."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    lam: float
    self_loops: bool
    converged: bool
    iterations: int


@dataclass(frozen=True)
class StrengthDegreeFit:
    """Fitted exact ME fixed-strength-degree ME model."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    w: NDArray[np.float64]
    self_loops: bool
    converged: bool
    iterations: int


@dataclass(frozen=True)
class FitResult:
    """Lagrange multiplier fitting result."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    converged: bool = True
    iterations: int = 0


def _validate_balanced_sequences(
    out_sequence: NDArray[np.float64],
    in_sequence: NDArray[np.float64],
    *,
    name: str,
) -> None:
    if len(out_sequence) != len(in_sequence):
        msg = f"{name}_out and {name}_in must have the same length"
        raise ValueError(msg)
    if np.any(out_sequence < 0) or np.any(in_sequence < 0):
        msg = f"fixed-{name} fitting requires non-negative sequences"
        raise ValueError(msg)
    if not np.isclose(out_sequence.sum(), in_sequence.sum()):
        msg = f"fixed-{name} fitting requires balanced in/out sequences"
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
    if target_cost <= 0.0:
        msg = "target_cost must be positive"
        raise ValueError(msg)

    c_src = np.asarray(cost_sources, dtype=np.int64).tolist()
    c_tgt = np.asarray(cost_targets, dtype=np.int64).tolist()
    c_val = np.asarray(cost_values, dtype=np.float64).tolist()

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
    )
    _log_fit_result(
        "fit_strength_cost_poisson", converged, iters, time.perf_counter() - t0, verbose
    )
    return result


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
    _validate_balanced_sequences(s_out, s_in, name="strength")
    if target_edges <= 0.0 or target_edges > s_out.sum():
        msg = "target_edges must be positive and no larger than total strength"
        raise ValueError(msg)
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
    )
    _log_fit_result(
        "fit_strength_edges_poisson",
        converged,
        iters,
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

    total_out = s_out.sum()
    n = len(s_out)
    if total_out == 0:
        return FitResult(
            node=np.arange(n, dtype=np.uint64),
            x=np.zeros(n),
            y=np.zeros(n),
        )

    if self_loops:
        sqrt_t = np.sqrt(total_out)
        x = s_out / sqrt_t
        y = s_in / sqrt_t
    else:
        t0 = time.perf_counter()
        x_list, y_list, converged, iters = _odme.fit_strength_poisson_no_self_loops(
            s_out.tolist(), s_in.tolist(), tolerance, max_iterations
        )
        x = np.array(x_list)
        y = np.array(y_list)
        _log_fit_result(
            "fit_strength_poisson",
            converged,
            iters,
            time.perf_counter() - t0,
            verbose,
        )

    return FitResult(
        node=np.arange(n, dtype=np.uint64),
        x=x,
        y=y,
        converged=converged if not self_loops else True,
        iterations=iters if not self_loops else 0,
    )


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

    if k_out.sum() == 0:
        return FitResult(
            node=np.arange(n, dtype=np.uint64),
            x=np.zeros(n),
            y=np.zeros(n),
        )

    t0 = time.perf_counter()
    x_list, y_list, converged, iters = _odme.fit_degree_bernoulli(
        k_out.tolist(), k_in.tolist(), self_loops, tolerance, max_iterations
    )
    result = FitResult(
        node=np.arange(n, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
        converged=converged,
        iterations=iters,
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
        converged=converged,
        iterations=iters,
    )


__all__ = [
    "FitResult",
    "StrengthCostFit",
    "StrengthDegreeFit",
    "StrengthEdgesFit",
    "fit_degree_bernoulli",
    "fit_strength_binomial",
    "fit_strength_cost_poisson",
    "fit_strength_degree_poisson",
    "fit_strength_edges_poisson",
    "fit_strength_poisson",
    "validate_strength_degree_constraints",
]
