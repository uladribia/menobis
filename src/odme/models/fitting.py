"""Lagrange multiplier fitting for ODME maximum-entropy models."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme


@dataclass(frozen=True)
class StrengthDegreeZipFit:
    """Fitted grand-canonical zero-inflated shifted-Poisson model."""

    node: NDArray[np.uint64]
    degree_x: NDArray[np.float64]
    degree_y: NDArray[np.float64]
    excess_x: NDArray[np.float64]
    excess_y: NDArray[np.float64]
    self_loops: bool


@dataclass(frozen=True)
class FitResult:
    """Lagrange multiplier fitting result."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]


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


def fit_strength_degree_zip(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 50000,
) -> StrengthDegreeZipFit:
    """Fit grand-canonical zero-inflated shifted-Poisson constraints.

    Edge presence follows the fixed-degree binary model. Conditional on edge
    presence, the weight is ``1 + Poisson(lambda_ij)`` with
    ``lambda_ij = excess_x_i * excess_y_j``. Therefore
    ``E[t_ij] = p_ij * (1 + lambda_ij)``.

    Args:
        strength_out: Expected outgoing strength per node.
        strength_in: Expected incoming strength per node.
        degree_out: Expected outgoing binary degree per node.
        degree_in: Expected incoming binary degree per node.
        self_loops: Whether self loops are allowed.
        tolerance: Solver tolerance.
        max_iterations: Maximum solver iterations.

    Returns:
        StrengthDegreeZipFit with degree and excess multipliers.
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    validate_strength_degree_constraints(s_out, s_in, k_out, k_in)

    degree_fit = fit_fixed_degree_binary(
        k_out,
        k_in,
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    excess_out = s_out - k_out
    excess_in = s_in - k_in
    x_list, y_list, _converged, _iters = _odme.fit_weighted_factors(
        excess_out.tolist(),
        excess_in.tolist(),
        degree_fit.x.tolist(),
        degree_fit.y.tolist(),
        self_loops,
        tolerance,
        max_iterations,
    )
    n = len(s_out)
    return StrengthDegreeZipFit(
        node=np.arange(n, dtype=np.uint64),
        degree_x=degree_fit.x,
        degree_y=degree_fit.y,
        excess_x=np.asarray(x_list, dtype=np.float64),
        excess_y=np.asarray(y_list, dtype=np.float64),
        self_loops=self_loops,
    )


def fit_fixed_strength_me(
    strength_out: NDArray[np.integer],
    strength_in: NDArray[np.integer],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> FitResult:
    """Fit Lagrange multipliers for the directed ME fixed-strength model.

    Args:
        strength_out: Outgoing strength per node.
        strength_in: Incoming strength per node.
        self_loops: Whether self loops are allowed.
        tolerance: Convergence tolerance for iterative balancing.
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
        x_list, y_list, _converged, _iters = _odme.fit_balance_no_self_loops(
            s_out.tolist(), s_in.tolist(), tolerance, max_iterations
        )
        x = np.array(x_list)
        y = np.array(y_list)

    return FitResult(
        node=np.arange(n, dtype=np.uint64),
        x=x,
        y=y,
    )


def fit_fixed_degree_binary(
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
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

    x_list, y_list, _converged, _iters = _odme.fit_binary_degrees(
        k_out.tolist(), k_in.tolist(), self_loops, tolerance, max_iterations
    )
    return FitResult(
        node=np.arange(n, dtype=np.uint64),
        x=np.asarray(x_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.float64),
    )


__all__ = [
    "FitResult",
    "StrengthDegreeZipFit",
    "fit_fixed_degree_binary",
    "fit_fixed_strength_me",
    "fit_strength_degree_zip",
    "validate_strength_degree_constraints",
]
