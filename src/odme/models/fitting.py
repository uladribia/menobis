"""Lagrange multiplier fitting for ODME maximum-entropy models."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme


@dataclass(frozen=True)
class FitResult:
    """Lagrange multiplier fitting result."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]


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

    if len(s_out) != len(s_in):
        msg = "strength_out and strength_in must have the same length"
        raise ValueError(msg)

    total_out = s_out.sum()
    total_in = s_in.sum()
    if not np.isclose(total_out, total_in):
        msg = "fixed-strength fitting requires balanced in/out strengths"
        raise ValueError(msg)

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


__all__ = ["FitResult", "fit_fixed_strength_me"]
