"""Tests for W fixed-strength-plus-cost fitting wrappers."""

import numpy as np
import pytest


def _expected_strength_cost(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
    cost_src: np.ndarray,
    cost_tgt: np.ndarray,
    cost_val: np.ndarray,
    layers: int,
    self_loops: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(x)
    costs = {
        (int(i), int(j)): float(d)
        for i, j, d in zip(cost_src, cost_tgt, cost_val, strict=True)
    }
    out = np.zeros(n)
    inc = np.zeros(n)
    total_cost = 0.0
    for i in range(n):
        for j in range(n):
            if not self_loops and i == j:
                continue
            d = costs.get((i, j), 0.0)
            q = x[i] * y[j] * np.exp(-gamma * d)
            mean = layers * q / (1.0 - q)
            out[i] += mean
            inc[j] += mean
            total_cost += d * mean
    return out, inc, total_cost


def test_fit_strength_cost_geometric_recovers_homogeneous_constraints() -> None:
    """Geometric W strength-cost fitting recovers strengths and target cost."""
    from odme.models.fitting import fit_strength_cost_geometric

    cost_src = np.array([0, 1], dtype=np.uint64)
    cost_tgt = np.array([0, 1], dtype=np.uint64)
    cost_val = np.array([1.0, 1.0])
    result = fit_strength_cost_geometric(
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        cost_src,
        cost_tgt,
        cost_val,
        target_cost=1.0,
        self_loops=True,
        tolerance=1e-8,
        max_iterations=500,
    )

    out, inc, cost = _expected_strength_cost(
        result.x, result.y, result.gamma, cost_src, cost_tgt, cost_val, 1, True
    )
    assert result.converged
    assert result.layers == 1
    assert np.max(np.abs(out - 1.0)) < 1e-4
    assert np.max(np.abs(inc - 1.0)) < 1e-4
    assert abs(cost - 1.0) < 1e-4
    assert result.max_q is not None and result.max_q < 1.0


def test_fit_strength_cost_negative_binomial_rejects_single_layer() -> None:
    """Negative-binomial strength-cost API reserves M=1 for geometric."""
    from odme.models.fitting import fit_strength_cost_negative_binomial

    with pytest.raises(ValueError, match="layers > 1"):
        fit_strength_cost_negative_binomial(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0], dtype=np.uint64),
            np.array([0], dtype=np.uint64),
            np.array([1.0]),
            target_cost=1.0,
            layers=1,
        )
