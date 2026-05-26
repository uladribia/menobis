"""Saturation regression tests for mixed strength-degree models."""

import numpy as np

from menobis.models import (
    fit_strength_degree_geometric,
    fit_strength_degree_negative_binomial,
    fit_strength_degree_poisson,
)


def _saturated_constraints() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return feasible constraints with saturated out/in degree for N=3."""
    return (
        np.array([5.0, 2.0, 3.0]),
        np.array([3.0, 4.0, 3.0]),
        np.array([2.0, 1.0, 1.0]),
        np.array([1.0, 2.0, 1.0]),
    )


def test_me_strength_degree_converges_with_degree_saturation() -> None:
    """ME handles k=capacity by fixing occupation multipliers high."""
    s_out, s_in, k_out, k_in = _saturated_constraints()
    fit = fit_strength_degree_poisson(
        s_out, s_in, k_out, k_in, self_loops=False, tolerance=1e-6
    )

    assert fit.converged
    assert fit.z[0] > 1e20
    assert fit.w[1] > 1e20


def test_w_strength_degree_converges_with_degree_saturation() -> None:
    """Geometric W handles mixed strength plus saturated degree constraints."""
    s_out, s_in, k_out, k_in = _saturated_constraints()
    fit = fit_strength_degree_geometric(
        s_out, s_in, k_out, k_in, self_loops=False, tolerance=1e-6
    )

    assert fit.converged
    assert fit.z[0] > 1e20
    assert fit.w[1] > 1e20


def test_wnb_strength_degree_converges_with_degree_saturation() -> None:
    """Negative-binomial W handles mixed strength plus saturated degree constraints."""
    s_out, s_in, k_out, k_in = _saturated_constraints()
    fit = fit_strength_degree_negative_binomial(
        s_out, s_in, k_out, k_in, layers=3, self_loops=False, tolerance=1e-6
    )

    assert fit.converged
    assert fit.z[0] > 1e20
    assert fit.w[1] > 1e20
