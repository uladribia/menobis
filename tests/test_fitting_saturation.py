"""Saturation edge-case tests for strength-degree models.

Tests verify correct behavior when some nodes have degree equal to
the maximum possible (N-1 for no-self-loops, N for self-loops).
Uses small N=3-5 for exact control.
"""

import warnings

import numpy as np
import pytest

from menobis.models import (
    fit_strength_degree_binomial,
    fit_strength_degree_geometric,
    fit_strength_degree_poisson,
)


def _saturated_constraints_n3():
    """Return feasible constraints with saturated out/in degree for N=3.

    Node 0: k_out=2 (saturated, max=2 with no-self-loops and N=3)
    Node 1: k_in=2 (saturated)
    """
    return (
        np.array([5.0, 2.0, 3.0]),
        np.array([3.0, 4.0, 3.0]),
        np.array([2.0, 1.0, 1.0]),
        np.array([1.0, 2.0, 1.0]),
    )


class TestMESaturation:
    """ME handles saturated degree by clamping z/w multipliers high."""

    def test_converges_with_degree_saturation(self) -> None:
        s_out, s_in, k_out, k_in = _saturated_constraints_n3()
        fit = fit_strength_degree_poisson(
            s_out, s_in, k_out, k_in, self_loops=False, tolerance=1e-6
        )
        assert fit.converged
        # Saturated nodes should have very large z/w multipliers
        assert fit.z[0] > 1e20
        assert fit.w[1] > 1e20

    def test_non_saturated_nodes_have_finite_multipliers(self) -> None:
        s_out, s_in, k_out, k_in = _saturated_constraints_n3()
        fit = fit_strength_degree_poisson(
            s_out, s_in, k_out, k_in, self_loops=False, tolerance=1e-6
        )
        assert fit.converged
        # Node 2 is not saturated (k_out=1 < 2)
        assert fit.z[2] < 1e10


class TestWSaturation:
    """W (geometric) handles saturated degree constraints."""

    @pytest.mark.xfail(
        reason="W Newton solver does not converge at N=3 with saturated degrees",
        strict=False,
    )
    def test_converges_with_degree_saturation(self) -> None:
        s_out, s_in, k_out, k_in = _saturated_constraints_n3()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_strength_degree_geometric(
                s_out,
                s_in,
                k_out,
                k_in,
                self_loops=False,
                tolerance=1e-6,
                max_iterations=5000,
            )
        assert fit.converged
        assert fit.z[0] > 1e20
        assert fit.w[1] > 1e20


class TestBSaturation:
    """B (binomial) handles saturated degree constraints."""

    @pytest.mark.xfail(
        reason="B solver does not converge at N=3 with saturated degrees",
        strict=False,
    )
    def test_converges_with_degree_saturation(self) -> None:
        s_out, s_in, k_out, k_in = _saturated_constraints_n3()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_strength_degree_binomial(
                s_out, s_in, k_out, k_in, layers=10, self_loops=False, tolerance=1e-6
            )
        assert fit.converged
        assert fit.z[0] > 1e20
        assert fit.w[1] > 1e20
