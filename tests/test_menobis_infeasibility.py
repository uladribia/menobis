"""Tests for infeasibility rejection and convergence diagnostics."""

import numpy as np
import pytest

from menobis.data.frames import normalize_probabilities
from menobis.models import (
    fit_degree_bernoulli,
    fit_strength_degree_poisson,
    fit_strength_edges_poisson,
    fit_strength_poisson,
    validate_strength_degree_constraints,
)


def test_strength_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        fit_strength_poisson(np.array([-1.0]), np.array([1.0]))


def test_strength_rejects_unbalanced() -> None:
    with pytest.raises(ValueError, match="balanced"):
        fit_strength_poisson(np.array([10.0]), np.array([20.0]))


def test_strength_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_strength_poisson(np.array([10.0, 20.0]), np.array([30.0]))


def test_degree_rejects_unbalanced() -> None:
    with pytest.raises(ValueError, match="balanced"):
        fit_degree_bernoulli(np.array([1.0]), np.array([0.5]))


def test_degree_rejects_infeasible() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        fit_degree_bernoulli(np.array([5.0, 5.0]), np.array([5.0, 5.0]))


def test_degree_rejects_boundary_capacity_without_self_loops() -> None:
    with pytest.raises(ValueError, match="boundary"):
        fit_degree_bernoulli(
            np.array([2.0, 1.0, 1.0]),
            np.array([1.0, 2.0, 1.0]),
            self_loops=False,
        )


def test_strength_degree_handles_boundary_strength_equal_degree() -> None:
    """Boundary s==k case converges (possibly slowly) without crashing."""
    result = fit_strength_degree_poisson(
        np.array([1.0, 3.0]),
        np.array([2.0, 2.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
    )
    # May or may not converge, but must not crash
    assert result.x is not None
    assert result.y is not None


def test_strength_edges_rejects_zero_edges() -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_strength_edges_poisson(np.array([10.0, 10.0]), np.array([10.0, 10.0]), 0.0)


def test_strength_edges_rejects_edges_above_total() -> None:
    with pytest.raises(ValueError, match="no larger"):
        fit_strength_edges_poisson(np.array([10.0, 10.0]), np.array([10.0, 10.0]), 30.0)


def test_strength_degree_rejects_strength_below_degree() -> None:
    with pytest.raises(ValueError, match=r"strength.*degree"):
        fit_strength_degree_poisson(
            np.array([0.5, 1.0]),
            np.array([1.0, 0.5]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        )


def test_strength_degree_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        validate_strength_degree_constraints(
            np.array([2.0]),
            np.array([2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        )


def test_probabilities_reject_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        normalize_probabilities(
            np.array([0, 0]), np.array([1, 1]), np.array([0.3, 0.4])
        )


def test_probabilities_reject_out_of_range() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        normalize_probabilities(np.array([0]), np.array([1]), np.array([1.5]))


def test_fit_result_has_convergence_fields() -> None:
    r = fit_strength_poisson(np.array([10, 20]), np.array([15, 15]))
    assert r.converged is True
    assert r.iterations >= 0


def test_degree_fit_has_convergence_fields() -> None:
    r = fit_degree_bernoulli(np.array([0.8, 1.2]), np.array([1.0, 1.0]))
    assert r.converged is True
    assert r.iterations >= 1


def test_strength_edges_fit_has_convergence_fields() -> None:
    r = fit_strength_edges_poisson(np.array([10.0, 10.0]), np.array([10.0, 10.0]), 3.0)
    assert r.converged is True
    assert r.iterations >= 1


def test_strength_degree_fit_has_convergence_fields() -> None:
    r = fit_strength_degree_poisson(
        np.array([2.0, 3.0]),
        np.array([2.5, 2.5]),
        np.array([0.8, 1.2]),
        np.array([1.0, 1.0]),
    )
    assert r.converged is True
    assert r.iterations >= 1
