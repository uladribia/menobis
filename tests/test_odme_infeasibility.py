"""Tests for infeasibility rejection and convergence diagnostics."""

import numpy as np
import pytest

from odme.data.frames import normalize_probabilities
from odme.models import (
    fit_fixed_degree_binary,
    fit_fixed_strength_me,
    fit_strength_degree_me,
    fit_strength_edges_me,
    validate_strength_degree_constraints,
)


def test_strength_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        fit_fixed_strength_me(np.array([-1.0]), np.array([1.0]))


def test_strength_rejects_unbalanced() -> None:
    with pytest.raises(ValueError, match="balanced"):
        fit_fixed_strength_me(np.array([10.0]), np.array([20.0]))


def test_strength_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_fixed_strength_me(np.array([10.0, 20.0]), np.array([30.0]))


def test_degree_rejects_unbalanced() -> None:
    with pytest.raises(ValueError, match="balanced"):
        fit_fixed_degree_binary(np.array([1.0]), np.array([0.5]))


def test_degree_rejects_infeasible() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        fit_fixed_degree_binary(np.array([5.0, 5.0]), np.array([5.0, 5.0]))


def test_strength_edges_rejects_zero_edges() -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_strength_edges_me(np.array([10.0, 10.0]), np.array([10.0, 10.0]), 0.0)


def test_strength_edges_rejects_edges_above_total() -> None:
    with pytest.raises(ValueError, match="no larger"):
        fit_strength_edges_me(np.array([10.0, 10.0]), np.array([10.0, 10.0]), 30.0)


def test_strength_degree_rejects_strength_below_degree() -> None:
    with pytest.raises(ValueError, match=r"strength.*degree"):
        fit_strength_degree_me(
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
    r = fit_fixed_strength_me(np.array([10, 20]), np.array([15, 15]))
    assert r.converged is True
    assert r.iterations >= 0


def test_degree_fit_has_convergence_fields() -> None:
    r = fit_fixed_degree_binary(np.array([0.8, 1.2]), np.array([1.0, 1.0]))
    assert r.converged is True
    assert r.iterations >= 1


def test_strength_edges_fit_has_convergence_fields() -> None:
    r = fit_strength_edges_me(np.array([10.0, 10.0]), np.array([10.0, 10.0]), 3.0)
    assert r.converged is True
    assert r.iterations >= 1


def test_strength_degree_fit_has_convergence_fields() -> None:
    r = fit_strength_degree_me(
        np.array([2.0, 3.0]),
        np.array([2.5, 2.5]),
        np.array([0.8, 1.2]),
        np.array([1.0, 1.0]),
    )
    assert r.converged is True
    assert r.iterations >= 1
