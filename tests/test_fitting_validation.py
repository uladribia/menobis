"""Input validation tests for all fitting functions.

Tests verify that fitting wrappers reject invalid inputs at the boundary:
NaN/Inf, unbalanced sequences, capacity overflow, bad coordinates, etc.
"""

import numpy as np
import pytest

from menobis.models import (
    fit_strength_binomial,
    fit_strength_cost_geometric,
    fit_strength_cost_poisson,
    fit_strength_edges_binomial,
    fit_strength_edges_geometric,
    fit_strength_edges_poisson,
    fit_strength_geometric,
    fit_strength_poisson,
    validate_strength_degree_constraints,
)


class TestBalancedSequenceValidation:
    """All fitting functions reject unbalanced strength/degree sequences."""

    def test_me_strength_rejects_unbalanced(self) -> None:
        with pytest.raises(ValueError, match="balanced"):
            fit_strength_poisson(np.array([10.0]), np.array([20.0]))

    def test_w_strength_rejects_unbalanced(self) -> None:
        with pytest.raises(ValueError, match="balanced"):
            fit_strength_geometric(np.array([10.0]), np.array([20.0]))

    def test_b_strength_rejects_unbalanced(self) -> None:
        with pytest.raises(ValueError, match="balanced"):
            fit_strength_binomial(np.array([10.0]), np.array([20.0]), layers=5)


class TestNonFiniteRejection:
    """All fitting wrappers reject non-finite constraint sequences."""

    def test_nan_rejected_in_strength(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            fit_strength_geometric(np.array([1.0, np.nan]), np.array([1.0, np.nan]))

    def test_inf_rejected_in_strength(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            fit_strength_poisson(np.array([1.0, np.inf]), np.array([1.0, np.inf]))


class TestCapacityValidation:
    """Strength-edges fits reject target_edges exceeding pair capacity."""

    def test_me_edges_at_capacity(self) -> None:
        s = np.array([10.0, 10.0])
        with pytest.raises(ValueError, match="candidate-pair capacity"):
            fit_strength_edges_poisson(s, s, target_edges=4.0, self_loops=True)

    def test_w_edges_at_capacity(self) -> None:
        s = np.array([10.0, 10.0])
        with pytest.raises(ValueError, match="candidate-pair capacity"):
            fit_strength_edges_geometric(s, s, target_edges=4.0, self_loops=True)

    def test_b_edges_at_capacity(self) -> None:
        s = np.array([10.0, 10.0])
        with pytest.raises(ValueError, match="candidate-pair capacity"):
            fit_strength_edges_binomial(
                s, s, target_edges=4.0, layers=5, self_loops=True
            )

    def test_no_self_loops_reduces_capacity(self) -> None:
        s = np.array([5.0, 5.0, 5.0])
        # N=3, no self-loops → 6 pairs max
        with pytest.raises(ValueError, match="candidate-pair capacity"):
            fit_strength_edges_poisson(s, s, target_edges=6.0, self_loops=False)

    def test_nonfinite_target_edges_rejected(self) -> None:
        s = np.array([5.0, 5.0])
        with pytest.raises(ValueError, match="finite"):
            fit_strength_edges_geometric(
                s, s, target_edges=float("inf"), self_loops=True
            )
        with pytest.raises(ValueError, match="finite"):
            fit_strength_edges_geometric(
                s, s, target_edges=float("nan"), self_loops=True
            )

    def test_target_edges_above_total_strength(self) -> None:
        s = np.array([2.0, 2.0])
        with pytest.raises(ValueError, match="no larger"):
            fit_strength_edges_geometric(s, s, target_edges=5.0, self_loops=True)


class TestCoordinateValidation:
    """Strength-cost fits reject mismatched coordinate arrays."""

    @pytest.mark.parametrize(
        "fit_func", [fit_strength_cost_poisson, fit_strength_cost_geometric]
    )
    def test_mismatched_coordinate_length(self, fit_func) -> None:
        s_out = np.array([1.0, 1.0])
        s_in = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="same length"):
            fit_func(s_out, s_in, np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2]), 1.0)


class TestStrengthDegreeConstraintValidation:
    """validate_strength_degree_constraints rejects infeasible inputs."""

    def test_accepts_valid_sequences(self) -> None:
        validate_strength_degree_constraints(
            np.array([2.0, 1.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        )

    def test_rejects_strength_below_degree(self) -> None:
        with pytest.raises(ValueError, match=r"strength.*degree"):
            validate_strength_degree_constraints(
                np.array([0.5, 1.0]),
                np.array([1.0, 0.5]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
            )
