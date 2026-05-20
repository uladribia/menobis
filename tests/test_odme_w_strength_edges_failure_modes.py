"""W strength-edges solver convergence and failure-mode tests."""

import numpy as np
import pytest

from odme.models import (
    StrengthEdgesFit,
    fit_strength_edges_geometric,
    fit_strength_edges_negative_binomial,
)


def test_rejects_edges_at_capacity() -> None:
    """Strength-edges must reject target_edges >= candidate-pair capacity."""
    s = np.array([5.0, 5.0, 5.0])
    with pytest.raises(ValueError, match="candidate-pair capacity"):
        fit_strength_edges_geometric(s, s, target_edges=9.0, self_loops=True)
    with pytest.raises(ValueError, match="candidate-pair capacity"):
        fit_strength_edges_geometric(s, s, target_edges=6.0, self_loops=False)


def test_rejects_edges_above_total_strength() -> None:
    """Target edges cannot exceed total strength."""
    s = np.array([2.0, 2.0])
    with pytest.raises(ValueError, match="no larger"):
        fit_strength_edges_geometric(s, s, target_edges=5.0, self_loops=True)


def test_rejects_nonfinite_target_edges() -> None:
    """Non-finite target_edges should be rejected at the boundary."""
    s = np.array([5.0, 5.0])
    with pytest.raises(ValueError, match="finite"):
        fit_strength_edges_geometric(s, s, target_edges=float("inf"), self_loops=True)
    with pytest.raises(ValueError, match="finite"):
        fit_strength_edges_geometric(s, s, target_edges=float("nan"), self_loops=True)


def test_near_capacity_reports_status_not_silent_bad_fit() -> None:
    """When edges approach pair capacity, solver should not silently return NaN."""
    n = 5
    s = np.full(n, 20.0)
    capacity = n * n
    target = capacity - 1.0

    result = fit_strength_edges_geometric(
        s, s, target_edges=target, self_loops=True, tolerance=1e-6, max_iterations=500
    )
    assert isinstance(result, StrengthEdgesFit)
    if result.converged:
        assert not np.any(np.isnan(result.x))
        assert not np.any(np.isnan(result.y))
        assert np.isfinite(result.lam)
    else:
        assert result.status != "solved"


def test_very_sparse_edges_converges() -> None:
    """Very low edge count (high average weight) should still converge."""
    n = 5
    s = np.full(n, 10.0)
    target = 2.0

    result = fit_strength_edges_geometric(s, s, target_edges=target, self_loops=True)
    assert result.converged
    assert result.lam > 0.0
    assert result.max_q is not None and result.max_q < 1.0


def test_heterogeneous_pareto_no_self_loops() -> None:
    """Heterogeneous Pareto strengths without self-loops should converge."""
    rng = np.random.default_rng(42)
    n = 8
    raw = rng.pareto(2.0, n) + 1.0
    total = 40.0
    s_out = raw / raw.sum() * total
    s_in = np.roll(raw[::-1], 2) / raw.sum() * total
    target = 10.0
    capacity = n * (n - 1)
    assert target < capacity

    result = fit_strength_edges_geometric(
        s_out, s_in, target_edges=target, self_loops=False, tolerance=1e-7
    )
    assert isinstance(result, StrengthEdgesFit)
    if result.converged:
        assert not np.any(np.isnan(result.x))
        assert result.max_q is not None and result.max_q < 1.0


def test_negative_binomial_high_layers_converges() -> None:
    """Negative binomial with M=5 on moderate heterogeneity."""
    rng = np.random.default_rng(99)
    n = 6
    raw = rng.pareto(2.5, n) + 1.0
    total = 30.0
    s_out = raw / raw.sum() * total
    s_in = np.roll(raw[::-1], 1) / raw.sum() * total
    target = 8.0

    result = fit_strength_edges_negative_binomial(
        s_out, s_in, target_edges=target, layers=5, self_loops=True, tolerance=1e-7
    )
    assert isinstance(result, StrengthEdgesFit)
    assert result.converged
    assert result.lam > 0.0
    assert result.max_q is not None and result.max_q < 1.0


def test_solver_warns_on_nonconvergence() -> None:
    """If max_iterations is too low, the solver should warn and report status."""
    n = 8
    rng = np.random.default_rng(77)
    raw = rng.pareto(1.8, n) + 1.0
    total = 50.0
    s_out = raw / raw.sum() * total
    s_in = np.roll(raw[::-1], 3) / raw.sum() * total
    target = 12.0

    with pytest.warns(UserWarning, match="did not converge"):
        result = fit_strength_edges_geometric(
            s_out,
            s_in,
            target_edges=target,
            self_loops=True,
            tolerance=1e-12,
            max_iterations=2,
        )
    assert not result.converged
    assert result.status != "solved"
