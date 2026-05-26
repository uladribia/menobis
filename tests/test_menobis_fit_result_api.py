"""Tests for clean, constraint-oriented fitting result APIs."""

import numpy as np

from menobis.models import (
    StrengthCostFit,
    StrengthFit,
    fit_strength_cost_geometric,
    fit_strength_cost_poisson,
    fit_strength_geometric,
)


def test_strength_fit_uses_constraint_type_with_w_diagnostics() -> None:
    """W strength fitting returns the shared strength fit type plus diagnostics."""
    result = fit_strength_geometric(
        np.array([2.0, 2.0]),
        np.array([2.0, 2.0]),
        self_loops=True,
    )

    assert isinstance(result, StrengthFit)
    assert result.family == "geometric"
    assert result.layers == 1
    assert result.diagnostics is not None
    assert result.diagnostics.conic is not None
    assert result.diagnostics.conic.max_q < 1.0


def test_strength_cost_fits_share_public_shape() -> None:
    """ME and W strength-cost fits expose the same core public fields."""
    s_out = np.array([1.0, 1.0])
    s_in = np.array([1.0, 1.0])
    cost_src = np.array([0, 1], dtype=np.uint64)
    cost_tgt = np.array([0, 1], dtype=np.uint64)
    cost_val = np.array([1.0, 1.0])

    me_fit = fit_strength_cost_poisson(
        s_out, s_in, cost_src, cost_tgt, cost_val, target_cost=1.0
    )
    w_fit = fit_strength_cost_geometric(
        s_out, s_in, cost_src, cost_tgt, cost_val, target_cost=1.0
    )

    assert isinstance(me_fit, StrengthCostFit)
    assert isinstance(w_fit, StrengthCostFit)
    assert me_fit.family == "poisson"
    assert w_fit.family == "geometric"
    assert me_fit.diagnostics is not None
    assert w_fit.diagnostics is not None
    assert me_fit.diagnostics.conic is None
    assert w_fit.diagnostics.conic is not None
