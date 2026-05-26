"""Coordinate-based strength-cost fitting tests."""

from __future__ import annotations

import numpy as np

from menobis.models import (
    fit_strength_cost_binomial_coordinates,
    fit_strength_cost_geometric_coordinates,
    fit_strength_cost_negative_binomial_coordinates,
    fit_strength_cost_poisson,
    fit_strength_cost_poisson_coordinates,
    fit_strength_poisson,
)


def _fixture_me() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    strength_out = np.array([10.0, 20.0, 30.0])
    strength_in = np.array([15.0, 25.0, 20.0])
    x = np.array([0.0, 3.0, 0.0])
    y = np.array([0.0, 0.0, 4.0])
    base = fit_strength_poisson(strength_out, strength_in)
    coords = np.column_stack([x, y])
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    target_cost = float(np.sum(np.outer(base.x, base.y) * distance))
    return strength_out, strength_in, x, y, target_cost


def test_coordinate_strength_cost_matches_complete_sparse_costs() -> None:
    """Projected XY coordinate costs match equivalent complete sparse costs."""
    strength_out, strength_in, x, y, target_cost = _fixture_me()
    coords = np.column_stack([x, y])
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    src, tgt = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")

    sparse = fit_strength_cost_poisson(
        strength_out,
        strength_in,
        src.ravel(),
        tgt.ravel(),
        distance.ravel(),
        target_cost,
        tolerance=1e-6,
    )
    coord = fit_strength_cost_poisson_coordinates(
        strength_out,
        strength_in,
        x,
        y,
        target_cost,
        tolerance=1e-6,
    )

    assert sparse.converged
    assert coord.converged
    np.testing.assert_allclose(coord.gamma, sparse.gamma, atol=1e-5)


def test_all_families_converge_and_differ() -> None:
    """ME, B, W-geometric, W-NB produce different gamma on same input."""
    # Use feasible strengths for B(M=10): max s_out_i < M*n = 30
    strength_out = np.array([2.0, 3.0, 4.0])
    strength_in = np.array([2.5, 3.5, 3.0])
    x = np.array([0.0, 3.0, 0.0])
    y = np.array([0.0, 0.0, 4.0])
    target_cost = 15.0

    me = fit_strength_cost_poisson_coordinates(
        strength_out, strength_in, x, y, target_cost, tolerance=1e-4
    )
    b = fit_strength_cost_binomial_coordinates(
        strength_out, strength_in, x, y, target_cost, layers=3, tolerance=1e-4
    )
    wg = fit_strength_cost_geometric_coordinates(
        strength_out, strength_in, x, y, target_cost, tolerance=1e-4
    )
    wnb = fit_strength_cost_negative_binomial_coordinates(
        strength_out, strength_in, x, y, target_cost, layers=3, tolerance=1e-4
    )

    assert me.converged, "ME did not converge"
    assert b.converged, "B did not converge"
    assert wg.converged, "W geometric did not converge"
    assert wnb.converged, "W negative_binomial did not converge"

    # Families must produce different gamma (different E[t_ij] formulas)
    assert abs(me.gamma - b.gamma) > 1e-5, f"ME==B: {me.gamma}"
    assert abs(me.gamma - wg.gamma) > 1e-5, f"ME==Wg: {me.gamma}"
    assert abs(b.gamma - wg.gamma) > 1e-5, f"B==Wg: {b.gamma}"

    assert [fit.family for fit in [me, b, wg, wnb]] == [
        "poisson",
        "binomial",
        "geometric",
        "negative_binomial",
    ]
