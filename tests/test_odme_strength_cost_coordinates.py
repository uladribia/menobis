"""Coordinate-based strength-cost fitting tests."""

from __future__ import annotations

import numpy as np

from odme.models import (
    fit_strength_cost_binomial_coordinates,
    fit_strength_cost_geometric_coordinates,
    fit_strength_cost_negative_binomial_coordinates,
    fit_strength_cost_poisson,
    fit_strength_cost_poisson_coordinates,
    fit_strength_poisson,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
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
    strength_out, strength_in, x, y, target_cost = _fixture()
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


def test_coordinate_strength_cost_available_for_all_model_families() -> None:
    """Coordinate strength-cost wrappers exist for ME, B, and W families."""
    strength_out, strength_in, x, y, target_cost = _fixture()
    fits = [
        fit_strength_cost_poisson_coordinates(
            strength_out, strength_in, x, y, target_cost
        ),
        fit_strength_cost_binomial_coordinates(
            strength_out, strength_in, x, y, target_cost, layers=3
        ),
        fit_strength_cost_geometric_coordinates(
            strength_out, strength_in, x, y, target_cost
        ),
        fit_strength_cost_negative_binomial_coordinates(
            strength_out, strength_in, x, y, target_cost, layers=3
        ),
    ]
    assert all(fit.converged for fit in fits)
    assert [fit.family for fit in fits] == [
        "poisson",
        "binomial",
        "geometric",
        "negative_binomial",
    ]
