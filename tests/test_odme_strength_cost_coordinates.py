"""Coordinate-based strength-cost fitting tests."""

from __future__ import annotations

import numpy as np

from odme.models import (
    fit_strength_cost_poisson,
    fit_strength_cost_poisson_coordinates,
    fit_strength_poisson,
)


def test_coordinate_strength_cost_matches_complete_sparse_costs() -> None:
    """Projected XY coordinate costs match equivalent complete sparse costs."""
    strength_out = np.array([10.0, 20.0, 30.0])
    strength_in = np.array([15.0, 25.0, 20.0])
    x = np.array([0.0, 3.0, 0.0])
    y = np.array([0.0, 0.0, 4.0])
    base = fit_strength_poisson(strength_out, strength_in)
    coords = np.column_stack([x, y])
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    target_cost = float(np.sum(np.outer(base.x, base.y) * distance))
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
