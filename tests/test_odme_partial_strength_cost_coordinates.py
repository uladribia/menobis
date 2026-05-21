"""Partial coordinate strength-cost fitting tests."""

from __future__ import annotations

import numpy as np

from odme.models.partial import (
    fit_partial_strength_cost_binomial_coordinates,
    fit_partial_strength_cost_geometric_coordinates,
    fit_partial_strength_cost_negative_binomial_coordinates,
    fit_partial_strength_cost_poisson_coordinates,
)


def test_partial_coordinate_strength_cost_available_for_all_families() -> None:
    """Partial coordinate strength-cost wrappers converge for ME/B/W labels."""
    strength_out = np.array([10.0, 20.0, 30.0])
    strength_in = np.array([15.0, 25.0, 20.0])
    known_source = np.array([0], dtype=np.uint64)
    known_target = np.array([1], dtype=np.uint64)
    known_rate = np.array([2.0])
    x = np.array([0.0, 3.0, 0.0])
    y = np.array([0.0, 0.0, 4.0])
    coords = np.column_stack([x, y])
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    target_cost = float(
        0.85
        * np.sum(np.outer(strength_out, strength_in) / strength_out.sum() * distance)
    )

    fits = [
        fit_partial_strength_cost_poisson_coordinates(
            strength_out,
            strength_in,
            known_source,
            known_target,
            known_rate,
            x,
            y,
            target_cost,
        ),
        fit_partial_strength_cost_binomial_coordinates(
            strength_out,
            strength_in,
            known_source,
            known_target,
            known_rate,
            x,
            y,
            target_cost,
        ),
        fit_partial_strength_cost_geometric_coordinates(
            strength_out,
            strength_in,
            known_source,
            known_target,
            known_rate,
            x,
            y,
            target_cost,
        ),
        fit_partial_strength_cost_negative_binomial_coordinates(
            strength_out,
            strength_in,
            known_source,
            known_target,
            known_rate,
            x,
            y,
            target_cost,
        ),
    ]

    assert all(fit.converged for fit in fits)
    assert [fit.family for fit in fits] == [
        "poisson",
        "binomial",
        "geometric",
        "negative_binomial",
    ]
