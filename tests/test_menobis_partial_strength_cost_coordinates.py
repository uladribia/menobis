"""Partial coordinate strength-cost fitting tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from menobis.models.partial import (
    fit_partial_strength_cost_binomial_coordinates,
    fit_partial_strength_cost_geometric_coordinates,
    fit_partial_strength_cost_negative_binomial_coordinates,
    fit_partial_strength_cost_poisson_coordinates,
)


def test_partial_coordinate_strength_cost_available_for_all_families() -> None:
    """Partial coordinate strength-cost wrappers converge for ME/B/W labels."""
    # Use feasible strengths for B(M=10): max excess < M*(n-1)
    strength_out = np.array([2.0, 3.0, 4.0])
    strength_in = np.array([2.5, 3.5, 3.0])
    known_source = np.array([0], dtype=np.uint64)
    known_target = np.array([1], dtype=np.uint64)
    known_rate = np.array([0.5])
    x = np.array([0.0, 3.0, 0.0])
    y = np.array([0.0, 0.0, 4.0])
    target_cost = 12.0

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
            layers=10,
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
            layers=3,
        ),
    ]

    assert all(fit.converged for fit in fits), [
        (fit.family, fit.converged) for fit in fits
    ]
    assert [fit.family for fit in fits] == [
        "poisson",
        "binomial",
        "geometric",
        "negative_binomial",
    ]


def test_partial_coordinate_metric_must_be_euclidean() -> None:
    """Coordinate partials expose metric selection before adding more metrics."""
    with pytest.raises(ValueError, match=r"coordinate_metric.*euclidean"):
        fit_partial_strength_cost_poisson_coordinates(
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.float64),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            1.0,
            coordinate_metric="manhattan",
        )


def test_partial_coordinate_wrappers_do_not_allocate_dense_python_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B/W coordinate partial assembly must not allocate Python N x N masks."""
    original_zeros = cast(Callable[..., object], np.zeros)

    def guarded_zeros(shape: object, *args: object, **kwargs: object) -> object:
        if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == shape[1]:
            raise AssertionError("dense square mask allocated in Python")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", guarded_zeros)
    fit = fit_partial_strength_cost_binomial_coordinates(
        np.array([2.0, 3.0, 4.0]),
        np.array([2.5, 3.5, 3.0]),
        np.array([0], dtype=np.uint64),
        np.array([1], dtype=np.uint64),
        np.array([0.5]),
        np.array([0.0, 3.0, 0.0]),
        np.array([0.0, 0.0, 4.0]),
        12.0,
        layers=10,
        coordinate_metric="euclidean",
    )
    assert fit.family == "binomial"
