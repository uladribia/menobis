"""Tests for original fixed-degree ME weighted generation."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from odme.models import fit_degree_bernoulli, sample_degree_events_poisson


def _probability(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.outer(x, y)
    return z / (1.0 + z)


@given(
    values=st.lists(
        st.floats(min_value=0.05, max_value=0.8, allow_nan=False, allow_infinity=False),
        min_size=6,
        max_size=6,
    ),
)
@settings(deadline=None, max_examples=15)
def test_fixed_degree_equations_and_total_weight_expectation(
    values: list[float],
) -> None:
    arr = np.asarray(values, dtype=np.float64).reshape(2, 3)
    true_x, true_y = arr
    p = _probability(true_x, true_y)
    k_out = p.sum(axis=1)
    k_in = p.sum(axis=0)
    fit = fit_degree_bernoulli(k_out, k_in)
    p_fit = _probability(fit.x, fit.y)

    np.testing.assert_allclose(p_fit.sum(axis=1), k_out, atol=1e-6)
    np.testing.assert_allclose(p_fit.sum(axis=0), k_in, atol=1e-6)
    total_events = 1000
    average_existing_weight = total_events / p_fit.sum()
    np.testing.assert_allclose((p_fit * average_existing_weight).sum(), total_events)


def test_fixed_degree_events_me_sample_is_reproducible() -> None:
    k_out = np.array([0.8, 1.2, 1.0], dtype=np.float64)
    k_in = np.array([1.1, 0.9, 1.0], dtype=np.float64)
    fit = fit_degree_bernoulli(k_out, k_in)
    first = sample_degree_events_poisson(fit, total_events=100, seed=42)
    second = sample_degree_events_poisson(fit, total_events=100, seed=42)

    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.weight, second.weight)
    assert np.all(first.weight >= 1)
