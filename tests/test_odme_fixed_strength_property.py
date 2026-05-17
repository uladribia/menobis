"""Property tests for fixed-strength ME samplers."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from odme.models import (
    fit_strength_poisson,
    sample_strength_multinomial,
    sample_strength_poisson,
    sample_strength_poisson_multinomial,
)


@given(values=st.lists(st.integers(min_value=1, max_value=50), min_size=3, max_size=3))
@settings(deadline=None, max_examples=20)
def test_fixed_strength_multinomial_preserves_total(values: list[int]) -> None:
    s = np.asarray(values, dtype=np.float64)
    fit = fit_strength_poisson(s, s)
    total = int(s.sum())
    sample = sample_strength_multinomial(fit.x, fit.y, total_events=total, seed=42)
    assert sample.total_events == total
    assert np.all(sample.weight > 0)


@given(values=st.lists(st.integers(min_value=1, max_value=40), min_size=3, max_size=3))
@settings(deadline=None, max_examples=20)
def test_fixed_strength_poisson_is_seeded_and_non_negative(values: list[int]) -> None:
    s = np.asarray(values, dtype=np.float64)
    fit = fit_strength_poisson(s, s)
    first = sample_strength_poisson(fit.x, fit.y, seed=42)
    second = sample_strength_poisson(fit.x, fit.y, seed=42)
    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.weight, second.weight)
    assert np.all(first.weight > 0)


@given(values=st.lists(st.integers(min_value=1, max_value=40), min_size=3, max_size=3))
@settings(deadline=None, max_examples=20)
def test_fixed_strength_poisson_multinomial_is_seeded(values: list[int]) -> None:
    s = np.asarray(values, dtype=np.float64)
    fit = fit_strength_poisson(s, s)
    first = sample_strength_poisson_multinomial(fit.x, fit.y, seed=42)
    second = sample_strength_poisson_multinomial(fit.x, fit.y, seed=42)
    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.weight, second.weight)
    assert first.total_events >= 0
