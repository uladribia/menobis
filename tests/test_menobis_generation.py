"""Tests for generation samplers."""

import numpy as np

from menobis.models import (
    fit_strength_poisson,
    sample_strength_multinomial,
    sample_strength_poisson,
)


def _xy(n: int, total: int):
    s = np.full(n, total // n, dtype=np.float64)
    r = fit_strength_poisson(s, s)
    return r.x, r.y


def test_poisson_positive_weights() -> None:
    x, y = _xy(50, 5000)
    sample = sample_strength_poisson(x, y, seed=42)
    assert sample.num_edges > 0
    assert np.all(sample.weight > 0)


def test_poisson_reproducible() -> None:
    x, y = _xy(20, 2000)
    a = sample_strength_poisson(x, y, seed=7)
    b = sample_strength_poisson(x, y, seed=7)
    np.testing.assert_array_equal(a.source, b.source)
    np.testing.assert_array_equal(a.weight, b.weight)


def test_multinomial_preserves_total() -> None:
    x, y = _xy(50, 5000)
    sample = sample_strength_multinomial(x, y, total_events=5000, seed=42)
    assert sample.total_events == 5000


def test_multinomial_reproducible() -> None:
    x, y = _xy(20, 2000)
    a = sample_strength_multinomial(x, y, total_events=2000, seed=7)
    b = sample_strength_multinomial(x, y, total_events=2000, seed=7)
    np.testing.assert_array_equal(a.weight, b.weight)


def test_no_self_loops() -> None:
    x, y = _xy(10, 1000)
    sample = sample_strength_poisson(x, y, self_loops=False, seed=42)
    assert np.all(sample.source != sample.target)


def test_poisson_mean_approaches_expected() -> None:
    x, y = _xy(5, 500)
    total = np.zeros((5, 5))
    for seed in range(200):
        s = sample_strength_poisson(x, y, seed=seed)
        for i, j, w in zip(s.source, s.target, s.weight, strict=True):
            total[i, j] += w
    np.testing.assert_allclose(total / 200, np.outer(x, y), rtol=0.15)
