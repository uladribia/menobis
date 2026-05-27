"""Streaming-generation regression tests."""

import numpy as np

from menobis.models import (
    StrengthCostFit,
    sample_strength_cost_poisson,
    sample_strength_multinomial,
    sample_strength_poisson,
)


def test_factorized_poisson_generation_scales_to_1000_nodes() -> None:
    """Default generation handles 1e6 candidate pairs without dense matrices."""
    n = 1000
    x = np.full(n, 5e-4, dtype=np.float64)
    y = np.full(n, 5e-4, dtype=np.float64)

    sample = sample_strength_poisson(x, y, seed=123)

    assert len(sample.source) == len(sample.target) == len(sample.weight)
    assert np.all(sample.weight >= 1)


def test_factorized_multinomial_generation_scales_to_1000_nodes() -> None:
    """Canonical sampling uses row/column binomial passes and preserves T."""
    n = 1000
    x = np.linspace(1.0, 2.0, n, dtype=np.float64)
    y = np.linspace(2.0, 1.0, n, dtype=np.float64)

    sample = sample_strength_multinomial(x, y, total_events=5000, seed=123)

    assert int(sample.weight.sum()) == 5000
    assert len(sample.source) == len(sample.target) == len(sample.weight)


def test_strength_cost_generation_scales_to_1000_nodes() -> None:
    """Strength-cost sampling runs in Rust and streams pair rates on demand."""
    n = 1000
    fit = StrengthCostFit(
        node=np.arange(n, dtype=np.uint64),
        x=np.full(n, 5e-4, dtype=np.float64),
        y=np.full(n, 5e-4, dtype=np.float64),
        gamma=0.1,
        self_loops=True,
        converged=True,
        iterations=1,
    )

    sample = sample_strength_cost_poisson(
        fit,
        np.linspace(0.0, 1.0, n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        seed=123,
    )

    assert len(sample.source) == len(sample.target) == len(sample.weight)
    assert np.all(sample.weight >= 1)
