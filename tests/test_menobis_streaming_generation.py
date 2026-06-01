"""Streaming-generation regression tests."""

import numpy as np

from menobis.models.generation import (
    _sample_strength_cost_poisson as sample_strength_cost_poisson,
)
from menobis.models.generation import (
    _sample_strength_multinomial as sample_strength_multinomial,
)
from menobis.models.generation import (
    _sample_strength_poisson as sample_strength_poisson,
)
from menobis.models.types import StrengthCostFit


def test_factorized_poisson_generation_smoke() -> None:
    """Default generation handles factorized pair rates without dense matrices."""
    n = 10
    x = np.full(n, 5e-4, dtype=np.float64)
    y = np.full(n, 5e-4, dtype=np.float64)

    sample = sample_strength_poisson(x, y, seed=123)

    assert len(sample.source) == len(sample.target) == len(sample.weight)
    assert np.all(sample.weight >= 1)


def test_factorized_multinomial_generation_smoke() -> None:
    """Canonical sampling uses row/column binomial passes and preserves T."""
    n = 10
    x = np.linspace(1.0, 2.0, n, dtype=np.float64)
    y = np.linspace(2.0, 1.0, n, dtype=np.float64)

    sample = sample_strength_multinomial(x, y, total_events=100, seed=123)

    assert int(sample.weight.sum()) == 100
    assert len(sample.source) == len(sample.target) == len(sample.weight)


def test_strength_cost_generation_smoke() -> None:
    """Strength-cost sampling runs in Rust and streams pair rates on demand."""
    n = 10
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
