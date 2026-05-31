"""Tests for ensemble averaging."""

from pathlib import Path

import numpy as np

from menobis.analysis import (
    directed_strengths,
    ensemble_average,
    ensemble_scalar_average,
)
from menobis.models.fitting import _fit_strength_poisson as fit_strength_poisson
from menobis.models.generation import (
    _sample_strength_poisson as sample_strength_poisson,
)


def _gen(n: int, total: int):
    s = np.full(n, total // n, dtype=np.float64)
    r = fit_strength_poisson(s, s)
    return lambda seed: sample_strength_poisson(r.x, r.y, seed=seed)


def test_ensemble_module_belongs_to_analysis_package() -> None:
    """Ensemble helpers should live under the analysis domain."""
    assert not (Path(__file__).parents[1] / "src" / "menobis" / "ensemble.py").exists()


def test_ensemble_average() -> None:
    generate = _gen(2, 100)
    mean, std = ensemble_average(
        generate=generate,
        analyze=lambda e: directed_strengths(e).out.astype(float),
        repetitions=10,
    )
    assert len(mean) == 2
    assert len(std) == 2


def test_ensemble_scalar_average() -> None:
    generate = _gen(2, 100)
    mean, std = ensemble_scalar_average(
        generate=generate,
        compute=lambda e: float(e.total_events),
        repetitions=20,
    )
    assert mean > 0
    assert std >= 0
