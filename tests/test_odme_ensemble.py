"""Tests for ensemble averaging."""

import numpy as np

from odme.analysis import directed_strengths
from odme.ensemble import ensemble_average, ensemble_scalar_average
from odme.models import fit_strength_poisson, sample_strength_poisson


def _gen(n: int, total: int):
    s = np.full(n, total // n, dtype=np.float64)
    r = fit_strength_poisson(s, s)
    return lambda seed: sample_strength_poisson(r.x, r.y, seed=seed)


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
