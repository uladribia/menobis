"""Ensemble averaging utilities for MENoBiS analysis.

Example:
-------
```python
from menobis.analysis import directed_strengths, ensemble_average
from menobis.routing import fit_model, sample_model
from menobis.models import Constraint, ModelFamily

result = fit_model(
    family=ModelFamily.ME, constraint=Constraint.STRENGTH,
    strength_out=s_out, strength_in=s_in,
)

def generate(seed: int):
    return sample_model(
        family=ModelFamily.ME, constraint=Constraint.STRENGTH,
        fit=result, seed=seed,
    )

mean, std = ensemble_average(
    generate=generate,
    analyze=lambda e: directed_strengths(e).out.astype(float),
    repetitions=100,
)
```
"""

import statistics
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from menobis.data.frames import EdgeTable


def ensemble_average(
    *,
    generate: Callable[[int], EdgeTable],
    analyze: Callable[[EdgeTable], NDArray[np.floating]],
    repetitions: int,
    seed_start: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average a per-node statistic over ensemble samples.

    Args:
        generate: Takes a seed, returns an EdgeTable.
        analyze: Takes an EdgeTable, returns a 1D float array (one value per node).
        repetitions: Number of samples.
        seed_start: First seed.

    Returns:
        Tuple of (mean_array, std_array).
    """
    samples = [analyze(generate(seed_start + i)) for i in range(repetitions)]
    stacked = np.stack(samples)
    return stacked.mean(axis=0), stacked.std(axis=0)


def ensemble_scalar_average(
    *,
    generate: Callable[[int], EdgeTable],
    compute: Callable[[EdgeTable], float],
    repetitions: int,
    seed_start: int = 0,
) -> tuple[float, float]:
    """Average a scalar statistic over ensemble samples.

    Returns:
        Tuple of (mean, standard_deviation).
    """
    values = [compute(generate(seed_start + i)) for i in range(repetitions)]
    if not values:
        return 0.0, 0.0
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


__all__ = ["ensemble_average", "ensemble_scalar_average"]
