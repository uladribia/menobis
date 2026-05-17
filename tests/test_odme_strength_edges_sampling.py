"""Property tests for fixed-strength edge-count ME sampling."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from odme.models import fit_strength_edges_poisson, sample_strength_edges_poisson


def _expectations(
    x: np.ndarray, y: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    u = np.outer(x, y)
    exp_u = np.exp(u)
    den = 1.0 + lam * (exp_u - 1.0)
    p = lam * (exp_u - 1.0) / den
    expected = lam * u * exp_u / den
    return p, expected


@given(
    values=st.lists(
        st.floats(min_value=0.05, max_value=0.6, allow_nan=False, allow_infinity=False),
        min_size=7,
        max_size=7,
    )
)
@settings(deadline=None, max_examples=15)
def test_strength_edges_sampler_is_seeded(values: list[float]) -> None:
    arr = np.asarray(values, dtype=np.float64)
    x = arr[:3]
    y = arr[3:6]
    lam = float(arr[6])
    p, expected = _expectations(x, y, lam)
    fit = fit_strength_edges_poisson(
        expected.sum(axis=1), expected.sum(axis=0), float(p.sum())
    )

    first = sample_strength_edges_poisson(fit, seed=42)
    second = sample_strength_edges_poisson(fit, seed=42)
    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.weight, second.weight)
    assert np.all(first.weight > 0)
