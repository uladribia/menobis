"""Property tests for exact ME fixed-strength and edge-count model."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from odme.models import fit_strength_edges_zip


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
    ),
)
@settings(deadline=None, max_examples=15)
def test_fixed_strength_edges_equations_hold(values: list[float]) -> None:
    arr = np.asarray(values, dtype=np.float64)
    x = arr[:3]
    y = arr[3:6]
    lam = float(arr[6])
    p, expected = _expectations(x, y, lam)
    fit = fit_strength_edges_zip(
        expected.sum(axis=1), expected.sum(axis=0), float(p.sum())
    )
    p_fit, expected_fit = _expectations(fit.x, fit.y, fit.lam)

    np.testing.assert_allclose(
        expected_fit.sum(axis=1), expected.sum(axis=1), atol=1e-6
    )
    np.testing.assert_allclose(
        expected_fit.sum(axis=0), expected.sum(axis=0), atol=1e-6
    )
    np.testing.assert_allclose(p_fit.sum(), p.sum(), atol=1e-6)
