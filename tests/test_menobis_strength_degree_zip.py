"""Property tests for exact ME fixed-strength-degree ME model."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from menobis.analysis import directed_degrees, directed_strengths
from menobis.models import fit_strength_degree_poisson, sample_strength_degree_poisson


def _expectations(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    u = np.outer(x, y)
    v = np.outer(z, w)
    exp_u = np.exp(u)
    den = 1.0 + v * (exp_u - 1.0)
    p = v * (exp_u - 1.0) / den
    expected = v * u * exp_u / den
    return p, expected


@given(
    values=st.lists(
        st.floats(min_value=0.05, max_value=0.7, allow_nan=False, allow_infinity=False),
        min_size=12,
        max_size=12,
    ),
)
@settings(deadline=None, max_examples=15)
def test_zip_fit_recovers_expected_strengths_and_degrees(values: list[float]) -> None:
    """Fitted equations recover constraints generated from valid multipliers."""
    arr = np.asarray(values, dtype=np.float64).reshape(4, 3)
    true_x, true_y, true_z, true_w = arr
    p, expected = _expectations(true_x, true_y, true_z, true_w)
    k_out = p.sum(axis=1)
    k_in = p.sum(axis=0)
    s_out = expected.sum(axis=1)
    s_in = expected.sum(axis=0)

    fit = fit_strength_degree_poisson(s_out, s_in, k_out, k_in)
    p_fit, expected_fit = _expectations(fit.x, fit.y, fit.z, fit.w)

    np.testing.assert_allclose(p_fit.sum(axis=1), k_out, atol=1e-6)
    np.testing.assert_allclose(p_fit.sum(axis=0), k_in, atol=1e-6)
    np.testing.assert_allclose(expected_fit.sum(axis=1), s_out, atol=1e-6)
    np.testing.assert_allclose(expected_fit.sum(axis=0), s_in, atol=1e-6)


def test_zip_sample_is_reproducible_and_weighted_positive() -> None:
    """Samples are seeded and have positive integer weights on present edges."""
    x = np.array([0.3, 0.4, 0.5], dtype=np.float64)
    y = np.array([0.2, 0.6, 0.4], dtype=np.float64)
    z = np.array([0.7, 0.5, 0.8], dtype=np.float64)
    w = np.array([0.6, 0.9, 0.4], dtype=np.float64)
    p, expected = _expectations(x, y, z, w)
    fit = fit_strength_degree_poisson(
        expected.sum(axis=1), expected.sum(axis=0), p.sum(axis=1), p.sum(axis=0)
    )

    first = sample_strength_degree_poisson(fit, seed=42)
    second = sample_strength_degree_poisson(fit, seed=42)

    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.weight, second.weight)
    assert np.all(first.weight >= 1)

    strengths = directed_strengths(first)
    degrees = directed_degrees(first)
    assert np.all(strengths.out >= degrees.out)
    assert np.all(strengths.incoming >= degrees.incoming)
