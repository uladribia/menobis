"""Tests for fitting."""

import numpy as np
import pytest

from menobis.models import fit_strength_poisson


def test_recovers_strengths() -> None:
    s_out = np.array([10, 20, 30], dtype=np.float64)
    s_in = np.array([15, 25, 20], dtype=np.float64)
    r = fit_strength_poisson(s_out, s_in)
    expected = np.outer(r.x, r.y)
    np.testing.assert_allclose(expected.sum(axis=1), s_out, rtol=1e-6)
    np.testing.assert_allclose(expected.sum(axis=0), s_in, rtol=1e-6)


def test_rejects_unbalanced() -> None:
    with pytest.raises(ValueError, match="balanced"):
        fit_strength_poisson(np.array([10.0]), np.array([20.0]))


def test_zero_strength_nodes() -> None:
    s_out = np.array([0, 30, 30], dtype=np.float64)
    s_in = np.array([20, 40, 0], dtype=np.float64)
    r = fit_strength_poisson(s_out, s_in)
    assert r.x[0] == 0.0
    assert r.y[2] == 0.0
