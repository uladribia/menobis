"""Tests for no-self-loops fitting."""

import numpy as np

from odme.models import fit_fixed_strength_me


def test_no_self_loops_recovers_strengths() -> None:
    s_out = np.array([10, 20, 30], dtype=np.float64)
    s_in = np.array([15, 25, 20], dtype=np.float64)
    r = fit_fixed_strength_me(s_out, s_in, self_loops=False)
    expected = np.outer(r.x, r.y)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(expected.sum(axis=1), s_out, rtol=1e-6)
    np.testing.assert_allclose(expected.sum(axis=0), s_in, rtol=1e-6)


def test_differs_from_self_loops() -> None:
    s_out = np.array([10, 20, 30], dtype=np.float64)
    s_in = np.array([15, 25, 20], dtype=np.float64)
    with_self = fit_fixed_strength_me(s_out, s_in, self_loops=True)
    without_self = fit_fixed_strength_me(s_out, s_in, self_loops=False)
    assert not np.allclose(with_self.x, without_self.x, rtol=1e-10)
