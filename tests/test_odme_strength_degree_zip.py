"""Tests for grand-canonical fixed-strength-degree ZIP model."""

import numpy as np

from odme.analysis import directed_degrees, directed_strengths
from odme.models import fit_strength_degree_zip, sample_strength_degree_zip


def _probability(x: np.ndarray, y: np.ndarray, *, self_loops: bool) -> np.ndarray:
    z = np.outer(x, y)
    p = z / (1.0 + z)
    if not self_loops:
        np.fill_diagonal(p, 0.0)
    return p


def _lambda(a: np.ndarray, b: np.ndarray, *, self_loops: bool) -> np.ndarray:
    lam = np.outer(a, b)
    if not self_loops:
        np.fill_diagonal(lam, 0.0)
    return lam


def test_zip_fit_recovers_expected_strengths_and_degrees() -> None:
    k_out = np.array([0.8, 1.2, 1.0], dtype=np.float64)
    k_in = np.array([1.1, 0.9, 1.0], dtype=np.float64)
    s_out = np.array([2.0, 3.5, 2.5], dtype=np.float64)
    s_in = np.array([2.7, 2.4, 2.9], dtype=np.float64)

    fit = fit_strength_degree_zip(s_out, s_in, k_out, k_in)
    p = _probability(fit.degree_x, fit.degree_y, self_loops=True)
    lam = _lambda(fit.excess_x, fit.excess_y, self_loops=True)
    expected_strength = p * (1.0 + lam)

    np.testing.assert_allclose(p.sum(axis=1), k_out, atol=1e-6)
    np.testing.assert_allclose(p.sum(axis=0), k_in, atol=1e-6)
    np.testing.assert_allclose(expected_strength.sum(axis=1), s_out, atol=1e-6)
    np.testing.assert_allclose(expected_strength.sum(axis=0), s_in, atol=1e-6)


def test_zip_fit_no_self_loops() -> None:
    k_out = np.array([0.8, 1.1, 0.9], dtype=np.float64)
    k_in = np.array([0.9, 0.8, 1.1], dtype=np.float64)
    s_out = np.array([2.2, 3.0, 2.8], dtype=np.float64)
    s_in = np.array([2.5, 2.1, 3.4], dtype=np.float64)

    fit = fit_strength_degree_zip(s_out, s_in, k_out, k_in, self_loops=False)
    p = _probability(fit.degree_x, fit.degree_y, self_loops=False)
    lam = _lambda(fit.excess_x, fit.excess_y, self_loops=False)
    expected_strength = p * (1.0 + lam)

    np.testing.assert_allclose(np.diag(p), 0.0)
    np.testing.assert_allclose(expected_strength.sum(axis=1), s_out, atol=1e-6)
    np.testing.assert_allclose(expected_strength.sum(axis=0), s_in, atol=1e-6)


def test_zip_sample_is_reproducible_and_weighted_positive() -> None:
    k_out = np.array([0.8, 1.2, 1.0], dtype=np.float64)
    k_in = np.array([1.1, 0.9, 1.0], dtype=np.float64)
    s_out = np.array([20.0, 35.0, 25.0], dtype=np.float64)
    s_in = np.array([27.0, 24.0, 29.0], dtype=np.float64)
    fit = fit_strength_degree_zip(s_out, s_in, k_out, k_in)

    first = sample_strength_degree_zip(fit, seed=42)
    second = sample_strength_degree_zip(fit, seed=42)

    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.weight, second.weight)
    assert np.all(first.weight >= 1)

    strengths = directed_strengths(first)
    degrees = directed_degrees(first)
    assert np.all(strengths.out >= degrees.out)
    assert np.all(strengths.incoming >= degrees.incoming)
