"""Tests for fixed-degree model fitting."""

import numpy as np
import pytest

from menobis.models import fit_degree_bernoulli


def _probability_matrix(
    x: np.ndarray, y: np.ndarray, *, self_loops: bool
) -> np.ndarray:
    z = np.outer(x, y)
    p = z / (1.0 + z)
    if not self_loops:
        np.fill_diagonal(p, 0.0)
    return p


def test_fixed_degree_recovers_expected_degrees() -> None:
    k_out = np.array([0.8, 1.2, 1.0], dtype=np.float64)
    k_in = np.array([1.1, 0.9, 1.0], dtype=np.float64)

    result = fit_degree_bernoulli(k_out, k_in)
    p = _probability_matrix(result.x, result.y, self_loops=True)

    np.testing.assert_allclose(p.sum(axis=1), k_out, atol=1e-6)
    np.testing.assert_allclose(p.sum(axis=0), k_in, atol=1e-6)


def test_fixed_degree_no_self_loops_recovers_expected_degrees() -> None:
    k_out = np.array([0.8, 1.1, 0.9], dtype=np.float64)
    k_in = np.array([0.9, 0.8, 1.1], dtype=np.float64)

    result = fit_degree_bernoulli(k_out, k_in, self_loops=False)
    p = _probability_matrix(result.x, result.y, self_loops=False)

    np.testing.assert_allclose(p.sum(axis=1), k_out, atol=1e-6)
    np.testing.assert_allclose(p.sum(axis=0), k_in, atol=1e-6)
    np.testing.assert_allclose(np.diag(p), 0.0)


def test_fixed_degree_rejects_unbalanced_sequences() -> None:
    with pytest.raises(ValueError, match="balanced"):
        fit_degree_bernoulli(np.array([1.0]), np.array([0.5]))
