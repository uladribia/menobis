"""Tests for W fixed-strength-plus-degree fitting wrappers."""

import numpy as np
import pytest

from menobis.models import StrengthDegreeFit


def _expected_strength_degree(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    layers: int,
    self_loops: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    s_out = np.zeros(n)
    s_in = np.zeros(n)
    k_out = np.zeros(n)
    k_in = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if not self_loops and i == j:
                continue
            q = x[i] * y[j]
            v = z[i] * w[j]
            a = (1.0 - q) ** (-layers)
            g = a - 1.0
            zz = 1.0 + v * g
            occupation = v * g / zz
            mean = v * layers * q * (1.0 - q) ** (-layers - 1) / zz
            s_out[i] += mean
            s_in[j] += mean
            k_out[i] += occupation
            k_in[j] += occupation
    return s_out, s_in, k_out, k_in


def test_fit_strength_degree_geometric_recovers_homogeneous_constraints() -> None:
    """Geometric W strength-degree fitting recovers strengths and degrees."""
    from menobis.models import fit_strength_degree_geometric

    result = fit_strength_degree_geometric(
        np.array([2.0, 2.0]),
        np.array([2.0, 2.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        self_loops=True,
        tolerance=1e-8,
        max_iterations=1000,
    )

    s_out, s_in, k_out, k_in = _expected_strength_degree(
        result.x, result.y, result.z, result.w, result.layers or 1, True
    )
    assert isinstance(result, StrengthDegreeFit)
    assert result.converged
    assert result.family == "geometric"
    assert np.max(np.abs(s_out - 2.0)) < 1e-3
    assert np.max(np.abs(s_in - 2.0)) < 1e-3
    assert np.max(np.abs(k_out - 1.0)) < 1e-3
    assert np.max(np.abs(k_in - 1.0)) < 1e-3


def test_fit_strength_degree_negative_binomial_rejects_single_layer() -> None:
    """Negative-binomial strength-degree API reserves M=1 for geometric."""
    from menobis.models import fit_strength_degree_negative_binomial

    with pytest.raises(ValueError, match="layers > 1"):
        fit_strength_degree_negative_binomial(
            np.array([2.0]),
            np.array([2.0]),
            np.array([1.0]),
            np.array([1.0]),
            layers=1,
        )
