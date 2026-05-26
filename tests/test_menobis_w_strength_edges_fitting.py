"""Tests for W fixed-strength-plus-edge-count fitting wrappers."""

import numpy as np
import pytest

from menobis.models import StrengthEdgesFit, fit_strength_edges_geometric


def _expected_strength_edges(
    x: np.ndarray,
    y: np.ndarray,
    lam: float,
    layers: int,
    self_loops: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(x)
    out = np.zeros(n)
    inc = np.zeros(n)
    edges = 0.0
    for i in range(n):
        for j in range(n):
            if not self_loops and i == j:
                continue
            q = x[i] * y[j]
            a = (1.0 - q) ** (-layers)
            g = a - 1.0
            z = 1.0 + lam * g
            occupation = lam * g / z
            mean = lam * layers * q * (1.0 - q) ** (-layers - 1) / z
            out[i] += mean
            inc[j] += mean
            edges += occupation
    return out, inc, edges


def test_fit_strength_edges_geometric_recovers_homogeneous_constraints() -> None:
    """Geometric W strength-edges fitting recovers strengths and edge count."""
    result = fit_strength_edges_geometric(
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        target_edges=1.0,
        self_loops=True,
        tolerance=1e-8,
        max_iterations=500,
    )

    out, inc, edges = _expected_strength_edges(
        result.x, result.y, result.lam, result.layers or 1, True
    )
    assert isinstance(result, StrengthEdgesFit)
    assert result.converged
    assert result.family == "geometric"
    assert result.diagnostics is not None
    assert result.diagnostics.conic is not None
    assert np.max(np.abs(out - 1.0)) < 1e-4
    assert np.max(np.abs(inc - 1.0)) < 1e-4
    assert abs(edges - 1.0) < 1e-4


def test_fit_strength_edges_geometric_recovers_nonhomogeneous_constraints() -> None:
    """General geometric W strength-edges fitting handles heterogeneous strengths."""
    result = fit_strength_edges_geometric(
        np.array([1.5, 0.5]),
        np.array([1.0, 1.0]),
        target_edges=1.2,
        self_loops=True,
        tolerance=1e-8,
        max_iterations=1000,
    )

    out, inc, edges = _expected_strength_edges(
        result.x, result.y, result.lam, result.layers or 1, True
    )
    assert result.converged
    assert np.max(np.abs(out - np.array([1.5, 0.5]))) < 1e-4
    assert np.max(np.abs(inc - np.array([1.0, 1.0]))) < 1e-4
    assert abs(edges - 1.2) < 1e-4


def test_fit_strength_edges_negative_binomial_rejects_single_layer() -> None:
    """Negative-binomial strength-edges API reserves M=1 for geometric."""
    from menobis.models import fit_strength_edges_negative_binomial

    with pytest.raises(ValueError, match="layers > 1"):
        fit_strength_edges_negative_binomial(
            np.array([1.0]),
            np.array([1.0]),
            target_edges=1.0,
            layers=1,
        )
