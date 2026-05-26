"""Regression tests for zero-inflated binomial fitting kernels."""

import numpy as np

from menobis.models import fit_strength_degree_binomial, fit_strength_edges_binomial
from menobis.utilities.synthetic import generate_pa_geographic_network

LAYERS = 20


def _pa_constraints() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a PA-geographic network and derive feasible B constraints."""
    network = generate_pa_geographic_network(
        8,
        average_degree=2.0,
        events_per_edge=3.0,
        seed=20240525,
        self_loops=False,
    )
    edges = network.edges
    s_out = np.zeros(8, dtype=np.float64)
    s_in = np.zeros(8, dtype=np.float64)
    k_out = np.zeros(8, dtype=np.float64)
    k_in = np.zeros(8, dtype=np.float64)
    np.add.at(s_out, edges.source.astype(np.int64), edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target.astype(np.int64), edges.weight.astype(np.float64))
    np.add.at(k_out, edges.source.astype(np.int64), 1.0)
    np.add.at(k_in, edges.target.astype(np.int64), 1.0)
    return s_out, s_in, k_out, k_in, float(edges.num_edges)


def _b_mean(q: float, lam: float, layers: int) -> float:
    """Zero-inflated B mean from thesis G_B(q)=(1+q)^M-1."""
    return (
        lam
        * layers
        * q
        * (1.0 + q) ** (layers - 1)
        / (1.0 + lam * ((1.0 + q) ** layers - 1.0))
    )


def _b_occ(q: float, lam: float, layers: int) -> float:
    """Zero-inflated B occupation from thesis G_B(q)=(1+q)^M-1."""
    g = (1.0 + q) ** layers - 1.0
    return lam * g / (1.0 + lam * g)


def test_strength_edges_binomial_recovers_derived_constraints() -> None:
    """B strength-edges fit uses the B kernel on PA-geographic constraints."""
    s_out, s_in, _k_out, _k_in, target_edges = _pa_constraints()
    fit = fit_strength_edges_binomial(
        s_out, s_in, target_edges, layers=LAYERS, self_loops=False
    )

    assert fit.converged
    n = len(s_out)
    expected = np.zeros((n, n), dtype=np.float64)
    occ = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                q = float(fit.x[i] * fit.y[j])
                expected[i, j] = _b_mean(q, fit.lam, LAYERS)
                occ[i, j] = _b_occ(q, fit.lam, LAYERS)

    np.testing.assert_allclose(expected.sum(axis=1), s_out, atol=1e-5)
    np.testing.assert_allclose(expected.sum(axis=0), s_in, atol=1e-5)
    assert abs(float(occ.sum()) - target_edges) < 1e-5


def test_strength_degree_binomial_recovers_derived_constraints() -> None:
    """B strength-degree fit uses the B kernel on PA-geographic constraints."""
    s_out, s_in, k_out, k_in, _target_edges = _pa_constraints()
    fit = fit_strength_degree_binomial(
        s_out,
        s_in,
        k_out,
        k_in,
        layers=LAYERS,
        self_loops=False,
        tolerance=1e-4,
    )

    assert fit.converged
    n = len(s_out)
    expected = np.zeros((n, n), dtype=np.float64)
    occ = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                q = float(fit.x[i] * fit.y[j])
                v = float(fit.z[i] * fit.w[j])
                expected[i, j] = _b_mean(q, v, LAYERS)
                occ[i, j] = _b_occ(q, v, LAYERS)

    tolerance = 0.01 * max(float(s_out.max()), float(k_out.max()))
    np.testing.assert_allclose(expected.sum(axis=1), s_out, atol=tolerance)
    np.testing.assert_allclose(expected.sum(axis=0), s_in, atol=tolerance)
    np.testing.assert_allclose(occ.sum(axis=1), k_out, atol=tolerance)
    np.testing.assert_allclose(occ.sum(axis=0), k_in, atol=tolerance)
