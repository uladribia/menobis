"""Tests for W degree-events fitting (geometric and negative binomial)."""

import numpy as np


def test_fit_degree_events_geometric_recovers_degrees() -> None:
    """Geometric degree-events fitter recovers expected degrees."""
    from menobis.models.fitting import fit_degree_events_geometric

    # 3-node network with known degrees and T > E
    k_out = np.array([2.0, 1.0, 1.0])
    k_in = np.array([1.0, 2.0, 1.0])
    total_events = 10  # T > E=4

    result = fit_degree_events_geometric(k_out, k_in, total_events)

    assert result.converged
    assert result.q > 0.0
    assert result.q < 1.0
    # Expected degrees should recover input
    n = len(k_out)
    expected_k_out = np.zeros(n)
    for i in range(n):
        for j in range(n):
            z = result.x[i] * result.y[j]
            expected_k_out[i] += z / (1.0 + z)
    np.testing.assert_allclose(expected_k_out, k_out, atol=0.1)


def test_fit_degree_events_negative_binomial_recovers_degrees() -> None:
    """Negative binomial degree-events fitter recovers expected degrees."""
    from menobis.models.fitting import fit_degree_events_negative_binomial

    k_out = np.array([2.0, 1.0, 1.0])
    k_in = np.array([1.0, 2.0, 1.0])
    total_events = 12
    layers = 3

    result = fit_degree_events_negative_binomial(
        k_out, k_in, total_events, layers=layers
    )

    assert result.converged
    assert result.q > 0.0
    assert result.q < 1.0
    n = len(k_out)
    expected_k_out = np.zeros(n)
    for i in range(n):
        for j in range(n):
            z = result.x[i] * result.y[j]
            expected_k_out[i] += z / (1.0 + z)
    np.testing.assert_allclose(expected_k_out, k_out, atol=0.1)


def test_fit_degree_events_geometric_rejects_t_less_than_e() -> None:
    """Reject T < E."""
    import pytest

    from menobis.models.fitting import fit_degree_events_geometric

    k_out = np.array([2.0, 1.0])
    k_in = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="total_events"):
        fit_degree_events_geometric(k_out, k_in, total_events=2)
