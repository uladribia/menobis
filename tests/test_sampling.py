"""Sampling tests: reproducibility, non-negativity, constraint preservation.

Uses N=20 PA-geographic derived constraints for realistic inputs.
Tests all sampler families and constraint types.
"""

import numpy as np
import pytest

from menobis.models import (
    fit_strength_binomial,
    fit_strength_degree_poisson,
    fit_strength_edges_poisson,
    fit_strength_geometric,
    fit_strength_poisson,
    sample_strength_binomial,
    sample_strength_degree_poisson,
    sample_strength_edges_poisson,
    sample_strength_geometric,
    sample_strength_multinomial,
    sample_strength_poisson,
)
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

N = 20
SEED = 54320


@pytest.fixture(scope="module")
def network():
    return generate_pa_geographic_network(
        N, average_degree=5.0, events_per_edge=4.0, seed=SEED, self_loops=False
    )


@pytest.fixture(scope="module")
def constraints(network):
    return derive_synthetic_constraints(network)


# --- Reproducibility tests ---


class TestReproducibility:
    """Seeded sampling produces identical results."""

    def test_poisson_reproducible(self, constraints) -> None:
        fit = fit_strength_poisson(
            constraints.strength_out, constraints.strength_in, self_loops=False
        )
        a = sample_strength_poisson(fit.x, fit.y, self_loops=False, seed=42)
        b = sample_strength_poisson(fit.x, fit.y, self_loops=False, seed=42)
        np.testing.assert_array_equal(a.source, b.source)
        np.testing.assert_array_equal(a.target, b.target)
        np.testing.assert_array_equal(a.weight, b.weight)

    def test_multinomial_reproducible(self, constraints) -> None:
        fit = fit_strength_poisson(
            constraints.strength_out, constraints.strength_in, self_loops=False
        )
        total = int(constraints.strength_out.sum())
        a = sample_strength_multinomial(
            fit.x, fit.y, total_events=total, self_loops=False, seed=42
        )
        b = sample_strength_multinomial(
            fit.x, fit.y, total_events=total, self_loops=False, seed=42
        )
        np.testing.assert_array_equal(a.weight, b.weight)

    def test_geometric_reproducible(self, constraints) -> None:
        fit = fit_strength_geometric(
            constraints.strength_out, constraints.strength_in, self_loops=False
        )
        a = sample_strength_geometric(fit.x, fit.y, self_loops=False, seed=42)
        b = sample_strength_geometric(fit.x, fit.y, self_loops=False, seed=42)
        np.testing.assert_array_equal(a.source, b.source)
        np.testing.assert_array_equal(a.weight, b.weight)

    def test_binomial_reproducible(self, constraints) -> None:
        fit = fit_strength_binomial(
            constraints.strength_out,
            constraints.strength_in,
            layers=constraints.binomial_layers,
            self_loops=False,
        )
        a = sample_strength_binomial(
            fit.x, fit.y, layers=constraints.binomial_layers, self_loops=False, seed=42
        )
        b = sample_strength_binomial(
            fit.x, fit.y, layers=constraints.binomial_layers, self_loops=False, seed=42
        )
        np.testing.assert_array_equal(a.source, b.source)
        np.testing.assert_array_equal(a.weight, b.weight)

    def test_strength_edges_reproducible(self, constraints) -> None:
        fit = fit_strength_edges_poisson(
            constraints.strength_out,
            constraints.strength_in,
            constraints.total_edges,
            self_loops=False,
        )
        a = sample_strength_edges_poisson(fit, seed=42)
        b = sample_strength_edges_poisson(fit, seed=42)
        np.testing.assert_array_equal(a.source, b.source)
        np.testing.assert_array_equal(a.weight, b.weight)

    def test_strength_degree_reproducible(self, constraints) -> None:
        fit = fit_strength_degree_poisson(
            constraints.strength_out,
            constraints.strength_in,
            constraints.degree_out,
            constraints.degree_in,
            self_loops=False,
            tolerance=1e-4,
        )
        if not fit.converged:
            pytest.skip("solver did not converge")
        a = sample_strength_degree_poisson(fit, seed=42)
        b = sample_strength_degree_poisson(fit, seed=42)
        np.testing.assert_array_equal(a.source, b.source)
        np.testing.assert_array_equal(a.weight, b.weight)


# --- Non-negativity and structure tests ---


class TestSampleStructure:
    """Sampled networks have correct basic structure."""

    def test_poisson_positive_weights_no_self_loops(self, constraints) -> None:
        fit = fit_strength_poisson(
            constraints.strength_out, constraints.strength_in, self_loops=False
        )
        sample = sample_strength_poisson(fit.x, fit.y, self_loops=False, seed=0)
        assert sample.num_edges > 0
        assert np.all(sample.weight > 0)
        assert np.all(sample.source != sample.target)

    def test_geometric_positive_weights(self, constraints) -> None:
        fit = fit_strength_geometric(
            constraints.strength_out, constraints.strength_in, self_loops=False
        )
        sample = sample_strength_geometric(fit.x, fit.y, self_loops=False, seed=0)
        assert sample.num_edges > 0
        assert np.all(sample.weight > 0)

    def test_binomial_bounded_weights(self, constraints) -> None:
        layers = constraints.binomial_layers
        fit = fit_strength_binomial(
            constraints.strength_out,
            constraints.strength_in,
            layers=layers,
            self_loops=False,
        )
        sample = sample_strength_binomial(
            fit.x, fit.y, layers=layers, self_loops=False, seed=0
        )
        assert np.all(sample.weight > 0)
        assert np.all(sample.weight <= layers)

    def test_multinomial_preserves_total(self, constraints) -> None:
        fit = fit_strength_poisson(
            constraints.strength_out, constraints.strength_in, self_loops=False
        )
        total = int(constraints.strength_out.sum())
        sample = sample_strength_multinomial(
            fit.x, fit.y, total_events=total, self_loops=False, seed=42
        )
        assert sample.total_events == total

    def test_different_seeds_produce_different_samples(self, constraints) -> None:
        fit = fit_strength_poisson(
            constraints.strength_out, constraints.strength_in, self_loops=False
        )
        a = sample_strength_poisson(fit.x, fit.y, self_loops=False, seed=1)
        b = sample_strength_poisson(fit.x, fit.y, self_loops=False, seed=2)
        # Very unlikely to be identical
        assert not np.array_equal(a.weight, b.weight) or not np.array_equal(
            a.source, b.source
        )
