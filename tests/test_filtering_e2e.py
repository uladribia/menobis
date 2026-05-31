"""End-to-end filtering tests: false-positive rate under null model.

Pipeline: generate PA-geographic → derive → fit → sample from null → filter.
Verifies that filtering null samples produces FPR <= alpha + stochastic tolerance.
Uses N=20 for speed.
"""

from __future__ import annotations

import pytest

from menobis.filtering.models import (
    _filter_strength_binomial,
    _filter_strength_degree_poisson,
    _filter_strength_edges_poisson,
    _filter_strength_geometric,
    _filter_strength_poisson,
)
from menobis.models.fitting import (
    _fit_strength_binomial,
    _fit_strength_degree_poisson,
    _fit_strength_edges_poisson,
    _fit_strength_geometric,
    _fit_strength_poisson,
)
from menobis.models.generation import (
    _sample_strength_binomial,
    _sample_strength_degree_poisson,
    _sample_strength_edges_poisson,
    _sample_strength_geometric,
    _sample_strength_poisson,
)
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

N = 20
SEED = 54320
ALPHA = 0.05
# Allow FPR up to 3x alpha (stochastic tolerance at small sample sizes)
FPR_TOLERANCE = 3.0 * ALPHA
NULL_SAMPLES = 5


@pytest.fixture(scope="module")
def network():
    return generate_pa_geographic_network(
        N, average_degree=5.0, events_per_edge=4.0, seed=SEED, self_loops=False
    )


@pytest.fixture(scope="module")
def constraints(network):
    return derive_synthetic_constraints(network)


def _compute_fpr(
    filter_func, sample_func, fit, *, extra_filter_kwargs=None, extra_sample_kwargs=None
):
    """Compute empirical FPR by filtering null samples."""
    extra_filter_kwargs = extra_filter_kwargs or {}
    extra_sample_kwargs = extra_sample_kwargs or {}
    false_positives = 0
    total_edges = 0
    for offset in range(NULL_SAMPLES):
        sample = sample_func(seed=SEED + 1000 + offset, **extra_sample_kwargs)
        result = filter_func(
            sample, fit, alpha=ALPHA, tail="upper", **extra_filter_kwargs
        )
        false_positives += result.upper.edges.num_edges
        total_edges += sample.num_edges
    return false_positives / max(total_edges, 1)


class TestMEFilteringFPR:
    """ME family filtering produces bounded FPR."""

    def test_strength_fpr(self, constraints) -> None:
        fit = _fit_strength_poisson(
            constraints.strength_out, constraints.strength_in, self_loops=False
        )
        fpr = _compute_fpr(
            _filter_strength_poisson,
            lambda seed, **kw: _sample_strength_poisson(
                fit.x, fit.y, self_loops=False, seed=seed
            ),
            fit,
        )
        assert fpr <= FPR_TOLERANCE, f"FPR={fpr:.4f} exceeds tolerance"

    def test_strength_edges_fpr(self, constraints) -> None:
        fit = _fit_strength_edges_poisson(
            constraints.strength_out,
            constraints.strength_in,
            constraints.total_edges,
            self_loops=False,
        )
        fpr = _compute_fpr(
            _filter_strength_edges_poisson,
            lambda seed, **kw: _sample_strength_edges_poisson(fit, seed=seed),
            fit,
        )
        assert fpr <= FPR_TOLERANCE, f"FPR={fpr:.4f} exceeds tolerance"

    def test_strength_degree_fpr(self, constraints) -> None:
        fit = _fit_strength_degree_poisson(
            constraints.strength_out,
            constraints.strength_in,
            constraints.degree_out,
            constraints.degree_in,
            self_loops=False,
            tolerance=1e-4,
        )
        if not fit.converged:
            pytest.skip("solver did not converge")
        fpr = _compute_fpr(
            _filter_strength_degree_poisson,
            lambda seed, **kw: _sample_strength_degree_poisson(fit, seed=seed),
            fit,
        )
        assert fpr <= FPR_TOLERANCE, f"FPR={fpr:.4f} exceeds tolerance"


class TestBFilteringFPR:
    """B family filtering produces bounded FPR."""

    def test_strength_fpr(self, constraints) -> None:
        fit = _fit_strength_binomial(
            constraints.strength_out,
            constraints.strength_in,
            layers=constraints.binomial_layers,
            self_loops=False,
        )
        fpr = _compute_fpr(
            _filter_strength_binomial,
            lambda seed, **kw: _sample_strength_binomial(
                fit.x,
                fit.y,
                layers=constraints.binomial_layers,
                self_loops=False,
                seed=seed,
            ),
            fit,
            extra_filter_kwargs={"layers": constraints.binomial_layers},
        )
        assert fpr <= FPR_TOLERANCE, f"FPR={fpr:.4f} exceeds tolerance"


class TestWFilteringFPR:
    """W family filtering produces bounded FPR."""

    def test_strength_fpr(self, constraints) -> None:
        fit = _fit_strength_geometric(
            constraints.strength_out, constraints.strength_in, self_loops=False
        )
        fpr = _compute_fpr(
            _filter_strength_geometric,
            lambda seed, **kw: _sample_strength_geometric(
                fit.x, fit.y, self_loops=False, seed=seed
            ),
            fit,
        )
        assert fpr <= FPR_TOLERANCE, f"FPR={fpr:.4f} exceeds tolerance"
