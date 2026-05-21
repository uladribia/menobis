"""E2E pipeline tests: generate → derive → fit → sample → verify.

These tests follow the mandatory testing policy: constraints are always
derived from a generated network, guaranteeing feasibility.
"""

import numpy as np

from odme.data.frames import EdgeTable
from odme.models import (
    fit_degree_events_poisson,
    fit_strength_binomial,
    fit_strength_cost_poisson_coordinates,
    fit_strength_degree_poisson,
    fit_strength_edges_poisson,
    fit_strength_poisson,
    sample_degree_events_poisson,
    sample_strength_binomial,
    sample_strength_cost_poisson,
    sample_strength_degree_poisson,
    sample_strength_edges_poisson,
    sample_strength_poisson,
)


def _generate_network(
    n: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a realistic weighted directed network with coordinates.

    Uses a gravity model: rates = activity_i * attract_j * exp(-gamma * dist_ij).
    """
    rng = np.random.default_rng(seed)
    cx = rng.normal(size=n) * 10.0
    cy = rng.normal(size=n) * 10.0
    # Distance matrix
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    # Gravity rates
    activity = rng.pareto(1.2, size=n) + 1.0
    attract = rng.pareto(1.2, size=n) + 1.0
    gamma = 0.15
    rates = np.outer(activity, attract) * np.exp(-gamma * dist)
    np.fill_diagonal(rates, 0.0)  # no self-loops
    weights = rng.poisson(rates).astype(int)
    return weights, cx, cy, dist


def _derive_constraints(weights: np.ndarray, dist: np.ndarray) -> dict[str, object]:
    """Derive all constraint sequences from an observed network."""
    s_out = weights.sum(axis=1).astype(np.float64)
    s_in = weights.sum(axis=0).astype(np.float64)
    k_out = (weights > 0).sum(axis=1).astype(np.float64)
    k_in = (weights > 0).sum(axis=0).astype(np.float64)
    total_edges = float((weights > 0).sum())
    total_events = int(weights.sum())
    total_cost = float((weights * dist).sum())
    return {
        "s_out": s_out,
        "s_in": s_in,
        "k_out": k_out,
        "k_in": k_in,
        "total_edges": total_edges,
        "total_events": total_events,
        "total_cost": total_cost,
    }


def _check_strength_recovery(
    sample: EdgeTable,
    s_out: np.ndarray,
    s_in: np.ndarray,
    tolerance: float,
) -> None:
    """Assert sampled strengths recover input constraints within tolerance."""
    n = len(s_out)
    sampled_out = np.zeros(n)
    sampled_in = np.zeros(n)
    np.add.at(sampled_out, sample.source.astype(int), sample.weight.astype(float))
    np.add.at(sampled_in, sample.target.astype(int), sample.weight.astype(float))
    # Use 5x fitting tolerance for stochastic tolerance
    np.testing.assert_allclose(
        sampled_out,
        s_out,
        atol=tolerance * 5,
        err_msg="Out-strength mismatch",
    )
    np.testing.assert_allclose(
        sampled_in,
        s_in,
        atol=tolerance * 5,
        err_msg="In-strength mismatch",
    )


class TestMEStrengthE2E:
    """E2E pipeline for ME (Poisson) fixed-strength model."""

    def test_generate_fit_sample_verify(self) -> None:
        """Full pipeline: generate → derive → fit → sample → verify.

        Single-sample tolerance: 3 * sqrt(max_strength) accounts for
        Poisson stochastic noise. The ensemble test below is the proper
        statistical check.
        """
        n = 25
        weights, _cx, _cy, dist = _generate_network(n, seed=100)
        constraints = _derive_constraints(weights, dist)
        s_out = constraints["s_out"]
        s_in = constraints["s_in"]

        # Balance (may differ by rounding)
        diff = s_out.sum() - s_in.sum()
        if abs(diff) > 0.01:
            s_in[0] += diff

        # Fit
        fit = fit_strength_poisson(s_out, s_in, self_loops=False, tolerance=1e-4)
        assert fit.converged

        # Sample using correct API: (x, y, *, self_loops, seed)
        sample = sample_strength_poisson(
            fit.x, fit.y, self_loops=fit.self_loops, seed=42
        )
        assert isinstance(sample, EdgeTable)

        # Verify constraint recovery (single sample, Poisson noise ~ sqrt(lambda))
        # For max over n nodes, use 4*sqrt(max_strength) as generous bound
        tol = max(4.0 * np.sqrt(s_out.max()), 10.0)
        _check_strength_recovery(sample, s_out, s_in, tolerance=tol)

    def test_ensemble_z_score(self) -> None:
        """Ensemble sampling should recover constraints with z < 3."""
        n = 25
        weights, _cx, _cy, dist = _generate_network(n, seed=200)
        constraints = _derive_constraints(weights, dist)
        s_out = constraints["s_out"]
        s_in = constraints["s_in"]
        diff = s_out.sum() - s_in.sum()
        if abs(diff) > 0.01:
            s_in[0] += diff

        fit = fit_strength_poisson(s_out, s_in, self_loops=False, tolerance=1e-6)
        assert fit.converged

        # Ensemble
        ensemble_size = 200
        sums_out = np.zeros((ensemble_size, n))
        for i in range(ensemble_size):
            sample = sample_strength_poisson(
                fit.x, fit.y, self_loops=fit.self_loops, seed=1000 + i
            )
            np.add.at(
                sums_out[i], sample.source.astype(int), sample.weight.astype(float)
            )

        mean_out = sums_out.mean(axis=0)
        se_out = sums_out.std(axis=0) / np.sqrt(ensemble_size)
        # z-score where SE > 0
        mask = se_out > 0
        z_scores = np.abs((mean_out[mask] - s_out[mask]) / se_out[mask])
        assert z_scores.max() < 3.0, f"Max z-score {z_scores.max():.2f} >= 3"


class TestBStrengthE2E:
    """E2E pipeline for B (Binomial) fixed-strength model."""

    def test_generate_fit_sample_verify(self) -> None:
        """Full pipeline for binomial strength.

        Single-sample tolerance: 4*sqrt(max_strength) for stochastic noise.
        """
        n = 25
        weights, _cx, _cy, dist = _generate_network(n, seed=300)
        constraints = _derive_constraints(weights, dist)
        s_out = constraints["s_out"]
        s_in = constraints["s_in"]
        diff = s_out.sum() - s_in.sum()
        if abs(diff) > 0.01:
            s_in[0] += diff
        fit_tol = 0.02 * max(s_out.max(), 1.0)

        fit = fit_strength_binomial(
            s_out, s_in, layers=10, self_loops=False, tolerance=fit_tol
        )
        assert fit.converged

        # Sample: (x, y, *, layers, self_loops, seed)
        sample = sample_strength_binomial(
            fit.x,
            fit.y,
            layers=fit.layers or 10,
            self_loops=fit.self_loops,
            seed=42,
        )
        assert isinstance(sample, EdgeTable)

        # Verify (single sample, Binomial noise ~ sqrt(M*p*(1-p)))
        tol = max(4.0 * np.sqrt(s_out.max()), 10.0)
        _check_strength_recovery(sample, s_out, s_in, tolerance=tol)


class TestMEStrengthCostE2E:
    """E2E pipeline for ME strength-cost (coordinate) model."""

    def test_generate_fit_sample_verify(self) -> None:
        """Full pipeline for ME strength-cost with coordinates."""
        n = 25
        weights, cx, cy, dist = _generate_network(n, seed=400)
        constraints = _derive_constraints(weights, dist)
        s_out = constraints["s_out"]
        s_in = constraints["s_in"]
        total_cost = constraints["total_cost"]
        diff = s_out.sum() - s_in.sum()
        if abs(diff) > 0.01:
            s_in[0] += diff
        tol = 0.02 * max(s_out.max(), 1.0)

        fit = fit_strength_cost_poisson_coordinates(
            s_out, s_in, cx, cy, total_cost, self_loops=False, tolerance=tol
        )
        assert fit.converged

        # Cost triples for sampler
        n_ = len(s_out)
        ii, jj = np.meshgrid(range(n_), range(n_), indexing="ij")
        mask = ii != jj  # no self-loops
        cost_src = ii[mask].ravel().astype(np.uint64)
        cost_tgt = jj[mask].ravel().astype(np.uint64)
        cost_val = dist[mask].ravel().astype(np.float64)

        # Sample: (fit, cost_sources, cost_targets, cost_values, *, seed)
        sample = sample_strength_cost_poisson(
            fit, cost_src, cost_tgt, cost_val, seed=42
        )
        assert isinstance(sample, EdgeTable)
        _check_strength_recovery(sample, s_out, s_in, tolerance=tol * 2)


class TestMEStrengthEdgesE2E:
    """E2E pipeline for ME strength-edges model."""

    def test_generate_fit_sample_verify(self) -> None:
        """Full pipeline for ME strength-edges."""
        n = 25
        weights, _cx, _cy, dist = _generate_network(n, seed=500)
        constraints = _derive_constraints(weights, dist)
        s_out = constraints["s_out"]
        s_in = constraints["s_in"]
        total_edges = constraints["total_edges"]
        diff = s_out.sum() - s_in.sum()
        if abs(diff) > 0.01:
            s_in[0] += diff
        tol = 0.02 * max(s_out.max(), 1.0)

        fit = fit_strength_edges_poisson(
            s_out, s_in, total_edges, self_loops=False, tolerance=tol
        )
        assert fit.converged

        # Sample: (fit, *, seed)
        sample = sample_strength_edges_poisson(fit, seed=42)
        assert isinstance(sample, EdgeTable)
        _check_strength_recovery(sample, s_out, s_in, tolerance=tol * 2)


class TestMEStrengthDegreeE2E:
    """E2E pipeline for ME strength-degree model."""

    def test_generate_fit_sample_verify(self) -> None:
        """Full pipeline for ME strength-degree.

        Single-sample tolerance: 4*sqrt(max_strength).
        """
        n = 25
        weights, _cx, _cy, dist = _generate_network(n, seed=600)
        constraints = _derive_constraints(weights, dist)
        s_out = constraints["s_out"]
        s_in = constraints["s_in"]
        k_out = constraints["k_out"]
        k_in = constraints["k_in"]
        diff = s_out.sum() - s_in.sum()
        if abs(diff) > 0.01:
            s_in[0] += diff
        fit_tol = 0.02 * max(s_out.max(), 1.0)

        fit = fit_strength_degree_poisson(
            s_out, s_in, k_out, k_in, self_loops=False, tolerance=fit_tol
        )
        assert fit.converged

        # Sample: (fit, *, seed)
        sample = sample_strength_degree_poisson(fit, seed=42)
        assert isinstance(sample, EdgeTable)
        tol = max(4.0 * np.sqrt(s_out.max()), 10.0)
        _check_strength_recovery(sample, s_out, s_in, tolerance=tol)


class TestMEDegreeEventsE2E:
    """E2E pipeline for ME degree-events model."""

    def test_generate_fit_sample_verify(self) -> None:
        """Full pipeline for ME degree-events."""
        n = 25
        weights, _cx, _cy, dist = _generate_network(n, seed=700)
        constraints = _derive_constraints(weights, dist)
        k_out = constraints["k_out"]
        k_in = constraints["k_in"]
        total_events = constraints["total_events"]
        tol = 0.02 * max(k_out.max(), 1.0)

        fit = fit_degree_events_poisson(
            k_out, k_in, total_events, self_loops=False, tolerance=tol
        )
        assert fit.converged

        # Sample: (fit, *, total_events, seed, self_loops)
        sample = sample_degree_events_poisson(
            fit, total_events=total_events, seed=42, self_loops=fit.self_loops
        )
        assert isinstance(sample, EdgeTable)

        # Verify degree recovery (binary occupation)
        n_ = len(k_out)
        sampled_k_out = np.zeros(n_)
        sampled_k_in = np.zeros(n_)
        # Count unique targets per source
        for i in range(n_):
            sampled_k_out[i] = np.sum(sample.source == i)
        for j in range(n_):
            sampled_k_in[j] = np.sum(sample.target == j)
        # Stochastic tolerance for degree (generous)
        degree_tol = max(tol * 5, 5.0)
        np.testing.assert_allclose(
            sampled_k_out, k_out, atol=degree_tol, err_msg="k_out mismatch"
        )
