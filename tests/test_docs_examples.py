"""Deterministic smoke tests for the documented public workflows (§51).

Each test mirrors one critical public workflow exactly as it appears in the
public documentation. If a documentation example changes, the corresponding
test must change in the same commit (documentation rewrite plan §51, §90).

Constraints are ALWAYS derived from a valid witness network
(``generate_pa_geographic_network`` + ``derive_synthetic_constraints``) so
they are feasible by construction; no hand-picked arbitrary sequences are
used (AGENTS.md testing policy).

Tolerances (documented per test):
- grand-canonical samples fluctuate stochastically around fitted
  expectations, so single-sample checks are contract-level (convergence,
  reproducibility, total-event envelope);
- microcanonical routes fix their constrained quantities realization by
  realization, so exact recovery is asserted;
- the microcanonical strength+cost route is hybrid: strengths exact, cost
  matched in expectation through gamma fitting, asserted within 5%
  relative (stochastic).
"""

from __future__ import annotations

import numpy as np
import pytest

from menobis.analysis import (
    clustering_coefficient,
    compute_all_stats,
    occupation_clustering_coefficient,
    occupation_distribution,
)
from menobis.filtering import filter_model
from menobis.models import Constraint, Ensemble, ModelFamily, fit_model, sample_model
from menobis.routing import sample_model_detailed
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

SEED = 7
SELF_LOOPS = False


def _dense_network(node_count: int, *, seed: int = SEED) -> object:
    """Build a PA-geographic network in the dense regime (unsaturated support)."""
    return generate_pa_geographic_network(
        node_count,
        average_degree=node_count / 5.0,
        events_per_edge=8.0,
        seed=seed,
        self_loops=SELF_LOOPS,
    )


@pytest.fixture(scope="module")
def network() -> object:
    """A dense synthetic witness network used by the common ME workflows."""
    return _dense_network(25)


@pytest.fixture(scope="module")
def constraints(network: object) -> object:
    """Constraints derived from the witness network (feasible by construction)."""
    return derive_synthetic_constraints(network)


def _assert_total_events_envelope(sample: object, target: int) -> None:
    """Grand-canonical single draws fluctuate; keep a generous envelope."""
    total = int(sample.occ_num.sum())
    assert 0.5 * target <= total <= 2.0 * target, f"total events {total} vs {target}"


def test_getting_started_gc_strength_fit_and_sample(
    network: object, constraints: object
) -> None:
    """Getting-started workflow: GC ME strength fit + sample (§11, §51.1)."""
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=constraints.strength_out,
        strength_in=constraints.strength_in,
        self_loops=SELF_LOOPS,
    )
    assert fit.converged, fit.status

    result = sample_model_detailed(
        ensemble=Ensemble.GRAND_CANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        seed=3,
    )
    assert result.edges.num_edges > 0
    assert result.diagnostics.exactness == "exact_independent"
    _assert_total_events_envelope(result.edges, int(network.edges.total_events))

    # One high-level statistic on a sampled network.
    stats = compute_all_stats(result.edges)
    assert len(stats.strength_out) == 25


def test_filtering_workflow(network: object, constraints: object) -> None:
    """Filtering workflow: fit with convergence guard, filter with FDR (§27, §51.2)."""
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=constraints.strength_out,
        strength_in=constraints.strength_in,
        self_loops=SELF_LOOPS,
    )
    assert fit.converged, fit.status

    result = filter_model(
        network.edges,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        self_loops=SELF_LOOPS,
        correction="fdr",
    )
    assert result.upper is not None
    assert result.lower is not None
    reported = result.upper.edges.num_edges + result.lower.edges.num_edges
    assert reported < network.edges.num_edges


def test_ensemble_statistics_workflow(network: object) -> None:
    """Ensemble-statistics workflow: high-level metrics (§26, §51.3)."""
    stats = compute_all_stats(network.edges)
    assert len(stats.strength_out) == len(stats.degree_out)

    occ = occupation_distribution(network.edges)
    assert occ is not None

    binary_cc = clustering_coefficient(network.edges)
    occ_cc = occupation_clustering_coefficient(network.edges)
    assert binary_cc is not None and occ_cc is not None


def test_spatial_cost_gc_workflow(network: object, constraints: object) -> None:
    """Spatial-cost guide: GC strength+cost fit and sample (§24, §51.4)."""
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_COST,
        strength_out=constraints.strength_out,
        strength_in=constraints.strength_in,
        coord_x=network.x,
        coord_y=network.y,
        target_cost=float(constraints.total_cost),
        self_loops=SELF_LOOPS,
    )
    assert fit.converged, fit.status

    result = sample_model_detailed(
        ensemble=Ensemble.GRAND_CANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_COST,
        fit=fit,
        coord_x=network.x,
        coord_y=network.y,
        seed=4,
    )
    assert result.edges.num_edges > 0
    assert result.diagnostics.exactness == "exact_independent"


def test_fixed_pairs_partial_fit_residualization(
    network: object, constraints: object
) -> None:
    """Fixed-pairs guide: partial fit with a known positive occupation (§25, §51.5)."""
    known_source = network.edges.source[:5]
    known_target = network.edges.target[:5]
    known_occnum = network.edges.occ_num[:5]

    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=constraints.strength_out,
        strength_in=constraints.strength_in,
        known_source=known_source,
        known_target=known_target,
        known_occnum=known_occnum,
        self_loops=SELF_LOOPS,
    )
    assert fit.converged, fit.status


def test_microcanonical_edges_events(network: object) -> None:
    """Microcanonical (E,T): exact direct sampler, E and T exact (§51.6)."""
    target_edges = int(network.edges.num_edges)
    total_events = int(network.edges.total_events)
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.EDGES_EVENTS,
        node_count=len(network.x),
        target_edges=target_edges,
        total_events=total_events,
        self_loops=SELF_LOOPS,
        seed=6,
    )
    assert sample.num_edges == target_edges
    assert sample.total_events == total_events


def test_microcanonical_degree_events(network: object, constraints: object) -> None:
    """Microcanonical (k,T): degree sequences and T exact (§51.7)."""
    k_out = constraints.degree_out.astype(np.uint32)
    k_in = constraints.degree_in.astype(np.uint32)
    total_events = int(network.edges.total_events)
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.DEGREE_EVENTS,
        degree_out=k_out,
        degree_in=k_in,
        total_events=total_events,
        self_loops=SELF_LOOPS,
        seed=7,
    )
    s_out = np.zeros(len(k_out), dtype=np.int64)
    s_in = np.zeros(len(k_in), dtype=np.int64)
    np.add.at(s_out, sample.source, sample.occ_num)
    np.add.at(s_in, sample.target, sample.occ_num)
    k_out_sampled = np.bincount(sample.source, minlength=len(k_out)).astype(np.int64)
    k_in_sampled = np.bincount(sample.target, minlength=len(k_in)).astype(np.int64)
    assert np.array_equal(k_out_sampled, k_out)
    assert np.array_equal(k_in_sampled, k_in)
    assert sample.total_events == total_events


def test_microcanonical_fixed_strength(network: object, constraints: object) -> None:
    """Microcanonical fixed strengths: strengths exact, stationary MCMC (§51.8)."""
    s_out = constraints.strength_out.astype(np.uint64)
    s_in = constraints.strength_in.astype(np.uint64)
    result = sample_model_detailed(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
        self_loops=SELF_LOOPS,
        seed=8,
    )
    sampled_s_out = np.bincount(result.edges.source, weights=result.edges.occ_num)
    sampled_s_in = np.bincount(result.edges.target, weights=result.edges.occ_num)
    assert np.array_equal(np.round(sampled_s_out).astype(np.uint64), s_out)
    assert np.array_equal(np.round(sampled_s_in).astype(np.uint64), s_in)
    assert result.diagnostics.exactness == "exact_stationary_mcmc"


def test_microcanonical_strength_edges(network: object, constraints: object) -> None:
    """Microcanonical (s,E): strengths and E exact (§51.9)."""
    s_out = constraints.strength_out.astype(np.uint64)
    s_in = constraints.strength_in.astype(np.uint64)
    target_edges = int(network.edges.num_edges)
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=target_edges,
        self_loops=SELF_LOOPS,
        seed=9,
    )
    assert sample.num_edges == target_edges
    sampled_s_out = np.bincount(sample.source, weights=sample.occ_num)
    sampled_s_in = np.bincount(sample.target, weights=sample.occ_num)
    assert np.array_equal(np.round(sampled_s_out).astype(np.uint64), s_out)
    assert np.array_equal(np.round(sampled_s_in).astype(np.uint64), s_in)


def test_microcanonical_strength_degree(network: object, constraints: object) -> None:
    """Microcanonical (s,k): strengths and degree sequences exact (§51.10)."""
    s_out = constraints.strength_out.astype(np.uint64)
    s_in = constraints.strength_in.astype(np.uint64)
    k_out = constraints.degree_out.astype(np.uint32)
    k_in = constraints.degree_in.astype(np.uint32)
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_DEGREE,
        strength_out=s_out,
        strength_in=s_in,
        degree_out=k_out,
        degree_in=k_in,
        self_loops=SELF_LOOPS,
        seed=10,
    )
    sampled_s_out = np.bincount(sample.source, weights=sample.occ_num)
    sampled_s_in = np.bincount(sample.target, weights=sample.occ_num)
    sampled_k_out = np.bincount(sample.source, minlength=len(k_out))
    sampled_k_in = np.bincount(sample.target, minlength=len(k_in))
    assert np.array_equal(np.round(sampled_s_out).astype(np.uint64), s_out)
    assert np.array_equal(np.round(sampled_s_in).astype(np.uint64), s_in)
    assert np.array_equal(sampled_k_out, k_out)
    assert np.array_equal(sampled_k_in, k_in)


def test_microcanonical_strength_cost(network: object, constraints: object) -> None:
    """Microcanonical strength+cost: strengths exact, cost expected (§51.11).

    Hybrid semantics: strengths are exact; cost is matched in expectation
    through gamma fitting. Small N keeps the gamma-fit MCMC loop fast.
    """
    net = _dense_network(10, seed=SEED)
    c = derive_synthetic_constraints(net)
    s_out = c.strength_out.astype(np.uint64)
    s_in = c.strength_in.astype(np.uint64)
    target_cost = float(c.total_cost)
    result = sample_model_detailed(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_COST,
        strength_out=s_out,
        strength_in=s_in,
        coord_x=net.x,
        coord_y=net.y,
        target_cost=target_cost,
        self_loops=SELF_LOOPS,
        seed=11,
        max_iterations=60,
        samples_per_iteration=10,
        estimation_sweeps=50,
        adaptation_sweeps=50,
    )
    sampled_s_out = np.bincount(result.edges.source, weights=result.edges.occ_num)
    sampled_s_in = np.bincount(result.edges.target, weights=result.edges.occ_num)
    assert np.array_equal(np.round(sampled_s_out).astype(np.uint64), s_out)
    assert np.array_equal(np.round(sampled_s_in).astype(np.uint64), s_in)
    assert result.diagnostics.exactness == "exact_stationary_mcmc"
    expected_cost = result.diagnostics.expected_cost
    assert expected_cost is not None
    rel_residual = abs(expected_cost - target_cost) / target_cost
    assert rel_residual <= 0.05, f"cost residual {rel_residual:.3f}"
