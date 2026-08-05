"""E2E tests for the exact W microcanonical sampler with fixed (E,T).

Pipeline (AGENTS.md):
1. Generate a synthetic non-binary network (dense and sparse regimes).
2. Derive feasible (E, T) constraints.
3. Sample the W microcanonical fixed-(E,T) model with and without fixed
   pairs.
4. Verify exact recovery of E and T, positivity, and fixed-pair
   preservation.

Regimes: dense (large occupations, rejection path) and sparse (nearly
binary, exercises the DP fallback).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from menobis.data.frames import EdgeTable
from menobis.models.spec import Constraint, Ensemble, ModelFamily
from menobis.routing import sample_model
from menobis.utilities.synthetic import generate_pa_geographic_network

DENSE_NODE_COUNT = 30
DENSE_AVERAGE_DEGREE = DENSE_NODE_COUNT / 5
SPARSE_NODE_COUNT = 30
SPARSE_AVERAGE_DEGREE = 3.0

DENSE_LAYERS = 2
SPARSE_LAYERS = 2

N_FIXED = 10


def _generate(node_count: int, average_degree: float, events_per_edge: float):
    net = generate_pa_geographic_network(
        node_count,
        average_degree=average_degree,
        events_per_edge=events_per_edge,
        seed=7,
        self_loops=False,
    )
    return net.edges, len(net.edges), int(net.edges.occ_num.sum()), node_count


def _sample_w(
    node_count: int,
    e: int,
    t: int,
    layers: int,
    *,
    seed: int = 0,
    known=None,
) -> EdgeTable:
    kwargs: dict = {
        "ensemble": Ensemble.MICROCANONICAL,
        "family": ModelFamily.W,
        "constraint": Constraint.EDGES_EVENTS,
        "node_count": node_count,
        "target_edges": e,
        "total_events": t,
        "layers": layers,
        "self_loops": False,
        "seed": seed,
    }
    if known is not None:
        kwargs["known_source"], kwargs["known_target"], kwargs["known_occnum"] = known
    return sample_model(**kwargs)


def _verify(sampled, e, t):
    assert len(sampled) == e, f"E={len(sampled)} != {e}"
    assert int(sampled.occ_num.sum()) == t, f"T={sampled.occ_num.sum()} != {t}"
    assert (sampled.occ_num > 0).all(), "zero occupation found"
    assert (sampled.source != sampled.target).all()
    assert sampled.source.max() < 1000  # sanity


def _pick_fixed(edges, n_fixed):
    rng = np.random.default_rng(123)
    idx = np.sort(rng.choice(len(edges), size=n_fixed, replace=False))
    return (
        edges.source[idx].astype(np.uint64),
        edges.target[idx].astype(np.uint64),
        edges.occ_num[idx].astype(np.uint64),
    )


@pytest.fixture(scope="module")
def w_dense():
    return _generate(DENSE_NODE_COUNT, DENSE_AVERAGE_DEGREE, 8.0)


@pytest.fixture(scope="module")
def w_sparse():
    return _generate(SPARSE_NODE_COUNT, SPARSE_AVERAGE_DEGREE, 3.0)


@pytest.mark.parametrize("regime", ["w_dense", "w_sparse"])
def test_w_fixed_et_recovers_constraints(regime, request) -> None:
    _edges, e, t, n = request.getfixturevalue(regime)
    sampled = _sample_w(n, e, t, DENSE_LAYERS, seed=11)
    _verify(sampled, e, t)


@pytest.mark.parametrize("regime", ["w_dense", "w_sparse"])
def test_w_fixed_et_reproducible(regime, request) -> None:
    _edges, e, t, n = request.getfixturevalue(regime)
    a = _sample_w(n, e, t, DENSE_LAYERS, seed=42)
    b = _sample_w(n, e, t, DENSE_LAYERS, seed=42)
    assert (a.source == b.source).all()
    assert (a.target == b.target).all()
    assert (a.occ_num == b.occ_num).all()


@pytest.mark.parametrize("regime", ["w_dense", "w_sparse"])
def test_w_fixed_et_with_fixed_pairs(regime, request) -> None:
    edges, e, t, n = request.getfixturevalue(regime)
    known = _pick_fixed(edges, N_FIXED)
    sampled = _sample_w(n, e, t, DENSE_LAYERS, seed=17, known=known)
    _verify(sampled, e, t)
    for s, tgt, o in zip(known[0], known[1], known[2], strict=True):
        mask = (sampled.source == s) & (sampled.target == tgt)
        assert mask.any(), f"fixed ({s},{tgt}) missing"
        assert sampled.occ_num[mask][0] == o


def test_w_fixed_et_all_ones_when_t_equals_e() -> None:
    n, e, t, layers = 10, 5, 5, 2
    sampled = _sample_w(n, e, t, layers, seed=1)
    assert len(sampled) == e
    assert (sampled.occ_num == 1).all()


def test_w_fixed_et_rejects_t_below_e() -> None:
    with pytest.raises(ValueError):
        _sample_w(10, 5, 4, layers=2, seed=1)


def test_w_fixed_et_matches_exact_enumeration() -> None:
    """Tiny system: empirical vs exact W probabilities.

    L=4 pairs (2 nodes, self-loops), E=2, T=4, M=2.
    Weight ∝ C(M+t₁-1,t₁)·C(M+t₂-1,t₂).
    """
    n, e, t, layers = 2, 2, 4, 2
    pairs = [(i, j) for i in range(2) for j in range(2)]

    exact = {}
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            for n1, n2 in [(1, 3), (2, 2), (3, 1)]:
                w = math.comb(layers + n1 - 1, n1) * math.comb(layers + n2 - 1, n2)
                key = tuple(sorted([(*pairs[i], n1), (*pairs[j], n2)]))
                exact[key] = exact.get(key, 0) + w
    total = sum(exact.values())

    trials = 20_000
    counts = {}
    for seed in range(trials):
        sampled = sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.W,
            constraint=Constraint.EDGES_EVENTS,
            node_count=n,
            target_edges=e,
            total_events=t,
            layers=layers,
            self_loops=True,
            seed=seed,
        )
        config = tuple(
            sorted(
                zip(
                    sampled.source.tolist(),
                    sampled.target.tolist(),
                    sampled.occ_num.tolist(),
                    strict=True,
                )
            )
        )
        counts[config] = counts.get(config, 0) + 1

    for config, weight in exact.items():
        target_p = weight / total
        expected = target_p * trials
        observed = counts.get(config, 0)
        sigma = max((expected * (1.0 - target_p)) ** 0.5, 1.0)
        assert abs(observed - expected) < 5.0 * sigma, (
            f"config {config}: expected {expected:.1f}, observed {observed}"
        )


# ---------------------------------------------------------------------------
# M=1 and near-boundary cases
# ---------------------------------------------------------------------------


def test_w_fixed_et_m1_uniform_compositions() -> None:
    """M=1: degeneracy C(t, t)=1 → all positive compositions equally likely.

    Verify the marginal occupation distribution of a single pair is uniform
    over 1..T-E+1 (by symmetry all pairs share the same marginal).
    """
    n, e, t, layers = 6, 3, 8, 1
    trials = 8_000
    # Marginal: probability that a FIXED pair position is occupied with occ=k.
    # By exchangeability, the marginal of pair i (unconditioned on support)
    # is: P(i selected) = E/L, P(occ=k | selected) is what we test.
    # Simpler: sample and look at the histogram of occupations of all edges,
    # which should match the uniform-composition marginal.
    occ_counts = {}
    for seed in range(trials):
        s = sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.W,
            constraint=Constraint.EDGES_EVENTS,
            node_count=n,
            target_edges=e,
            total_events=t,
            layers=layers,
            self_loops=False,
            seed=seed,
        )
        for o in s.occ_num.tolist():
            occ_counts[o] = occ_counts.get(o, 0) + 1

    # Number of ordered compositions of T into E positive parts is C(T-1, E-1).
    # Each composition has equal probability 1/C(T-1,E-1).
    # The marginal of one *occupied* pair having occupation k:
    #   count over compositions where the first part = k / total compositions.
    total_compositions = math.comb(t - 1, e - 1)
    total_edges_sampled = trials * e
    for k in range(1, t - e + 2):
        # compositions with first part = k: distribute remaining T-k into E-1 parts
        n_comp_first_k = math.comb((t - k) - 1, (e - 1) - 1) if t - k >= e - 1 else 0
        expected_frac = n_comp_first_k / total_compositions
        expected_count = total_edges_sampled * expected_frac
        observed = occ_counts.get(k, 0)
        sigma = max((expected_count * (1.0 - expected_frac)) ** 0.5, 1.0)
        assert abs(observed - expected_count) < 5.0 * sigma, (
            f"M=1: occ={k} expected {expected_count:.1f}, observed {observed}"
        )


def test_w_fixed_et_near_lower_bound() -> None:
    """T = E + 1: exactly one pair has 2, the rest have 1."""
    n, e, layers = 10, 5, 2
    t = e + 1
    sampled = _sample_w(n, e, t, layers, seed=4)
    assert len(sampled) == e
    assert int(sampled.occ_num.sum()) == t
    assert (sampled.occ_num == 1).sum() == e - 1
    assert (sampled.occ_num == 2).sum() == 1
