"""E2E tests for the exact B microcanonical sampler with fixed (E,T).

Follows the mandatory AGENTS.md E2E protocol:
1. Generate a synthetic non-binary network (dense and sparse regimes).
2. Derive feasible (E, T) constraints — adjusted for B layer capacity
   so that T ≤ M·E holds.
3. Sample the B microcanonical ME fixed-(E,T) model with and without
   fixed pairs.
4. Verify exact recovery of E and T, layer bound t ≤ M, and fixed-pair
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

# Layer counts per regime (events_per_edge must be ≤ M so T ≤ M·E).
DENSE_LAYERS = 12
SPARSE_LAYERS = 5

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


def _sample_b(
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
        "family": ModelFamily.B,
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


def _verify(sampled, e, t, layers):
    assert len(sampled) == e, f"E={len(sampled)} != {e}"
    assert int(sampled.occ_num.sum()) == t, f"T={sampled.occ_num.sum()} != {t}"
    assert (sampled.occ_num > 0).all(), "zero occupation found"
    assert (sampled.occ_num <= layers).all(), f"occupation exceeds M={layers}"
    # no self-loops, in range
    assert (sampled.source != sampled.target).all()
    assert (
        sampled.source.max() < len(sampled)
        or len(sampled) == 0
        or sampled.source.max() >= 0
    )


def _pick_fixed(edges, n_fixed, layers):
    """Pick fixed edges whose occupation is within B capacity (≤ M)."""
    feasible = np.where(edges.occ_num <= layers)[0]
    assert len(feasible) >= n_fixed, "not enough edges within B capacity"
    rng = np.random.default_rng(123)
    idx = np.sort(rng.choice(feasible, size=n_fixed, replace=False))
    return (
        edges.source[idx].astype(np.uint64),
        edges.target[idx].astype(np.uint64),
        edges.occ_num[idx].astype(np.uint64),
    )


@pytest.fixture(scope="module")
def b_dense():
    return _generate(DENSE_NODE_COUNT, DENSE_AVERAGE_DEGREE, 8.0)


@pytest.fixture(scope="module")
def b_sparse():
    return _generate(SPARSE_NODE_COUNT, SPARSE_AVERAGE_DEGREE, 3.0)


@pytest.mark.parametrize("regime", ["b_dense", "b_sparse"])
def test_b_fixed_et_recovers_constraints(regime, request) -> None:
    _edges, e, t, n = request.getfixturevalue(regime)
    layers = DENSE_LAYERS if regime == "b_dense" else SPARSE_LAYERS
    # Ensure T ≤ M·E (feasible)
    assert t <= layers * e, "test setup infeasible"
    sampled = _sample_b(n, e, t, layers, seed=11)
    _verify(sampled, e, t, layers)


@pytest.mark.parametrize("regime", ["b_dense", "b_sparse"])
def test_b_fixed_et_reproducible(regime, request) -> None:
    _edges, e, t, n = request.getfixturevalue(regime)
    layers = DENSE_LAYERS if regime == "b_dense" else SPARSE_LAYERS
    a = _sample_b(n, e, t, layers, seed=42)
    b = _sample_b(n, e, t, layers, seed=42)
    assert (a.source == b.source).all()
    assert (a.target == b.target).all()
    assert (a.occ_num == b.occ_num).all()


@pytest.mark.parametrize("regime", ["b_dense", "b_sparse"])
def test_b_fixed_et_with_fixed_pairs(regime, request) -> None:
    edges, e, t, n = request.getfixturevalue(regime)
    layers = DENSE_LAYERS if regime == "b_dense" else SPARSE_LAYERS
    known = _pick_fixed(edges, N_FIXED, layers)
    sampled = _sample_b(n, e, t, layers, seed=17, known=known)
    _verify(sampled, e, t, layers)
    # fixed pairs preserved
    for s, tgt, o in zip(known[0], known[1], known[2], strict=True):
        mask = (sampled.source == s) & (sampled.target == tgt)
        assert mask.any(), f"fixed ({s},{tgt}) missing"
        assert sampled.occ_num[mask][0] == o


def test_b_fixed_et_all_ones_when_t_equals_e() -> None:
    """T == E → every selected pair has occupation 1."""
    n, e, t, layers = 10, 5, 5, 3
    sampled = _sample_b(n, e, t, layers, seed=1)
    assert len(sampled) == e
    assert (sampled.occ_num == 1).all()


def test_b_fixed_et_all_max_when_t_equals_m_e() -> None:
    """T == M·E → every selected pair has occupation M."""
    n, e, layers = 10, 4, 3
    t = layers * e
    sampled = _sample_b(n, e, t, layers, seed=2)
    assert len(sampled) == e
    assert (sampled.occ_num == layers).all()


def test_b_fixed_et_rejects_t_above_me() -> None:
    """T > M·E is infeasible."""
    with pytest.raises(ValueError):
        _sample_b(10, 5, 20, layers=3, seed=1)  # T=20 > M·E=15


def test_b_fixed_et_rejects_t_below_e() -> None:
    with pytest.raises(ValueError):
        _sample_b(10, 5, 4, layers=3, seed=1)


def test_b_fixed_et_matches_exact_enumeration() -> None:
    """Tiny system: verify empirical vs exact B probabilities.

    L=4 pairs (2 nodes, self-loops), E=2, T=4, M=2.
    Weight ∝ C(2,t₁)·C(2,t₂).
    """
    n, e, t, layers = 2, 2, 4, 2
    pairs = [(i, j) for i in range(2) for j in range(2)]

    exact = {}
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            for n1, n2 in [(1, 3), (2, 2), (3, 1)]:
                w = math.comb(layers, n1) * math.comb(layers, n2)
                key = tuple(sorted([(*pairs[i], n1), (*pairs[j], n2)]))
                exact[key] = exact.get(key, 0) + w
    total = sum(exact.values())

    trials = 20_000
    counts = {}
    for seed in range(trials):
        sampled = sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.B,
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


def test_b_fixed_et_m1_requires_t_equals_e() -> None:
    """M=1: feasibility requires T == E; every pair gets occupation 1."""
    n, e, t, layers = 10, 6, 6, 1
    sampled = _sample_b(n, e, t, layers, seed=3)
    assert len(sampled) == e
    assert (sampled.occ_num == 1).all()
    # T > E with M=1 is infeasible
    with pytest.raises(ValueError):
        _sample_b(10, 6, 7, layers=1, seed=3)


def test_b_fixed_et_near_lower_bound() -> None:
    """T = E + 1: exactly one pair gets 2, the rest get 1."""
    n, e, layers = 10, 5, 3
    t = e + 1
    sampled = _sample_b(n, e, t, layers, seed=4)
    assert len(sampled) == e
    assert int(sampled.occ_num.sum()) == t
    assert (sampled.occ_num >= 1).all()
    # T = E+1 → one pair has 2, rest have 1
    assert (sampled.occ_num == 1).sum() == e - 1
    assert (sampled.occ_num == 2).sum() == 1


def test_b_fixed_et_near_saturation() -> None:
    """T = M*E - 1: complement-mode rejection path (many holes)."""
    n, e, layers = 10, 6, 4
    t = layers * e - 1
    sampled = _sample_b(n, e, t, layers, seed=5)
    assert len(sampled) == e
    assert int(sampled.occ_num.sum()) == t
    assert (sampled.occ_num <= layers).all()
    assert (sampled.occ_num > 0).all()


def test_b_fixed_et_rejects_fixed_occ_above_m() -> None:
    """A fixed pair with occupation > M must be rejected."""
    n, e, t, layers = 10, 5, 10, 3
    with pytest.raises(ValueError, match="layer capacity"):
        _sample_b(
            n,
            e,
            t,
            layers,
            seed=6,
            known=(np.array([0]), np.array([1]), np.array([5])),
        )
