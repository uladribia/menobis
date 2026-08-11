"""E2E tests for the exact ME microcanonical sampler with fixed (E,T).

Pipeline follows the mandatory AGENTS.md E2E protocol:

1. Generate a realistic non-binary network with the synthetic gravity-like
   module (dense and sparse regimes).
2. Derive hard constraints (E = occupied-pair count, T = total occupation)
   from the generated network.  These are guaranteed feasible because they
   come from an actual network.
3. Sample the microcanonical ME fixed-(E,T) model, with and without fixed
   (known) pairs.
4. Verify the sampled network recovers the original constraints exactly.

Regimes:

- **dense**:  `average_degree=N/5`, `events_per_edge=8.0` (unsaturated
  binary constraints, non-sparse occupations; rejection fast path).
- **sparse**: `average_degree=3.0`, `events_per_edge=3.0` (nearly binary
  occupations; exercises the Stirling fallback path).

Tolerances: the hard constraints E and T are exact (integer equality), so
no statistical tolerance is needed for the constraint checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from menobis.data.frames import EdgeTable
from menobis.models.spec import Constraint, Ensemble, ModelFamily
from menobis.routing import sample_model
from menobis.utilities.synthetic import generate_pa_geographic_network

# ---------------------------------------------------------------------------
# Regime parameters (match the Phase-0 benchmark regimes)
# ---------------------------------------------------------------------------

DENSE_NODE_COUNT = 30
DENSE_AVERAGE_DEGREE = DENSE_NODE_COUNT / 5  # N/5
DENSE_EVENTS_PER_EDGE = 8.0

SPARSE_NODE_COUNT = 30
SPARSE_AVERAGE_DEGREE = 3.0
SPARSE_EVENTS_PER_EDGE = 3.0

N_FIXED = 10  # number of fixed pairs used in fixed-pair tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _generate(node_count: int, average_degree: float, events_per_edge: float) -> tuple:
    """Generate a synthetic network and return (edges, E, T, node_count)."""
    net = generate_pa_geographic_network(
        node_count,
        average_degree=average_degree,
        events_per_edge=events_per_edge,
        seed=7,
        self_loops=False,
    )
    edges = net.edges
    e = len(edges)
    t = int(edges.occ_num.sum())
    return edges, e, t, node_count


@pytest.fixture(scope="module")
def dense_network() -> tuple:
    """Dense synthetic network (unsaturated, non-sparse occupations)."""
    return _generate(DENSE_NODE_COUNT, DENSE_AVERAGE_DEGREE, DENSE_EVENTS_PER_EDGE)


@pytest.fixture(scope="module")
def sparse_network() -> tuple:
    """Sparse synthetic network (nearly binary occupations)."""
    return _generate(SPARSE_NODE_COUNT, SPARSE_AVERAGE_DEGREE, SPARSE_EVENTS_PER_EDGE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_fixed_et(
    node_count: int,
    e: int,
    t: int,
    *,
    seed: int = 0,
    known: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> EdgeTable:
    """Sample the microcanonical ME fixed-(E,T) model."""
    kwargs: dict = {
        "ensemble": Ensemble.MICROCANONICAL,
        "family": ModelFamily.ME,
        "constraint": Constraint.EDGES_EVENTS,
        "node_count": node_count,
        "target_edges": e,
        "total_events": t,
        "self_loops": False,
        "seed": seed,
    }
    if known is not None:
        kwargs["known_source"], kwargs["known_target"], kwargs["known_occnum"] = known
    return sample_model(**kwargs)


def _verify_constraints(sampled, e: int, t: int) -> None:
    """Exact hard-constraint checks (no tolerance needed)."""
    assert len(sampled) == e, f"edge count {len(sampled)} != E={e}"
    assert int(sampled.occ_num.sum()) == t, (
        f"total occupation {sampled.occ_num.sum()} != T={t}"
    )
    assert (sampled.occ_num > 0).all(), "zero occupation found in output"


def _verify_valid_pairs(sampled, node_count: int, known=None) -> None:
    """Every pair must be valid, in-range, self-loop-free, and unique."""
    src = sampled.source.astype(np.int64)
    tgt = sampled.target.astype(np.int64)
    # In range
    assert src.min() >= 0 and src.max() < node_count
    assert tgt.min() >= 0 and tgt.max() < node_count
    # No self-loops (self_loops=False in these tests)
    assert (src != tgt).all(), "self-loop found"
    # No duplicates
    keys = src.astype(np.int64) * node_count + tgt
    assert len(np.unique(keys)) == len(keys), "duplicate pair found"


def _verify_fixed_pairs(sampled, known) -> None:
    """Fixed pairs must be present with their exact occupation values."""
    k_src, k_tgt, k_occ = known
    for s, t, o in zip(k_src, k_tgt, k_occ, strict=True):
        if o == 0:
            continue  # fixed-zero must not appear as an edge
        mask = (sampled.source == s) & (sampled.target == t)
        assert mask.any(), f"fixed pair ({s},{t}) missing from output"
        assert sampled.occ_num[mask][0] == o, (
            f"fixed pair ({s},{t}) occupation {sampled.occ_num[mask][0]} != {o}"
        )


def _pick_fixed_pairs(edges, n_fixed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick a deterministic subset of real edges as fixed pairs."""
    rng = np.random.default_rng(123)
    idx = rng.choice(len(edges), size=n_fixed, replace=False)
    idx = np.sort(idx)
    return (
        edges.source[idx].astype(np.uint64),
        edges.target[idx].astype(np.uint64),
        edges.occ_num[idx].astype(np.uint64),
    )


# ---------------------------------------------------------------------------
# No fixed pairs — dense and sparse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ["dense", "sparse"])
def test_fixed_et_recovers_exact_constraints(regime, request) -> None:
    """E2E: sample with derived E,T and verify exact recovery."""
    edges, e, t, node_count = request.getfixturevalue(f"{regime}_network")
    sampled = _sample_fixed_et(node_count, e, t, seed=11)
    _verify_constraints(sampled, e, t)
    _verify_valid_pairs(sampled, node_count)
    # The sampled network is a *different* configuration than the synthetic
    # one (microcanonical is not conditioned on the source network), but E
    # and T are exact.
    assert len(sampled) == len(edges)


@pytest.mark.parametrize("regime", ["dense", "sparse"])
def test_fixed_et_reproducible(regime, request) -> None:
    """Same seed -> identical output."""
    _edges, e, t, node_count = request.getfixturevalue(f"{regime}_network")
    a = _sample_fixed_et(node_count, e, t, seed=42)
    b = _sample_fixed_et(node_count, e, t, seed=42)
    assert (a.source == b.source).all()
    assert (a.target == b.target).all()
    assert (a.occ_num == b.occ_num).all()


@pytest.mark.parametrize("regime", ["dense", "sparse"])
def test_fixed_et_varies_across_seeds(regime, request) -> None:
    """Different seeds -> different (but equally valid) outputs."""
    _edges, e, t, node_count = request.getfixturevalue(f"{regime}_network")
    a = _sample_fixed_et(node_count, e, t, seed=1)
    b = _sample_fixed_et(node_count, e, t, seed=2)
    # Both valid
    _verify_constraints(a, e, t)
    _verify_constraints(b, e, t)
    # Very likely different configurations (E is large enough that the
    # chance of an identical draw is negligible).
    assert not ((a.source == b.source).all() and (a.occ_num == b.occ_num).all())


# ---------------------------------------------------------------------------
# Fixed pairs — dense and sparse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ["dense", "sparse"])
def test_fixed_et_with_fixed_pairs_recovers_constraints(regime, request) -> None:
    """E2E with fixed pairs: residuals sampled, fixed pairs preserved."""
    edges, e, t, node_count = request.getfixturevalue(f"{regime}_network")
    known = _pick_fixed_pairs(edges, N_FIXED)
    sampled = _sample_fixed_et(node_count, e, t, seed=17, known=known)
    _verify_constraints(sampled, e, t)
    _verify_valid_pairs(sampled, node_count, known=known)
    _verify_fixed_pairs(sampled, known)


@pytest.mark.parametrize("regime", ["dense", "sparse"])
def test_fixed_et_with_fixed_zero_pairs(regime, request) -> None:
    """A fixed-zero pair is an ordinary frozen pair with occupation 0.

    It contributes nothing to E or T but must not be sampled.
    """
    edges, e, t, node_count = request.getfixturevalue(f"{regime}_network")
    # Fix 3 real edges and 2 zero pairs.
    pos = _pick_fixed_pairs(edges, 3)
    zero_src = np.array([node_count - 1, 0], dtype=np.uint64)
    zero_tgt = np.array([0, node_count - 1], dtype=np.uint64)
    # Ensure the zero pairs are not among the positive fixed pairs and are
    # admissible (no self-loop).
    zero_src, zero_tgt = zero_src[:1], zero_tgt[:1]
    if (int(zero_src[0]), int(zero_tgt[0])) in set(
        zip(pos[0].tolist(), pos[1].tolist(), strict=True)
    ):
        zero_src = np.array([1], dtype=np.uint64)
        zero_tgt = np.array([0], dtype=np.uint64)
    k_src = np.concatenate([pos[0], zero_src])
    k_tgt = np.concatenate([pos[1], zero_tgt])
    k_occ = np.concatenate([pos[2], np.zeros(1, dtype=np.uint64)])
    known = (k_src, k_tgt, k_occ)

    sampled = _sample_fixed_et(node_count, e, t, seed=23, known=known)
    _verify_constraints(sampled, e, t)
    _verify_valid_pairs(sampled, node_count)
    _verify_fixed_pairs(sampled, known)
    # The fixed-zero pair must not appear as an occupied edge
    mask = (sampled.source == zero_src[0]) & (sampled.target == zero_tgt[0])
    assert not mask.any(), "fixed-zero pair appeared as an edge"


@pytest.mark.parametrize("regime", ["dense", "sparse"])
def test_fixed_et_reproducible_with_fixed_pairs(regime, request) -> None:
    """Same seed + same fixed pairs -> identical output."""
    edges, e, t, node_count = request.getfixturevalue(f"{regime}_network")
    known = _pick_fixed_pairs(edges, N_FIXED)
    a = _sample_fixed_et(node_count, e, t, seed=99, known=known)
    b = _sample_fixed_et(node_count, e, t, seed=99, known=known)
    assert (a.source == b.source).all()
    assert (a.target == b.target).all()
    assert (a.occ_num == b.occ_num).all()


# ---------------------------------------------------------------------------
# Infeasible problems
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ["dense", "sparse"])
def test_fixed_et_rejects_over_satisfying_fixed_pairs(regime, request) -> None:
    """Fixed pairs whose contribution exceeds the request must be rejected."""
    _edges, e, t, node_count = request.getfixturevalue(f"{regime}_network")
    # One fixed pair with occ > T makes the residual infeasible.
    k_src = np.array([0], dtype=np.uint64)
    k_tgt = np.array([1], dtype=np.uint64)
    k_occ = np.array([t + 1], dtype=np.uint64)
    with pytest.raises(ValueError, match="exceeds"):
        _sample_fixed_et(node_count, e, t, seed=1, known=(k_src, k_tgt, k_occ))


def test_fixed_et_rejects_t_below_e() -> None:
    """T < E is infeasible (each occupied pair needs >= 1 event)."""
    with pytest.raises(ValueError):
        _sample_fixed_et(10, 5, 4, seed=1)


def test_fixed_et_rejects_e_above_admissible_pairs() -> None:
    """E > L is infeasible."""
    with pytest.raises(ValueError):
        _sample_fixed_et(4, 13, 13, seed=1)  # L = 4*3 = 12


def test_fixed_et_rejects_fixed_pairs_mismatched_length() -> None:
    """known_source/target/occnum must have the same length."""
    with pytest.raises(ValueError, match="same length"):
        _sample_fixed_et(
            10,
            5,
            10,
            seed=1,
            known=(np.array([0]), np.array([1]), np.array([1, 2])),
        )


# ---------------------------------------------------------------------------
# Self-loops allowed
# ---------------------------------------------------------------------------


def test_fixed_et_with_self_loops_dense() -> None:
    """Self-loops allowed: all N² pairs admissible."""
    node_count = 12
    # Build a network with self-loops for feasible constraints
    net = generate_pa_geographic_network(
        node_count,
        average_degree=node_count / 5,
        events_per_edge=8.0,
        seed=7,
        self_loops=True,
    )
    edges = net.edges
    e = len(edges)
    t = int(edges.occ_num.sum())

    sampled = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.EDGES_EVENTS,
        node_count=node_count,
        target_edges=e,
        total_events=t,
        self_loops=True,
        seed=5,
    )
    assert len(sampled) == e
    assert int(sampled.occ_num.sum()) == t
    # No duplicates
    keys = sampled.source.astype(np.int64) * node_count + sampled.target
    assert len(np.unique(keys)) == len(keys)
    # Self-loops are allowed, so don't forbid them, but all pairs in range
    assert sampled.source.max() < node_count  # type: ignore
    assert sampled.target.max() < node_count  # type: ignore


# ---------------------------------------------------------------------------
# Distribution-level sanity: exact enumeration on a tiny system
# ---------------------------------------------------------------------------


def test_fixed_et_matches_exact_enumeration_small() -> None:
    """On a tiny system, empirical frequencies match exact ME probabilities.

    L = 3 pairs, E = 2, T = 4.  Target law is proportional to 1/prod(t_i!).
    """
    node_count = 2
    # self_loops = True (all 4 pairs admissible, L = 4)
    e = 2
    t = 4

    # Enumerate all C(4,2) * compositions(4,2) = 6 * 3 = 18 configurations.
    exact = {}
    pairs = [(i, j) for i in range(2) for j in range(2)]
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            for n1, n2 in [(1, 3), (2, 2), (3, 1)]:
                weight = 1.0 / (math.factorial(n1) * math.factorial(n2))
                key = tuple(sorted([(*pairs[i], n1), (*pairs[j], n2)]))
                exact[key] = exact.get(key, 0.0) + weight
    total_weight = sum(exact.values())

    trials = 20_000
    counts = {}
    for seed in range(trials):
        sampled = sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.ME,
            constraint=Constraint.EDGES_EVENTS,
            node_count=node_count,
            target_edges=e,
            total_events=t,
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
        target_p = weight / total_weight
        expected = target_p * trials
        observed = counts.get(config, 0)
        sigma = max((expected * (1.0 - target_p)) ** 0.5, 1.0)
        assert abs(observed - expected) < 5.0 * sigma, (
            f"config {config}: expected {expected:.1f}, observed {observed}"
        )
