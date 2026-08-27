"""Tests for microcanonical fixed-(s,E) sampling (ME/B/W).

E2E pipeline per the testing policy: generate a realistic weighted
network (PA-geographic, dense scenario), derive constraints (strengths
and binary edge count), sample with exact constraints, and verify exact
recovery.  Microcanonical constraints are exact, so recovery checks use
``array_equal`` rather than stochastic tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

from menobis.capabilities import capability
from menobis.models.spec import (
    Constraint,
    Ensemble,
    ModelFamily,
    Verb,
)
from menobis.routing import sample_model
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

# These E2E tests exercise the microcanonical STRENGTH_EDGES route, which
# is only exposed in the capability registry after the exact small-state
# gates pass.  Until then they skip so the fast suite stays green;
# removing the capability entry flips them on automatically.
_SKIP_REASON = "microcanonical STRENGTH_EDGES not yet exposed"
pytestmark = pytest.mark.skipif(
    capability(
        Verb.SAMPLE,
        Ensemble.MICROCANONICAL,
        ModelFamily.ME,
        Constraint.STRENGTH_EDGES,
    )
    is None,
    reason=_SKIP_REASON,
)


def _edge_count(net) -> int:
    return int(len(net.source))


def _strengths(net, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute out/in strengths from a sampled EdgeTable."""
    out = np.zeros(n, dtype=np.uint64)
    inp = np.zeros(n, dtype=np.uint64)
    np.add.at(out, net.source.astype(np.int64), net.occ_num)
    np.add.at(inp, net.target.astype(np.int64), net.occ_num)
    return out, inp


def _dense_case() -> tuple:
    """Dense PA-geographic network with derived (s, E) constraints."""
    net = generate_pa_geographic_network(
        15, seed=11, self_loops=False, average_degree=4.0, events_per_edge=5.0
    )
    constraints = derive_synthetic_constraints(net)
    n = len(net.x)
    s_out = np.round(constraints.strength_out).astype(np.uint64)
    s_in = np.round(constraints.strength_in).astype(np.uint64)
    # Rebalance rounding drift (guaranteed-feasible totals by design).
    diff = int(s_out.sum()) - int(s_in.sum())
    if diff > 0:
        s_in[np.argmax(s_in)] += diff
    elif diff < 0:
        s_out[np.argmax(s_out)] -= diff
    edges = int(np.round(constraints.total_edges))
    return net, n, s_out, s_in, edges


def _b_feasible_case() -> tuple[int, np.ndarray, np.ndarray, int]:
    """B-feasible variant: lower occupations so every margin fits within
    M × (available cells) under the family capacity.

    The dense fixture's rounded margins concentrate up to 81 events on a
    single node, which exceeds the B M=4 capacity budget of 4 × 14 cells
    (plain microcanonical STRENGTH fails there too); events_per_edge=2
    keeps occupations within capacity.
    """
    net = generate_pa_geographic_network(
        15, seed=11, self_loops=False, average_degree=4.0, events_per_edge=2.0
    )
    constraints = derive_synthetic_constraints(net)
    n = len(net.x)
    s_out = np.round(constraints.strength_out).astype(np.uint64)
    s_in = np.round(constraints.strength_in).astype(np.uint64)
    diff = int(s_out.sum()) - int(s_in.sum())
    if diff > 0:
        s_in[np.argmax(s_in)] += diff
    elif diff < 0:
        s_out[np.argmax(s_out)] -= diff
    edges = int(np.round(constraints.total_edges))
    return n, s_out, s_in, edges


@pytest.mark.parametrize("family", [ModelFamily.ME, ModelFamily.B, ModelFamily.W])
def test_fixed_se_exact_recovery(family) -> None:
    """Strengths and edge count are recovered exactly (all families)."""
    if family is ModelFamily.B:
        n, s_out, s_in, edges = _b_feasible_case()
        kwargs = {"layers": 4}
    else:
        _, n, s_out, s_in, edges = _dense_case()
        kwargs = {}
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=family,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=edges,
        self_loops=False,
        seed=42,
        **kwargs,
    )
    assert _edge_count(sample) == edges
    out, inp = _strengths(sample, n)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)
    if family is ModelFamily.B:
        assert int(sample.occ_num.max()) <= 4
    assert bool((sample.occ_num >= 1).all())


def test_fixed_se_families_differ() -> None:
    """ME, B, and W are distinct models: same constraints, different samples."""
    # B-feasible margins so all three families accept the same input.
    _, s_out, s_in, edges = _b_feasible_case()
    samples = {}
    for family in (ModelFamily.ME, ModelFamily.B, ModelFamily.W):
        kwargs = {"layers": 4} if family is not ModelFamily.ME else {}
        samples[family] = sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=family,
            constraint=Constraint.STRENGTH_EDGES,
            strength_out=s_out,
            strength_in=s_in,
            target_edges=edges,
            self_loops=False,
            seed=7,
            **kwargs,
        )
    me, b, w = (
        samples[ModelFamily.ME],
        samples[ModelFamily.B],
        samples[ModelFamily.W],
    )
    keys = [
        sorted(zip(net.source, net.target, net.occ_num, strict=True))
        for net in (me, b, w)
    ]
    assert not (keys[0] == keys[1] and keys[1] == keys[2])


def test_fixed_se_deterministic() -> None:
    """Same seed → identical sample."""
    _, _, s_out, s_in, edges = _dense_case()
    kwargs = {}
    a = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=edges,
        self_loops=False,
        seed=123,
        **kwargs,
    )
    b = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=edges,
        self_loops=False,
        seed=123,
        **kwargs,
    )
    np.testing.assert_array_equal(a.source, b.source)
    np.testing.assert_array_equal(a.target, b.target)
    np.testing.assert_array_equal(a.occ_num, b.occ_num)


def test_fixed_se_self_loops() -> None:
    """Self-loop policy is respected on both sides."""
    # With self-loops: N=3, E=5.
    s_out = np.array([3, 3, 3], dtype=np.uint64)
    s_in = np.array([3, 3, 3], dtype=np.uint64)
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=5,
        self_loops=True,
        seed=42,
    )
    out, inp = _strengths(sample, 3)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)
    assert len(sample.source) == 5

    # Loopless N=4 (loopless N=3 with equal margins forces a perfect
    # matching whose rectangle move set is disconnected — documented
    # corner case surfaced as a structured error).
    s_out = np.array([3, 2, 2, 1], dtype=np.uint64)
    s_in = np.array([1, 2, 2, 3], dtype=np.uint64)
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=5,
        self_loops=False,
        seed=42,
    )
    assert not bool((sample.source == sample.target).any())
    out, inp = _strengths(sample, 4)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)
    assert len(sample.source) == 5


def test_fixed_se_fixed_pairs() -> None:
    """Fixed positive pairs are subtracted from (s, E) and merged back."""
    _, n, s_out, s_in, edges = _dense_case()
    # Fix one positive pair: (0, 1) with occupation 2.
    known_source = np.array([0])
    known_target = np.array([1])
    known_occnum = np.array([2], dtype=np.uint64)
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=edges,
        known_source=known_source,
        known_target=known_target,
        known_occnum=known_occnum,
        self_loops=False,
        seed=42,
    )
    fixed_present = bool(
        ((sample.source == 0) & (sample.target == 1) & (sample.occ_num == 2)).any()
    )
    assert fixed_present
    assert _edge_count(sample) == edges
    out, inp = _strengths(sample, n)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)


def test_fixed_se_infeasible_targets_rejected() -> None:
    """Necessary-bound violations raise informative errors."""
    s_out = np.array([4, 4, 4, 4], dtype=np.uint64)
    s_in = np.array([4, 4, 4, 4], dtype=np.uint64)
    # E above total occupation.
    with pytest.raises(ValueError, match="exceeds total occupation"):
        sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH_EDGES,
            strength_out=s_out,
            strength_in=s_in,
            target_edges=20,
            self_loops=True,
            seed=1,
        )
    # E below the per-node cell demand (B M=2, strengths 4 → ⌈4/2⌉·4 = 8).
    with pytest.raises(ValueError, match="required minimum"):
        sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.B,
            constraint=Constraint.STRENGTH_EDGES,
            strength_out=s_out,
            strength_in=s_in,
            target_edges=7,
            layers=2,
            self_loops=True,
            seed=1,
        )


def test_fixed_se_requires_arguments() -> None:
    """Missing constraint arguments raise ValueError."""
    s_out = np.array([2, 2], dtype=np.uint64)
    s_in = np.array([2, 2], dtype=np.uint64)
    with pytest.raises(ValueError, match="target_edges"):
        sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH_EDGES,
            strength_out=s_out,
            strength_in=s_in,
            self_loops=True,
            seed=1,
        )


@pytest.mark.heavy
def test_fixed_se_dense_n100() -> None:
    """Larger dense scenario: exact recovery within a bounded budget."""
    net = generate_pa_geographic_network(
        100, seed=5, self_loops=False, average_degree=6.0, events_per_edge=4.0
    )
    constraints = derive_synthetic_constraints(net)
    n = len(net.x)
    s_out = np.round(constraints.strength_out).astype(np.uint64)
    s_in = np.round(constraints.strength_in).astype(np.uint64)
    diff = int(s_out.sum()) - int(s_in.sum())
    if diff > 0:
        s_in[np.argmax(s_in)] += diff
    elif diff < 0:
        s_out[np.argmax(s_out)] -= diff
    edges = int(np.round(constraints.total_edges))
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=edges,
        self_loops=False,
        seed=42,
    )
    assert _edge_count(sample) == edges
    out, inp = _strengths(sample, n)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)
