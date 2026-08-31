"""Tests for microcanonical fixed-(s,k) sampling (ME/B/W).

E2E pipeline per the testing policy: generate a realistic weighted
network (PA-geographic, dense scenario), derive constraints (strength
and binary degree sequences), sample with the fixed-(s,k) route —
extras-first exact constructor + capped first-return degree trace — and
verify exact recovery with ``array_equal`` (microcanonical constraints
are exact, so no stochastic tolerance is needed).

Includes the §80 routing release blocker: a target whose per-node
strength differs from its per-node degree must reproduce BOTH exactly,
proving the call never silently degrades to fixed-(k,T) (which would
ignore the strengths).
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

# These E2E tests exercise the microcanonical STRENGTH_DEGREE route,
# exposed in the capability registry only after the N=1000 Rust gates
# (Gate C/D + Part G E2E) pass.  Until then they skip so the fast suite
# stays green; the capability entry flips them on automatically.
_SKIP_REASON = "microcanonical STRENGTH_DEGREE not yet exposed"
pytestmark = pytest.mark.skipif(
    capability(
        Verb.SAMPLE,
        Ensemble.MICROCANONICAL,
        ModelFamily.ME,
        Constraint.STRENGTH_DEGREE,
    )
    is None,
    reason=_SKIP_REASON,
)


def _degrees(net, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute out/in binary degrees from a sampled EdgeTable."""
    out = np.zeros(n, dtype=np.uint32)
    inp = np.zeros(n, dtype=np.uint32)
    np.add.at(out, net.source.astype(np.int64), 1)
    np.add.at(inp, net.target.astype(np.int64), 1)
    return out, inp


def _strengths(net, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute out/in strengths (total occupation) from a sampled EdgeTable."""
    out = np.zeros(n, dtype=np.uint64)
    inp = np.zeros(n, dtype=np.uint64)
    np.add.at(out, net.source.astype(np.int64), net.occ_num)
    np.add.at(inp, net.target.astype(np.int64), net.occ_num)
    return out, inp


def _dense_case() -> tuple:
    """Dense PA-geographic network with derived (s, k) constraints."""
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
    k_out = np.round(constraints.degree_out).astype(np.uint32)
    k_in = np.round(constraints.degree_in).astype(np.uint32)
    return net, n, s_out, s_in, k_out, k_in


def _b_feasible_case() -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """B-feasible variant: low occupations within the family capacity."""
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
    k_out = np.round(constraints.degree_out).astype(np.uint32)
    k_in = np.round(constraints.degree_in).astype(np.uint32)
    return n, s_out, s_in, k_out, k_in


def _sample_sk(
    family: ModelFamily,
    s_out: np.ndarray,
    s_in: np.ndarray,
    k_out: np.ndarray,
    k_in: np.ndarray,
    *,
    self_loops: bool = False,
    seed: int = 42,
    layers: int = 1,
    fixed: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
):
    kwargs = {}
    if fixed is not None:
        kwargs["known_source"], kwargs["known_target"], kwargs["known_occnum"] = fixed
    return sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=family,
        constraint=Constraint.STRENGTH_DEGREE,
        strength_out=s_out,
        strength_in=s_in,
        degree_out=k_out,
        degree_in=k_in,
        self_loops=self_loops,
        seed=seed,
        layers=layers,
        burn_in_sweeps=3,
        sweeps_per_sample=2,
        **kwargs,
    )


@pytest.mark.parametrize("family", [ModelFamily.ME, ModelFamily.B, ModelFamily.W])
def test_fixed_sk_exact_recovery(family) -> None:
    """Strengths and degrees are recovered exactly (all families)."""
    if family is ModelFamily.B:
        n, s_out, s_in, k_out, k_in = _b_feasible_case()
        layers = 4
    else:
        _, n, s_out, s_in, k_out, k_in = _dense_case()
        layers = 1
    sample = _sample_sk(family, s_out, s_in, k_out, k_in, seed=42, layers=layers)
    out, inp = _strengths(sample, n)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)
    dout, din = _degrees(sample, n)
    np.testing.assert_array_equal(dout, k_out)
    np.testing.assert_array_equal(din, k_in)
    assert len(sample.source) == int(k_out.sum())
    if family is ModelFamily.B:
        assert int(np.max(sample.occ_num)) <= 4
    assert bool((sample.occ_num >= 1).all())


def test_fixed_sk_routing_release_blocker() -> None:
    """§80: s+k must route to occupation MCMC, never silently to fixed-(k,T).

    The fixture has heterogeneous strengths with per-node strength
    clearly different from per-node degree (residual extras > 0).  Exact
    recovery of both sequences is only possible via the fixed-(s,k)
    backend; fixed-(k,T) would reproduce the degrees but not the
    strengths.
    """
    # N=4 directed cycle of degrees (1,1,1,1) with heterogeneous extras:
    # s = k + r, r = (2,1,1,1), c = (1,2,1,1).
    s_out = np.array([3, 2, 2, 2], dtype=np.uint64)
    s_in = np.array([2, 3, 2, 2], dtype=np.uint64)
    k_out = np.array([1, 1, 1, 1], dtype=np.uint32)
    k_in = np.array([1, 1, 1, 1], dtype=np.uint32)
    sample = _sample_sk(ModelFamily.ME, s_out, s_in, k_out, k_in, self_loops=False)
    out, inp = _strengths(sample, 4)
    np.testing.assert_array_equal(out, s_out, err_msg="strengths must be exact (s+k)")
    np.testing.assert_array_equal(inp, s_in)
    dout, din = _degrees(sample, 4)
    np.testing.assert_array_equal(dout, k_out, err_msg="degrees must be exact (s+k)")
    np.testing.assert_array_equal(din, k_in)
    # With strength totals above degree totals, any fixed-(k,T) fallback
    # would have returned the wrong total occupation: sum(s) = 9.
    assert int(sample.occ_num.sum()) == int(s_out.sum())


def test_fixed_sk_deterministic() -> None:
    """Same seed → identical sample."""
    _, _, s_out, s_in, k_out, k_in = _dense_case()
    a = _sample_sk(ModelFamily.ME, s_out, s_in, k_out, k_in, seed=123)
    b = _sample_sk(ModelFamily.ME, s_out, s_in, k_out, k_in, seed=123)
    np.testing.assert_array_equal(a.source, b.source)
    np.testing.assert_array_equal(a.target, b.target)
    np.testing.assert_array_equal(a.occ_num, b.occ_num)


def test_fixed_sk_self_loops() -> None:
    """Self-loop policy is respected on both sides."""
    # With self-loops: N=3, k=(1,1,1)/(1,1,1), s=[3,2,2]/[2,3,2].
    s_out = np.array([3, 2, 2], dtype=np.uint64)
    s_in = np.array([2, 3, 2], dtype=np.uint64)
    k_out = np.array([1, 1, 1], dtype=np.uint32)
    k_in = np.array([1, 1, 1], dtype=np.uint32)
    sample = _sample_sk(ModelFamily.ME, s_out, s_in, k_out, k_in, self_loops=True)
    out, inp = _strengths(sample, 3)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)
    assert len(sample.source) == 3

    # Loopless variant (degree sum 3, no diagonal allowed).
    sample2 = _sample_sk(ModelFamily.ME, s_out, s_in, k_out, k_in, self_loops=False)
    assert not bool((sample2.source == sample2.target).any())
    dout, din = _degrees(sample2, 3)
    np.testing.assert_array_equal(dout, k_out)
    np.testing.assert_array_equal(din, k_in)


def test_fixed_sk_fixed_pairs() -> None:
    """Fixed positive pairs are subtracted from (s,k) and merged back."""
    _, n, s_out, s_in, k_out, k_in = _dense_case()
    fixed = (np.array([0]), np.array([1]), np.array([2], dtype=np.uint64))
    sample = _sample_sk(ModelFamily.ME, s_out, s_in, k_out, k_in, seed=42, fixed=fixed)
    assert bool(
        ((sample.source == 0) & (sample.target == 1) & (sample.occ_num == 2)).any()
    )
    out, inp = _strengths(sample, n)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)
    dout, din = _degrees(sample, n)
    np.testing.assert_array_equal(dout, k_out)
    np.testing.assert_array_equal(din, k_in)


def test_fixed_sk_b_m1_early_rejection() -> None:
    """B M=1 (Bernoulli) forces strength == degree: rejected early."""
    s_out = np.array([3, 2], dtype=np.uint64)
    s_in = np.array([2, 3], dtype=np.uint64)
    k_out = np.array([1, 1], dtype=np.uint32)
    k_in = np.array([1, 1], dtype=np.uint32)
    with pytest.raises(ValueError, match="M=1"):
        sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.B,
            constraint=Constraint.STRENGTH_DEGREE,
            strength_out=s_out,
            strength_in=s_in,
            degree_out=k_out,
            degree_in=k_in,
            layers=1,
            self_loops=True,
            seed=1,
        )


def test_fixed_sk_requires_arguments() -> None:
    """Missing degree constraints raise ValueError."""
    s_out = np.array([2, 2], dtype=np.uint64)
    s_in = np.array([2, 2], dtype=np.uint64)
    k_out = np.array([1, 1], dtype=np.uint32)
    with pytest.raises(ValueError, match="degree_out and degree_in"):
        sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH_DEGREE,
            strength_out=s_out,
            strength_in=s_in,
            degree_out=k_out,
            self_loops=True,
            seed=1,
        )


@pytest.mark.heavy
def test_fixed_sk_dense_n100() -> None:
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
    k_out = np.round(constraints.degree_out).astype(np.uint32)
    k_in = np.round(constraints.degree_in).astype(np.uint32)
    sample = _sample_sk(ModelFamily.ME, s_out, s_in, k_out, k_in, seed=42)
    out, inp = _strengths(sample, n)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)
    dout, din = _degrees(sample, n)
    np.testing.assert_array_equal(dout, k_out)
    np.testing.assert_array_equal(din, k_in)
