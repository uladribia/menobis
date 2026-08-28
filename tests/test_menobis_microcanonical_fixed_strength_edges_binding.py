"""Binding-level E2E for microcanonical fixed-(s,E) sampling.

The public ``sample_model`` route for ``STRENGTH_EDGES`` stays
capability-gated until the final phase (mathematical + scalability gates
must pass first).  These tests exercise the Rust binding directly so the
Python↔Rust path is verified end-to-end while the feature is still
unadvertised.

Pipeline per the testing policy: generate a realistic weighted network
(PA-geographic, dense scenario), derive constraints (strengths and
occupied-pair count), sample, and verify exact recovery (microcanonical
constraints are exact, so recovery checks are ``array_equal``).
"""

from __future__ import annotations

import numpy as np
import pytest

import menobis._menobis as _menobis
from menobis.models.spec import Constraint, Ensemble, ModelFamily, Verb
from menobis.routing import sample_model, sample_model_detailed
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)


def _strengths(source: np.ndarray, target: np.ndarray, occ_num: np.ndarray, n: int):
    out = np.zeros(n, dtype=np.uint64)
    inp = np.zeros(n, dtype=np.uint64)
    np.add.at(out, source.astype(np.int64), occ_num)
    np.add.at(inp, target.astype(np.int64), occ_num)
    return out, inp


def _sample(
    family: str,
    s_out,
    s_in,
    edges: int,
    *,
    layers: int = 1,
    fixed: tuple | None = None,
    seed: int = 42,
    self_loops: bool = False,
    burn: int = 50,
):
    fixed = fixed or ([], [], [])
    sources, targets, occ_nums = _menobis.sample_fixed_strength_edges(
        family,
        np.asarray(s_out, dtype=np.uint64).tolist(),
        np.asarray(s_in, dtype=np.uint64).tolist(),
        int(edges),
        bool(self_loops),
        list(fixed[0]),
        list(fixed[1]),
        list(fixed[2]),
        int(layers),
        int(burn),  # burn_in_sweeps
        10,  # sweeps_per_sample
        int(seed),
    )
    return (
        np.asarray(sources, dtype=np.uint64),
        np.asarray(targets, dtype=np.uint64),
        np.asarray(occ_nums, dtype=np.uint64),
    )


def _dense_case():
    """Dense PA-geographic network with derived (s, E) constraints."""
    net = generate_pa_geographic_network(
        15, seed=11, self_loops=False, average_degree=4.0, events_per_edge=5.0
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


def _b_feasible_case():
    """B-feasible variant: lower occupations within the family capacity."""
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


@pytest.mark.parametrize("family", ["ME", "B", "W"])
def test_binding_exact_recovery_all_families(family: str) -> None:
    """Strengths and occupied-pair count are recovered exactly (ME/B/W)."""
    if family == "B":
        n, s_out, s_in, edges = _b_feasible_case()
        layers = 4
    else:
        n, s_out, s_in, edges = _dense_case()
        layers = 4 if family == "W" else 1
    source, target, occ_num = _sample(family, s_out, s_in, edges, layers=layers)
    assert len(source) == edges, f"{family}: E={len(source)} != target {edges}"
    out, inp = _strengths(source, target, occ_num, n)
    np.testing.assert_array_equal(out, s_out, err_msg=f"{family}: out-strength drift")
    np.testing.assert_array_equal(inp, s_in, err_msg=f"{family}: in-strength drift")
    if family == "B":
        assert int(occ_num.max()) <= layers
    assert bool((occ_num >= 1).all())


def test_binding_deterministic_by_seed() -> None:
    _, s_out, s_in, edges = _dense_case()
    a = _sample("ME", s_out, s_in, edges, seed=123)
    b = _sample("ME", s_out, s_in, edges, seed=123)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


def test_binding_fixed_pairs_merged_exactly() -> None:
    _, s_out, s_in, edges = _dense_case()
    fixed = (np.array([0]), np.array([1]), np.array([2], dtype=np.uint64))
    source, target, occ_num = _sample("ME", s_out, s_in, edges, fixed=fixed, seed=42)
    assert len(source) == edges
    out, inp = _strengths(source, target, occ_num, 15)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)
    # Fixed pair present with the right occupation, unique key.
    keys = set(zip(source.tolist(), target.tolist(), strict=True))
    assert (0, 1) in {(s, t) for s, t in zip(source, target, strict=True)}
    assert len(keys) == len(source), "duplicate output coordinates"


def test_binding_rejects_infeasible_targets() -> None:
    s_out = np.array([4, 4, 4, 4], dtype=np.uint64)
    s_in = np.array([4, 4, 4, 4], dtype=np.uint64)
    with pytest.raises(ValueError, match="exceeds total occupation"):
        _sample("ME", s_out, s_in, 20)


def test_public_route_now_supported_and_labeled() -> None:
    """Phase 11: the public sample_model route is advertised.

    sample_model_detailed must label the method explicitly as the
    fixed-strength-edges exact stationary MCMC, never the generic
    fixed-strength else-branch.
    """
    from menobis.capabilities import SamplingExactness, capability

    n, s_out, s_in, edges = _dense_case()
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=edges,
        self_loops=False,
        seed=11,
    )
    assert len(sample) == edges
    out, inp = _strengths(sample.source, sample.target, sample.occ_num, n)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)

    detailed = sample_model_detailed(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=edges,
        self_loops=False,
        seed=11,
    )
    assert detailed.method == "microcanonical_fixed_strength_edges"
    assert detailed.exactness == SamplingExactness.EXACT_STATIONARY_MCMC

    cap = capability(
        Verb.SAMPLE,
        Ensemble.MICROCANONICAL,
        ModelFamily.ME,
        Constraint.STRENGTH_EDGES,
    )
    assert cap is not None and cap.supported
    assert not cap.requires_fit
    assert cap.backend == "microcanonical_fixed_strength_edges"


# --------------------------------------------------------------------------
# Phase 10: N=1000 heavy scalability gates (§32) via the direct binding
# --------------------------------------------------------------------------


def _derive_from_generated(net):
    """Derive constraints from a generated network.

    Rebalances rounding drift (guaranteed feasible by construction).
    """
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


@pytest.mark.heavy
@pytest.mark.parametrize(
    "family,layers,events_per_edge",
    [("ME", 1, 5.0), ("B", 5, 2.0), ("W", 3, 3.0)],
)
def test_n1000_scalability_exact_recovery(family, layers, events_per_edge) -> None:
    """N=1000 sparse generated network: exact constraints, loopless.

    Runs through the Rust binding; the public route stays
    capability-gated until the final phase.
    """
    net = generate_pa_geographic_network(
        1000,
        seed=5,
        self_loops=False,
        average_degree=6.0,
        events_per_edge=events_per_edge,
    )
    n, s_out, s_in, edges = _derive_from_generated(net)
    source, target, occ_num = _sample(
        family, s_out, s_in, edges, layers=layers, seed=42, self_loops=False
    )
    assert len(source) == edges, f"{family}: E={len(source)} != {edges}"
    out, inp = _strengths(source, target, occ_num, n)
    np.testing.assert_array_equal(out, s_out, err_msg=f"{family}: out-strength drift")
    np.testing.assert_array_equal(inp, s_in, err_msg=f"{family}: in-strength drift")
    assert not bool((source == target).any()), f"{family}: self-loop appeared"
    assert bool((occ_num >= 1).all())


@pytest.mark.heavy
def test_n1000_fixed_pair_scalability() -> None:
    """N=1000 fixed pairs: exact recovery, unique output coordinates.

    The sparse residual domain is verified at the Rust core level by the
    CompleteMinus oracle tests.
    """
    net = generate_pa_geographic_network(
        1000, seed=5, self_loops=False, average_degree=6.0, events_per_edge=5.0
    )
    n, s_out, s_in, edges = _derive_from_generated(net)
    fixed = (
        np.array([0, 1, 2, 3], dtype=np.uint64),
        np.array([10, 11, 12, 13], dtype=np.uint64),
        np.array([1, 1, 1, 1], dtype=np.uint64),
    )
    source, target, occ_num = _sample(
        "ME", s_out, s_in, edges, fixed=fixed, seed=42, self_loops=False
    )
    assert len(source) == edges
    out, inp = _strengths(source, target, occ_num, n)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)
    assert not bool((source == target).any())
    assert len(set(zip(source.tolist(), target.tolist(), strict=True))) == len(source)


@pytest.mark.heavy
def test_n5000_smoke() -> None:
    """N=5000 sparse ME smoke (§33): exact recovery, no quadratic blowup.

    Timing is informational; the hard requirement is completion without
    dense materialization.
    """
    net = generate_pa_geographic_network(
        5000, seed=9, self_loops=False, average_degree=4.0, events_per_edge=4.0
    )
    n, s_out, s_in, edges = _derive_from_generated(net)
    source, target, occ_num = _sample(
        "ME", s_out, s_in, edges, seed=7, self_loops=False, burn=10
    )
    assert len(source) == edges
    out, inp = _strengths(source, target, occ_num, n)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)
    assert not bool((source == target).any())
