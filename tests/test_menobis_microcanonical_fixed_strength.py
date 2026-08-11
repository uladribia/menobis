"""Tests for microcanonical fixed-strength sampling (ME/B/W)."""

from __future__ import annotations

import numpy as np
import pytest

from menobis.models.generation import _sample_strength_fixed_strength_mcmc
from menobis.models.spec import Constraint, Ensemble, ModelFamily
from menobis.routing import sample_model


def _strengths(net) -> tuple[np.ndarray, np.ndarray]:
    """Compute out/in strengths from a sampled EdgeTable."""
    n = int(max(net.source.max(), net.target.max())) + 1
    out = np.zeros(n, dtype=np.uint64)
    inp = np.zeros(n, dtype=np.uint64)
    np.add.at(out, net.source.astype(np.int64), net.occ_num)
    np.add.at(inp, net.target.astype(np.int64), net.occ_num)
    return out, inp


def test_me_mcmc_no_self_loops_preserves_strengths() -> None:
    """ME without self-loops routes through MCMC and preserves strengths."""
    s_out = np.array([3, 5, 7, 2], dtype=np.uint64)
    s_in = np.array([4, 6, 3, 4], dtype=np.uint64)
    net = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
        self_loops=False,
        seed=42,
    )
    # No self-loops.
    assert not np.any(net.source == net.target)
    # Exact strengths.
    out, inp = _strengths(net)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)


def test_b_fixed_strength_via_mcmc() -> None:
    """B fixed-strength routes through MCMC and respects layer capacity."""
    s_out = np.array([4, 4, 4], dtype=np.uint64)
    s_in = np.array([4, 4, 4], dtype=np.uint64)
    net = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.B,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
        layers=4,
        self_loops=True,
        seed=42,
        burn_in_sweeps=30,
        sweeps_per_sample=10,
    )
    # B occupations must not exceed layers.
    assert np.all(net.occ_num <= 4)
    out, inp = _strengths(net)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)


def test_w_fixed_strength_via_mcmc() -> None:
    """W fixed-strength routes through MCMC and preserves strengths."""
    s_out = np.array([5, 5, 5], dtype=np.uint64)
    s_in = np.array([5, 5, 5], dtype=np.uint64)
    net = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.W,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
        layers=2,
        self_loops=True,
        seed=42,
    )
    out, inp = _strengths(net)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)


def test_me_mcmc_preserves_strengths_with_self_loops() -> None:
    """ME with self-loops uses the MCMC chain and preserves strengths."""
    s_out = np.array([10, 20, 30], dtype=np.uint64)
    s_in = np.array([15, 25, 20], dtype=np.uint64)
    net = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
        self_loops=True,
        seed=42,
    )
    out, inp = _strengths(net)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)


def test_fixed_pairs_are_respected() -> None:
    """Fixed pairs are subtracted and merged correctly."""
    s_out = np.array([5, 5], dtype=np.uint64)
    s_in = np.array([5, 5], dtype=np.uint64)
    net = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
        self_loops=True,
        seed=42,
        known_source=np.array([0], dtype=np.uint64),
        known_target=np.array([1], dtype=np.uint64),
        known_occnum=np.array([2], dtype=np.uint64),
    )
    out, inp = _strengths(net)
    np.testing.assert_array_equal(out, s_out)
    np.testing.assert_array_equal(inp, s_in)


def test_strength_mcmc_deterministic() -> None:
    """Same seed gives reproducible results."""
    s_out = np.array([3, 3, 3], dtype=np.uint64)
    s_in = np.array([3, 3, 3], dtype=np.uint64)
    a = _sample_strength_fixed_strength_mcmc(
        family="B",
        strength_out=s_out,
        strength_in=s_in,
        self_loops=True,
        layers=3,
        seed=42,
    )
    b = _sample_strength_fixed_strength_mcmc(
        family="B",
        strength_out=s_out,
        strength_in=s_in,
        self_loops=True,
        layers=3,
        seed=42,
    )
    np.testing.assert_array_equal(a.source, b.source)
    np.testing.assert_array_equal(a.target, b.target)
    np.testing.assert_array_equal(a.occ_num, b.occ_num)


def test_infeasible_strength_rejected() -> None:
    """Infeasible strength sequences are rejected with a clear error."""
    # s_out total (11) != s_in total (9)
    with pytest.raises((ValueError, RuntimeError)):
        sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH,
            strength_out=np.array([5, 6], dtype=np.uint64),
            strength_in=np.array([5, 4], dtype=np.uint64),
            self_loops=True,
        )


@pytest.mark.heavy
@pytest.mark.parametrize(
    "family,kwargs",
    [
        (ModelFamily.ME, {}),
        (ModelFamily.B, {"layers": 8}),
        (ModelFamily.W, {"layers": 4}),
    ],
)
def test_microcanonical_strength_cost_e2e(family: ModelFamily, kwargs: dict) -> None:
    """STRENGTH_COST E2E: strength+cost sampler recovers both constraints.

    Uses PA-geographic network (N=10, dense) to derive feasible strength
    sequences and an empirical target cost, then verifies the microcanonical
    STRENGTH_COST sampler recovers both constraints.
    """
    from menobis.analysis import directed_strengths
    from menobis.utilities.synthetic import generate_pa_geographic_network

    # 1. Generate a small PA-geographic network (N=10, dense)
    net = generate_pa_geographic_network(
        10, average_degree=4.0, events_per_edge=5.0, seed=42, self_loops=True
    )

    # 2. Derive strength_out/in, coordinates, and target cost
    s = directed_strengths(net.edges)
    cx, cy = net.x, net.y
    source = net.edges.source
    target = net.edges.target
    occ = net.edges.occ_num
    dx = cx[source] - cx[target]
    dy = cy[source] - cy[target]
    target_cost = float(np.sqrt(dx**2 + dy**2) @ occ)

    # 3. Sample via microcanonical STRENGTH_COST
    result = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=family,
        constraint=Constraint.STRENGTH_COST,
        strength_out=s.out,
        strength_in=s.incoming,
        coord_x=cx,
        coord_y=cy,
        target_cost=target_cost,
        self_loops=True,
        seed=42,
        **kwargs,
    )

    # 4. Verify strengths are exactly preserved
    n = len(cx)
    assert result.total_events == int(s.out.sum())
    actual_out = np.bincount(result.source, weights=result.occ_num, minlength=n)
    actual_inp = np.bincount(result.target, weights=result.occ_num, minlength=n)
    np.testing.assert_array_equal(actual_out, s.out)
    np.testing.assert_array_equal(actual_inp, s.incoming)

    # 5. Verify observed cost is within tolerance of target cost
    dx_sampled = cx[result.source] - cx[result.target]
    dy_sampled = cy[result.source] - cy[result.target]
    observed_cost = float(np.sqrt(dx_sampled**2 + dy_sampled**2) @ result.occ_num)
    rel_err = abs(observed_cost - target_cost) / target_cost
    # Allow 20 % relative tolerance for MCMC noise
    assert rel_err < 0.20, (
        f"cost rel_err={rel_err:.4f} exceeds 0.20 "
        f"(target={target_cost:.2f}, observed={observed_cost:.2f})"
    )
