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


def test_me_direct_still_works() -> None:
    """ME with self-loops still uses the exact stub-matching fast path."""
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
