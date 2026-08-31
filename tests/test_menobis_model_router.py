"""Unified model routing tests."""

import numpy as np
import pytest

from menobis.data.frames import EdgeTable
from menobis.models import (
    Constraint,
    Ensemble,
    ModelFamily,
    StrengthFit,
    UnsupportedModelCaseError,
    fit_model,
    sample_model,
)
from menobis.routing import sample_model_detailed

S_OUT = np.array([3.0, 2.0], dtype=np.float64)
S_IN = np.array([2.0, 3.0], dtype=np.float64)


# ---- Microcanonical fixed-(E,T): all families ----


@pytest.mark.parametrize(
    "family,kwargs",
    [
        (ModelFamily.ME, {}),
        (ModelFamily.B, {"layers": 5}),
        (ModelFamily.W, {"layers": 2}),
    ],
)
def test_microcanonical_edges_events_preserves_constraints(
    family: ModelFamily, kwargs: dict
) -> None:
    """MC EdgesEvents preserves exact E and T for all families."""
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=family,
        constraint=Constraint.EDGES_EVENTS,
        node_count=6,
        target_edges=4,
        total_events=10,
        self_loops=False,
        seed=7,
        **kwargs,
    )
    assert len(sample.source) == 4
    assert sample.total_events == 10


# ---- Microcanonical fixed-(k,T): all families ----


@pytest.mark.parametrize(
    "family,kwargs",
    [
        (ModelFamily.ME, {}),
        (ModelFamily.B, {"layers": 5}),
        (ModelFamily.W, {"layers": 2}),
    ],
)
def test_microcanonical_degree_events_preserves_constraints(
    family: ModelFamily, kwargs: dict
) -> None:
    """MC DegreeEvents preserves exact degrees and total for all families.

    N=5 without self-loops (max degree 4), deg=2 is feasible.
    """
    n = 5
    deg_out = np.array([2, 2, 2, 2, 2], dtype=np.uint64)
    deg_inp = np.array([2, 2, 2, 2, 2], dtype=np.uint64)
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=family,
        constraint=Constraint.DEGREE_EVENTS,
        degree_out=deg_out,
        degree_in=deg_inp,
        total_events=10,
        self_loops=False,
        seed=7,
        **kwargs,
    )
    assert sample.total_events == 10
    actual_out = np.bincount(sample.source, weights=sample.occ_num, minlength=n)
    actual_inp = np.bincount(sample.target, weights=sample.occ_num, minlength=n)
    assert actual_out.tolist() == deg_out.tolist()
    assert actual_inp.tolist() == deg_inp.tolist()


def test_fit_model_grandcanonical_strength_me() -> None:
    """Grand-canonical ME fixed strengths routes to Poisson fitter."""
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=S_OUT,
        strength_in=S_IN,
    )
    assert isinstance(fit, StrengthFit)
    assert fit.family == "poisson"


@pytest.mark.parametrize(
    "family,kwargs",
    [
        (ModelFamily.ME, {}),
        (ModelFamily.B, {"layers": 4}),
        (ModelFamily.W, {"layers": 2}),
    ],
)
def test_microcanonical_strength_preserves_constraints(
    family: ModelFamily, kwargs: dict
) -> None:
    """MC Strength preserves exact strength sequences for all families."""
    s_out = np.array([2, 3, 1], dtype=np.uint64)
    s_inp = np.array([3, 1, 2], dtype=np.uint64)
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=family,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_inp,
        self_loops=True,
        seed=7,
        **kwargs,
    )
    assert sample.total_events == 6
    actual_out = np.bincount(sample.source, weights=sample.occ_num, minlength=3)
    actual_inp = np.bincount(sample.target, weights=sample.occ_num, minlength=3)
    assert actual_out.tolist() == [2, 3, 1], f"out-strengths: {actual_out}"
    assert actual_inp.tolist() == [3, 1, 2], f"in-strengths: {actual_inp}"


@pytest.mark.parametrize(
    "family,kwargs",
    [
        (ModelFamily.ME, {}),
        (ModelFamily.B, {"layers": 10}),
        (ModelFamily.W, {"layers": 5}),
    ],
)
def test_microcanonical_strength_cost_fast(family: ModelFamily, kwargs: dict) -> None:
    """MC StrengthCost with target derived from a tiny synthetic network."""
    from menobis.analysis import directed_strengths
    from menobis.utilities.synthetic import generate_pa_geographic_network

    net = generate_pa_geographic_network(
        4, average_degree=2.0, events_per_edge=3.0, seed=7, self_loops=True
    )
    s = directed_strengths(net.edges)
    cx, cy = net.x, net.y
    source = net.edges.source
    target = net.edges.target
    occ = net.edges.occ_num
    dx = cx[source] - cx[target]
    dy = cy[source] - cy[target]
    target_cost = float(np.sqrt(dx**2 + dy**2) @ occ)

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
        seed=7,
        **kwargs,
    )
    n = len(cx)
    assert result.total_events == int(s.out.sum())
    actual_out = np.bincount(result.source, weights=result.occ_num, minlength=n)
    actual_inp = np.bincount(result.target, weights=result.occ_num, minlength=n)
    np.testing.assert_array_equal(actual_out, s.out)
    np.testing.assert_array_equal(actual_inp, s.incoming)


def test_sample_model_canonical_me_strength_uses_multinomial() -> None:
    """Canonical ME fixed-strength sampling has fixed total events."""
    fit = fit_model(
        ensemble=Ensemble.CANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=S_OUT,
        strength_in=S_IN,
    )
    sample = sample_model(
        ensemble=Ensemble.CANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        total_events=50,
        seed=7,
    )
    assert isinstance(sample, EdgeTable)
    assert sample.total_events == 50


def test_sample_model_microcanonical_me_strength_preserves_strengths() -> None:
    """Microcanonical ME fixed-strength sampling preserves strengths exactly."""
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=np.array([2, 1], dtype=np.uint64),
        strength_in=np.array([1, 2], dtype=np.uint64),
        seed=9,
    )
    assert sample.total_events == 3
    s_out = np.bincount(sample.source, weights=sample.occ_num, minlength=2)
    s_in = np.bincount(sample.target, weights=sample.occ_num, minlength=2)
    assert s_out.tolist() == [2, 1]
    assert s_in.tolist() == [1, 2]


def test_canonical_rejects_non_me_family() -> None:
    """Canonical ensemble is ME-only in MENoBiS."""
    with pytest.raises(UnsupportedModelCaseError, match=r"canonical.*ME"):
        fit_model(
            ensemble=Ensemble.CANONICAL,
            family=ModelFamily.B,
            constraint=Constraint.STRENGTH,
            strength_out=S_OUT,
            strength_in=S_IN,
            layers=3,
        )


def test_microcanonical_rejects_unsupported_constraint() -> None:
    """Microcanonical rejects an unknown ensemble combination.

    All six Constraint values now have microcanonical routes; an
    unsupported case is still rejected with the structured router error
    (here: an impossible strength target that violates a feasibility
    bound surfaces as a routed backend error, not a silent success).
    """
    # Strength with a B M=1 family and strength != degree must be
    # rejected early (Bernoulli invariant) rather than silently sampled.
    with pytest.raises(ValueError, match="M=1"):
        sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.B,
            constraint=Constraint.STRENGTH_DEGREE,
            strength_out=np.array([2], dtype=np.uint64),
            strength_in=np.array([2], dtype=np.uint64),
            degree_out=np.array([1], dtype=np.uint64),
            degree_in=np.array([1], dtype=np.uint64),
            layers=1,
        )


def test_microcanonical_strength_degree_now_supported() -> None:
    """The fixed-(s,k) feature exposes microcanonical STRENGTH_DEGREE.

    Extras-first exact constructor + capped first-return degree trace;
    exact strengths and degrees, no fit required.
    """
    from menobis.capabilities import SamplingExactness

    res = sample_model_detailed(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_DEGREE,
        strength_out=np.array([2, 2], dtype=np.uint64),
        strength_in=np.array([2, 2], dtype=np.uint64),
        degree_out=np.array([1, 1], dtype=np.uint32),
        degree_in=np.array([1, 1], dtype=np.uint32),
        self_loops=True,
        seed=7,
    )
    assert res.method == "microcanonical_fixed_strength_degree"
    assert res.exactness == SamplingExactness.EXACT_STATIONARY_MCMC
    assert len(res.edges) == 2
    net = res.edges
    dout = np.zeros(2, dtype=np.uint32)
    din = np.zeros(2, dtype=np.uint32)
    np.add.at(dout, net.source.astype(int), 1)
    np.add.at(din, net.target.astype(int), 1)
    np.testing.assert_array_equal(dout, np.array([1, 1], dtype=np.uint32))
    np.testing.assert_array_equal(din, np.array([1, 1], dtype=np.uint32))


def test_microcanonical_strength_edges_now_supported() -> None:
    """The fixed-(s,E) feature exposes microcanonical STRENGTH_EDGES.

    Exact stationary MCMC, no fit required.
    """
    from menobis.capabilities import SamplingExactness

    res = sample_model_detailed(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=np.array([2, 2], dtype=np.uint64),
        strength_in=np.array([2, 2], dtype=np.uint64),
        target_edges=2,
        self_loops=True,
        seed=1,
    )
    assert res.method == "microcanonical_fixed_strength_edges"
    assert res.exactness == SamplingExactness.EXACT_STATIONARY_MCMC
    assert len(res.edges) == 2


def test_fit_model_rejects_invalid_enum_value() -> None:
    """Constructing a ModelFamily from an invalid string raises ValueError."""
    with pytest.raises(ValueError):
        ModelFamily("invalid_family")
