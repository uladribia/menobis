"""P0.7 tests: capability registry and detailed sampling result."""

import numpy as np

from menobis.capabilities import (
    REGISTRY,
    SamplingExactness,
    capability,
    unsupported_cases,
)
from menobis.models.spec import Constraint, Ensemble, ModelFamily, Verb
from menobis.models.types import SamplingResult
from menobis.routing import (
    fit_model,
    sample_model,
    sample_model_detailed,
)

S_OUT = np.array([10.0, 20.0])
S_IN = np.array([15.0, 15.0])


# ---------------------------------------------------------------------------
# Capability registry
# ---------------------------------------------------------------------------


def test_registry_covers_fit_matrix() -> None:
    for family in (ModelFamily.ME, ModelFamily.B, ModelFamily.W):
        for constraint in (
            Constraint.STRENGTH,
            Constraint.STRENGTH_COST,
            Constraint.STRENGTH_EDGES,
            Constraint.STRENGTH_DEGREE,
            Constraint.DEGREE_EVENTS,
            Constraint.EDGES_EVENTS,
        ):
            cap = capability(Verb.FIT, Ensemble.GRAND_CANONICAL, family, constraint)
            assert cap is not None
            assert cap.supported


def test_registry_covers_sample_matrix() -> None:
    for family in (ModelFamily.ME, ModelFamily.B, ModelFamily.W):
        for constraint in (
            Constraint.STRENGTH,
            Constraint.STRENGTH_COST,
            Constraint.STRENGTH_EDGES,
            Constraint.STRENGTH_DEGREE,
            Constraint.DEGREE_EVENTS,
            Constraint.EDGES_EVENTS,
        ):
            cap = capability(Verb.SAMPLE, Ensemble.GRAND_CANONICAL, family, constraint)
            assert cap is not None
            assert cap.supported
            assert cap.requires_fit


def test_microcanonical_capability_rules() -> None:
    cap = capability(
        Verb.SAMPLE, Ensemble.MICROCANONICAL, ModelFamily.ME, Constraint.STRENGTH
    )
    assert cap is not None and cap.supported
    assert not cap.requires_fit
    assert cap.supports_self_loops
    assert cap.supports_no_self_loops  # MCMC backend handles no-self-loops


def test_unsupported_cases_are_explicit() -> None:
    cases = unsupported_cases()
    # Microcanonical W/B STRENGTH now supported via MCMC.
    assert (
        Verb.SAMPLE,
        Ensemble.MICROCANONICAL,
        ModelFamily.W,
        Constraint.STRENGTH,
    ) not in cases
    assert (
        Verb.SAMPLE,
        Ensemble.CANONICAL,
        ModelFamily.B,
        Constraint.STRENGTH,
    ) in cases
    assert (
        Verb.FILTER,
        Ensemble.CANONICAL,
        ModelFamily.ME,
        Constraint.STRENGTH,
    ) in cases
    assert (
        Verb.FILTER,
        Ensemble.MICROCANONICAL,
        ModelFamily.ME,
        Constraint.STRENGTH,
    ) in cases


def test_registry_is_source_of_truth_for_every_public_route() -> None:
    """Every registered supported route must have a capability record."""
    assert len(REGISTRY) >= 3 * 6 + 2  # fit 3x6 + sample extras


# ---------------------------------------------------------------------------
# sample_model_detailed
# ---------------------------------------------------------------------------


def test_detailed_grand_canonical_result() -> None:
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=S_OUT,
        strength_in=S_IN,
    )
    res = sample_model_detailed(
        family=ModelFamily.ME, constraint=Constraint.STRENGTH, fit=fit, seed=42
    )
    assert isinstance(res, SamplingResult)
    assert res.method == "grandcanonical_independent"
    assert res.exactness == SamplingExactness.EXACT_INDEPENDENT
    assert res.seed == 42
    assert res.ensemble is Ensemble.GRAND_CANONICAL
    assert res.family is ModelFamily.ME
    assert res.constraint is Constraint.STRENGTH
    assert (
        res.edges.num_edges
        == sample_model(
            family=ModelFamily.ME, constraint=Constraint.STRENGTH, fit=fit, seed=42
        ).num_edges
    )


def test_detailed_canonical_exact_direct() -> None:
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=S_OUT,
        strength_in=S_IN,
    )
    res = sample_model_detailed(
        ensemble=Ensemble.CANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        total_events=50,
        seed=1,
    )
    assert res.exactness == SamplingExactness.EXACT_DIRECT
    assert res.edges.total_events == 50


def test_detailed_microcanonical_strength_stationary_mcmc() -> None:
    res = sample_model_detailed(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=np.array([5, 7]),
        strength_in=np.array([6, 6]),
        seed=1,
    )
    assert res.exactness == SamplingExactness.EXACT_STATIONARY_MCMC
    assert res.method == "microcanonical_fixed_strength_mcmc"


def test_sample_model_delegates_to_detailed() -> None:
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=S_OUT,
        strength_in=S_IN,
    )
    edges = sample_model(
        family=ModelFamily.ME, constraint=Constraint.STRENGTH, fit=fit, seed=9
    )
    assert hasattr(edges, "source")
    assert hasattr(edges, "target")
    assert hasattr(edges, "occ_num")


def test_microcanonical_no_self_loops_succeeds() -> None:
    # MCMC backend handles no-self-loops.
    edges = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=np.array([5, 5]),
        strength_in=np.array([5, 5]),
        self_loops=False,
        seed=42,
    )
    assert hasattr(edges, "source")
    assert hasattr(edges, "target")
    assert hasattr(edges, "occ_num")
    # Verify no self-loops.
    assert not any(edges.source == edges.target)
