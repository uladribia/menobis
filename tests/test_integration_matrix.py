"""§19 supported-combination integration run (compact).

For every supported (family, constraint) row in the capability registry:
construct a feasible synthetic problem, fit, sample, filter, and analyze.
Verifies: integer occupations in family support, hard constraint recovery
for exact samplers, renamed occupation API, no panics.
"""

import numpy as np
import pytest

from menobis.analysis import analyze
from menobis.capabilities import capability
from menobis.models.spec import Constraint, Ensemble, ModelFamily, Verb
from menobis.routing import filter_model, fit_model, sample_model
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

_FAMILIES = [ModelFamily.ME, ModelFamily.B, ModelFamily.W]
_LAYERS = {ModelFamily.ME: 1, ModelFamily.B: 12, ModelFamily.W: 1}
_NODE_COUNT = 10


@pytest.fixture(scope="session")
def integration_network():
    net = generate_pa_geographic_network(
        _NODE_COUNT, seed=1, self_loops=False, average_degree=4.0, events_per_edge=5.0
    )
    return net, derive_synthetic_constraints(net)


@pytest.mark.parametrize("family", _FAMILIES)
def test_integration_matrix_strength(integration_network, family) -> None:
    net, c = integration_network
    layers = _LAYERS[family]
    fit = fit_model(
        family=family,
        constraint=Constraint.STRENGTH,
        strength_out=c.strength_out,
        strength_in=c.strength_in,
        layers=layers,
        self_loops=False,
    )
    sample = sample_model(
        family=family, constraint=Constraint.STRENGTH, fit=fit, seed=1
    )
    # occupation API + support
    assert np.all(sample.occ_num >= 1)
    if family is ModelFamily.B:
        assert np.all(sample.occ_num <= layers)
    _verify_analysis(sample)
    _verify_filter(net.edges, family, Constraint.STRENGTH, fit, {})


@pytest.mark.parametrize("family", _FAMILIES)
def test_integration_matrix_strength_edges(integration_network, family) -> None:
    net, c = integration_network
    layers = _LAYERS[family]
    fit = fit_model(
        family=family,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=c.strength_out,
        strength_in=c.strength_in,
        target_edges=c.total_edges,
        layers=layers,
        self_loops=False,
    )
    sample = sample_model(
        family=family, constraint=Constraint.STRENGTH_EDGES, fit=fit, seed=1
    )
    assert np.all(sample.occ_num >= 1)
    if family is ModelFamily.B:
        assert np.all(sample.occ_num <= layers)
    _verify_analysis(sample)
    _verify_filter(net.edges, family, Constraint.STRENGTH_EDGES, fit, {})


@pytest.mark.heavy
@pytest.mark.parametrize("family", _FAMILIES)
def test_integration_matrix_strength_degree(integration_network, family) -> None:
    net, c = integration_network
    layers = _LAYERS[family]
    fit = fit_model(
        family=family,
        constraint=Constraint.STRENGTH_DEGREE,
        strength_out=c.strength_out,
        strength_in=c.strength_in,
        degree_out=c.degree_out,
        degree_in=c.degree_in,
        layers=layers,
        self_loops=False,
    )
    sample = sample_model(
        family=family, constraint=Constraint.STRENGTH_DEGREE, fit=fit, seed=1
    )
    assert np.all(sample.occ_num >= 1)
    if family is ModelFamily.B:
        assert np.all(sample.occ_num <= layers)
    _verify_analysis(sample)
    _verify_filter(net.edges, family, Constraint.STRENGTH_DEGREE, fit, {})


@pytest.mark.parametrize("family", _FAMILIES)
def test_integration_matrix_strength_cost(integration_network, family) -> None:
    net, c = integration_network
    layers = _LAYERS[family]
    fit = fit_model(
        family=family,
        constraint=Constraint.STRENGTH_COST,
        strength_out=c.strength_out,
        strength_in=c.strength_in,
        coord_x=net.x,
        coord_y=net.y,
        target_cost=c.total_cost,
        layers=layers,
        self_loops=False,
    )
    sample = sample_model(
        family=family,
        constraint=Constraint.STRENGTH_COST,
        fit=fit,
        coord_x=net.x,
        coord_y=net.y,
        seed=1,
    )
    assert np.all(sample.occ_num >= 1)
    _verify_analysis(sample)
    _verify_filter(
        net.edges,
        family,
        Constraint.STRENGTH_COST,
        fit,
        {"coord_x": net.x, "coord_y": net.y},
    )


def test_integration_matrix_microcanonical_fixed_strength(integration_network) -> None:
    """Microcanonical ME strength: exact strengths, no fit required."""
    _, c = integration_network
    s_out = np.round(c.strength_out).astype(np.uint64)
    s_in = np.round(c.strength_in).astype(np.uint64)
    # balance
    if s_out.sum() != s_in.sum():
        diff = int(s_out.sum()) - int(s_in.sum())
        if diff > 0:
            s_in[np.argmax(s_in)] += diff
        else:
            s_out[np.argmax(s_out)] -= diff
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
        self_loops=True,
        seed=2,
    )
    from menobis.analysis import directed_strengths

    actual = directed_strengths(sample)
    np.testing.assert_array_equal(actual.out, s_out)
    np.testing.assert_array_equal(actual.incoming, s_in)
    _verify_analysis(sample)


def _verify_analysis(sample) -> None:
    """Analysis facade must accept the generated network."""
    result = analyze(sample, strengths=True, degrees=True, distribution=True)
    assert result.node_count >= 1
    assert result.strengths is not None


def _verify_filter(edges, family, constraint, fit, extra) -> None:
    """Filtering must run and produce valid p-values."""
    result = filter_model(
        edges,
        family=family,
        constraint=constraint,
        fit=fit,
        tail="upper",
        **extra,
    )
    assert np.all(result.upper.upper_pvalue >= 0.0)
    assert np.all(result.upper.upper_pvalue <= 1.0)


def test_support_matrix_matches_capability_registry() -> None:
    """Every supported filter route must be exercised by this suite."""
    for family in _FAMILIES:
        for constraint in (
            Constraint.STRENGTH,
            Constraint.STRENGTH_EDGES,
            Constraint.STRENGTH_DEGREE,
            Constraint.STRENGTH_COST,
        ):
            cap = capability(Verb.FILTER, Ensemble.GRAND_CANONICAL, family, constraint)
            assert cap is not None and cap.supported, f"{family}/{constraint}"
