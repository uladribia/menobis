"""P0.10 filtering regression: every supported family/constraint combination.

Tests the shared pair-law path: filtering obtains p-values and expectations
from the same PairDistribution kernels used by generation.
"""

import numpy as np
import pytest

from menobis.models.spec import Constraint, ModelFamily
from menobis.models.types import EdgesEventsFit
from menobis.routing import filter_model, fit_model
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)


@pytest.fixture(scope="session")
def filtering_case():
    """PA-geographic network for filtering regression."""
    net = generate_pa_geographic_network(
        12, seed=1, self_loops=False, average_degree=4.0, events_per_edge=5.0
    )
    c = derive_synthetic_constraints(net)
    return {"network": net, "constraints": c}


_FAMILIES = [ModelFamily.ME, ModelFamily.B, ModelFamily.W]
_LAYERS = {ModelFamily.ME: 1, ModelFamily.B: 12, ModelFamily.W: 1}


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize(
    "constraint,extra",
    [
        (Constraint.STRENGTH, {}),
        (Constraint.STRENGTH_EDGES, {}),
        pytest.param(Constraint.STRENGTH_DEGREE, {}, marks=pytest.mark.heavy),
        (Constraint.DEGREE_EVENTS, {}),
    ],
)
def test_filtering_across_families_and_constraints(
    filtering_case, family, constraint, extra
) -> None:
    """filter_model runs without error for every supported combination.

    The null p-values are at least defined for the observed edges.
    """
    case = filtering_case
    net = case["network"]
    c = case["constraints"]
    layers = _LAYERS[family]

    fit = fit_model(
        family=family,
        constraint=constraint,
        strength_out=c.strength_out,
        strength_in=c.strength_in,
        degree_out=c.degree_out,
        degree_in=c.degree_in,
        target_edges=c.total_edges,
        total_events=c.total_events,
        self_loops=False,
        layers=layers,
    )
    result = filter_model(
        net.edges, family=family, constraint=constraint, fit=fit, tail="upper"
    )
    assert 0 <= len(result.upper.edges) <= len(net.edges)
    assert np.all(result.upper.upper_pvalue >= 0.0)
    assert np.all(result.upper.upper_pvalue <= 1.0)
    assert np.all(result.upper.expected > 0.0)
    assert np.all(result.upper.occupation > 0.0)


@pytest.mark.parametrize("family", _FAMILIES)
def test_filtering_strength_cost(family, filtering_case) -> None:
    """strength-cost filtering with coordinates."""
    case = filtering_case
    net = case["network"]
    c = case["constraints"]
    layers = _LAYERS[family]

    fit = fit_model(
        family=family,
        constraint=Constraint.STRENGTH_COST,
        strength_out=c.strength_out,
        strength_in=c.strength_in,
        coord_x=net.x,
        coord_y=net.y,
        target_cost=c.total_cost,
        self_loops=False,
        layers=layers,
    )
    result = filter_model(
        net.edges,
        family=family,
        constraint=Constraint.STRENGTH_COST,
        fit=fit,
        coord_x=net.x,
        coord_y=net.y,
        tail="upper",
    )
    assert np.all(result.upper.upper_pvalue >= 0.0)


def test_filtering_edges_events_symmetric(filtering_case) -> None:
    """EDGES_EVENTS filtering produces uniform expected values."""
    case = filtering_case
    net = case["network"]
    c = case["constraints"]
    for family, layers in [(ModelFamily.ME, 1), (ModelFamily.W, 1)]:
        fit = fit_model(
            family=family,
            constraint=Constraint.EDGES_EVENTS,
            target_edges=float(c.total_edges),
            total_events=int(c.total_events),
            node_count=len(c.strength_out),
            self_loops=False,
            layers=layers,
        )
        result = filter_model(
            net.edges,
            family=family,
            constraint=Constraint.EDGES_EVENTS,
            fit=fit,
            tail="upper",
        )
        assert isinstance(fit, EdgesEventsFit)
        expected = fit.occupation * fit.positive_mean
        if len(result.upper.edges) > 0:
            np.testing.assert_allclose(result.upper.expected, expected, rtol=1e-10)


def test_filtering_zero_occupation_and_boundary() -> None:
    """Zero-occupation pairs and B layer-capacity tests."""
    import numpy as np

    net = generate_pa_geographic_network(
        12, seed=1, self_loops=False, average_degree=4.0, events_per_edge=5.0
    )
    c = derive_synthetic_constraints(net)

    # B with layers well above max occupation
    fit = fit_model(
        family=ModelFamily.B,
        constraint=Constraint.STRENGTH,
        strength_out=c.strength_out,
        strength_in=c.strength_in,
        layers=20,
        self_loops=False,
    )
    result = filter_model(
        net.edges,
        family=ModelFamily.B,
        constraint=Constraint.STRENGTH,
        fit=fit,
        tail="upper",
    )
    assert np.all(result.upper.occupation > 0.0)
    assert np.all(result.upper.expected > 0.0)
    # All observed occupations must be within B capacity
    assert np.all(net.edges.occ_num <= 20)
