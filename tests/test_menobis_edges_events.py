"""P0.3 E2E pipeline tests for the EDGES_EVENTS constraint.

Mandatory E2E pipeline (testing policy): generate a realistic network,
derive E/T constraints from it (guaranteed feasible), fit the
grand-canonical EDGES_EVENTS model, sample, and verify constraint
recovery in expectation.
"""

import numpy as np
import pytest

from menobis.models.spec import Constraint, ModelFamily
from menobis.models.types import EdgesEventsFit
from menobis.routing import filter_model, fit_model, sample_model
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)


@pytest.fixture(scope="module")
def edges_events_case():
    """Dense PA-geographic network with derived feasible E/T constraints."""
    net = generate_pa_geographic_network(
        12, seed=1, self_loops=False, average_degree=4.0, events_per_edge=5.0
    )
    c = derive_synthetic_constraints(net)
    return {
        "network": net,
        "total_edges": float(c.total_edges),
        "total_events": int(c.total_events),
        "node_count": len(c.strength_out),
    }


def _n_pairs(node_count: int, self_loops: bool) -> int:
    return node_count * node_count if self_loops else node_count * (node_count - 1)


@pytest.mark.parametrize(
    "family,layers",
    [
        (ModelFamily.ME, 1),
        (ModelFamily.B, 12),
        (ModelFamily.W, 1),
    ],
)
def test_edges_events_fit_recovers_constraints_in_expectation(
    edges_events_case, family, layers
) -> None:
    """Fitted expectations must reproduce E and T within tolerance."""
    case = edges_events_case
    fit = fit_model(
        family=family,
        constraint=Constraint.EDGES_EVENTS,
        target_edges=case["total_edges"],
        total_events=case["total_events"],
        node_count=case["node_count"],
        self_loops=False,
        layers=layers,
    )
    assert isinstance(fit, EdgesEventsFit)
    assert fit.converged
    n_pairs = _n_pairs(case["node_count"], False)
    # occupation = E / n_pairs
    assert fit.occupation == pytest.approx(case["total_edges"] / n_pairs, rel=1e-10)
    # conditional mean = T / E
    assert fit.positive_mean == pytest.approx(
        case["total_events"] / case["total_edges"], rel=1e-10
    )
    # expected events per pair = occupation * positive_mean
    expected_per_pair = fit.occupation * fit.positive_mean
    assert expected_per_pair * n_pairs == pytest.approx(case["total_events"], rel=1e-9)
    # For B, positive mean cannot exceed layers
    if family is ModelFamily.B:
        assert fit.positive_mean <= layers


@pytest.mark.parametrize("family", [ModelFamily.ME, ModelFamily.B, ModelFamily.W])
def test_edges_events_sample_recovers_constraints_stochastically(
    edges_events_case, family
) -> None:
    """Sampled networks recover E and T within 5x fitting tolerance."""
    case = edges_events_case
    layers = 12 if family is ModelFamily.B else 1
    fit = fit_model(
        family=family,
        constraint=Constraint.EDGES_EVENTS,
        target_edges=case["total_edges"],
        total_events=case["total_events"],
        node_count=case["node_count"],
        self_loops=False,
        layers=layers,
    )
    edges_total = 0
    events_total = 0
    n_samples = 30
    for seed in range(n_samples):
        sample = sample_model(
            family=family,
            constraint=Constraint.EDGES_EVENTS,
            fit=fit,
            seed=seed,
        )
        # support: every sampled occupation is positive
        assert np.all(sample.occ_num >= 1)
        if family is ModelFamily.B:
            assert np.all(sample.occ_num <= layers)
        edges_total += sample.num_edges
        events_total += sample.total_events
    rel_tol = 0.05 * max(1.0, case["total_edges"] / case["node_count"])
    assert edges_total / n_samples == pytest.approx(case["total_edges"], rel=rel_tol)
    assert events_total / n_samples == pytest.approx(case["total_events"], rel=rel_tol)


@pytest.mark.parametrize("family", [ModelFamily.ME, ModelFamily.B, ModelFamily.W])
def test_edges_events_filter_runs_on_all_families(edges_events_case, family) -> None:
    """Filtering against the EDGES_EVENTS null returns a valid FilterResult."""
    case = edges_events_case
    layers = 12 if family is ModelFamily.B else 1
    fit = fit_model(
        family=family,
        constraint=Constraint.EDGES_EVENTS,
        target_edges=case["total_edges"],
        total_events=case["total_events"],
        node_count=case["node_count"],
        self_loops=False,
        layers=layers,
    )
    assert isinstance(fit, EdgesEventsFit)
    result = filter_model(
        case["network"].edges,
        family=family,
        constraint=Constraint.EDGES_EVENTS,
        fit=fit,
        tail="upper",
    )
    # significant subset is contained in the observed edges
    assert len(result.upper.edges) <= len(case["network"].edges)
    # p-values are probabilities
    assert np.all(result.upper.upper_pvalue >= 0.0)
    assert np.all(result.upper.upper_pvalue <= 1.0)
    # expected values match the symmetric model: occupation * positive_mean
    expected = fit.occupation * fit.positive_mean
    if len(result.upper.edges) > 0:
        assert np.allclose(result.upper.expected, expected)


def test_edges_events_fit_rejects_infeasible_inputs(edges_events_case) -> None:
    case = edges_events_case
    with pytest.raises(ValueError, match="total_events must be >= total_edges"):
        fit_model(
            family=ModelFamily.ME,
            constraint=Constraint.EDGES_EVENTS,
            target_edges=case["total_edges"],
            total_events=1,
            node_count=case["node_count"],
            self_loops=False,
        )
    with pytest.raises(ValueError, match="candidate pairs"):
        fit_model(
            family=ModelFamily.ME,
            constraint=Constraint.EDGES_EVENTS,
            target_edges=_n_pairs(case["node_count"], False) + 1,
            total_events=_n_pairs(case["node_count"], False) + 10,
            node_count=case["node_count"],
            self_loops=False,
        )
    with pytest.raises(ValueError, match="node_count"):
        fit_model(
            family=ModelFamily.ME,
            constraint=Constraint.EDGES_EVENTS,
            target_edges=case["total_edges"],
            total_events=case["total_events"],
            self_loops=False,
        )


def test_edges_events_sample_is_seed_reproducible(edges_events_case) -> None:
    case = edges_events_case
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.EDGES_EVENTS,
        target_edges=case["total_edges"],
        total_events=case["total_events"],
        node_count=case["node_count"],
        self_loops=False,
    )
    a = sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.EDGES_EVENTS,
        fit=fit,
        seed=11,
    )
    b = sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.EDGES_EVENTS,
        fit=fit,
        seed=11,
    )
    np.testing.assert_array_equal(a.source, b.source)
    np.testing.assert_array_equal(a.target, b.target)
    np.testing.assert_array_equal(a.occ_num, b.occ_num)


def test_edges_events_families_differ(edges_events_case) -> None:
    """ME != B != W on identical E/T inputs (model differentiation)."""
    case = edges_events_case
    qs = []
    for family, layers in [
        (ModelFamily.ME, 1),
        (ModelFamily.B, 12),
        (ModelFamily.W, 1),
    ]:
        fit = fit_model(
            family=family,
            constraint=Constraint.EDGES_EVENTS,
            target_edges=case["total_edges"],
            total_events=case["total_events"],
            node_count=case["node_count"],
            self_loops=False,
            layers=layers,
        )
        assert isinstance(fit, EdgesEventsFit)
        qs.append(fit.q)
    assert len({round(q, 6) for q in qs}) == 3
