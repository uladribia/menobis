"""Tests for canonical PA geographic synthetic benchmark networks."""

from __future__ import annotations

import numpy as np

from odme.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)


def test_pa_geographic_network_controls_density_and_events() -> None:
    """Generator fixes binary support size and total integer events."""
    network = generate_pa_geographic_network(
        50,
        density=0.08,
        events_per_edge=6.0,
        seed=123,
        self_loops=False,
    )

    expected_edges = round(0.08 * 50 * 49)
    assert network.edges.num_edges == expected_edges
    assert network.edges.total_events == round(expected_edges * 6.0)
    assert np.all(network.edges.source != network.edges.target)
    assert network.x.shape == (50,)
    assert network.y.shape == (50,)


def test_pa_geographic_network_self_loops_variant() -> None:
    """Self-loop variant includes diagonal edges and respects density."""
    network = generate_pa_geographic_network(
        50,
        density=0.08,
        events_per_edge=6.0,
        seed=123,
        self_loops=True,
    )

    expected_edges = round(0.08 * 50 * 50)
    assert network.edges.num_edges == expected_edges
    assert network.edges.total_events == round(expected_edges * 6.0)
    # Must contain at least one self-loop in the support
    has_self_loop = np.any(network.edges.source == network.edges.target)
    assert has_self_loop, "self_loops=True should produce diagonal edges"
    assert network.self_loops is True


def test_pa_geographic_weights_follow_degree_distance_scores() -> None:
    """Weights are normalized from degree product damped by distance."""
    network = generate_pa_geographic_network(
        80,
        average_degree=8.0,
        events_per_edge=10.0,
        seed=321,
        distance_decay=2.0,
        self_loops=False,
    )

    score = network.edge_scores()
    weights = network.edges.weight.astype(np.float64)
    assert np.all(score > 0.0)
    assert np.corrcoef(score, weights)[0, 1] > 0.35


def test_derived_constraints_match_generated_network() -> None:
    """All benchmark constraints are derived exactly from the PA network."""
    network = generate_pa_geographic_network(40, average_degree=6.0, seed=555)
    constraints = derive_synthetic_constraints(network)

    assert constraints.total_edges == float(network.edges.num_edges)
    assert constraints.total_events == network.edges.total_events
    np.testing.assert_allclose(
        constraints.strength_out.sum(), constraints.strength_in.sum()
    )
    np.testing.assert_allclose(
        constraints.degree_out.sum(), constraints.degree_in.sum()
    )
    assert constraints.total_cost > 0.0
    assert constraints.binomial_layers >= 1
