"""End-to-end ensemble validation on canonical PA geographic networks."""

import numpy as np

from odme.analysis import directed_degrees, directed_strengths
from odme.data.frames import EdgeTable
from odme.models import (
    fit_degree_bernoulli,
    fit_strength_cost_poisson,
    fit_strength_degree_poisson,
    fit_strength_edges_poisson,
    fit_strength_poisson,
    sample_degree_events_poisson,
    sample_strength_cost_poisson,
    sample_strength_degree_poisson,
    sample_strength_edges_poisson,
    sample_strength_multinomial,
    sample_strength_poisson,
    sample_strength_poisson_multinomial,
)
from odme.utilities.synthetic import generate_pa_geographic_network


def _pa_network(seed: int = 20240521) -> EdgeTable:
    """Create the canonical PA geographic E2E fixture."""
    return generate_pa_geographic_network(
        8,
        average_degree=2.0,
        events_per_edge=5.0,
        seed=seed,
        self_loops=False,
    ).edges


def _observed_cost(
    edges: EdgeTable,
    cost_src: np.ndarray,
    cost_tgt: np.ndarray,
    cost_val: np.ndarray,
) -> float:
    cost_map = {
        (int(s), int(t)): float(c)
        for s, t, c in zip(cost_src, cost_tgt, cost_val, strict=True)
    }
    return sum(
        float(w) * cost_map.get((int(s), int(t)), 0.0)
        for s, t, w in zip(edges.source, edges.target, edges.weight, strict=True)
    )


def _strength_out(edges: EdgeTable, node_count: int) -> np.ndarray:
    out = np.zeros(node_count, dtype=np.float64)
    np.add.at(out, edges.source.astype(int), edges.weight.astype(float))
    return out


def _degree_out(edges: EdgeTable, node_count: int) -> np.ndarray:
    out = np.zeros(node_count, dtype=np.float64)
    np.add.at(out, edges.source.astype(int), 1.0)
    return out


def _mean_std(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack(values)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=1)


def _assert_mean_close(
    mean: np.ndarray, std: np.ndarray, expected: np.ndarray, repetitions: int
) -> None:
    standard_error = std / np.sqrt(repetitions)
    tolerance = 6.0 * standard_error + 1.0
    assert np.all(np.abs(mean - expected) <= tolerance + 0.08 * np.abs(expected))


def test_fixed_strength_me_ensembles_match_observed_constraints() -> None:
    """Fixed-strength ME samplers recover PA-derived strengths in ensemble mean."""
    edges = _pa_network()
    strengths = directed_strengths(edges)
    fit = fit_strength_poisson(
        strengths.out.astype(float), strengths.incoming.astype(float), self_loops=False
    )
    repetitions = 300

    poisson_out: list[np.ndarray] = []
    multinomial_out: list[np.ndarray] = []
    poisson_multinomial_out: list[np.ndarray] = []
    multinomial_totals = []
    for seed in range(repetitions):
        p_sample = sample_strength_poisson(fit.x, fit.y, self_loops=False, seed=seed)
        m_sample = sample_strength_multinomial(
            fit.x,
            fit.y,
            total_events=edges.total_events,
            self_loops=False,
            seed=seed,
        )
        pm_sample = sample_strength_poisson_multinomial(
            fit.x, fit.y, self_loops=False, seed=seed
        )
        poisson_out.append(_strength_out(p_sample, len(strengths.out)))
        multinomial_out.append(_strength_out(m_sample, len(strengths.out)))
        poisson_multinomial_out.append(_strength_out(pm_sample, len(strengths.out)))
        multinomial_totals.append(m_sample.total_events)

    for observed in [poisson_out, multinomial_out, poisson_multinomial_out]:
        mean, std = _mean_std(observed)
        _assert_mean_close(mean, std, strengths.out.astype(float), repetitions)
    assert set(multinomial_totals) == {edges.total_events}


def test_fixed_degree_total_events_ensemble_matches_degrees_and_trips() -> None:
    """Fixed-degree ME samples recover PA-derived degrees and trips in mean."""
    edges = _pa_network()
    degrees = directed_degrees(edges)
    repetitions = 400
    fit = fit_degree_bernoulli(
        degrees.out.astype(float), degrees.incoming.astype(float), self_loops=False
    )

    sampled_degrees: list[np.ndarray] = []
    totals = []
    for seed in range(repetitions):
        sample = sample_degree_events_poisson(
            fit, total_events=edges.total_events, self_loops=False, seed=seed
        )
        sampled_degrees.append(_degree_out(sample, len(degrees.out)))
        totals.append(sample.total_events)

    mean, std = _mean_std(sampled_degrees)
    _assert_mean_close(mean, std, degrees.out.astype(float), repetitions)
    assert abs(float(np.mean(totals)) - edges.total_events) < 0.12 * edges.total_events
    assert float(np.std(totals, ddof=1)) > 0.0


def test_strength_degree_me_ensemble_matches_strengths_and_degrees() -> None:
    """Strength-degree ME samples recover PA-derived constraints in mean."""
    edges = _pa_network()
    strengths = directed_strengths(edges)
    degrees = directed_degrees(edges)
    repetitions = 500
    fit = fit_strength_degree_poisson(
        strengths.out.astype(float),
        strengths.incoming.astype(float),
        degrees.out.astype(float),
        degrees.incoming.astype(float),
        self_loops=False,
    )

    sampled_strengths: list[np.ndarray] = []
    sampled_degrees: list[np.ndarray] = []
    for seed in range(repetitions):
        sample = sample_strength_degree_poisson(fit, seed=seed)
        sampled_strengths.append(_strength_out(sample, len(strengths.out)))
        sampled_degrees.append(_degree_out(sample, len(degrees.out)))

    mean_s, std_s = _mean_std(sampled_strengths)
    mean_k, std_k = _mean_std(sampled_degrees)
    _assert_mean_close(mean_s, std_s, strengths.out.astype(float), repetitions)
    _assert_mean_close(mean_k, std_k, degrees.out.astype(float), repetitions)


def test_strength_edges_me_ensemble_matches_strengths_and_binary_edges() -> None:
    """Strength+edge-count ME samples recover PA-derived constraints in mean."""
    edges = _pa_network()
    strengths = directed_strengths(edges)
    target_edges = float(edges.num_edges)
    repetitions = 500
    fit = fit_strength_edges_poisson(
        strengths.out.astype(float),
        strengths.incoming.astype(float),
        target_edges,
        self_loops=False,
    )

    sampled_strengths: list[np.ndarray] = []
    edge_counts = []
    for seed in range(repetitions):
        sample = sample_strength_edges_poisson(fit, seed=seed)
        sampled_strengths.append(_strength_out(sample, len(strengths.out)))
        edge_counts.append(sample.num_edges)

    mean_s, std_s = _mean_std(sampled_strengths)
    _assert_mean_close(mean_s, std_s, strengths.out.astype(float), repetitions)
    assert (
        abs(float(np.mean(edge_counts)) - target_edges)
        < 4.0 * float(np.std(edge_counts, ddof=1)) / np.sqrt(repetitions) + 1.0
    )


def test_strength_cost_me_ensemble_matches_strengths_and_cost() -> None:
    """Strength-cost ME samples recover PA-derived strengths and cost in mean."""
    network = generate_pa_geographic_network(
        8,
        average_degree=2.0,
        events_per_edge=5.0,
        seed=20240522,
        self_loops=False,
    )
    edges = network.edges
    cost_src, cost_tgt, cost_val = network.complete_cost_triples()
    strengths = directed_strengths(edges)
    target_cost = _observed_cost(edges, cost_src, cost_tgt, cost_val)

    repetitions = 300
    fit = fit_strength_cost_poisson(
        strengths.out.astype(float),
        strengths.incoming.astype(float),
        cost_src,
        cost_tgt,
        cost_val,
        target_cost,
        self_loops=False,
        tolerance=1e-4,
        max_iterations=20000,
    )
    assert fit.converged, "fit_strength_cost_poisson did not converge"

    sampled_strengths: list[np.ndarray] = []
    sampled_costs: list[float] = []
    for seed in range(repetitions):
        sample = sample_strength_cost_poisson(
            fit, cost_src, cost_tgt, cost_val, seed=seed
        )
        sampled_strengths.append(_strength_out(sample, len(strengths.out)))
        sampled_costs.append(_observed_cost(sample, cost_src, cost_tgt, cost_val))

    mean_s, std_s = _mean_std(sampled_strengths)
    _assert_mean_close(mean_s, std_s, strengths.out.astype(float), repetitions)

    mean_cost = float(np.mean(sampled_costs))
    std_cost = float(np.std(sampled_costs, ddof=1))
    se_cost = std_cost / np.sqrt(repetitions)
    assert abs(mean_cost - target_cost) < 6.0 * se_cost + 0.08 * target_cost + 1.0
