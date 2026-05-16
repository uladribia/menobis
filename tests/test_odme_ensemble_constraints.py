"""End-to-end ensemble validation on a non-trivial synthetic network."""

import numpy as np

from odme.analysis import directed_degrees, directed_strengths
from odme.data.frames import EdgeTable
from odme.models import (
    fit_fixed_degree_binary,
    fit_fixed_strength_me,
    fit_strength_cost_me,
    fit_strength_degree_me,
    fit_strength_edges_me,
    sample_fixed_degree_events_me,
    sample_multinomial,
    sample_poisson,
    sample_poisson_multinomial,
    sample_strength_cost_me,
    sample_strength_degree_me,
    sample_strength_edges_me,
)


def _metric_distance_matrix(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a valid metric distance matrix from 2-D node positions."""
    rng = np.random.default_rng(123)
    positions = rng.uniform(0.0, 10.0, size=(n, 2))
    sources: list[int] = []
    targets: list[int] = []
    costs: list[float] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(np.linalg.norm(positions[i] - positions[j]))
            sources.append(i)
            targets.append(j)
            costs.append(d)
    return np.array(sources), np.array(targets), np.array(costs)


def _observed_cost(
    edges: EdgeTable,
    cost_src: np.ndarray,
    cost_tgt: np.ndarray,
    cost_val: np.ndarray,
) -> float:
    cost_map: dict[tuple[int, int], float] = {}
    for s, t, c in zip(cost_src, cost_tgt, cost_val, strict=True):
        cost_map[(int(s), int(t))] = float(c)
    return sum(
        float(w) * cost_map.get((int(s), int(t)), 0.0)
        for s, t, w in zip(edges.source, edges.target, edges.weight, strict=True)
    )


def _pareto_like_network() -> EdgeTable:
    """Create a deterministic heavy-tailed directed weighted network."""
    out_factor = np.array([30, 18, 11, 7, 5, 3, 2, 1], dtype=np.uint64)
    in_factor = np.array([23, 15, 10, 6, 4, 3, 2, 1], dtype=np.uint64)
    sources: list[int] = []
    targets: list[int] = []
    weights: list[int] = []
    for i, out_value in enumerate(out_factor):
        for j, in_value in enumerate(in_factor):
            if i == j:
                continue
            sources.append(i)
            targets.append(j)
            weights.append(int(max(1, (out_value * in_value) // 9)))
    return EdgeTable(
        source=np.asarray(sources, dtype=np.uint64),
        target=np.asarray(targets, dtype=np.uint64),
        weight=np.asarray(weights, dtype=np.uint64),
    )


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
    """Fixed-strength ME samplers recover observed strengths in ensemble mean."""
    edges = _pareto_like_network()
    strengths = directed_strengths(edges)
    fit = fit_fixed_strength_me(
        strengths.out.astype(float), strengths.incoming.astype(float)
    )
    repetitions = 300

    poisson_out: list[np.ndarray] = []
    multinomial_out: list[np.ndarray] = []
    poisson_multinomial_out: list[np.ndarray] = []
    multinomial_totals = []
    for seed in range(repetitions):
        p_sample = sample_poisson(fit.x, fit.y, seed=seed)
        m_sample = sample_multinomial(
            fit.x, fit.y, total_events=edges.total_events, seed=seed
        )
        pm_sample = sample_poisson_multinomial(fit.x, fit.y, seed=seed)
        poisson_out.append(directed_strengths(p_sample).out.astype(float))
        multinomial_out.append(directed_strengths(m_sample).out.astype(float))
        poisson_multinomial_out.append(directed_strengths(pm_sample).out.astype(float))
        multinomial_totals.append(m_sample.total_events)

    for observed in [poisson_out, multinomial_out, poisson_multinomial_out]:
        mean, std = _mean_std(observed)
        _assert_mean_close(mean, std, strengths.out.astype(float), repetitions)
    assert set(multinomial_totals) == {edges.total_events}


def test_fixed_degree_total_events_ensemble_matches_degrees_and_trips() -> None:
    """Fixed-degree ME ME samples recover degrees and total trips in mean."""
    edges = _pareto_like_network()
    degrees = directed_degrees(edges)
    repetitions = 400
    fit = fit_fixed_degree_binary(
        degrees.out.astype(float), degrees.incoming.astype(float), self_loops=True
    )

    sampled_degrees: list[np.ndarray] = []
    totals = []
    for seed in range(repetitions):
        sample = sample_fixed_degree_events_me(
            fit, total_events=edges.total_events, seed=seed
        )
        sampled_degrees.append(directed_degrees(sample).out.astype(float))
        totals.append(sample.total_events)

    mean, std = _mean_std(sampled_degrees)
    _assert_mean_close(mean, std, degrees.out.astype(float), repetitions)
    assert abs(float(np.mean(totals)) - edges.total_events) < 0.12 * edges.total_events
    assert float(np.std(totals, ddof=1)) > 0.0


def test_strength_degree_me_ensemble_matches_strengths_and_degrees() -> None:
    """Exact ME strength-degree ME samples recover both constraints in mean."""
    edges = _pareto_like_network()
    strengths = directed_strengths(edges)
    degrees = directed_degrees(edges)
    repetitions = 500
    fit = fit_strength_degree_me(
        strengths.out.astype(float),
        strengths.incoming.astype(float),
        degrees.out.astype(float),
        degrees.incoming.astype(float),
        self_loops=True,
    )

    sampled_strengths: list[np.ndarray] = []
    sampled_degrees: list[np.ndarray] = []
    for seed in range(repetitions):
        sample = sample_strength_degree_me(fit, seed=seed)
        sampled_strengths.append(directed_strengths(sample).out.astype(float))
        sampled_degrees.append(directed_degrees(sample).out.astype(float))

    mean_s, std_s = _mean_std(sampled_strengths)
    mean_k, std_k = _mean_std(sampled_degrees)
    _assert_mean_close(mean_s, std_s, strengths.out.astype(float), repetitions)
    _assert_mean_close(mean_k, std_k, degrees.out.astype(float), repetitions)


def test_strength_edges_me_ensemble_matches_strengths_and_binary_edges() -> None:
    """Fixed-strength+edge-count ME samples recover constraints in mean."""
    edges = _pareto_like_network()
    strengths = directed_strengths(edges)
    target_edges = float(edges.num_edges)
    repetitions = 500
    fit = fit_strength_edges_me(
        strengths.out.astype(float),
        strengths.incoming.astype(float),
        target_edges,
        self_loops=True,
    )

    sampled_strengths: list[np.ndarray] = []
    edge_counts = []
    for seed in range(repetitions):
        sample = sample_strength_edges_me(fit, seed=seed)
        sampled_strengths.append(directed_strengths(sample).out.astype(float))
        edge_counts.append(sample.num_edges)

    mean_s, std_s = _mean_std(sampled_strengths)
    _assert_mean_close(mean_s, std_s, strengths.out.astype(float), repetitions)
    assert (
        abs(float(np.mean(edge_counts)) - target_edges)
        < 4.0 * float(np.std(edge_counts, ddof=1)) / np.sqrt(repetitions) + 1.0
    )


def test_strength_cost_me_ensemble_matches_strengths_and_cost() -> None:
    """Strength-cost ME samples recover strengths and total cost in mean."""
    # Build a spatially-structured network where cost matters.
    n = 5
    rng = np.random.default_rng(42)
    positions = rng.uniform(0.0, 10.0, size=(n, 2))
    true_x = np.array([3.0, 2.5, 2.0, 1.5, 1.0])
    true_y = np.array([2.0, 2.5, 1.5, 2.0, 1.0])
    true_gamma = 0.15
    sources_list: list[int] = []
    targets_list: list[int] = []
    costs_list: list[float] = []
    weights_list: list[int] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(np.linalg.norm(positions[i] - positions[j]))
            rate = true_x[i] * true_y[j] * np.exp(-true_gamma * d)
            w = max(1, round(rate))
            sources_list.append(i)
            targets_list.append(j)
            costs_list.append(d)
            weights_list.append(w)
    edges = EdgeTable(
        source=np.array(sources_list, dtype=np.uint64),
        target=np.array(targets_list, dtype=np.uint64),
        weight=np.array(weights_list, dtype=np.uint64),
    )
    cost_src = np.array(sources_list)
    cost_tgt = np.array(targets_list)
    cost_val = np.array(costs_list)
    strengths = directed_strengths(edges)
    target_cost = _observed_cost(edges, cost_src, cost_tgt, cost_val)

    repetitions = 300
    fit = fit_strength_cost_me(
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
    assert fit.converged, "fit_strength_cost_me did not converge"

    sampled_strengths: list[np.ndarray] = []
    sampled_costs: list[float] = []
    for seed in range(repetitions):
        sample = sample_strength_cost_me(fit, cost_src, cost_tgt, cost_val, seed=seed)
        sampled_strengths.append(directed_strengths(sample).out.astype(float))
        sampled_costs.append(_observed_cost(sample, cost_src, cost_tgt, cost_val))

    mean_s, std_s = _mean_std(sampled_strengths)
    _assert_mean_close(mean_s, std_s, strengths.out.astype(float), repetitions)

    mean_cost = float(np.mean(sampled_costs))
    std_cost = float(np.std(sampled_costs, ddof=1))
    se_cost = std_cost / np.sqrt(repetitions)
    assert abs(mean_cost - target_cost) < 6.0 * se_cost + 0.08 * target_cost + 1.0
