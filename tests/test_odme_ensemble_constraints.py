"""End-to-end ensemble validation on a non-trivial synthetic network."""

import numpy as np

from odme.analysis import directed_degrees, directed_strengths
from odme.data.frames import EdgeTable
from odme.models import (
    fit_fixed_degree_binary,
    fit_fixed_strength_me,
    fit_strength_degree_me,
    fit_strength_edges_me,
    sample_fixed_degree_events_me,
    sample_multinomial,
    sample_poisson,
    sample_poisson_multinomial,
    sample_strength_degree_me,
    sample_strength_edges_me,
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
