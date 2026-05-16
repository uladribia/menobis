"""Seeded graph generation samplers backed by Rust kernels."""

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.data.frames import EdgeTable, ProbabilityTable
from odme.models.fitting import (
    FitResult,
    GravityMeFit,
    StrengthDegreeMeFit,
    StrengthEdgesMeFit,
)


def _edge_table_from_lists(
    sources: list[int], targets: list[int], weights: list[int]
) -> EdgeTable:
    return EdgeTable(
        source=np.asarray(sources, dtype=np.uint64),
        target=np.asarray(targets, dtype=np.uint64),
        weight=np.asarray(weights, dtype=np.uint64),
    )


def sample_gravity_me(
    fit: GravityMeFit,
    cost_sources: NDArray[np.integer],
    cost_targets: NDArray[np.integer],
    cost_values: NDArray[np.floating],
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample from the gravity ME model: E[t_ij] = x_i y_j exp(-gamma d_ij)."""
    n = len(fit.x)
    gamma = fit.gamma
    c_src = np.asarray(cost_sources, dtype=np.int64)
    c_tgt = np.asarray(cost_targets, dtype=np.int64)
    c_val = np.asarray(cost_values, dtype=np.float64)

    # Build sparse rate map
    cost_map: dict[tuple[int, int], float] = {}
    for s_val, t_val, d_val in zip(c_src, c_tgt, c_val, strict=True):
        cost_map[(int(s_val), int(t_val))] = float(d_val)

    # Compute rates and delegate to Poisson sampler via _odme
    sources_list: list[float] = []
    targets_list: list[float] = []
    rates: list[float] = []
    for i in range(n):
        for j in range(n):
            if not fit.self_loops and i == j:
                continue
            d = cost_map.get((i, j), 0.0)
            rate = fit.x[i] * fit.y[j] * np.exp(-gamma * d)
            if rate > 0:
                sources_list.append(float(i))
                targets_list.append(float(j))
                rates.append(rate)
    prob_table = ProbabilityTable(
        source=np.asarray(sources_list, dtype=np.uint64),
        target=np.asarray(targets_list, dtype=np.uint64),
        probability=np.asarray(rates, dtype=np.float64),
    )
    total_events = round(sum(rates))
    if total_events <= 0:
        total_events = 1
    # Use Poisson sampling with rates as unnormalized probabilities
    result_sources, result_targets, result_weights = (
        _odme.sample_custom_pij_events_poisson(
            prob_table.source.tolist(),
            prob_table.target.tolist(),
            prob_table.probability.tolist(),
            total_events,
            seed,
        )
    )
    return _edge_table_from_lists(result_sources, result_targets, result_weights)


def sample_custom_pij_events_poisson(
    probabilities: ProbabilityTable,
    *,
    total_events: int,
    seed: int = 0,
) -> EdgeTable:
    """Grand-canonical custom p_ij sampling with ``E[t_ij] = T p_ij``."""
    sources, targets, weights = _odme.sample_custom_pij_events_poisson(
        probabilities.source.tolist(),
        probabilities.target.tolist(),
        probabilities.probability.tolist(),
        total_events,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_custom_pij_events_multinomial(
    probabilities: ProbabilityTable,
    *,
    total_events: int,
    seed: int = 0,
) -> EdgeTable:
    """Canonical custom p_ij multinomial sampling with fixed ``T``."""
    sources, targets, weights = _odme.sample_custom_pij_events_multinomial(
        probabilities.source.tolist(),
        probabilities.target.tolist(),
        probabilities.probability.tolist(),
        total_events,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_poisson_multinomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Poisson-total multinomial sampling for fixed-strength ME."""
    sources, targets, weights = _odme.sample_poisson_multinomial(
        x.tolist(), y.tolist(), self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_edges_me(
    fit: StrengthEdgesMeFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample exact ME fixed-strength-and-edge-count ME model."""
    sources, targets, weights = _odme.sample_strength_edges_me(
        fit.x.tolist(), fit.y.tolist(), fit.lam, fit.self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_poisson(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent Poisson(x_i * y_j)."""
    sources, targets, weights = _odme.sample_poisson(
        x.tolist(), y.tolist(), self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_fixed_degree_events_me(
    fit: FitResult,
    *,
    total_events: int,
    seed: int = 0,
    self_loops: bool = True,
) -> EdgeTable:
    """Sample original fixed-degree ME weighted ME model."""
    sources, targets, weights = _odme.sample_fixed_degree_events_me(
        fit.x.tolist(), fit.y.tolist(), total_events, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_degree_me(
    fit: StrengthDegreeMeFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample exact ME fixed-strength-degree ME model."""
    sources, targets, weights = _odme.sample_strength_degree_me(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_multinomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    total_events: int,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Multinomial sampling with node-factorized probabilities."""
    sources, targets, weights = _odme.sample_multinomial(
        x.tolist(), y.tolist(), total_events, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


__all__ = [
    "sample_custom_pij_events_multinomial",
    "sample_custom_pij_events_poisson",
    "sample_fixed_degree_events_me",
    "sample_gravity_me",
    "sample_multinomial",
    "sample_poisson",
    "sample_poisson_multinomial",
    "sample_strength_degree_me",
    "sample_strength_edges_me",
]
