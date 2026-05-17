"""Seeded graph generation samplers backed by Rust kernels."""

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.data.frames import EdgeTable, ProbabilityTable
from odme.models.fitting import (
    FitResult,
    StrengthCostFit,
    StrengthDegreeFit,
    StrengthEdgesFit,
)


def _edge_table_from_lists(
    sources: list[int], targets: list[int], weights: list[int]
) -> EdgeTable:
    return EdgeTable(
        source=np.asarray(sources, dtype=np.uint64),
        target=np.asarray(targets, dtype=np.uint64),
        weight=np.asarray(weights, dtype=np.uint64),
    )


def sample_strength_cost_poisson(
    fit: StrengthCostFit,
    cost_sources: NDArray[np.integer],
    cost_targets: NDArray[np.integer],
    cost_values: NDArray[np.floating],
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample from the strength-cost ME model: E[t_ij] = x_i y_j exp(-gamma d_ij)."""
    c_src = np.asarray(cost_sources, dtype=np.int64)
    c_tgt = np.asarray(cost_targets, dtype=np.int64)
    c_val = np.asarray(cost_values, dtype=np.float64)
    sources, targets, weights = _odme.sample_strength_cost_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.gamma,
        c_src.tolist(),
        c_tgt.tolist(),
        c_val.tolist(),
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_microcanonical(
    strength_out: NDArray[np.integer],
    strength_in: NDArray[np.integer],
    *,
    seed: int = 0,
) -> EdgeTable:
    """Microcanonical stub-matching sampler for fixed-strength ME.

    Produces an unbiased uniform sample from the space of all integer-weight
    directed graphs with the exact given strength sequence. Self-loops are
    always allowed because uniform sampling without self-loops requires
    more sophisticated algorithms to avoid bias.

    Args:
        strength_out: Exact outgoing strength per node (positive integers).
        strength_in: Exact incoming strength per node (positive integers).
        seed: Random seed.

    Returns:
        EdgeTable with exact strength preservation.
    """
    s_out = np.asarray(strength_out, dtype=np.uint64)
    s_in = np.asarray(strength_in, dtype=np.uint64)
    sources, targets, weights = _odme.sample_strength_microcanonical(
        s_out.tolist(), s_in.tolist(), seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_custom_poisson(
    probabilities: ProbabilityTable,
    *,
    total_events: int,
    seed: int = 0,
) -> EdgeTable:
    """Grand-canonical custom p_ij sampling with ``E[t_ij] = T p_ij``."""
    sources, targets, weights = _odme.sample_custom_poisson(
        probabilities.source.tolist(),
        probabilities.target.tolist(),
        probabilities.probability.tolist(),
        total_events,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_custom_multinomial(
    probabilities: ProbabilityTable,
    *,
    total_events: int,
    seed: int = 0,
) -> EdgeTable:
    """Canonical custom p_ij multinomial sampling with fixed ``T``."""
    sources, targets, weights = _odme.sample_custom_multinomial(
        probabilities.source.tolist(),
        probabilities.target.tolist(),
        probabilities.probability.tolist(),
        total_events,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_poisson_multinomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Poisson-total multinomial sampling for fixed-strength ME."""
    sources, targets, weights = _odme.sample_strength_poisson_multinomial(
        x.tolist(), y.tolist(), self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_edges_poisson(
    fit: StrengthEdgesFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample exact ME fixed-strength-and-edge-count ME model."""
    sources, targets, weights = _odme.sample_strength_edges_poisson(
        fit.x.tolist(), fit.y.tolist(), fit.lam, fit.self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_poisson(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent Poisson(x_i * y_j)."""
    sources, targets, weights = _odme.sample_strength_poisson(
        x.tolist(), y.tolist(), self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_degree_events_poisson(
    fit: FitResult,
    *,
    total_events: int,
    seed: int = 0,
    self_loops: bool = True,
) -> EdgeTable:
    """Sample original fixed-degree ME weighted ME model."""
    sources, targets, weights = _odme.sample_degree_events_poisson(
        fit.x.tolist(), fit.y.tolist(), total_events, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_degree_poisson(
    fit: StrengthDegreeFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample exact ME fixed-strength-degree ME model."""
    sources, targets, weights = _odme.sample_strength_degree_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_multinomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    total_events: int,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Multinomial sampling with node-factorized probabilities."""
    sources, targets, weights = _odme.sample_strength_multinomial(
        x.tolist(), y.tolist(), total_events, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_geometric(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent Geometric(1 - x_i*y_j)."""
    sources, targets, weights = _odme.sample_strength_geometric(
        x.tolist(), y.tolist(), self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_binomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    layers: int = 1,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent Binomial(M, x_i*y_j/(1+x_i*y_j))."""
    sources, targets, weights = _odme.sample_strength_binomial(
        x.tolist(), y.tolist(), layers, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_neg_binomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    layers: int = 1,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent NegBinomial(M, 1-x_i*y_j)."""
    sources, targets, weights = _odme.sample_strength_neg_binomial(
        x.tolist(), y.tolist(), layers, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


__all__ = [
    "sample_custom_multinomial",
    "sample_custom_poisson",
    "sample_degree_events_poisson",
    "sample_strength_binomial",
    "sample_strength_cost_poisson",
    "sample_strength_degree_poisson",
    "sample_strength_edges_poisson",
    "sample_strength_geometric",
    "sample_strength_microcanonical",
    "sample_strength_multinomial",
    "sample_strength_neg_binomial",
    "sample_strength_poisson",
    "sample_strength_poisson_multinomial",
]
