"""Seeded graph generation samplers backed by Rust kernels."""

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.data.frames import EdgeTable, ProbabilityTable
from odme.models.fitting import FitResult, StrengthDegreeZipFit, StrengthEdgesZipFit


def _edge_table_from_lists(
    sources: list[int], targets: list[int], weights: list[int]
) -> EdgeTable:
    return EdgeTable(
        source=np.asarray(sources, dtype=np.uint64),
        target=np.asarray(targets, dtype=np.uint64),
        weight=np.asarray(weights, dtype=np.uint64),
    )


def sample_custom_pij_poisson(
    probabilities: ProbabilityTable,
    *,
    total_events: int,
    seed: int = 0,
) -> EdgeTable:
    """Grand-canonical custom p_ij sampling with ``E[t_ij] = T p_ij``."""
    sources, targets, weights = _odme.sample_custom_pij_poisson(
        probabilities.source.tolist(),
        probabilities.target.tolist(),
        probabilities.probability.tolist(),
        total_events,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_custom_pij_multinomial(
    probabilities: ProbabilityTable,
    *,
    total_events: int,
    seed: int = 0,
) -> EdgeTable:
    """Canonical custom p_ij multinomial sampling with fixed ``T``."""
    sources, targets, weights = _odme.sample_custom_pij_multinomial(
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


def sample_strength_edges_zip(
    fit: StrengthEdgesZipFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample exact ME fixed-strength-and-edge-count ZIP model."""
    sources, targets, weights = _odme.sample_strength_edges_zip(
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


def sample_fixed_degree_zip(
    fit: FitResult,
    *,
    total_events: int,
    seed: int = 0,
    self_loops: bool = True,
) -> EdgeTable:
    """Sample original fixed-degree ME weighted ZIP model."""
    sources, targets, weights = _odme.sample_fixed_degree_zip(
        fit.x.tolist(), fit.y.tolist(), total_events, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def sample_strength_degree_zip(
    fit: StrengthDegreeZipFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample exact ME fixed-strength-degree ZIP model."""
    sources, targets, weights = _odme.sample_strength_degree_zip(
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
    "sample_custom_pij_multinomial",
    "sample_custom_pij_poisson",
    "sample_fixed_degree_zip",
    "sample_multinomial",
    "sample_poisson",
    "sample_poisson_multinomial",
    "sample_strength_degree_zip",
    "sample_strength_edges_zip",
]
