"""Seeded graph generation samplers backed by Rust kernels."""

import numpy as np
from numpy.typing import NDArray

import menobis._menobis as _menobis
from menobis.data.frames import EdgeTable, ProbabilityTable
from menobis.models.types import (
    DegreeEventsFit,
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


def _sample_native(function_name: str, *args: object) -> EdgeTable:
    """Call a native sampler and normalize its edge-list output."""
    sources, targets, weights = getattr(_menobis, function_name)(*args)
    return _edge_table_from_lists(sources, targets, weights)


def _as_float_list(values: NDArray[np.floating]) -> list[float]:
    return np.asarray(values, dtype=np.float64).tolist()


def _as_int_list(values: NDArray[np.integer]) -> list[int]:
    return np.asarray(values, dtype=np.int64).tolist()


def _sample_strength_cost_poisson(
    fit: StrengthCostFit,
    coord_x: NDArray[np.floating],
    coord_y: NDArray[np.floating],
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample from the strength-cost ME model using Euclidean coordinate costs."""
    x_coord = np.asarray(coord_x, dtype=np.float64)
    y_coord = np.asarray(coord_y, dtype=np.float64)
    sources, targets, weights = _menobis.sample_strength_cost_poisson_coordinates(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.gamma,
        x_coord.tolist(),
        y_coord.tolist(),
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_stub_matching(
    strength_out: NDArray[np.integer],
    strength_in: NDArray[np.integer],
    *,
    seed: int = 0,
) -> EdgeTable:
    """Exact-strength stub-matching sampler for fixed-strength ME.

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
    sources, targets, weights = _menobis.sample_strength_stub_matching(
        s_out.tolist(), s_in.tolist(), seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_custom_poisson(
    probabilities: ProbabilityTable,
    *,
    total_events: int,
    seed: int = 0,
) -> EdgeTable:
    """Grand-canonical custom p_ij sampling with ``E[t_ij] = T p_ij``."""
    sources, targets, weights = _menobis.sample_custom_poisson(
        probabilities.source.tolist(),
        probabilities.target.tolist(),
        probabilities.probability.tolist(),
        total_events,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_custom_multinomial(
    probabilities: ProbabilityTable,
    *,
    total_events: int,
    seed: int = 0,
) -> EdgeTable:
    """Canonical custom p_ij multinomial sampling with fixed ``T``."""
    sources, targets, weights = _menobis.sample_custom_multinomial(
        probabilities.source.tolist(),
        probabilities.target.tolist(),
        probabilities.probability.tolist(),
        total_events,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_edges_poisson(
    fit: StrengthEdgesFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample exact ME fixed-strength-and-edge-count ME model."""
    sources, targets, weights = _menobis.sample_strength_edges_poisson(
        fit.x.tolist(), fit.y.tolist(), fit.lam, fit.self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_poisson(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent Poisson(x_i * y_j)."""
    sources, targets, weights = _menobis.sample_strength_poisson(
        x.tolist(), y.tolist(), self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_degree_events_poisson(
    fit: DegreeEventsFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample degree-events ME from a fitted zero-truncated Poisson rate."""
    sources, targets, weights = _menobis.sample_degree_events_poisson(
        fit.x.tolist(), fit.y.tolist(), fit.q, fit.self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_degree_poisson(
    fit: StrengthDegreeFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample exact ME fixed-strength-degree ME model."""
    sources, targets, weights = _menobis.sample_strength_degree_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_multinomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    total_events: int,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Multinomial sampling with node-factorized probabilities."""
    sources, targets, weights = _menobis.sample_strength_multinomial(
        x.tolist(), y.tolist(), total_events, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_geometric(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent Geometric(1 - x_i*y_j)."""
    sources, targets, weights = _menobis.sample_strength_geometric(
        x.tolist(), y.tolist(), self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_binomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    layers: int = 1,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent Binomial(M, x_i*y_j/(1+x_i*y_j))."""
    sources, targets, weights = _menobis.sample_strength_binomial(
        x.tolist(), y.tolist(), layers, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_negative_binomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    layers: int = 1,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent NegativeBinomial(M, 1-x_i*y_j)."""
    sources, targets, weights = _menobis.sample_strength_negative_binomial(
        x.tolist(), y.tolist(), layers, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_cost_binomial(
    fit: "StrengthCostFit",
    coord_x: NDArray[np.floating],
    coord_y: NDArray[np.floating],
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-cost binomial using Euclidean coordinate costs."""
    sources, targets, weights = _menobis.sample_strength_cost_binomial_coordinates(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.gamma,
        np.asarray(coord_x, dtype=np.float64).tolist(),
        np.asarray(coord_y, dtype=np.float64).tolist(),
        layers,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_cost_geometric(
    fit: "StrengthCostFit",
    coord_x: NDArray[np.floating],
    coord_y: NDArray[np.floating],
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-cost geometric using Euclidean coordinate costs."""
    sources, targets, weights = _menobis.sample_strength_cost_geometric_coordinates(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.gamma,
        np.asarray(coord_x, dtype=np.float64).tolist(),
        np.asarray(coord_y, dtype=np.float64).tolist(),
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_cost_negative_binomial(
    fit: "StrengthCostFit",
    coord_x: NDArray[np.floating],
    coord_y: NDArray[np.floating],
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-cost negative binomial using Euclidean coordinate costs."""
    sources, targets, weights = (
        _menobis.sample_strength_cost_negative_binomial_coordinates(
            fit.x.tolist(),
            fit.y.tolist(),
            fit.gamma,
            np.asarray(coord_x, dtype=np.float64).tolist(),
            np.asarray(coord_y, dtype=np.float64).tolist(),
            layers,
            fit.self_loops,
            seed,
        )
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_edges_binomial(
    fit: "StrengthEdgesFit",
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-edges zero-inflated binomial."""
    sources, targets, weights = _menobis.sample_strength_edges_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.lam,
        layers,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_degree_binomial(
    fit: "StrengthDegreeFit",
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-degree zero-inflated binomial."""
    sources, targets, weights = _menobis.sample_strength_degree_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        layers,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_degree_events_binomial(
    fit: "DegreeEventsFit",
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample degree-events zero-inflated binomial from a fit result."""
    sources, targets, weights = _menobis.sample_degree_events_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.q,
        fit.layers or 1,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_edges_geometric(
    fit: "StrengthEdgesFit",
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-edges zero-inflated geometric."""
    sources, targets, weights = _menobis.sample_strength_edges_geometric(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.lam,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_edges_negative_binomial(
    fit: "StrengthEdgesFit",
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-edges zero-inflated negative binomial."""
    sources, targets, weights = _menobis.sample_strength_edges_negative_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.lam,
        layers,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_degree_geometric(
    fit: "StrengthDegreeFit",
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-degree zero-inflated geometric."""
    sources, targets, weights = _menobis.sample_strength_degree_geometric(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_strength_degree_negative_binomial(
    fit: "StrengthDegreeFit",
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-degree zero-inflated negative binomial."""
    sources, targets, weights = _menobis.sample_strength_degree_negative_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        layers,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_degree_events_geometric(
    fit: "DegreeEventsFit",
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample degree-events zero-inflated geometric."""
    sources, targets, weights = _menobis.sample_degree_events_geometric(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.q,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)


def _sample_degree_events_negative_binomial(
    fit: "DegreeEventsFit",
    *,
    layers: int | None = None,
    seed: int = 0,
) -> EdgeTable:
    """Sample degree-events zero-inflated negative binomial."""
    m = layers if layers is not None else (fit.layers or 1)
    sources, targets, weights = _menobis.sample_degree_events_negative_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.q,
        m,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, weights)
