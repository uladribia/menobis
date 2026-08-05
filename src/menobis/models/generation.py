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
    sources: list[int], targets: list[int], occ_nums: list[int]
) -> EdgeTable:
    return EdgeTable(
        source=np.asarray(sources, dtype=np.uint64),
        target=np.asarray(targets, dtype=np.uint64),
        occ_num=np.asarray(occ_nums, dtype=np.uint64),
    )


def _sample_native(function_name: str, *args: object) -> EdgeTable:
    """Call a native sampler and normalize its edge-list output."""
    sources, targets, occ_nums = getattr(_menobis, function_name)(*args)
    return _edge_table_from_lists(sources, targets, occ_nums)


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
    sources, targets, occ_nums = _menobis.sample_strength_cost_poisson_coordinates(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.gamma,
        x_coord.tolist(),
        y_coord.tolist(),
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_edges_events(
    node_count: int,
    q: float,
    occupation: float,
    family: str,
    *,
    layers: int = 1,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample the symmetric EDGES_EVENTS model from fitted multipliers.

    Every candidate pair draws from the same zero-inflated distribution
    with positive-support parameter `q` and global occupation probability.
    """
    sources, targets, occ_nums = _menobis.sample_edges_events(
        int(node_count),
        float(q),
        float(occupation),
        family,
        int(layers),
        bool(self_loops),
        int(seed),
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_stub_matching(
    strength_out: NDArray[np.integer],
    strength_in: NDArray[np.integer],
    *,
    seed: int = 0,
) -> EdgeTable:
    """Exact-strength stub-matching sampler for fixed-strength ME.

    Produces an unbiased uniform sample from the space of all integer-occupation
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
    sources, targets, occ_nums = _menobis.sample_strength_stub_matching(
        s_out.tolist(), s_in.tolist(), seed
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_me_fixed_et(
    node_count: int,
    *,
    self_loops: bool = True,
    residual_edges: int,
    residual_total: int,
    seed: int = 0,
) -> EdgeTable:
    """Exact ME microcanonical sampler with fixed (E,T) over all pairs.

    Draws an exact sample from the ME microcanonical distribution over
    the full admissible set of ``node_count`` nodes (all N² or N(N-1)
    candidate pairs depending on ``self_loops``).  No pair list is
    materialised: the Rust kernel maps linear indices to pairs on the fly.

    Args:
        node_count: Number of nodes.
        self_loops: Whether diagonal pairs are admissible.
        residual_edges: Number of occupied pairs E.
        residual_total: Total occupation T.
        seed: Random seed.

    Returns:
        EdgeTable with exactly ``residual_edges`` occupied pairs and
        total occupation ``residual_total``.
    """
    sources, targets, occ_nums = _menobis.sample_me_fixed_et(
        int(node_count),
        bool(self_loops),
        int(residual_edges),
        int(residual_total),
        int(seed),
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_me_fixed_et_explicit(
    admissible_sources: NDArray[np.uint64],
    admissible_targets: NDArray[np.uint64],
    residual_edges: int,
    residual_total: int,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Exact ME microcanonical sampler on an explicit admissible-pair set.

    Used when masks or fixed pairs reduce the admissible domain.  The
    caller is responsible for computing residual E and T by subtracting
    fixed-pair contributions, and for merging fixed pairs afterwards.

    Args:
        admissible_sources: Source nodes of admissible (free) pairs.
        admissible_targets: Target nodes of admissible (free) pairs.
        residual_edges: Number of occupied pairs E in the residual graph.
        residual_total: Total occupation T in the residual graph.
        seed: Random seed.

    Returns:
        EdgeTable with exactly ``residual_edges`` occupied pairs and
        total occupation ``residual_total``.
    """
    sources_list = np.asarray(admissible_sources, dtype=np.uint64).tolist()
    targets_list = np.asarray(admissible_targets, dtype=np.uint64).tolist()
    sources, targets, occ_nums = _menobis.sample_me_fixed_et_explicit(
        sources_list,
        targets_list,
        int(residual_edges),
        int(residual_total),
        int(seed),
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_b_fixed_et(
    node_count: int,
    *,
    self_loops: bool = True,
    layers: int = 1,
    residual_edges: int,
    residual_total: int,
    seed: int = 0,
) -> EdgeTable:
    """Exact B microcanonical sampler with fixed (E,T) and M layers."""
    sources, targets, occ_nums = _menobis.sample_b_fixed_et(
        int(node_count),
        bool(self_loops),
        int(layers),
        int(residual_edges),
        int(residual_total),
        int(seed),
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_b_fixed_et_explicit(
    admissible_sources: NDArray[np.uint64],
    admissible_targets: NDArray[np.uint64],
    layers: int = 1,
    residual_edges: int = 0,
    residual_total: int = 0,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Exact B microcanonical sampler on an explicit admissible-pair set."""
    sources_list = np.asarray(admissible_sources, dtype=np.uint64).tolist()
    targets_list = np.asarray(admissible_targets, dtype=np.uint64).tolist()
    sources, targets, occ_nums = _menobis.sample_b_fixed_et_explicit(
        sources_list,
        targets_list,
        int(layers),
        int(residual_edges),
        int(residual_total),
        int(seed),
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_w_fixed_et(
    node_count: int,
    *,
    self_loops: bool = True,
    layers: int = 1,
    residual_edges: int,
    residual_total: int,
    seed: int = 0,
) -> EdgeTable:
    """Exact W microcanonical sampler with fixed (E,T) and M layers."""
    sources, targets, occ_nums = _menobis.sample_w_fixed_et(
        int(node_count),
        bool(self_loops),
        int(layers),
        int(residual_edges),
        int(residual_total),
        int(seed),
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_w_fixed_et_explicit(
    admissible_sources: NDArray[np.uint64],
    admissible_targets: NDArray[np.uint64],
    layers: int = 1,
    residual_edges: int = 0,
    residual_total: int = 0,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Exact W microcanonical sampler on an explicit admissible-pair set."""
    sources_list = np.asarray(admissible_sources, dtype=np.uint64).tolist()
    targets_list = np.asarray(admissible_targets, dtype=np.uint64).tolist()
    sources, targets, occ_nums = _menobis.sample_w_fixed_et_explicit(
        sources_list,
        targets_list,
        int(layers),
        int(residual_edges),
        int(residual_total),
        int(seed),
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_custom_poisson(
    probabilities: ProbabilityTable,
    *,
    total_events: int,
    seed: int = 0,
) -> EdgeTable:
    """Grand-canonical custom p_ij sampling with ``E[t_ij] = T p_ij``."""
    sources, targets, occ_nums = _menobis.sample_custom_poisson(
        probabilities.source.tolist(),
        probabilities.target.tolist(),
        probabilities.probability.tolist(),
        total_events,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_custom_multinomial(
    probabilities: ProbabilityTable,
    *,
    total_events: int,
    seed: int = 0,
) -> EdgeTable:
    """Canonical custom p_ij multinomial sampling with fixed ``T``."""
    sources, targets, occ_nums = _menobis.sample_custom_multinomial(
        probabilities.source.tolist(),
        probabilities.target.tolist(),
        probabilities.probability.tolist(),
        total_events,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_edges_poisson(
    fit: StrengthEdgesFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample exact ME fixed-strength-and-edge-count ME model."""
    sources, targets, occ_nums = _menobis.sample_strength_edges_poisson(
        fit.x.tolist(), fit.y.tolist(), fit.lam, fit.self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_poisson(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent Poisson(x_i * y_j)."""
    sources, targets, occ_nums = _menobis.sample_strength_poisson(
        x.tolist(), y.tolist(), self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_degree_events_poisson(
    fit: DegreeEventsFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample degree-events ME from a fitted zero-truncated Poisson rate."""
    sources, targets, occ_nums = _menobis.sample_degree_events_poisson(
        fit.x.tolist(), fit.y.tolist(), fit.q, fit.self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_degree_poisson(
    fit: StrengthDegreeFit,
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample exact ME fixed-strength-degree ME model."""
    sources, targets, occ_nums = _menobis.sample_strength_degree_poisson(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_multinomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    total_events: int,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Multinomial sampling with node-factorized probabilities."""
    sources, targets, occ_nums = _menobis.sample_strength_multinomial(
        x.tolist(), y.tolist(), total_events, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_geometric(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent Geometric(1 - x_i*y_j)."""
    sources, targets, occ_nums = _menobis.sample_strength_geometric(
        x.tolist(), y.tolist(), self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_binomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    layers: int = 1,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent Binomial(M, x_i*y_j/(1+x_i*y_j))."""
    sources, targets, occ_nums = _menobis.sample_strength_binomial(
        x.tolist(), y.tolist(), layers, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_negative_binomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    layers: int = 1,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent NegativeBinomial(M, 1-x_i*y_j)."""
    sources, targets, occ_nums = _menobis.sample_strength_negative_binomial(
        x.tolist(), y.tolist(), layers, self_loops, seed
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_cost_binomial(
    fit: "StrengthCostFit",
    coord_x: NDArray[np.floating],
    coord_y: NDArray[np.floating],
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-cost binomial using Euclidean coordinate costs."""
    sources, targets, occ_nums = _menobis.sample_strength_cost_binomial_coordinates(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.gamma,
        np.asarray(coord_x, dtype=np.float64).tolist(),
        np.asarray(coord_y, dtype=np.float64).tolist(),
        layers,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_cost_geometric(
    fit: "StrengthCostFit",
    coord_x: NDArray[np.floating],
    coord_y: NDArray[np.floating],
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-cost geometric using Euclidean coordinate costs."""
    sources, targets, occ_nums = _menobis.sample_strength_cost_geometric_coordinates(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.gamma,
        np.asarray(coord_x, dtype=np.float64).tolist(),
        np.asarray(coord_y, dtype=np.float64).tolist(),
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_cost_negative_binomial(
    fit: "StrengthCostFit",
    coord_x: NDArray[np.floating],
    coord_y: NDArray[np.floating],
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-cost negative binomial using Euclidean coordinate costs."""
    sources, targets, occ_nums = (
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
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_edges_binomial(
    fit: "StrengthEdgesFit",
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-edges zero-inflated binomial."""
    sources, targets, occ_nums = _menobis.sample_strength_edges_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.lam,
        layers,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_degree_binomial(
    fit: "StrengthDegreeFit",
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-degree zero-inflated binomial."""
    sources, targets, occ_nums = _menobis.sample_strength_degree_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        layers,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_degree_events_binomial(
    fit: "DegreeEventsFit",
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample degree-events zero-inflated binomial from a fit result."""
    sources, targets, occ_nums = _menobis.sample_degree_events_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.q,
        fit.layers or 1,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_edges_geometric(
    fit: "StrengthEdgesFit",
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-edges zero-inflated geometric."""
    sources, targets, occ_nums = _menobis.sample_strength_edges_geometric(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.lam,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_edges_negative_binomial(
    fit: "StrengthEdgesFit",
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-edges zero-inflated negative binomial."""
    sources, targets, occ_nums = _menobis.sample_strength_edges_negative_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.lam,
        layers,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_degree_geometric(
    fit: "StrengthDegreeFit",
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-degree zero-inflated geometric."""
    sources, targets, occ_nums = _menobis.sample_strength_degree_geometric(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_strength_degree_negative_binomial(
    fit: "StrengthDegreeFit",
    *,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample strength-degree zero-inflated negative binomial."""
    sources, targets, occ_nums = _menobis.sample_strength_degree_negative_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.z.tolist(),
        fit.w.tolist(),
        layers,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_degree_events_geometric(
    fit: "DegreeEventsFit",
    *,
    seed: int = 0,
) -> EdgeTable:
    """Sample degree-events zero-inflated geometric."""
    sources, targets, occ_nums = _menobis.sample_degree_events_geometric(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.q,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_degree_events_negative_binomial(
    fit: "DegreeEventsFit",
    *,
    layers: int | None = None,
    seed: int = 0,
) -> EdgeTable:
    """Sample degree-events zero-inflated negative binomial."""
    m = layers if layers is not None else (fit.layers or 1)
    sources, targets, occ_nums = _menobis.sample_degree_events_negative_binomial(
        fit.x.tolist(),
        fit.y.tolist(),
        fit.q,
        m,
        fit.self_loops,
        seed,
    )
    return _edge_table_from_lists(sources, targets, occ_nums)


def _sample_degree_events_fixed_kt(
    *,
    family: str,
    degree_out: list[int],
    degree_in: list[int],
    total_events: int,
    layers: int = 1,
    seed: int = 0,
    self_loops: bool = False,
    burn_in_sweeps: int = 50,
    sweeps_per_sample: int = 10,
) -> EdgeTable:
    """Sample microcanonical DEGREE_EVENTS via MCMC support + occupation allocator."""
    import menobis._menobis as _menobis

    sources, targets, occ_nums = _menobis.sample_degree_events_fixed_kt(  # type: ignore
        family,
        degree_out,
        degree_in,
        total_events,
        layers,
        burn_in_sweeps,
        sweeps_per_sample,
        seed,
        self_loops,
    )
    return EdgeTable(
        source=np.asarray(sources, dtype=np.uint64),
        target=np.asarray(targets, dtype=np.uint64),
        occ_num=np.asarray(occ_nums, dtype=np.uint64),
    )
