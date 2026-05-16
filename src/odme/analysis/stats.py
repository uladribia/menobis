"""Statistics backed by Rust single-pass kernel."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.data.frames import EdgeTable


@dataclass(frozen=True)
class WeightDistribution:
    """Weight distribution P(w)."""

    weight: NDArray[np.uint64]
    count: NDArray[np.uint64]


@dataclass(frozen=True)
class NodeStats:
    """All per-node statistics from a single Rust pass."""

    node: NDArray[np.uint64]
    strength_out: NDArray[np.uint64]
    strength_in: NDArray[np.uint64]
    degree_out: NDArray[np.uint64]
    degree_in: NDArray[np.uint64]
    y2_out: NDArray[np.float64]
    y2_in: NDArray[np.float64]
    s_nn_out: NDArray[np.float64]
    s_nn_in: NDArray[np.float64]
    k_nn_out: NDArray[np.float64]
    k_nn_in: NDArray[np.float64]


def weight_distribution(edges: EdgeTable) -> WeightDistribution:
    """Compute the weight distribution P(w)."""
    w, c = _odme.weight_distribution(
        edges.source.tolist(), edges.target.tolist(), edges.weight.tolist()
    )
    return WeightDistribution(
        weight=np.asarray(w, dtype=np.uint64),
        count=np.asarray(c, dtype=np.uint64),
    )


def compute_all_stats(edges: EdgeTable) -> NodeStats:
    """Compute all per-node statistics in a single Rust pass."""
    nc = _node_count(edges)
    (s_out, s_in, k_out, k_in, y2_o, y2_i, snn_o, snn_i, knn_o, knn_i) = (
        _odme.compute_all_node_stats(
            nc,
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        )
    )
    return NodeStats(
        node=np.arange(nc, dtype=np.uint64),
        strength_out=np.asarray(s_out, dtype=np.uint64),
        strength_in=np.asarray(s_in, dtype=np.uint64),
        degree_out=np.asarray(k_out, dtype=np.uint64),
        degree_in=np.asarray(k_in, dtype=np.uint64),
        y2_out=np.asarray(y2_o),
        y2_in=np.asarray(y2_i),
        s_nn_out=np.asarray(snn_o),
        s_nn_in=np.asarray(snn_i),
        k_nn_out=np.asarray(knn_o),
        k_nn_in=np.asarray(knn_i),
    )


def _node_count(edges: EdgeTable) -> int:
    if len(edges) == 0:
        return 0
    return int(max(edges.source.max(), edges.target.max())) + 1


__all__ = [
    "NodeStats",
    "WeightDistribution",
    "compute_all_stats",
    "weight_distribution",
]
