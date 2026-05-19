"""Clustering coefficients backed by Rust kernels."""

import numpy as np

import odme._odme as _odme
from odme.analysis.types import ClusteringResult
from odme.data.frames import EdgeTable


def clustering_coefficient(edges: EdgeTable) -> ClusteringResult:
    """Compute binary clustering coefficient per node (Rust kernel)."""
    nc = _node_count(edges)
    values = _odme.clustering_coefficients(
        nc, edges.source.tolist(), edges.target.tolist(), edges.weight.tolist()
    )
    return ClusteringResult(
        node=np.arange(nc, dtype=np.uint64),
        values=np.asarray(values),
    )


def weighted_clustering_coefficient(edges: EdgeTable) -> ClusteringResult:
    """Compute weighted clustering coefficient per node (Rust kernel)."""
    nc = _node_count(edges)
    values = _odme.weighted_clustering_coefficients(
        nc, edges.source.tolist(), edges.target.tolist(), edges.weight.tolist()
    )
    return ClusteringResult(
        node=np.arange(nc, dtype=np.uint64),
        values=np.asarray(values),
    )


def _node_count(edges: EdgeTable) -> int:
    if len(edges) == 0:
        return 0
    return int(max(edges.source.max(), edges.target.max())) + 1


__all__ = [
    "ClusteringResult",
    "clustering_coefficient",
    "weighted_clustering_coefficient",
]
