"""Clustering coefficients backed by Rust kernels."""

import numpy as np

import menobis._menobis as _menobis
from menobis.analysis.types import ClusteringResult
from menobis.data.frames import EdgeTable


def clustering_coefficient(edges: EdgeTable) -> ClusteringResult:
    """Compute binary clustering coefficient per node (Rust kernel)."""
    nc = _node_count(edges)
    values = _menobis.clustering_coefficients(
        nc, edges.source.tolist(), edges.target.tolist(), edges.occ_num.tolist()
    )
    return ClusteringResult(
        node=np.arange(nc, dtype=np.uint64),
        values=np.asarray(values),
    )


def occupation_clustering_coefficient(edges: EdgeTable) -> ClusteringResult:
    """Compute occupation-based clustering coefficient per node (Rust kernel)."""
    nc = _node_count(edges)
    values = _menobis.occupation_clustering_coefficients(
        nc, edges.source.tolist(), edges.target.tolist(), edges.occ_num.tolist()
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
    "occupation_clustering_coefficient",
]
