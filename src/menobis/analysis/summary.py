"""Analysis helpers backed by Rust kernels.

Python is a thin wrapper: extract arrays from EdgeTable, call Rust, return results.
"""

import numpy as np

import menobis._menobis as _menobis
from menobis.analysis.types import DirectedSequences
from menobis.data.frames import EdgeTable


def directed_strengths(edges: EdgeTable) -> DirectedSequences:
    """Compute directed in/out strength sequences."""
    nc = _node_count(edges)
    s_out, s_in = _menobis.directed_strengths(
        nc, edges.source.tolist(), edges.target.tolist(), edges.weight.tolist()
    )
    return DirectedSequences(
        node=np.arange(nc, dtype=np.uint64),
        out=np.asarray(s_out, dtype=np.uint64),
        incoming=np.asarray(s_in, dtype=np.uint64),
    )


def directed_degrees(edges: EdgeTable) -> DirectedSequences:
    """Compute directed in/out binary degrees."""
    nc = _node_count(edges)
    k_out, k_in = _menobis.directed_degrees(
        nc, edges.source.tolist(), edges.target.tolist(), edges.weight.tolist()
    )
    return DirectedSequences(
        node=np.arange(nc, dtype=np.uint64),
        out=np.asarray(k_out, dtype=np.uint64),
        incoming=np.asarray(k_in, dtype=np.uint64),
    )


def _node_count(edges: EdgeTable) -> int:
    if len(edges) == 0:
        return 0
    return int(max(edges.source.max(), edges.target.max())) + 1


__all__ = ["DirectedSequences", "directed_degrees", "directed_strengths"]
