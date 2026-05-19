"""Result types for ODME analysis operations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DirectedSequences:
    """Directed per-node sequences (strengths or degrees)."""

    node: NDArray[np.uint64]
    out: NDArray[np.uint64]
    incoming: NDArray[np.uint64]


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


@dataclass(frozen=True)
class ClusteringResult:
    """Per-node clustering coefficient."""

    node: NDArray[np.uint64]
    values: NDArray[np.float64]


__all__ = [
    "ClusteringResult",
    "DirectedSequences",
    "NodeStats",
    "WeightDistribution",
]
