"""Core data types for MENoBiS edge tables.

MENoBiS uses numpy arrays as the canonical in-memory representation for
edge data, avoiding heavy dataframe dependencies. Results are returned
as simple dataclasses with numpy arrays.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ProbabilityTable:
    """Sparse custom p_ij probability table."""

    source: NDArray[np.uint64]
    target: NDArray[np.uint64]
    probability: NDArray[np.float64]

    @property
    def num_edges(self) -> int:
        """Number of candidate edges."""
        return len(self.source)


@dataclass(frozen=True)
class EdgeTable:
    """Canonical MENoBiS weighted edge table stored as numpy arrays."""

    source: NDArray[np.uint64]
    target: NDArray[np.uint64]
    weight: NDArray[np.uint64]

    @property
    def num_edges(self) -> int:
        """Number of edges with positive weight."""
        return len(self.source)

    @property
    def total_events(self) -> int:
        """Total weight sum."""
        return int(self.weight.sum())

    def __len__(self) -> int:
        """Number of edges."""
        return len(self.source)


def normalize_probabilities(
    source: NDArray[np.integer],
    target: NDArray[np.integer],
    probability: NDArray[np.floating],
) -> ProbabilityTable:
    """Validate and normalize sparse custom probabilities."""
    if len(source) != len(target) or len(source) != len(probability):
        msg = "source, target, and probability arrays must have the same length"
        raise ValueError(msg)
    p = np.asarray(probability, dtype=np.float64)
    if np.any(p < 0.0) or np.any(p > 1.0):
        msg = "probabilities must be in [0, 1]"
        raise ValueError(msg)
    mask = p > 0.0
    pairs = set[tuple[int, int]]()
    for s_val, t_val in zip(
        np.asarray(source)[mask], np.asarray(target)[mask], strict=True
    ):
        pair = (int(s_val), int(t_val))
        if pair in pairs:
            msg = "duplicate probability entries are not allowed"
            raise ValueError(msg)
        pairs.add(pair)
    return ProbabilityTable(
        source=np.asarray(source, dtype=np.uint64)[mask],
        target=np.asarray(target, dtype=np.uint64)[mask],
        probability=p[mask],
    )


def normalize_edges(
    source: NDArray[np.integer],
    target: NDArray[np.integer],
    weight: NDArray[np.integer],
) -> EdgeTable:
    """Validate and normalize raw edge arrays.

    Drops zero-weight edges, rejects negative or fractional weights.

    Args:
        source: Source node ids.
        target: Target node ids.
        weight: Edge weights.

    Returns:
        Normalized EdgeTable.

    Raises:
        ValueError: If arrays have different lengths or weights are invalid.
    """
    if len(source) != len(target) or len(source) != len(weight):
        msg = "source, target, and weight arrays must have the same length"
        raise ValueError(msg)

    raw_weight = np.asarray(weight)
    if not np.issubdtype(raw_weight.dtype, np.integer) and not np.all(
        np.equal(raw_weight, np.floor(raw_weight))
    ):
        msg = "edge weights must be non-negative integers"
        raise ValueError(msg)

    w = raw_weight.astype(np.int64)
    if np.any(w < 0):
        msg = "edge weights must be non-negative integers"
        raise ValueError(msg)

    mask = w > 0
    return EdgeTable(
        source=np.asarray(source, dtype=np.uint64)[mask],
        target=np.asarray(target, dtype=np.uint64)[mask],
        weight=np.asarray(w, dtype=np.uint64)[mask],
    )


__all__ = [
    "EdgeTable",
    "ProbabilityTable",
    "normalize_edges",
    "normalize_probabilities",
]
