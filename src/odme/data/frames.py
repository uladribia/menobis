"""Core data types for ODME edge tables.

ODME uses numpy arrays as the canonical in-memory representation for
edge data, avoiding heavy dataframe dependencies. Results are returned
as simple dataclasses with numpy arrays.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EdgeTable:
    """Canonical ODME weighted edge table stored as numpy arrays."""

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


def normalize_edges(
    source: NDArray[np.integer],
    target: NDArray[np.integer],
    weight: NDArray[np.integer],
) -> EdgeTable:
    """Validate and normalize raw edge arrays.

    Drops zero-weight edges, rejects negative weights.

    Args:
        source: Source node ids.
        target: Target node ids.
        weight: Edge weights.

    Returns:
        Normalized EdgeTable.

    Raises:
        ValueError: If arrays have different lengths or weights are negative.
    """
    if len(source) != len(target) or len(source) != len(weight):
        msg = "source, target, and weight arrays must have the same length"
        raise ValueError(msg)

    w = np.asarray(weight, dtype=np.int64)
    if np.any(w < 0):
        msg = "edge weights must be non-negative integers"
        raise ValueError(msg)

    mask = w > 0
    return EdgeTable(
        source=np.asarray(source, dtype=np.uint64)[mask],
        target=np.asarray(target, dtype=np.uint64)[mask],
        weight=np.asarray(w, dtype=np.uint64)[mask],
    )


__all__ = ["EdgeTable", "normalize_edges"]
