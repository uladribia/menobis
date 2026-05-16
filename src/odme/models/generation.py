"""Seeded graph generation samplers backed by Rust kernels."""

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.data.frames import EdgeTable


def sample_poisson(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Sample from independent Poisson(x_i * y_j).

    Args:
        x: Outgoing Lagrange multipliers, length N.
        y: Incoming Lagrange multipliers, length N.
        self_loops: Whether to allow i==j edges.
        seed: Random seed.

    Returns:
        EdgeTable with sampled edges.
    """
    sources, targets, weights = _odme.sample_poisson(
        x.tolist(), y.tolist(), self_loops, seed
    )
    return EdgeTable(
        source=np.asarray(sources, dtype=np.uint64),
        target=np.asarray(targets, dtype=np.uint64),
        weight=np.asarray(weights, dtype=np.uint64),
    )


def sample_multinomial(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    total_events: int,
    self_loops: bool = True,
    seed: int = 0,
) -> EdgeTable:
    """Multinomial sampling with node-factorized probabilities.

    Args:
        x: Outgoing Lagrange multipliers, length N.
        y: Incoming Lagrange multipliers, length N.
        total_events: Total number of events T.
        self_loops: Whether to allow i==j edges.
        seed: Random seed.

    Returns:
        EdgeTable with sampled edges.
    """
    sources, targets, weights = _odme.sample_multinomial(
        x.tolist(), y.tolist(), total_events, self_loops, seed
    )
    return EdgeTable(
        source=np.asarray(sources, dtype=np.uint64),
        target=np.asarray(targets, dtype=np.uint64),
        weight=np.asarray(weights, dtype=np.uint64),
    )


__all__ = ["sample_multinomial", "sample_poisson"]
