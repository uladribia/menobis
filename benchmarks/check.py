"""Constraint checks for benchmark samples."""

from __future__ import annotations

import numpy as np

from odme.data.frames import EdgeTable


def sample_strengths(sample: EdgeTable, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return outgoing and incoming weighted strengths."""
    out = np.zeros(node_count, dtype=np.float64)
    incoming = np.zeros(node_count, dtype=np.float64)
    np.add.at(out, sample.source.astype(np.int64), sample.weight.astype(np.float64))
    np.add.at(incoming, sample.target.astype(np.int64), sample.weight.astype(np.float64))
    return out, incoming


def sample_degrees(sample: EdgeTable, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return outgoing and incoming binary degrees."""
    out = np.zeros(node_count, dtype=np.float64)
    incoming = np.zeros(node_count, dtype=np.float64)
    np.add.at(out, sample.source.astype(np.int64), 1.0)
    np.add.at(incoming, sample.target.astype(np.int64), 1.0)
    return out, incoming


def max_pair_error(
    actual_out: np.ndarray,
    actual_in: np.ndarray,
    expected_out: np.ndarray,
    expected_in: np.ndarray,
) -> float:
    """Return max absolute in/out sequence error."""
    return float(
        max(
            np.max(np.abs(actual_out - expected_out), initial=0.0),
            np.max(np.abs(actual_in - expected_in), initial=0.0),
        )
    )
