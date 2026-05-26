"""Tests for EdgeTable normalization."""

import numpy as np
import pytest

from menobis.data.frames import normalize_edges


def test_normalize_preserves_totals() -> None:
    edges = normalize_edges(np.array([0, 1]), np.array([1, 2]), np.array([3, 4]))
    assert edges.total_events == 7
    assert edges.num_edges == 2


def test_normalize_ignores_zero_weights() -> None:
    edges = normalize_edges(np.array([0, 1]), np.array([1, 2]), np.array([0, 3]))
    assert edges.num_edges == 1
    assert edges.total_events == 3


def test_normalize_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        normalize_edges(np.array([0]), np.array([1]), np.array([-1]))


def test_normalize_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        normalize_edges(np.array([0]), np.array([1, 2]), np.array([3]))
