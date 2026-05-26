"""Tests for Rust-backed statistics."""

import numpy as np
import pytest

from menobis.analysis import compute_all_stats, weight_distribution
from menobis.data.frames import normalize_edges


def test_weight_distribution() -> None:
    edges = normalize_edges(
        np.array([0, 0, 1]), np.array([1, 2, 2]), np.array([3, 3, 5])
    )
    dist = weight_distribution(edges)
    np.testing.assert_array_equal(dist.weight, [3, 5])
    np.testing.assert_array_equal(dist.count, [2, 1])


def test_compute_all_stats_y2() -> None:
    edges = normalize_edges(
        np.array([0, 0, 0]), np.array([1, 2, 3]), np.array([4, 4, 4])
    )
    stats = compute_all_stats(edges)
    assert stats.y2_out[0] == pytest.approx(1 / 3)


def test_compute_all_stats_knn() -> None:
    edges = normalize_edges(
        np.array([0, 0, 1]), np.array([1, 2, 2]), np.array([1, 1, 1])
    )
    stats = compute_all_stats(edges)
    # k_nn_out(0) = (k_in(1) + k_in(2)) / k_out(0) = (1+2)/2 = 1.5
    assert stats.k_nn_out[0] == pytest.approx(1.5)


def test_compute_all_stats_snn() -> None:
    edges = normalize_edges(np.array([0, 0]), np.array([1, 2]), np.array([2, 3]))
    stats = compute_all_stats(edges)
    # s_nn_out(0) = (2*s_in(1) + 3*s_in(2)) / s_out(0) = (2*2 + 3*3)/5 = 13/5
    assert stats.s_nn_out[0] == pytest.approx(13 / 5)
