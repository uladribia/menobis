"""Tests for geometric, binomial, and negative binomial distribution families."""

from __future__ import annotations

import numpy as np

from odme.data.frames import EdgeTable
from odme.filtering import (
    FilterResult,
    filter_strength_binomial,
    filter_strength_geometric,
    filter_strength_negative_binomial,
)
from odme.models.fitting import fit_strength_binomial
from odme.models.generation import (
    sample_strength_binomial,
    sample_strength_geometric,
    sample_strength_negative_binomial,
)


def _small_edges() -> EdgeTable:
    """Build a small 3-node directed weighted graph."""
    return EdgeTable(
        source=np.array([0, 0, 1, 1, 2], dtype=np.uint64),
        target=np.array([1, 2, 0, 2, 0], dtype=np.uint64),
        weight=np.array([3, 1, 2, 4, 1], dtype=np.uint64),
    )


def _assert_filter_partitions(result: FilterResult, n_edges: int) -> None:
    """Assert upper + lower + compatible == total observed edges."""
    total = (
        len(result.upper.edges) + len(result.lower.edges) + len(result.compatible.edges)
    )
    assert total == n_edges


class TestGeometricGeneration:
    """Tests for geometric sampling."""

    def test_reproducible(self) -> None:
        """Seeded geometric sampling is reproducible."""
        x = np.array([0.3, 0.4, 0.2])
        y = np.array([0.25, 0.35, 0.3])
        a = sample_strength_geometric(x, y, seed=42)
        b = sample_strength_geometric(x, y, seed=42)
        np.testing.assert_array_equal(a.source, b.source)
        np.testing.assert_array_equal(a.weight, b.weight)

    def test_nonneg_weights(self) -> None:
        """All geometric weights are non-negative."""
        x = np.array([0.3, 0.4, 0.2])
        y = np.array([0.25, 0.35, 0.3])
        edges = sample_strength_geometric(x, y, seed=0)
        assert np.all(edges.weight >= 0)


class TestBinomialGeneration:
    """Tests for binomial sampling."""

    def test_reproducible(self) -> None:
        """Seeded binomial sampling is reproducible."""
        x = np.array([0.5, 0.6])
        y = np.array([0.4, 0.5])
        a = sample_strength_binomial(x, y, layers=10, seed=42)
        b = sample_strength_binomial(x, y, layers=10, seed=42)
        np.testing.assert_array_equal(a.source, b.source)
        np.testing.assert_array_equal(a.weight, b.weight)

    def test_max_weight_bounded_by_layers(self) -> None:
        """Binomial weights cannot exceed M layers."""
        x = np.array([0.9, 0.8])
        y = np.array([0.9, 0.8])
        edges = sample_strength_binomial(x, y, layers=5, seed=0)
        assert np.all(edges.weight <= 5)


class TestNegativeBinomialGeneration:
    """Tests for negative binomial sampling."""

    def test_reproducible(self) -> None:
        """Seeded negative binomial sampling is reproducible."""
        x = np.array([0.3, 0.4])
        y = np.array([0.25, 0.35])
        a = sample_strength_negative_binomial(x, y, layers=3, seed=42)
        b = sample_strength_negative_binomial(x, y, layers=3, seed=42)
        np.testing.assert_array_equal(a.source, b.source)
        np.testing.assert_array_equal(a.weight, b.weight)

    def test_nonneg_weights(self) -> None:
        """All negative binomial weights are non-negative."""
        x = np.array([0.3, 0.4])
        y = np.array([0.25, 0.35])
        edges = sample_strength_negative_binomial(x, y, layers=3, seed=0)
        assert np.all(edges.weight >= 0)


class TestBinomialFitting:
    """Tests for binomial IPF fitting."""

    def test_converges_small(self) -> None:
        """Binomial fitting converges on a small graph."""
        s_out = np.array([10.0, 15.0, 5.0])
        s_in = np.array([8.0, 12.0, 10.0])
        fit = fit_strength_binomial(
            s_out, s_in, layers=5, tolerance=0.01, max_iterations=50000
        )
        assert fit.converged

    def test_recovers_constraints(self) -> None:
        """Fitted binomial model approximately recovers strength constraints."""
        s_out = np.array([20.0, 30.0, 10.0, 40.0])
        s_in = np.array([25.0, 25.0, 20.0, 30.0])
        fit = fit_strength_binomial(s_out, s_in, layers=10, max_iterations=50000)
        n = len(s_out)
        pred_out = np.zeros(n)
        pred_in = np.zeros(n)
        for i in range(n):
            for j in range(n):
                xy = fit.x[i] * fit.y[j]
                expected = 10.0 * xy / (1.0 + xy)
                pred_out[i] += expected
                pred_in[j] += expected
        np.testing.assert_allclose(pred_out, s_out, rtol=0.05)
        np.testing.assert_allclose(pred_in, s_in, rtol=0.05)


class TestGeometricFiltering:
    """Tests for geometric filtering."""

    def test_partitions_edges(self) -> None:
        """Filter partitions cover all edges."""
        edges = _small_edges()
        x = np.array([0.3, 0.4, 0.2])
        y = np.array([0.25, 0.35, 0.3])
        result = filter_strength_geometric(edges, x, y, alpha=0.05)
        _assert_filter_partitions(result, len(edges))


class TestBinomialFiltering:
    """Tests for binomial filtering."""

    def test_partitions_edges(self) -> None:
        """Filter partitions cover all edges."""
        edges = _small_edges()
        x = np.array([0.5, 0.6, 0.4])
        y = np.array([0.4, 0.5, 0.5])
        result = filter_strength_binomial(edges, x, y, layers=10, alpha=0.05)
        _assert_filter_partitions(result, len(edges))

    def test_pvalues_in_range(self) -> None:
        """All p-values are in [0, 1]."""
        edges = _small_edges()
        x = np.array([0.5, 0.6, 0.4])
        y = np.array([0.4, 0.5, 0.5])
        result = filter_strength_binomial(edges, x, y, layers=10, alpha=0.05)
        for group in [result.upper, result.lower, result.compatible]:
            assert np.all(group.upper_pvalue >= 0.0)
            assert np.all(group.upper_pvalue <= 1.0)


class TestNegativeBinomialFiltering:
    """Tests for negative binomial filtering."""

    def test_partitions_edges(self) -> None:
        """Filter partitions cover all edges."""
        edges = _small_edges()
        x = np.array([0.3, 0.4, 0.2])
        y = np.array([0.25, 0.35, 0.3])
        result = filter_strength_negative_binomial(edges, x, y, layers=3, alpha=0.05)
        _assert_filter_partitions(result, len(edges))
