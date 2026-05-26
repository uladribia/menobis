"""Tests for binomial distribution family across all constraint cases."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import menobis._menobis as _menobis
from menobis.data.frames import EdgeTable
from menobis.filtering import FilterResult


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


# ---------------------------------------------------------------------------
# Strength-binomial (already tested elsewhere, add property tests)
# ---------------------------------------------------------------------------


class TestStrengthBinomial:
    """Property tests for strength-binomial sampling."""

    def test_weights_bounded_by_layers(self) -> None:
        """All binomial weights must be <= M."""
        x = np.array([0.5, 0.6, 0.4])
        y = np.array([0.4, 0.5, 0.5])
        _src, _tgt, wgt = _menobis.sample_strength_binomial(
            x.tolist(), y.tolist(), 5, True, 42
        )
        assert all(w <= 5 for w in wgt)

    def test_reproducible(self) -> None:
        """Seeded sampling is reproducible."""
        x = np.array([0.5, 0.6])
        y = np.array([0.4, 0.5])
        a = _menobis.sample_strength_binomial(x.tolist(), y.tolist(), 10, True, 42)
        b = _menobis.sample_strength_binomial(x.tolist(), y.tolist(), 10, True, 42)
        assert a == b


# ---------------------------------------------------------------------------
# Strength-cost binomial
# ---------------------------------------------------------------------------


class TestStrengthCostBinomial:
    """Tests for strength-cost binomial sampling and filtering."""

    def test_sample_weights_bounded(self) -> None:
        """Weights bounded by M layers."""
        x = [0.5, 0.6, 0.4]
        y = [0.4, 0.5, 0.5]
        cost_src = [0, 0, 1, 1, 2, 2]
        cost_tgt = [1, 2, 0, 2, 0, 1]
        cost_val = [1.0, 2.0, 1.0, 1.5, 2.0, 1.5]
        _src, _tgt, wgt = _menobis.sample_strength_cost_binomial(
            x, y, 0.5, cost_src, cost_tgt, cost_val, 5, True, 42
        )
        assert all(w <= 5 for w in wgt)

    def test_filter_partitions(self) -> None:
        """Filter partitions cover all edges."""
        edges = _small_edges()
        x = [0.5, 0.6, 0.4]
        y = [0.4, 0.5, 0.5]
        cost_src = [0, 0, 1, 1, 2, 2]
        cost_tgt = [1, 2, 0, 2, 0, 1]
        cost_val = [1.0, 2.0, 1.0, 1.5, 2.0, 1.5]
        upper, _lower, _expected, _occ = _menobis.filter_strength_cost_binomial(
            x,
            y,
            0.5,
            cost_src,
            cost_tgt,
            cost_val,
            5,
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        )
        assert len(upper) == len(edges)


# ---------------------------------------------------------------------------
# Strength-edges binomial (zero-inflated)
# ---------------------------------------------------------------------------


class TestStrengthEdgesBinomial:
    """Tests for strength-edges binomial zero-inflated sampling and filtering."""

    def test_sample_weights_bounded(self) -> None:
        """Weights bounded by M layers."""
        x = [0.5, 0.6, 0.4]
        y = [0.4, 0.5, 0.5]
        _src, _tgt, wgt = _menobis.sample_strength_edges_binomial(
            x, y, 0.5, 5, True, 42
        )
        assert all(w <= 5 for w in wgt)

    def test_sample_reproducible(self) -> None:
        """Seeded sampling is reproducible."""
        x = [0.5, 0.6]
        y = [0.4, 0.5]
        a = _menobis.sample_strength_edges_binomial(x, y, 0.5, 10, True, 42)
        b = _menobis.sample_strength_edges_binomial(x, y, 0.5, 10, True, 42)
        assert a == b

    def test_filter_partitions(self) -> None:
        """Filter partitions cover all edges."""
        edges = _small_edges()
        x = [0.5, 0.6, 0.4]
        y = [0.4, 0.5, 0.5]
        upper, _lower, _expected, _occ = _menobis.filter_strength_edges_binomial(
            x,
            y,
            0.5,
            5,
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        )
        assert len(upper) == len(edges)

    def test_pvalues_in_range(self) -> None:
        """All p-values in [0, 1]."""
        edges = _small_edges()
        x = [0.5, 0.6, 0.4]
        y = [0.4, 0.5, 0.5]
        upper, lower, _, _ = _menobis.filter_strength_edges_binomial(
            x,
            y,
            0.5,
            5,
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        )
        assert all(0.0 <= p <= 1.0 for p in upper)
        assert all(0.0 <= p <= 1.0 for p in lower)


# ---------------------------------------------------------------------------
# Strength-degree binomial (zero-inflated)
# ---------------------------------------------------------------------------


class TestStrengthDegreeBinomial:
    """Tests for strength-degree binomial zero-inflated sampling and filtering."""

    def test_sample_weights_bounded(self) -> None:
        """Weights bounded by M layers."""
        x = [0.5, 0.6, 0.4]
        y = [0.4, 0.5, 0.5]
        z = [0.3, 0.4, 0.2]
        w = [0.25, 0.35, 0.3]
        _src, _tgt, wgt = _menobis.sample_strength_degree_binomial(
            x, y, z, w, 5, True, 42
        )
        assert all(ww <= 5 for ww in wgt)

    def test_filter_partitions(self) -> None:
        """Filter partitions cover all edges."""
        edges = _small_edges()
        x = [0.5, 0.6, 0.4]
        y = [0.4, 0.5, 0.5]
        z = [0.3, 0.4, 0.2]
        w = [0.25, 0.35, 0.3]
        upper, _lower, _expected, _occ = _menobis.filter_strength_degree_binomial(
            x,
            y,
            z,
            w,
            5,
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        )
        assert len(upper) == len(edges)


# ---------------------------------------------------------------------------
# Degree-events binomial (zero-inflated)
# ---------------------------------------------------------------------------


class TestDegreeEventsBinomial:
    """Tests for degree-events binomial zero-inflated sampling and filtering."""

    def test_sample_weights_bounded(self) -> None:
        """Weights bounded by M layers."""
        x = [0.5, 0.6, 0.4]
        y = [0.4, 0.5, 0.5]
        _src, _tgt, wgt = _menobis.sample_degree_events_binomial(x, y, 0.5, 5, True, 42)
        assert all(w <= 5 for w in wgt)

    def test_filter_partitions(self) -> None:
        """Filter partitions cover all edges."""
        edges = _small_edges()
        x = [0.5, 0.6, 0.4]
        y = [0.4, 0.5, 0.5]
        upper, _lower, _expected, _occ = _menobis.filter_degree_events_binomial(
            x,
            y,
            0.5,
            5,
            edges.source.tolist(),
            edges.target.tolist(),
            edges.weight.tolist(),
        )
        assert len(upper) == len(edges)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(
    layers=st.integers(min_value=1, max_value=20),
    xy=st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=50)
def test_binomial_weight_bounded_by_layers(layers: int, xy: float) -> None:
    """Binomial samples never exceed M layers."""
    x = [xy]
    y = [1.0]
    _src, _tgt, wgt = _menobis.sample_strength_binomial(x, y, layers, True, 42)
    for w in wgt:
        assert w <= layers, f"weight {w} > layers {layers}"


@given(
    layers=st.integers(min_value=1, max_value=10),
    xy=st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=50)
def test_zip_binomial_weight_bounded(layers: int, xy: float) -> None:
    """zero-inflated binomial samples never exceed M layers."""
    x = [xy, xy]
    y = [1.0, 1.0]
    _src, _tgt, wgt = _menobis.sample_strength_edges_binomial(
        x, y, 0.5, layers, True, 42
    )
    for w in wgt:
        assert w <= layers, f"weight {w} > layers {layers}"


def test_binomial_total_weight_bounded_by_layers_times_pairs() -> None:
    """Total weight T <= M * candidate_pairs for binomial models."""
    n = 10
    layers = 5
    x = [0.9] * n
    y = [0.9] * n
    _src, _tgt, wgt = _menobis.sample_strength_binomial(x, y, layers, True, 42)
    max_t = layers * n * n  # with self-loops
    assert sum(wgt) <= max_t


def test_binomial_occupation_converges_with_layers() -> None:
    """As M grows, more pairs are occupied, converging to full occupation."""
    n = 5
    x = [0.5] * n
    y = [0.5] * n
    candidate_pairs = n * n
    prev_edges = 0
    for layers in [1, 5, 20, 100]:
        _src, _tgt, wgt = _menobis.sample_strength_binomial(x, y, layers, True, 42)
        n_edges = len(wgt)
        assert n_edges >= prev_edges or layers == 1
        prev_edges = n_edges
    # At M=100, nearly all pairs should be occupied
    assert prev_edges >= candidate_pairs * 0.9
