"""Tests for all MENoBiS filtering models."""

from __future__ import annotations

import numpy as np

from menobis.data.frames import EdgeTable, ProbabilityTable
from menobis.filtering import (
    FilterResult,
    _solve_ztp_rate,
    filter_custom_poisson,
    filter_degree_events_poisson,
    filter_strength_cost_poisson,
    filter_strength_degree_poisson,
    filter_strength_edges_poisson,
)
from menobis.models.fitting import (
    fit_strength_cost_poisson,
    fit_strength_degree_poisson,
    fit_strength_edges_poisson,
)
from menobis.routing import Constraint, ModelFamily, filter_model


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


class TestFixedStrength:
    """Tests for fixed-strength Poisson filtering."""

    def test_partitions_edges(self) -> None:
        """Upper + lower + compatible == total edges."""
        edges = _small_edges()
        result = filter_model(
            edges, family=ModelFamily.ME, constraint=Constraint.STRENGTH, alpha=0.05
        )
        _assert_filter_partitions(result, len(edges))

    def test_pvalues_in_range(self) -> None:
        """All p-values are in [0, 1]."""
        edges = _small_edges()
        result = filter_model(
            edges, family=ModelFamily.ME, constraint=Constraint.STRENGTH, alpha=0.05
        )
        for group in [result.upper, result.lower, result.compatible]:
            assert np.all(group.upper_pvalue >= 0.0)
            assert np.all(group.upper_pvalue <= 1.0)
            assert np.all(group.lower_pvalue >= 0.0)
            assert np.all(group.lower_pvalue <= 1.0)

    def test_absent_detection(self) -> None:
        """Absent detection runs without error."""
        edges = _small_edges()
        result = filter_model(
            edges,
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH,
            alpha=0.05,
            detect_absent=True,
            min_occupation=0.0,
        )
        assert len(result.absent_lower.edges) >= 0


class TestCustomRates:
    """Tests for custom Poisson rate filtering."""

    def test_partitions_edges(self) -> None:
        """Upper + lower + compatible == total edges."""
        edges = _small_edges()
        rates = ProbabilityTable(
            source=edges.source,
            target=edges.target,
            probability=np.array([2.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float64),
        )
        result = filter_custom_poisson(edges, rates, alpha=0.05)
        _assert_filter_partitions(result, len(edges))


class TestStrengthEdges:
    """Tests for strength-edges zero-inflated filtering."""

    def test_partitions_edges(self) -> None:
        """Upper + lower + compatible == total edges."""
        edges = _small_edges()
        n = 3
        s_out = np.zeros(n, dtype=np.float64)
        s_in = np.zeros(n, dtype=np.float64)
        np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
        np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
        fit = fit_strength_edges_poisson(s_out, s_in, float(len(edges)))
        result = filter_strength_edges_poisson(edges, fit, alpha=0.05)
        _assert_filter_partitions(result, len(edges))

    def test_absent_detection(self) -> None:
        """Absent detection runs without error."""
        edges = _small_edges()
        n = 3
        s_out = np.zeros(n, dtype=np.float64)
        s_in = np.zeros(n, dtype=np.float64)
        np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
        np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
        fit = fit_strength_edges_poisson(s_out, s_in, float(len(edges)))
        result = filter_strength_edges_poisson(
            edges, fit, alpha=0.05, detect_absent=True, min_occupation=0.0
        )
        assert len(result.absent_lower.edges) >= 0


class TestStrengthCost:
    """Tests for strength-cost Poisson filtering."""

    def test_partitions_edges(self) -> None:
        """Upper + lower + compatible == total edges."""
        edges = _small_edges()
        n = 3
        s_out = np.zeros(n, dtype=np.float64)
        s_in = np.zeros(n, dtype=np.float64)
        np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
        np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
        cost_src = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint64)
        cost_tgt = np.array([1, 2, 0, 2, 0, 1], dtype=np.uint64)
        cost_val = np.array([1.0, 2.0, 1.0, 1.5, 2.0, 1.5], dtype=np.float64)
        fit = fit_strength_cost_poisson(s_out, s_in, cost_src, cost_tgt, cost_val, 1.5)
        result = filter_strength_cost_poisson(
            edges, fit, cost_src, cost_tgt, cost_val, alpha=0.05
        )
        _assert_filter_partitions(result, len(edges))

    def test_pvalues_in_range(self) -> None:
        """All p-values are in [0, 1]."""
        edges = _small_edges()
        n = 3
        s_out = np.zeros(n, dtype=np.float64)
        s_in = np.zeros(n, dtype=np.float64)
        np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
        np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
        cost_src = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint64)
        cost_tgt = np.array([1, 2, 0, 2, 0, 1], dtype=np.uint64)
        cost_val = np.array([1.0, 2.0, 1.0, 1.5, 2.0, 1.5], dtype=np.float64)
        fit = fit_strength_cost_poisson(s_out, s_in, cost_src, cost_tgt, cost_val, 1.5)
        result = filter_strength_cost_poisson(
            edges, fit, cost_src, cost_tgt, cost_val, alpha=0.05
        )
        for group in [result.upper, result.lower, result.compatible]:
            assert np.all(group.upper_pvalue >= 0.0)
            assert np.all(group.upper_pvalue <= 1.0)


class TestStrengthDegree:
    """Tests for strength-degree zero-inflated filtering."""

    def test_partitions_edges(self) -> None:
        """Upper + lower + compatible == total edges."""
        edges = _small_edges()
        n = 3
        s_out = np.zeros(n, dtype=np.float64)
        s_in = np.zeros(n, dtype=np.float64)
        d_out = np.zeros(n, dtype=np.float64)
        d_in = np.zeros(n, dtype=np.float64)
        np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
        np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
        for src in edges.source:
            d_out[src] += 1
        for tgt in edges.target:
            d_in[tgt] += 1
        fit = fit_strength_degree_poisson(s_out, s_in, d_out, d_in)
        result = filter_strength_degree_poisson(edges, fit, alpha=0.05)
        _assert_filter_partitions(result, len(edges))

    def test_absent_detection(self) -> None:
        """Absent detection runs without error."""
        edges = _small_edges()
        n = 3
        s_out = np.zeros(n, dtype=np.float64)
        s_in = np.zeros(n, dtype=np.float64)
        d_out = np.zeros(n, dtype=np.float64)
        d_in = np.zeros(n, dtype=np.float64)
        np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
        np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
        for src in edges.source:
            d_out[src] += 1
        for tgt in edges.target:
            d_in[tgt] += 1
        fit = fit_strength_degree_poisson(s_out, s_in, d_out, d_in)
        result = filter_strength_degree_poisson(
            edges, fit, alpha=0.05, detect_absent=True, min_occupation=0.0
        )
        assert len(result.absent_lower.edges) >= 0


class TestDegreeEvents:
    """Tests for degree-events zero-inflated filtering."""

    def test_partitions_edges(self) -> None:
        """Upper + lower + compatible == total edges."""
        edges = _small_edges()
        n = 3
        d_out = np.zeros(n, dtype=np.float64)
        d_in = np.zeros(n, dtype=np.float64)
        for src in edges.source:
            d_out[src] += 1
        for tgt in edges.target:
            d_in[tgt] += 1
        from menobis.models.fitting import fit_degree_events_poisson

        fit = fit_degree_events_poisson(
            d_out, d_in, int(edges.weight.sum()), self_loops=False
        )
        result = filter_degree_events_poisson(edges, fit, alpha=0.05)
        _assert_filter_partitions(result, len(edges))

    def test_absent_detection(self) -> None:
        """Absent detection runs without error."""
        edges = _small_edges()
        n = 3
        d_out = np.zeros(n, dtype=np.float64)
        d_in = np.zeros(n, dtype=np.float64)
        for src in edges.source:
            d_out[src] += 1
        for tgt in edges.target:
            d_in[tgt] += 1
        from menobis.models.fitting import fit_degree_events_poisson

        fit = fit_degree_events_poisson(
            d_out, d_in, int(edges.weight.sum()), self_loops=False
        )
        result = filter_degree_events_poisson(
            edges,
            fit,
            alpha=0.05,
            detect_absent=True,
            min_occupation=0.0,
        )
        assert len(result.absent_lower.edges) >= 0


class TestPartialConstraints:
    """Tests for partial-constraint filtering via custom rates."""

    def test_partial_fit_feeds_custom_rates_filter(self) -> None:
        """PartialFitResult feeds directly into filter_custom_poisson."""
        from menobis.models.partial import fit_partial_strength_poisson

        edges = _small_edges()
        n = 3
        s_out = np.zeros(n, dtype=np.float64)
        s_in = np.zeros(n, dtype=np.float64)
        np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
        np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
        known_src = np.array([0], dtype=np.uint64)
        known_tgt = np.array([1], dtype=np.uint64)
        known_rate = np.array([3.0], dtype=np.float64)
        partial = fit_partial_strength_poisson(
            s_out, s_in, known_src, known_tgt, known_rate
        )
        rates = partial.as_probability_table()
        result = filter_custom_poisson(edges, rates, alpha=0.05)
        _assert_filter_partitions(result, len(edges))


class TestSolveZtpRate:
    """Tests for the positive Poisson rate solver."""

    def test_unit_mean_returns_zero(self) -> None:
        """Mean of 1.0 implies rate 0."""
        assert _solve_ztp_rate(1.0) == 0.0

    def test_above_one_is_positive(self) -> None:
        """Mean > 1 requires positive rate."""
        assert _solve_ztp_rate(2.0) > 0.0

    def test_below_one_returns_zero(self) -> None:
        """Mean < 1 is infeasible, returns 0."""
        assert _solve_ztp_rate(0.5) == 0.0
