"""Tests for W zero-inflated sampling across all constraint types."""

import numpy as np

from menobis.models import (
    StrengthDegreeFit,
    StrengthEdgesFit,
)
from menobis.models.generation import (
    sample_degree_events_geometric,
    sample_degree_events_negative_binomial,
    sample_strength_cost_geometric,
    sample_strength_cost_negative_binomial,
    sample_strength_degree_geometric,
    sample_strength_degree_negative_binomial,
    sample_strength_edges_geometric,
    sample_strength_edges_negative_binomial,
)


def _strength_edges_fit() -> StrengthEdgesFit:
    return StrengthEdgesFit(
        node=np.array([0, 1], dtype=np.uint64),
        x=np.array([0.5, 0.8]),
        y=np.array([0.6, 0.7]),
        lam=1.5,
        converged=True,
        iterations=10,
        self_loops=True,
    )


def _strength_degree_fit() -> StrengthDegreeFit:
    return StrengthDegreeFit(
        node=np.array([0, 1], dtype=np.uint64),
        x=np.array([0.3, 0.4]),
        y=np.array([0.35, 0.45]),
        z=np.array([0.6, 0.7]),
        w=np.array([0.65, 0.75]),
        converged=True,
        iterations=10,
        self_loops=True,
    )


def _degree_fit():
    from menobis.models.types import DegreeEventsFit

    return DegreeEventsFit(
        node=np.array([0, 1], dtype=np.uint64),
        x=np.array([0.8, 1.2]),
        y=np.array([1.0, 1.0]),
        q=0.5,
        positive_mean=2.0,
        self_loops=True,
        converged=True,
        iterations=10,
        family="geometric",
        layers=1,
    )


class TestStrengthEdgesGeometric:
    """Tests for strength-edges geometric zero-inflated sampling."""

    def test_produces_edges(self) -> None:
        """Sampler returns a valid EdgeTable."""
        fit = _strength_edges_fit()
        sample = sample_strength_edges_geometric(fit, seed=42)
        assert sample.num_edges >= 0

    def test_seeded_reproducibility(self) -> None:
        """Same seed gives same result."""
        fit = _strength_edges_fit()
        a = sample_strength_edges_geometric(fit, seed=99)
        b = sample_strength_edges_geometric(fit, seed=99)
        np.testing.assert_array_equal(a.weight, b.weight)


class TestStrengthEdgesNegativeBinomial:
    """Tests for strength-edges negative binomial zero-inflated sampling."""

    def test_produces_edges(self) -> None:
        """Sampler returns a valid EdgeTable."""
        fit = _strength_edges_fit()
        sample = sample_strength_edges_negative_binomial(fit, layers=3, seed=42)
        assert sample.num_edges >= 0

    def test_seeded_reproducibility(self) -> None:
        """Same seed gives same result."""
        fit = _strength_edges_fit()
        a = sample_strength_edges_negative_binomial(fit, layers=3, seed=99)
        b = sample_strength_edges_negative_binomial(fit, layers=3, seed=99)
        np.testing.assert_array_equal(a.weight, b.weight)


class TestStrengthDegreeGeometric:
    """Tests for strength-degree geometric zero-inflated sampling."""

    def test_produces_edges(self) -> None:
        """Sampler returns a valid EdgeTable."""
        fit = _strength_degree_fit()
        sample = sample_strength_degree_geometric(fit, seed=42)
        assert sample.num_edges >= 0

    def test_seeded_reproducibility(self) -> None:
        """Same seed gives same result."""
        fit = _strength_degree_fit()
        a = sample_strength_degree_geometric(fit, seed=99)
        b = sample_strength_degree_geometric(fit, seed=99)
        np.testing.assert_array_equal(a.weight, b.weight)


class TestStrengthDegreeNegativeBinomial:
    """Tests for strength-degree negative binomial zero-inflated sampling."""

    def test_produces_edges(self) -> None:
        """Sampler returns a valid EdgeTable."""
        fit = _strength_degree_fit()
        sample = sample_strength_degree_negative_binomial(fit, layers=3, seed=42)
        assert sample.num_edges >= 0

    def test_seeded_reproducibility(self) -> None:
        """Same seed gives same result."""
        fit = _strength_degree_fit()
        a = sample_strength_degree_negative_binomial(fit, layers=3, seed=99)
        b = sample_strength_degree_negative_binomial(fit, layers=3, seed=99)
        np.testing.assert_array_equal(a.weight, b.weight)


class TestDegreeEventsGeometric:
    """Tests for degree-events geometric zero-inflated sampling."""

    def test_produces_edges(self) -> None:
        """Sampler returns a valid EdgeTable."""
        fit = _degree_fit()
        sample = sample_degree_events_geometric(fit, seed=42)
        assert sample.num_edges >= 0

    def test_seeded_reproducibility(self) -> None:
        """Same seed gives same result."""
        fit = _degree_fit()
        a = sample_degree_events_geometric(fit, seed=99)
        b = sample_degree_events_geometric(fit, seed=99)
        np.testing.assert_array_equal(a.weight, b.weight)


class TestDegreeEventsNegativeBinomial:
    """Tests for degree-events negative binomial zero-inflated sampling."""

    def test_produces_edges(self) -> None:
        """Sampler returns a valid EdgeTable."""
        fit = _degree_fit()
        sample = sample_degree_events_negative_binomial(fit, layers=3, seed=42)
        assert sample.num_edges >= 0

    def test_seeded_reproducibility(self) -> None:
        """Same seed gives same result."""
        fit = _degree_fit()
        a = sample_degree_events_negative_binomial(fit, layers=3, seed=99)
        b = sample_degree_events_negative_binomial(fit, layers=3, seed=99)
        np.testing.assert_array_equal(a.weight, b.weight)


def _strength_cost_fit():
    """Build a mock StrengthCostFit."""
    from menobis.models.fitting import StrengthCostFit

    return StrengthCostFit(
        node=np.array([0, 1], dtype=np.uint64),
        x=np.array([0.5, 0.8]),
        y=np.array([0.6, 0.7]),
        gamma=0.1,
        converged=True,
        iterations=10,
        self_loops=True,
    )


class TestStrengthCostGeometric:
    """Tests for strength-cost geometric sampling."""

    def test_produces_edges(self) -> None:
        """Sampler returns a valid EdgeTable."""
        fit = _strength_cost_fit()
        c_src = np.array([0, 1], dtype=np.int64)
        c_tgt = np.array([1, 0], dtype=np.int64)
        c_val = np.array([1.0, 2.0])
        sample = sample_strength_cost_geometric(fit, c_src, c_tgt, c_val, seed=42)
        assert sample.num_edges >= 0

    def test_seeded_reproducibility(self) -> None:
        """Same seed gives same result."""
        fit = _strength_cost_fit()
        c_src = np.array([0, 1], dtype=np.int64)
        c_tgt = np.array([1, 0], dtype=np.int64)
        c_val = np.array([1.0, 2.0])
        a = sample_strength_cost_geometric(fit, c_src, c_tgt, c_val, seed=99)
        b = sample_strength_cost_geometric(fit, c_src, c_tgt, c_val, seed=99)
        np.testing.assert_array_equal(a.weight, b.weight)


class TestStrengthCostNegativeBinomial:
    """Tests for strength-cost negative binomial sampling."""

    def test_produces_edges(self) -> None:
        """Sampler returns a valid EdgeTable."""
        fit = _strength_cost_fit()
        c_src = np.array([0, 1], dtype=np.int64)
        c_tgt = np.array([1, 0], dtype=np.int64)
        c_val = np.array([1.0, 2.0])
        sample = sample_strength_cost_negative_binomial(
            fit, c_src, c_tgt, c_val, layers=3, seed=42
        )
        assert sample.num_edges >= 0

    def test_seeded_reproducibility(self) -> None:
        """Same seed gives same result."""
        fit = _strength_cost_fit()
        c_src = np.array([0, 1], dtype=np.int64)
        c_tgt = np.array([1, 0], dtype=np.int64)
        c_val = np.array([1.0, 2.0])
        a = sample_strength_cost_negative_binomial(
            fit, c_src, c_tgt, c_val, layers=3, seed=99
        )
        b = sample_strength_cost_negative_binomial(
            fit, c_src, c_tgt, c_val, layers=3, seed=99
        )
        np.testing.assert_array_equal(a.weight, b.weight)
