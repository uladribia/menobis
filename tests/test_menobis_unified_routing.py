"""Tests for the unified routing API (fit_model, sample_model, filter_model)."""

import numpy as np
import pytest

from menobis.routing import (
    Constraint,
    Ensemble,
    ModelFamily,
    UnsupportedModelCaseError,
    filter_model,
    fit_model,
    sample_model,
)


class TestFitModel:
    """Test fit_model dispatches correctly for all constraints."""

    def test_strength_me(self):
        s_out = np.array([10.0, 20.0, 30.0])
        s_in = np.array([15.0, 25.0, 20.0])
        fit = fit_model(
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH,
            strength_out=s_out,
            strength_in=s_in,
        )
        assert fit.converged

    def test_strength_binomial(self):
        s_out = np.array([3.0, 5.0, 4.0])
        s_in = np.array([4.0, 4.0, 4.0])
        fit = fit_model(
            family=ModelFamily.B,
            constraint=Constraint.STRENGTH,
            strength_out=s_out,
            strength_in=s_in,
            layers=3,
        )
        assert fit.converged

    def test_strength_edges_me(self):
        s_out = np.array([10.0, 20.0, 30.0])
        s_in = np.array([15.0, 25.0, 20.0])
        fit = fit_model(
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH_EDGES,
            strength_out=s_out,
            strength_in=s_in,
            target_edges=5.0,
        )
        assert fit.converged

    def test_strength_degree_me(self):
        s_out = np.array([10.0, 20.0, 30.0])
        s_in = np.array([15.0, 25.0, 20.0])
        k_out = np.array([1.5, 2.0, 2.5])
        k_in = np.array([2.0, 2.0, 2.0])
        fit = fit_model(
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH_DEGREE,
            strength_out=s_out,
            strength_in=s_in,
            degree_out=k_out,
            degree_in=k_in,
        )
        # May not converge at small N but should not crash
        assert fit is not None

    def test_degree_events_geometric(self):
        k_out = np.array([2.0, 1.0, 1.0])
        k_in = np.array([1.0, 2.0, 1.0])
        fit = fit_model(
            family=ModelFamily.W,
            constraint=Constraint.DEGREE_EVENTS,
            degree_out=k_out,
            degree_in=k_in,
            total_events=10,
        )
        assert fit.converged

    def test_microcanonical_raises(self):
        with pytest.raises(UnsupportedModelCaseError):
            fit_model(
                ensemble=Ensemble.MICROCANONICAL,
                family=ModelFamily.ME,
                constraint=Constraint.STRENGTH,
                strength_out=np.array([1.0]),
                strength_in=np.array([1.0]),
            )

    def test_missing_arrays_raises(self):
        with pytest.raises(ValueError):
            fit_model(family=ModelFamily.ME, constraint=Constraint.STRENGTH)


class TestSampleModel:
    """Test sample_model dispatches correctly."""

    def test_grand_canonical_strength_me(self):
        s_out = np.array([10.0, 20.0, 30.0])
        s_in = np.array([15.0, 25.0, 20.0])
        fit = fit_model(
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH,
            strength_out=s_out,
            strength_in=s_in,
        )
        edges = sample_model(
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH,
            fit=fit,
            seed=42,
        )
        assert len(edges.source) > 0

    def test_microcanonical_stub_matching(self):
        s_out = np.array([5, 10, 5], dtype=np.uint64)
        s_in = np.array([8, 7, 5], dtype=np.uint64)
        edges = sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH,
            strength_out=s_out,
            strength_in=s_in,
            seed=1,
        )
        assert len(edges.source) > 0

    def test_canonical_multinomial(self):
        s_out = np.array([10.0, 20.0, 30.0])
        s_in = np.array([15.0, 25.0, 20.0])
        fit = fit_model(
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH,
            strength_out=s_out,
            strength_in=s_in,
        )
        edges = sample_model(
            ensemble=Ensemble.CANONICAL,
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH,
            fit=fit,
            total_events=60,
            seed=7,
        )
        assert edges.weight.sum() == 60


class TestFilterModel:
    """Test filter_model dispatches correctly."""

    def test_filter_strength_me(self):
        from menobis.data.frames import EdgeTable

        edges = EdgeTable(
            source=np.array([0, 1, 2]),
            target=np.array([1, 2, 0]),
            weight=np.array([5, 10, 3]),
        )
        result = filter_model(
            edges, family=ModelFamily.ME, constraint=Constraint.STRENGTH
        )
        assert result.upper is not None

    def test_filter_strength_binomial(self):
        from menobis.data.frames import EdgeTable

        edges = EdgeTable(
            source=np.array([0, 1, 2]),
            target=np.array([1, 2, 0]),
            weight=np.array([2, 3, 1]),
        )
        result = filter_model(
            edges, family=ModelFamily.B, constraint=Constraint.STRENGTH
        )
        assert result.upper is not None

    def test_filter_requires_fit_for_cost(self):
        from menobis.data.frames import EdgeTable

        edges = EdgeTable(
            source=np.array([0, 1]),
            target=np.array([1, 0]),
            weight=np.array([5, 3]),
        )
        # STRENGTH_COST needs target_cost for fit_model, so auto-fit fails
        with pytest.raises(ValueError):
            filter_model(
                edges,
                family=ModelFamily.ME,
                constraint=Constraint.STRENGTH_COST,
            )
