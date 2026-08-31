"""P0.2 public API inventory and clean API definition.

Snapshots the public surface: enum members, function signatures, module
exports, default behaviours, and seed reproducibility. This is the
contract that later Phase 0 work packages must not silently break.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np

import menobis
import menobis.analysis
import menobis.data
import menobis.filtering
import menobis.models
import menobis.routing
from menobis.models.spec import Constraint, Ensemble, ModelFamily, Verb
from menobis.models.types import (
    DegreeEventsFit,
    DegreeFit,
    EdgesEventsFit,
    FitResult,
    PartialFitResult,
    StrengthCostFit,
    StrengthDegreeFit,
    StrengthEdgesFit,
    StrengthFit,
)
from menobis.routing import filter_model, fit_model, sample_model

# ---------------------------------------------------------------------------
# Enum inventories
# ---------------------------------------------------------------------------


def test_verb_enum_members() -> None:
    assert set(Verb) == {Verb.FIT, Verb.SAMPLE, Verb.FILTER}


def test_ensemble_enum_members() -> None:
    assert set(Ensemble) == {
        Ensemble.GRAND_CANONICAL,
        Ensemble.CANONICAL,
        Ensemble.MICROCANONICAL,
    }


def test_family_enum_members() -> None:
    assert set(ModelFamily) == {ModelFamily.ME, ModelFamily.B, ModelFamily.W}


def test_constraint_enum_members() -> None:
    assert set(Constraint) == {
        Constraint.STRENGTH,
        Constraint.STRENGTH_COST,
        Constraint.STRENGTH_EDGES,
        Constraint.STRENGTH_DEGREE,
        Constraint.DEGREE_EVENTS,
        Constraint.EDGES_EVENTS,
    }


# ---------------------------------------------------------------------------
# Public function signatures
# ---------------------------------------------------------------------------


def test_fit_model_signature() -> None:
    sig = inspect.signature(fit_model)
    params = sig.parameters
    assert params["family"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["constraint"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["ensemble"].default is Ensemble.GRAND_CANONICAL
    assert params["family"].default is inspect.Parameter.empty
    assert params["constraint"].default is inspect.Parameter.empty
    for name in (
        "strength_out",
        "strength_in",
        "degree_out",
        "degree_in",
        "total_events",
        "target_edges",
        "target_cost",
        "coord_x",
        "coord_y",
        "known_source",
        "known_target",
        "known_occnum",
        "layers",
        "self_loops",
        "tolerance",
        "max_iterations",
    ):
        assert name in params, f"fit_model missing parameter {name}"


def test_sample_model_signature() -> None:
    sig = inspect.signature(sample_model)
    params = sig.parameters
    assert params["family"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["constraint"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["ensemble"].default is Ensemble.GRAND_CANONICAL
    for name in (
        "fit",
        "strength_out",
        "strength_in",
        "total_events",
        "coord_x",
        "coord_y",
        "layers",
        "seed",
    ):
        assert name in params, f"sample_model missing parameter {name}"


def test_filter_model_signature() -> None:
    sig = inspect.signature(filter_model)
    params = sig.parameters
    assert "edges" in params
    assert params["family"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["constraint"].kind == inspect.Parameter.KEYWORD_ONLY
    for name in (
        "fit",
        "layers",
        "self_loops",
        "alpha",
        "tail",
        "correction",
    ):
        assert name in params, f"filter_model missing parameter {name}"


def test_fit_result_dataclass_fields() -> None:
    """Every public fit result exposes the documented field names."""
    expected: dict[type, set[str]] = {
        StrengthFit: {"node", "x", "y", "self_loops", "converged", "iterations"},
        DegreeFit: {"node", "x", "y", "self_loops", "converged", "iterations"},
        StrengthCostFit: {
            "node",
            "x",
            "y",
            "gamma",
            "self_loops",
            "converged",
            "iterations",
        },
        StrengthEdgesFit: {
            "node",
            "x",
            "y",
            "lam",
            "self_loops",
            "converged",
            "iterations",
        },
        StrengthDegreeFit: {
            "node",
            "x",
            "y",
            "z",
            "w",
            "self_loops",
            "converged",
            "iterations",
        },
        DegreeEventsFit: {
            "node",
            "x",
            "y",
            "q",
            "positive_mean",
            "self_loops",
            "converged",
            "iterations",
        },
        EdgesEventsFit: {
            "q",
            "lam",
            "occupation",
            "positive_mean",
            "node_count",
            "self_loops",
            "converged",
            "iterations",
        },
        PartialFitResult: {
            "source",
            "target",
            "intensity",
            "constraint",
            "family",
            "self_loops",
            "converged",
            "iterations",
        },
    }
    for cls, required in expected.items():
        actual = {f.name for f in fields(cls) if not f.name.startswith("_")}  # type: ignore
        missing = required - actual
        assert not missing, f"{cls.__name__} missing fields: {missing}"


def test_all_fit_results_inherit_fit_result() -> None:
    """All concrete fit results share the FitResult diagnostics surface."""
    for cls in (
        StrengthFit,
        DegreeFit,
        StrengthCostFit,
        StrengthEdgesFit,
        StrengthDegreeFit,
        DegreeEventsFit,
        EdgesEventsFit,
        PartialFitResult,
    ):
        assert issubclass(cls, FitResult)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


def test_package_exports_version() -> None:
    assert menobis.__version__ == "1.1.0"
    assert menobis.__all__ == ["__version__"]


def test_routing_exports() -> None:
    expected = {
        "Constraint",
        "Ensemble",
        "ModelFamily",
        "UnsupportedModelCaseError",
        "filter_model",
        "fit_model",
        "sample_model",
        "sample_model_detailed",
    }
    assert set(menobis.routing.__all__) == expected


def test_models_exports() -> None:
    for name in (
        "Constraint",
        "Ensemble",
        "ModelFamily",
        "UnsupportedModelCaseError",
        "FitResult",
        "StrengthFit",
        "StrengthCostFit",
        "StrengthEdgesFit",
        "StrengthDegreeFit",
        "DegreeFit",
        "DegreeEventsFit",
        "PartialFitResult",
        "OptimizationDiagnostics",
        "ConicDiagnostics",
        "fit_model",
        "sample_model",
    ):
        assert hasattr(menobis.models, name), f"menobis.models missing {name}"


def test_analysis_exports() -> None:
    for name in (
        "directed_strengths",
        "directed_degrees",
        "compute_all_stats",
        "occupation_distribution",
        "clustering_coefficient",
        "occupation_clustering_coefficient",
        "NodeStats",
        "OccupationDistribution",
        "ClusteringResult",
        "DirectedSequences",
    ):
        assert hasattr(menobis.analysis, name), f"menobis.analysis missing {name}"


def test_enum_values_are_stable_strings() -> None:
    """Enum string values are the public wire format (CLI, JSON, files)."""
    assert Constraint.STRENGTH.value == "strength"
    assert Constraint.STRENGTH_COST.value == "strength_cost"
    assert Constraint.STRENGTH_EDGES.value == "strength_edges"
    assert Constraint.STRENGTH_DEGREE.value == "strength_degree"
    assert Constraint.DEGREE_EVENTS.value == "degree_events"
    assert Constraint.EDGES_EVENTS.value == "edges_events"
    assert Ensemble.GRAND_CANONICAL.value == "grandcanonical"
    assert Ensemble.CANONICAL.value == "canonical"
    assert Ensemble.MICROCANONICAL.value == "microcanonical"
    assert ModelFamily.ME.value == "me"
    assert ModelFamily.B.value == "b"
    assert ModelFamily.W.value == "w"


# ---------------------------------------------------------------------------
# Default behaviours
# ---------------------------------------------------------------------------


def test_fit_model_defaults_are_grand_canonical() -> None:
    """Default ensemble is grand-canonical; self-loops default on."""
    from menobis.models.spec import Ensemble

    sig = inspect.signature(fit_model)
    assert sig.parameters["ensemble"].default is Ensemble.GRAND_CANONICAL
    assert sig.parameters["self_loops"].default is True
    assert sig.parameters["tolerance"].default == 1e-8
    assert sig.parameters["max_iterations"].default == 10000


def test_sample_seed_reproducibility() -> None:
    """Same seed -> identical sampled network; different seed -> independent."""
    s_out = np.array([10.0, 20.0])
    s_in = np.array([15.0, 15.0])
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
    )
    a = sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        seed=42,
    )
    b = sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        seed=42,
    )
    c = sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        seed=43,
    )
    np.testing.assert_array_equal(a.source, b.source)
    np.testing.assert_array_equal(a.target, b.target)
    np.testing.assert_array_equal(a.occ_num, b.occ_num)
    assert not (
        np.array_equal(a.source, c.source)
        and np.array_equal(a.target, c.target)
        and np.array_equal(a.occ_num, c.occ_num)
    )


def test_edge_table_default_positive_occupation() -> None:
    """Sampled edge tables only contain positive occupation numbers."""
    s_out = np.array([5.0, 5.0])
    s_in = np.array([5.0, 5.0])
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
    )
    sample = sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        seed=7,
    )
    assert np.all(sample.occ_num >= 1)
    assert len(sample.source) == len(sample.target) == len(sample.occ_num)
