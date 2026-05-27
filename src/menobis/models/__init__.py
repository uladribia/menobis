"""Multi-edge model helpers for MENoBiS."""

from typing import Any

from menobis.data.frames import EdgeTable
from menobis.models.fitting import (
    fit_degree_bernoulli,
    fit_degree_events_binomial,
    fit_degree_events_geometric,
    fit_degree_events_negative_binomial,
    fit_degree_events_poisson,
    fit_strength_binomial,
    fit_strength_cost_binomial,
    fit_strength_cost_geometric,
    fit_strength_cost_negative_binomial,
    fit_strength_cost_poisson,
    fit_strength_degree_binomial,
    fit_strength_degree_geometric,
    fit_strength_degree_negative_binomial,
    fit_strength_degree_poisson,
    fit_strength_edges_binomial,
    fit_strength_edges_geometric,
    fit_strength_edges_negative_binomial,
    fit_strength_edges_poisson,
    fit_strength_geometric,
    fit_strength_negative_binomial,
    fit_strength_poisson,
    validate_strength_degree_constraints,
)
from menobis.models.generation import (
    sample_custom_multinomial,
    sample_custom_poisson,
    sample_degree_events_binomial,
    sample_degree_events_geometric,
    sample_degree_events_negative_binomial,
    sample_degree_events_poisson,
    sample_strength_binomial,
    sample_strength_cost_binomial,
    sample_strength_cost_geometric,
    sample_strength_cost_negative_binomial,
    sample_strength_cost_poisson,
    sample_strength_degree_binomial,
    sample_strength_degree_geometric,
    sample_strength_degree_negative_binomial,
    sample_strength_degree_poisson,
    sample_strength_edges_binomial,
    sample_strength_edges_geometric,
    sample_strength_edges_negative_binomial,
    sample_strength_edges_poisson,
    sample_strength_geometric,
    sample_strength_multinomial,
    sample_strength_negative_binomial,
    sample_strength_poisson,
    sample_strength_stub_matching,
)
from menobis.models.spec import (
    Constraint,
    Ensemble,
    ModelFamily,
    UnsupportedModelCaseError,
)
from menobis.models.types import (
    ConicDiagnostics,
    DegreeEventsFit,
    DegreeFit,
    FitResult,
    OptimizationDiagnostics,
    PartialFitResult,
    StrengthCostFit,
    StrengthDegreeFit,
    StrengthEdgesFit,
    StrengthFit,
)


def fit_model(*args: Any, **kwargs: Any) -> FitResult:  # noqa: ANN401
    """Fit a model through the unified router."""
    from menobis.routing import fit_model as route_fit_model

    return route_fit_model(*args, **kwargs)


def sample_model(*args: Any, **kwargs: Any) -> EdgeTable:  # noqa: ANN401
    """Sample a model through the unified router."""
    from menobis.routing import sample_model as route_sample_model

    return route_sample_model(*args, **kwargs)


__all__ = [
    "ConicDiagnostics",
    "Constraint",
    "DegreeEventsFit",
    "DegreeFit",
    "Ensemble",
    "FitResult",
    "ModelFamily",
    "OptimizationDiagnostics",
    "PartialFitResult",
    "StrengthCostFit",
    "StrengthDegreeFit",
    "StrengthEdgesFit",
    "StrengthFit",
    "UnsupportedModelCaseError",
    "fit_degree_bernoulli",
    "fit_degree_events_binomial",
    "fit_degree_events_geometric",
    "fit_degree_events_negative_binomial",
    "fit_degree_events_poisson",
    "fit_model",
    "fit_strength_binomial",
    "fit_strength_cost_binomial",
    "fit_strength_cost_geometric",
    "fit_strength_cost_negative_binomial",
    "fit_strength_cost_poisson",
    "fit_strength_degree_binomial",
    "fit_strength_degree_geometric",
    "fit_strength_degree_negative_binomial",
    "fit_strength_degree_poisson",
    "fit_strength_edges_binomial",
    "fit_strength_edges_geometric",
    "fit_strength_edges_negative_binomial",
    "fit_strength_edges_poisson",
    "fit_strength_geometric",
    "fit_strength_negative_binomial",
    "fit_strength_poisson",
    "sample_custom_multinomial",
    "sample_custom_poisson",
    "sample_degree_events_binomial",
    "sample_degree_events_geometric",
    "sample_degree_events_negative_binomial",
    "sample_degree_events_poisson",
    "sample_model",
    "sample_strength_binomial",
    "sample_strength_cost_binomial",
    "sample_strength_cost_geometric",
    "sample_strength_cost_negative_binomial",
    "sample_strength_cost_poisson",
    "sample_strength_degree_binomial",
    "sample_strength_degree_geometric",
    "sample_strength_degree_negative_binomial",
    "sample_strength_degree_poisson",
    "sample_strength_edges_binomial",
    "sample_strength_edges_geometric",
    "sample_strength_edges_negative_binomial",
    "sample_strength_edges_poisson",
    "sample_strength_geometric",
    "sample_strength_multinomial",
    "sample_strength_negative_binomial",
    "sample_strength_poisson",
    "sample_strength_stub_matching",
    "validate_strength_degree_constraints",
]
