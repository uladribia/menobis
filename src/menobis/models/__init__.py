"""Multi-edge model helpers for MENoBiS."""

from typing import Any

from menobis.data.frames import EdgeTable
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
    "fit_model",
    "sample_model",
]
