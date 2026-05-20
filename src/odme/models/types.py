"""Result types for ODME model fitting, filtering, and partial constraints."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from odme.data.frames import ProbabilityTable


@dataclass(frozen=True)
class ConicDiagnostics:
    """Lifted conic solver diagnostics for W fitting problems."""

    min_margin: float
    max_q: float
    variables: int
    auxiliary_variables: int
    exponential_cones: int
    power_cones: int
    linear_constraints: int
    sparse_nonzeros: int


@dataclass(frozen=True)
class OptimizationDiagnostics:
    """Shared optimization diagnostics for fitted models."""

    converged: bool
    status: str
    iterations: int
    objective: float | None = None
    max_strength_residual: float | None = None
    total_strength_residual: float | None = None
    cost_residual: float | None = None
    conic: ConicDiagnostics | None = None


@dataclass(frozen=True)
class FitResult:
    """Fitted strength/degree Lagrange multipliers."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    self_loops: bool = True
    converged: bool = True
    iterations: int = 0
    family: str = "poisson"
    layers: int | None = None
    diagnostics: OptimizationDiagnostics | None = None

    @property
    def status(self) -> str:
        """Optimization status string."""
        if self.diagnostics is not None:
            return self.diagnostics.status
        return "solved" if self.converged else "failed"

    @property
    def objective(self) -> float | None:
        """Optimization objective, when available."""
        return None if self.diagnostics is None else self.diagnostics.objective

    @property
    def min_margin(self) -> float | None:
        """Minimum W inverse/log margin, when conic diagnostics are available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.min_margin

    @property
    def max_q(self) -> float | None:
        """Maximum W pair parameter q, when conic diagnostics are available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.max_q

    @property
    def max_strength_residual(self) -> float | None:
        """Maximum absolute strength residual, when available."""
        if self.diagnostics is None:
            return None
        return self.diagnostics.max_strength_residual

    @property
    def total_strength_residual(self) -> float | None:
        """Total absolute strength residual, when available."""
        if self.diagnostics is None:
            return None
        return self.diagnostics.total_strength_residual

    @property
    def variables(self) -> int | None:
        """Number of original conic variables, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.variables

    @property
    def auxiliary_variables(self) -> int | None:
        """Number of auxiliary conic variables, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.auxiliary_variables

    @property
    def exponential_cones(self) -> int | None:
        """Number of exponential cones, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.exponential_cones

    @property
    def power_cones(self) -> int | None:
        """Number of power cones, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.power_cones

    @property
    def linear_constraints(self) -> int | None:
        """Number of lifted linear constraints, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.linear_constraints

    @property
    def sparse_nonzeros(self) -> int | None:
        """Number of lifted sparse nonzeros, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.sparse_nonzeros


@dataclass(frozen=True)
class StrengthCostFit:
    """Fitted strength-cost model for Poisson, geometric, or binomial families."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    gamma: float
    self_loops: bool
    converged: bool
    iterations: int
    family: str = "poisson"
    layers: int | None = None
    diagnostics: OptimizationDiagnostics | None = None

    @property
    def status(self) -> str:
        """Optimization status string."""
        if self.diagnostics is not None:
            return self.diagnostics.status
        return "solved" if self.converged else "failed"

    @property
    def objective(self) -> float | None:
        """Optimization objective, when available."""
        return None if self.diagnostics is None else self.diagnostics.objective

    @property
    def min_margin(self) -> float | None:
        """Minimum W inverse/log margin, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.min_margin

    @property
    def max_q(self) -> float | None:
        """Maximum W pair parameter q, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.max_q

    @property
    def max_strength_residual(self) -> float | None:
        """Maximum absolute strength residual, when available."""
        if self.diagnostics is None:
            return None
        return self.diagnostics.max_strength_residual

    @property
    def total_strength_residual(self) -> float | None:
        """Total absolute strength residual, when available."""
        if self.diagnostics is None:
            return None
        return self.diagnostics.total_strength_residual

    @property
    def cost_residual(self) -> float | None:
        """Cost residual, when available."""
        if self.diagnostics is None:
            return None
        return self.diagnostics.cost_residual

    @property
    def variables(self) -> int | None:
        """Number of original conic variables, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.variables

    @property
    def auxiliary_variables(self) -> int | None:
        """Number of auxiliary conic variables, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.auxiliary_variables

    @property
    def exponential_cones(self) -> int | None:
        """Number of exponential cones, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.exponential_cones

    @property
    def power_cones(self) -> int | None:
        """Number of power cones, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.power_cones

    @property
    def linear_constraints(self) -> int | None:
        """Number of lifted linear constraints, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.linear_constraints

    @property
    def sparse_nonzeros(self) -> int | None:
        """Number of lifted sparse nonzeros, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.sparse_nonzeros


# Internal compatibility aliases while callers migrate to constraint types.
WStrengthFit = FitResult
WStrengthCostFit = StrengthCostFit


@dataclass(frozen=True)
class StrengthEdgesFit:
    """Fitted fixed-strength-and-edge-count model."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    lam: float
    self_loops: bool
    converged: bool
    iterations: int
    family: str = "poisson"
    layers: int | None = None
    diagnostics: OptimizationDiagnostics | None = None

    @property
    def status(self) -> str:
        """Optimization status string."""
        if self.diagnostics is not None:
            return self.diagnostics.status
        return "solved" if self.converged else "failed"

    @property
    def objective(self) -> float | None:
        """Optimization objective, when available."""
        return None if self.diagnostics is None else self.diagnostics.objective

    @property
    def min_margin(self) -> float | None:
        """Minimum W inverse/log margin, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.min_margin

    @property
    def max_q(self) -> float | None:
        """Maximum W pair parameter q, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.max_q

    @property
    def max_strength_residual(self) -> float | None:
        """Maximum absolute strength residual, when available."""
        if self.diagnostics is None:
            return None
        return self.diagnostics.max_strength_residual

    @property
    def total_strength_residual(self) -> float | None:
        """Total absolute strength residual, when available."""
        if self.diagnostics is None:
            return None
        return self.diagnostics.total_strength_residual

    @property
    def variables(self) -> int | None:
        """Number of original conic variables, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.variables

    @property
    def auxiliary_variables(self) -> int | None:
        """Number of auxiliary conic variables, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.auxiliary_variables

    @property
    def exponential_cones(self) -> int | None:
        """Number of exponential cones, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.exponential_cones

    @property
    def power_cones(self) -> int | None:
        """Number of power cones, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.power_cones

    @property
    def linear_constraints(self) -> int | None:
        """Number of lifted linear constraints, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.linear_constraints

    @property
    def sparse_nonzeros(self) -> int | None:
        """Number of lifted sparse nonzeros, when available."""
        if self.diagnostics is None or self.diagnostics.conic is None:
            return None
        return self.diagnostics.conic.sparse_nonzeros


@dataclass(frozen=True)
class StrengthDegreeFit:
    """Fitted fixed-strength-degree model."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    w: NDArray[np.float64]
    self_loops: bool
    converged: bool
    iterations: int
    family: str = "poisson"
    layers: int | None = None
    diagnostics: OptimizationDiagnostics | None = None


@dataclass(frozen=True)
class DegreeEventsFit:
    """Fitting result for degree-events models (all families).

    The model factorizes into occupation (Bernoulli via x, y) and
    positive-weight distribution parameterized by q.
    """

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    q: float
    positive_mean: float
    self_loops: bool
    converged: bool = True
    iterations: int = 0
    family: str = "geometric"
    layers: int | None = None
    diagnostics: OptimizationDiagnostics | None = None


@dataclass(frozen=True)
class PartialFitResult:
    """Combined rate table from partial-constraint fitting."""

    source: NDArray[np.uint64]
    target: NDArray[np.uint64]
    rate: NDArray[np.float64]

    def as_probability_table(self) -> ProbabilityTable:
        """Convert to ProbabilityTable for sampling (rates as weights)."""
        return ProbabilityTable(
            source=self.source,
            target=self.target,
            probability=self.rate,
        )


__all__ = [
    "ConicDiagnostics",
    "DegreeEventsFit",
    "FitResult",
    "OptimizationDiagnostics",
    "PartialFitResult",
    "StrengthCostFit",
    "StrengthDegreeFit",
    "StrengthEdgesFit",
]
