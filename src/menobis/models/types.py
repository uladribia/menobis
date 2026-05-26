"""Result types for MENoBiS model fitting, filtering, and partial constraints."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from menobis.data.frames import ProbabilityTable


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


class _OptimizationView:
    """Shared read-only convenience properties for fitted result dataclasses."""

    diagnostics: OptimizationDiagnostics | None
    converged: bool

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
        conic = None if self.diagnostics is None else self.diagnostics.conic
        return None if conic is None else conic.min_margin

    @property
    def max_q(self) -> float | None:
        """Maximum W pair parameter q, when conic diagnostics are available."""
        conic = None if self.diagnostics is None else self.diagnostics.conic
        return None if conic is None else conic.max_q

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
        conic = None if self.diagnostics is None else self.diagnostics.conic
        return None if conic is None else conic.variables

    @property
    def auxiliary_variables(self) -> int | None:
        """Number of auxiliary conic variables, when available."""
        conic = None if self.diagnostics is None else self.diagnostics.conic
        return None if conic is None else conic.auxiliary_variables

    @property
    def exponential_cones(self) -> int | None:
        """Number of exponential cones, when available."""
        conic = None if self.diagnostics is None else self.diagnostics.conic
        return None if conic is None else conic.exponential_cones

    @property
    def power_cones(self) -> int | None:
        """Number of power cones, when available."""
        conic = None if self.diagnostics is None else self.diagnostics.conic
        return None if conic is None else conic.power_cones

    @property
    def linear_constraints(self) -> int | None:
        """Number of lifted linear constraints, when available."""
        conic = None if self.diagnostics is None else self.diagnostics.conic
        return None if conic is None else conic.linear_constraints

    @property
    def sparse_nonzeros(self) -> int | None:
        """Number of lifted sparse nonzeros, when available."""
        conic = None if self.diagnostics is None else self.diagnostics.conic
        return None if conic is None else conic.sparse_nonzeros


@dataclass(frozen=True)
class StrengthFit(_OptimizationView):
    """Fitted fixed-strength model with node Lagrange multipliers."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    self_loops: bool = True
    converged: bool = True
    iterations: int = 0
    family: str = "poisson"
    layers: int | None = None
    diagnostics: OptimizationDiagnostics | None = None


@dataclass(frozen=True)
class DegreeFit(_OptimizationView):
    """Fitted fixed-degree Bernoulli model with node multipliers."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    self_loops: bool = True
    converged: bool = True
    iterations: int = 0
    family: str = "bernoulli"
    layers: int | None = None
    diagnostics: OptimizationDiagnostics | None = None


@dataclass(frozen=True)
class StrengthCostFit(_OptimizationView):
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


@dataclass(frozen=True)
class StrengthEdgesFit(_OptimizationView):
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


@dataclass(frozen=True)
class StrengthDegreeFit(_OptimizationView):
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
class DegreeEventsFit(_OptimizationView):
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
class PartialFitResult(_OptimizationView):
    """Rate table from partial-constraint fitting with diagnostics."""

    source: NDArray[np.uint64]
    target: NDArray[np.uint64]
    rate: NDArray[np.float64]
    constraint: str = "strength"
    family: str = "poisson"
    self_loops: bool = True
    converged: bool = True
    iterations: int = 0
    diagnostics: OptimizationDiagnostics | None = None

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
    "DegreeFit",
    "OptimizationDiagnostics",
    "PartialFitResult",
    "StrengthCostFit",
    "StrengthDegreeFit",
    "StrengthEdgesFit",
    "StrengthFit",
]
