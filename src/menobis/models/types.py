"""Result types for MENoBiS model fitting, filtering, and partial constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from menobis.data.frames import ProbabilityTable

if TYPE_CHECKING:
    from menobis.data.frames import EdgeTable
    from menobis.models.spec import Constraint, Ensemble, ModelFamily


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


class FitResult:
    """Shared read-only convenience properties for fitted result dataclasses."""

    diagnostics: OptimizationDiagnostics | None
    converged: bool
    self_loops: bool

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
class StrengthFit(FitResult):
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
class DegreeFit(FitResult):
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
class StrengthCostFit(FitResult):
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
class StrengthEdgesFit(FitResult):
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
class StrengthDegreeFit(FitResult):
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
class EdgesEventsFit(FitResult):
    """Fitted global EDGES_EVENTS multipliers (all families).

    The model is symmetric: every candidate pair shares one zero-inflated
    distribution with positive-support parameter `q` and global occupation
    multiplier `lam`.
    """

    q: float
    lam: float
    occupation: float
    positive_mean: float
    node_count: int
    self_loops: bool = True
    converged: bool = True
    iterations: int = 0
    family: str = "poisson"
    layers: int | None = None
    diagnostics: OptimizationDiagnostics | None = None


@dataclass(frozen=True)
class DegreeEventsFit(FitResult):
    """Fitting result for degree-events models (all families).

    The model factorizes into occupation (Bernoulli via x, y) and
    positive-occupation distribution parameterized by q.
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
class SamplingDiagnostics:
    """Diagnostics for a sampling run."""

    method: str
    exactness: str
    iterations: int | None = None
    accepted: int | None = None
    total_steps: int | None = None
    message: str | None = None
    # Strength-cost gamma-fit diagnostics (microcanonical only).
    gamma: float | None = None
    expected_cost: float | None = None
    expected_cost_standard_error: float | None = None
    observed_cost: float | None = None
    cost_residual: float | None = None
    converged: bool | None = None
    proposals: int | None = None


@dataclass(frozen=True)
class StrengthCostDiagnostics:
    """Gamma-fit diagnostics for microcanonical strength-cost sampling.

    Mirrors the Rust ``FixedStrengthCostFitResult`` fields exposed through
    the Python binding ``sample_fixed_strength_with_cost``.
    """

    gamma: float
    expected_cost_estimate: float
    expected_cost_standard_error: float
    observed_cost: float
    residual: float
    converged: bool
    iterations: int
    proposals: int
    accepted: int
    fixed_cost: float = 0.0


@dataclass(frozen=True)
class SamplingResult:
    """Detailed result of a sampling call.

    ``edges`` is the sampled occupied-pair table. ``exactness`` records the
    generation-method category (independent, direct, MCMC, approximate...).
    """

    edges: EdgeTable
    ensemble: Ensemble
    family: ModelFamily
    constraint: Constraint
    method: str
    exactness: str
    seed: int
    diagnostics: SamplingDiagnostics | None = None


@dataclass(frozen=True)
class PartialFitResult(FitResult):
    """Sparse intensity table from partial-constraint fitting with diagnostics."""

    source: NDArray[np.uint64]
    target: NDArray[np.uint64]
    intensity: NDArray[np.float64]
    constraint: str = "strength"
    family: str = "poisson"
    self_loops: bool = True
    converged: bool = True
    iterations: int = 0
    diagnostics: OptimizationDiagnostics | None = None

    def as_probability_table(self) -> ProbabilityTable:
        """Convert to ProbabilityTable for sampling (intensities as probabilities)."""
        return ProbabilityTable(
            source=self.source,
            target=self.target,
            probability=self.intensity,
        )


__all__ = [
    "ConicDiagnostics",
    "DegreeEventsFit",
    "DegreeFit",
    "EdgesEventsFit",
    "FitResult",
    "OptimizationDiagnostics",
    "PartialFitResult",
    "SamplingDiagnostics",
    "SamplingResult",
    "StrengthCostDiagnostics",
    "StrengthCostFit",
    "StrengthDegreeFit",
    "StrengthEdgesFit",
    "StrengthFit",
]
