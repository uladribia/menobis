"""Result types for ODME model fitting, filtering, and partial constraints."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from odme.data.frames import ProbabilityTable


@dataclass(frozen=True)
class FitResult:
    """Lagrange multiplier fitting result."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    converged: bool = True
    iterations: int = 0


@dataclass(frozen=True)
class StrengthCostFit:
    """Fitted strength-cost ME model: E[t_ij] = x_i y_j exp(-gamma d_ij)."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    gamma: float
    self_loops: bool
    converged: bool
    iterations: int


@dataclass(frozen=True)
class StrengthEdgesFit:
    """Fitted exact ME fixed-strength-and-edge-count model."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    lam: float
    self_loops: bool
    converged: bool
    iterations: int


@dataclass(frozen=True)
class StrengthDegreeFit:
    """Fitted exact ME fixed-strength-degree model."""

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    w: NDArray[np.float64]
    self_loops: bool
    converged: bool
    iterations: int


@dataclass(frozen=True)
class DegreeEventsFit:
    """Fitting result for degree-events W models (geometric/negative binomial).

    The model factorizes into occupation (Bernoulli via x, y) and
    positive-weight distribution (geometric or negative binomial with q).
    """

    node: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    q: float
    positive_mean: float
    converged: bool = True
    iterations: int = 0


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
    "DegreeEventsFit",
    "FitResult",
    "PartialFitResult",
    "StrengthCostFit",
    "StrengthDegreeFit",
    "StrengthEdgesFit",
]
