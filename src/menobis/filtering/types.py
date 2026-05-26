"""Result types for MENoBiS statistical filtering."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from menobis.data.frames import EdgeTable

Tail = Literal["upper", "lower", "two-sided"]
Correction = Literal["none", "bonferroni", "fdr"]


@dataclass(frozen=True)
class FilteredEdges:
    """Filtered edge table plus p-values and null expectations."""

    edges: EdgeTable
    upper_pvalue: NDArray[np.float64]
    lower_pvalue: NDArray[np.float64]
    expected: NDArray[np.float64]
    occupation: NDArray[np.float64]


@dataclass(frozen=True)
class FilterResult:
    """Statistical filtering result for observed and absent edges."""

    upper: FilteredEdges
    lower: FilteredEdges
    compatible: FilteredEdges
    absent_lower: FilteredEdges
    alpha: float
    tail: Tail
    correction: Correction


__all__ = [
    "Correction",
    "FilterResult",
    "FilteredEdges",
    "Tail",
]
