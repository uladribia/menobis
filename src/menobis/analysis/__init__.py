"""Analysis routines for MENoBiS non-binary networks."""

from menobis.analysis.common import node_count
from menobis.analysis.ensemble import (
    ensemble_average,
    ensemble_scalar_average,
)
from menobis.analysis.facade import AnalysisResult, analyze
from menobis.analysis.graph_algorithms import (
    clustering_coefficient,
    occupation_clustering_coefficient,
)
from menobis.analysis.stats import (
    compute_all_stats,
    occupation_distribution,
)
from menobis.analysis.summary import (
    directed_degrees,
    directed_strengths,
)
from menobis.analysis.types import (
    ClusteringResult,
    DirectedSequences,
    NodeStats,
    OccupationDistribution,
)

__all__ = [
    "AnalysisResult",
    "ClusteringResult",
    "DirectedSequences",
    "NodeStats",
    "OccupationDistribution",
    "analyze",
    "clustering_coefficient",
    "compute_all_stats",
    "directed_degrees",
    "directed_strengths",
    "ensemble_average",
    "ensemble_scalar_average",
    "node_count",
    "occupation_clustering_coefficient",
    "occupation_distribution",
]
