"""Composable analysis facade for MENoBiS networks.

All analysis paths use O(N + E) memory: per-node vectors plus the sparse
edge list. Clustering is excluded from the default pass because it builds
a sparse adjacency and is algorithmically distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from menobis.analysis.common import node_count
from menobis.analysis.graph_algorithms import (
    clustering_coefficient,
    occupation_clustering_coefficient,
)
from menobis.analysis.stats import compute_all_stats, occupation_distribution
from menobis.analysis.summary import directed_degrees, directed_strengths

if TYPE_CHECKING:
    from menobis.analysis.types import (
        ClusteringResult,
        DirectedSequences,
        NodeStats,
        OccupationDistribution,
    )
    from menobis.data.frames import EdgeTable


@dataclass(frozen=True)
class AnalysisResult:
    """Composable analysis result.

    Only requested metrics are populated; ``node_count`` is always set.
    """

    node_count: int
    strengths: DirectedSequences | None = None
    degrees: DirectedSequences | None = None
    stats: NodeStats | None = None
    distribution: OccupationDistribution | None = None
    clustering: ClusteringResult | None = None
    occupation_clustering: ClusteringResult | None = None
    _fields: frozenset[str] = field(default_factory=frozenset)

    @property
    def requested(self) -> frozenset[str]:
        """Names of the populated metrics."""
        return self._fields

    def __getitem__(self, name: str) -> object:
        """Return the named metric field."""
        return getattr(self, name)


def analyze(
    edges: EdgeTable,
    *,
    strengths: bool = False,
    degrees: bool = False,
    stats: bool = False,
    distribution: bool = False,
    clustering: bool = False,
    occupation_clustering: bool = False,
) -> AnalysisResult:
    """Compute the requested network metrics in one call.

    Args:
        edges: Occupied-pair table.
        strengths: Compute directed strengths.
        degrees: Compute directed degrees.
        stats: Compute the full per-node statistics pass.
        distribution: Compute the occupation-number distribution.
        clustering: Compute binary clustering coefficients (builds adjacency).
        occupation_clustering: Compute occupation-based clustering.

    Returns:
        AnalysisResult with the requested fields populated.
    """
    requested = frozenset(
        name
        for name, flag in (
            ("strengths", strengths),
            ("degrees", degrees),
            ("stats", stats),
            ("distribution", distribution),
            ("clustering", clustering),
            ("occupation_clustering", occupation_clustering),
        )
        if flag
    )
    nc = node_count(edges)
    result = AnalysisResult(node_count=nc, _fields=requested)

    if strengths:
        result = replace(result, strengths=directed_strengths(edges))
    if degrees:
        result = replace(result, degrees=directed_degrees(edges))
    if stats:
        result = replace(result, stats=compute_all_stats(edges))
    if distribution:
        result = replace(result, distribution=occupation_distribution(edges))
    if clustering:
        result = replace(result, clustering=clustering_coefficient(edges))
    if occupation_clustering:
        result = replace(
            result, occupation_clustering=occupation_clustering_coefficient(edges)
        )
    return result


__all__ = ["AnalysisResult", "analyze", "node_count"]
