"""Shared types for the canonical ODME benchmark pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from odme.utilities.synthetic import SyntheticConstraints, SyntheticNetwork


@dataclass(frozen=True)
class BenchmarkOptions:
    """Runtime options for the benchmark pipeline."""

    nodes: tuple[int, ...]
    families: tuple[str, ...]
    constraints: tuple[str, ...]
    seed: int
    average_degree: float
    density: float | None
    events_per_edge: float
    tolerance_factor: float
    max_iterations: int
    sample_count: int
    filter_sample_count: int
    alpha: float
    self_loops: bool


@dataclass(frozen=True)
class GeneratedCase:
    """One generated synthetic network and its constraints."""

    node_count: int
    network: SyntheticNetwork
    constraints: SyntheticConstraints


@dataclass(frozen=True)
class CaseSpec:
    """One family/constraint fitting case."""

    family: str
    constraint: str
    layers: int | None = None
    skip_reason: str | None = None

    @property
    def name(self) -> str:
        """Stable case name."""
        return f"{self.family}-{self.constraint}"


@dataclass
class FitArtifact:
    """A fitted model plus metadata needed for later stages."""

    node_count: int
    spec: CaseSpec
    fit: Any
    network: SyntheticNetwork
    constraints: SyntheticConstraints
    seconds: float


@dataclass(frozen=True)
class BenchmarkRow:
    """Serializable benchmark event row."""

    stage: str
    node_count: int
    case: str
    family: str | None
    constraint: str | None
    status: str
    seconds: float = 0.0
    message: str = ""
    converged: bool | None = None
    iterations: int | None = None
    max_strength_error: float | None = None
    max_degree_error: float | None = None
    edge_count_error: float | None = None
    total_event_error: float | None = None
    false_positive_rate: float | None = None
    sampled_edges_mean: float | None = None

    def to_json(self) -> dict[str, object]:
        """Convert row to a compact JSON dictionary."""
        return {key: value for key, value in asdict(self).items() if value is not None}
