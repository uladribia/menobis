"""Canonical synthetic networks for ODME tests and benchmarks.

The generator intentionally does *not* draw from an ODME null model. It builds a
heterogeneous directed binary support with preferential attachment, assigns
projected XY coordinates, and normalizes geographic degree-driven weights to a
fixed total number of events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from odme.data.frames import EdgeTable


@dataclass(frozen=True)
class SyntheticNetwork:
    """Generated PA geographic weighted directed network."""

    edges: EdgeTable
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    support_out_degree: NDArray[np.float64]
    support_in_degree: NDArray[np.float64]
    self_loops: bool
    distance_decay: float
    distance_scale: float
    degree_attractiveness: float
    origin_degree_exponent: float
    destination_degree_exponent: float

    def edge_distances(self) -> NDArray[np.float64]:
        """Return Euclidean distance for each occupied edge."""
        source = self.edges.source.astype(np.int64)
        target = self.edges.target.astype(np.int64)
        return np.hypot(
            self.x[source] - self.x[target], self.y[source] - self.y[target]
        )

    def edge_scores(self) -> NDArray[np.float64]:
        """Return unnormalized degree-distance weight scores for occupied edges."""
        source = self.edges.source.astype(np.int64)
        target = self.edges.target.astype(np.int64)
        degree_part = (
            self.support_out_degree[source] + self.degree_attractiveness
        ) ** self.origin_degree_exponent * (
            self.support_in_degree[target] + self.degree_attractiveness
        ) ** self.destination_degree_exponent
        distance_part = np.exp(
            -self.distance_decay
            * self.edge_distances()
            / max(self.distance_scale, 1e-12)
        )
        return degree_part * distance_part

    def complete_cost_triples(
        self,
    ) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.float64]]:
        """Return complete Euclidean cost triples for model APIs."""
        node_count = len(self.x)
        source, target = np.meshgrid(
            np.arange(node_count, dtype=np.uint64),
            np.arange(node_count, dtype=np.uint64),
            indexing="ij",
        )
        mask = np.ones((node_count, node_count), dtype=bool)
        if not self.self_loops:
            np.fill_diagonal(mask, False)
        distance = np.hypot(
            self.x[source.astype(np.int64)] - self.x[target.astype(np.int64)],
            self.y[source.astype(np.int64)] - self.y[target.astype(np.int64)],
        )
        return source[mask], target[mask], distance[mask].astype(np.float64)


@dataclass(frozen=True)
class SyntheticConstraints:
    """Constraints derived exactly from a synthetic network."""

    strength_out: NDArray[np.float64]
    strength_in: NDArray[np.float64]
    degree_out: NDArray[np.float64]
    degree_in: NDArray[np.float64]
    total_edges: float
    total_events: int
    total_cost: float
    binomial_layers: int


def generate_pa_geographic_network(
    node_count: int,
    *,
    density: float | None = None,
    average_degree: float | None = 8.0,
    events_per_edge: float = 8.0,
    total_events: int | None = None,
    seed: int = 0,
    self_loops: bool = False,
    coordinate_scale: float = 1.0,
    distance_decay: float = 2.0,
    degree_attractiveness: float = 1.0,
    origin_degree_exponent: float = 1.0,
    destination_degree_exponent: float = 1.0,
) -> SyntheticNetwork:
    """Generate a PA geographic weighted directed network.

    Binary support is created by directed preferential attachment. Positive
    integer weights are then allocated only on existing edges with probabilities
    proportional to origin out-degree, destination in-degree, and
    ``exp(-distance_decay * distance / distance_scale)``.

    Args:
        node_count: Number of nodes.
        density: Target directed density. Overrides ``average_degree`` when set.
        average_degree: Target mean out-degree when ``density`` is omitted.
        events_per_edge: Mean positive weight used when ``total_events`` is omitted.
        total_events: Exact total weight sum. Must be at least the edge count.
        seed: Random seed.
        self_loops: Whether self-loops are allowed in the support.
        coordinate_scale: Width/height of the square coordinate domain.
        distance_decay: Geographic damping strength for weights.
        degree_attractiveness: Additive degree offset used in weight scores.
        origin_degree_exponent: Exponent for origin support out-degree.
        destination_degree_exponent: Exponent for destination support in-degree.

    Returns:
        SyntheticNetwork with exact edge count and total event count.
    """
    if node_count <= 0:
        msg = "node_count must be positive"
        raise ValueError(msg)
    if density is not None and not (0.0 <= density <= 1.0):
        msg = "density must be in [0, 1]"
        raise ValueError(msg)
    if average_degree is not None and average_degree < 0.0:
        msg = "average_degree must be non-negative"
        raise ValueError(msg)
    if events_per_edge < 1.0:
        msg = "events_per_edge must be at least 1.0"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, coordinate_scale, size=node_count).astype(np.float64)
    y = rng.uniform(0.0, coordinate_scale, size=node_count).astype(np.float64)
    edge_count = _target_edge_count(
        node_count,
        density=density,
        average_degree=average_degree,
        self_loops=self_loops,
    )
    sources, targets, out_degree, in_degree = _preferential_support(
        node_count, edge_count, self_loops=self_loops, rng=rng
    )

    distances = np.hypot(x[sources] - x[targets], y[sources] - y[targets])
    positive_distances = distances[distances > 0.0]
    distance_scale = (
        float(np.median(positive_distances)) if positive_distances.size else 1.0
    )
    scores = (
        (out_degree[sources] + degree_attractiveness) ** origin_degree_exponent
        * (in_degree[targets] + degree_attractiveness) ** destination_degree_exponent
        * np.exp(-distance_decay * distances / max(distance_scale, 1e-12))
    )
    weights = _allocate_positive_integer_weights(
        scores,
        edge_count=edge_count,
        total_events=total_events,
        events_per_edge=events_per_edge,
        rng=rng,
    )
    order = np.lexsort((targets, sources))
    edges = EdgeTable(
        source=sources[order].astype(np.uint64),
        target=targets[order].astype(np.uint64),
        weight=weights[order].astype(np.uint64),
    )
    return SyntheticNetwork(
        edges=edges,
        x=x,
        y=y,
        support_out_degree=out_degree.astype(np.float64),
        support_in_degree=in_degree.astype(np.float64),
        self_loops=self_loops,
        distance_decay=distance_decay,
        distance_scale=distance_scale,
        degree_attractiveness=degree_attractiveness,
        origin_degree_exponent=origin_degree_exponent,
        destination_degree_exponent=destination_degree_exponent,
    )


def derive_synthetic_constraints(network: SyntheticNetwork) -> SyntheticConstraints:
    """Derive ODME benchmark constraints from a generated network."""
    node_count = len(network.x)
    strength_out = np.zeros(node_count, dtype=np.float64)
    strength_in = np.zeros(node_count, dtype=np.float64)
    degree_out = np.zeros(node_count, dtype=np.float64)
    degree_in = np.zeros(node_count, dtype=np.float64)
    source = network.edges.source.astype(np.int64)
    target = network.edges.target.astype(np.int64)
    weight = network.edges.weight.astype(np.float64)
    np.add.at(strength_out, source, weight)
    np.add.at(strength_in, target, weight)
    np.add.at(degree_out, source, 1.0)
    np.add.at(degree_in, target, 1.0)
    distances = network.edge_distances()
    total_cost = float(np.sum(weight * distances))
    pairs_per_node = node_count if network.self_loops else max(node_count - 1, 1)
    max_strength = float(
        max(strength_out.max(initial=0.0), strength_in.max(initial=0.0))
    )
    binomial_layers = max(10, 4 * int(np.ceil(max_strength / pairs_per_node)))
    return SyntheticConstraints(
        strength_out=strength_out,
        strength_in=strength_in,
        degree_out=degree_out,
        degree_in=degree_in,
        total_edges=float(network.edges.num_edges),
        total_events=network.edges.total_events,
        total_cost=total_cost,
        binomial_layers=binomial_layers,
    )


def known_pairs_from_network(
    network: SyntheticNetwork,
    *,
    fraction: float = 0.15,
) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.float64]]:
    """Select deterministic known weighted pairs from strongest observed edges."""
    if not (0.0 <= fraction <= 1.0):
        msg = "fraction must be in [0, 1]"
        raise ValueError(msg)
    count = min(
        network.edges.num_edges, max(0, round(fraction * network.edges.num_edges))
    )
    if count == 0:
        empty_u = np.array([], dtype=np.uint64)
        return empty_u, empty_u, np.array([], dtype=np.float64)
    order = np.argsort(network.edges.weight)[::-1][:count]
    return (
        network.edges.source[order].astype(np.uint64),
        network.edges.target[order].astype(np.uint64),
        network.edges.weight[order].astype(np.float64),
    )


def _target_edge_count(
    node_count: int,
    *,
    density: float | None,
    average_degree: float | None,
    self_loops: bool,
) -> int:
    candidate_count = node_count * (
        node_count if self_loops else max(node_count - 1, 0)
    )
    if density is not None:
        target = round(density * candidate_count)
    else:
        avg = 0.0 if average_degree is None else average_degree
        target = round(avg * node_count)
    return min(candidate_count, max(0, target))


def _preferential_support(
    node_count: int,
    edge_count: int,
    *,
    self_loops: bool,
    rng: np.random.Generator,
) -> tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]
]:
    edges: set[tuple[int, int]] = set()
    out_degree = np.zeros(node_count, dtype=np.float64)
    in_degree = np.zeros(node_count, dtype=np.float64)
    source_urn = list(range(node_count))
    target_urn = list(range(node_count))

    def add_edge(source: int, target: int) -> None:
        edges.add((source, target))
        out_degree[source] += 1.0
        in_degree[target] += 1.0
        source_urn.append(source)
        target_urn.append(target)

    for source in range(node_count):
        if len(edges) >= edge_count:
            break
        target = (source + 1) % node_count
        if self_loops or source != target:
            add_edge(source, target)

    attempts = 0
    max_attempts = max(10_000, edge_count * 20)
    while len(edges) < edge_count and attempts < max_attempts:
        attempts += 1
        source = source_urn[int(rng.integers(0, len(source_urn)))]
        target = target_urn[int(rng.integers(0, len(target_urn)))]
        if (not self_loops and source == target) or (source, target) in edges:
            continue
        add_edge(source, target)

    while len(edges) < edge_count:
        remaining = _remaining_pairs(node_count, edges, self_loops=self_loops)
        if not remaining:
            break
        weights = np.array(
            [(out_degree[s] + 1.0) * (in_degree[t] + 1.0) for s, t in remaining],
            dtype=np.float64,
        )
        probabilities = weights / weights.sum()
        source, target = remaining[int(rng.choice(len(remaining), p=probabilities))]
        add_edge(source, target)

    ordered = np.array(sorted(edges), dtype=np.int64)
    if ordered.size == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            out_degree,
            in_degree,
        )
    return ordered[:, 0], ordered[:, 1], out_degree, in_degree


def _remaining_pairs(
    node_count: int,
    edges: set[tuple[int, int]],
    *,
    self_loops: bool,
) -> list[tuple[int, int]]:
    return [
        (source, target)
        for source in range(node_count)
        for target in range(node_count)
        if (self_loops or source != target) and (source, target) not in edges
    ]


def _allocate_positive_integer_weights(
    scores: NDArray[np.float64],
    *,
    edge_count: int,
    total_events: int | None,
    events_per_edge: float,
    rng: np.random.Generator,
) -> NDArray[np.uint64]:
    if edge_count == 0:
        return np.array([], dtype=np.uint64)
    target_events = (
        round(edge_count * events_per_edge) if total_events is None else total_events
    )
    if target_events < edge_count:
        msg = "total_events must be at least the binary edge count"
        raise ValueError(msg)
    probabilities = (
        scores / scores.sum()
        if scores.sum() > 0.0
        else np.full(edge_count, 1.0 / edge_count)
    )
    extra = rng.multinomial(target_events - edge_count, probabilities)
    return (extra + 1).astype(np.uint64)


__all__ = [
    "SyntheticConstraints",
    "SyntheticNetwork",
    "derive_synthetic_constraints",
    "generate_pa_geographic_network",
    "known_pairs_from_network",
]
