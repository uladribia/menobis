"""Generate canonical PA geographic benchmark networks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.types import BenchmarkOptions, BenchmarkRow, GeneratedCase
from odme.data.frames import EdgeTable
from odme.synthetic import (
    SyntheticNetwork,
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)


def generate_cases(options: BenchmarkOptions, output: Path) -> tuple[list[GeneratedCase], list[BenchmarkRow]]:
    """Generate and persist one PA geographic network for each requested N."""
    network_dir = output / "networks"
    network_dir.mkdir(parents=True, exist_ok=True)
    cases: list[GeneratedCase] = []
    rows: list[BenchmarkRow] = []
    for node_count in options.nodes:
        network = generate_pa_geographic_network(
            node_count,
            density=options.density,
            average_degree=options.average_degree,
            events_per_edge=options.events_per_edge,
            seed=options.seed + node_count,
            self_loops=options.self_loops,
        )
        constraints = derive_synthetic_constraints(network)
        cases.append(GeneratedCase(node_count, network, constraints))
        save_network(network, network_dir / f"n{node_count}.npz")
        rows.append(
            BenchmarkRow(
                stage="generate",
                node_count=node_count,
                case="pa-geographic",
                family=None,
                constraint=None,
                status="ok",
                message=(
                    f"edges={network.edges.num_edges} events={network.edges.total_events} "
                    f"max_strength={max(constraints.strength_out.max(), constraints.strength_in.max()):.0f}"
                ),
                sampled_edges_mean=float(network.edges.num_edges),
            )
        )
    return cases, rows


def save_network(network: SyntheticNetwork, path: Path) -> None:
    """Save a generated network as a compressed npz file."""
    np.savez_compressed(
        path,
        source=network.edges.source,
        target=network.edges.target,
        weight=network.edges.weight,
        x=network.x,
        y=network.y,
        support_out_degree=network.support_out_degree,
        support_in_degree=network.support_in_degree,
        self_loops=np.array([network.self_loops]),
        distance_decay=np.array([network.distance_decay]),
        distance_scale=np.array([network.distance_scale]),
        degree_attractiveness=np.array([network.degree_attractiveness]),
        origin_degree_exponent=np.array([network.origin_degree_exponent]),
        destination_degree_exponent=np.array([network.destination_degree_exponent]),
    )


def load_network(path: Path) -> SyntheticNetwork:
    """Load a network saved by :func:`save_network`."""
    data = np.load(path)
    return SyntheticNetwork(
        edges=EdgeTable(
            source=data["source"].astype(np.uint64),
            target=data["target"].astype(np.uint64),
            weight=data["weight"].astype(np.uint64),
        ),
        x=data["x"].astype(np.float64),
        y=data["y"].astype(np.float64),
        support_out_degree=data["support_out_degree"].astype(np.float64),
        support_in_degree=data["support_in_degree"].astype(np.float64),
        self_loops=bool(data["self_loops"][0]),
        distance_decay=float(data["distance_decay"][0]),
        distance_scale=float(data["distance_scale"][0]),
        degree_attractiveness=float(data["degree_attractiveness"][0]),
        origin_degree_exponent=float(data["origin_degree_exponent"][0]),
        destination_degree_exponent=float(data["destination_degree_exponent"][0]),
    )
