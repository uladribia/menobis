"""Benchmark pipeline stages: generate, fit, sample, filter.

This module consolidates the four pipeline stages and constraint-check helpers
into a single cohesive unit that uses the dispatch registry.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from benchmarks.dispatch import (
    REGISTRY,
    build_fit_kwargs,
    build_sample_kwargs,
    filter_one,
)
from benchmarks.types import (
    BenchmarkOptions,
    BenchmarkRow,
    CaseSpec,
    FitArtifact,
    GeneratedCase,
)
from menobis.data.frames import EdgeTable
from menobis.utilities.synthetic import (
    SyntheticNetwork,
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

# ---------------------------------------------------------------------------
# Constraint checks (formerly check.py)
# ---------------------------------------------------------------------------


def sample_strengths(sample: EdgeTable, node_count: int) -> tuple[NDArray, NDArray]:
    """Return outgoing and incoming weighted strengths."""
    out = np.zeros(node_count, dtype=np.float64)
    incoming = np.zeros(node_count, dtype=np.float64)
    np.add.at(out, sample.source.astype(np.int64), sample.weight.astype(np.float64))
    np.add.at(
        incoming, sample.target.astype(np.int64), sample.weight.astype(np.float64)
    )
    return out, incoming


def sample_degrees(sample: EdgeTable, node_count: int) -> tuple[NDArray, NDArray]:
    """Return outgoing and incoming binary degrees."""
    out = np.zeros(node_count, dtype=np.float64)
    incoming = np.zeros(node_count, dtype=np.float64)
    np.add.at(out, sample.source.astype(np.int64), 1.0)
    np.add.at(incoming, sample.target.astype(np.int64), 1.0)
    return out, incoming


def max_pair_error(
    actual_out: NDArray,
    actual_in: NDArray,
    expected_out: NDArray,
    expected_in: NDArray,
) -> float:
    """Return max absolute in/out sequence error."""
    return float(
        max(
            np.max(np.abs(actual_out - expected_out), initial=0.0),
            np.max(np.abs(actual_in - expected_in), initial=0.0),
        )
    )


# ---------------------------------------------------------------------------
# Generate stage
# ---------------------------------------------------------------------------


def generate_cases(
    options: BenchmarkOptions, output: Path
) -> tuple[list[GeneratedCase], list[BenchmarkRow]]:
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
        _save_network(network, network_dir / f"n{node_count}.npz")
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


# ---------------------------------------------------------------------------
# Fit stage
# ---------------------------------------------------------------------------


def case_specs(
    families: Iterable[str], constraints: Iterable[str], binomial_layers: int
) -> list[CaseSpec]:
    """Build benchmark case specs."""
    specs: list[CaseSpec] = []
    for family in families:
        for constraint in constraints:
            skip_reason = None
            layers = (
                binomial_layers if family == "b" else 3 if family == "wnb" else None
            )
            specs.append(
                CaseSpec(
                    family=family,
                    constraint=constraint,
                    layers=layers,
                    skip_reason=skip_reason,
                )
            )
    return specs


def fit_cases(
    generated: list[GeneratedCase],
    options: BenchmarkOptions,
) -> tuple[list[FitArtifact], list[BenchmarkRow]]:
    """Fit requested cases and return successful artifacts plus log rows."""
    artifacts: list[FitArtifact] = []
    rows: list[BenchmarkRow] = []
    for generated_case in generated:
        constraints = generated_case.constraints
        for spec in case_specs(
            options.families, options.constraints, constraints.binomial_layers
        ):
            if spec.skip_reason is not None:
                rows.append(
                    BenchmarkRow(
                        stage="fit",
                        node_count=generated_case.node_count,
                        case=spec.name,
                        family=spec.family,
                        constraint=spec.constraint,
                        status="skipped",
                        message=spec.skip_reason,
                    )
                )
                continue
            started = time.perf_counter()
            try:
                fit = _fit_one(generated_case, spec, options)
            except Exception as exc:
                rows.append(
                    BenchmarkRow(
                        stage="fit",
                        node_count=generated_case.node_count,
                        case=spec.name,
                        family=spec.family,
                        constraint=spec.constraint,
                        status="error",
                        seconds=time.perf_counter() - started,
                        message=str(exc)[:200],
                    )
                )
                continue
            seconds = time.perf_counter() - started
            converged = bool(getattr(fit, "converged", True))
            status = "ok" if converged else "not_converged"
            rows.append(
                BenchmarkRow(
                    stage="fit",
                    node_count=generated_case.node_count,
                    case=spec.name,
                    family=spec.family,
                    constraint=spec.constraint,
                    status=status,
                    seconds=seconds,
                    converged=converged,
                    iterations=int(getattr(fit, "iterations", 0)),
                )
            )
            if converged:
                artifacts.append(
                    FitArtifact(
                        node_count=generated_case.node_count,
                        spec=spec,
                        fit=fit,
                        network=generated_case.network,
                        constraints=constraints,
                        seconds=seconds,
                    )
                )
    return artifacts, rows


def _fit_one(generated: GeneratedCase, spec: CaseSpec, options: BenchmarkOptions):
    """Fit a single case using the dispatch registry."""
    c = generated.constraints
    strength_tol = options.tolerance_factor * max(float(c.strength_out.max()), 1.0)
    degree_tol = max(
        options.tolerance_factor * max(float(c.degree_out.max()), 1.0), 1.0
    )
    tolerance = max(strength_tol, 1.0) if spec.family in {"w", "wnb"} else strength_tol

    dispatch = REGISTRY.get((spec.family, spec.constraint))
    if dispatch is None:
        msg = f"unsupported case: {spec.name}"
        raise ValueError(msg)

    kwargs = build_fit_kwargs(
        spec.family,
        spec.constraint,
        c,
        generated.network,
        self_loops=options.self_loops,
        tolerance=tolerance,
        degree_tolerance=degree_tol,
        max_iterations=options.max_iterations,
        layers=spec.layers,
    )
    return dispatch.fit(**kwargs)


# ---------------------------------------------------------------------------
# Sample stage
# ---------------------------------------------------------------------------


def sample_and_check(
    artifacts: list[FitArtifact], options: BenchmarkOptions
) -> list[BenchmarkRow]:
    """Sample each fitted model and compare ensemble means to PA constraints."""
    rows: list[BenchmarkRow] = []
    for artifact in artifacts:
        started = time.perf_counter()
        try:
            samples = [
                _sample_one(artifact, seed=options.seed + 50_000 + idx)
                for idx in range(options.sample_count)
            ]
        except Exception as exc:
            rows.append(
                BenchmarkRow(
                    stage="sample-check",
                    node_count=artifact.node_count,
                    case=artifact.spec.name,
                    family=artifact.spec.family,
                    constraint=artifact.spec.constraint,
                    status="error",
                    seconds=time.perf_counter() - started,
                    message=str(exc)[:200],
                )
            )
            continue
        strength_out, strength_in, degree_out, degree_in = _ensemble_means(
            samples, artifact.node_count
        )
        c = artifact.constraints
        strength_error = max_pair_error(
            strength_out, strength_in, c.strength_out, c.strength_in
        )
        degree_error = max_pair_error(degree_out, degree_in, c.degree_out, c.degree_in)
        edge_error = abs(float(np.mean([s.num_edges for s in samples])) - c.total_edges)
        event_error = abs(
            float(np.mean([s.total_events for s in samples])) - c.total_events
        )
        rows.append(
            BenchmarkRow(
                stage="sample-check",
                node_count=artifact.node_count,
                case=artifact.spec.name,
                family=artifact.spec.family,
                constraint=artifact.spec.constraint,
                status="ok",
                seconds=time.perf_counter() - started,
                max_strength_error=strength_error,
                max_degree_error=degree_error,
                edge_count_error=edge_error,
                total_event_error=event_error,
                sampled_edges_mean=float(np.mean([s.num_edges for s in samples])),
            )
        )
    return rows


def _sample_one(artifact: FitArtifact, *, seed: int) -> EdgeTable:
    """Sample one network from a fitted artifact using the dispatch registry."""
    spec = artifact.spec
    layers = int(getattr(artifact.fit, "layers", None) or spec.layers or 1)

    dispatch = REGISTRY.get((spec.family, spec.constraint))
    if dispatch is None:
        msg = f"unsupported sample case: {spec.name}"
        raise ValueError(msg)

    kwargs = build_sample_kwargs(
        spec.family,
        spec.constraint,
        artifact.fit,
        artifact.network,
        artifact.constraints,
        layers=layers,
        seed=seed,
    )
    return dispatch.sample(**kwargs)


def _ensemble_means(
    samples: list[EdgeTable], node_count: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute ensemble mean strengths and degrees across samples."""
    if not samples:
        zeros = np.zeros(node_count, dtype=np.float64)
        return zeros, zeros, zeros, zeros
    s_out_list, s_in_list, k_out_list, k_in_list = [], [], [], []
    for sample in samples:
        s_out, s_in = sample_strengths(sample, node_count)
        k_out, k_in = sample_degrees(sample, node_count)
        s_out_list.append(s_out)
        s_in_list.append(s_in)
        k_out_list.append(k_out)
        k_in_list.append(k_in)
    return (
        np.mean(s_out_list, axis=0),
        np.mean(s_in_list, axis=0),
        np.mean(k_out_list, axis=0),
        np.mean(k_in_list, axis=0),
    )


# ---------------------------------------------------------------------------
# Filter stage
# ---------------------------------------------------------------------------


def filter_null_samples(
    artifacts: list[FitArtifact], options: BenchmarkOptions
) -> list[BenchmarkRow]:
    """Sample from each fitted null and filter those null samples."""
    rows: list[BenchmarkRow] = []
    for artifact in artifacts:
        started = time.perf_counter()
        flagged = 0
        observed = 0
        try:
            for idx in range(options.filter_sample_count):
                sample = _sample_one(artifact, seed=options.seed + 90_000 + idx)
                layers = int(
                    getattr(artifact.fit, "layers", None) or artifact.spec.layers or 1
                )
                result = filter_one(
                    artifact.spec.family,
                    artifact.spec.constraint,
                    sample,
                    artifact.fit,
                    artifact.network,
                    layers=layers,
                    alpha=options.alpha,
                )
                flagged += result.upper.edges.num_edges + result.lower.edges.num_edges
                observed += sample.num_edges
        except Exception as exc:
            rows.append(
                BenchmarkRow(
                    stage="filter-null",
                    node_count=artifact.node_count,
                    case=artifact.spec.name,
                    family=artifact.spec.family,
                    constraint=artifact.spec.constraint,
                    status="error",
                    seconds=time.perf_counter() - started,
                    message=str(exc)[:200],
                )
            )
            continue
        candidate_pairs = artifact.node_count * (
            artifact.node_count
            if artifact.network.self_loops
            else artifact.node_count - 1
        )
        denominator = max(candidate_pairs * options.filter_sample_count, 1)
        fpr = flagged / denominator
        rows.append(
            BenchmarkRow(
                stage="filter-null",
                node_count=artifact.node_count,
                case=artifact.spec.name,
                family=artifact.spec.family,
                constraint=artifact.spec.constraint,
                status="ok",
                seconds=time.perf_counter() - started,
                false_positive_rate=fpr,
                sampled_edges_mean=float(
                    observed / max(options.filter_sample_count, 1)
                ),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Network I/O helpers
# ---------------------------------------------------------------------------


def _save_network(network: SyntheticNetwork, path: Path) -> None:
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
    """Load a network saved by :func:`_save_network`."""
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
