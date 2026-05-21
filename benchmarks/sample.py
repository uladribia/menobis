"""Sample fitted benchmark models and compare recovered constraints."""

from __future__ import annotations

import time

import numpy as np

from benchmarks.check import max_pair_error, sample_degrees, sample_strengths
from benchmarks.types import BenchmarkOptions, BenchmarkRow, FitArtifact
from odme.data.frames import EdgeTable
from odme.models import (
    sample_degree_events_binomial,
    sample_degree_events_geometric,
    sample_degree_events_negative_binomial,
    sample_degree_events_poisson,
    sample_strength_binomial,
    sample_strength_cost_binomial,
    sample_strength_cost_geometric,
    sample_strength_cost_negative_binomial,
    sample_strength_cost_poisson,
    sample_strength_degree_binomial,
    sample_strength_degree_geometric,
    sample_strength_degree_negative_binomial,
    sample_strength_degree_poisson,
    sample_strength_edges_binomial,
    sample_strength_edges_geometric,
    sample_strength_edges_negative_binomial,
    sample_strength_edges_poisson,
    sample_strength_geometric,
    sample_strength_negative_binomial,
    sample_strength_poisson,
)


def sample_and_check(artifacts: list[FitArtifact], options: BenchmarkOptions) -> list[BenchmarkRow]:
    """Sample each fitted model and compare ensemble means to PA constraints."""
    rows: list[BenchmarkRow] = []
    for artifact in artifacts:
        started = time.perf_counter()
        try:
            samples = [sample_from_artifact(artifact, seed=options.seed + 50_000 + idx) for idx in range(options.sample_count)]
        except Exception as exc:  # noqa: BLE001 - benchmarks must continue.
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
        strength_out, strength_in, degree_out, degree_in = _ensemble_means(samples, artifact.node_count)
        c = artifact.constraints
        strength_error = max_pair_error(strength_out, strength_in, c.strength_out, c.strength_in)
        degree_error = max_pair_error(degree_out, degree_in, c.degree_out, c.degree_in)
        edge_error = abs(float(np.mean([sample.num_edges for sample in samples])) - c.total_edges)
        event_error = abs(float(np.mean([sample.total_events for sample in samples])) - c.total_events)
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
                sampled_edges_mean=float(np.mean([sample.num_edges for sample in samples])),
            )
        )
    return rows


def sample_from_artifact(artifact: FitArtifact, *, seed: int) -> EdgeTable:
    """Dispatch sampling for one fitted artifact."""
    fit = artifact.fit
    family = artifact.spec.family
    constraint = artifact.spec.constraint
    layers = int(getattr(fit, "layers", None) or artifact.spec.layers or 1)
    if constraint == "strength":
        if family == "me":
            return sample_strength_poisson(fit.x, fit.y, self_loops=fit.self_loops, seed=seed)
        if family == "b":
            return sample_strength_binomial(fit.x, fit.y, layers=layers, self_loops=fit.self_loops, seed=seed)
        if family == "w":
            return sample_strength_geometric(fit.x, fit.y, self_loops=fit.self_loops, seed=seed)
        if family == "wnb":
            return sample_strength_negative_binomial(fit.x, fit.y, layers=layers, self_loops=fit.self_loops, seed=seed)
    if constraint == "strength-cost":
        cost_src, cost_tgt, cost_val = artifact.network.complete_cost_triples()
        if family == "me":
            return sample_strength_cost_poisson(fit, cost_src, cost_tgt, cost_val, seed=seed)
        if family == "b":
            return sample_strength_cost_binomial(fit, cost_src, cost_tgt, cost_val, layers=layers, seed=seed)
        if family == "w":
            return sample_strength_cost_geometric(fit, cost_src, cost_tgt, cost_val, seed=seed)
        if family == "wnb":
            return sample_strength_cost_negative_binomial(fit, cost_src, cost_tgt, cost_val, layers=layers, seed=seed)
    if constraint == "strength-edges":
        if family == "me":
            return sample_strength_edges_poisson(fit, seed=seed)
        if family == "b":
            return sample_strength_edges_binomial(fit, layers=layers, seed=seed)
        if family == "w":
            return sample_strength_edges_geometric(fit, seed=seed)
        if family == "wnb":
            return sample_strength_edges_negative_binomial(fit, layers=layers, seed=seed)
    if constraint == "strength-degree":
        if family == "me":
            return sample_strength_degree_poisson(fit, seed=seed)
        if family == "b":
            return sample_strength_degree_binomial(fit, layers=layers, seed=seed)
        if family == "w":
            return sample_strength_degree_geometric(fit, seed=seed)
        if family == "wnb":
            return sample_strength_degree_negative_binomial(fit, layers=layers, seed=seed)
    if constraint == "degree-events":
        if family == "me":
            return sample_degree_events_poisson(
                fit, total_events=artifact.constraints.total_events, seed=seed, self_loops=fit.self_loops
            )
        if family == "b":
            return sample_degree_events_binomial(fit, layers=layers, seed=seed)
        if family == "w":
            return sample_degree_events_geometric(fit, seed=seed)
        if family == "wnb":
            return sample_degree_events_negative_binomial(fit, layers=layers, seed=seed)
    msg = f"unsupported sample case: {artifact.spec.name}"
    raise ValueError(msg)


def _ensemble_means(
    samples: list[EdgeTable], node_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not samples:
        zeros = np.zeros(node_count, dtype=np.float64)
        return zeros, zeros, zeros, zeros
    strength_out = []
    strength_in = []
    degree_out = []
    degree_in = []
    for sample in samples:
        s_out, s_in = sample_strengths(sample, node_count)
        k_out, k_in = sample_degrees(sample, node_count)
        strength_out.append(s_out)
        strength_in.append(s_in)
        degree_out.append(k_out)
        degree_in.append(k_in)
    return (
        np.mean(strength_out, axis=0),
        np.mean(strength_in, axis=0),
        np.mean(degree_out, axis=0),
        np.mean(degree_in, axis=0),
    )
