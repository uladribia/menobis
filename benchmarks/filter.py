"""Null-filter calibration stage for PA geographic benchmarks."""

from __future__ import annotations

import time

from benchmarks.sample import sample_from_artifact
from benchmarks.types import BenchmarkOptions, BenchmarkRow, FitArtifact
from odme.data.frames import EdgeTable, ProbabilityTable
from odme.filtering import (
    FilterResult,
    _solve_ztp_rate,
    filter_custom_poisson,
    filter_degree_events_binomial,
    filter_degree_events_geometric,
    filter_degree_events_negative_binomial,
    filter_degree_events_poisson,
    filter_strength_binomial,
    filter_strength_cost_binomial,
    filter_strength_cost_geometric,
    filter_strength_cost_negative_binomial,
    filter_strength_cost_poisson,
    filter_strength_degree_geometric,
    filter_strength_degree_negative_binomial,
    filter_strength_degree_poisson,
    filter_strength_edges_geometric,
    filter_strength_edges_negative_binomial,
    filter_strength_edges_poisson,
    filter_strength_geometric,
    filter_strength_negative_binomial,
)


def filter_null_samples(artifacts: list[FitArtifact], options: BenchmarkOptions) -> list[BenchmarkRow]:
    """Sample from each fitted null and filter those null samples.

    This stage intentionally does not estimate FPR on the PA network itself.
    The PA network supplies constraints; calibration is assessed on samples from
    the fitted null model.
    """
    rows: list[BenchmarkRow] = []
    for artifact in artifacts:
        started = time.perf_counter()
        flagged = 0
        observed = 0
        try:
            for idx in range(options.filter_sample_count):
                sample = sample_from_artifact(artifact, seed=options.seed + 90_000 + idx)
                result = filter_against_artifact(sample, artifact, alpha=options.alpha)
                flagged += result.upper.edges.num_edges + result.lower.edges.num_edges
                observed += sample.num_edges
        except Exception as exc:  # noqa: BLE001 - benchmarks must continue.
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
            artifact.node_count if artifact.network.self_loops else artifact.node_count - 1
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
                sampled_edges_mean=float(observed / max(options.filter_sample_count, 1)),
            )
        )
    return rows


def filter_against_artifact(sample: EdgeTable, artifact: FitArtifact, *, alpha: float) -> FilterResult:
    """Dispatch filtering of one null sample against its fitted PA-constraint null."""
    fit = artifact.fit
    family = artifact.spec.family
    constraint = artifact.spec.constraint
    layers = int(getattr(fit, "layers", None) or artifact.spec.layers or 1)
    if constraint == "strength":
        if family == "me":
            source = sample.source.astype(int)
            target = sample.target.astype(int)
            rates = ProbabilityTable(
                source=sample.source,
                target=sample.target,
                probability=fit.x[source] * fit.y[target],
            )
            return filter_custom_poisson(sample, rates, alpha=alpha, tail="upper")
        if family == "b":
            return filter_strength_binomial(sample, fit.x, fit.y, layers=layers, alpha=alpha, tail="upper", self_loops=fit.self_loops)
        if family == "w":
            return filter_strength_geometric(sample, fit.x, fit.y, alpha=alpha, tail="upper", self_loops=fit.self_loops)
        if family == "wnb":
            return filter_strength_negative_binomial(
                sample, fit.x, fit.y, layers=layers, alpha=alpha, tail="upper", self_loops=fit.self_loops
            )
    if constraint == "strength-cost":
        cost_src, cost_tgt, cost_val = artifact.network.complete_cost_triples()
        if family == "me":
            return filter_strength_cost_poisson(sample, fit, cost_src, cost_tgt, cost_val, alpha=alpha, tail="upper")
        if family == "b":
            return filter_strength_cost_binomial(
                sample,
                fit,
                cost_src,
                cost_tgt,
                cost_val,
                layers=layers,
                alpha=alpha,
                tail="upper",
            )
        if family == "w":
            return filter_strength_cost_geometric(sample, fit, cost_src, cost_tgt, cost_val, alpha=alpha, tail="upper")
        if family == "wnb":
            return filter_strength_cost_negative_binomial(
                sample,
                fit,
                cost_src,
                cost_tgt,
                cost_val,
                layers=layers,
                alpha=alpha,
                tail="upper",
            )
    if constraint == "strength-edges":
        if family == "me":
            return filter_strength_edges_poisson(sample, fit, alpha=alpha, tail="upper")
        if family == "w":
            return filter_strength_edges_geometric(sample, fit, alpha=alpha, tail="upper")
        if family == "wnb":
            return filter_strength_edges_negative_binomial(sample, fit, layers=layers, alpha=alpha, tail="upper")
    if constraint == "strength-degree":
        if family == "me":
            return filter_strength_degree_poisson(sample, fit, alpha=alpha, tail="upper")
        if family == "w":
            return filter_strength_degree_geometric(sample, fit, alpha=alpha, tail="upper")
        if family == "wnb":
            return filter_strength_degree_negative_binomial(sample, fit, layers=layers, alpha=alpha, tail="upper")
    if constraint == "degree-events":
        if family == "me":
            return filter_degree_events_poisson(
                sample,
                fit.x,
                fit.y,
                _solve_ztp_rate(fit.positive_mean),
                alpha=alpha,
                tail="upper",
                self_loops=fit.self_loops,
            )
        if family == "b":
            return filter_degree_events_binomial(
                sample, fit.x, fit.y, fit.q, layers=layers, alpha=alpha, tail="upper", self_loops=fit.self_loops
            )
        if family == "w":
            return filter_degree_events_geometric(
                sample, fit, positive_weight_rate=fit.q, alpha=alpha, tail="upper", self_loops=fit.self_loops
            )
        if family == "wnb":
            return filter_degree_events_negative_binomial(
                sample,
                fit,
                positive_weight_rate=fit.q,
                layers=layers,
                alpha=alpha,
                tail="upper",
                self_loops=fit.self_loops,
            )
    msg = f"unsupported filter case: {artifact.spec.name}"
    raise ValueError(msg)
