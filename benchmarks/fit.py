"""Fit ODME models on PA geographic benchmark constraints."""

from __future__ import annotations

import time
from collections.abc import Iterable

from benchmarks.types import BenchmarkOptions, BenchmarkRow, CaseSpec, FitArtifact, GeneratedCase
from odme.models import (
    fit_degree_events_binomial,
    fit_degree_events_geometric,
    fit_degree_events_negative_binomial,
    fit_degree_events_poisson,
    fit_strength_binomial,
    fit_strength_cost_binomial_coordinates,
    fit_strength_cost_geometric_coordinates,
    fit_strength_cost_negative_binomial_coordinates,
    fit_strength_cost_poisson_coordinates,
    fit_strength_degree_binomial,
    fit_strength_degree_geometric,
    fit_strength_degree_negative_binomial,
    fit_strength_degree_poisson,
    fit_strength_edges_binomial,
    fit_strength_edges_geometric,
    fit_strength_edges_negative_binomial,
    fit_strength_edges_poisson,
    fit_strength_geometric,
    fit_strength_negative_binomial,
    fit_strength_poisson,
)

CONSTRAINTS = (
    "strength",
    "strength-cost",
    "strength-edges",
    "strength-degree",
    "degree-events",
)
FAMILIES = ("me", "b", "w", "wnb")


def case_specs(families: Iterable[str], constraints: Iterable[str], binomial_layers: int) -> list[CaseSpec]:
    """Build benchmark case specs, marking known-invalid cases as skipped."""
    specs: list[CaseSpec] = []
    for family in families:
        for constraint in constraints:
            skip_reason = None
            if family == "b" and constraint in {"strength-edges", "strength-degree"}:
                skip_reason = "known P5: Python wrapper calls ME kernel"
            layers = binomial_layers if family == "b" else 3 if family == "wnb" else None
            specs.append(CaseSpec(family=family, constraint=constraint, layers=layers, skip_reason=skip_reason))
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
        for spec in case_specs(options.families, options.constraints, constraints.binomial_layers):
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
            except Exception as exc:  # noqa: BLE001 - benchmarks must continue.
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
    c = generated.constraints
    network = generated.network
    strength_tol = options.tolerance_factor * max(float(c.strength_out.max()), 1.0)
    degree_tol = max(options.tolerance_factor * max(float(c.degree_out.max()), 1.0), 1.0)
    tolerance = max(strength_tol, 1.0) if spec.family in {"w", "wnb"} else strength_tol
    common = {"self_loops": options.self_loops, "tolerance": tolerance, "max_iterations": options.max_iterations}
    degree_common = {"self_loops": options.self_loops, "tolerance": degree_tol, "max_iterations": options.max_iterations}

    if spec.family == "me":
        if spec.constraint == "strength":
            return fit_strength_poisson(c.strength_out, c.strength_in, **common)
        if spec.constraint == "strength-cost":
            return fit_strength_cost_poisson_coordinates(
                c.strength_out, c.strength_in, network.x, network.y, c.total_cost, **common
            )
        if spec.constraint == "strength-edges":
            return fit_strength_edges_poisson(c.strength_out, c.strength_in, c.total_edges, **common)
        if spec.constraint == "strength-degree":
            return fit_strength_degree_poisson(
                c.strength_out, c.strength_in, c.degree_out, c.degree_in, **common
            )
        if spec.constraint == "degree-events":
            return fit_degree_events_poisson(c.degree_out, c.degree_in, c.total_events, **degree_common)
    if spec.family == "b":
        layers = int(spec.layers or c.binomial_layers)
        if spec.constraint == "strength":
            return fit_strength_binomial(c.strength_out, c.strength_in, layers=layers, **common)
        if spec.constraint == "strength-cost":
            return fit_strength_cost_binomial_coordinates(
                c.strength_out,
                c.strength_in,
                network.x,
                network.y,
                c.total_cost,
                layers=layers,
                **common,
            )
        if spec.constraint == "strength-edges":
            return fit_strength_edges_binomial(c.strength_out, c.strength_in, c.total_edges, layers=layers, **common)
        if spec.constraint == "strength-degree":
            return fit_strength_degree_binomial(
                c.strength_out,
                c.strength_in,
                c.degree_out,
                c.degree_in,
                layers=layers,
                **common,
            )
        if spec.constraint == "degree-events":
            return fit_degree_events_binomial(c.degree_out, c.degree_in, c.total_events, layers=layers, **degree_common)
    if spec.family == "w":
        if spec.constraint == "strength":
            return fit_strength_geometric(c.strength_out, c.strength_in, **common)
        if spec.constraint == "strength-cost":
            return fit_strength_cost_geometric_coordinates(
                c.strength_out, c.strength_in, network.x, network.y, c.total_cost, **common
            )
        if spec.constraint == "strength-edges":
            return fit_strength_edges_geometric(c.strength_out, c.strength_in, c.total_edges, **common)
        if spec.constraint == "strength-degree":
            return fit_strength_degree_geometric(
                c.strength_out, c.strength_in, c.degree_out, c.degree_in, **common
            )
        if spec.constraint == "degree-events":
            return fit_degree_events_geometric(c.degree_out, c.degree_in, c.total_events, **degree_common)
    if spec.family == "wnb":
        layers = int(spec.layers or 3)
        if spec.constraint == "strength":
            return fit_strength_negative_binomial(c.strength_out, c.strength_in, layers=layers, **common)
        if spec.constraint == "strength-cost":
            return fit_strength_cost_negative_binomial_coordinates(
                c.strength_out,
                c.strength_in,
                network.x,
                network.y,
                c.total_cost,
                layers=layers,
                **common,
            )
        if spec.constraint == "strength-edges":
            return fit_strength_edges_negative_binomial(
                c.strength_out, c.strength_in, c.total_edges, layers=layers, **common
            )
        if spec.constraint == "strength-degree":
            return fit_strength_degree_negative_binomial(
                c.strength_out,
                c.strength_in,
                c.degree_out,
                c.degree_in,
                layers=layers,
                **common,
            )
        if spec.constraint == "degree-events":
            return fit_degree_events_negative_binomial(
                c.degree_out, c.degree_in, c.total_events, layers=layers, **degree_common
            )
    msg = f"unsupported case: {spec.name}"
    raise ValueError(msg)
