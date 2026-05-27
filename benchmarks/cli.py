"""Modern MENoBiS end-to-end benchmarks.

The benchmark uses the PA geographic synthetic generator and exercises the modern
Rust/Python stack only: derive constraints, fit a null model, sample one null
network, and estimate upper-tail filtering false-positive rate from null samples.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import typer

from menobis.analysis import directed_strengths
from menobis.data.frames import EdgeTable
from menobis.filtering import (
    filter_strength_binomial,
    filter_strength_cost_binomial,
    filter_strength_cost_geometric,
    filter_strength_cost_poisson,
    filter_strength_geometric,
    filter_strength_poisson,
)
from menobis.models import (
    fit_strength_binomial,
    fit_strength_cost_binomial,
    fit_strength_cost_geometric,
    fit_strength_cost_poisson,
    fit_strength_geometric,
    fit_strength_poisson,
    sample_strength_binomial,
    sample_strength_cost_binomial,
    sample_strength_cost_geometric,
    sample_strength_cost_poisson,
    sample_strength_geometric,
    sample_strength_poisson,
)
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

app = typer.Typer(help="Modern MENoBiS PA-geographic E2E benchmarks.")

Family = Literal["me", "b", "w"]
Constraint = Literal["strength", "strength-cost"]
FAMILIES: tuple[str, ...] = ("me", "b", "w")
CONSTRAINTS: tuple[str, ...] = ("strength", "strength-cost")


@dataclass(frozen=True)
class BenchmarkRow:
    """One benchmark measurement row."""

    stage: str
    node_count: int
    family: str
    constraint: str
    self_loops: bool
    seconds: float
    status: str
    sampled_edges: int | None = None
    max_strength_error: float | None = None
    false_positive_rate: float | None = None
    message: str = ""


@app.command("all")
def all_command(
    nodes: Annotated[str, typer.Option("--nodes", help="Comma-separated node sizes.")] = "100,500",
    families: Annotated[
        str, typer.Option("--families", help="Comma-separated families: me,b,w.")
    ] = "me,b,w",
    constraints: Annotated[
        str,
        typer.Option("--constraints", help="Comma-separated constraints: strength,strength-cost."),
    ] = "strength,strength-cost",
    seed: Annotated[int, typer.Option("--seed", help="Base random seed.")] = 10_000,
    average_degree: Annotated[
        float, typer.Option("--average-degree", help="Mean synthetic support out-degree.")
    ] = 8.0,
    events_per_edge: Annotated[
        float, typer.Option("--events-per-edge", help="Mean observed weight per edge.")
    ] = 6.0,
    filter_samples: Annotated[
        int, typer.Option("--filter-samples", help="Null samples for FPR estimate.")
    ] = 3,
    alpha: Annotated[float, typer.Option("--alpha", help="Upper-tail alpha.")] = 0.05,
    self_loops: Annotated[
        bool, typer.Option("--self-loops/--no-self-loops", help="Allow self-loops.")
    ] = False,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output JSON path.")
    ] = Path("benchmarks/results/e2e-modern.json"),
    output_json: Annotated[bool, typer.Option("--json", help="Print JSON rows.")] = False,
) -> None:
    """Run generate → fit → sample-check → filter-FPR benchmarks."""
    rows = run_benchmark(
        nodes=tuple(_parse_ints(nodes)),
        families=tuple(_parse_tokens(families, FAMILIES, "family")),
        constraints=tuple(_parse_tokens(constraints, CONSTRAINTS, "constraint")),
        seed=seed,
        average_degree=average_degree,
        events_per_edge=events_per_edge,
        filter_samples=filter_samples,
        alpha=alpha,
        self_loops=self_loops,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if output_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        _print_rows(rows)
        typer.echo(f"Results: {output}", err=True)


@app.command("fit")
def fit_command(
    nodes: Annotated[str, typer.Option("--nodes", help="Comma-separated node sizes.")] = "100,500",
    families: Annotated[
        str, typer.Option("--families", help="Comma-separated families: me,b,w.")
    ] = "me,b,w",
    seed: Annotated[int, typer.Option("--seed", help="Base random seed.")] = 10_000,
    self_loops: Annotated[
        bool, typer.Option("--self-loops/--no-self-loops", help="Allow self-loops.")
    ] = False,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output JSON path.")
    ] = Path("benchmarks/results/fit-modern.json"),
) -> None:
    """Run only the fit stage for strength and strength-cost cases."""
    rows = [
        row
        for row in run_benchmark(
            nodes=tuple(_parse_ints(nodes)),
            families=tuple(_parse_tokens(families, FAMILIES, "family")),
            constraints=CONSTRAINTS,
            seed=seed,
            average_degree=8.0,
            events_per_edge=6.0,
            filter_samples=0,
            alpha=0.05,
            self_loops=self_loops,
            fit_only=True,
        )
        if row.stage in {"generate", "fit"}
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    _print_rows(rows)


def run_benchmark(
    *,
    nodes: tuple[int, ...],
    families: tuple[str, ...],
    constraints: tuple[str, ...],
    seed: int,
    average_degree: float,
    events_per_edge: float,
    filter_samples: int,
    alpha: float,
    self_loops: bool,
    fit_only: bool = False,
) -> list[BenchmarkRow]:
    """Run the modern benchmark matrix."""
    rows: list[BenchmarkRow] = []
    for n_index, node_count in enumerate(nodes):
        start = time.perf_counter()
        network = generate_pa_geographic_network(
            node_count,
            average_degree=average_degree,
            events_per_edge=events_per_edge,
            seed=seed + n_index,
            self_loops=self_loops,
        )
        constraints_data = derive_synthetic_constraints(network)
        rows.append(
            BenchmarkRow(
                stage="generate",
                node_count=node_count,
                family="all",
                constraint="pa-geographic",
                self_loops=self_loops,
                seconds=time.perf_counter() - start,
                status="ok",
                sampled_edges=network.edges.num_edges,
                message=f"events={network.edges.total_events}",
            )
        )
        for family in families:
            for constraint in constraints:
                fit, fit_row = _fit_case(family, constraint, network, constraints_data)
                rows.append(fit_row)
                if fit_only or fit_row.status != "ok":
                    continue
                sample, sample_row = _sample_case(family, constraint, fit, network, constraints_data, seed)
                rows.append(sample_row)
                if filter_samples > 0 and sample_row.status == "ok":
                    rows.append(
                        _filter_fpr_case(
                            family,
                            constraint,
                            fit,
                            network,
                            constraints_data,
                            filter_samples=filter_samples,
                            alpha=alpha,
                            seed=seed,
                        )
                    )
    return rows


def _fit_case(family: str, constraint: str, network, constraints_data) -> tuple[object, BenchmarkRow]:
    start = time.perf_counter()
    fit = None
    status = "ok"
    message = ""
    try:
        if constraint == "strength":
            fit = _fit_strength(family, constraints_data, network.self_loops)
        elif constraint == "strength-cost":
            fit = _fit_strength_cost(family, network, constraints_data)
        else:
            raise ValueError(f"unknown constraint: {constraint}")
    except Exception as exc:  # pragma: no cover - benchmark diagnostics
        status = "error"
        message = f"{type(exc).__name__}: {exc}"
    return fit, BenchmarkRow(
        stage="fit",
        node_count=len(network.x),
        family=family,
        constraint=constraint,
        self_loops=network.self_loops,
        seconds=time.perf_counter() - start,
        status=status,
        message=message,
    )


def _fit_strength(family: str, constraints_data, self_loops: bool) -> object:
    if family == "me":
        return fit_strength_poisson(
            constraints_data.strength_out, constraints_data.strength_in, self_loops=self_loops
        )
    if family == "b":
        return fit_strength_binomial(
            constraints_data.strength_out,
            constraints_data.strength_in,
            layers=constraints_data.binomial_layers,
            self_loops=self_loops,
        )
    return fit_strength_geometric(
        constraints_data.strength_out, constraints_data.strength_in, self_loops=self_loops
    )


def _fit_strength_cost(family: str, network, constraints_data) -> object:
    if family == "me":
        return fit_strength_cost_poisson(
            constraints_data.strength_out,
            constraints_data.strength_in,
            network.x,
            network.y,
            constraints_data.total_cost,
            self_loops=network.self_loops,
        )
    if family == "b":
        return fit_strength_cost_binomial(
            constraints_data.strength_out,
            constraints_data.strength_in,
            network.x,
            network.y,
            constraints_data.total_cost,
            layers=constraints_data.binomial_layers,
            self_loops=network.self_loops,
        )
    return fit_strength_cost_geometric(
        constraints_data.strength_out,
        constraints_data.strength_in,
        network.x,
        network.y,
        constraints_data.total_cost,
        self_loops=network.self_loops,
    )


def _sample_case(family: str, constraint: str, fit: object, network, constraints_data, seed: int) -> tuple[EdgeTable | None, BenchmarkRow]:
    start = time.perf_counter()
    sample = None
    status = "ok"
    message = ""
    max_strength_error = None
    try:
        sample = _sample(family, constraint, fit, network, constraints_data, seed)
        sampled_strength = directed_strengths(sample)
        max_strength_error = float(
            max(
                np.max(np.abs(sampled_strength.out - constraints_data.strength_out), initial=0.0),
                np.max(np.abs(sampled_strength.incoming - constraints_data.strength_in), initial=0.0),
            )
        )
    except Exception as exc:  # pragma: no cover - benchmark diagnostics
        status = "error"
        message = f"{type(exc).__name__}: {exc}"
    return sample, BenchmarkRow(
        stage="sample-check",
        node_count=len(network.x),
        family=family,
        constraint=constraint,
        self_loops=network.self_loops,
        seconds=time.perf_counter() - start,
        status=status,
        sampled_edges=None if sample is None else sample.num_edges,
        max_strength_error=max_strength_error,
        message=message,
    )


def _sample(family: str, constraint: str, fit: object, network, constraints_data, seed: int) -> EdgeTable:
    if constraint == "strength":
        if family == "me":
            return sample_strength_poisson(fit.x, fit.y, self_loops=fit.self_loops, seed=seed)
        if family == "b":
            return sample_strength_binomial(
                fit.x, fit.y, layers=constraints_data.binomial_layers, self_loops=fit.self_loops, seed=seed
            )
        return sample_strength_geometric(fit.x, fit.y, self_loops=fit.self_loops, seed=seed)
    if family == "me":
        return sample_strength_cost_poisson(fit, network.x, network.y, seed=seed)
    if family == "b":
        return sample_strength_cost_binomial(
            fit, network.x, network.y, layers=constraints_data.binomial_layers, seed=seed
        )
    return sample_strength_cost_geometric(fit, network.x, network.y, seed=seed)


def _filter_fpr_case(
    family: str,
    constraint: str,
    fit: object,
    network,
    constraints_data,
    *,
    filter_samples: int,
    alpha: float,
    seed: int,
) -> BenchmarkRow:
    start = time.perf_counter()
    false_positives = 0
    tested_edges = 0
    status = "ok"
    message = ""
    try:
        for offset in range(filter_samples):
            sample = _sample(family, constraint, fit, network, constraints_data, seed + 10_000 + offset)
            result = _filter(family, constraint, sample, fit, network, constraints_data, alpha)
            false_positives += result.upper.edges.num_edges
            tested_edges += sample.num_edges
        fpr = false_positives / max(tested_edges, 1)
    except Exception as exc:  # pragma: no cover - benchmark diagnostics
        status = "error"
        message = f"{type(exc).__name__}: {exc}"
        fpr = None
    return BenchmarkRow(
        stage="filter-fpr",
        node_count=len(network.x),
        family=family,
        constraint=constraint,
        self_loops=network.self_loops,
        seconds=time.perf_counter() - start,
        status=status,
        false_positive_rate=fpr,
        message=message,
    )


def _filter(family: str, constraint: str, sample: EdgeTable, fit: object, network, constraints_data, alpha: float):
    if constraint == "strength":
        if family == "me":
            return filter_strength_poisson(sample, fit, alpha=alpha, tail="upper")
        if family == "b":
            return filter_strength_binomial(
                sample, fit, layers=constraints_data.binomial_layers, alpha=alpha, tail="upper"
            )
        return filter_strength_geometric(sample, fit, alpha=alpha, tail="upper")
    if family == "me":
        return filter_strength_cost_poisson(sample, fit, network.x, network.y, alpha=alpha, tail="upper")
    if family == "b":
        return filter_strength_cost_binomial(
            sample, fit, network.x, network.y, layers=constraints_data.binomial_layers, alpha=alpha, tail="upper"
        )
    return filter_strength_cost_geometric(sample, fit, network.x, network.y, alpha=alpha, tail="upper")


def _parse_ints(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise typer.BadParameter("expected at least one node count")
    return result


def _parse_tokens(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    result = [part.strip().lower() for part in value.split(",") if part.strip()]
    bad = sorted(set(result) - set(allowed))
    if bad:
        raise typer.BadParameter(f"unknown {name}: {','.join(bad)}")
    return result


def _print_rows(rows: list[BenchmarkRow]) -> None:
    header = f"{'stage':<13} {'N':>7} {'family':<4} {'constraint':<14} {'status':<7} {'sec':>8} details"
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in rows:
        detail = row.message
        if row.stage == "generate":
            detail = f"edges={row.sampled_edges} {row.message}"
        elif row.stage == "sample-check":
            detail = f"edges={row.sampled_edges} max_s_err={row.max_strength_error:.2f}"
        elif row.stage == "filter-fpr" and row.false_positive_rate is not None:
            detail = f"fpr={row.false_positive_rate:.4f}"
        typer.echo(
            f"{row.stage:<13} {row.node_count:>7} {row.family:<4} {row.constraint:<14} "
            f"{row.status:<7} {row.seconds:>8.3f} {detail}"
        )


if __name__ == "__main__":
    app()
