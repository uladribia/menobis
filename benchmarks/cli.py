"""Canonical MENoBiS benchmark CLI using PA geographic synthetic networks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from benchmarks.dispatch import CONSTRAINTS, FAMILIES
from benchmarks.pipeline import (
    filter_null_samples,
    fit_cases,
    generate_cases,
    sample_and_check,
)
from benchmarks.types import BenchmarkOptions, BenchmarkRow

app = typer.Typer(help="MENoBiS benchmark suite.", no_args_is_help=True)

Stage = Literal["all", "generate", "fit", "sample", "filter"]


@app.command("run")
def run_command(
    stage: Annotated[Stage, typer.Argument(help="Stage to run.")] = "all",
    nodes: Annotated[
        str, typer.Option("--nodes", help="Comma-separated node sizes.")
    ] = "100,500",
    families: Annotated[
        str, typer.Option("--families", help="Comma-separated families: me,b,w,wnb.")
    ] = "me,b,w,wnb",
    constraints: Annotated[
        str,
        typer.Option("--constraints", help="Comma-separated constraints."),
    ] = ",".join(CONSTRAINTS),
    seed: Annotated[int, typer.Option("--seed", help="Base random seed.")] = 10_000,
    average_degree: Annotated[
        float,
        typer.Option(
            "--average-degree",
            help="PA support mean out-degree when density is omitted.",
        ),
    ] = 8.0,
    density: Annotated[
        float | None,
        typer.Option(
            "--density", help="Directed graph density. Overrides average degree."
        ),
    ] = None,
    events_per_edge: Annotated[
        float,
        typer.Option("--events-per-edge", help="Total events divided by binary edges."),
    ] = 6.0,
    tolerance_factor: Annotated[
        float,
        typer.Option("--tolerance-factor", help="Relative fitting tolerance factor."),
    ] = 0.02,
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", help="Maximum fitting iterations."),
    ] = 5_000,
    sample_count: Annotated[
        int, typer.Option("--samples", help="Samples per fitted model check.")
    ] = 5,
    filter_sample_count: Annotated[
        int,
        typer.Option("--filter-samples", help="Null samples per filter calibration."),
    ] = 5,
    alpha: Annotated[
        float, typer.Option("--alpha", help="Filter significance level.")
    ] = 0.05,
    self_loops: Annotated[
        bool,
        typer.Option(
            "--self-loops/--no-self-loops",
            help="Allow self-loops in generated networks and models.",
        ),
    ] = False,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory.")
    ] = Path("benchmarks/results/canonical"),
    output_json: Annotated[
        bool, typer.Option("--json", help="Emit summary JSON to stdout.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", help="Suppress progress table.")
    ] = False,
) -> None:
    """Run canonical generate → fit → sample → filter benchmarks."""
    rows = run_pipeline(
        BenchmarkOptions(
            nodes=tuple(_parse_ints(nodes)),
            families=tuple(_parse_tokens(families, FAMILIES, "family")),
            constraints=tuple(_parse_tokens(constraints, CONSTRAINTS, "constraint")),
            seed=seed,
            average_degree=average_degree,
            density=density,
            events_per_edge=events_per_edge,
            tolerance_factor=tolerance_factor,
            max_iterations=max_iterations,
            sample_count=sample_count,
            filter_sample_count=filter_sample_count,
            alpha=alpha,
            self_loops=self_loops,
        ),
        stage=stage,
        output=output,
    )
    if output_json:
        typer.echo(json.dumps([row.to_json() for row in rows], indent=2))
    elif not quiet:
        _print_rows(rows)
        typer.echo(f"Results: {output / 'summary.json'}", err=True)


def run_pipeline(
    options: BenchmarkOptions, *, stage: Stage, output: Path
) -> list[BenchmarkRow]:
    """Run the benchmark pipeline up to the requested stage."""
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[BenchmarkRow] = []
    generated, rows = generate_cases(options, output)
    all_rows.extend(rows)
    if stage == "generate":
        return _finalize(all_rows, output)
    artifacts, rows = fit_cases(generated, options)
    all_rows.extend(rows)
    if stage == "fit":
        return _finalize(all_rows, output)
    if stage in {"all", "sample"}:
        all_rows.extend(sample_and_check(artifacts, options))
    if stage in {"all", "filter"}:
        all_rows.extend(filter_null_samples(artifacts, options))
    return _finalize(all_rows, output)


def _finalize(rows: list[BenchmarkRow], output: Path) -> list[BenchmarkRow]:
    """Persist summary and per-stage JSONL logs."""
    (output / "summary.json").write_text(
        json.dumps([row.to_json() for row in rows], indent=2), encoding="utf-8"
    )
    by_stage: dict[str, list[BenchmarkRow]] = {}
    for row in rows:
        by_stage.setdefault(row.stage, []).append(row)
    for stage_name, stage_rows in by_stage.items():
        path = output / f"{stage_name.replace('-', '_')}_log.jsonl"
        path.write_text(
            "".join(json.dumps(row.to_json()) + "\n" for row in stage_rows),
            encoding="utf-8",
        )
    return rows


def _parse_ints(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        msg = "--nodes must include at least one integer"
        raise typer.BadParameter(msg)
    return result


def _parse_tokens(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    tokens = [part.strip().lower() for part in value.split(",") if part.strip()]
    bad = sorted(set(tokens) - set(allowed))
    if bad:
        msg = f"unknown {name}: {','.join(bad)}"
        raise typer.BadParameter(msg)
    return tokens


def _print_rows(rows: list[BenchmarkRow]) -> None:
    """Print a compact human summary."""
    header = f"{'stage':<13} {'N':>5} {'case':<25} {'status':<14} {'sec':>8} details"
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in rows:
        detail = row.message
        if row.stage == "sample-check" and row.max_strength_error is not None:
            detail = (
                f"s_err={row.max_strength_error:.2f} k_err={row.max_degree_error:.2f} "
                f"e_err={row.edge_count_error:.2f}"
            )
        if row.stage == "filter-null" and row.false_positive_rate is not None:
            detail = (
                f"fpr={row.false_positive_rate:.4f} edges≈{row.sampled_edges_mean:.1f}"
            )
        typer.echo(
            f"{row.stage:<13} {row.node_count:>5} {row.case:<25} "
            f"{row.status:<14} {row.seconds:>8.2f} {detail}"
        )


if __name__ == "__main__":
    app()
