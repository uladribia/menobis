"""Typer commands for MENoBiS statistical filtering."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pyarrow.csv as pa_csv
import typer
from typer import Option, Typer

from menobis.data.frames import ProbabilityTable
from menobis.data.io import read_edges
from menobis.filtering import (
    Correction,
    FilteredEdges,
    FilterResult,
    Tail,
    filter_custom_poisson,
    filter_degree_events_poisson,
    filter_strength_cost_poisson,
    filter_strength_degree_poisson,
    filter_strength_edges_poisson,
)
from menobis.models.routing import (
    Constraint,
    Family,
    filter_model,
    fit_model,
)

app = Typer(no_args_is_help=True)


def _read_rate_table(path: Path) -> ProbabilityTable:
    table = pa_csv.read_csv(path)
    column = "rate" if "rate" in table.column_names else "probability"
    return ProbabilityTable(
        source=table.column("source").to_numpy().astype(np.uint64),
        target=table.column("target").to_numpy().astype(np.uint64),
        probability=table.column(column).to_numpy().astype(np.float64),
    )


@app.command("custom-poisson")
def custom_rates(
    input_path: Path,
    rates_path: Annotated[
        Path, Option("--rates", help="CSV with source,target,rate columns.")
    ],
    output_prefix: Annotated[
        Path, Option("--output-prefix", help="Prefix/directory for output CSV files.")
    ],
    alpha: Annotated[float, Option("--alpha", help="Significance level.")] = 0.05,
    tail: Annotated[
        Tail, Option("--tail", help="upper, lower, or two-sided.")
    ] = "two-sided",
    correction: Annotated[
        Correction, Option("--correction", help="none, bonferroni, or fdr.")
    ] = "none",
    detect_absent: Annotated[
        bool, Option("--detect-absent", help="Detect significant absent pairs.")
    ] = False,
    min_occupation: Annotated[
        float, Option("--min-occupation", help="Absent-edge occupation threshold.")
    ] = 0.5,
    min_expected: Annotated[
        float,
        Option("--min-expected", help="Optional absent expected-weight threshold."),
    ] = 0.0,
    max_absent: Annotated[
        int | None, Option("--max-absent", help="Maximum absent pairs to output.")
    ] = None,
) -> None:
    """Filter against user-supplied Poisson rates t_ij = T p_ij."""
    result = filter_custom_poisson(
        read_edges(input_path),
        _read_rate_table(rates_path),
        alpha=alpha,
        tail=tail,
        correction=correction,
        detect_absent=detect_absent,
        min_occupation=min_occupation,
        min_expected=min_expected,
        max_absent=max_absent,
    )
    _write_outputs(result, output_prefix)


@app.command("strength-poisson")
def fixed_strength(
    input_path: Path,
    output_prefix: Annotated[
        Path, Option("--output-prefix", help="Prefix/directory for output CSV files.")
    ],
    alpha: Annotated[float, Option("--alpha", help="Significance level.")] = 0.05,
    tail: Annotated[
        Tail, Option("--tail", help="upper, lower, or two-sided.")
    ] = "two-sided",
    correction: Annotated[
        Correction, Option("--correction", help="none, bonferroni, or fdr.")
    ] = "none",
    detect_absent: Annotated[
        bool, Option("--detect-absent", help="Detect significant absent pairs.")
    ] = False,
    min_occupation: Annotated[
        float, Option("--min-occupation", help="Absent-edge occupation threshold.")
    ] = 0.5,
    max_absent: Annotated[
        int | None, Option("--max-absent", help="Maximum absent pairs to output.")
    ] = None,
    self_loops: Annotated[
        bool, Option("--self-loops/--no-self-loops", help="Include diagonal pairs.")
    ] = True,
) -> None:
    """Fit fixed-strength ME, then filter against its Poisson null."""
    result = filter_model(
        read_edges(input_path),
        family=Family.ME,
        constraint=Constraint.STRENGTH,
        alpha=alpha,
        tail=tail,
        correction=correction,
        detect_absent=detect_absent,
        min_occupation=min_occupation,
        max_absent=max_absent,
        self_loops=self_loops,
    )
    _write_outputs(result, output_prefix)


@app.command("strength-edges-poisson")
def strength_edges(
    input_path: Path,
    output_prefix: Annotated[
        Path, Option("--output-prefix", help="Prefix/directory for output CSV files.")
    ],
    target_edges: Annotated[
        float, Option("--target-edges", help="Target number of edges for fitting.")
    ],
    alpha: Annotated[float, Option("--alpha", help="Significance level.")] = 0.05,
    tail: Annotated[
        Tail, Option("--tail", help="upper, lower, or two-sided.")
    ] = "two-sided",
    correction: Annotated[
        Correction, Option("--correction", help="none, bonferroni, or fdr.")
    ] = "none",
    detect_absent: Annotated[
        bool, Option("--detect-absent", help="Detect significant absent pairs.")
    ] = False,
    min_occupation: Annotated[
        float, Option("--min-occupation", help="Absent-edge occupation threshold.")
    ] = 0.5,
    min_expected: Annotated[
        float,
        Option("--min-expected", help="Absent expected-weight threshold."),
    ] = 0.0,
    max_absent: Annotated[
        int | None, Option("--max-absent", help="Maximum absent pairs to output.")
    ] = None,
    self_loops: Annotated[
        bool, Option("--self-loops/--no-self-loops", help="Include diagonal pairs.")
    ] = True,
) -> None:
    """Fit strength-edges ME, then filter against its zero-inflated null."""
    edges = read_edges(input_path)
    node_count = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(node_count, dtype=np.float64)
    s_in = np.zeros(node_count, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    fit = fit_model(
        family=Family.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=target_edges,
        self_loops=self_loops,
    )
    result = filter_strength_edges_poisson(
        edges,
        fit,
        alpha=alpha,
        tail=tail,
        correction=correction,
        detect_absent=detect_absent,
        min_occupation=min_occupation,
        min_expected=min_expected,
        max_absent=max_absent,
    )
    _write_outputs(result, output_prefix)


@app.command("strength-cost-poisson")
def strength_cost(
    input_path: Path,
    costs_path: Annotated[
        Path, Option("--costs", help="CSV with source,target,cost columns.")
    ],
    target_cost: Annotated[
        float, Option("--target-cost", help="Target average cost for fitting.")
    ],
    output_prefix: Annotated[
        Path, Option("--output-prefix", help="Prefix/directory for output CSV files.")
    ],
    alpha: Annotated[float, Option("--alpha", help="Significance level.")] = 0.05,
    tail: Annotated[
        Tail, Option("--tail", help="upper, lower, or two-sided.")
    ] = "two-sided",
    correction: Annotated[
        Correction, Option("--correction", help="none, bonferroni, or fdr.")
    ] = "none",
    detect_absent: Annotated[
        bool, Option("--detect-absent", help="Detect significant absent pairs.")
    ] = False,
    min_occupation: Annotated[
        float, Option("--min-occupation", help="Absent-edge occupation threshold.")
    ] = 0.5,
    min_expected: Annotated[
        float,
        Option("--min-expected", help="Absent expected-weight threshold."),
    ] = 0.0,
    max_absent: Annotated[
        int | None, Option("--max-absent", help="Maximum absent pairs to output.")
    ] = None,
    self_loops: Annotated[
        bool, Option("--self-loops/--no-self-loops", help="Include diagonal pairs.")
    ] = True,
) -> None:
    """Fit strength-cost ME, then filter against its Poisson null."""
    edges = read_edges(input_path)
    cost_table = pa_csv.read_csv(costs_path)
    cost_sources = cost_table.column("source").to_numpy().astype(np.uint64)
    cost_targets = cost_table.column("target").to_numpy().astype(np.uint64)
    cost_values = cost_table.column("cost").to_numpy().astype(np.float64)
    node_count = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(node_count, dtype=np.float64)
    s_in = np.zeros(node_count, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    fit = fit_model(
        family=Family.ME,
        constraint=Constraint.STRENGTH_COST,
        strength_out=s_out,
        strength_in=s_in,
        cost_sources=cost_sources,
        cost_targets=cost_targets,
        cost_values=cost_values,
        target_cost=target_cost,
        self_loops=self_loops,
    )
    result = filter_strength_cost_poisson(
        edges,
        fit,
        cost_sources,
        cost_targets,
        cost_values,
        alpha=alpha,
        tail=tail,
        correction=correction,
        detect_absent=detect_absent,
        min_occupation=min_occupation,
        min_expected=min_expected,
        max_absent=max_absent,
    )
    _write_outputs(result, output_prefix)


@app.command("strength-degree-poisson")
def strength_degree(
    input_path: Path,
    output_prefix: Annotated[
        Path, Option("--output-prefix", help="Prefix/directory for output CSV files.")
    ],
    alpha: Annotated[float, Option("--alpha", help="Significance level.")] = 0.05,
    tail: Annotated[
        Tail, Option("--tail", help="upper, lower, or two-sided.")
    ] = "two-sided",
    correction: Annotated[
        Correction, Option("--correction", help="none, bonferroni, or fdr.")
    ] = "none",
    detect_absent: Annotated[
        bool, Option("--detect-absent", help="Detect significant absent pairs.")
    ] = False,
    min_occupation: Annotated[
        float, Option("--min-occupation", help="Absent-edge occupation threshold.")
    ] = 0.5,
    min_expected: Annotated[
        float,
        Option("--min-expected", help="Absent expected-weight threshold."),
    ] = 0.0,
    max_absent: Annotated[
        int | None, Option("--max-absent", help="Maximum absent pairs to output.")
    ] = None,
    self_loops: Annotated[
        bool, Option("--self-loops/--no-self-loops", help="Include diagonal pairs.")
    ] = True,
) -> None:
    """Fit strength-degree ME, then filter against its zero-inflated null."""
    edges = read_edges(input_path)
    node_count = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(node_count, dtype=np.float64)
    s_in = np.zeros(node_count, dtype=np.float64)
    d_out = np.zeros(node_count, dtype=np.float64)
    d_in = np.zeros(node_count, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    for src in edges.source:
        d_out[src] += 1
    for tgt in edges.target:
        d_in[tgt] += 1
    fit = fit_model(
        family=Family.ME,
        constraint=Constraint.STRENGTH_DEGREE,
        strength_out=s_out,
        strength_in=s_in,
        degree_out=d_out,
        degree_in=d_in,
        self_loops=self_loops,
    )
    result = filter_strength_degree_poisson(
        edges,
        fit,
        alpha=alpha,
        tail=tail,
        correction=correction,
        detect_absent=detect_absent,
        min_occupation=min_occupation,
        min_expected=min_expected,
        max_absent=max_absent,
    )
    _write_outputs(result, output_prefix)


@app.command("degree-events-poisson")
def degree_events(
    input_path: Path,
    output_prefix: Annotated[
        Path, Option("--output-prefix", help="Prefix/directory for output CSV files.")
    ],
    alpha: Annotated[float, Option("--alpha", help="Significance level.")] = 0.05,
    tail: Annotated[
        Tail, Option("--tail", help="upper, lower, or two-sided.")
    ] = "two-sided",
    correction: Annotated[
        Correction, Option("--correction", help="none, bonferroni, or fdr.")
    ] = "none",
    detect_absent: Annotated[
        bool, Option("--detect-absent", help="Detect significant absent pairs.")
    ] = False,
    min_occupation: Annotated[
        float, Option("--min-occupation", help="Absent-edge occupation threshold.")
    ] = 0.5,
    min_expected: Annotated[
        float,
        Option("--min-expected", help="Absent expected-weight threshold."),
    ] = 0.0,
    max_absent: Annotated[
        int | None, Option("--max-absent", help="Maximum absent pairs to output.")
    ] = None,
    self_loops: Annotated[
        bool, Option("--self-loops/--no-self-loops", help="Include diagonal pairs.")
    ] = True,
) -> None:
    """Fit degree-events ME, then filter against its zero-inflated null."""
    edges = read_edges(input_path)
    node_count = int(max(edges.source.max(), edges.target.max())) + 1
    d_out = np.zeros(node_count, dtype=np.float64)
    d_in = np.zeros(node_count, dtype=np.float64)
    for src in edges.source:
        d_out[src] += 1
    for tgt in edges.target:
        d_in[tgt] += 1
    fit = fit_model(
        family=Family.ME,
        constraint=Constraint.DEGREE_EVENTS,
        degree_out=d_out,
        degree_in=d_in,
        total_events=int(edges.weight.sum()),
        self_loops=self_loops,
    )
    result = filter_degree_events_poisson(
        edges,
        fit,
        alpha=alpha,
        tail=tail,
        correction=correction,
        detect_absent=detect_absent,
        min_occupation=min_occupation,
        min_expected=min_expected,
        max_absent=max_absent,
    )
    _write_outputs(result, output_prefix)


def _write_outputs(result: FilterResult, output_prefix: Path) -> None:
    output_prefix.mkdir(parents=True, exist_ok=True)
    _write_filtered(result.upper, output_prefix / "upper.csv")
    _write_filtered(result.lower, output_prefix / "lower.csv")
    _write_filtered(result.compatible, output_prefix / "compatible.csv")
    _write_filtered(result.absent_lower, output_prefix / "absent_lower.csv")
    typer.echo(f"Wrote filter outputs to {output_prefix}", err=True)


def _write_filtered(filtered: FilteredEdges, path: Path) -> None:
    lines = ["source,target,weight,upper_pvalue,lower_pvalue,expected,occupation"]
    for row in zip(
        filtered.edges.source,
        filtered.edges.target,
        filtered.edges.weight,
        filtered.upper_pvalue,
        filtered.lower_pvalue,
        filtered.expected,
        filtered.occupation,
        strict=True,
    ):
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
