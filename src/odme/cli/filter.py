"""Typer commands for ODME statistical filtering."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pyarrow.csv as pa_csv
import typer
from typer import Option, Typer

from odme.data.frames import ProbabilityTable
from odme.data.io import read_edges
from odme.filtering import (
    Correction,
    FilteredEdges,
    FilterResult,
    Tail,
    filter_custom_rates_poisson,
    filter_fixed_strength_me,
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


@app.command("custom-rates")
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
    result = filter_custom_rates_poisson(
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


@app.command("fixed-strength")
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
    result = filter_fixed_strength_me(
        read_edges(input_path),
        alpha=alpha,
        tail=tail,
        correction=correction,
        detect_absent=detect_absent,
        min_occupation=min_occupation,
        max_absent=max_absent,
        self_loops=self_loops,
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
