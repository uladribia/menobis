"""Generation subcommands for the MENoBiS CLI."""

import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from typer import Option, Typer

from menobis.data.frames import EdgeTable
from menobis.data.io import read_edges, read_probabilities
from menobis.models.generation import (
    _sample_custom_multinomial as sample_custom_multinomial,
)
from menobis.models.generation import (
    _sample_custom_poisson as sample_custom_poisson,
)
from menobis.routing import (
    Constraint,
    Ensemble,
    ModelFamily,
    fit_model,
    sample_model,
)

app = Typer(no_args_is_help=True)


def _emit_edges(sample: EdgeTable, output: Path | None, output_json: bool) -> None:
    if output_json:
        data = [
            {"source": int(s), "target": int(t), "weight": int(w)}
            for s, t, w in zip(sample.source, sample.target, sample.weight, strict=True)
        ]
        _write_output(json.dumps(data, indent=2), output)
    else:
        lines = ["source,target,weight"]
        for s_val, t_val, w_val in zip(
            sample.source, sample.target, sample.weight, strict=True
        ):
            lines.append(f"{s_val},{t_val},{w_val}")
        _write_output("\n".join(lines) + "\n", output)


def _progress(
    message: str, output: Path | None, quiet: bool, output_json: bool
) -> None:
    if not (quiet or output_json):
        dest = str(output) if output else "stdout"
        typer.echo(f"{message} to {dest}", err=True)


@app.command("strength-poisson")
def poisson(
    input_path: Path,
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    seed: Annotated[int, Option("--seed", "-s", help="Random seed.")] = 0,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
) -> None:
    """Generate a Poisson sample from a fixed-strength ME model."""
    edges = read_edges(input_path)
    nc = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(nc, dtype=np.float64)
    s_in = np.zeros(nc, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
    )
    sample = sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        seed=seed,
    )
    _emit_edges(sample, output, output_json)
    _progress("Wrote Poisson sample", output, quiet, output_json)


@app.command("strength-multinomial")
def multinomial(
    input_path: Path,
    total_events: Annotated[int, Option("--total-events", help="Fixed total events.")],
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    seed: Annotated[int, Option("--seed", "-s", help="Random seed.")] = 0,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
) -> None:
    """Generate a multinomial fixed-strength ME sample."""
    edges = read_edges(input_path)
    nc = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(nc, dtype=np.float64)
    s_in = np.zeros(nc, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
    )
    sample = sample_model(
        ensemble=Ensemble.CANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        total_events=total_events,
        seed=seed,
    )
    _emit_edges(sample, output, output_json)
    _progress("Wrote multinomial sample", output, quiet, output_json)


@app.command("degree-events-poisson")
def degree_events_me(
    input_path: Path,
    total_events: Annotated[int, Option("--total-events", help="Expected events T.")],
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    seed: Annotated[int, Option("--seed", "-s", help="Random seed.")] = 0,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
) -> None:
    """Generate a fixed-degree ME sample with expected total events T."""
    edges = read_edges(input_path)
    nc = int(max(edges.source.max(), edges.target.max())) + 1
    d_out = np.zeros(nc, dtype=np.float64)
    d_in = np.zeros(nc, dtype=np.float64)
    for src in edges.source:
        d_out[src] += 1
    for tgt in edges.target:
        d_in[tgt] += 1
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.DEGREE_EVENTS,
        degree_out=d_out,
        degree_in=d_in,
        total_events=total_events,
        self_loops=self_loops,
    )
    sample = sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.DEGREE_EVENTS,
        fit=fit,
        seed=seed,
    )
    _emit_edges(sample, output, output_json)
    _progress("Wrote fixed-degree events ME sample", output, quiet, output_json)


@app.command("strength-degree-poisson")
def strength_degree_me(
    input_path: Path,
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    seed: Annotated[int, Option("--seed", "-s", help="Random seed.")] = 0,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
) -> None:
    """Generate a sample from the fixed-strength-degree ME model."""
    edges = read_edges(input_path)
    nc = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(nc, dtype=np.float64)
    s_in = np.zeros(nc, dtype=np.float64)
    d_out = np.zeros(nc, dtype=np.float64)
    d_in = np.zeros(nc, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    for src in edges.source:
        d_out[src] += 1
    for tgt in edges.target:
        d_in[tgt] += 1
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_DEGREE,
        strength_out=s_out,
        strength_in=s_in,
        degree_out=d_out,
        degree_in=d_in,
        self_loops=self_loops,
    )
    sample = sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_DEGREE,
        fit=fit,
        seed=seed,
    )
    _emit_edges(sample, output, output_json)
    _progress("Wrote strength-degree ME sample", output, quiet, output_json)


@app.command("strength-edges-poisson")
def strength_edges_me(
    input_path: Path,
    target_edges: Annotated[
        float | None,
        Option(
            "--target-edges", help="Expected binary edge count. Defaults to observed."
        ),
    ] = None,
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    seed: Annotated[int, Option("--seed", "-s", help="Random seed.")] = 0,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
) -> None:
    """Generate a sample from the fixed-strength-and-edge-count ME model."""
    edges = read_edges(input_path)
    nc = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(nc, dtype=np.float64)
    s_in = np.zeros(nc, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    if target_edges is None:
        target_edges = float(len(edges))
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=target_edges,
        self_loops=self_loops,
    )
    sample = sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        fit=fit,
        seed=seed,
    )
    _emit_edges(sample, output, output_json)
    _progress("Wrote strength-edges ME sample", output, quiet, output_json)


@app.command("custom-poisson")
def custom_pij(
    probabilities_path: Path,
    total_events: Annotated[int, Option("--total-events", help="Expected events T.")],
    ensemble: Annotated[
        str, Option("--ensemble", help="Either 'poisson' or 'multinomial'.")
    ] = "poisson",
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    seed: Annotated[int, Option("--seed", "-s", help="Random seed.")] = 0,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
) -> None:
    """Generate a custom p_ij ME sample from a probability table."""
    probabilities = read_probabilities(probabilities_path)
    if ensemble == "poisson":
        sample = sample_custom_poisson(
            probabilities, total_events=total_events, seed=seed
        )
    elif ensemble == "multinomial":
        sample = sample_custom_multinomial(
            probabilities, total_events=total_events, seed=seed
        )
    else:
        msg = "--ensemble must be 'poisson' or 'multinomial'"
        raise typer.BadParameter(msg)
    _emit_edges(sample, output, output_json)
    _progress("Wrote custom p_ij ME sample", output, quiet, output_json)


@app.command("strength-cost-poisson")
def strength_cost_me_cmd(
    input_path: Path,
    coordinates_path: Annotated[
        Path,
        Option("--coordinates", help="Projected coordinates CSV with x,y columns."),
    ],
    target_cost: Annotated[
        float | None,
        Option("--target-cost", help="Target total cost. Defaults to observed."),
    ] = None,
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    seed: Annotated[int, Option("--seed", "-s", help="Random seed.")] = 0,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
) -> None:
    """Generate from the strength-cost ME model: fixed strength + total cost."""
    import pyarrow.csv as pa_csv

    edges = read_edges(input_path)
    nc = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(nc, dtype=np.float64)
    s_in = np.zeros(nc, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    coordinate_table = pa_csv.read_csv(coordinates_path)
    coord_x = coordinate_table.column("x").to_numpy().astype(np.float64)
    coord_y = coordinate_table.column("y").to_numpy().astype(np.float64)
    if target_cost is None:
        target_cost = sum(
            float(w_val)
            * float(
                np.hypot(
                    coord_x[int(s_val)] - coord_x[int(t_val)],
                    coord_y[int(s_val)] - coord_y[int(t_val)],
                )
            )
            for s_val, t_val, w_val in zip(
                edges.source, edges.target, edges.weight, strict=True
            )
        )
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_COST,
        strength_out=s_out,
        strength_in=s_in,
        coord_x=coord_x,
        coord_y=coord_y,
        target_cost=target_cost,
        self_loops=self_loops,
    )
    sample = sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_COST,
        fit=fit,
        coord_x=coord_x,
        coord_y=coord_y,
        seed=seed,
    )
    _emit_edges(sample, output, output_json)
    _progress("Wrote strength-cost ME sample", output, quiet, output_json)


def _write_output(content: str, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    else:
        sys.stdout.write(content)
