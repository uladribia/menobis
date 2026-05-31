"""Fitting subcommands for the MENoBiS CLI — all route through fit_model."""

import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from typer import Option, Typer

from menobis.data.io import read_edges
from menobis.routing import (
    Constraint,
    ModelFamily,
    fit_model,
)

app = Typer(no_args_is_help=True)


def _write_json_to_output(data: dict | list, path: Path | None) -> None:
    output = json.dumps(data, indent=2)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
    else:
        sys.stdout.write(output)


@app.command("strength-poisson")
def strength_me(
    input_path: Path,
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
    tolerance: Annotated[
        float,
        Option("--tolerance", help="Convergence tolerance."),
    ] = 1e-8,
    max_iterations: Annotated[
        int,
        Option("--max-iterations", help="Maximum iterations."),
    ] = 10000,
) -> None:
    """Fit Lagrange multipliers for the fixed-strength ME model."""
    edges = read_edges(input_path)
    nc = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(nc, dtype=np.float64)
    s_in = np.zeros(nc, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    result = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    if output_json:
        data = [
            {"node": int(n), "x": float(x), "y": float(y)}
            for n, x, y in zip(result.node, result.x, result.y, strict=True)
        ]
        _write_json_to_output(data, output)
    else:
        lines = ["node,x,y"]
        for n, x, y in zip(result.node, result.x, result.y, strict=True):
            lines.append(f"{n},{x},{y}")
        _write_json_to_output("\n".join(lines) + "\n", output)

    if not (quiet or output_json):
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote Lagrange multipliers to {dest}", err=True)


@app.command("strength-geometric")
def strength_geometric(
    input_path: Path,
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
    tolerance: Annotated[
        float,
        Option("--tolerance", help="Conic solver tolerance."),
    ] = 1e-8,
    max_iterations: Annotated[
        int,
        Option("--max-iterations", help="Maximum conic solver iterations."),
    ] = 1000,
) -> None:
    """Fit the W fixed-strength geometric model."""
    edges = read_edges(input_path)
    nc = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(nc, dtype=np.float64)
    s_in = np.zeros(nc, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    result = fit_model(
        family=ModelFamily.W,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
        layers=1,
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    if not output_json:
        lines = ["node,x,y"]
        for n, x, y in zip(result.node, result.x, result.y, strict=True):
            lines.append(f"{n},{x},{y}")
        lines.append("")
        path_txt = "\n".join(lines)
        _write_json_to_output(path_txt, output)
    else:
        _write_json_to_output(
            {
                "status": result.status,
                "iterations": result.iterations,
                "nodes": [
                    {"node": int(n), "x": float(x), "y": float(y)}
                    for n, x, y in zip(result.node, result.x, result.y, strict=True)
                ],
            },
            output,
        )

    if not (quiet or output_json):
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote W geometric multipliers to {dest}", err=True)


@app.command("strength-negative-binomial")
def strength_negative_binomial(
    input_path: Path,
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
    layers: Annotated[
        int,
        Option("--layers", help="Negative-binomial layer count M (>1)."),
    ] = 3,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
    tolerance: Annotated[
        float,
        Option("--tolerance", help="Conic solver tolerance."),
    ] = 1e-8,
    max_iterations: Annotated[
        int,
        Option("--max-iterations", help="Maximum conic solver iterations."),
    ] = 1000,
) -> None:
    """Fit the W fixed-strength negative-binomial model."""
    edges = read_edges(input_path)
    nc = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(nc, dtype=np.float64)
    s_in = np.zeros(nc, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    result = fit_model(
        family=ModelFamily.W,
        constraint=Constraint.STRENGTH,
        strength_out=s_out,
        strength_in=s_in,
        layers=layers,
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    if not output_json:
        lines = ["node,x,y"]
        for n, x, y in zip(result.node, result.x, result.y, strict=True):
            lines.append(f"{n},{x},{y}")
        lines.append("")
        _write_json_to_output("\n".join(lines), output)
    else:
        _write_json_to_output(
            {
                "status": result.status,
                "iterations": result.iterations,
                "nodes": [
                    {"node": int(n), "x": float(x), "y": float(y)}
                    for n, x, y in zip(result.node, result.x, result.y, strict=True)
                ],
            },
            output,
        )

    if not (quiet or output_json):
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote W negative-binomial multipliers to {dest}", err=True)


@app.command("degree-bernoulli")
def degree_bernoulli(
    input_path: Path,
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
) -> None:
    """Fit Lagrange multipliers for the fixed-degree binary model."""
    edges = read_edges(input_path)
    nc = int(max(edges.source.max(), edges.target.max())) + 1
    d_out = np.zeros(nc, dtype=np.float64)
    d_in = np.zeros(nc, dtype=np.float64)
    for src in edges.source:
        d_out[src] += 1
    for tgt in edges.target:
        d_in[tgt] += 1
    result = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.DEGREE_EVENTS,
        degree_out=d_out,
        degree_in=d_in,
        total_events=int(edges.weight.sum()),
        self_loops=self_loops,
    )

    if output_json:
        data = [
            {"node": int(n), "x": float(x), "y": float(y)}
            for n, x, y in zip(result.node, result.x, result.y, strict=True)
        ]
        _write_json_to_output(data, output)
    else:
        lines = ["node,x,y"]
        for n, x, y in zip(result.node, result.x, result.y, strict=True):
            lines.append(f"{n},{x},{y}")
        _write_json_to_output("\n".join(lines) + "\n", output)

    if not (quiet or output_json):
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote degree multipliers to {dest}", err=True)


@app.command("strength-degree-poisson")
def strength_degree_me(
    input_path: Path,
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
) -> None:
    """Fit the fixed-strength-degree zero-inflated Poisson model."""
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
    result = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_DEGREE,
        strength_out=s_out,
        strength_in=s_in,
        degree_out=d_out,
        degree_in=d_in,
        self_loops=self_loops,
    )

    if output_json:
        data = [
            {
                "node": int(n),
                "x": float(dx),
                "y": float(dy),
                "z": float(ex),
                "w": float(ey),
            }
            for n, dx, dy, ex, ey in zip(
                result.node,
                result.x,
                result.y,
                result.z,
                result.w,
                strict=True,
            )
        ]
        _write_json_to_output(data, output)
    else:
        lines = ["node,x,y,z,w"]
        for n, dx, dy, ex, ey in zip(
            result.node,
            result.x,
            result.y,
            result.z,
            result.w,
            strict=True,
        ):
            lines.append(f"{n},{dx},{dy},{ex},{ey}")
        _write_json_to_output("\n".join(lines) + "\n", output)

    if not (quiet or output_json):
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote strength-degree ME multipliers to {dest}", err=True)


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
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
) -> None:
    """Fit the fixed-strength-and-edge-count ME model."""
    edges = read_edges(input_path)
    nc = int(max(edges.source.max(), edges.target.max())) + 1
    s_out = np.zeros(nc, dtype=np.float64)
    s_in = np.zeros(nc, dtype=np.float64)
    np.add.at(s_out, edges.source, edges.weight.astype(np.float64))
    np.add.at(s_in, edges.target, edges.weight.astype(np.float64))
    if target_edges is None:
        target_edges = float(len(edges))
    result = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        strength_out=s_out,
        strength_in=s_in,
        target_edges=target_edges,
        self_loops=self_loops,
    )

    if output_json:
        data = [
            {"node": int(n), "x": float(x), "y": float(y), "lambda": result.lam}
            for n, x, y in zip(result.node, result.x, result.y, strict=True)
        ]
        _write_json_to_output(data, output)
    else:
        lines = ["node,x,y,lambda"]
        for n, x, y in zip(result.node, result.x, result.y, strict=True):
            lines.append(f"{n},{x},{y},{result.lam}")
        _write_json_to_output("\n".join(lines) + "\n", output)

    if not (quiet or output_json):
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote strength-edges ME multipliers to {dest}", err=True)


@app.command("strength-cost-poisson")
def strength_cost_me(
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
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
) -> None:
    """Fit the strength-cost ME model: fixed strength + fixed total cost."""
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
    result = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_COST,
        strength_out=s_out,
        strength_in=s_in,
        coord_x=coord_x,
        coord_y=coord_y,
        target_cost=target_cost,
        self_loops=self_loops,
    )

    if output_json:
        data = [
            {"node": int(n), "x": float(x), "y": float(y), "gamma": result.gamma}
            for n, x, y in zip(result.node, result.x, result.y, strict=True)
        ]
        _write_json_to_output(data, output)
    else:
        lines = ["node,x,y,gamma"]
        for n, x, y in zip(result.node, result.x, result.y, strict=True):
            lines.append(f"{n},{x},{y},{result.gamma}")
        _write_json_to_output("\n".join(lines) + "\n", output)

    if not (quiet or output_json):
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote strength-cost ME multipliers to {dest}", err=True)
