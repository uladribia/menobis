"""Fitting subcommands for the MENoBiS CLI."""

import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from typer import Option, Typer

from menobis.analysis import directed_degrees, directed_strengths
from menobis.data.io import read_edges
from menobis.models import (
    StrengthFit,
    fit_degree_bernoulli,
    fit_strength_cost_poisson,
    fit_strength_degree_poisson,
    fit_strength_edges_poisson,
    fit_strength_geometric,
    fit_strength_negative_binomial,
    fit_strength_poisson,
)

app = Typer(no_args_is_help=True)


@app.command("strength-poisson")
def strengths(
    input_path: Path,
    output: Annotated[
        Path | None, Option("--output", "-o", help="Output path. Stdout if omitted.")
    ] = None,
    output_json: Annotated[bool, Option("--json", help="Output as JSON.")] = False,
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
) -> None:
    """Fit Lagrange multipliers for the fixed-strength ME model."""
    effective_quiet = quiet or output_json
    edges = read_edges(input_path)
    s = directed_strengths(edges)
    result = fit_strength_poisson(s.out, s.incoming)

    if output_json:
        data = [
            {"node": int(n), "x": float(x), "y": float(y)}
            for n, x, y in zip(result.node, result.x, result.y, strict=True)
        ]
        _write_output(json.dumps(data, indent=2), output)
    else:
        lines = ["node,x,y"]
        for n, x, y in zip(result.node, result.x, result.y, strict=True):
            lines.append(f"{n},{x},{y}")
        _write_output("\n".join(lines) + "\n", output)

    if not effective_quiet:
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
        float, Option("--tolerance", help="Conic solver tolerance.")
    ] = 1e-8,
    max_iterations: Annotated[
        int, Option("--max-iterations", help="Maximum conic solver iterations.")
    ] = 1000,
) -> None:
    """Fit the W fixed-strength geometric model."""
    effective_quiet = quiet or output_json
    edges = read_edges(input_path)
    s = directed_strengths(edges)
    result = fit_strength_geometric(
        s.out.astype(np.float64),
        s.incoming.astype(np.float64),
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _write_w_strength_result(result, output, output_json)
    if not effective_quiet:
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
        int, Option("--layers", help="Negative-binomial layer count M (>1).")
    ] = 3,
    self_loops: Annotated[
        bool,
        Option("--self-loops/--no-self-loops", help="Whether model self loops."),
    ] = True,
    tolerance: Annotated[
        float, Option("--tolerance", help="Conic solver tolerance.")
    ] = 1e-8,
    max_iterations: Annotated[
        int, Option("--max-iterations", help="Maximum conic solver iterations.")
    ] = 1000,
) -> None:
    """Fit the W fixed-strength negative-binomial model."""
    effective_quiet = quiet or output_json
    edges = read_edges(input_path)
    s = directed_strengths(edges)
    result = fit_strength_negative_binomial(
        s.out.astype(np.float64),
        s.incoming.astype(np.float64),
        layers=layers,
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _write_w_strength_result(result, output, output_json)
    if not effective_quiet:
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote W negative-binomial multipliers to {dest}", err=True)


@app.command("degree-bernoulli")
def degrees(
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
    effective_quiet = quiet or output_json
    edges = read_edges(input_path)
    k = directed_degrees(edges)
    result = fit_degree_bernoulli(
        k.out.astype(np.float64), k.incoming.astype(np.float64), self_loops=self_loops
    )

    if output_json:
        data = [
            {"node": int(n), "x": float(x), "y": float(y)}
            for n, x, y in zip(result.node, result.x, result.y, strict=True)
        ]
        _write_output(json.dumps(data, indent=2), output)
    else:
        lines = ["node,x,y"]
        for n, x, y in zip(result.node, result.x, result.y, strict=True):
            lines.append(f"{n},{x},{y}")
        _write_output("\n".join(lines) + "\n", output)

    if not effective_quiet:
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
    effective_quiet = quiet or output_json
    edges = read_edges(input_path)
    s = directed_strengths(edges)
    k = directed_degrees(edges)
    result = fit_strength_degree_poisson(
        s.out.astype(np.float64),
        s.incoming.astype(np.float64),
        k.out.astype(np.float64),
        k.incoming.astype(np.float64),
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
        _write_output(json.dumps(data, indent=2), output)
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
        _write_output("\n".join(lines) + "\n", output)

    if not effective_quiet:
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
    effective_quiet = quiet or output_json
    edges = read_edges(input_path)
    s = directed_strengths(edges)
    result = fit_strength_edges_poisson(
        s.out.astype(np.float64),
        s.incoming.astype(np.float64),
        float(edges.num_edges if target_edges is None else target_edges),
        self_loops=self_loops,
    )

    if output_json:
        data = [
            {"node": int(n), "x": float(x), "y": float(y), "lambda": result.lam}
            for n, x, y in zip(result.node, result.x, result.y, strict=True)
        ]
        _write_output(json.dumps(data, indent=2), output)
    else:
        lines = ["node,x,y,lambda"]
        for n, x, y in zip(result.node, result.x, result.y, strict=True):
            lines.append(f"{n},{x},{y},{result.lam}")
        _write_output("\n".join(lines) + "\n", output)

    if not effective_quiet:
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
    effective_quiet = quiet or output_json
    edges = read_edges(input_path)
    s = directed_strengths(edges)
    import pyarrow.csv as pa_csv

    coordinate_table = pa_csv.read_csv(coordinates_path)
    coord_x = coordinate_table.column("x").to_numpy().astype(np.float64)
    coord_y = coordinate_table.column("y").to_numpy().astype(np.float64)
    if target_cost is None:
        target_cost = sum(
            float(w)
            * float(
                np.hypot(
                    coord_x[int(s_val)] - coord_x[int(t_val)],
                    coord_y[int(s_val)] - coord_y[int(t_val)],
                )
            )
            for s_val, t_val, w in zip(
                edges.source, edges.target, edges.weight, strict=True
            )
        )
    result = fit_strength_cost_poisson(
        s.out.astype(np.float64),
        s.incoming.astype(np.float64),
        coord_x,
        coord_y,
        target_cost,
        self_loops=self_loops,
    )

    if output_json:
        data = [
            {"node": int(n), "x": float(x), "y": float(y), "gamma": result.gamma}
            for n, x, y in zip(result.node, result.x, result.y, strict=True)
        ]
        _write_output(json.dumps(data, indent=2), output)
    else:
        lines = ["node,x,y,gamma"]
        for n, x, y in zip(result.node, result.x, result.y, strict=True):
            lines.append(f"{n},{x},{y},{result.gamma}")
        _write_output("\n".join(lines) + "\n", output)

    if not effective_quiet:
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote strength-cost ME multipliers to {dest}", err=True)


def _write_w_strength_result(
    result: StrengthFit, path: Path | None, output_json: bool
) -> None:
    def _float_or_nan(value: float | None) -> float:
        return float("nan") if value is None else float(value)

    def _int_or_zero(value: int | None) -> int:
        return 0 if value is None else int(value)

    if output_json:
        data = {
            "status": result.status,
            "layers": _int_or_zero(result.layers),
            "objective": _float_or_nan(result.objective),
            "iterations": int(result.iterations),
            "min_margin": _float_or_nan(result.min_margin),
            "max_q": _float_or_nan(result.max_q),
            "max_strength_residual": _float_or_nan(result.max_strength_residual),
            "total_strength_residual": _float_or_nan(result.total_strength_residual),
            "metrics": {
                "variables": _int_or_zero(result.variables),
                "auxiliary_variables": _int_or_zero(result.auxiliary_variables),
                "exponential_cones": _int_or_zero(result.exponential_cones),
                "power_cones": _int_or_zero(result.power_cones),
                "linear_constraints": _int_or_zero(result.linear_constraints),
                "sparse_nonzeros": _int_or_zero(result.sparse_nonzeros),
            },
            "nodes": [
                {"node": int(n), "x": float(x), "y": float(y)}
                for n, x, y in zip(result.node, result.x, result.y, strict=True)
            ],
        }
        _write_output(json.dumps(data, indent=2), path)
        return

    lines = [
        "node,x,y,layers,status,objective,iterations,min_margin,max_q,"
        "max_strength_residual,total_strength_residual"
    ]
    for n, x, y in zip(result.node, result.x, result.y, strict=True):
        lines.append(
            f"{n},{x},{y},{result.layers},{result.status},{result.objective},"
            f"{result.iterations},{result.min_margin},{result.max_q},"
            f"{result.max_strength_residual},{result.total_strength_residual}"
        )
    _write_output("\n".join(lines) + "\n", path)


def _write_output(content: str, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    else:
        sys.stdout.write(content)
