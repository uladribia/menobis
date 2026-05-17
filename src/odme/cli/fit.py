"""Fitting subcommands for the ODME CLI."""

import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from typer import Option, Typer

from odme.analysis import directed_degrees, directed_strengths
from odme.data.io import read_edges
from odme.models import (
    fit_degree_bernoulli,
    fit_strength_cost_poisson,
    fit_strength_degree_poisson,
    fit_strength_edges_poisson,
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
    cost_path: Annotated[
        Path, Option("--costs", help="Cost matrix CSV with source,target,cost columns.")
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

    cost_table = pa_csv.read_csv(cost_path)
    c_src = cost_table.column("source").to_numpy()
    c_tgt = cost_table.column("target").to_numpy()
    c_val = cost_table.column("cost").to_numpy().astype(np.float64)
    if target_cost is None:
        cost_map: dict[tuple[int, int], float] = {}
        for cs, ct, cv in zip(c_src, c_tgt, c_val, strict=True):
            cost_map[(int(cs), int(ct))] = float(cv)
        target_cost = sum(
            float(w) * cost_map.get((int(s_val), int(t_val)), 0.0)
            for s_val, t_val, w in zip(
                edges.source, edges.target, edges.weight, strict=True
            )
        )
    result = fit_strength_cost_poisson(
        s.out.astype(np.float64),
        s.incoming.astype(np.float64),
        c_src,
        c_tgt,
        c_val,
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


def _write_output(content: str, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    else:
        sys.stdout.write(content)
