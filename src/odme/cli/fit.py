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
    fit_fixed_degree_binary,
    fit_fixed_strength_me,
    fit_strength_degree_zip,
)

app = Typer(no_args_is_help=True)


@app.command()
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
    result = fit_fixed_strength_me(s.out, s.incoming)

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


@app.command()
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
    result = fit_fixed_degree_binary(
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


@app.command("strength-degree-zip")
def strength_degree_zip(
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
    result = fit_strength_degree_zip(
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
                "degree_x": float(dx),
                "degree_y": float(dy),
                "excess_x": float(ex),
                "excess_y": float(ey),
            }
            for n, dx, dy, ex, ey in zip(
                result.node,
                result.degree_x,
                result.degree_y,
                result.excess_x,
                result.excess_y,
                strict=True,
            )
        ]
        _write_output(json.dumps(data, indent=2), output)
    else:
        lines = ["node,degree_x,degree_y,excess_x,excess_y"]
        for n, dx, dy, ex, ey in zip(
            result.node,
            result.degree_x,
            result.degree_y,
            result.excess_x,
            result.excess_y,
            strict=True,
        ):
            lines.append(f"{n},{dx},{dy},{ex},{ey}")
        _write_output("\n".join(lines) + "\n", output)

    if not effective_quiet:
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote strength-degree ZIP multipliers to {dest}", err=True)


def _write_output(content: str, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    else:
        sys.stdout.write(content)
