"""Fitting subcommands for the ODME CLI."""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from typer import Option, Typer

from odme.analysis import directed_strengths
from odme.data.io import read_edges
from odme.models import fit_fixed_strength_me

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


def _write_output(content: str, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    else:
        sys.stdout.write(content)
