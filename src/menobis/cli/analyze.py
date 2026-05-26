"""Analysis subcommands for the MENoBiS CLI."""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from typer import Option, Typer

from menobis.analysis import directed_strengths
from menobis.data.io import read_edges

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
    """Compute directed in/out strength sequences from an edge table."""
    effective_quiet = quiet or output_json
    edges = read_edges(input_path)
    result = directed_strengths(edges)

    if output_json:
        data = [
            {"node": int(n), "strength_out": int(o), "strength_in": int(i)}
            for n, o, i in zip(result.node, result.out, result.incoming, strict=True)
        ]
        _write_output(json.dumps(data, indent=2), output)
    else:
        lines = ["node,strength_out,strength_in"]
        for n, o, i in zip(result.node, result.out, result.incoming, strict=True):
            lines.append(f"{n},{o},{i}")
        _write_output("\n".join(lines) + "\n", output)

    if not effective_quiet:
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote directed strengths to {dest}", err=True)


def _write_output(content: str, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    else:
        sys.stdout.write(content)
