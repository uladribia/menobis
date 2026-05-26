"""Typer command for converting between MENoBiS edge-list formats."""

from pathlib import Path
from typing import Annotated

import typer
from typer import Argument, Option, Typer

from menobis.data.io import read_edges, write_edges

app = Typer(no_args_is_help=True)


@app.command()
def convert(
    input_path: Annotated[Path, Argument(help="Input edge file.")],
    output_path: Annotated[Path, Option("--output", "-o", help="Output edge file.")],
    quiet: Annotated[
        bool, Option("--quiet", help="Suppress progress messages.")
    ] = False,
) -> None:
    """Convert an edge table between supported formats."""
    edges = read_edges(input_path)
    write_edges(edges, output_path)
    if not quiet:
        typer.echo(
            f"Converted {len(edges)} edges: {input_path} -> {output_path}",
            err=True,
        )
