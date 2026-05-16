"""Generation subcommands for the ODME CLI."""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from typer import Option, Typer

from odme.analysis import directed_strengths
from odme.data.io import read_edges
from odme.models import fit_fixed_strength_me, sample_poisson

app = Typer(no_args_is_help=True)


@app.command()
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
    effective_quiet = quiet or output_json
    edges = read_edges(input_path)
    s = directed_strengths(edges)
    fit = fit_fixed_strength_me(s.out, s.incoming)
    sample = sample_poisson(fit.x, fit.y, seed=seed)

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

    if not effective_quiet:
        dest = str(output) if output else "stdout"
        typer.echo(f"Wrote Poisson sample to {dest}", err=True)


def _write_output(content: str, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    else:
        sys.stdout.write(content)
