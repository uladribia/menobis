"""Generation subcommands for the ODME CLI."""

import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from typer import Option, Typer

from odme.analysis import directed_degrees, directed_strengths
from odme.data.frames import EdgeTable
from odme.data.io import read_edges, read_probabilities
from odme.models import (
    fit_fixed_degree_binary,
    fit_fixed_strength_me,
    fit_gravity_me,
    fit_strength_degree_me,
    fit_strength_edges_me,
    sample_custom_pij_events_multinomial,
    sample_custom_pij_events_poisson,
    sample_fixed_degree_events_me,
    sample_gravity_me,
    sample_multinomial,
    sample_poisson,
    sample_poisson_multinomial,
    sample_strength_degree_me,
    sample_strength_edges_me,
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
    edges = read_edges(input_path)
    s = directed_strengths(edges)
    fit = fit_fixed_strength_me(s.out, s.incoming)
    _emit_edges(sample_poisson(fit.x, fit.y, seed=seed), output, output_json)
    _progress("Wrote Poisson sample", output, quiet, output_json)


@app.command()
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
    s = directed_strengths(edges)
    fit = fit_fixed_strength_me(s.out, s.incoming)
    sample = sample_multinomial(fit.x, fit.y, total_events=total_events, seed=seed)
    _emit_edges(sample, output, output_json)
    _progress("Wrote multinomial sample", output, quiet, output_json)


@app.command("poisson-multinomial")
def poisson_multinomial(
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
    """Generate a Poisson-total multinomial fixed-strength ME sample."""
    edges = read_edges(input_path)
    s = directed_strengths(edges)
    fit = fit_fixed_strength_me(s.out, s.incoming)
    _emit_edges(
        sample_poisson_multinomial(fit.x, fit.y, seed=seed), output, output_json
    )
    _progress("Wrote Poisson-multinomial sample", output, quiet, output_json)


@app.command("degree-events-me")
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
    k = directed_degrees(edges)
    fit = fit_fixed_degree_binary(
        k.out.astype(np.float64), k.incoming.astype(np.float64), self_loops=self_loops
    )
    sample = sample_fixed_degree_events_me(
        fit, total_events=total_events, seed=seed, self_loops=self_loops
    )
    _emit_edges(sample, output, output_json)
    _progress("Wrote fixed-degree events ME sample", output, quiet, output_json)


@app.command("strength-degree-me")
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
    s = directed_strengths(edges)
    k = directed_degrees(edges)
    fit = fit_strength_degree_me(
        s.out.astype(np.float64),
        s.incoming.astype(np.float64),
        k.out.astype(np.float64),
        k.incoming.astype(np.float64),
        self_loops=self_loops,
    )
    _emit_edges(sample_strength_degree_me(fit, seed=seed), output, output_json)
    _progress("Wrote strength-degree ME sample", output, quiet, output_json)


@app.command("strength-edges-me")
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
    s = directed_strengths(edges)
    fit = fit_strength_edges_me(
        s.out.astype(np.float64),
        s.incoming.astype(np.float64),
        float(edges.num_edges if target_edges is None else target_edges),
        self_loops=self_loops,
    )
    _emit_edges(sample_strength_edges_me(fit, seed=seed), output, output_json)
    _progress("Wrote strength-edges ME sample", output, quiet, output_json)


@app.command("custom-pij")
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
        sample = sample_custom_pij_events_poisson(
            probabilities, total_events=total_events, seed=seed
        )
    elif ensemble == "multinomial":
        sample = sample_custom_pij_events_multinomial(
            probabilities, total_events=total_events, seed=seed
        )
    else:
        msg = "--ensemble must be 'poisson' or 'multinomial'"
        raise typer.BadParameter(msg)
    _emit_edges(sample, output, output_json)
    _progress("Wrote custom p_ij ME sample", output, quiet, output_json)


@app.command("gravity-me")
def gravity_me_cmd(
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
    """Generate from the gravity ME model: fixed strength + total cost."""
    import pyarrow.csv as pa_csv

    edges = read_edges(input_path)
    s = directed_strengths(edges)
    cost_table = pa_csv.read_csv(cost_path)
    c_src = cost_table.column("source").to_numpy()
    c_tgt = cost_table.column("target").to_numpy()
    c_val = cost_table.column("cost").to_numpy().astype(np.float64)
    if target_cost is None:
        cost_map: dict[tuple[int, int], float] = {}
        for cs, ct, cv in zip(c_src, c_tgt, c_val, strict=True):
            cost_map[(int(cs), int(ct))] = float(cv)
        target_cost = sum(
            float(w_val) * cost_map.get((int(s_val), int(t_val)), 0.0)
            for s_val, t_val, w_val in zip(
                edges.source, edges.target, edges.weight, strict=True
            )
        )
    fit = fit_gravity_me(
        s.out.astype(np.float64),
        s.incoming.astype(np.float64),
        c_src,
        c_tgt,
        c_val,
        target_cost,
        self_loops=self_loops,
    )
    _emit_edges(
        sample_gravity_me(fit, c_src, c_tgt, c_val, seed=seed), output, output_json
    )
    _progress("Wrote gravity ME sample", output, quiet, output_json)


def _write_output(content: str, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    else:
        sys.stdout.write(content)
