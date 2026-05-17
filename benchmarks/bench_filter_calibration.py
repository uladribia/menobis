"""Calibration benchmark for ODME statistical filters."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import fmean
from typing import Annotated

import numpy as np
import typer

from odme.data.frames import EdgeTable, ProbabilityTable
from odme.filtering import (
    FilterResult,
    filter_custom_poisson,
    filter_strength_poisson,
    filter_strength_edges_poisson,
)
from odme.models import StrengthEdgesFit, fit_strength_poisson, sample_strength_poisson
from odme.models.generation import sample_strength_edges_poisson

app = typer.Typer(help="Benchmark filtering calibration across alpha levels.")
ALPHAS = [0.01, 0.05, 0.1]
RATE = 10.0


def _complete_rates(n: int, rate: float) -> ProbabilityTable:
    source = np.repeat(np.arange(n, dtype=np.uint64), n)
    target = np.tile(np.arange(n, dtype=np.uint64), n)
    rates = np.full(n * n, rate, dtype=np.float64)
    return ProbabilityTable(source=source, target=target, probability=rates)


def _make_strength_sample(n: int, seed: int) -> EdgeTable:
    strength = np.full(n, round(n * RATE), dtype=np.uint64)
    fit = fit_strength_poisson(strength, strength)
    return sample_strength_poisson(fit.x, fit.y, seed=seed)


def _make_strength_edges_fit(n: int) -> StrengthEdgesFit:
    occupation = 0.98
    x = np.full(n, np.sqrt(RATE), dtype=np.float64)
    y = np.full(n, np.sqrt(RATE), dtype=np.float64)
    lam = occupation / ((1.0 - occupation) * np.expm1(RATE))
    return StrengthEdgesFit(
        node=np.arange(n, dtype=np.uint64),
        x=x,
        y=y,
        lam=float(lam),
        self_loops=True,
        converged=True,
        iterations=0,
    )


@app.command()
def main(
    output: Annotated[Path, typer.Option("--output", help="Output CSV path.")] = Path(
        "benchmarks/results/filter_calibration.csv"
    ),
    n: Annotated[int, typer.Option("--nodes", help="Node count.")] = 80,
    repeats: Annotated[int, typer.Option("--repeats", help="Repeats per alpha.")] = 30,
) -> None:
    """Run calibration and write flagged fractions over existing pairs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    custom_rates = _complete_rates(n, RATE)
    zip_fit = _make_strength_edges_fit(n)
    for alpha in ALPHAS:
        for repeat in range(repeats):
            sample = _make_strength_sample(n, repeat)
            filtered = filter_strength_poisson(sample, alpha=alpha)
            _append_row(rows, "fixed_strength_poisson", alpha, repeat, sample, filtered)

            custom_sample = sample_strength_poisson(
                np.full(n, np.sqrt(RATE)), np.full(n, np.sqrt(RATE)), seed=repeat
            )
            custom_filtered = filter_custom_poisson(
                custom_sample, custom_rates, alpha=alpha
            )
            _append_row(
                rows,
                "custom_rates_poisson",
                alpha,
                repeat,
                custom_sample,
                custom_filtered,
            )

            zip_sample = sample_strength_edges_poisson(zip_fit, seed=repeat)
            zip_filtered = filter_strength_edges_poisson(zip_sample, zip_fit, alpha=alpha)
            _append_row(
                rows, "strength_edges_zip", alpha, repeat, zip_sample, zip_filtered
            )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for model in sorted({str(row["model"]) for row in rows}):
        for alpha in ALPHAS:
            vals = [
                float(row["flagged_fraction"])
                for row in rows
                if row["model"] == model and row["alpha"] == alpha
            ]
            typer.echo(f"{model} alpha={alpha}: mean={fmean(vals):.4f}")
    typer.echo(f"Wrote {output}")


def _append_row(
    rows: list[dict[str, object]],
    model: str,
    alpha: float,
    repeat: int,
    sample: EdgeTable,
    filtered: FilterResult,
) -> None:
    flagged = filtered.upper.edges.num_edges + filtered.lower.edges.num_edges
    rows.append(
        {
            "model": model,
            "alpha": alpha,
            "repeat": repeat,
            "existing_edges": sample.num_edges,
            "flagged_fraction": flagged / max(sample.num_edges, 1),
        }
    )


if __name__ == "__main__":
    app()
