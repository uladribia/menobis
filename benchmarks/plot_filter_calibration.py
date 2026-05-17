"""Plot filter calibration benchmark results."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Annotated

import matplotlib
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = typer.Typer(help="Plot filter calibration results.")


@app.command()
def main(
    input_csv: Annotated[Path, typer.Option("--input", help="Calibration CSV.")] = Path(
        "benchmarks/results/filter_calibration.csv"
    ),
    output: Annotated[Path, typer.Option("--output", help="Figure path.")] = Path(
        "docs/figures/filter_calibration.png"
    ),
) -> None:
    """Plot mean flagged fraction against alpha."""
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["model"]][float(row["alpha"])].append(
            float(row["flagged_fraction"])
        )
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, by_alpha in sorted(grouped.items()):
        xs = sorted(by_alpha)
        ys = [fmean(by_alpha[x]) for x in xs]
        ax.plot(xs, ys, marker="o", label=model.replace("_", " "))
    ax.plot([0, 0.1], [0, 0.1], "k--", label="ideal")
    ax.set_xlabel("alpha")
    ax.set_ylabel("fraction of existing edges flagged")
    ax.set_title("Filter calibration under generated null samples")
    ax.grid(True, alpha=0.3)
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    app()
