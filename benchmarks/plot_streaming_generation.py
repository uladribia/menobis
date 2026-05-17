"""Plot streaming-generation benchmark CSV files."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Annotated

import matplotlib
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = typer.Typer(help="Plot ODME streaming-generation benchmark results.")

FIGURES_DIR = Path("docs/figures")
RESULTS_DIR = Path("benchmarks/results")


def _latest_csv() -> Path:
    """Return the newest streaming-generation CSV file."""
    paths = sorted(RESULTS_DIR.glob("streaming_generation_*.csv"))
    if not paths:
        msg = "no streaming_generation_*.csv files found"
        raise FileNotFoundError(msg)
    return paths[-1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read successful benchmark rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["status"] == "ok"]


def _aggregate_time(
    rows: list[dict[str, str]],
) -> dict[str, list[tuple[int, float, float]]]:
    """Aggregate runtime by case and N as (N, mean, std)."""
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        buckets[(row["case"], int(row["n"]))].append(float(row["seconds"]))

    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for (case, n), values in buckets.items():
        std = stdev(values) if len(values) > 1 else 0.0
        grouped[case].append((n, fmean(values), std))
    for values in grouped.values():
        values.sort()
    return grouped


def _aggregate_rss(rows: list[dict[str, str]]) -> list[tuple[int, float]]:
    """Aggregate peak RSS by N using the maximum observed RSS."""
    by_n: dict[int, float] = defaultdict(float)
    for row in rows:
        n = int(row["n"])
        by_n[n] = max(by_n[n], float(row["max_rss_mb"]))
    return sorted(by_n.items())


def _plot_time(
    grouped: dict[str, list[tuple[int, float, float]]], output: Path
) -> None:
    """Plot mean runtime by N."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for case, values in sorted(grouped.items()):
        ns = [n for n, _, _ in values]
        seconds = [mean for _, mean, _ in values]
        ax.loglog(ns, seconds, marker="o", label=case.replace("_", " "))
    ax.set_xlabel("N nodes")
    ax.set_ylabel("Mean seconds over repeats")
    ax.set_title("Streaming generation runtime")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_rss(values: list[tuple[int, float]], output: Path) -> None:
    """Plot peak RSS by N."""
    ns = [n for n, _ in values]
    rss = [value for _, value in values]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, rss, marker="o")
    ax.set_xlabel("N nodes")
    ax.set_ylabel("Peak RSS (MiB)")
    ax.set_title("Streaming generation peak process RSS")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


@app.command()
def main(
    input_csv: Annotated[
        Path | None,
        typer.Option("--input", help="Benchmark CSV. Defaults to newest result."),
    ] = None,
) -> None:
    """Generate streaming benchmark figures under docs/figures."""
    path = input_csv or _latest_csv()
    rows = _read_rows(path)
    time_grouped = _aggregate_time(rows)
    rss_values = _aggregate_rss(rows)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    time_output = FIGURES_DIR / "streaming_generation_time.png"
    rss_output = FIGURES_DIR / "streaming_generation_rss.png"
    _plot_time(time_grouped, time_output)
    _plot_rss(rss_values, rss_output)
    typer.echo(f"Read {path}")
    typer.echo(f"Wrote {time_output}")
    typer.echo(f"Wrote {rss_output}")


if __name__ == "__main__":
    app()
