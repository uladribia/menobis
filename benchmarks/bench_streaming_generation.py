"""Verbose streaming-generation benchmark across ODME model cases.

Writes incremental CSV and JSONL results under ``benchmarks/results/`` so a
long run can be inspected even if it is interrupted.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import resource
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from odme.data.frames import EdgeTable, ProbabilityTable
from odme.models import (
    FitResult,
    StrengthCostFit,
    StrengthDegreeFit,
    StrengthEdgesFit,
    fit_degree_bernoulli,
    fit_strength_binomial,
    fit_strength_cost_poisson,
    fit_strength_degree_poisson,
    fit_strength_edges_poisson,
    fit_strength_poisson,
    sample_custom_multinomial,
    sample_custom_poisson,
    sample_degree_events_binomial,
    sample_degree_events_poisson,
    sample_strength_binomial,
    sample_strength_cost_binomial,
    sample_strength_cost_poisson,
    sample_strength_degree_binomial,
    sample_strength_degree_poisson,
    sample_strength_edges_binomial,
    sample_strength_edges_poisson,
    sample_strength_microcanonical,
    sample_strength_multinomial,
    sample_strength_poisson,
    sample_strength_poisson_multinomial,
)

DEFAULT_NODES = [10, 100, 500, 1000, 5000, 10000, 20000, 30000]
DEFAULT_CASES = [
    "strength_poisson",
    "strength_multinomial",
    "strength_poisson_multinomial",
    "strength_microcanonical",
    "strength_binomial",
    "custom_poisson_sparse",
    "custom_multinomial_sparse",
    "degree_events_poisson",
    "strength_cost_poisson",
    "strength_cost_binomial",
    "strength_edges_poisson",
    "strength_edges_binomial",
    "strength_degree_poisson",
]
app = typer.Typer(help="Verbose streaming-generation benchmark for ODME model cases.")

CSV_FIELDS = [
    "run_id",
    "timestamp_utc",
    "n",
    "case",
    "repeat",
    "seconds",
    "sampled_edges",
    "total_weight",
    "max_rss_mb",
    "status",
    "error",
]


@dataclass(frozen=True)
class Inputs:
    """Reusable synthetic inputs for one node count."""

    n: int
    total_events: int
    x_strength: np.ndarray
    y_strength: np.ndarray
    strength_out: np.ndarray
    strength_in: np.ndarray
    custom_probabilities: ProbabilityTable
    degree_fit: FitResult
    strength_cost_fit: StrengthCostFit
    strength_edges_fit: StrengthEdgesFit
    strength_degree_fit: StrengthDegreeFit
    strength_binomial_fit: FitResult
    cost_sources: np.ndarray
    cost_targets: np.ndarray
    cost_values: np.ndarray


def _max_rss_mb() -> float:
    """Return maximum resident set size in MiB for the current process."""
    # Linux reports KiB; macOS reports bytes. This repo benchmark is Linux-first.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    if value == "all":
        return list(DEFAULT_CASES)
    return [part.strip() for part in value.split(",") if part.strip()]


def _make_sparse_probability_table(n: int, degree: int) -> ProbabilityTable:
    """Build deterministic sparse custom p_ij support with O(N * degree) entries."""
    degree = max(1, min(degree, n))
    sources = np.repeat(np.arange(n, dtype=np.uint64), degree)
    offsets = np.arange(1, degree + 1, dtype=np.uint64)
    targets = (sources.reshape(n, degree) + offsets.reshape(1, degree)) % n
    probabilities = np.ones(n * degree, dtype=np.float64)
    return ProbabilityTable(
        source=sources,
        target=targets.reshape(-1).astype(np.uint64),
        probability=probabilities,
    )


def _make_inputs(n: int, average_strength: int, sparse_degree: int) -> Inputs:
    """Generate a real synthetic network and fit all models from it."""
    rng = np.random.default_rng(42)
    total_events = n * average_strength

    # 1. Create heterogeneous strength targets
    raw = rng.pareto(1.5, size=n) + 1.0
    s_out_target = np.round(raw / raw.sum() * total_events).astype(np.uint64)
    raw = rng.pareto(1.5, size=n) + 1.0
    s_in_target = np.round(raw / raw.sum() * total_events).astype(np.uint64)
    diff = int(s_out_target.sum()) - int(s_in_target.sum())
    if diff > 0:
        s_in_target[np.argmax(s_in_target)] += abs(diff)
    elif diff < 0:
        s_out_target[np.argmax(s_out_target)] += abs(diff)

    # 2. Fit and sample a real Poisson network
    pre_fit = fit_strength_poisson(s_out_target, s_in_target)
    network = sample_strength_poisson(pre_fit.x, pre_fit.y, seed=42)

    # 3. Extract actual strengths and degrees from the real network
    strength_out = np.zeros(n, dtype=np.uint64)
    strength_in = np.zeros(n, dtype=np.uint64)
    d_out = np.zeros(n, dtype=np.float64)
    d_in = np.zeros(n, dtype=np.float64)
    np.add.at(strength_out, network.source, network.weight)
    np.add.at(strength_in, network.target, network.weight)
    for src in network.source:
        if src < n:
            d_out[src] += 1
    for tgt in network.target:
        if tgt < n:
            d_in[tgt] += 1
    # Balance (rounding may cause tiny imbalance)
    diff = int(strength_out.sum()) - int(strength_in.sum())
    if diff > 0:
        strength_in[np.argmax(strength_in)] += abs(diff)
    elif diff < 0:
        strength_out[np.argmax(strength_out)] += abs(diff)
    total_events = int(strength_out.sum())

    s_out_f = strength_out.astype(np.float64)
    s_in_f = strength_in.astype(np.float64)

    # 4. Fit all models from the real network
    strength_fit = fit_strength_poisson(strength_out, strength_in)
    x_strength = strength_fit.x
    y_strength = strength_fit.y

    strength_binomial_fit = fit_strength_binomial(
        strength_out, strength_in, layers=10, tolerance=0.1, max_iterations=1000,
    )

    custom = _make_sparse_probability_table(n, sparse_degree)

    degree_fit = fit_degree_bernoulli(d_out, d_in, max_iterations=1000)

    target_edges = float(d_out.sum())
    strength_edges_fit = fit_strength_edges_poisson(
        s_out_f, s_in_f, target_edges, max_iterations=1000,
    )

    strength_degree_fit = fit_strength_degree_poisson(
        s_out_f, s_in_f, d_out, d_in, max_iterations=1000,
    )

    cost_degree = min(sparse_degree, n)
    cost_sources = np.repeat(np.arange(n, dtype=np.uint64), cost_degree)
    cost_offsets = np.arange(1, cost_degree + 1, dtype=np.uint64)
    cost_targets = (
        (cost_sources.reshape(n, cost_degree) + cost_offsets.reshape(1, cost_degree)) % n
    ).reshape(-1).astype(np.uint64)
    cost_values = rng.exponential(1.0, size=len(cost_sources))
    target_cost = float(np.mean(cost_values))
    strength_cost_fit = fit_strength_cost_poisson(
        s_out_f, s_in_f, cost_sources, cost_targets, cost_values,
        target_cost, max_iterations=1000,
    )

    return Inputs(
        n=n,
        total_events=total_events,
        x_strength=x_strength,
        y_strength=y_strength,
        strength_out=strength_out,
        strength_in=strength_in,
        custom_probabilities=custom,
        degree_fit=degree_fit,
        strength_cost_fit=strength_cost_fit,
        strength_edges_fit=strength_edges_fit,
        strength_degree_fit=strength_degree_fit,
        strength_binomial_fit=strength_binomial_fit,
        cost_sources=cost_sources,
        cost_targets=cost_targets,
        cost_values=cost_values,
    )


def _case_functions(inputs: Inputs, seed: int) -> dict[str, Callable[[], EdgeTable]]:
    """Return benchmark callables for every generation case."""
    return {
        "strength_poisson": lambda: sample_strength_poisson(
            inputs.x_strength,
            inputs.y_strength,
            seed=seed,
        ),
        "strength_multinomial": lambda: sample_strength_multinomial(
            inputs.x_strength,
            inputs.y_strength,
            total_events=inputs.total_events,
            seed=seed,
        ),
        "strength_poisson_multinomial": lambda: sample_strength_poisson_multinomial(
            inputs.x_strength,
            inputs.y_strength,
            seed=seed,
        ),
        "strength_microcanonical": lambda: sample_strength_microcanonical(
            inputs.strength_out,
            inputs.strength_in,
            seed=seed,
        ),
        "strength_binomial": lambda: sample_strength_binomial(
            inputs.strength_binomial_fit.x,
            inputs.strength_binomial_fit.y,
            layers=10,
            seed=seed,
        ),
        "custom_poisson_sparse": lambda: sample_custom_poisson(
            inputs.custom_probabilities,
            total_events=inputs.total_events,
            seed=seed,
        ),
        "custom_multinomial_sparse": lambda: sample_custom_multinomial(
            inputs.custom_probabilities,
            total_events=inputs.total_events,
            seed=seed,
        ),
        "degree_events_poisson": lambda: sample_degree_events_poisson(
            inputs.degree_fit,
            total_events=inputs.total_events,
            seed=seed,
        ),
        "strength_cost_poisson": lambda: sample_strength_cost_poisson(
            inputs.strength_cost_fit,
            inputs.cost_sources,
            inputs.cost_targets,
            inputs.cost_values,
            seed=seed,
        ),
        "strength_cost_binomial": lambda: sample_strength_cost_binomial(
            inputs.strength_cost_fit,
            inputs.cost_sources,
            inputs.cost_targets,
            inputs.cost_values,
            layers=10,
            seed=seed,
        ),
        "strength_edges_poisson": lambda: sample_strength_edges_poisson(
            inputs.strength_edges_fit,
            seed=seed,
        ),
        "strength_edges_binomial": lambda: sample_strength_edges_binomial(
            inputs.strength_edges_fit,
            layers=10,
            seed=seed,
        ),
        "strength_degree_poisson": lambda: sample_strength_degree_poisson(
            inputs.strength_degree_fit,
            seed=seed,
        ),
    }


def _write_row(csv_path: Path, jsonl_path: Path, row: dict[str, object]) -> None:
    """Append a result row to both CSV and JSONL outputs."""
    csv_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not csv_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _candidate_description(n: int, case: str, sparse_degree: int) -> str:
    """Describe the work expected for a benchmark case."""
    if case in {"custom_pij_poisson_sparse", "custom_pij_multinomial_sparse"}:
        return f"candidate_pairs≈{n * min(sparse_degree, n):,} sparse support"
    if case == "fixed_strength_microcanonical":
        return "candidate_pairs=stubs only, no all-pairs scan"
    return f"candidate_pairs≈{n * n:,} all-pairs stream"


def _start_progress_reporter(
    *,
    n: int,
    case: str,
    repeat: int,
    start: float,
    done: threading.Event,
    interval: float,
) -> threading.Thread | None:
    """Start a lightweight progress reporter for long Rust calls."""
    if interval <= 0.0:
        return None

    def report() -> None:
        while not done.wait(interval):
            elapsed = time.perf_counter() - start
            print(
                f"      still running case={case} N={n} repeat={repeat} "
                f"elapsed={elapsed:.1f}s rss={_max_rss_mb():.1f}MiB",
                flush=True,
            )

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    return thread


def _run_one(
    run_id: str,
    n: int,
    case: str,
    repeat: int,
    fn: Callable[[], EdgeTable],
    *,
    sparse_degree: int,
    progress_interval: float,
) -> dict[str, object]:
    """Run one benchmark case and return a serializable result row."""
    description = _candidate_description(n, case, sparse_degree)
    print(f"    case={case:<34} repeat={repeat}", flush=True)
    print(f"      start: {description} rss={_max_rss_mb():.1f}MiB", flush=True)
    gc.collect()
    start = time.perf_counter()
    done = threading.Event()
    reporter = _start_progress_reporter(
        n=n,
        case=case,
        repeat=repeat,
        start=start,
        done=done,
        interval=progress_interval,
    )
    status = "ok"
    error = ""
    sampled_edges = 0
    total_weight = 0
    try:
        sample = fn()
        sampled_edges = int(sample.num_edges)
        total_weight = int(sample.total_events)
    except Exception as exc:
        status = "error"
        error = repr(exc)
    finally:
        done.set()
        if reporter is not None:
            reporter.join(timeout=0.1)
    seconds = time.perf_counter() - start
    print(
        f"      done:  {status} {seconds:.3f}s edges={sampled_edges} "
        f"T={total_weight} rss={_max_rss_mb():.1f}MiB",
        flush=True,
    )
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "n": n,
        "case": case,
        "repeat": repeat,
        "seconds": seconds,
        "sampled_edges": sampled_edges,
        "total_weight": total_weight,
        "max_rss_mb": _max_rss_mb(),
        "status": status,
        "error": error,
    }


def run_benchmarks(
    *,
    nodes_value: str,
    cases_value: str,
    repeats: int,
    average_strength: int,
    sparse_degree: int,
    seed: int,
    output_dir_value: str,
    progress_interval: float,
) -> tuple[Path, Path]:
    """Run the configured benchmark matrix."""
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"streaming_generation_{run_id}.csv"
    jsonl_path = output_dir / f"streaming_generation_{run_id}.jsonl"
    meta_path = output_dir / f"streaming_generation_{run_id}.metadata.json"

    nodes = _parse_csv_ints(nodes_value)
    cases = _parse_csv_strings(cases_value)
    unknown = sorted(set(cases) - set(DEFAULT_CASES))
    if unknown:
        msg = f"unknown cases: {', '.join(unknown)}"
        raise ValueError(msg)

    metadata = {
        "run_id": run_id,
        "nodes": nodes,
        "cases": cases,
        "repeats": repeats,
        "average_strength": average_strength,
        "sparse_degree": sparse_degree,
        "seed": seed,
        "progress_interval": progress_interval,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("ODME streaming generation benchmark")
    print(f"  output CSV : {csv_path}")
    print(f"  output JSONL: {jsonl_path}")
    print(f"  nodes      : {nodes}")
    print(f"  cases      : {cases}")
    print(f"  repeats    : {repeats}")
    print(f"  progress   : every {progress_interval}s while a case is running")
    print("  warning    : all-pairs generators are O(N^2) time but streaming memory")

    for n in nodes:
        print("\n" + "=" * 80)
        print(f"N={n}: preparing synthetic parameters", flush=True)
        inputs = _make_inputs(n, average_strength, sparse_degree)
        print(
            f"N={n}: T={inputs.total_events} sparse_pij_entries="
            f"{inputs.custom_probabilities.num_edges} rss={_max_rss_mb():.1f}MiB",
            flush=True,
        )
        for repeat in range(repeats):
            functions = _case_functions(inputs, seed + repeat)
            for case in cases:
                row = _run_one(
                    run_id,
                    n,
                    case,
                    repeat,
                    functions[case],
                    sparse_degree=sparse_degree,
                    progress_interval=progress_interval,
                )
                _write_row(csv_path, jsonl_path, row)

    print("\nDone.")
    print(f"Results written to {csv_path}")
    print(f"JSONL written to {jsonl_path}")
    return csv_path, jsonl_path


@app.command()
def main(
    nodes: Annotated[
        str,
        typer.Option("--nodes", help="Comma-separated node counts."),
    ] = ",".join(str(n) for n in DEFAULT_NODES),
    cases: Annotated[
        str,
        typer.Option(
            "--cases",
            help=(
                f"Comma-separated cases or 'all'. Available: {', '.join(DEFAULT_CASES)}"
            ),
        ),
    ] = "all",
    repeats: Annotated[int, typer.Option("--repeats", help="Repeats per case.")] = 5,
    average_strength: Annotated[
        int,
        typer.Option(
            "--average-strength",
            help="Expected total events per node for synthetic inputs.",
        ),
    ] = 20,
    sparse_degree: Annotated[
        int,
        typer.Option(
            "--sparse-degree",
            help="Candidate entries per source for sparse custom p_ij benchmarks.",
        ),
    ] = 20,
    seed: Annotated[int, typer.Option("--seed", help="Base RNG seed.")] = 42,
    output_dir: Annotated[
        str,
        typer.Option(
            "--output-dir",
            help="Directory for incremental CSV/JSONL benchmark results.",
        ),
    ] = "benchmarks/results",
    progress_interval: Annotated[
        float,
        typer.Option(
            "--progress-interval",
            help=(
                "Seconds between progress messages while one case is running; "
                "0 disables."
            ),
        ),
    ] = 5.0,
) -> None:
    """Run the benchmark CLI."""
    run_benchmarks(
        nodes_value=nodes,
        cases_value=cases,
        repeats=repeats,
        average_strength=average_strength,
        sparse_degree=sparse_degree,
        seed=seed,
        output_dir_value=output_dir,
        progress_interval=progress_interval,
    )


if __name__ == "__main__":
    app()
