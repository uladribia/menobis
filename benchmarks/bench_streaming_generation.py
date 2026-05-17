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
    sample_custom_multinomial,
    sample_custom_poisson,
    sample_degree_events_poisson,
    sample_strength_binomial,
    sample_strength_cost_binomial,
    sample_strength_cost_poisson,
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
    """Create model parameters without iterative fitting.

    The parameters target sparse sampled outputs while still forcing all-pairs
    generators to visit their full candidate support.
    """
    total_events = n * average_strength
    factor = math.sqrt(average_strength / n)
    x_strength = np.full(n, factor, dtype=np.float64)
    y_strength = np.full(n, factor, dtype=np.float64)
    strength_out = np.full(n, average_strength, dtype=np.uint64)
    strength_in = np.full(n, average_strength, dtype=np.uint64)

    custom = _make_sparse_probability_table(n, sparse_degree)

    occupation_probability = min(0.2, max(1e-12, sparse_degree / max(n, 1)))
    binary_z = occupation_probability / max(1.0 - occupation_probability, 1e-12)
    degree_factor = math.sqrt(binary_z)
    degree_fit = FitResult(
        node=np.arange(n, dtype=np.uint64),
        x=np.full(n, degree_factor, dtype=np.float64),
        y=np.full(n, degree_factor, dtype=np.float64),
    )

    strength_cost_fit = StrengthCostFit(
        node=np.arange(n, dtype=np.uint64),
        x=x_strength,
        y=y_strength,
        gamma=0.0,
        self_loops=True,
        converged=True,
        iterations=0,
    )

    zip_u = 1.0
    exp_minus_one = math.expm1(zip_u)
    lam = occupation_probability / max(
        (1.0 - occupation_probability) * exp_minus_one,
        1e-12,
    )
    strength_edges_fit = StrengthEdgesFit(
        node=np.arange(n, dtype=np.uint64),
        x=np.ones(n, dtype=np.float64),
        y=np.ones(n, dtype=np.float64),
        lam=lam,
        self_loops=True,
        converged=True,
        iterations=0,
    )
    zw = math.sqrt(lam)
    strength_degree_fit = StrengthDegreeFit(
        node=np.arange(n, dtype=np.uint64),
        x=np.ones(n, dtype=np.float64),
        y=np.ones(n, dtype=np.float64),
        z=np.full(n, zw, dtype=np.float64),
        w=np.full(n, zw, dtype=np.float64),
        self_loops=True,
        converged=True,
        iterations=0,
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
    )


def _case_functions(inputs: Inputs, seed: int) -> dict[str, Callable[[], EdgeTable]]:
    """Return benchmark callables for every generation case."""
    empty_u64 = np.array([], dtype=np.uint64)
    empty_f64 = np.array([], dtype=np.float64)
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
            inputs.x_strength,
            inputs.y_strength,
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
            empty_u64,
            empty_u64,
            empty_f64,
            seed=seed,
        ),
        "strength_cost_binomial": lambda: sample_strength_cost_binomial(
            inputs.strength_cost_fit,
            empty_u64,
            empty_u64,
            empty_f64,
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
