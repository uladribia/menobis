"""Compare archived thesis analyzer with modern ODME on identical inputs.

This is intentionally a benchmark/regression utility, not a packaged ODME API.
It extracts the removed C analyzer from git history, compiles it in a temporary
folder, generates realistic PA-geographic networks, and compares key node-level
outputs with modern Rust-backed ODME statistics.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from odme.utilities.synthetic import generate_pa_geographic_network


@dataclass(frozen=True)
class Timing:
    """Process timing and peak resident memory."""

    seconds: float
    peak_rss_kb: int | None


@dataclass(frozen=True)
class ComparisonResult:
    """One legacy-vs-modern analyzer comparison."""

    node_count: int
    edge_count: int
    total_events: int
    modern: Timing
    legacy: Timing
    modern_load_and_compute_seconds: float
    max_strength_out_abs_diff: float
    max_strength_in_abs_diff: float
    max_degree_out_abs_diff: float
    max_degree_in_abs_diff: float
    max_y2_out_abs_diff: float
    max_y2_in_abs_diff: float


def main() -> None:
    """Run the comparison benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="100,500", help="Comma-separated N values")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--average-degree", type=float, default=8.0)
    parser.add_argument("--events-per-edge", type=float, default=8.0)
    parser.add_argument(
        "--legacy-ref", default=None, help="Git ref containing legacy folders"
    )
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()

    node_counts = [int(value) for value in args.nodes.split(",") if value]
    with tempfile.TemporaryDirectory(prefix="odme-legacy-compare-") as tmp_name:
        workdir = Path(tmp_name)
        if args.keep_workdir:
            workdir = Path(tempfile.mkdtemp(prefix="odme-legacy-compare-keep-"))
        try:
            analyzer = prepare_legacy_analyzer(workdir, args.legacy_ref)
            results = [
                compare_one(
                    analyzer=analyzer,
                    workdir=workdir,
                    node_count=node_count,
                    seed=args.seed + node_count,
                    average_degree=args.average_degree,
                    events_per_edge=args.events_per_edge,
                )
                for node_count in node_counts
            ]
            print(json.dumps([asdict(result) for result in results], indent=2))
        finally:
            if args.keep_workdir:
                print(f"kept workdir: {workdir}", file=sys.stderr)


def prepare_legacy_analyzer(workdir: Path, legacy_ref: str | None) -> Path:
    """Extract and compile the archived legacy C analyzer."""
    if shutil.which("gcc") is None or shutil.which("make") is None:
        raise SystemExit("gcc and make are required for legacy comparison")
    ref = legacy_ref or infer_legacy_ref()
    archive = workdir / "legacy.tar"
    run(["git", "archive", "--format=tar", "-o", str(archive), ref])
    legacy_root = workdir / "legacy"
    legacy_root.mkdir()
    run(["tar", "xf", str(archive), "-C", str(legacy_root)])
    analyzer_dir = legacy_root / "1. Network analysis"
    if not analyzer_dir.exists():
        raise SystemExit(f"legacy analyzer not found at ref {ref!r}")
    compiled = subprocess.run(
        ["make", "CC=gcc"], cwd=analyzer_dir, text=True, capture_output=True
    )
    if compiled.returncode != 0:
        raise SystemExit(
            "failed to compile legacy analyzer; install GSL headers/libs "
            "(for example libgsl-dev) and retry\n" + compiled.stderr[-1000:]
        )
    return analyzer_dir / "MultiEdgeAnalyzer"


def infer_legacy_ref() -> str:
    """Find a git ref where the removed legacy analyzer still exists."""
    path = "1. Network analysis/src/main.c"
    commits = run(
        ["git", "rev-list", "-n", "20", "HEAD", "--", path]
    ).stdout.splitlines()
    for commit in commits:
        candidates = [commit, f"{commit}^"]
        for candidate in candidates:
            exists = subprocess.run(
                ["git", "cat-file", "-e", f"{candidate}:{path}"],
                text=True,
                capture_output=True,
            )
            if exists.returncode == 0:
                return candidate
    raise SystemExit("could not infer a legacy git ref; pass --legacy-ref")


def compare_one(
    *,
    analyzer: Path,
    workdir: Path,
    node_count: int,
    seed: int,
    average_degree: float,
    events_per_edge: float,
) -> ComparisonResult:
    """Compare one generated network."""
    case_dir = workdir / f"n{node_count}"
    case_dir.mkdir(exist_ok=True)
    network = generate_pa_geographic_network(
        node_count,
        average_degree=average_degree,
        events_per_edge=events_per_edge,
        seed=seed,
        self_loops=False,
    )
    edge_file = case_dir / "edges.tr"
    write_legacy_edge_file(
        edge_file, network.edges.source, network.edges.target, network.edges.weight
    )

    modern_json = case_dir / "modern.json"
    modern_script = case_dir / "modern_stats.py"
    modern_script.write_text(_modern_script(edge_file, modern_json))
    modern = timed([sys.executable, str(modern_script)], cwd=case_dir)
    modern_stats = json.loads(modern_json.read_text())

    legacy_run = case_dir / "legacy"
    legacy_run.mkdir(exist_ok=True)
    legacy = timed(
        [
            str(analyzer),
            "-N",
            str(node_count),
            "-d",
            "1",
            "-f",
            str(edge_file),
            "-h",
            "0",
            "-l",
            "0",
        ],
        cwd=legacy_run,
    )
    node_files = list(legacy_run.glob("*node_list.list"))
    if not node_files:
        raise RuntimeError(
            f"legacy analyzer produced no node_list.list for N={node_count}"
        )
    legacy_nodes = np.loadtxt(node_files[0], comments="#")
    if legacy_nodes.ndim == 1:
        legacy_nodes = legacy_nodes.reshape(1, -1)

    return ComparisonResult(
        node_count=node_count,
        edge_count=len(network.edges),
        total_events=network.edges.total_events,
        modern=modern,
        legacy=legacy,
        modern_load_and_compute_seconds=float(modern_stats["seconds"]),
        max_strength_out_abs_diff=max_abs(
            legacy_nodes[:, 3], modern_stats["strength_out"]
        ),
        max_strength_in_abs_diff=max_abs(
            legacy_nodes[:, 10], modern_stats["strength_in"]
        ),
        max_degree_out_abs_diff=max_abs(legacy_nodes[:, 1], modern_stats["degree_out"]),
        max_degree_in_abs_diff=max_abs(legacy_nodes[:, 8], modern_stats["degree_in"]),
        max_y2_out_abs_diff=max_abs(legacy_nodes[:, 4], modern_stats["y2_out"]),
        max_y2_in_abs_diff=max_abs(legacy_nodes[:, 11], modern_stats["y2_in"]),
    )


def _modern_script(edge_file: Path, output_file: Path) -> str:
    return textwrap.dedent(
        f"""
        import json
        import time
        from pathlib import Path
        import numpy as np
        from odme.analysis import compute_all_stats
        from odme.data import normalize_edges

        started = time.perf_counter()
        raw = np.loadtxt({str(edge_file)!r}, dtype=np.uint64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        edges = normalize_edges(raw[:, 0], raw[:, 1], raw[:, 2])
        stats = compute_all_stats(edges)
        seconds = time.perf_counter() - started
        data = {{
            "seconds": seconds,
            "strength_out": stats.strength_out.tolist(),
            "strength_in": stats.strength_in.tolist(),
            "degree_out": stats.degree_out.tolist(),
            "degree_in": stats.degree_in.tolist(),
            "y2_out": stats.y2_out.tolist(),
            "y2_in": stats.y2_in.tolist(),
        }}
        Path({str(output_file)!r}).write_text(json.dumps(data))
        """
    )


def timed(command: list[str], *, cwd: Path) -> Timing:
    """Run a command and parse GNU time peak RSS when available."""
    time_bin = shutil.which("time") or "/usr/bin/time"
    started = time.perf_counter()
    completed = subprocess.run(
        [time_bin, "-v", *command], cwd=cwd, text=True, capture_output=True
    )
    seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:] + completed.stdout[-1000:])
    return Timing(seconds=seconds, peak_rss_kb=parse_peak_rss(completed.stderr))


def parse_peak_rss(stderr: str) -> int | None:
    """Parse GNU time maximum RSS."""
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    return int(match.group(1)) if match else None


def write_legacy_edge_file(
    path: Path,
    source: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    weight: np.ndarray[Any, Any],
) -> None:
    """Write the whitespace legacy edge-list format."""
    with path.open("w") as handle:
        for src, dst, value in zip(source, target, weight, strict=True):
            handle.write(f"{int(src)} {int(dst)} {int(value)}\n")


def max_abs(left: np.ndarray[Any, Any], right: list[float]) -> float:
    """Return maximum absolute difference between arrays."""
    return float(
        np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))
    )


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and fail with captured output."""
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr + completed.stdout)
    return completed


if __name__ == "__main__":
    main()
