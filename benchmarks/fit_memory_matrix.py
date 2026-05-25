"""Benchmark memory and time for every supported fitting case.

Each case runs in a fresh Python process under GNU ``time -v`` so the reported
peak RSS is per fit case instead of one shared benchmark process. The generated
input is the canonical PA-geographic network used by the ODME benchmark suite.
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


@dataclass(frozen=True)
class FitMemoryResult:
    """One per-process fitting benchmark result."""

    node_count: int
    family: str
    constraint: str
    layers: int | None
    status: str
    process_seconds: float
    peak_rss_kb: int | None
    inner_fit_seconds: float | None = None
    converged: bool | None = None
    iterations: int | None = None
    message: str = ""


def main() -> None:
    """Run all requested fitting cases in fresh subprocesses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="1000", help="Comma-separated N values")
    parser.add_argument("--families", default="me,b,w,wnb")
    parser.add_argument(
        "--constraints",
        default="strength,strength-cost,strength-edges,strength-degree,degree-events",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--average-degree", type=float, default=8.0)
    parser.add_argument("--events-per-edge", type=float, default=8.0)
    parser.add_argument("--tolerance-factor", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=50_000)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    node_counts = parse_csv_ints(args.nodes)
    families = parse_csv(args.families)
    constraints = parse_csv(args.constraints)
    results: list[FitMemoryResult] = []
    with tempfile.TemporaryDirectory(prefix="odme-fit-memory-") as tmp_name:
        workdir = Path(tmp_name)
        for node_count in node_counts:
            for family in families:
                for constraint in constraints:
                    results.append(
                        run_case(
                            workdir=workdir,
                            node_count=node_count,
                            family=family,
                            constraint=constraint,
                            seed=args.seed + node_count,
                            average_degree=args.average_degree,
                            events_per_edge=args.events_per_edge,
                            tolerance_factor=args.tolerance_factor,
                            max_iterations=args.max_iterations,
                            timeout=args.timeout,
                        )
                    )
                    print(json.dumps(asdict(results[-1])), file=sys.stderr)
    print(json.dumps([asdict(result) for result in results], indent=2))


def run_case(
    *,
    workdir: Path,
    node_count: int,
    family: str,
    constraint: str,
    seed: int,
    average_degree: float,
    events_per_edge: float,
    tolerance_factor: float,
    max_iterations: int,
    timeout: float,
) -> FitMemoryResult:
    """Run one fit case in a fresh process."""
    case_dir = workdir / f"n{node_count}-{family}-{constraint}"
    case_dir.mkdir(parents=True, exist_ok=True)
    output = case_dir / "result.json"
    script = case_dir / "case.py"
    script.write_text(
        case_script(
            node_count=node_count,
            family=family,
            constraint=constraint,
            seed=seed,
            average_degree=average_degree,
            events_per_edge=events_per_edge,
            tolerance_factor=tolerance_factor,
            max_iterations=max_iterations,
            output=output,
            repo_root=Path.cwd(),
        )
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [time_binary(), "-v", sys.executable, str(script)],
        cwd=case_dir,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    process_seconds = time.perf_counter() - started
    peak_rss = parse_peak_rss(completed.stderr)
    payload: dict[str, Any] = {}
    if output.exists():
        payload = json.loads(output.read_text())
    if completed.returncode != 0:
        return FitMemoryResult(
            node_count=node_count,
            family=family,
            constraint=constraint,
            layers=payload.get("layers"),
            status="error",
            process_seconds=process_seconds,
            peak_rss_kb=peak_rss,
            inner_fit_seconds=payload.get("inner_fit_seconds"),
            converged=payload.get("converged"),
            iterations=payload.get("iterations"),
            message=(completed.stderr + completed.stdout)[-500:],
        )
    return FitMemoryResult(
        node_count=node_count,
        family=family,
        constraint=constraint,
        layers=payload.get("layers"),
        status=payload.get("status", "ok"),
        process_seconds=process_seconds,
        peak_rss_kb=peak_rss,
        inner_fit_seconds=payload.get("inner_fit_seconds"),
        converged=payload.get("converged"),
        iterations=payload.get("iterations"),
        message=payload.get("message", ""),
    )


def case_script(
    *,
    node_count: int,
    family: str,
    constraint: str,
    seed: int,
    average_degree: float,
    events_per_edge: float,
    tolerance_factor: float,
    max_iterations: int,
    output: Path,
    repo_root: Path,
) -> str:
    """Return the subprocess Python code for one case."""
    return textwrap.dedent(
        f"""
        import json
        import time
        import warnings
        import sys
        from pathlib import Path

        import numpy as np

        sys.path.insert(0, {str(repo_root)!r})
        from benchmarks.dispatch import REGISTRY, build_fit_kwargs
        from odme.utilities.synthetic import derive_synthetic_constraints, generate_pa_geographic_network

        output = Path({str(output)!r})
        family = {family!r}
        constraint = {constraint!r}
        network = generate_pa_geographic_network(
            {node_count},
            average_degree={average_degree!r},
            events_per_edge={events_per_edge!r},
            seed={seed},
            self_loops=False,
        )
        constraints = derive_synthetic_constraints(network)
        layers = constraints.binomial_layers if family == "b" else 3 if family == "wnb" else None
        result = {{"layers": layers}}
        dispatch = REGISTRY.get((family, constraint))
        if dispatch is None:
            result.update(status="unsupported", message="not registered")
            output.write_text(json.dumps(result))
            raise SystemExit(0)
        strength_tol = {tolerance_factor!r} * max(float(constraints.strength_out.max()), 1.0)
        degree_tol = max({tolerance_factor!r} * max(float(constraints.degree_out.max()), 1.0), 1.0)
        tolerance = max(strength_tol, 1.0) if family in {{"w", "wnb"}} else strength_tol
        kwargs = build_fit_kwargs(
            family,
            constraint,
            constraints,
            network,
            self_loops=False,
            tolerance=tolerance,
            degree_tolerance=degree_tol,
            max_iterations={max_iterations},
            layers=layers,
        )
        try:
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = dispatch.fit(**kwargs)
            result.update(
                status="ok" if bool(getattr(fit, "converged", True)) else "not_converged",
                inner_fit_seconds=time.perf_counter() - started,
                converged=bool(getattr(fit, "converged", True)),
                iterations=int(getattr(fit, "iterations", 0)),
            )
        except Exception as exc:
            result.update(status="error", message=str(exc)[:500])
        output.write_text(json.dumps(result))
        """
    )


def parse_csv(value: str) -> list[str]:
    """Parse comma-separated strings."""
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_csv_ints(value: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(part) for part in parse_csv(value)]


def time_binary() -> str:
    """Return GNU time binary."""
    return shutil.which("time") or "/usr/bin/time"


def parse_peak_rss(stderr: str) -> int | None:
    """Parse GNU time maximum RSS."""
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    main()
