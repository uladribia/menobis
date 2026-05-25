"""Compare archived thesis strength fitters with modern ODME solvers.

The benchmark extracts the removed Python fitter from git history, converts it
with Python 3.12's ``lib2to3`` in a temporary directory, and runs the archived
``fitter_s.balance_xy`` solver on the same realistic PA-geographic constraints
used by modern ODME. It reports constraint recovery, expectation differences,
time, and peak RSS for ME/Poisson and B/Binomial fixed-strength fits.
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
from typing import Any, Literal

import numpy as np

from odme.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

Family = Literal["me", "b"]


@dataclass(frozen=True)
class Timing:
    """Process timing and peak resident memory."""

    seconds: float
    peak_rss_kb: int | None


@dataclass(frozen=True)
class FitComparisonResult:
    """One legacy-vs-modern fitter comparison."""

    node_count: int
    family: Family
    layers: int
    total_events: int
    modern: Timing
    legacy: Timing
    modern_inner_seconds: float
    legacy_inner_seconds: float
    modern_max_strength_error: float
    legacy_max_strength_error: float
    max_expected_weight_abs_diff: float


def main() -> None:
    """Run legacy-vs-modern fitter benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="100,500", help="Comma-separated N values")
    parser.add_argument("--families", default="me,b", help="Comma-separated: me,b")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--average-degree", type=float, default=8.0)
    parser.add_argument("--events-per-edge", type=float, default=8.0)
    parser.add_argument(
        "--legacy-ref", default=None, help="Git ref containing legacy fitter"
    )
    parser.add_argument("--max-iterations", type=int, default=50_000)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()

    node_counts = [int(value) for value in args.nodes.split(",") if value]
    families = [value.strip() for value in args.families.split(",") if value]
    invalid = sorted(set(families) - {"me", "b"})
    if invalid:
        raise SystemExit(f"unsupported families: {invalid}")

    with tempfile.TemporaryDirectory(prefix="odme-legacy-fit-") as tmp_name:
        workdir = Path(tmp_name)
        if args.keep_workdir:
            workdir = Path(tempfile.mkdtemp(prefix="odme-legacy-fit-keep-"))
        try:
            legacy_pkg = prepare_legacy_fitter(workdir, args.legacy_ref)
            results: list[FitComparisonResult] = []
            for node_count in node_counts:
                for family in families:
                    results.append(
                        compare_fit(
                            legacy_pkg=legacy_pkg,
                            workdir=workdir,
                            node_count=node_count,
                            family=family,  # type: ignore[arg-type]
                            seed=args.seed + node_count,
                            average_degree=args.average_degree,
                            events_per_edge=args.events_per_edge,
                            max_iterations=args.max_iterations,
                        )
                    )
            print(json.dumps([asdict(result) for result in results], indent=2))
        finally:
            if args.keep_workdir:
                print(f"kept workdir: {workdir}", file=sys.stderr)


def prepare_legacy_fitter(workdir: Path, legacy_ref: str | None) -> Path:
    """Extract, convert, and patch the archived Python fitter."""
    ref = legacy_ref or infer_legacy_ref()
    archive = workdir / "legacy.tar"
    run(["git", "archive", "--format=tar", "-o", str(archive), ref])
    legacy_root = workdir / "legacy"
    legacy_root.mkdir()
    run(["tar", "xf", str(archive), "-C", str(legacy_root)])
    package = legacy_root / "2. Model Fitting" / "multi_edge_fitter"
    fitter_s = package / "fitter_s.py"
    if not fitter_s.exists():
        raise SystemExit(f"legacy fitter_s.py not found at ref {ref!r}")
    fitter_s.write_text(fitter_s.read_text().expandtabs(4))

    if shutil.which("python3.12") is None:
        raise SystemExit("python3.12 is required to run lib2to3 conversion")
    converted = subprocess.run(
        ["python3.12", "-m", "lib2to3", "-w", str(fitter_s)],
        text=True,
        capture_output=True,
    )
    if converted.returncode != 0:
        raise SystemExit(converted.stderr + converted.stdout)

    text = fitter_s.read_text()
    text = text.replace(
        "from scipy import optimize as opt",
        "try:\n    from scipy import optimize as opt\nexcept ImportError:\n    opt = None",
    )
    text = text.replace(
        "from scipy import sparse",
        "try:\n    from scipy import sparse\nexcept ImportError:\n    sparse = None",
    )
    text = text.replace("np.int", "int")
    fitter_s.write_text(text)
    return package


def infer_legacy_ref() -> str:
    """Find a git ref where the removed legacy fitter still exists."""
    path = "2. Model Fitting/multi_edge_fitter/fitter_s.py"
    commits = run(
        ["git", "rev-list", "-n", "20", "HEAD", "--", path]
    ).stdout.splitlines()
    for commit in commits:
        for candidate in [commit, f"{commit}^"]:
            exists = subprocess.run(
                ["git", "cat-file", "-e", f"{candidate}:{path}"],
                text=True,
                capture_output=True,
            )
            if exists.returncode == 0:
                return candidate
    raise SystemExit("could not infer a legacy git ref; pass --legacy-ref")


def compare_fit(
    *,
    legacy_pkg: Path,
    workdir: Path,
    node_count: int,
    family: Family,
    seed: int,
    average_degree: float,
    events_per_edge: float,
    max_iterations: int,
) -> FitComparisonResult:
    """Compare one family and size."""
    case_dir = workdir / f"fit-n{node_count}-{family}"
    case_dir.mkdir(exist_ok=True)
    network = generate_pa_geographic_network(
        node_count,
        average_degree=average_degree,
        events_per_edge=events_per_edge,
        seed=seed,
        self_loops=False,
    )
    constraints = derive_synthetic_constraints(network)
    layers = constraints.binomial_layers if family == "b" else 1

    np.save(case_dir / "sout.npy", constraints.strength_out)
    np.save(case_dir / "sin.npy", constraints.strength_in)

    modern_json = case_dir / "modern.json"
    legacy_json = case_dir / "legacy.json"
    modern_script = case_dir / "modern_fit.py"
    legacy_script = case_dir / "legacy_fit.py"
    modern_script.write_text(
        _modern_script(family, layers, max_iterations, case_dir, modern_json)
    )
    legacy_script.write_text(
        _legacy_script(
            legacy_pkg, family, layers, max_iterations, case_dir, legacy_json
        )
    )

    modern = timed([sys.executable, str(modern_script)], cwd=case_dir)
    legacy = timed([sys.executable, str(legacy_script)], cwd=case_dir)
    modern_data = json.loads(modern_json.read_text())
    legacy_data = json.loads(legacy_json.read_text())

    return FitComparisonResult(
        node_count=node_count,
        family=family,
        layers=layers,
        total_events=constraints.total_events,
        modern=modern,
        legacy=legacy,
        modern_inner_seconds=float(modern_data["seconds"]),
        legacy_inner_seconds=float(legacy_data["seconds"]),
        modern_max_strength_error=float(modern_data["max_strength_error"]),
        legacy_max_strength_error=float(legacy_data["max_strength_error"]),
        max_expected_weight_abs_diff=expectation_diff(
            modern_data, legacy_data, family=family, layers=layers
        ),
    )


def _modern_script(
    family: Family, layers: int, max_iterations: int, case_dir: Path, output_file: Path
) -> str:
    fit_call = (
        f"fit_strength_binomial(sout, sin_, layers={layers}, self_loops=False, max_iterations={max_iterations})"
        if family == "b"
        else f"fit_strength_poisson(sout, sin_, self_loops=False, max_iterations={max_iterations})"
    )
    expectation = f"{layers} * q / (1.0 + q)" if family == "b" else "q"
    import_line = (
        "from odme.models import fit_strength_binomial"
        if family == "b"
        else "from odme.models import fit_strength_poisson"
    )
    return textwrap.dedent(
        f"""
        import json
        import time
        from pathlib import Path
        import numpy as np
        {import_line}

        sout = np.load({str(case_dir / "sout.npy")!r})
        sin_ = np.load({str(case_dir / "sin.npy")!r})
        started = time.perf_counter()
        fit = {fit_call}
        seconds = time.perf_counter() - started
        q = np.outer(fit.x, fit.y)
        np.fill_diagonal(q, 0.0)
        expected = {expectation}
        out_error = np.max(np.abs(expected.sum(axis=1) - sout))
        in_error = np.max(np.abs(expected.sum(axis=0) - sin_))
        data = {{
            "seconds": seconds,
            "max_strength_error": float(max(out_error, in_error)),
            "x": fit.x.tolist(),
            "y": fit.y.tolist(),
        }}
        Path({str(output_file)!r}).write_text(json.dumps(data))
        """
    )


def _legacy_script(
    legacy_pkg: Path,
    family: Family,
    layers: int,
    max_iterations: int,
    case_dir: Path,
    output_file: Path,
) -> str:
    case = "B" if family == "b" else "ME"
    kwargs = f", M={layers}" if family == "b" else ""
    expectation = f"{layers} * q / (1.0 + q)" if family == "b" else "q"
    return textwrap.dedent(
        f"""
        import importlib.util
        import json
        import sys
        import time
        from pathlib import Path
        import numpy as np

        spec = importlib.util.spec_from_file_location('legacy_fitter_s', {str(legacy_pkg / "fitter_s.py")!r})
        fitter_s = importlib.util.module_from_spec(spec)
        sys.modules['legacy_fitter_s'] = fitter_s
        spec.loader.exec_module(fitter_s)

        sout = np.load({str(case_dir / "sout.npy")!r})
        sin_ = np.load({str(case_dir / "sin.npy")!r})
        n = len(sout)
        inds_selfs = (np.diag_indices(n)[0] * n + np.diag_indices(n)[1]).tolist()
        started = time.perf_counter()
        x, y = fitter_s.balance_xy(
            sin_,
            sout,
            tol=1e-9,
            tol_c=1e-9,
            maxreps={max_iterations},
            verbose=False,
            selfs=False,
            case={case!r},
            inds_selfs=inds_selfs{kwargs},
        )
        seconds = time.perf_counter() - started
        q = np.outer(x, y)
        np.fill_diagonal(q, 0.0)
        expected = {expectation}
        out_error = np.max(np.abs(expected.sum(axis=1) - sout))
        in_error = np.max(np.abs(expected.sum(axis=0) - sin_))
        data = {{
            "seconds": seconds,
            "max_strength_error": float(max(out_error, in_error)),
            "x": x.tolist(),
            "y": y.tolist(),
        }}
        Path({str(output_file)!r}).write_text(json.dumps(data))
        """
    )


def timed(command: list[str], *, cwd: Path) -> Timing:
    """Run a command and parse GNU time peak RSS."""
    time_bin = shutil.which("time") or "/usr/bin/time"
    started = time.perf_counter()
    completed = subprocess.run(
        [time_bin, "-v", *command], cwd=cwd, text=True, capture_output=True
    )
    seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-3000:] + completed.stdout[-1000:])
    return Timing(seconds=seconds, peak_rss_kb=parse_peak_rss(completed.stderr))


def parse_peak_rss(stderr: str) -> int | None:
    """Parse GNU time maximum RSS."""
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    return int(match.group(1)) if match else None


def expectation_diff(
    modern_data: dict[str, Any],
    legacy_data: dict[str, Any],
    *,
    family: Family,
    layers: int,
) -> float:
    """Return max expected-weight difference without subprocess dense JSON."""
    modern_x = np.asarray(modern_data["x"], dtype=float)
    modern_y = np.asarray(modern_data["y"], dtype=float)
    legacy_x = np.asarray(legacy_data["x"], dtype=float)
    legacy_y = np.asarray(legacy_data["y"], dtype=float)
    modern_q = np.outer(modern_x, modern_y)
    legacy_q = np.outer(legacy_x, legacy_y)
    np.fill_diagonal(modern_q, 0.0)
    np.fill_diagonal(legacy_q, 0.0)
    if family == "b":
        modern_q = layers * modern_q / (1.0 + modern_q)
        legacy_q = layers * legacy_q / (1.0 + legacy_q)
    return float(np.max(np.abs(modern_q - legacy_q)))


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and fail with captured output."""
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr + completed.stdout)
    return completed


if __name__ == "__main__":
    main()
