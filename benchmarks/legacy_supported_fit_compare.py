"""Compare every legacy-supported fitter case with modern ODME where possible.

Run with SciPy available because archived W and nonlinear fitters import SciPy:

    uv run --with scipy python -m benchmarks.legacy_supported_fit_compare --nodes 1000

The script extracts the removed Python fitters from git history, converts them
with Python 3.12's ``lib2to3``, patches Python-3 incompatibilities in a temporary
copy, and runs each legacy case in a fresh subprocess under GNU ``time -v``.
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

from odme.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

LEGACY_CASES = (
    "me-strength",
    "b-strength",
    "w-strength",
    "me-strength-cost",
    "me-strength-edges",
    "b-strength-edges",
    "me-strength-degree",
    "b-strength-degree",
    "degree",
)


@dataclass(frozen=True)
class Timing:
    """Process timing and peak resident memory."""

    seconds: float
    peak_rss_kb: int | None


@dataclass(frozen=True)
class LegacyCaseResult:
    """Comparison result for one legacy-supported case."""

    node_count: int
    case: str
    layers: int | None
    modern: Timing
    legacy: Timing
    modern_status: str
    legacy_status: str
    modern_inner_seconds: float | None
    legacy_inner_seconds: float | None
    modern_iterations: int | None
    max_observable_abs_diff: float | None
    modern_max_error: float | None
    legacy_max_error: float | None
    message: str = ""


def main() -> None:
    """Run all legacy-supported comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="1000")
    parser.add_argument("--cases", default=",".join(LEGACY_CASES))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--average-degree", type=float, default=8.0)
    parser.add_argument("--events-per-edge", type=float, default=8.0)
    parser.add_argument("--max-iterations", type=int, default=50_000)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--legacy-ref", default=None)
    parser.add_argument("--legacy-only", action="store_true", help="skip modern reruns")
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()

    node_counts = [int(part) for part in args.nodes.split(",") if part]
    cases = [part.strip() for part in args.cases.split(",") if part.strip()]
    unknown = sorted(set(cases) - set(LEGACY_CASES))
    if unknown:
        raise SystemExit(f"unknown legacy cases: {unknown}")

    with tempfile.TemporaryDirectory(prefix="odme-legacy-supported-") as tmp_name:
        workdir = Path(tmp_name)
        if args.keep_workdir:
            workdir = Path(tempfile.mkdtemp(prefix="odme-legacy-supported-keep-"))
        try:
            legacy_pkg = prepare_legacy_package(workdir, args.legacy_ref)
            results: list[LegacyCaseResult] = []
            for node_count in node_counts:
                generated = generate_inputs(
                    workdir,
                    node_count=node_count,
                    seed=args.seed + node_count,
                    average_degree=args.average_degree,
                    events_per_edge=args.events_per_edge,
                )
                for case in cases:
                    result = run_case(
                        workdir=workdir,
                        legacy_pkg=legacy_pkg,
                        generated=generated,
                        node_count=node_count,
                        case=case,
                        max_iterations=args.max_iterations,
                        timeout=args.timeout,
                        legacy_only=args.legacy_only,
                    )
                    results.append(result)
                    print(json.dumps(asdict(result)), file=sys.stderr)
            print(json.dumps([asdict(result) for result in results], indent=2))
        finally:
            if args.keep_workdir:
                print(f"kept workdir: {workdir}", file=sys.stderr)


def prepare_legacy_package(workdir: Path, legacy_ref: str | None) -> Path:
    """Extract and Python-3 patch archived multi_edge_fitter modules."""
    ref = legacy_ref or infer_legacy_ref()
    archive = workdir / "legacy.tar"
    run(["git", "archive", "--format=tar", "-o", str(archive), ref])
    legacy_root = workdir / "legacy"
    legacy_root.mkdir()
    run(["tar", "xf", str(archive), "-C", str(legacy_root)])
    package = legacy_root / "2. Model Fitting" / "multi_edge_fitter"
    if not package.exists():
        raise SystemExit(f"legacy package not found at {ref!r}")
    for path in package.glob("*.py"):
        path.write_text(path.read_text().expandtabs(4))
    if shutil.which("python3.12") is None:
        raise SystemExit("python3.12 is required for lib2to3 conversion")
    converted = subprocess.run(
        ["python3.12", "-m", "lib2to3", "-w", *map(str, package.glob("*.py"))],
        text=True,
        capture_output=True,
    )
    if converted.returncode != 0:
        raise SystemExit(converted.stderr + converted.stdout)
    for path in package.glob("*.py"):
        text = path.read_text()
        text = text.replace("np.int", "int")
        text = text.replace("np.float128", "np.longdouble")
        text = text.replace("float128", "np.longdouble")
        text = text.replace("np.float", "float")
        text = text.replace("np.longdouble", "np.longdouble")
        path.write_text(text)
    return package


def infer_legacy_ref() -> str:
    """Find a git ref containing the archived Python fitter package."""
    path = "2. Model Fitting/multi_edge_fitter/fitter_s.py"
    commits = run(
        ["git", "rev-list", "-n", "20", "HEAD", "--", path]
    ).stdout.splitlines()
    for commit in commits:
        for candidate in (commit, f"{commit}^"):
            exists = subprocess.run(
                ["git", "cat-file", "-e", f"{candidate}:{path}"],
                text=True,
                capture_output=True,
            )
            if exists.returncode == 0:
                return candidate
    raise SystemExit("could not infer legacy ref")


def generate_inputs(
    workdir: Path,
    *,
    node_count: int,
    seed: int,
    average_degree: float,
    events_per_edge: float,
) -> Path:
    """Generate and persist shared synthetic constraints."""
    data_dir = workdir / f"data-n{node_count}"
    data_dir.mkdir(exist_ok=True)
    network = generate_pa_geographic_network(
        node_count,
        average_degree=average_degree,
        events_per_edge=events_per_edge,
        seed=seed,
        self_loops=False,
    )
    constraints = derive_synthetic_constraints(network)
    np.save(data_dir / "sout.npy", constraints.strength_out)
    np.save(data_dir / "sin.npy", constraints.strength_in)
    np.save(data_dir / "kout.npy", constraints.degree_out)
    np.save(data_dir / "kin.npy", constraints.degree_in)
    np.save(data_dir / "xcoord.npy", network.x)
    np.save(data_dir / "ycoord.npy", network.y)
    np.save(data_dir / "total_cost.npy", np.asarray([constraints.total_cost]))
    np.save(data_dir / "total_edges.npy", np.asarray([constraints.total_edges]))
    np.save(data_dir / "layers.npy", np.asarray([constraints.binomial_layers]))
    return data_dir


def run_case(
    *,
    workdir: Path,
    legacy_pkg: Path,
    generated: Path,
    node_count: int,
    case: str,
    max_iterations: int,
    timeout: float,
    legacy_only: bool,
) -> LegacyCaseResult:
    """Run modern and legacy scripts for one case."""
    case_dir = workdir / f"run-n{node_count}-{case}"
    case_dir.mkdir(exist_ok=True)
    modern_json = case_dir / "modern.json"
    legacy_json = case_dir / "legacy.json"
    modern_script = case_dir / "modern.py"
    legacy_script = case_dir / "legacy.py"
    modern_script.write_text(
        _modern_script(case, generated, modern_json, max_iterations)
    )
    legacy_script.write_text(
        _legacy_script(case, legacy_pkg, generated, legacy_json, max_iterations)
    )

    if legacy_only:
        modern = Timing(seconds=0.0, peak_rss_kb=None)
        modern_data = {"status": "skipped", "message": "legacy-only run"}
    else:
        modern = timed(
            [sys.executable, str(modern_script)], cwd=case_dir, timeout=timeout
        )
        modern_data = read_payload(modern_json)
    legacy = timed([sys.executable, str(legacy_script)], cwd=case_dir, timeout=timeout)
    legacy_data = read_payload(legacy_json)
    return LegacyCaseResult(
        node_count=node_count,
        case=case,
        layers=modern_data.get("layers") or legacy_data.get("layers"),
        modern=modern,
        legacy=legacy,
        modern_status=modern_data.get("status", "process_error"),
        legacy_status=legacy_data.get("status", "process_error"),
        modern_inner_seconds=modern_data.get("seconds"),
        legacy_inner_seconds=legacy_data.get("seconds"),
        modern_iterations=modern_data.get("iterations"),
        max_observable_abs_diff=observable_diff(modern_data, legacy_data),
        modern_max_error=modern_data.get("max_error"),
        legacy_max_error=legacy_data.get("max_error"),
        message=(
            legacy_data.get("message")
            if modern_data.get("status") == "skipped"
            else modern_data.get("message") or legacy_data.get("message") or ""
        ),
    )


def _modern_script(case: str, data_dir: Path, output: Path, max_iterations: int) -> str:
    return textwrap.dedent(
        f"""
        import json
        import sys
        import time
        import warnings
        from pathlib import Path
        import numpy as np

        from odme.models import (
            fit_degree_bernoulli,
            fit_strength_binomial,
            fit_strength_cost_poisson_coordinates,
            fit_strength_degree_binomial,
            fit_strength_degree_poisson,
            fit_strength_edges_binomial,
            fit_strength_edges_poisson,
            fit_strength_geometric,
            fit_strength_poisson,
        )

        case = {case!r}
        data_dir = Path({str(data_dir)!r})
        output = Path({str(output)!r})
        sout = np.load(data_dir / 'sout.npy')
        sin_ = np.load(data_dir / 'sin.npy')
        kout = np.load(data_dir / 'kout.npy')
        kin = np.load(data_dir / 'kin.npy')
        layers = int(np.load(data_dir / 'layers.npy')[0])
        payload = {{'layers': layers if case.startswith('b-') else None}}
        try:
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if case == 'me-strength':
                    fit = fit_strength_poisson(sout, sin_, self_loops=False, max_iterations={max_iterations})
                    obs = np.outer(fit.x, fit.y); np.fill_diagonal(obs, 0.0)
                    target_out, target_in = sout, sin_
                elif case == 'b-strength':
                    fit = fit_strength_binomial(sout, sin_, layers=layers, self_loops=False, max_iterations={max_iterations})
                    q = np.outer(fit.x, fit.y); np.fill_diagonal(q, 0.0); obs = layers*q/(1+q)
                    target_out, target_in = sout, sin_
                elif case == 'w-strength':
                    fit = fit_strength_geometric(sout, sin_, self_loops=False, max_iterations={max_iterations})
                    q = np.outer(fit.x, fit.y); np.fill_diagonal(q, 0.0); obs = q/(1-q)
                    target_out, target_in = sout, sin_
                elif case == 'me-strength-cost':
                    fit = fit_strength_cost_poisson_coordinates(sout, sin_, np.load(data_dir/'xcoord.npy'), np.load(data_dir/'ycoord.npy'), float(np.load(data_dir/'total_cost.npy')[0]), self_loops=False, max_iterations={max_iterations})
                    x = np.load(data_dir/'xcoord.npy'); y = np.load(data_dir/'ycoord.npy')
                    src, tgt = np.meshgrid(np.arange(len(x)), np.arange(len(x)), indexing='ij')
                    d = np.hypot(x[src]-x[tgt], y[src]-y[tgt])
                    obs = np.outer(fit.x, fit.y)*np.exp(-fit.gamma*d); np.fill_diagonal(obs, 0.0)
                    target_out, target_in = sout, sin_
                elif case == 'me-strength-edges':
                    fit = fit_strength_edges_poisson(sout, sin_, float(np.load(data_dir/'total_edges.npy')[0]), self_loops=False, max_iterations={max_iterations})
                    q = np.outer(fit.x, fit.y); np.fill_diagonal(q, 0.0); ell = fit.lam
                    obs = ell*q*np.exp(q)/(1+ell*(np.exp(q)-1))
                    target_out, target_in = sout, sin_
                elif case == 'b-strength-edges':
                    fit = fit_strength_edges_binomial(sout, sin_, float(np.load(data_dir/'total_edges.npy')[0]), layers=layers, self_loops=False, max_iterations={max_iterations})
                    q = np.outer(fit.x, fit.y); np.fill_diagonal(q, 0.0); ell = fit.lam
                    obs = ell*layers*q*(1+q)**(layers-1)/(1+ell*((1+q)**layers-1))
                    target_out, target_in = sout, sin_
                elif case == 'me-strength-degree':
                    fit = fit_strength_degree_poisson(sout, sin_, kout, kin, self_loops=False, max_iterations={max_iterations})
                    q = np.outer(fit.x, fit.y); z = np.outer(fit.z, fit.w); np.fill_diagonal(q, 0.0); np.fill_diagonal(z, 0.0)
                    obs = z*q*np.exp(q)/(1+z*(np.exp(q)-1))
                    target_out, target_in = sout, sin_
                elif case == 'b-strength-degree':
                    fit = fit_strength_degree_binomial(sout, sin_, kout, kin, layers=layers, self_loops=False, max_iterations={max_iterations})
                    q = np.outer(fit.x, fit.y); z = np.outer(fit.z, fit.w); np.fill_diagonal(q, 0.0); np.fill_diagonal(z, 0.0)
                    obs = z*layers*q*(1+q)**(layers-1)/(1+z*((1+q)**layers-1))
                    target_out, target_in = sout, sin_
                elif case == 'degree':
                    fit = fit_degree_bernoulli(kout, kin, self_loops=False, max_iterations={max_iterations})
                    q = np.outer(fit.x, fit.y); np.fill_diagonal(q, 0.0); obs = q/(1+q)
                    target_out, target_in = kout, kin
                else:
                    raise ValueError(case)
            max_error = max(float(np.max(np.abs(obs.sum(axis=1)-target_out))), float(np.max(np.abs(obs.sum(axis=0)-target_in))))
            payload.update(status='ok', seconds=time.perf_counter()-started, iterations=int(getattr(fit, 'iterations', 0)), max_error=max_error, observable=obs.ravel().tolist())
        except Exception as exc:
            payload.update(status='error', message=str(exc)[:500])
        output.write_text(json.dumps(payload))
        """
    )


def _legacy_script(
    case: str, package: Path, data_dir: Path, output: Path, max_iterations: int
) -> str:
    return textwrap.dedent(
        f"""
        import importlib.util
        import json
        import sys
        import time
        from pathlib import Path
        import numpy as np

        package = Path({str(package)!r})
        def load(name):
            spec = importlib.util.spec_from_file_location(name, str(package / (name + '.py')))
            mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod

        case = {case!r}
        data_dir = Path({str(data_dir)!r})
        output = Path({str(output)!r})
        sout = np.load(data_dir / 'sout.npy')
        sin_ = np.load(data_dir / 'sin.npy')
        kout = np.load(data_dir / 'kout.npy')
        kin = np.load(data_dir / 'kin.npy')
        layers = int(np.load(data_dir / 'layers.npy')[0])
        n = len(sout)
        payload = {{'layers': layers if case.startswith('b-') else None}}
        try:
            started = time.perf_counter()
            if case in ('me-strength', 'b-strength'):
                mod = load('fitter_s')
                legacy_case = 'B' if case == 'b-strength' else 'ME'
                kwargs = {{'M': layers}} if legacy_case == 'B' else {{}}
                x, y = mod.balance_xy(sin_, sout, tol=1e-9, tol_c=1e-9, maxreps={max_iterations}, verbose=False, selfs=False, case=legacy_case, inds_selfs=(np.diag_indices(n)[0]*n+np.diag_indices(n)[1]).tolist(), **kwargs)
                q = np.outer(x, y); np.fill_diagonal(q, 0.0)
                obs = layers*q/(1+q) if legacy_case == 'B' else q
                target_out, target_in = sout, sin_
            elif case == 'w-strength':
                mod = load('fitter_s')
                solver = mod.Edge_Maximizer(sin_, sout, selfs=False, M=1, case='W')
                solver.precondition(verbose=False)
                qres = solver.minimize('TNC', verbose=False, eta=0.5, xtol=1e-9, tol=1e-9, epsi=1e-10, bmax=1.0)
                x, y = solver.var_result().T
                q = np.outer(x, y); np.fill_diagonal(q, 0.0); obs = q/(1-q)
                target_out, target_in = sout, sin_
            elif case == 'me-strength-cost':
                mod = load('fitter_grav')
                xcoord = np.load(data_dir/'xcoord.npy'); ycoord = np.load(data_dir/'ycoord.npy')
                src, tgt = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
                d = np.hypot(xcoord[src]-xcoord[tgt], ycoord[src]-ycoord[tgt])
                x, y, gamma = mod.fit_gamma(sin_, sout, float(np.load(data_dir/'total_cost.npy')[0]), d, tol_s=1e-9, tol_gamma=1e-9, maxreps={max_iterations}, selfs=False, verbose=False)
                obs = np.outer(x, y)*np.exp(-gamma*d); np.fill_diagonal(obs, 0.0)
                target_out, target_in = sout, sin_
            elif case in ('me-strength-edges', 'b-strength-edges'):
                mod = load('fitter_E')
                agg = case == 'b-strength-edges'
                x, y, lam = mod.fit_lambda(sin_, sout, float(np.load(data_dir/'total_edges.npy')[0]), tol_s=1e-9, tol_gamma=1e-5, maxreps={max_iterations}, selfs=False, verbose=False, agg=agg, M=layers)
                q = np.outer(x, y); np.fill_diagonal(q, 0.0)
                if agg:
                    obs = lam*layers*q*(1+q)**(layers-1)/(1+lam*((1+q)**layers-1))
                else:
                    obs = lam*q*np.exp(q)/(1+lam*(np.exp(q)-1))
                target_out, target_in = sout, sin_
            elif case in ('me-strength-degree', 'b-strength-degree'):
                mod = load('fitter_sk')
                agg = case == 'b-strength-degree'
                x, y, z, w = mod.balance_xyzw(sin_, sout, kin, kout, tol=1e-9, tol_c=1e-6, maxreps={max_iterations}, verbose=False, selfs=False, print_c=False, agg=agg, M=layers)
                q = np.outer(x, y); zz = np.outer(z, w); np.fill_diagonal(q, 0.0); np.fill_diagonal(zz, 0.0)
                if agg:
                    obs = zz*layers*q*(1+q)**(layers-1)/(1+zz*((1+q)**layers-1))
                else:
                    obs = zz*q*np.exp(q)/(1+zz*(np.exp(q)-1))
                target_out, target_in = sout, sin_
            elif case == 'degree':
                mod = load('fitter_k')
                x, y = mod.balance_xy(kin, kout, tol=1e-9, tol_c=1e-9, maxreps={max_iterations}, verbose=False, selfs=False, print_c=False)
                q = np.outer(x, y); np.fill_diagonal(q, 0.0); obs = q/(1+q)
                target_out, target_in = kout, kin
            else:
                raise ValueError(case)
            max_error = max(float(np.max(np.abs(obs.sum(axis=1)-target_out))), float(np.max(np.abs(obs.sum(axis=0)-target_in))))
            payload.update(status='ok', seconds=time.perf_counter()-started, max_error=max_error, observable=obs.ravel().tolist())
        except Exception as exc:
            payload.update(status='error', message=str(exc)[:500])
        output.write_text(json.dumps(payload))
        """
    )


def timed(command: list[str], *, cwd: Path, timeout: float) -> Timing:
    """Run a command with GNU time."""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [time_binary(), "-v", *command],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        seconds = time.perf_counter() - started
        stderr = (
            exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
        )
        return Timing(seconds=seconds, peak_rss_kb=parse_peak_rss(stderr))
    seconds = time.perf_counter() - started
    return Timing(seconds=seconds, peak_rss_kb=parse_peak_rss(completed.stderr))


def observable_diff(modern: dict[str, Any], legacy: dict[str, Any]) -> float | None:
    """Compare serialized expected observables."""
    if modern.get("status") != "ok" or legacy.get("status") != "ok":
        return None
    return float(
        np.max(
            np.abs(np.asarray(modern["observable"]) - np.asarray(legacy["observable"]))
        )
    )


def read_payload(path: Path) -> dict[str, Any]:
    """Read JSON if produced."""
    if not path.exists():
        return {"status": "process_error", "message": "no output JSON"}
    return json.loads(path.read_text())


def time_binary() -> str:
    """Return GNU time binary."""
    return shutil.which("time") or "/usr/bin/time"


def parse_peak_rss(stderr: str) -> int | None:
    """Parse peak RSS."""
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    return int(match.group(1)) if match else None


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run command or fail."""
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr + completed.stdout)
    return completed


if __name__ == "__main__":
    main()
