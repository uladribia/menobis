"""Modern MENoBiS end-to-end benchmarks.

The benchmark uses the PA geographic synthetic generator and exercises the modern
Rust/Python stack: derive constraints, fit a null model, sample one null
network, and estimate upper-tail filtering false-positive rate from null samples.

Measures: time (wall + CPU), memory (Python heap + RSS), convergence, precision,
and parallelism factor (CPU/wall ratio).
"""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from menobis.data.frames import EdgeTable
from menobis.models.spec import Constraint, ModelFamily
from menobis.routing import filter_model, fit_model, sample_model
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

app = typer.Typer(help="MENoBiS PA-geographic E2E benchmarks.")

FAMILIES: tuple[str, ...] = ("me", "b", "w")
CONSTRAINTS: tuple[str, ...] = (
    "strength",
    "strength-cost",
    "strength-edges",
    "strength-degree",
)
REGIMES: tuple[str, ...] = ("sparse", "saturated")


# --- Measurement infrastructure ---


@dataclass(frozen=True)
class Measurement:
    """Timing and memory measurement for one stage."""

    wall_seconds: float
    cpu_seconds: float
    parallel_factor: float
    memory_python_peak_mb: float
    memory_rss_peak_mb: float


@dataclass
class BenchmarkRow:
    """One benchmark measurement row."""

    stage: str
    node_count: int
    family: str
    constraint: str
    self_loops: bool
    regime: str
    known_pair_fraction: float
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    parallel_factor: float = 1.0
    memory_python_peak_mb: float = 0.0
    memory_rss_peak_mb: float = 0.0
    converged: bool | None = None
    iterations: int | None = None
    max_iterations_hit: bool | None = None
    max_s_out_err: float | None = None
    max_s_in_err: float | None = None
    rel_s_err: float | None = None
    max_k_out_err: float | None = None
    max_k_in_err: float | None = None
    edge_count_err: float | None = None
    cost_err: float | None = None
    false_positive_rate: float | None = None
    sampled_edges: int | None = None
    status: str = "ok"
    message: str = ""


def _get_rss_mb() -> float:
    """Read VmRSS from /proc/self/status (Linux)."""
    try:
        with Path("/proc/self/status").open() as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return 0.0


def _measure(fn, *, track_memory: bool = True) -> tuple[Any, Measurement]:
    """Measure wall, CPU, memory for a callable."""
    if track_memory:
        tracemalloc.start()
        tracemalloc.reset_peak()
    rss_before = _get_rss_mb()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    result = fn()

    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start

    python_peak_mb = 0.0
    if track_memory:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        python_peak_mb = peak_bytes / 1e6

    rss_after = _get_rss_mb()
    rss_delta = max(0.0, rss_after - rss_before)

    parallel_factor = cpu_elapsed / wall_elapsed if wall_elapsed > 0.001 else 1.0

    return result, Measurement(
        wall_seconds=wall_elapsed,
        cpu_seconds=cpu_elapsed,
        parallel_factor=parallel_factor,
        memory_python_peak_mb=python_peak_mb,
        memory_rss_peak_mb=rss_delta,
    )


# --- Regime parameters ---


def _regime_params(regime: str, node_count: int) -> dict[str, float]:
    """Return PA-geographic parameters for a regime."""
    if regime == "sparse":
        return {"average_degree": 3.0, "events_per_edge": 3.0}
    # saturated: degree near N-1
    return {"average_degree": 0.85 * (node_count - 1), "events_per_edge": 8.0}


# --- Model dispatch via router ---

_FAMILY_MAP = {"me": ModelFamily.ME, "b": ModelFamily.B, "w": ModelFamily.W}
_CONSTRAINT_MAP = {
    "strength": Constraint.STRENGTH,
    "strength-cost": Constraint.STRENGTH_COST,
    "strength-edges": Constraint.STRENGTH_EDGES,
    "strength-degree": Constraint.STRENGTH_DEGREE,
}


def _fit_case(
    family: str,
    constraint: str,
    network,
    constraints_data,
    *,
    self_loops: bool,
    tolerance: float,
    known_fraction: float = 0.0,
    seed: int = 0,
):
    """Fit a model (full or partial) via the unified router."""
    import warnings

    c = constraints_data
    kwargs: dict[str, Any] = {
        "family": _FAMILY_MAP[family],
        "constraint": _CONSTRAINT_MAP[constraint],
        "strength_out": c.strength_out,
        "strength_in": c.strength_in,
        "self_loops": self_loops,
        "tolerance": tolerance,
    }
    if constraint == "strength-degree":
        kwargs["degree_out"] = c.degree_out
        kwargs["degree_in"] = c.degree_in
    if constraint == "strength-edges":
        kwargs["target_edges"] = c.total_edges
    if constraint == "strength-cost":
        kwargs["coord_x"] = network.x
        kwargs["coord_y"] = network.y
        kwargs["target_cost"] = c.total_cost
    if family == "b":
        kwargs["layers"] = c.binomial_layers

    if known_fraction > 0.0:
        n_known = max(1, int(known_fraction * network.edges.num_edges))
        rng = np.random.default_rng(seed)
        indices = rng.choice(network.edges.num_edges, size=n_known, replace=False)
        kwargs["known_source"] = network.edges.source[indices].astype(np.uint64)
        kwargs["known_target"] = network.edges.target[indices].astype(np.uint64)
        kwargs["known_rate"] = network.edges.weight[indices].astype(np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_model(**kwargs)


def _sample_case(family: str, constraint: str, fit, network, c, seed: int) -> EdgeTable:
    """Sample one null network via the unified router."""
    kwargs: dict[str, Any] = {
        "family": _FAMILY_MAP[family],
        "constraint": _CONSTRAINT_MAP[constraint],
        "fit": fit,
        "seed": seed,
    }
    if family == "b":
        kwargs["layers"] = c.binomial_layers
    if constraint == "strength-cost":
        kwargs["coord_x"] = network.x
        kwargs["coord_y"] = network.y
    return sample_model(**kwargs)


def _filter_case(family, constraint, sample, fit, network, c, alpha):
    """Filter one sample via the unified router."""
    kwargs: dict[str, Any] = {
        "family": _FAMILY_MAP[family],
        "constraint": _CONSTRAINT_MAP[constraint],
        "fit": fit,
        "alpha": alpha,
        "tail": "upper",
    }
    if family == "b":
        kwargs["layers"] = c.binomial_layers
    if constraint == "strength-cost":
        kwargs["coord_x"] = network.x
        kwargs["coord_y"] = network.y
    return filter_model(sample, **kwargs)


# --- Precision computation ---


def _compute_partial_precision(
    fit, constraint: str, c, network, *, self_loops: bool
) -> dict[str, float | None]:
    """Compute precision for a PartialFitResult (source/target/rate format).

    The partial fit returns expected rates for free (fitted) pairs.
    Known pairs are edges present in network.edges but not in the free-pair
    list; they contribute their observed weights directly.
    """
    result: dict[str, float | None] = {
        "max_s_out_err": None,
        "max_s_in_err": None,
        "rel_s_err": None,
        "max_k_out_err": None,
        "max_k_in_err": None,
        "edge_count_err": None,
        "cost_err": None,
    }

    fit_src = np.asarray(fit.source, dtype=np.int64)
    fit_tgt = np.asarray(fit.target, dtype=np.int64)
    fit_rate = np.asarray(fit.rate, dtype=np.float64)

    # Build a lookup set of free (source, target) pairs
    free_pairs: set[tuple[int, int]] = set()
    for i in range(len(fit_src)):
        free_pairs.add((int(fit_src[i]), int(fit_tgt[i])))

    # Determine total node count
    max_node = 0
    if len(fit_src) > 0:
        max_node = max(max_node, int(fit_src.max()), int(fit_tgt.max()))
    if network.edges is not None and len(network.edges) > 0:
        max_node = max(
            max_node,
            int(network.edges.source.max()),
            int(network.edges.target.max()),
        )
    n = max(max_node + 1, len(c.strength_out))

    # 1. Expected strengths from free-pair rates
    s_out = np.zeros(n, dtype=np.float64)
    s_in = np.zeros(n, dtype=np.float64)
    np.add.at(s_out, fit_src, fit_rate)
    np.add.at(s_in, fit_tgt, fit_rate)

    # 2. Add known-pair contributions (edges in network but NOT in free pairs)
    for i_edge in range(len(network.edges)):
        s = int(network.edges.source[i_edge])
        t = int(network.edges.target[i_edge])
        w = float(network.edges.weight[i_edge])
        if (s, t) not in free_pairs:
            s_out[s] += w
            s_in[t] += w

    # Strength errors
    max_s = max(float(c.strength_out.max()), float(c.strength_in.max()), 1.0)
    s_out_err = float(np.abs(s_out[: len(c.strength_out)] - c.strength_out).max())
    s_in_err = float(np.abs(s_in[: len(c.strength_in)] - c.strength_in).max())
    result["max_s_out_err"] = s_out_err
    result["max_s_in_err"] = s_in_err
    result["rel_s_err"] = max(s_out_err, s_in_err) / max_s

    # Edge count: known edges are counted as 1 each (occupied).
    # Free pairs contribute P(t_ij > 0). For ME/Poisson free pairs
    # (non-ZI), this is 1 - exp(-rate). For ZI free pairs, this is
    # approximate. Known pairs count as 1.
    if constraint == "strength-edges":
        total_edges = float(len(network.edges) - len(free_pairs))  # known edges
        total_edges += float(np.sum(1.0 - np.exp(-fit_rate)))
        result["edge_count_err"] = abs(total_edges - c.total_edges)

    # Cost: free + known pairs, each weighted by Euclidean distance
    if constraint == "strength-cost":
        total_cost = 0.0
        for i in range(len(fit_src)):
            s, t = int(fit_src[i]), int(fit_tgt[i])
            if not self_loops and s == t:
                continue
            d = float(
                np.hypot(
                    float(network.x[s]) - float(network.x[t]),
                    float(network.y[s]) - float(network.y[t]),
                )
            )
            total_cost += float(fit_rate[i]) * d
        for i_edge in range(len(network.edges)):
            s = int(network.edges.source[i_edge])
            t = int(network.edges.target[i_edge])
            if (s, t) in free_pairs:
                continue
            if not self_loops and s == t:
                continue
            d = float(
                np.hypot(
                    float(network.x[s]) - float(network.x[t]),
                    float(network.y[s]) - float(network.y[t]),
                )
            )
            total_cost += float(network.edges.weight[i_edge]) * d
        result["cost_err"] = abs(total_cost - c.total_cost)

    return result


def _compute_precision(
    fit, constraint: str, constraints_data, network, *, self_loops: bool
) -> dict[str, float | None]:
    """Compute constraint precision metrics from the fitted model.

    Recomputes expected strengths/degrees/edges from fitted multipliers
    for all constraint types, giving consistent reporting.
    """
    c = constraints_data
    result: dict[str, float | None] = {
        "max_s_out_err": None,
        "max_s_in_err": None,
        "rel_s_err": None,
        "max_k_out_err": None,
        "max_k_in_err": None,
        "edge_count_err": None,
        "cost_err": None,
    }

    # PartialFitResult: use source/target/rate based computation
    if not hasattr(fit, "x") or fit.x is None:
        if hasattr(fit, "source") and fit.source is not None and len(fit.source) > 0:
            return _compute_partial_precision(
                fit, constraint, c, network, self_loops=self_loops
            )
        return result

    x, y = np.asarray(fit.x), np.asarray(fit.y)
    n = len(x)
    layers = getattr(fit, "layers", 1) or 1
    m = float(layers)
    family = getattr(fit, "family", None) or "poisson"

    # Determine lam (for edges constraints) and z/w (for degree constraints)
    lam = getattr(fit, "lam", None)
    z = np.asarray(fit.z) if hasattr(fit, "z") and fit.z is not None else None
    w = np.asarray(fit.w) if hasattr(fit, "w") and fit.w is not None else None
    gamma = getattr(fit, "gamma", None)

    # Build expected strengths, degrees, edge count from multipliers
    s_out = np.zeros(n)
    s_in = np.zeros(n)
    k_out = np.zeros(n)
    k_in = np.zeros(n)
    total_edges = 0.0
    total_cost = 0.0

    for i in range(n):
        for j in range(n):
            if not self_loops and i == j:
                continue

            # Compute q_ij
            if constraint == "strength-cost" and gamma is not None:
                d = float(
                    np.hypot(network.x[i] - network.x[j], network.y[i] - network.y[j])
                )
                q = float(x[i] * y[j]) * np.exp(-gamma * d)
            else:
                q = float(x[i] * y[j])

            if q <= 0.0:
                continue

            # Compute expected weight and occupation based on family + constraint
            if constraint == "strength" or constraint == "strength-cost":
                # Non-ZI: independent statistics
                if "binomial" in family or "Binomial" in str(type(fit)):
                    weight = m * q / (1.0 + q)
                    occ = 1.0  # not ZI
                elif "geometric" in family or "Geometric" in str(type(fit)):
                    if q >= 1.0:
                        continue
                    weight = m * q / (1.0 - q)
                    occ = 1.0
                else:
                    # ME/Poisson
                    weight = q
                    occ = 1.0
            elif constraint == "strength-edges":
                # ZI with scalar lam
                if lam is None:
                    continue
                if "binomial" in family or "Binomial" in str(type(fit)):
                    g = (1.0 + q) ** layers - 1.0
                    den = 1.0 + lam * g
                    occ = lam * g / den
                    weight = lam * m * q * (1.0 + q) ** (layers - 1) / den
                elif "geometric" in family or "Geometric" in str(type(fit)):
                    if q >= 1.0:
                        continue
                    g = (1.0 - q) ** (-layers) - 1.0
                    den = 1.0 + lam * g
                    occ = lam * g / den
                    weight = lam * m * q * (1.0 - q) ** (-layers - 1) / den
                else:
                    # ME
                    exp_q = np.exp(q)
                    g = exp_q - 1.0
                    den = 1.0 + lam * g
                    occ = lam * g / den
                    weight = lam * q * exp_q / den
            elif constraint == "strength-degree":
                # ZI with per-node z/w
                if z is None or w is None:
                    continue
                v = float(z[i] * w[j])
                if v <= 0.0:
                    continue
                if "binomial" in family or "Binomial" in str(type(fit)):
                    g = (1.0 + q) ** layers - 1.0
                    den = 1.0 + v * g
                    occ = v * g / den
                    weight = v * m * q * (1.0 + q) ** (layers - 1) / den
                elif "geometric" in family or "Geometric" in str(type(fit)):
                    if q >= 1.0:
                        continue
                    g = (1.0 - q) ** (-layers) - 1.0
                    den = 1.0 + v * g
                    occ = v * g / den
                    weight = v * m * q * (1.0 - q) ** (-layers - 1) / den
                else:
                    # ME
                    exp_q = np.exp(q)
                    g = exp_q - 1.0
                    den = 1.0 + v * g
                    occ = v * g / den
                    weight = v * q * exp_q / den
            else:
                continue

            s_out[i] += weight
            s_in[j] += weight
            k_out[i] += occ
            k_in[j] += occ
            total_edges += occ
            if constraint == "strength-cost" and gamma is not None:
                d = float(
                    np.hypot(network.x[i] - network.x[j], network.y[i] - network.y[j])
                )
                total_cost += weight * d

    # Compute errors
    max_s = max(c.strength_out.max(), c.strength_in.max(), 1.0)
    s_out_err = float(np.abs(s_out - c.strength_out).max())
    s_in_err = float(np.abs(s_in - c.strength_in).max())
    result["max_s_out_err"] = s_out_err
    result["max_s_in_err"] = s_in_err
    result["rel_s_err"] = max(s_out_err, s_in_err) / max_s

    if constraint == "strength-degree":
        k_out_err = float(np.abs(k_out - c.degree_out).max())
        k_in_err = float(np.abs(k_in - c.degree_in).max())
        result["max_k_out_err"] = k_out_err
        result["max_k_in_err"] = k_in_err

    if constraint == "strength-edges":
        result["edge_count_err"] = abs(total_edges - c.total_edges)

    if constraint == "strength-cost":
        result["cost_err"] = abs(total_cost - c.total_cost)

    return result


# --- Main benchmark runner ---


def _format_row(row: BenchmarkRow) -> str:
    """Format a single row for immediate display."""
    conv_str = ""
    if row.converged is not None:
        conv_str = "\u2713" if row.converged else "\u2717"
    iters_str = str(row.iterations) if row.iterations else ""
    max_s = ""
    if row.max_s_out_err is not None:
        max_s = f"{max(row.max_s_out_err, row.max_s_in_err or 0):.2e}"
    max_k = ""
    if row.max_k_out_err is not None:
        max_k = f"{max(row.max_k_out_err, row.max_k_in_err or 0):.2e}"
    edge_err = ""
    if row.edge_count_err is not None:
        edge_err = f"{row.edge_count_err:.2e}"
    cost_err = ""
    if row.cost_err is not None:
        cost_err = f"{row.cost_err:.2e}"
    kp_str = (
        f"{row.known_pair_fraction * 100:.0f}" if row.known_pair_fraction > 0 else "0"
    )
    return (
        f"{row.stage:<12} {row.node_count:>5} {row.family:<3} {row.constraint:<16} "
        f"{row.regime:<9} {kp_str:>4} {row.wall_seconds:>8.4f} {row.cpu_seconds:>8.4f} "
        f"{row.parallel_factor:>5.1f} {row.memory_python_peak_mb:>6.1f} "
        f"{row.memory_rss_peak_mb:>7.1f} {conv_str:>5} {iters_str:>6} "
        f"{max_s:>10} {max_k:>10} {edge_err:>10} {cost_err:>10} {row.status:<8}"
    )


def _print_header() -> None:
    """Print the table header."""
    header = (
        f"{'stage':<12} {'N':>5} {'fam':<3} {'constraint':<16} {'regime':<9} "
        f"{'kp%':>4} {'wall(s)':>8} {'cpu(s)':>8} {'parX':>5} "
        f"{'py_MB':>6} {'rss_MB':>7} {'conv':>5} {'iters':>6} "
        f"{'max_s_err':>10} {'max_k_err':>10} {'edge_err':>10} {'cost_err':>10} {'status':<8}"
    )
    typer.echo(header, err=True)
    typer.echo("\u2500" * len(header), err=True)


def run_benchmark(
    *,
    nodes: tuple[int, ...],
    families: tuple[str, ...],
    constraints: tuple[str, ...],
    regimes: tuple[str, ...],
    known_pairs: tuple[float, ...],
    seed: int,
    self_loops: bool,
    filter_samples: int,
    alpha: float,
    track_memory: bool,
    tol_sparse: float = 1e-6,
    tol_saturated: float = 1e-6,
    fit_only: bool = False,
    verbose: bool = True,
) -> list[BenchmarkRow]:
    """Run the benchmark matrix."""
    if verbose:
        _print_header()

    def _emit(row: BenchmarkRow) -> None:
        rows.append(row)
        if verbose:
            typer.echo(_format_row(row), err=True)

    rows: list[BenchmarkRow] = []

    for n_index, node_count in enumerate(nodes):
        for regime in regimes:
            params = _regime_params(regime, node_count)

            # Generate network
            net, gen_m = _measure(
                lambda _nc=node_count, _ni=n_index, _p=params: (
                    generate_pa_geographic_network(
                        _nc, seed=seed + _ni, self_loops=self_loops, **_p
                    )
                ),
                track_memory=track_memory,
            )
            c = derive_synthetic_constraints(net)
            _emit(
                BenchmarkRow(
                    stage="generate",
                    node_count=node_count,
                    family="all",
                    constraint="pa-geographic",
                    self_loops=self_loops,
                    regime=regime,
                    known_pair_fraction=0.0,
                    wall_seconds=gen_m.wall_seconds,
                    cpu_seconds=gen_m.cpu_seconds,
                    parallel_factor=gen_m.parallel_factor,
                    memory_python_peak_mb=gen_m.memory_python_peak_mb,
                    memory_rss_peak_mb=gen_m.memory_rss_peak_mb,
                    sampled_edges=net.edges.num_edges,
                    message=f"events={net.edges.total_events}",
                )
            )

            for family in families:
                for constraint in constraints:
                    for kp_fraction in known_pairs:
                        tolerance = tol_sparse if regime == "sparse" else tol_saturated
                        cell_rows = _run_one_cell(
                            family=family,
                            constraint=constraint,
                            regime=regime,
                            kp_fraction=kp_fraction,
                            network=net,
                            constraints_data=c,
                            self_loops=self_loops,
                            seed=seed,
                            filter_samples=filter_samples,
                            alpha=alpha,
                            track_memory=track_memory,
                            fit_only=fit_only,
                            tolerance=tolerance,
                        )
                        for r in cell_rows:
                            _emit(r)
    return rows


def _run_one_cell(
    *,
    family,
    constraint,
    regime,
    kp_fraction,
    network,
    constraints_data,
    self_loops,
    seed,
    filter_samples,
    alpha,
    track_memory,
    fit_only,
    tolerance,
) -> list[BenchmarkRow]:
    """Run fit (+ optional sample + filter) for one matrix cell."""
    rows: list[BenchmarkRow] = []
    node_count = len(constraints_data.strength_out)
    c = constraints_data

    # Fit
    try:
        fit, m = _measure(
            lambda: _fit_case(
                family,
                constraint,
                network,
                c,
                self_loops=self_loops,
                tolerance=tolerance,
                known_fraction=kp_fraction,
                seed=seed,
            ),
            track_memory=track_memory,
        )
    except Exception as exc:
        rows.append(
            BenchmarkRow(
                stage="fit",
                node_count=node_count,
                family=family,
                constraint=constraint,
                self_loops=self_loops,
                regime=regime,
                known_pair_fraction=kp_fraction,
                status="error",
                message=f"{type(exc).__name__}: {exc}",
            )
        )
        return rows

    if fit is None:
        rows.append(
            BenchmarkRow(
                stage="fit",
                node_count=node_count,
                family=family,
                constraint=constraint,
                self_loops=self_loops,
                regime=regime,
                known_pair_fraction=kp_fraction,
                status="unsupported",
                message="partial not available for this family/constraint",
            )
        )
        return rows

    # Convergence info
    converged = getattr(fit, "converged", None)
    iterations = getattr(fit, "iterations", None)

    # Precision
    precision = _compute_precision(fit, constraint, c, network, self_loops=self_loops)

    fit_row = BenchmarkRow(
        stage="fit",
        node_count=node_count,
        family=family,
        constraint=constraint,
        self_loops=self_loops,
        regime=regime,
        known_pair_fraction=kp_fraction,
        wall_seconds=m.wall_seconds,
        cpu_seconds=m.cpu_seconds,
        parallel_factor=m.parallel_factor,
        memory_python_peak_mb=m.memory_python_peak_mb,
        memory_rss_peak_mb=m.memory_rss_peak_mb,
        converged=converged,
        iterations=iterations,
        max_iterations_hit=(iterations is not None and not converged),
        **precision,
    )
    rows.append(fit_row)

    if fit_only or kp_fraction > 0:
        return rows

    # Sample
    try:
        sample, sm = _measure(
            lambda: _sample_case(family, constraint, fit, network, c, seed),
            track_memory=track_memory,
        )
        rows.append(
            BenchmarkRow(
                stage="sample",
                node_count=node_count,
                family=family,
                constraint=constraint,
                self_loops=self_loops,
                regime=regime,
                known_pair_fraction=kp_fraction,
                wall_seconds=sm.wall_seconds,
                cpu_seconds=sm.cpu_seconds,
                parallel_factor=sm.parallel_factor,
                memory_python_peak_mb=sm.memory_python_peak_mb,
                memory_rss_peak_mb=sm.memory_rss_peak_mb,
                sampled_edges=sample.num_edges if sample else None,
            )
        )
    except Exception as exc:
        rows.append(
            BenchmarkRow(
                stage="sample",
                node_count=node_count,
                family=family,
                constraint=constraint,
                self_loops=self_loops,
                regime=regime,
                known_pair_fraction=kp_fraction,
                status="error",
                message=f"{type(exc).__name__}: {exc}",
            )
        )
        return rows

    # Filter FPR
    if filter_samples > 0 and sample is not None:
        try:
            fp_total = 0
            edge_total = 0
            for offset in range(filter_samples):
                null_sample = _sample_case(
                    family, constraint, fit, network, c, seed + 10000 + offset
                )
                if null_sample is None:
                    continue
                result = _filter_case(
                    family, constraint, null_sample, fit, network, c, alpha
                )
                fp_total += result.upper.edges.num_edges
                edge_total += null_sample.num_edges
            fpr = fp_total / max(edge_total, 1)
            rows.append(
                BenchmarkRow(
                    stage="filter-fpr",
                    node_count=node_count,
                    family=family,
                    constraint=constraint,
                    self_loops=self_loops,
                    regime=regime,
                    known_pair_fraction=kp_fraction,
                    false_positive_rate=fpr,
                )
            )
        except Exception as exc:
            rows.append(
                BenchmarkRow(
                    stage="filter-fpr",
                    node_count=node_count,
                    family=family,
                    constraint=constraint,
                    self_loops=self_loops,
                    regime=regime,
                    known_pair_fraction=kp_fraction,
                    status="error",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )

    return rows


# --- CLI commands ---


@app.command("all")
def all_command(
    nodes: Annotated[
        str, typer.Option("--nodes", help="Comma-separated node sizes.")
    ] = "100,500",
    families: Annotated[
        str, typer.Option("--families", help="Comma-separated families: me,b,w.")
    ] = "me,b,w",
    constraints: Annotated[
        str, typer.Option("--constraints", help="Comma-separated constraints.")
    ] = "strength,strength-cost,strength-edges,strength-degree",
    regimes: Annotated[
        str, typer.Option("--regime", help="Comma-separated regimes: sparse,saturated.")
    ] = "sparse,saturated",
    known_pairs: Annotated[
        str, typer.Option("--known-pairs", help="Comma-separated known-pair fractions.")
    ] = "0.0,0.05,0.20",
    seed: Annotated[int, typer.Option("--seed", help="Base random seed.")] = 10_000,
    self_loops: Annotated[bool, typer.Option("--self-loops/--no-self-loops")] = False,
    filter_samples: Annotated[
        int, typer.Option("--filter-samples", help="Null samples for FPR.")
    ] = 3,
    alpha: Annotated[float, typer.Option("--alpha", help="Upper-tail alpha.")] = 0.05,
    no_memory: Annotated[
        bool, typer.Option("--no-memory", help="Skip memory profiling.")
    ] = False,
    tol_sparse: Annotated[
        float, typer.Option("--tol-sparse", help="Solver tolerance for sparse regime.")
    ] = 1e-6,
    tol_saturated: Annotated[
        float,
        typer.Option("--tol-saturated", help="Solver tolerance for saturated regime."),
    ] = 1e-6,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "benchmarks/results/e2e-modern.json"
    ),
    output_json: Annotated[
        bool, typer.Option("--json", help="Print JSON to stdout.")
    ] = False,
) -> None:
    """Run generate → fit → sample → filter-FPR benchmarks."""
    rows = run_benchmark(
        nodes=tuple(_parse_ints(nodes)),
        families=tuple(_parse_tokens(families, FAMILIES, "family")),
        constraints=tuple(_parse_tokens(constraints, CONSTRAINTS, "constraint")),
        regimes=tuple(_parse_tokens(regimes, REGIMES, "regime")),
        known_pairs=tuple(_parse_floats(known_pairs)),
        seed=seed,
        self_loops=self_loops,
        filter_samples=filter_samples,
        alpha=alpha,
        track_memory=not no_memory,
        tol_sparse=tol_sparse,
        tol_saturated=tol_saturated,
    )
    _save_and_display(rows, output, output_json)


@app.command("fit")
def fit_command(
    nodes: Annotated[str, typer.Option("--nodes")] = "100,500",
    families: Annotated[str, typer.Option("--families")] = "me,b,w",
    constraints: Annotated[
        str, typer.Option("--constraints")
    ] = "strength,strength-cost,strength-edges,strength-degree",
    regimes: Annotated[str, typer.Option("--regime")] = "sparse,saturated",
    known_pairs: Annotated[str, typer.Option("--known-pairs")] = "0.0,0.05,0.20",
    seed: Annotated[int, typer.Option("--seed")] = 10_000,
    self_loops: Annotated[bool, typer.Option("--self-loops/--no-self-loops")] = False,
    no_memory: Annotated[bool, typer.Option("--no-memory")] = False,
    tol_sparse: Annotated[float, typer.Option("--tol-sparse")] = 1e-6,
    tol_saturated: Annotated[float, typer.Option("--tol-saturated")] = 1e-6,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "benchmarks/results/fit-modern.json"
    ),
) -> None:
    """Run only the fit stage."""
    rows = run_benchmark(
        nodes=tuple(_parse_ints(nodes)),
        families=tuple(_parse_tokens(families, FAMILIES, "family")),
        constraints=tuple(_parse_tokens(constraints, CONSTRAINTS, "constraint")),
        regimes=tuple(_parse_tokens(regimes, REGIMES, "regime")),
        known_pairs=tuple(_parse_floats(known_pairs)),
        seed=seed,
        self_loops=self_loops,
        filter_samples=0,
        alpha=0.05,
        track_memory=not no_memory,
        tol_sparse=tol_sparse,
        tol_saturated=tol_saturated,
        fit_only=True,
    )
    _save_and_display(rows, output, False)


@app.command("compare")
def compare_command(
    nodes: Annotated[str, typer.Option("--nodes")] = "500",
    families: Annotated[str, typer.Option("--families")] = "me,w",
    constraints: Annotated[str, typer.Option("--constraints")] = "strength-degree",
    seed: Annotated[int, typer.Option("--seed")] = 10_000,
    self_loops: Annotated[bool, typer.Option("--self-loops/--no-self-loops")] = False,
    no_memory: Annotated[bool, typer.Option("--no-memory")] = False,
    tol_sparse: Annotated[float, typer.Option("--tol-sparse")] = 1e-6,
    tol_saturated: Annotated[float, typer.Option("--tol-saturated")] = 1e-6,
) -> None:
    """Compare regimes side-by-side."""
    rows = run_benchmark(
        nodes=tuple(_parse_ints(nodes)),
        families=tuple(_parse_tokens(families, FAMILIES, "family")),
        constraints=tuple(_parse_tokens(constraints, CONSTRAINTS, "constraint")),
        regimes=("sparse", "saturated"),
        known_pairs=(0.0,),
        seed=seed,
        self_loops=self_loops,
        filter_samples=0,
        alpha=0.05,
        track_memory=not no_memory,
        tol_sparse=tol_sparse,
        tol_saturated=tol_saturated,
        fit_only=True,
    )
    _print_comparison(rows)


# --- Display ---


def _save_and_display(
    rows: list[BenchmarkRow], output: Path, output_json: bool
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if output_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        _print_summary(rows)
        typer.echo(f"\nResults: {output}", err=True)


def _print_summary(rows: list[BenchmarkRow]) -> None:
    """Print summary totals after streaming rows."""
    converged_count = sum(1 for r in rows if r.converged is True)
    error_count = sum(1 for r in rows if r.status == "error")
    total_wall = sum(r.wall_seconds for r in rows)
    typer.echo(
        f"\nTotals: {len(rows)} rows | {converged_count} converged | {error_count} errors | wall={total_wall:.1f}s",
        err=True,
    )


def _print_comparison(rows: list[BenchmarkRow]) -> None:
    """Print regime comparison."""
    fit_rows = [r for r in rows if r.stage == "fit"]
    by_key: dict[tuple, list[BenchmarkRow]] = {}
    for row in fit_rows:
        key = (row.node_count, row.family, row.constraint)
        by_key.setdefault(key, []).append(row)

    for key, group in sorted(by_key.items()):
        n, fam, con = key
        typer.echo(f"\nRegime Comparison (N={n}, {fam}, {con}):", err=True)
        for row in sorted(group, key=lambda r: r.regime):
            par_str = f"x{row.parallel_factor:.1f} par"
            iters_str = f"{row.iterations} iters" if row.iterations else "—"
            s_err = (
                f"max_s={row.max_s_out_err:.2e}"
                if row.max_s_out_err is not None
                else ""
            )
            typer.echo(
                f"  {row.regime:<10} {row.wall_seconds:.3f}s  {par_str}  {iters_str}  {s_err}",
                err=True,
            )


# --- Parsing helpers ---


def _parse_ints(value: str) -> list[int]:
    result = [int(p.strip()) for p in value.split(",") if p.strip()]
    if not result:
        raise typer.BadParameter("expected at least one integer")
    return result


def _parse_floats(value: str) -> list[float]:
    result = [float(p.strip()) for p in value.split(",") if p.strip()]
    if not result:
        raise typer.BadParameter("expected at least one float")
    return result


def _parse_tokens(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    result = [p.strip().lower() for p in value.split(",") if p.strip()]
    bad = sorted(set(result) - set(allowed))
    if bad:
        raise typer.BadParameter(f"unknown {name}: {','.join(bad)}")
    return result


if __name__ == "__main__":
    app()
