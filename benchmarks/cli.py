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
from menobis.filtering import (
    filter_strength_binomial,
    filter_strength_cost_binomial,
    filter_strength_cost_geometric,
    filter_strength_cost_poisson,
    filter_strength_degree_binomial,
    filter_strength_degree_geometric,
    filter_strength_degree_poisson,
    filter_strength_edges_binomial,
    filter_strength_edges_geometric,
    filter_strength_edges_poisson,
    filter_strength_geometric,
    filter_strength_poisson,
)
from menobis.models import (
    fit_strength_binomial,
    fit_strength_cost_binomial,
    fit_strength_cost_geometric,
    fit_strength_cost_poisson,
    fit_strength_degree_binomial,
    fit_strength_degree_geometric,
    fit_strength_degree_poisson,
    fit_strength_edges_binomial,
    fit_strength_edges_geometric,
    fit_strength_edges_poisson,
    fit_strength_geometric,
    fit_strength_poisson,
    sample_strength_binomial,
    sample_strength_cost_binomial,
    sample_strength_cost_geometric,
    sample_strength_cost_poisson,
    sample_strength_degree_binomial,
    sample_strength_degree_geometric,
    sample_strength_degree_poisson,
    sample_strength_edges_binomial,
    sample_strength_edges_geometric,
    sample_strength_edges_poisson,
    sample_strength_geometric,
    sample_strength_poisson,
)
from menobis.models.partial import (
    fit_partial_strength_binomial,
    fit_partial_strength_cost_binomial_coordinates,
    fit_partial_strength_cost_geometric_coordinates,
    fit_partial_strength_cost_poisson_coordinates,
    fit_partial_strength_degree_binomial,
    fit_partial_strength_degree_poisson,
    fit_partial_strength_edges_binomial,
    fit_partial_strength_edges_geometric,
    fit_partial_strength_edges_poisson,
    fit_partial_strength_poisson,
)
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
        return {"average_degree": 3.0, "events_per_edge": 1.5}
    # saturated: degree near N-1
    return {"average_degree": 0.85 * (node_count - 1), "events_per_edge": 5.0}


# --- Fitting dispatchers ---


def _fit_full(
    family: str, constraint: str, network, constraints_data, *, self_loops: bool
):
    """Dispatch to the correct full-fit function."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if constraint == "strength":
            return _fit_strength(family, constraints_data, self_loops)
        if constraint == "strength-cost":
            return _fit_strength_cost(family, network, constraints_data)
        if constraint == "strength-edges":
            return _fit_strength_edges(family, constraints_data, self_loops)
        if constraint == "strength-degree":
            return _fit_strength_degree(family, constraints_data, self_loops)
    msg = f"unknown constraint: {constraint}"
    raise ValueError(msg)


def _fit_strength(family: str, c, self_loops: bool):
    if family == "me":
        return fit_strength_poisson(
            c.strength_out, c.strength_in, self_loops=self_loops
        )
    if family == "b":
        return fit_strength_binomial(
            c.strength_out,
            c.strength_in,
            layers=c.binomial_layers,
            self_loops=self_loops,
        )
    return fit_strength_geometric(c.strength_out, c.strength_in, self_loops=self_loops)


def _fit_strength_cost(family: str, network, c):
    if family == "me":
        return fit_strength_cost_poisson(
            c.strength_out,
            c.strength_in,
            network.x,
            network.y,
            c.total_cost,
            self_loops=network.self_loops,
        )
    if family == "b":
        return fit_strength_cost_binomial(
            c.strength_out,
            c.strength_in,
            network.x,
            network.y,
            c.total_cost,
            layers=c.binomial_layers,
            self_loops=network.self_loops,
        )
    return fit_strength_cost_geometric(
        c.strength_out,
        c.strength_in,
        network.x,
        network.y,
        c.total_cost,
        self_loops=network.self_loops,
    )


def _fit_strength_edges(family: str, c, self_loops: bool):
    if family == "me":
        return fit_strength_edges_poisson(
            c.strength_out, c.strength_in, c.total_edges, self_loops=self_loops
        )
    if family == "b":
        return fit_strength_edges_binomial(
            c.strength_out,
            c.strength_in,
            c.total_edges,
            layers=c.binomial_layers,
            self_loops=self_loops,
        )
    return fit_strength_edges_geometric(
        c.strength_out, c.strength_in, c.total_edges, self_loops=self_loops
    )


def _fit_strength_degree(family: str, c, self_loops: bool):
    if family == "me":
        return fit_strength_degree_poisson(
            c.strength_out,
            c.strength_in,
            c.degree_out,
            c.degree_in,
            self_loops=self_loops,
            tolerance=1e-4,
        )
    if family == "b":
        return fit_strength_degree_binomial(
            c.strength_out,
            c.strength_in,
            c.degree_out,
            c.degree_in,
            layers=c.binomial_layers,
            self_loops=self_loops,
            tolerance=1e-4,
        )
    return fit_strength_degree_geometric(
        c.strength_out,
        c.strength_in,
        c.degree_out,
        c.degree_in,
        self_loops=self_loops,
        tolerance=1e-3,
        max_iterations=5000,
    )


# --- Partial fitting ---


def _fit_partial(
    family: str,
    constraint: str,
    network,
    constraints_data,
    known_fraction: float,
    *,
    self_loops: bool,
    seed: int,
):
    """Dispatch to the correct partial-fit function."""
    import warnings

    n_known = max(1, int(known_fraction * network.edges.num_edges))
    rng = np.random.default_rng(seed)
    indices = rng.choice(network.edges.num_edges, size=n_known, replace=False)
    known_src = network.edges.source[indices].astype(np.uint64)
    known_tgt = network.edges.target[indices].astype(np.uint64)
    known_rate = network.edges.weight[indices].astype(np.float64)
    c = constraints_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if constraint == "strength":
            if family == "me":
                return fit_partial_strength_poisson(
                    c.strength_out,
                    c.strength_in,
                    known_src,
                    known_tgt,
                    known_rate,
                    self_loops=self_loops,
                )
            if family == "b":
                return fit_partial_strength_binomial(
                    c.strength_out,
                    c.strength_in,
                    known_src,
                    known_tgt,
                    known_rate,
                    layers=c.binomial_layers,
                    self_loops=self_loops,
                )
        elif constraint == "strength-cost":
            if family == "me":
                return fit_partial_strength_cost_poisson_coordinates(
                    c.strength_out,
                    c.strength_in,
                    known_src,
                    known_tgt,
                    known_rate,
                    network.x,
                    network.y,
                    c.total_cost,
                    self_loops=self_loops,
                )
            if family == "b":
                return fit_partial_strength_cost_binomial_coordinates(
                    c.strength_out,
                    c.strength_in,
                    known_src,
                    known_tgt,
                    known_rate,
                    network.x,
                    network.y,
                    c.total_cost,
                    layers=c.binomial_layers,
                    self_loops=self_loops,
                )
            if family == "w":
                return fit_partial_strength_cost_geometric_coordinates(
                    c.strength_out,
                    c.strength_in,
                    known_src,
                    known_tgt,
                    known_rate,
                    network.x,
                    network.y,
                    c.total_cost,
                    self_loops=self_loops,
                )
        elif constraint == "strength-edges":
            if family == "me":
                return fit_partial_strength_edges_poisson(
                    c.strength_out,
                    c.strength_in,
                    known_src,
                    known_tgt,
                    known_rate,
                    c.total_edges,
                    self_loops=self_loops,
                )
            if family == "b":
                return fit_partial_strength_edges_binomial(
                    c.strength_out,
                    c.strength_in,
                    known_src,
                    known_tgt,
                    known_rate,
                    c.total_edges,
                    layers=c.binomial_layers,
                    self_loops=self_loops,
                )
            if family == "w":
                return fit_partial_strength_edges_geometric(
                    c.strength_out,
                    c.strength_in,
                    known_src,
                    known_tgt,
                    known_rate,
                    c.total_edges,
                    self_loops=self_loops,
                )
        elif constraint == "strength-degree":
            if family == "me":
                return fit_partial_strength_degree_poisson(
                    c.strength_out,
                    c.strength_in,
                    c.degree_out,
                    c.degree_in,
                    known_src,
                    known_tgt,
                    known_rate,
                    self_loops=self_loops,
                    tolerance=1e-4,
                )
            if family == "b":
                return fit_partial_strength_degree_binomial(
                    c.strength_out,
                    c.strength_in,
                    c.degree_out,
                    c.degree_in,
                    known_src,
                    known_tgt,
                    known_rate,
                    layers=c.binomial_layers,
                    self_loops=self_loops,
                    tolerance=1e-4,
                )
    # Partial not supported for this family/constraint combo
    return None


# --- Sampling ---


def _sample(
    family: str, constraint: str, fit, network, c, seed: int
) -> EdgeTable | None:
    """Sample one null network from a fitted model."""
    if constraint == "strength":
        if family == "me":
            return sample_strength_poisson(
                fit.x, fit.y, self_loops=fit.self_loops, seed=seed
            )
        if family == "b":
            return sample_strength_binomial(
                fit.x,
                fit.y,
                layers=c.binomial_layers,
                self_loops=fit.self_loops,
                seed=seed,
            )
        return sample_strength_geometric(
            fit.x, fit.y, self_loops=fit.self_loops, seed=seed
        )
    if constraint == "strength-cost":
        if family == "me":
            return sample_strength_cost_poisson(fit, network.x, network.y, seed=seed)
        if family == "b":
            return sample_strength_cost_binomial(
                fit, network.x, network.y, layers=c.binomial_layers, seed=seed
            )
        return sample_strength_cost_geometric(fit, network.x, network.y, seed=seed)
    if constraint == "strength-edges":
        if family == "me":
            return sample_strength_edges_poisson(fit, seed=seed)
        if family == "b":
            return sample_strength_edges_binomial(
                fit, layers=c.binomial_layers, seed=seed
            )
        return sample_strength_edges_geometric(fit, seed=seed)
    # strength-degree
    if family == "me":
        return sample_strength_degree_poisson(fit, seed=seed)
    if family == "b":
        return sample_strength_degree_binomial(fit, layers=c.binomial_layers, seed=seed)
    return sample_strength_degree_geometric(fit, seed=seed)


# --- Precision computation ---


def _compute_precision(
    fit, constraint: str, constraints_data, network, *, self_loops: bool
) -> dict[str, float | None]:
    """Compute constraint precision metrics from the fitted model."""
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

    if not hasattr(fit, "x") or fit.x is None:
        return result

    x, y = fit.x, fit.y

    if constraint == "strength":
        expected = np.outer(x, y)
        if not self_loops:
            np.fill_diagonal(expected, 0.0)
        s_out_err = np.abs(expected.sum(axis=1) - c.strength_out)
        s_in_err = np.abs(expected.sum(axis=0) - c.strength_in)
        max_s = max(c.strength_out.max(), c.strength_in.max(), 1.0)
        result["max_s_out_err"] = float(s_out_err.max())
        result["max_s_in_err"] = float(s_in_err.max())
        result["rel_s_err"] = float(max(s_out_err.max(), s_in_err.max()) / max_s)

    elif constraint in ("strength-edges", "strength-degree", "strength-cost"):
        # For ZI and cost models, use max_strength_residual if available
        if (
            hasattr(fit, "max_strength_residual")
            and fit.max_strength_residual is not None
        ):
            max_s = max(c.strength_out.max(), c.strength_in.max(), 1.0)
            result["max_s_out_err"] = float(fit.max_strength_residual)
            result["max_s_in_err"] = float(fit.max_strength_residual)
            result["rel_s_err"] = float(fit.max_strength_residual / max_s)

    return result


# --- Filter FPR ---


def _filter_one(family, constraint, sample, fit, network, c, alpha):
    """Apply filter to one sample."""
    if constraint == "strength":
        if family == "me":
            return filter_strength_poisson(sample, fit, alpha=alpha, tail="upper")
        if family == "b":
            return filter_strength_binomial(
                sample, fit, layers=c.binomial_layers, alpha=alpha, tail="upper"
            )
        return filter_strength_geometric(sample, fit, alpha=alpha, tail="upper")
    if constraint == "strength-cost":
        if family == "me":
            return filter_strength_cost_poisson(
                sample, fit, network.x, network.y, alpha=alpha, tail="upper"
            )
        if family == "b":
            return filter_strength_cost_binomial(
                sample,
                fit,
                network.x,
                network.y,
                layers=c.binomial_layers,
                alpha=alpha,
                tail="upper",
            )
        return filter_strength_cost_geometric(
            sample, fit, network.x, network.y, alpha=alpha, tail="upper"
        )
    if constraint == "strength-edges":
        if family == "me":
            return filter_strength_edges_poisson(sample, fit, alpha=alpha, tail="upper")
        if family == "b":
            return filter_strength_edges_binomial(
                sample, fit, layers=c.binomial_layers, alpha=alpha, tail="upper"
            )
        return filter_strength_edges_geometric(sample, fit, alpha=alpha, tail="upper")
    # strength-degree
    if family == "me":
        return filter_strength_degree_poisson(sample, fit, alpha=alpha, tail="upper")
    if family == "b":
        return filter_strength_degree_binomial(
            sample, fit, layers=c.binomial_layers, alpha=alpha, tail="upper"
        )
    return filter_strength_degree_geometric(sample, fit, alpha=alpha, tail="upper")


# --- Main benchmark runner ---


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
    fit_only: bool = False,
) -> list[BenchmarkRow]:
    """Run the benchmark matrix."""
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
            rows.append(
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
                        row = _run_one_cell(
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
                        )
                        rows.extend(row)
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
) -> list[BenchmarkRow]:
    """Run fit (+ optional sample + filter) for one matrix cell."""
    rows: list[BenchmarkRow] = []
    node_count = len(constraints_data.strength_out)
    c = constraints_data

    # Fit
    try:
        if kp_fraction > 0:
            fit, m = _measure(
                lambda: _fit_partial(
                    family,
                    constraint,
                    network,
                    c,
                    kp_fraction,
                    self_loops=self_loops,
                    seed=seed,
                ),
                track_memory=track_memory,
            )
        else:
            fit, m = _measure(
                lambda: _fit_full(
                    family, constraint, network, c, self_loops=self_loops
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
            lambda: _sample(family, constraint, fit, network, c, seed),
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
                null_sample = _sample(
                    family, constraint, fit, network, c, seed + 10000 + offset
                )
                if null_sample is None:
                    continue
                result = _filter_one(
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
        _print_table(rows)
        typer.echo(f"\nResults: {output}", err=True)


def _print_table(rows: list[BenchmarkRow]) -> None:
    """Print human-readable summary table."""
    header = (
        f"{'stage':<12} {'N':>5} {'fam':<3} {'constraint':<16} {'regime':<9} "
        f"{'kp%':>4} {'wall(s)':>8} {'cpu(s)':>8} {'parX':>5} "
        f"{'py_MB':>6} {'rss_MB':>7} {'conv':>5} {'iters':>6} "
        f"{'max_s_err':>10} {'max_k_err':>10} {'status':<8}"
    )
    typer.echo(header, err=True)
    typer.echo("─" * len(header), err=True)

    converged_count = 0
    error_count = 0
    total_wall = 0.0

    for row in rows:
        conv_str = ""
        if row.converged is not None:
            conv_str = "✓" if row.converged else "✗"
            if row.converged:
                converged_count += 1
        iters_str = str(row.iterations) if row.iterations else ""
        max_s = ""
        if row.max_s_out_err is not None:
            max_s = f"{max(row.max_s_out_err, row.max_s_in_err or 0):.2e}"
        max_k = ""
        if row.max_k_out_err is not None:
            max_k = f"{max(row.max_k_out_err, row.max_k_in_err or 0):.2e}"
        if row.status == "error":
            error_count += 1
        total_wall += row.wall_seconds
        kp_str = (
            f"{row.known_pair_fraction * 100:.0f}"
            if row.known_pair_fraction > 0
            else "0"
        )

        typer.echo(
            f"{row.stage:<12} {row.node_count:>5} {row.family:<3} {row.constraint:<16} "
            f"{row.regime:<9} {kp_str:>4} {row.wall_seconds:>8.4f} {row.cpu_seconds:>8.4f} "
            f"{row.parallel_factor:>5.1f} {row.memory_python_peak_mb:>6.1f} "
            f"{row.memory_rss_peak_mb:>7.1f} {conv_str:>5} {iters_str:>6} "
            f"{max_s:>10} {max_k:>10} {row.status:<8}",
            err=True,
        )

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
