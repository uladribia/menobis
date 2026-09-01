"""Microcanonical vs grand-canonical practical comparison benchmark.

Compares the current MENoBiS grand-canonical (GC) and microcanonical (MC)
ensembles over ``N x family x constraint x ensemble x samples`` from a
practical user perspective: end-to-end public-API time, constraint
recovery, and higher-order observables (degree, Y2, kNN, sNN, strength,
realized cost).

Design rules (see the implementation plan):

- public model API only (``fit_model`` / ``sample_model`` /
  ``compute_all_stats``);
- one observed network per N used for every family/constraint/ensemble;
- deterministic seed construction, never a bare ``default_rng()``;
- sparse scaling regime: ``average_degree=8``, ``events_per_edge=8``;
- checkpoint every completed cell; atomic file replacement;
- do not compare low-level internals or private ``_*`` bindings.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from menobis.analysis.stats import compute_all_stats
from menobis.data.frames import EdgeTable
from menobis.models.spec import Constraint, Ensemble, ModelFamily
from menobis.models.types import FitResult
from menobis.routing import fit_model, sample_model
from menobis.utilities.synthetic import (
    SyntheticConstraints,
    SyntheticNetwork,
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

# ---------------------------------------------------------------------------
# Mandatory experimental configuration (non-negotiable)
# ---------------------------------------------------------------------------

NODE_COUNTS: tuple[int, ...] = (100, 500, 2000)
SAMPLES_PER_CELL: int = 10
SELF_LOOPS: bool = False
AVERAGE_DEGREE: float = 8.0
EVENTS_PER_EDGE: float = 8.0
OBSERVED_SEED: int = 42  # main observed network seed rule: seed = 42 + n

FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily.ME,
    ModelFamily.B,
    ModelFamily.W,
)
CONSTRAINTS: tuple[Constraint, ...] = (
    Constraint.STRENGTH,
    Constraint.STRENGTH_COST,
    Constraint.STRENGTH_EDGES,
    Constraint.STRENGTH_DEGREE,
    Constraint.DEGREE_EVENTS,
    Constraint.EDGES_EVENTS,
)
ENSEMBLES: tuple[Ensemble, ...] = (
    Ensemble.GRAND_CANONICAL,
    Ensemble.MICROCANONICAL,
)

# Sparsity-sensitivity experiment (Section 3 of the plan).
SPARSITY_NODE_COUNT: int = 500
SPARSITY_DEGREES: tuple[float, ...] = (3.0, 8.0, 20.0)
SPARSITY_EVENTS_PER_EDGE: float = 8.0
SPARSITY_FAMILY: ModelFamily = ModelFamily.ME
SPARSITY_CONSTRAINTS: tuple[Constraint, ...] = (
    Constraint.STRENGTH,
    Constraint.STRENGTH_DEGREE,
    Constraint.EDGES_EVENTS,
)
SPARSITY_SAMPLES: int = 10
SPARSITY_SELF_LOOPS: bool = False
SPARSITY_OBSERVED_SEED: int = 142  # rule: seed = 142 + int(average_degree)

# MCMC budget (Section 13).
BURN_IN_SWEEPS: int = 3
SWEEPS_PER_SAMPLE: int = 1

# Timing repeats (Section 16).
TIMING_REPEATS_BY_N: dict[int, int] = {100: 3, 500: 3, 2000: 1}

# Seed indices (Section 12).
FAMILY_INDEX: dict[ModelFamily, int] = {
    ModelFamily.ME: 0,
    ModelFamily.B: 1,
    ModelFamily.W: 2,
}
CONSTRAINT_INDEX: dict[Constraint, int] = {
    Constraint.STRENGTH: 0,
    Constraint.STRENGTH_COST: 1,
    Constraint.STRENGTH_EDGES: 2,
    Constraint.STRENGTH_DEGREE: 3,
    Constraint.DEGREE_EVENTS: 4,
    Constraint.EDGES_EVENTS: 5,
}
ENSEMBLE_INDEX: dict[Ensemble, int] = {
    Ensemble.GRAND_CANONICAL: 0,
    Ensemble.MICROCANONICAL: 1,
}

RESULTS_DIR = Path("benchmarks/results/ensemble-comparison")

# Observable identity: (name, direction) with direction in out/in/global.
Observable = tuple[str, str]

# All stored node-level and global observables (Section 20).
STORE_OBSERVABLES: tuple[Observable, ...] = (
    ("strength", "out"),
    ("strength", "in"),
    ("degree", "out"),
    ("degree", "in"),
    ("Y2", "out"),
    ("Y2", "in"),
    ("sNN", "out"),
    ("sNN", "in"),
    ("kNN", "out"),
    ("kNN", "in"),
    ("E", "global"),
    ("T", "global"),
    ("mean_positive_occupation", "global"),
    ("cost", "global"),
)

# Primary (comparison) observables by constraint (Section 21).
PRIMARY_OBSERVABLES: dict[Constraint, tuple[Observable, ...]] = {
    Constraint.STRENGTH: (
        ("E", "global"),
        ("degree", "out"),
        ("degree", "in"),
        ("Y2", "out"),
        ("Y2", "in"),
        ("kNN", "out"),
        ("kNN", "in"),
        ("sNN", "out"),
        ("sNN", "in"),
    ),
    Constraint.STRENGTH_COST: (
        ("cost", "global"),
        ("E", "global"),
        ("degree", "out"),
        ("degree", "in"),
        ("Y2", "out"),
        ("Y2", "in"),
        ("kNN", "out"),
        ("kNN", "in"),
        ("sNN", "out"),
        ("sNN", "in"),
    ),
    Constraint.STRENGTH_EDGES: (
        ("degree", "out"),
        ("degree", "in"),
        ("Y2", "out"),
        ("Y2", "in"),
        ("kNN", "out"),
        ("kNN", "in"),
        ("sNN", "out"),
        ("sNN", "in"),
    ),
    Constraint.STRENGTH_DEGREE: (
        ("Y2", "out"),
        ("Y2", "in"),
        ("kNN", "out"),
        ("kNN", "in"),
        ("sNN", "out"),
        ("sNN", "in"),
        ("mean_positive_occupation", "global"),
    ),
    Constraint.DEGREE_EVENTS: (
        ("strength", "out"),
        ("strength", "in"),
        ("Y2", "out"),
        ("Y2", "in"),
        ("kNN", "out"),
        ("kNN", "in"),
        ("sNN", "out"),
        ("sNN", "in"),
    ),
    Constraint.EDGES_EVENTS: (
        ("strength", "out"),
        ("strength", "in"),
        ("degree", "out"),
        ("degree", "in"),
        ("Y2", "out"),
        ("Y2", "in"),
        ("kNN", "out"),
        ("kNN", "in"),
        ("sNN", "out"),
        ("sNN", "in"),
    ),
}

# Validation-only (hard-fixed / fitted) observables per constraint, used to
# mask the disagreement heatmaps (Section 38).
VALIDATION_OBSERVABLES: dict[Constraint, frozenset[str]] = {
    Constraint.STRENGTH: frozenset({"strength"}),
    Constraint.STRENGTH_COST: frozenset({"strength"}),
    Constraint.STRENGTH_EDGES: frozenset({"strength", "E"}),
    Constraint.STRENGTH_DEGREE: frozenset({"strength", "degree"}),
    Constraint.DEGREE_EVENTS: frozenset({"degree", "T"}),
    Constraint.EDGES_EVENTS: frozenset({"E", "T"}),
}

# Budget gate budgets (Section 14, extended per user decision): ladder A..E.
BUDGET_A: tuple[int, int] = (3, 1)
BUDGET_B: tuple[int, int] = (10, 2)
BUDGET_C: tuple[int, int] = (20, 5)
BUDGET_D: tuple[int, int] = (50, 10)
BUDGET_E: tuple[int, int] = (200, 50)
BUDGET_LADDER: tuple[tuple[int, int], ...] = (
    BUDGET_A,
    BUDGET_B,
    BUDGET_C,
    BUDGET_D,
    BUDGET_E,
)
BUDGET_GATE_CONSTRAINTS: tuple[Constraint, ...] = (
    Constraint.STRENGTH,
    Constraint.STRENGTH_EDGES,
    Constraint.STRENGTH_DEGREE,
)

# Feasibility caps applied to the gate-chosen micro budget at larger N.
# The gate ladder runs fully at N=100; the heavy cells (STRENGTH_COST,
# STRENGTH_DEGREE) would be infeasibly slow at extended budgets for
# N >= 500, so their budget is capped while the instability is flagged
# explicitly in results and the notebook.
BUDGET_CAP_BY_N: dict[int, dict[Constraint, tuple[int, int]]] = {
    100: {},
    500: {
        Constraint.STRENGTH_COST: BUDGET_A,
        Constraint.STRENGTH_DEGREE: BUDGET_B,
    },
    2000: {
        Constraint.STRENGTH_COST: BUDGET_A,
        Constraint.STRENGTH_DEGREE: BUDGET_A,
    },
}

# ---------------------------------------------------------------------------
# Seed construction
# ---------------------------------------------------------------------------


def ensemble_seed(
    n: int,
    family: ModelFamily,
    constraint: Constraint,
    ensemble: Ensemble,
    sample_index: int,
) -> int:
    """Deterministic seed for one ensemble sample (Section 12)."""
    return (
        1_000_000
        + 100_000 * NODE_COUNTS.index(n)
        + 10_000 * FAMILY_INDEX[family]
        + 1_000 * CONSTRAINT_INDEX[constraint]
        + 100 * ENSEMBLE_INDEX[ensemble]
        + sample_index
    )


# ---------------------------------------------------------------------------
# Observed case and constraint kwargs
# ---------------------------------------------------------------------------


def make_observed_case(
    node_count: int,
    *,
    average_degree: float = AVERAGE_DEGREE,
    events_per_edge: float = EVENTS_PER_EDGE,
    seed: int,
    self_loops: bool = SELF_LOOPS,
) -> tuple[SyntheticNetwork, SyntheticConstraints]:
    """Generate the observed network and derive feasible constraints."""
    observed = generate_pa_geographic_network(
        node_count=node_count,
        average_degree=average_degree,
        events_per_edge=events_per_edge,
        seed=seed,
        self_loops=self_loops,
    )
    return observed, derive_synthetic_constraints(observed)


def build_constraint_kwargs(
    *,
    constraint: Constraint,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
) -> dict[str, object]:
    """Constraint arguments shared by GC fit and micro sampling (Section 10)."""
    match constraint:
        case Constraint.STRENGTH:
            return {
                "strength_out": derived.strength_out,
                "strength_in": derived.strength_in,
            }
        case Constraint.STRENGTH_COST:
            return {
                "strength_out": derived.strength_out,
                "strength_in": derived.strength_in,
                "coord_x": observed.x,
                "coord_y": observed.y,
                "target_cost": derived.total_cost,
            }
        case Constraint.STRENGTH_EDGES:
            return {
                "strength_out": derived.strength_out,
                "strength_in": derived.strength_in,
                "target_edges": int(derived.total_edges),
            }
        case Constraint.STRENGTH_DEGREE:
            return {
                "strength_out": derived.strength_out,
                "strength_in": derived.strength_in,
                "degree_out": derived.degree_out,
                "degree_in": derived.degree_in,
            }
        case Constraint.DEGREE_EVENTS:
            return {
                "degree_out": derived.degree_out,
                "degree_in": derived.degree_in,
                "total_events": int(derived.total_events),
            }
        case Constraint.EDGES_EVENTS:
            return {
                "node_count": len(observed.x),
                "target_edges": int(derived.total_edges),
                "total_events": int(derived.total_events),
            }
        case _:
            msg = f"unsupported constraint: {constraint!r}"
            raise ValueError(msg)


def family_layers(family: ModelFamily, derived: SyntheticConstraints) -> int:
    """ME and W use 1 layer; B uses the derived binomial layer count."""
    if family is ModelFamily.B:
        return derived.binomial_layers
    return 1


def make_fit_kwargs(
    *,
    constraint: Constraint,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
    family: ModelFamily,
    self_loops: bool = SELF_LOOPS,
) -> dict[str, object]:
    """Keyword arguments for the grand-canonical fit route."""
    kwargs = build_constraint_kwargs(
        constraint=constraint, observed=observed, derived=derived
    )
    kwargs["layers"] = family_layers(family, derived)
    kwargs["self_loops"] = self_loops
    return kwargs


def make_sample_kwargs(
    *,
    ensemble: Ensemble,
    family: ModelFamily,
    constraint: Constraint,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
    self_loops: bool = SELF_LOOPS,
    fit: FitResult | None = None,
    seed: int = 0,
    burn_in_sweeps: int = BURN_IN_SWEEPS,
    sweeps_per_sample: int = SWEEPS_PER_SAMPLE,
) -> dict[str, object]:
    """Keyword arguments for one sample route call."""
    kwargs = build_constraint_kwargs(
        constraint=constraint, observed=observed, derived=derived
    )
    kwargs["layers"] = family_layers(family, derived)
    kwargs["self_loops"] = self_loops
    kwargs["seed"] = seed
    if ensemble is Ensemble.MICROCANONICAL:
        kwargs["burn_in_sweeps"] = burn_in_sweeps
        kwargs["sweeps_per_sample"] = sweeps_per_sample
    else:
        if fit is None:
            msg = "grand-canonical sampling requires a fitted model"
            raise ValueError(msg)
        kwargs["fit"] = fit
    return kwargs


# ---------------------------------------------------------------------------
# Core route calls
# ---------------------------------------------------------------------------


def fit_gc_case(
    *,
    family: ModelFamily,
    constraint: Constraint,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
    self_loops: bool = SELF_LOOPS,
) -> FitResult:
    """Fit the grand-canonical model for one cell."""
    return fit_model(
        ensemble=Ensemble.GRAND_CANONICAL,
        family=family,
        constraint=constraint,
        **make_fit_kwargs(
            constraint=constraint,
            observed=observed,
            derived=derived,
            family=family,
            self_loops=self_loops,
        ),
    )


def sample_gc_case(
    *,
    fit: FitResult,
    family: ModelFamily,
    constraint: Constraint,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
    self_loops: bool = SELF_LOOPS,
    seed: int,
) -> EdgeTable:
    """Generate one grand-canonical sample from a fitted model."""
    kwargs = make_sample_kwargs(
        ensemble=Ensemble.GRAND_CANONICAL,
        family=family,
        constraint=constraint,
        observed=observed,
        derived=derived,
        self_loops=self_loops,
        fit=fit,
        seed=seed,
    )
    return sample_model(
        ensemble=Ensemble.GRAND_CANONICAL,
        family=family,
        constraint=constraint,
        **kwargs,
    )


def sample_micro_case(
    *,
    family: ModelFamily,
    constraint: Constraint,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
    self_loops: bool = SELF_LOOPS,
    seed: int,
    burn_in_sweeps: int = BURN_IN_SWEEPS,
    sweeps_per_sample: int = SWEEPS_PER_SAMPLE,
) -> EdgeTable:
    """Generate one microcanonical sample directly from constraints."""
    kwargs = make_sample_kwargs(
        ensemble=Ensemble.MICROCANONICAL,
        family=family,
        constraint=constraint,
        observed=observed,
        derived=derived,
        self_loops=self_loops,
        seed=seed,
        burn_in_sweeps=burn_in_sweeps,
        sweeps_per_sample=sweeps_per_sample,
    )
    return sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=family,
        constraint=constraint,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------


def realized_cost(edges: EdgeTable, observed: SyntheticNetwork) -> float:
    """C = sum over occupied pairs of t_ij * Euclidean distance (Section 20)."""
    source = edges.source.astype(np.int64)
    target = edges.target.astype(np.int64)
    distances = np.hypot(
        observed.x[source] - observed.x[target],
        observed.y[source] - observed.y[target],
    )
    return float(np.sum(edges.occ_num.astype(np.float64) * distances))


def extract_observables(
    edges: EdgeTable,
    *,
    constraint: Constraint,
    observed: SyntheticNetwork,
) -> dict[Observable, float | NDArray[np.float64]]:
    """Extract all stored observables from one sampled network."""
    del constraint  # every stored observable is computed for every constraint
    stats = compute_all_stats(edges)
    out: dict[Observable, float | NDArray[np.float64]] = {
        ("strength", "out"): stats.strength_out.astype(np.float64),
        ("strength", "in"): stats.strength_in.astype(np.float64),
        ("degree", "out"): stats.degree_out.astype(np.float64),
        ("degree", "in"): stats.degree_in.astype(np.float64),
        ("Y2", "out"): stats.y2_out.astype(np.float64),
        ("Y2", "in"): stats.y2_in.astype(np.float64),
        ("sNN", "out"): stats.s_nn_out.astype(np.float64),
        ("sNN", "in"): stats.s_nn_in.astype(np.float64),
        ("kNN", "out"): stats.k_nn_out.astype(np.float64),
        ("kNN", "in"): stats.k_nn_in.astype(np.float64),
    }
    e = float(edges.num_edges)
    t = float(edges.total_events)
    out[("E", "global")] = e
    out[("T", "global")] = t
    out[("mean_positive_occupation", "global")] = t / e if e > 0 else np.nan
    out[("cost", "global")] = realized_cost(edges, observed)
    return out


def _constraint_residuals(
    edges: EdgeTable, derived: SyntheticConstraints
) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Strengths/degrees/global flags against the derived targets."""
    stats = compute_all_stats(edges)
    return (
        bool(
            np.array_equal(stats.strength_out, derived.strength_out.astype(np.uint64))
        ),
        bool(np.array_equal(stats.strength_in, derived.strength_in.astype(np.uint64))),
        bool(np.array_equal(stats.degree_out, derived.degree_out.astype(np.uint64))),
        bool(np.array_equal(stats.degree_in, derived.degree_in.astype(np.uint64))),
        edges.num_edges == int(derived.total_edges),
        edges.total_events == derived.total_events,
    )


def validate_micro_constraints(
    edges: EdgeTable,
    *,
    constraint: Constraint,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
) -> tuple[bool, str, dict[str, float]]:
    """Validate one microcanonical sample against its hard constraints.

    Returns ``(valid, message, extra)`` where ``extra`` records realized vs
    target cost for STRENGTH_COST (cost is matched in expectation by the
    internal gamma fit, never claimed exact).
    """
    s_out_ok, s_in_ok, k_out_ok, k_in_ok, e_ok, t_ok = _constraint_residuals(
        edges, derived
    )
    extra: dict[str, float] = {}
    match constraint:
        case Constraint.STRENGTH:
            ok = s_out_ok and s_in_ok
            msg = "" if ok else "strength mismatch"
        case Constraint.STRENGTH_COST:
            ok = s_out_ok and s_in_ok  # cost is soft, matched in expectation
            msg = "" if ok else "strength mismatch"
            extra["realized_cost"] = realized_cost(edges, observed)
            extra["target_cost"] = float(derived.total_cost)
        case Constraint.STRENGTH_EDGES:
            ok = s_out_ok and s_in_ok and e_ok
            msg = "" if ok else "strength/edges mismatch"
        case Constraint.STRENGTH_DEGREE:
            ok = s_out_ok and s_in_ok and k_out_ok and k_in_ok
            msg = "" if ok else "strength/degree mismatch"
        case Constraint.DEGREE_EVENTS:
            ok = k_out_ok and k_in_ok and t_ok
            msg = "" if ok else "degree/events mismatch"
        case Constraint.EDGES_EVENTS:
            ok = e_ok and t_ok
            msg = "" if ok else "edges/events mismatch"
        case _:
            msg = f"unsupported constraint: {constraint!r}"
            raise ValueError(msg)
    return ok, msg, extra


# ---------------------------------------------------------------------------
# Ensemble aggregation
# ---------------------------------------------------------------------------


@dataclass
class EnsembleAggregate:
    """Ten-sample aggregation for one cell and ensemble."""

    node_count: int
    node_level: dict[Observable, NDArray[np.float64]] = field(default_factory=dict)
    global_level: dict[Observable, NDArray[np.float64]] = field(default_factory=dict)
    status: str = "ok"
    message: str = ""
    fit_converged: bool | None = None
    fit_iterations: int | None = None
    invalid_sample: int | None = None

    def observable_values(self, observable: Observable) -> NDArray[np.float64]:
        """10-sample values for an observable (node or global)."""
        if observable[1] == "global":
            return self.global_level.get(observable, np.array([], dtype=np.float64))
        return self.node_level.get(observable, np.array([], dtype=np.float64))


def run_gc_ensemble(
    *,
    family: ModelFamily,
    constraint: Constraint,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
    node_count: int,
    sample_count: int = SAMPLES_PER_CELL,
    self_loops: bool = SELF_LOOPS,
    fit: FitResult | None = None,
) -> EnsembleAggregate:
    """Fit once and draw ``sample_count`` grand-canonical samples (Section 15)."""
    aggregate = EnsembleAggregate(node_count=node_count)
    if fit is None:
        fit = fit_gc_case(
            family=family,
            constraint=constraint,
            observed=observed,
            derived=derived,
            self_loops=self_loops,
        )
    aggregate.fit_converged = bool(fit.converged)
    aggregate.fit_iterations = (
        None if fit.diagnostics is None else fit.diagnostics.iterations
    )
    if not fit.converged:
        aggregate.status = "fit_failed"
        aggregate.message = f"gc fit not converged (status={fit.status!r})"
        return aggregate

    per_sample: dict[Observable, list[float | NDArray[np.float64]]] = {
        obs: [] for obs in STORE_OBSERVABLES
    }
    for sample_index in range(sample_count):
        seed = ensemble_seed(
            node_count, family, constraint, Ensemble.GRAND_CANONICAL, sample_index
        )
        edges = sample_gc_case(
            fit=fit,
            family=family,
            constraint=constraint,
            observed=observed,
            derived=derived,
            self_loops=self_loops,
            seed=seed,
        )
        values = extract_observables(edges, constraint=constraint, observed=observed)
        for obs in STORE_OBSERVABLES:
            per_sample[obs].append(values[obs])

    _aggregate_samples(aggregate, per_sample)
    return aggregate


def run_micro_ensemble(
    *,
    family: ModelFamily,
    constraint: Constraint,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
    node_count: int,
    sample_count: int = SAMPLES_PER_CELL,
    self_loops: bool = SELF_LOOPS,
    burn_in_sweeps: int = BURN_IN_SWEEPS,
    sweeps_per_sample: int = SWEEPS_PER_SAMPLE,
    validate: bool = True,
) -> EnsembleAggregate:
    """Draw ``sample_count`` microcanonical samples and validate them."""
    aggregate = EnsembleAggregate(node_count=node_count, fit_converged=True)
    per_sample: dict[Observable, list[float | NDArray[np.float64]]] = {
        obs: [] for obs in STORE_OBSERVABLES
    }
    for sample_index in range(sample_count):
        seed = ensemble_seed(
            node_count, family, constraint, Ensemble.MICROCANONICAL, sample_index
        )
        edges = sample_micro_case(
            family=family,
            constraint=constraint,
            observed=observed,
            derived=derived,
            self_loops=self_loops,
            seed=seed,
            burn_in_sweeps=burn_in_sweeps,
            sweeps_per_sample=sweeps_per_sample,
        )
        if validate:
            valid, message, extra = validate_micro_constraints(
                edges, constraint=constraint, observed=observed, derived=derived
            )
            if not valid:
                aggregate.status = "invalid_micro_sample"
                aggregate.message = f"sample {sample_index}: {message}"
                aggregate.invalid_sample = sample_index
                return aggregate
            if constraint is Constraint.STRENGTH_COST:
                _log(
                    f"  micro STRENGTH_COST sample {sample_index}: "
                    f"realized_cost={extra['realized_cost']:.3f} "
                    f"target_cost={extra['target_cost']:.3f}"
                )
        values = extract_observables(edges, constraint=constraint, observed=observed)
        for obs in STORE_OBSERVABLES:
            per_sample[obs].append(values[obs])

    _aggregate_samples(aggregate, per_sample)
    return aggregate


def _aggregate_samples(
    aggregate: EnsembleAggregate,
    per_sample: dict[Observable, list[float | NDArray[np.float64]]],
) -> None:
    """Stack per-sample observables into (10, N) or (10,) arrays."""
    for obs, samples in per_sample.items():
        first = samples[0]
        if isinstance(first, np.ndarray):
            aggregate.node_level[obs] = np.stack(samples).astype(np.float64)
        else:
            aggregate.global_level[obs] = np.asarray(samples, dtype=np.float64)


def summarize_ensemble(
    aggregate: EnsembleAggregate,
    *,
    node_count: int,
    family: ModelFamily,
    constraint: Constraint,
    ensemble: Ensemble,
    sample_count: int,
) -> list[dict[str, object]]:
    """Scientific-summary rows for one cell (Sections 22, 28)."""
    rows: list[dict[str, object]] = []
    for obs, direction in STORE_OBSERVABLES:
        values = aggregate.observable_values((obs, direction))
        if values.size == 0:
            continue
        if direction == "global":
            with np.errstate(invalid="ignore"):
                mean = float(np.nanmean(values))
                sd = (
                    float(np.nanstd(values, ddof=1))
                    if values.size > 1
                    else float("nan")
                )
                se = float(sd / np.sqrt(values.size))
            valid = 1 if np.isfinite(mean) else 0
        else:
            node_mean = np.nanmean(values, axis=0)
            node_sd = (
                np.nanstd(values, axis=0, ddof=1)
                if values.shape[0] > 1
                else np.full(values.shape[1], np.nan)
            )
            node_se = np.where(
                np.isfinite(node_sd), node_sd / np.sqrt(values.shape[0]), np.nan
            )
            valid = int(np.sum(np.isfinite(node_mean)))
            mean = float(np.nanmean(node_mean)) if valid else float("nan")
            sd = float(np.nanmean(node_sd)) if valid else float("nan")
            se = float(np.nanmean(node_se)) if valid else float("nan")
        rows.append(
            {
                "node_count": node_count,
                "family": family.value,
                "constraint": constraint.value,
                "ensemble": ensemble.value,
                "observable": obs,
                "direction": direction,
                "sample_count": sample_count,
                "mean_node_average": mean,
                "mean_within_ensemble_sd": sd,
                "mean_mc_se": se,
                "valid_node_count": valid,
                "status": aggregate.status,
                "message": aggregate.message,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Comparison metrics (Section 23)
# ---------------------------------------------------------------------------


def _spearman_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    _, inverse = np.unique(values, return_inverse=True)
    for value_index in range(int(inverse.max()) + 1):
        mask = inverse == value_index
        if np.sum(mask) > 1:
            ranks[mask] = np.mean(ranks[mask])
    return ranks


def spearman_correlation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Spearman rho via average ranks + Pearson on ranks (no scipy)."""
    if len(a) < 3 or len(a) != len(b):
        return float("nan")
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 3:
        return float("nan")
    ra = _spearman_ranks(a[valid])
    rb = _spearman_ranks(b[valid])
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def d_rel_node_level(
    micro_mean: NDArray[np.float64], gc_mean: NDArray[np.float64]
) -> tuple[float, int]:
    """Symmetric normalized mean absolute difference (Section 23.1)."""
    valid = np.isfinite(micro_mean) & np.isfinite(gc_mean)
    m = int(np.sum(valid))
    if m == 0:
        return float("nan"), 0
    diff = np.abs(micro_mean[valid] - gc_mean[valid])
    scale = np.mean(np.abs(micro_mean[valid]) + np.abs(gc_mean[valid])) / 2.0 + 1e-12
    return float(np.mean(diff) / scale), m


def d_rel_global(micro_value: float, gc_value: float) -> float:
    """Symmetric scalar relative difference with a 1e-12 floor."""
    denom = (abs(micro_value) + abs(gc_value)) / 2.0 + 1e-12
    return float(abs(micro_value - gc_value) / denom)


def compare_ensembles(
    micro: EnsembleAggregate,
    gc: EnsembleAggregate,
    *,
    node_count: int,
    family: ModelFamily,
    constraint: Constraint,
    sample_count: int,
    observables: tuple[Observable, ...] = STORE_OBSERVABLES,
) -> list[dict[str, object]]:
    """Comparison rows for one cell (micro vs GC), Sections 23/29."""
    rows: list[dict[str, object]] = []
    for obs, direction in observables:
        m_values = micro.observable_values((obs, direction))
        g_values = gc.observable_values((obs, direction))
        if m_values.size == 0 or g_values.size == 0:
            continue
        if direction == "global":
            with np.errstate(invalid="ignore"):
                m_mean = float(np.nanmean(m_values)) if m_values.size else float("nan")
                g_mean = float(np.nanmean(g_values)) if g_values.size else float("nan")
                m_se = (
                    float(np.nanstd(m_values, ddof=1) / np.sqrt(m_values.size))
                    if m_values.size > 1
                    else float("nan")
                )
                g_se = (
                    float(np.nanstd(g_values, ddof=1) / np.sqrt(g_values.size))
                    if g_values.size > 1
                    else float("nan")
                )
            finite = np.isfinite(m_mean) and np.isfinite(g_mean)
            d_rel = d_rel_global(m_mean, g_mean) if finite else float("nan")
            rho = float("nan")  # global observables have no rank correlation
            valid = 1 if finite else 0
        else:
            m_mean = np.nanmean(m_values, axis=0)
            g_mean = np.nanmean(g_values, axis=0)
            m_se = (
                np.nanstd(m_values, axis=0, ddof=1) / np.sqrt(m_values.shape[0])
                if m_values.shape[0] > 1
                else np.full(m_values.shape[1], np.nan)
            )
            g_se = (
                np.nanstd(g_values, axis=0, ddof=1) / np.sqrt(g_values.shape[0])
                if g_values.shape[0] > 1
                else np.full(g_values.shape[1], np.nan)
            )
            d_rel, valid = d_rel_node_level(m_mean, g_mean)
            rho = spearman_correlation(m_mean, g_mean)
        rows.append(
            {
                "node_count": node_count,
                "family": family.value,
                "constraint": constraint.value,
                "observable": obs,
                "direction": direction,
                "micro_mean_node_average": (
                    float(np.nanmean(m_mean))
                    if isinstance(m_mean, np.ndarray)
                    else m_mean
                ),
                "gc_mean_node_average": (
                    float(np.nanmean(g_mean))
                    if isinstance(g_mean, np.ndarray)
                    else g_mean
                ),
                "d_rel": d_rel,
                "spearman": rho,
                "valid_node_count": valid,
                "micro_mean_mc_se": (
                    float(np.nanmean(m_se)) if isinstance(m_se, np.ndarray) else m_se
                ),
                "gc_mean_mc_se": (
                    float(np.nanmean(g_se)) if isinstance(g_se, np.ndarray) else g_se
                ),
                "status": _cell_status(micro, gc),
                "message": _cell_message(micro, gc),
            }
        )
    return rows


def _cell_status(micro: EnsembleAggregate, gc: EnsembleAggregate) -> str:
    for aggregate in (micro, gc):
        if aggregate.status in ("error", "invalid_micro_sample", "fit_failed"):
            return aggregate.status
    if micro.status != "ok":
        return micro.status
    if gc.status != "ok":
        return gc.status
    return "ok"


def _cell_message(micro: EnsembleAggregate, gc: EnsembleAggregate) -> str:
    if micro.status != "ok":
        return micro.message
    if gc.status != "ok":
        return gc.message
    return ""


# ---------------------------------------------------------------------------
# Result storage (atomic, checkpointed)
# ---------------------------------------------------------------------------


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a dataframe atomically via a temp file + ``Path.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            frame.to_csv(handle, index=False)
        Path(temp_name).replace(path)
    except BaseException:
        with suppress(OSError):
            Path(temp_name).unlink()
        raise


class ResultStore:
    """Checkpointed CSV/state storage for the comparison benchmark."""

    def __init__(self, results_dir: Path = RESULTS_DIR) -> None:
        """Create the results directory and named result paths."""
        self.dir = results_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timings_path = self.dir / "main-timings.csv"
        self.science_path = self.dir / "main-scientific-summary.csv"
        self.comparison_path = self.dir / "main-comparison.csv"
        self.sparsity_path = self.dir / "sparsity-sensitivity.csv"
        self.gate_path = self.dir / "mcmc-budget-gate.csv"
        self.metadata_path = self.dir / "metadata.json"
        self.npz_path = self.dir / "per-node-means.npz"
        self.state_path = self.dir / "cell-state.json"

    # --- cell state -------------------------------------------------------
    def _load_state(self) -> dict[str, dict[str, object]]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {}

    def _save_state(self, state: dict[str, dict[str, object]]) -> None:
        fd, temp_name = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
            Path(temp_name).replace(self.state_path)
        except BaseException:
            with suppress(OSError):
                Path(temp_name).unlink()
            raise

    @staticmethod
    def cell_key(
        node_count: int, family: ModelFamily, constraint: Constraint, ensemble: Ensemble
    ) -> str:
        """Stable cell identifier used for checkpointing."""
        return f"{node_count}|{family.value}|{constraint.value}|{ensemble.value}"

    def cell_done(self, key: str) -> bool:
        """True when a cell is checkpointed as ok with full sample count."""
        state = self._load_state()
        entry = state.get(key)
        return (
            entry is not None
            and entry.get("status") == "ok"
            and entry.get("sample_count") == SAMPLES_PER_CELL
        )

    def set_cell(
        self, key: str, status: str, sample_count: int, message: str = ""
    ) -> None:
        """Record the final status of one cell in the checkpoint state."""
        state = self._load_state()
        state[key] = {
            "status": status,
            "sample_count": sample_count,
            "message": message,
        }
        self._save_state(state)

    # --- row tables -------------------------------------------------------
    def _load_rows(self, path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        return pd.read_csv(path).to_dict(orient="records")  # type: ignore[no-any-return]

    def append_rows(self, path: Path, rows: list[dict[str, object]]) -> None:
        """Merge rows with the existing table and write it atomically."""
        combined = pd.DataFrame(self._load_rows(path) + rows)
        _atomic_csv(combined, path)

    def write_rows(self, path: Path, rows: list[dict[str, object]]) -> None:
        """Replace the whole table atomically (idempotent full rewrite)."""
        _atomic_csv(pd.DataFrame(rows), path)

    # --- npz ---------------------------------------------------------------
    def save_per_node_npz(self, arrays: dict[str, NDArray[np.float64]]) -> None:
        """Atomically write the per-node ensemble arrays archive."""
        fd, temp_name = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            os.close(fd)
            # numpy's savez_compressed silently produces an empty archive when
            # the target path already exists (e.g. the mkstemp file), so write
            # through a file object instead.
            with Path(temp_name).open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            Path(temp_name).replace(self.npz_path)
        except BaseException:
            with suppress(OSError):
                Path(temp_name).unlink()
            raise


# ---------------------------------------------------------------------------
# Timing workload (Sections 15-18)
# ---------------------------------------------------------------------------


def _time_call(callable_: Callable[[], object]) -> float:
    start = time.perf_counter()
    callable_()
    return time.perf_counter() - start


def run_timing_workload(
    *,
    node_count: int,
    family: ModelFamily,
    constraint: Constraint,
    ensemble: Ensemble,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
    timing_repeat: int,
    self_loops: bool = SELF_LOOPS,
    sample_count: int = SAMPLES_PER_CELL,
    burn_in_sweeps: int = BURN_IN_SWEEPS,
    sweeps_per_sample: int = SWEEPS_PER_SAMPLE,
) -> dict[str, object]:
    """Run one complete 10-sample timing workload for a cell (Section 15).

    The measured total is exactly ``fit + sampling + stats``. Synthetic
    generation, derive, dataframe construction, and saving are excluded.
    """
    layers = family_layers(family, derived)
    samples: list[EdgeTable] = []
    status = "ok"
    message = ""

    if ensemble is Ensemble.MICROCANONICAL:
        fit_seconds = 0.0
        fit_converged: bool | None = None
        fit_iterations: int | None = None

        def generate() -> None:
            for sample_index in range(sample_count):
                seed = ensemble_seed(
                    node_count, family, constraint, ensemble, sample_index
                )
                samples.append(
                    sample_micro_case(
                        family=family,
                        constraint=constraint,
                        observed=observed,
                        derived=derived,
                        self_loops=self_loops,
                        seed=seed,
                        burn_in_sweeps=burn_in_sweeps,
                        sweeps_per_sample=sweeps_per_sample,
                    )
                )

        try:
            sampling_seconds = _time_call(generate)
            stats_seconds = _time_call(
                lambda: [
                    extract_observables(e, constraint=constraint, observed=observed)
                    for e in samples
                ]
            )
        except Exception as error:  # cell-local failure must not kill the run
            status = "error"
            message = f"micro: {type(error).__name__}: {error}"
            sampling_seconds = 0.0
            stats_seconds = 0.0
    else:
        fit_start = time.perf_counter()
        fit = fit_gc_case(
            family=family,
            constraint=constraint,
            observed=observed,
            derived=derived,
            self_loops=self_loops,
        )
        fit_seconds = time.perf_counter() - fit_start
        fit_converged = bool(fit.converged)
        fit_iterations = None if fit.diagnostics is None else fit.diagnostics.iterations
        if not fit.converged:
            status = "fit_failed"
            message = f"gc fit not converged (status={fit.status!r})"
            sampling_seconds = 0.0
            stats_seconds = 0.0
        else:

            def generate() -> None:
                for sample_index in range(sample_count):
                    seed = ensemble_seed(
                        node_count, family, constraint, ensemble, sample_index
                    )
                    samples.append(
                        sample_gc_case(
                            fit=fit,
                            family=family,
                            constraint=constraint,
                            observed=observed,
                            derived=derived,
                            self_loops=self_loops,
                            seed=seed,
                        )
                    )

            try:
                sampling_seconds = _time_call(generate)
                stats_seconds = _time_call(
                    lambda: [
                        extract_observables(e, constraint=constraint, observed=observed)
                        for e in samples
                    ]
                )
            except Exception as error:  # cell-local failure must not kill the run
                status = "error"
                message = f"gc sampling: {type(error).__name__}: {error}"
                sampling_seconds = 0.0
                stats_seconds = 0.0
    total_seconds = fit_seconds + sampling_seconds + stats_seconds
    return {
        "node_count": node_count,
        "family": family.value,
        "constraint": constraint.value,
        "ensemble": ensemble.value,
        "self_loops": self_loops,
        "layers": layers,
        "sample_count": sample_count,
        "burn_in_sweeps": burn_in_sweeps,
        "sweeps_per_sample": sweeps_per_sample,
        "timing_repeat": timing_repeat,
        "fit_seconds": fit_seconds,
        "sampling_seconds": sampling_seconds,
        "stats_seconds": stats_seconds,
        "total_seconds": total_seconds,
        "fit_converged": fit_converged,
        "fit_iterations": fit_iterations,
        "status": status,
        "message": message,
    }


def warm_up() -> None:
    """One unmeasured smoke call to load Rust-backed code (Section 17)."""
    observed, derived = make_observed_case(20, seed=62, self_loops=False)
    fit = fit_gc_case(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        observed=observed,
        derived=derived,
        self_loops=False,
    )
    sample_gc_case(
        fit=fit,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        observed=observed,
        derived=derived,
        self_loops=False,
        seed=0,
    )
    sample_micro_case(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        observed=observed,
        derived=derived,
        self_loops=False,
        seed=0,
    )


# ---------------------------------------------------------------------------
# Main matrix (Sections 33, 51-52)
# ---------------------------------------------------------------------------


def run_main_matrix(
    *,
    node_counts: tuple[int, ...] = NODE_COUNTS,
    sample_count: int = SAMPLES_PER_CELL,
    budget_overrides: dict[tuple[ModelFamily, Constraint], tuple[int, int]]
    | None = None,
    results_dir: Path = RESULTS_DIR,
    warm: bool = True,
    run_timing: bool = True,
) -> dict[str, int]:
    """Run the full N x family x constraint x ensemble matrix with checkpoints.

    Returns a dict mapping cell outcome statuses to counts. A cell is
    ``N x family x constraint x ensemble``; completed ``ok`` cells are
    skipped on resume.
    """
    if warm:
        warm_up()
    store = ResultStore(results_dir)
    timings: list[dict[str, object]] = store._load_rows(store.timings_path)
    science: list[dict[str, object]] = store._load_rows(store.science_path)
    comparison: list[dict[str, object]] = store._load_rows(store.comparison_path)
    per_node: dict[str, NDArray[np.float64]] = _load_per_node(store.npz_path)
    outcomes: dict[str, int] = {}

    for node_count in node_counts:
        observed, derived = make_observed_case(
            node_count, seed=OBSERVED_SEED + node_count, self_loops=SELF_LOOPS
        )
        _log(
            f"N={node_count}: edges={observed.edges.num_edges} "
            f"events={observed.edges.total_events} B layers={derived.binomial_layers}"
        )
        for family in FAMILIES:
            for constraint in CONSTRAINTS:
                gate_budget = (
                    budget_overrides.get(
                        (family, constraint), (BURN_IN_SWEEPS, SWEEPS_PER_SAMPLE)
                    )
                    if budget_overrides
                    else (BURN_IN_SWEEPS, SWEEPS_PER_SAMPLE)
                )
                burn_in, sweeps = effective_budget(node_count, constraint, gate_budget)
                budget_note = ""
                if (burn_in, sweeps) != gate_budget:
                    budget_note = (
                        f"micro budget capped from gate {gate_budget[0]}+{gate_budget[1]} "
                        f"to {burn_in}+{sweeps} for feasibility at N={node_count}"
                    )
                status = _process_family_constraint(
                    store,
                    node_count=node_count,
                    family=family,
                    constraint=constraint,
                    observed=observed,
                    derived=derived,
                    sample_count=sample_count,
                    burn_in_sweeps=burn_in,
                    sweeps_per_sample=sweeps,
                    budget_note=budget_note,
                    timings=timings,
                    science=science,
                    comparison=comparison,
                    per_node=per_node,
                    run_timing=run_timing,
                )
                outcomes[status] = outcomes.get(status, 0) + 1
                # Flush incrementally so interrupted runs never lose cells.
                _flush_results(
                    store,
                    timings=timings,
                    science=science,
                    comparison=comparison,
                    per_node=per_node,
                )

    _flush_results(
        store,
        timings=timings,
        science=science,
        comparison=comparison,
        per_node=per_node,
    )
    return outcomes


def _load_per_node(path: Path) -> dict[str, NDArray[np.float64]]:
    """Load the per-node means archive if it exists and is readable."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with np.load(path) as data:
            return {key: data[key].copy() for key in data.files}
    except (EOFError, OSError, ValueError):
        # A previously interrupted atomic write leaves an empty archive.
        return {}


def _flush_results(
    store: ResultStore,
    *,
    timings: list[dict[str, object]],
    science: list[dict[str, object]],
    comparison: list[dict[str, object]],
    per_node: dict[str, NDArray[np.float64]],
) -> None:
    """Persist the full tables atomically (idempotent on resume)."""
    store.write_rows(store.timings_path, timings)
    store.write_rows(store.science_path, science)
    store.write_rows(store.comparison_path, comparison)
    if per_node:
        store.save_per_node_npz(per_node)
    _write_metadata(store, derived_by_n=_b_layers_by_n())


def _process_family_constraint(
    store: ResultStore,
    *,
    node_count: int,
    family: ModelFamily,
    constraint: Constraint,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
    sample_count: int,
    burn_in_sweeps: int,
    sweeps_per_sample: int,
    budget_note: str = "",
    timings: list[dict[str, object]],
    science: list[dict[str, object]],
    comparison: list[dict[str, object]],
    per_node: dict[str, NDArray[np.float64]],
    run_timing: bool,
) -> str:
    """Compute and checkpoint both ensemble cells for one (N, family, constraint).

    Returns the overall status: ok / fit_failed / invalid_micro_sample.
    """
    gc_keys = store.cell_key(node_count, family, constraint, Ensemble.GRAND_CANONICAL)
    micro_keys = store.cell_key(node_count, family, constraint, Ensemble.MICROCANONICAL)
    gc_done = store.cell_done(gc_keys)
    micro_done = store.cell_done(micro_keys)

    if gc_done and micro_done:
        _log(f"[{node_count}|{family.value}|{constraint.value}] skipped (checkpoint)")
        return "skipped"

    # Idempotent recompute: drop any previously recorded rows for this
    # (N, family, constraint) so a partial-cell rerun never duplicates rows.
    _drop_cell_rows(timings, node_count, family, constraint)
    _drop_cell_rows(science, node_count, family, constraint)
    _drop_cell_rows(comparison, node_count, family, constraint)
    _drop_cell_nodes(per_node, node_count, family, constraint)

    try:
        gc = run_gc_ensemble(
            family=family,
            constraint=constraint,
            observed=observed,
            derived=derived,
            node_count=node_count,
            sample_count=sample_count,
        )
    except Exception as error:
        gc = EnsembleAggregate(node_count=node_count)
        gc.status = "error"
        gc.message = f"gc: {type(error).__name__}: {error}"
        _log(f"  gc cell error: {gc.message}")
    try:
        micro = run_micro_ensemble(
            family=family,
            constraint=constraint,
            observed=observed,
            derived=derived,
            node_count=node_count,
            sample_count=sample_count,
            burn_in_sweeps=burn_in_sweeps,
            sweeps_per_sample=sweeps_per_sample,
        )
    except Exception as error:
        micro = EnsembleAggregate(node_count=node_count)
        micro.status = "error"
        micro.message = f"micro: {type(error).__name__}: {error}"
        _log(f"  micro cell error: {micro.message}")
    _log(
        f"[{node_count}|{family.value}|{constraint.value}] "
        f"gc={gc.status} micro={micro.status}"
    )

    if run_timing:
        for ensemble in ENSEMBLES:
            for timing_repeat in range(TIMING_REPEATS_BY_N.get(node_count, 1)):
                timings.append(
                    run_timing_workload(
                        node_count=node_count,
                        family=family,
                        constraint=constraint,
                        ensemble=ensemble,
                        observed=observed,
                        derived=derived,
                        timing_repeat=timing_repeat,
                    )
                )

    for ensemble, aggregate in (
        (Ensemble.GRAND_CANONICAL, gc),
        (Ensemble.MICROCANONICAL, micro),
    ):
        if aggregate.status != "ok":
            continue
        science.extend(
            summarize_ensemble(
                aggregate,
                node_count=node_count,
                family=family,
                constraint=constraint,
                ensemble=ensemble,
                sample_count=sample_count,
            )
        )
    comparison.extend(
        compare_ensembles(
            micro,
            gc,
            node_count=node_count,
            family=family,
            constraint=constraint,
            sample_count=sample_count,
        )
    )
    _store_node_arrays(per_node, gc, micro, node_count, family, constraint)

    status = _cell_status(micro, gc)
    message = _cell_message(micro, gc)
    if budget_note:
        message = (message + "; " if message else "") + budget_note
    store.set_cell(
        gc_keys,
        status=gc.status,
        sample_count=sample_count,
        message=message,
    )
    store.set_cell(
        micro_keys,
        status=micro.status,
        sample_count=sample_count,
        message=message,
    )
    return status


def _drop_cell_rows(
    rows: list[dict[str, object]],
    node_count: int,
    family: ModelFamily,
    constraint: Constraint,
) -> None:
    """Remove every recorded row for one (N, family, constraint)."""
    rows[:] = [
        row
        for row in rows
        if not (
            int(row.get("node_count", -1)) == node_count
            and str(row.get("family", "")) == family.value
            and str(row.get("constraint", "")) == constraint.value
        )
    ]


def _drop_cell_nodes(
    per_node: dict[str, NDArray[np.float64]],
    node_count: int,
    family: ModelFamily,
    constraint: Constraint,
) -> None:
    """Remove stored per-node arrays for one (N, family, constraint)."""
    prefix = f"{node_count}-{family.value}-{constraint.value}-"
    for key in [key for key in per_node if key.startswith(prefix)]:
        del per_node[key]


def _store_node_arrays(
    per_node: dict[str, NDArray[np.float64]],
    gc: EnsembleAggregate,
    micro: EnsembleAggregate,
    node_count: int,
    family: ModelFamily,
    constraint: Constraint,
) -> None:
    for ensemble, aggregate in (
        (Ensemble.GRAND_CANONICAL, gc),
        (Ensemble.MICROCANONICAL, micro),
    ):
        if aggregate.status != "ok":
            continue
        tag = f"{node_count}-{family.value}-{constraint.value}-{ensemble.value}"
        for (obs, direction), values in aggregate.node_level.items():
            mean = np.nanmean(values, axis=0)
            sd = np.nanstd(values, axis=0, ddof=1)
            se = np.where(np.isfinite(sd), sd / np.sqrt(values.shape[0]), np.nan)
            per_node[f"{tag}|{obs}_{direction}|mean"] = mean
            per_node[f"{tag}|{obs}_{direction}|sd"] = sd
            per_node[f"{tag}|{obs}_{direction}|se"] = se


# ---------------------------------------------------------------------------
# Result writing (Sections 26-30)
# ---------------------------------------------------------------------------


def save_results(
    store: ResultStore,
    timings: list[dict[str, object]],
    science: list[dict[str, object]],
    comparison: list[dict[str, object]],
    per_node: dict[str, NDArray[np.float64]],
) -> None:
    """Write all result files atomically (Section 26)."""
    if timings:
        store.append_rows(store.timings_path, timings)
    if science:
        store.append_rows(store.science_path, science)
    if comparison:
        store.append_rows(store.comparison_path, comparison)
    if per_node:
        store.save_per_node_npz(per_node)
    _write_metadata(store, derived_by_n=_b_layers_by_n())


def _b_layers_by_n() -> dict[str, int]:
    layers: dict[str, int] = {}
    for node_count in NODE_COUNTS:
        _, derived = make_observed_case(
            node_count, seed=OBSERVED_SEED + node_count, self_loops=SELF_LOOPS
        )
        layers[str(node_count)] = derived.binomial_layers
    return layers


def _write_metadata(store: ResultStore, derived_by_n: dict[str, int]) -> None:
    metadata = collect_metadata(derived_by_n=derived_by_n)
    fd, temp_name = tempfile.mkstemp(dir=store.dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        Path(temp_name).replace(store.metadata_path)
    except BaseException:
        with suppress(OSError):
            Path(temp_name).unlink()
        raise


def collect_metadata(derived_by_n: dict[str, int] | None = None) -> dict[str, object]:
    """Collect environment metadata (Section 30)."""
    commit = "unknown"
    try:
        if Path(".git").exists():
            process = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            commit = process.stdout.strip().splitlines()[0]
    except (FileNotFoundError, OSError):
        commit = "unknown"
    import menobis

    metadata: dict[str, object] = {
        "git_commit": commit,
        "menobis_version": getattr(menobis, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "node_counts": list(NODE_COUNTS),
        "sample_count": SAMPLES_PER_CELL,
        "self_loops": SELF_LOOPS,
        "average_degree": AVERAGE_DEGREE,
        "events_per_edge": EVENTS_PER_EDGE,
        "observed_seed_rule": "main: 42 + n; sparsity: 142 + int(average_degree)",
        "burn_in_sweeps": BURN_IN_SWEEPS,
        "sweeps_per_sample": SWEEPS_PER_SAMPLE,
        "budget_ladder": [list(b) for b in BUDGET_LADDER],
        "budget_caps": {
            str(n): {c.value: list(b) for c, b in caps.items()}
            for n, caps in BUDGET_CAP_BY_N.items()
        },
        "timing_repeats_by_n": TIMING_REPEATS_BY_N,
        "families": [family.value for family in FAMILIES],
        "constraints": [constraint.value for constraint in CONSTRAINTS],
        "B_layers_per_N": derived_by_n or {},
    }
    gate_rows = _gate_decisions()
    if gate_rows:
        metadata["budget_gate"] = gate_rows
    return metadata


def _gate_decisions() -> dict[str, dict[str, dict[str, object]]]:
    """Per family/constraint gate decision recovered from the gate CSV."""
    path = RESULTS_DIR / "mcmc-budget-gate.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    decisions = frame[frame.get("budget_pair", "") == "decision"]
    result: dict[str, dict[str, dict[str, object]]] = {}
    for _index, row in decisions.iterrows():  # type: ignore[union-attr]
        family = str(row["family"])
        constraint = str(row["constraint"])
        result.setdefault(family, {})[constraint] = {
            "decision": str(row["decision"]),
            "chosen_burn_in_sweeps": _nan_none(row.get("chosen_burn_in_sweeps")),
            "chosen_sweeps_per_sample": _nan_none(row.get("chosen_sweeps_per_sample")),
            "max_d_rel_a_vs_b": _nan_none(row.get("max_d_rel_a_vs_b")),
            "max_d_rel_b_vs_c": _nan_none(row.get("max_d_rel_b_vs_c")),
            "max_d_rel_c_vs_d": _nan_none(row.get("max_d_rel_c_vs_d")),
            "max_d_rel_d_vs_e": _nan_none(row.get("max_d_rel_d_vs_e")),
        }
    return result


def _nan_none(value: object) -> object:
    """Convert pandas NaN to None for JSON serialization."""
    if value is None:
        return None
    try:
        if float(value) != float(value):  # NaN check
            return None
        return float(value)
    except (TypeError, ValueError):
        return value


def _budget_size(budget: tuple[int, int]) -> int:
    """Total effective sweeps: burn-in + 10 sample sweeps."""
    return budget[0] + 10 * budget[1]


def effective_budget(
    node_count: int,
    constraint: Constraint,
    gate_budget: tuple[int, int],
) -> tuple[int, int]:
    """Gate-chosen micro budget capped for feasibility at large N."""
    cap = BUDGET_CAP_BY_N.get(node_count, {}).get(constraint)
    if cap is None or _budget_size(gate_budget) <= _budget_size(cap):
        return gate_budget
    return cap


# ---------------------------------------------------------------------------
# Budget gate (Section 14, extended ladder)
# ---------------------------------------------------------------------------


def _micro_with_budget(
    *,
    family: ModelFamily,
    constraint: Constraint,
    observed: SyntheticNetwork,
    derived: SyntheticConstraints,
    node_count: int,
    budget: tuple[int, int],
) -> EnsembleAggregate:
    """Run a micro ensemble; return an error aggregate if the cell is infeasible."""
    try:
        return run_micro_ensemble(
            family=family,
            constraint=constraint,
            observed=observed,
            derived=derived,
            node_count=node_count,
            sample_count=SAMPLES_PER_CELL,
            burn_in_sweeps=budget[0],
            sweeps_per_sample=budget[1],
        )
    except Exception as error:
        aggregate = EnsembleAggregate(node_count=node_count)
        aggregate.status = "error"
        aggregate.message = f"{type(error).__name__}: {error}"
        return aggregate


def run_mcmc_budget_gate(
    results_dir: Path = RESULTS_DIR,
) -> tuple[bool, dict[tuple[ModelFamily, Constraint], tuple[int, int]]]:
    """Within-micro MCMC budget stability check at N=100 (Section 14).

    Extended ladder (user decision): walk budget pairs ``(A,B), (B,C),
    (C,D), (D,E)`` and choose the smallest budget whose ensemble is within
    ``D_rel <= 0.05`` of the next; if no pair is stable, keep the largest
    budget and mark the cell ``unstable_at_max_budget`` instead of stopping.

    Returns ``(overall_stop, chosen_budget_per_family_constraint)`` with
    ``overall_stop`` always False in the extended decision.
    """
    store = ResultStore(results_dir)
    rows: list[dict[str, object]] = []
    chosen: dict[tuple[ModelFamily, Constraint], tuple[int, int]] = {}
    overall_stop = False
    node_count = 100

    for family in FAMILIES:
        for constraint in BUDGET_GATE_CONSTRAINTS:
            observed, derived = make_observed_case(
                node_count, seed=OBSERVED_SEED + node_count, self_loops=SELF_LOOPS
            )
            primary = PRIMARY_OBSERVABLES[constraint]
            aggregates: dict[tuple[int, int], EnsembleAggregate] = {}
            for budget in BUDGET_LADDER:
                aggregates[budget] = _micro_with_budget(
                    family=family,
                    constraint=constraint,
                    observed=observed,
                    derived=derived,
                    node_count=node_count,
                    budget=budget,
                )
            ok = all(aggregates[b].status == "ok" for b in BUDGET_LADDER)
            pair_drel: dict[tuple[tuple[int, int], tuple[int, int]], float | None] = {}
            for budget_prev, budget_next in itertools.pairwise(BUDGET_LADDER):
                d_rows = _primary_d_rel_table(
                    aggregates[budget_prev], aggregates[budget_next], primary
                )
                for row in d_rows:
                    rows.append(
                        {
                            **row,
                            "family": family.value,
                            "constraint": constraint.value,
                            "budget_pair": (
                                f"{_budget_label(budget_prev)}_vs_{_budget_label(budget_next)}"
                            ),
                        }
                    )
                pair_drel[(budget_prev, budget_next)] = _max_finite(d_rows)

            decision: str
            if not ok:
                decision = "infeasible_or_error"
                chosen[(family, constraint)] = BUDGET_A
            else:
                chosen_budget: tuple[int, int] | None = None
                for budget_prev, budget_next in itertools.pairwise(BUDGET_LADDER):
                    max_d = pair_drel[(budget_prev, budget_next)]
                    if max_d is not None and max_d <= 0.05:
                        chosen_budget = budget_prev
                        break
                if chosen_budget is None:
                    decision = "unstable_at_max_budget"
                    chosen_budget = BUDGET_E
                else:
                    decision = f"use_{chosen_budget[0]}_{chosen_budget[1]}"
                chosen[(family, constraint)] = chosen_budget
            chosen_burn, chosen_sweeps = chosen[(family, constraint)]
            rows.append(
                {
                    "family": family.value,
                    "constraint": constraint.value,
                    "budget_pair": "decision",
                    "observable": "",
                    "direction": "",
                    "d_rel": float("nan"),
                    "max_d_rel_a_vs_b": pair_drel.get((BUDGET_A, BUDGET_B)),
                    "max_d_rel_b_vs_c": pair_drel.get((BUDGET_B, BUDGET_C)),
                    "max_d_rel_c_vs_d": pair_drel.get((BUDGET_C, BUDGET_D)),
                    "max_d_rel_d_vs_e": pair_drel.get((BUDGET_D, BUDGET_E)),
                    "decision": decision,
                    "chosen_burn_in_sweeps": chosen_burn,
                    "chosen_sweeps_per_sample": chosen_sweeps,
                }
            )
            _log(
                f"budget gate {family.value}/{constraint.value}: "
                f"D_rel(A,B)={_fmt(pair_drel.get((BUDGET_A, BUDGET_B)))} "
                f"(B,C)={_fmt(pair_drel.get((BUDGET_B, BUDGET_C)))} "
                f"(C,D)={_fmt(pair_drel.get((BUDGET_C, BUDGET_D)))} "
                f"(D,E)={_fmt(pair_drel.get((BUDGET_D, BUDGET_E)))} "
                f"decision={decision} budget={chosen_burn}+{chosen_sweeps}"
            )

    _atomic_csv(pd.DataFrame(rows), store.gate_path)
    return overall_stop, chosen


def _budget_label(budget: tuple[int, int]) -> str:
    """Budget letter used in the gate CSV (A..E)."""
    labels = {
        budget: letter for letter, budget in zip("ABCDE", BUDGET_LADDER, strict=True)
    }
    return labels.get(budget, f"{budget[0]}_{budget[1]}")


def _fmt(value: float | None) -> str:
    """Format an optional D_rel value for console output."""
    return "NaN" if value is None else f"{value:.4f}"


def _primary_d_rel_table(
    first: EnsembleAggregate,
    second: EnsembleAggregate,
    primary: tuple[Observable, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for obs, direction in primary:
        a = first.observable_values((obs, direction))
        b = second.observable_values((obs, direction))
        if a.size == 0 or b.size == 0:
            continue
        if direction == "global":
            d_rel = d_rel_global(float(np.nanmean(a)), float(np.nanmean(b)))
        else:
            d_rel, _ = d_rel_node_level(np.nanmean(a, axis=0), np.nanmean(b, axis=0))
        rows.append({"observable": obs, "direction": direction, "d_rel": d_rel})
    return rows


def _max_finite(rows: list[dict[str, object]]) -> float | None:
    values = [float(row["d_rel"]) for row in rows if np.isfinite(float(row["d_rel"]))]
    return max(values) if values else None


# ---------------------------------------------------------------------------
# Sparsity sensitivity (Section 3)
# ---------------------------------------------------------------------------


def _gate_chosen_budgets(
    results_dir: Path = RESULTS_DIR,
) -> dict[tuple[ModelFamily, Constraint], tuple[int, int]]:
    """Recover per (family, constraint) gate-chosen budgets from the CSV."""
    path = results_dir / "mcmc-budget-gate.csv"
    chosen: dict[tuple[ModelFamily, Constraint], tuple[int, int]] = {}
    if not path.exists():
        return chosen
    frame = pd.read_csv(path)
    decisions = frame[frame.get("budget_pair", "") == "decision"]
    for _index, row in decisions.iterrows():  # type: ignore[union-attr]
        burn = row.get("chosen_burn_in_sweeps")
        sweeps = row.get("chosen_sweeps_per_sample")
        if burn is None or sweeps is None or pd.isna(burn) or pd.isna(sweeps):
            continue
        family = ModelFamily(str(row["family"]))
        constraint = Constraint(str(row["constraint"]))
        chosen[(family, constraint)] = (int(burn), int(sweeps))
    return chosen


# ---------------------------------------------------------------------------
# Sparsity sensitivity (Section 3)
# ---------------------------------------------------------------------------


def run_sparsity_sensitivity(
    *,
    results_dir: Path = RESULTS_DIR,
    warm: bool = True,
    budget_overrides: dict[tuple[ModelFamily, Constraint], tuple[int, int]]
    | None = None,
) -> None:
    """Sparsity sensitivity at N=500, ME only (Section 3)."""
    if warm:
        warm_up()
    if budget_overrides is None:
        budget_overrides = _gate_chosen_budgets(results_dir)
    store = ResultStore(results_dir)
    rows: list[dict[str, object]] = []
    for average_degree in SPARSITY_DEGREES:
        observed, derived = make_observed_case(
            SPARSITY_NODE_COUNT,
            average_degree=average_degree,
            events_per_edge=SPARSITY_EVENTS_PER_EDGE,
            seed=SPARSITY_OBSERVED_SEED + int(average_degree),
            self_loops=SPARSITY_SELF_LOOPS,
        )
        _log(
            f"sparsity k={average_degree}: edges={observed.edges.num_edges} "
            f"events={observed.edges.total_events} "
            f"T/E={observed.edges.total_events / observed.edges.num_edges:.2f}"
        )
        for constraint in SPARSITY_CONSTRAINTS:
            burn_in, sweeps = effective_budget(
                SPARSITY_NODE_COUNT,
                constraint,
                budget_overrides.get(
                    (SPARSITY_FAMILY, constraint), (BURN_IN_SWEEPS, SWEEPS_PER_SAMPLE)
                ),
            )
            try:
                gc = run_gc_ensemble(
                    family=SPARSITY_FAMILY,
                    constraint=constraint,
                    observed=observed,
                    derived=derived,
                    node_count=SPARSITY_NODE_COUNT,
                    sample_count=SPARSITY_SAMPLES,
                )
            except Exception as error:
                gc = EnsembleAggregate(node_count=SPARSITY_NODE_COUNT)
                gc.status = "error"
                gc.message = f"gc: {type(error).__name__}: {error}"
            try:
                micro = run_micro_ensemble(
                    family=SPARSITY_FAMILY,
                    constraint=constraint,
                    observed=observed,
                    derived=derived,
                    node_count=SPARSITY_NODE_COUNT,
                    sample_count=SPARSITY_SAMPLES,
                    burn_in_sweeps=burn_in,
                    sweeps_per_sample=sweeps,
                )
            except Exception as error:
                micro = EnsembleAggregate(node_count=SPARSITY_NODE_COUNT)
                micro.status = "error"
                micro.message = f"micro: {type(error).__name__}: {error}"
            _log(f"  [{constraint.value}] gc={gc.status} micro={micro.status}")
            for ens, aggregate in (
                (Ensemble.GRAND_CANONICAL, gc),
                (Ensemble.MICROCANONICAL, micro),
            ):
                if aggregate.status != "ok":
                    rows.append(
                        {
                            "node_count": SPARSITY_NODE_COUNT,
                            "average_degree": float(average_degree),
                            "family": SPARSITY_FAMILY.value,
                            "constraint": constraint.value,
                            "ensemble": ens.value,
                            "observable": "",
                            "direction": "",
                            "sample_count": SPARSITY_SAMPLES,
                            "mean_node_average": float("nan"),
                            "mean_within_ensemble_sd": float("nan"),
                            "mean_mc_se": float("nan"),
                            "valid_node_count": 0,
                            "d_rel": float("nan"),
                            "spearman": float("nan"),
                            "status": aggregate.status,
                            "message": aggregate.message,
                        }
                    )
                    continue
                rows.extend(
                    {
                        **row,
                        "average_degree": float(average_degree),
                        "d_rel": float("nan"),
                        "spearman": float("nan"),
                    }
                    for row in summarize_ensemble(
                        aggregate,
                        node_count=SPARSITY_NODE_COUNT,
                        family=SPARSITY_FAMILY,
                        constraint=constraint,
                        ensemble=ens,
                        sample_count=SPARSITY_SAMPLES,
                    )
                )
            rows.extend(
                {
                    **row,
                    "average_degree": float(average_degree),
                    "sample_count": SPARSITY_SAMPLES,
                    "mean_node_average": float("nan"),
                    "mean_within_ensemble_sd": float("nan"),
                    "mean_mc_se": float("nan"),
                    "d_rel": row["d_rel"],
                    "spearman": row["spearman"],
                }
                for row in compare_ensembles(
                    micro,
                    gc,
                    node_count=SPARSITY_NODE_COUNT,
                    family=SPARSITY_FAMILY,
                    constraint=constraint,
                    sample_count=SPARSITY_SAMPLES,
                )
            )
    _atomic_csv(pd.DataFrame(rows), store.sparsity_path)


# ---------------------------------------------------------------------------
# Smoke command (Section 49)
# ---------------------------------------------------------------------------


def run_smoke(results_dir: Path = RESULTS_DIR) -> None:
    """N=100 ME STRENGTH + EDGES_EVENTS, both ensembles, 2 samples."""
    failures: list[str] = []
    for constraint in (Constraint.STRENGTH, Constraint.EDGES_EVENTS):
        observed, derived = make_observed_case(
            100, seed=OBSERVED_SEED + 100, self_loops=False
        )
        gc = run_gc_ensemble(
            family=ModelFamily.ME,
            constraint=constraint,
            observed=observed,
            derived=derived,
            node_count=100,
            sample_count=2,
        )
        if gc.status != "ok":
            failures.append(f"ME/{constraint.value}/GC: {gc.message}")
        micro = run_micro_ensemble(
            family=ModelFamily.ME,
            constraint=constraint,
            observed=observed,
            derived=derived,
            node_count=100,
            sample_count=2,
        )
        if micro.status != "ok":
            failures.append(f"ME/{constraint.value}/micro: {micro.message}")
        _log(f"smoke ME/{constraint.value}: gc={gc.status} micro={micro.status}")
    if failures:
        _log("SMOKE FAILED:\n" + "\n".join(failures))
        raise SystemExit(1)
    _log("SMOKE PASS")


# ---------------------------------------------------------------------------
# CLI (Section 32)
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    print(message, flush=True)


def main(argv: list[str] | None = None) -> None:
    """Runner CLI: smoke | budget-gate | main | sparsity | all (Section 32)."""
    parser = argparse.ArgumentParser(
        prog="benchmarks.ensemble_comparison",
        description="MENoBiS microcanonical vs grand-canonical comparison.",
    )
    parser.add_argument(
        "command",
        choices=("smoke", "budget-gate", "main", "sparsity", "all"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="results directory (default: benchmarks/results/ensemble-comparison)",
    )
    parser.add_argument(
        "--no-warm", action="store_true", help="skip the warm-up smoke call"
    )
    args = parser.parse_args(argv)

    if args.command == "smoke":
        run_smoke(args.results_dir)
        return

    if args.command == "budget-gate":
        stop, chosen = run_mcmc_budget_gate(args.results_dir)
        _log(f"budget gate: stop={stop}")
        _log(
            "chosen budgets: "
            + json.dumps(
                {
                    f"{fam.value}/{cons.value}": list(budget)
                    for (fam, cons), budget in chosen.items()
                },
                sort_keys=True,
            )
        )
        if stop:
            _log("BUDGET GATE FAILED: sampling-budget stability not established")
            raise SystemExit(2)
        _log("BUDGET GATE PASS")
        return

    if args.command == "main":
        stop, chosen = run_mcmc_budget_gate(args.results_dir)
        if stop:
            _log("Budget gate failed; not running main matrix.")
            raise SystemExit(2)
        outcomes = run_main_matrix(
            results_dir=args.results_dir,
            warm=not args.no_warm,
            budget_overrides=chosen,
        )
        _log("main matrix outcomes: " + json.dumps(outcomes, sort_keys=True))
        return

    if args.command == "sparsity":
        run_sparsity_sensitivity(results_dir=args.results_dir, warm=not args.no_warm)
        return

    if args.command == "all":
        stop, chosen = run_mcmc_budget_gate(args.results_dir)
        if stop:
            _log("Budget gate failed; skipping main and sparsity.")
            raise SystemExit(2)
        outcomes = run_main_matrix(
            results_dir=args.results_dir,
            warm=not args.no_warm,
            budget_overrides=chosen,
        )
        run_sparsity_sensitivity(results_dir=args.results_dir, warm=False)
        _log("main matrix outcomes: " + json.dumps(outcomes, sort_keys=True))
        return


if __name__ == "__main__":
    main()
