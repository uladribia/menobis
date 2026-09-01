"""Contract and unit tests for the ensemble comparison runner.

Covers the ten mandatory runner tests from the implementation plan:

1. constraint-kwargs builder covers all six constraints;
2. N=20 ME STRENGTH + EDGES_EVENTS smoke for GC and micro;
3. correct micro STRENGTH validation passes;
4. deliberate target mismatch is caught;
5. D_rel of identical vectors is zero;
6. finite-mask NaN handling gives the exact valid-node count;
7. Spearman with fewer than 3 valid nodes is NaN;
8. timing helper excludes observed-network generation;
9. seed rule is deterministic;
10. result schemas contain all mandatory columns.

N=500/2000 are intentionally not exercised here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import benchmarks.ensemble_comparison as ec
from menobis.models.spec import Constraint, Ensemble, ModelFamily

# Column sets from Sections 27-29 of the plan.
TIMING_COLUMNS = (
    "node_count",
    "family",
    "constraint",
    "ensemble",
    "self_loops",
    "layers",
    "sample_count",
    "burn_in_sweeps",
    "sweeps_per_sample",
    "timing_repeat",
    "fit_seconds",
    "sampling_seconds",
    "stats_seconds",
    "total_seconds",
    "fit_converged",
    "fit_iterations",
    "status",
    "message",
)
SCIENCE_COLUMNS = (
    "node_count",
    "family",
    "constraint",
    "ensemble",
    "observable",
    "direction",
    "sample_count",
    "mean_node_average",
    "mean_within_ensemble_sd",
    "mean_mc_se",
    "valid_node_count",
    "status",
    "message",
)
COMPARISON_COLUMNS = (
    "node_count",
    "family",
    "constraint",
    "observable",
    "direction",
    "micro_mean_node_average",
    "gc_mean_node_average",
    "d_rel",
    "spearman",
    "valid_node_count",
    "micro_mean_mc_se",
    "gc_mean_mc_se",
    "status",
    "message",
)


def _small_case() -> tuple[ec.SyntheticNetwork, ec.SyntheticConstraints]:
    """N=100 ME STRENGTH observed case shared by several tests."""
    return ec.make_observed_case(100, seed=ec.OBSERVED_SEED + 100, self_loops=False)


# ---------------------------------------------------------------------------
# 1. Constraint-kwargs builder covers all six constraints
# ---------------------------------------------------------------------------


def test_builder_covers_all_six_constraints() -> None:
    observed, derived = _small_case()
    expectation: dict[Constraint, frozenset[str]] = {
        Constraint.STRENGTH: frozenset({"strength_out", "strength_in"}),
        Constraint.STRENGTH_COST: frozenset(
            {"strength_out", "strength_in", "coord_x", "coord_y", "target_cost"}
        ),
        Constraint.STRENGTH_EDGES: frozenset(
            {"strength_out", "strength_in", "target_edges"}
        ),
        Constraint.STRENGTH_DEGREE: frozenset(
            {"strength_out", "strength_in", "degree_out", "degree_in"}
        ),
        Constraint.DEGREE_EVENTS: frozenset(
            {"degree_out", "degree_in", "total_events"}
        ),
        Constraint.EDGES_EVENTS: frozenset(
            {"node_count", "target_edges", "total_events"}
        ),
    }
    for constraint, expected in expectation.items():
        kwargs = ec.build_constraint_kwargs(
            constraint=constraint, observed=observed, derived=derived
        )
        assert frozenset(kwargs) == expected, (
            f"{constraint.value}: {sorted(kwargs)} != {sorted(expected)}"
        )
    # self_loops=False must be threaded into the fit kwargs.
    fit_kwargs = ec.make_fit_kwargs(
        constraint=Constraint.STRENGTH,
        observed=observed,
        derived=derived,
        family=ModelFamily.ME,
        self_loops=False,
    )
    assert fit_kwargs["self_loops"] is False


# ---------------------------------------------------------------------------
# 2. N=20 ME STRENGTH + EDGES_EVENTS smoke for GC and micro
# ---------------------------------------------------------------------------


def test_small_smoke_gc_and_micro() -> None:
    observed, derived = _small_case()
    for constraint in (Constraint.STRENGTH, Constraint.EDGES_EVENTS):
        gc = ec.run_gc_ensemble(
            family=ModelFamily.ME,
            constraint=constraint,
            observed=observed,
            derived=derived,
            node_count=100,
            sample_count=2,
        )
        assert gc.status == "ok", gc.message
        assert gc.fit_converged is True
        micro = ec.run_micro_ensemble(
            family=ModelFamily.ME,
            constraint=constraint,
            observed=observed,
            derived=derived,
            node_count=100,
            sample_count=2,
        )
        assert micro.status == "ok", micro.message


# ---------------------------------------------------------------------------
# 3. Correct micro STRENGTH validation passes
# ---------------------------------------------------------------------------


def test_micro_strength_validation_passes() -> None:
    observed, derived = _small_case()
    edges = ec.sample_micro_case(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        observed=observed,
        derived=derived,
        self_loops=False,
        seed=ec.ensemble_seed(
            100, ModelFamily.ME, Constraint.STRENGTH, Ensemble.MICROCANONICAL, 0
        ),
    )
    valid, message, _extra = ec.validate_micro_constraints(
        edges, constraint=Constraint.STRENGTH, observed=observed, derived=derived
    )
    assert valid, message


# ---------------------------------------------------------------------------
# 4. Deliberate target mismatch is caught
# ---------------------------------------------------------------------------


def test_deliberate_target_mismatch_caught() -> None:
    observed, derived = _small_case()
    edges = ec.sample_micro_case(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        observed=observed,
        derived=derived,
        self_loops=False,
        seed=1,
    )
    # Corrupt the derived strengths so validation must fail.
    mismatched = derived.__class__(
        strength_out=derived.strength_out + 1.0,
        strength_in=derived.strength_in,
        degree_out=derived.degree_out,
        degree_in=derived.degree_in,
        total_edges=derived.total_edges,
        total_events=derived.total_events,
        total_cost=derived.total_cost,
        binomial_layers=derived.binomial_layers,
    )
    valid, message, _extra = ec.validate_micro_constraints(
        edges, constraint=Constraint.STRENGTH, observed=observed, derived=mismatched
    )
    assert not valid
    assert message


# ---------------------------------------------------------------------------
# 5. D_rel of identical vectors is zero
# ---------------------------------------------------------------------------


def test_d_rel_identical_vectors_zero() -> None:
    x = np.array([0.0, 1.0, 5.0, 10.0, 0.5])
    d_rel, valid = ec.d_rel_node_level(x, x.copy())
    assert valid == len(x)
    assert d_rel == 0.0


# ---------------------------------------------------------------------------
# 6. Finite-mask NaN handling gives exact valid-node count
# ---------------------------------------------------------------------------


def test_d_rel_nan_mask_valid_count() -> None:
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    b = np.array([1.1, 2.2, 3.3, np.nan, 5.5])
    d_rel, valid = ec.d_rel_node_level(a, b)
    assert valid == 3  # nodes 0, 1, 4
    assert np.isfinite(d_rel)


def test_d_rel_all_nan_gives_nan() -> None:
    a = np.array([np.nan, np.nan])
    b = np.array([1.0, 2.0])
    d_rel, valid = ec.d_rel_node_level(a, b)
    assert valid == 0
    assert np.isnan(d_rel)


# ---------------------------------------------------------------------------
# 7. Spearman with fewer than 3 valid nodes is NaN
# ---------------------------------------------------------------------------


def test_spearman_fewer_than_three_nodes_nan() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([2.0, 1.0])
    assert np.isnan(ec.spearman_correlation(a, b))


def test_spearman_three_plus_nodes_finite() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    rho = ec.spearman_correlation(a, b)
    assert np.isfinite(rho)
    assert rho > 0.99


# ---------------------------------------------------------------------------
# 8. Timing helper excludes observed-network generation
# ---------------------------------------------------------------------------


def test_timing_helper_excludes_network_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed, derived = _small_case()

    def explode(*_args: object, **_kwargs: object) -> None:
        msg = "observed-network generation must not run inside timing"
        raise AssertionError(msg)

    monkeypatch.setattr(ec, "generate_pa_geographic_network", explode)
    monkeypatch.setattr(ec, "derive_synthetic_constraints", explode)

    row = ec.run_timing_workload(
        node_count=100,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        ensemble=Ensemble.GRAND_CANONICAL,
        observed=observed,
        derived=derived,
        timing_repeat=0,
        sample_count=2,
    )
    assert row["status"] == "ok"
    total = float(cast(float, row["total_seconds"]))
    parts = sum(
        float(cast(float, row[key]))
        for key in ("fit_seconds", "sampling_seconds", "stats_seconds")
    )
    assert total == pytest.approx(parts)

    micro_row = ec.run_timing_workload(
        node_count=100,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        ensemble=Ensemble.MICROCANONICAL,
        observed=observed,
        derived=derived,
        timing_repeat=0,
        sample_count=2,
    )
    assert micro_row["status"] == "ok"
    assert micro_row["fit_seconds"] == 0.0


# ---------------------------------------------------------------------------
# 9. Seed rule is deterministic
# ---------------------------------------------------------------------------


def test_seed_rule_deterministic() -> None:
    (
        n,
        family,
        constraint,
        ensemble,
        sample_index,
    ) = (100, ModelFamily.ME, Constraint.STRENGTH, Ensemble.GRAND_CANONICAL, 3)
    seed_a = ec.ensemble_seed(n, family, constraint, ensemble, sample_index)
    seed_b = ec.ensemble_seed(n, family, constraint, ensemble, sample_index)
    other = ec.ensemble_seed(n, family, constraint, ensemble, 4)
    assert seed_a == seed_b
    assert seed_a != other
    assert 0 <= seed_a - 1_000_000 < 700_000
    # Distinct families/constraints/ensembles produce distinct seeds.
    others = {
        ec.ensemble_seed(n, fam, cons, ens, sample_index)
        for fam in ec.FAMILIES
        for cons in ec.CONSTRAINTS
        for ens in ec.ENSEMBLES
    }
    assert len(others) == len(ec.FAMILIES) * len(ec.CONSTRAINTS) * len(ec.ENSEMBLES)


# ---------------------------------------------------------------------------
# 10. Result schemas contain all mandatory columns
# ---------------------------------------------------------------------------


def test_result_schemas_contain_mandatory_columns(tmp_path: Path) -> None:
    observed, derived = _small_case()
    gc = ec.run_gc_ensemble(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        observed=observed,
        derived=derived,
        node_count=100,
        sample_count=2,
    )
    micro = ec.run_micro_ensemble(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        observed=observed,
        derived=derived,
        node_count=100,
        sample_count=2,
    )
    timing = ec.run_timing_workload(
        node_count=100,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        ensemble=Ensemble.GRAND_CANONICAL,
        observed=observed,
        derived=derived,
        timing_repeat=0,
        sample_count=2,
    )
    assert set(TIMING_COLUMNS) <= set(timing)

    science = ec.summarize_ensemble(
        gc,
        node_count=100,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        ensemble=Ensemble.GRAND_CANONICAL,
        sample_count=2,
    )
    assert science
    assert set(SCIENCE_COLUMNS) <= set(science[0])

    comparison = ec.compare_ensembles(
        micro,
        gc,
        node_count=100,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        sample_count=2,
    )
    assert comparison
    assert set(COMPARISON_COLUMNS) <= set(comparison[0])

    # Roundtrip through the store: headers must survive writing + reading.
    store = ec.ResultStore(tmp_path)
    store.append_rows(store.timings_path, [timing])
    store.append_rows(store.science_path, science)
    store.append_rows(store.comparison_path, comparison)
    stored_timing = pd.read_csv(store.timings_path)
    stored_science = pd.read_csv(store.science_path)
    stored_comparison = pd.read_csv(store.comparison_path)
    assert set(TIMING_COLUMNS) <= set(stored_timing.columns)
    assert set(SCIENCE_COLUMNS) <= set(stored_science.columns)
    assert set(COMPARISON_COLUMNS) <= set(stored_comparison.columns)


def test_runner_module_importable_without_sys_path_hacks() -> None:
    """The benchmark package must be resolvable under plain pytest."""
    assert "benchmarks" in sys.modules
    assert Path(ec.__file__).name == "ensemble_comparison.py"


def test_write_rows_is_idempotent(tmp_path: Path) -> None:
    """Full-table writes must not duplicate rows on resume."""
    store = ec.ResultStore(tmp_path)
    rows: list[dict[str, object]] = [{"a": 1, "b": 2.0}, {"a": 2, "b": 3.5}]
    store.write_rows(store.timings_path, rows)
    store.write_rows(store.timings_path, rows)
    loaded = pd.read_csv(store.timings_path)
    assert len(loaded) == len(rows)


def test_drop_cell_rows_removes_only_target_cell() -> None:
    """Row dropping for a (N, family, constraint) must be exact."""
    rows: list[dict[str, object]] = [
        {
            "node_count": 100,
            "family": "me",
            "constraint": "strength",
            "observable": "E",
        },
        {
            "node_count": 100,
            "family": "me",
            "constraint": "strength_edges",
            "observable": "E",
        },
        {
            "node_count": 500,
            "family": "me",
            "constraint": "strength",
            "observable": "E",
        },
        {"node_count": 100, "family": "b", "constraint": "strength", "observable": "E"},
    ]
    ec._drop_cell_rows(rows, 100, ModelFamily.ME, Constraint.STRENGTH)
    assert len(rows) == 3
    remaining = {
        (int(cast(int, r["node_count"])), str(r["family"]), str(r["constraint"]))
        for r in rows
    }
    assert (100, "me", "strength") not in remaining
    assert (100, "me", "strength_edges") in remaining
    assert (500, "me", "strength") in remaining
    assert (100, "b", "strength") in remaining
