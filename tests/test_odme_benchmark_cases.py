"""Benchmark case coverage tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bench_fitting import CONSTRAINTS, fitting_case_registry


def _case_key(case: dict[str, object]) -> tuple[object, ...]:
    return (
        case["ensemble"],
        case["family"],
        case["constraint"],
        case["layers"],
        case["partial"],
    )


def test_fitting_case_registry_covers_all_full_model_combinations() -> None:
    """Canonical fitting benchmarks include ME/B/W for every constraint."""
    expected = (
        {("ME", "poisson", constraint, None, False) for constraint in CONSTRAINTS}
        | {("B", "binomial", constraint, 3, False) for constraint in CONSTRAINTS}
        | {("W", "geometric", constraint, None, False) for constraint in CONSTRAINTS}
        | {
            ("W", "negative-binomial", constraint, 3, False)
            for constraint in CONSTRAINTS
        }
    )

    actual = {_case_key(case) for case in fitting_case_registry(include_partial=False)}

    assert actual == expected


def test_fitting_case_registry_includes_all_partial_combinations() -> None:
    """Partial benchmark coverage mirrors the full ME/B/W matrix."""
    expected = (
        {("ME", "poisson", constraint, None, True) for constraint in CONSTRAINTS}
        | {("B", "binomial", constraint, 3, True) for constraint in CONSTRAINTS}
        | {("W", "geometric", constraint, None, True) for constraint in CONSTRAINTS}
        | {
            ("W", "negative-binomial", constraint, 3, True)
            for constraint in CONSTRAINTS
        }
    )

    actual = {_case_key(case) for case in fitting_case_registry(include_partial=True)}

    assert expected <= actual
