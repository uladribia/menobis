"""Generate public capability and route tables from the MENoBiS capability registry.

The capability registry in ``menobis.capabilities`` is the single machine-readable
source of truth for supported ``(verb, ensemble, family, constraint)`` routes.
This script renders that registry into Markdown tables used by the public
documentation, so support claims can never drift from code.

Usage (from the repository root)::

    uv run python scripts/docs/generate_capabilities.py            # write tables
    uv run python scripts/docs/generate_capabilities.py --check    # verify no drift

``--check`` regenerates the tables in memory and fails with exit code 1 if the
committed files differ (the timestamp and git SHA lines are ignored when
comparing).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
GENERATED_DIR = DOCS_DIR / "_generated"
CAPABILITIES_FILE = GENERATED_DIR / "capabilities.md"
ROUTES_FILE = GENERATED_DIR / "microcanonical-routes.md"

# Make the package importable regardless of the invocation directory.
sys.path.insert(0, str(ROOT / "src"))

from menobis.capabilities import REGISTRY, ModelCapability  # noqa: E402
from menobis.models.spec import (  # noqa: E402
    Constraint,
    Ensemble,
    ModelFamily,
    Verb,
)

GENERATED_MARKER = "<!-- GENERATED FILE. DO NOT EDIT BY HAND. -->"

VERB_ORDER = [Verb.FIT, Verb.SAMPLE, Verb.FILTER]
ENSEMBLE_ORDER = [
    Ensemble.GRAND_CANONICAL,
    Ensemble.CANONICAL,
    Ensemble.MICROCANONICAL,
]
FAMILY_ORDER = [ModelFamily.ME, ModelFamily.B, ModelFamily.W]
CONSTRAINT_ORDER = [
    Constraint.STRENGTH,
    Constraint.STRENGTH_COST,
    Constraint.STRENGTH_EDGES,
    Constraint.STRENGTH_DEGREE,
    Constraint.DEGREE_EVENTS,
    Constraint.EDGES_EVENTS,
]

# Exactness semantics for the SAMPLE verb, keyed by (ensemble, constraint).
# Mirrors the exactness assigned at runtime in ``menobis.routing``:
# ``sample_model_detailed`` reports the same category on
# ``result.diagnostics.exactness``.
SAMPLE_SEMANTICS: dict[tuple[Ensemble, Constraint], str] = {
    (Ensemble.GRAND_CANONICAL, Constraint.STRENGTH): (
        "exact independent draws; constraints matched in expectation"
    ),
    (Ensemble.GRAND_CANONICAL, Constraint.STRENGTH_COST): (
        "exact independent draws; constraints matched in expectation"
    ),
    (Ensemble.GRAND_CANONICAL, Constraint.STRENGTH_EDGES): (
        "exact independent draws; constraints matched in expectation"
    ),
    (Ensemble.GRAND_CANONICAL, Constraint.STRENGTH_DEGREE): (
        "exact independent draws; constraints matched in expectation"
    ),
    (Ensemble.GRAND_CANONICAL, Constraint.DEGREE_EVENTS): (
        "exact independent draws; constraints matched in expectation"
    ),
    (Ensemble.GRAND_CANONICAL, Constraint.EDGES_EVENTS): (
        "exact independent draws; constraints matched in expectation"
    ),
    (Ensemble.CANONICAL, Constraint.STRENGTH): (
        "exact direct; total occupation T fixed, remaining structure soft"
    ),
    (Ensemble.MICROCANONICAL, Constraint.STRENGTH): (
        "exact stationary MCMC; strengths exact"
    ),
    (Ensemble.MICROCANONICAL, Constraint.STRENGTH_COST): (
        "exact stationary MCMC; strengths exact, cost matched in expectation"
    ),
    (Ensemble.MICROCANONICAL, Constraint.STRENGTH_EDGES): (
        "exact stationary MCMC; strengths and occupied-pair count E exact"
    ),
    (Ensemble.MICROCANONICAL, Constraint.STRENGTH_DEGREE): (
        "exact stationary MCMC; strengths and degree sequences exact"
    ),
    (Ensemble.MICROCANONICAL, Constraint.DEGREE_EVENTS): (
        "exact stationary MCMC; degree sequences and total events T exact"
    ),
    (Ensemble.MICROCANONICAL, Constraint.EDGES_EVENTS): (
        "exact direct; occupied-pair count E and total events T exact"
    ),
}

# Exact/controlled quantities for the microcanonical route table (guide page).
MC_CONTROLLED: dict[Constraint, str] = {
    Constraint.STRENGTH: "strengths exact",
    Constraint.STRENGTH_COST: "strengths exact; cost expected (gamma)",
    Constraint.STRENGTH_EDGES: "strengths, E exact",
    Constraint.STRENGTH_DEGREE: "strengths, degree sequences exact",
    Constraint.DEGREE_EVENTS: "degree sequences, T exact",
    Constraint.EDGES_EVENTS: "E, T exact",
}

# Registry argument names -> public API keyword names.
PUBLIC_ARG_ALIASES: dict[str, str] = {
    "observed_total_cost": "target_cost",
}


def public_arg_names(cap: ModelCapability) -> tuple[str, ...]:
    """Return required arguments using the public Python API keyword names."""
    return tuple(
        sorted(PUBLIC_ARG_ALIASES.get(arg, arg) for arg in cap.required_arguments)
    )


def git_short_sha() -> str:
    """Return the current short git SHA, or ``unknown`` when unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _check(
    verb: Verb, ensemble: Ensemble, family: ModelFamily, constraint: Constraint
) -> str:
    """Render a verb support cell: ``yes`` when the registry supports it."""
    cap = REGISTRY.get((verb, ensemble, family, constraint))
    return "yes" if cap is not None and cap.supported else "\u2014"


def _sample_semantics(
    ensemble: Ensemble, family: ModelFamily, constraint: Constraint
) -> str:
    """Return the SAMPLE exactness semantics for a route, or ``n/a``."""
    cap = REGISTRY.get((Verb.SAMPLE, ensemble, family, constraint))
    if cap is None or not cap.supported:
        return "n/a"
    return SAMPLE_SEMANTICS[(ensemble, constraint)]


def _family_requires_layers(family: ModelFamily) -> bool:
    """B and W routes use a layer parameter M; ME does not."""
    return family in (ModelFamily.B, ModelFamily.W)


def _row_key(
    ensemble: Ensemble, family: ModelFamily, constraint: Constraint
) -> tuple[int, ...]:
    """Deterministic sort key: ensemble, family, constraint."""
    return (
        ENSEMBLE_ORDER.index(ensemble),
        FAMILY_ORDER.index(family),
        CONSTRAINT_ORDER.index(constraint),
    )


def generate_capabilities() -> str:
    """Render the full capability matrix Markdown table."""
    lines: list[str] = [
        GENERATED_MARKER,
        "",
        "# Supported model matrix",
        "",
        "> This table is generated from MENoBiS' capability registry"
        " (`menobis.capabilities`). If the code and another documentation page"
        " disagree, this table is authoritative for public support.",
        "",
        f"_Registry source SHA: `{git_short_sha()}` · generated:"
        f" {datetime.now(UTC).isoformat(timespec='seconds')}_",
        "",
        "| Ensemble | Family | Constraint | Fit | Sample | Filter |",
        "| Exactness / semantics |",
        "|---|---|---|---|---|---|---|",
    ]

    cells: set[tuple[Ensemble, ModelFamily, Constraint]] = set()
    for verb, ensemble, family, constraint in REGISTRY:
        if verb is not Verb.FIT and verb is not Verb.SAMPLE and verb is not Verb.FILTER:
            continue
        if REGISTRY[(verb, ensemble, family, constraint)].supported:
            cells.add((ensemble, family, constraint))
    ordered = sorted(cells, key=lambda t: _row_key(*t))

    for ensemble, family, constraint in ordered:
        row = [
            ensemble.value,
            family.value.upper(),
            constraint.value,
            _check(Verb.FIT, ensemble, family, constraint),
            _check(Verb.SAMPLE, ensemble, family, constraint),
            _check(Verb.FILTER, ensemble, family, constraint),
            _sample_semantics(ensemble, family, constraint),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def generate_microcanonical_routes() -> str:
    """Render the microcanonical route summary table for the user guide."""
    lines: list[str] = [
        GENERATED_MARKER,
        "",
        "# Microcanonical sampling routes",
        "",
        "> Generated from MENoBiS' capability registry (`menobis.capabilities`)."
        " The exactness and backend columns reflect the current source tree.",
        "",
        f"_Registry source SHA: `{git_short_sha()}` · generated:"
        f" {datetime.now(UTC).isoformat(timespec='seconds')}_",
        "",
        "| Constraint | Family | Exact / controlled | Backend |",
        "| Required arguments | Exactness |",
        "|---|---|---|---|---|---|",
    ]

    rows: list[
        tuple[
            tuple[int, ...], tuple[Ensemble, ModelFamily, Constraint, ModelCapability]
        ]
    ] = []
    for (verb, ensemble, family, constraint), cap in REGISTRY.items():
        if verb is not Verb.SAMPLE or ensemble is not Ensemble.MICROCANONICAL:
            continue
        if not cap.supported:
            continue
        key = _row_key(ensemble, family, constraint)
        rows.append((key, (ensemble, family, constraint, cap)))
    rows.sort(key=lambda item: item[0])

    for _, (_, family, constraint, cap) in rows:
        args = ", ".join(public_arg_names(cap)) or "\u2014"
        row = [
            constraint.value,
            family.value.upper(),
            MC_CONTROLLED[constraint],
            f"`{cap.backend}`",
            args,
            SAMPLE_SEMANTICS[(Ensemble.MICROCANONICAL, constraint)],
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_tables() -> None:
    """Write both generated Markdown files under ``docs/_generated/``."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    CAPABILITIES_FILE.write_text(generate_capabilities(), encoding="utf-8")
    ROUTES_FILE.write_text(generate_microcanonical_routes(), encoding="utf-8")


def _strip_metadata(text: str) -> str:
    """Remove timestamp/SHA metadata lines so --check ignores volatile text."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith("_Registry source SHA:")
    )


def check_tables() -> bool:
    """Return True when committed tables match freshly generated ones."""
    expected = {
        CAPABILITIES_FILE.name: _strip_metadata(generate_capabilities()),
        ROUTES_FILE.name: _strip_metadata(generate_microcanonical_routes()),
    }
    ok = True
    for name, rendered in expected.items():
        path = GENERATED_DIR / name
        if not path.exists():
            print(f"missing generated file: {path}")
            ok = False
            continue
        committed = _strip_metadata(path.read_text(encoding="utf-8"))
        if committed != rendered:
            print(f"drift detected in: {path}")
            print("run `uv run python scripts/docs/generate_capabilities.py` to update")
            ok = False
    return ok


def main() -> None:
    """CLI entry point: write tables by default, verify drift with --check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed generated tables match the registry; exit 1 on drift",
    )
    args = parser.parse_args()
    if args.check:
        if not check_tables():
            raise SystemExit(1)
        print("capability tables are up to date")
        return
    write_tables()
    print(f"wrote {CAPABILITIES_FILE}")
    print(f"wrote {ROUTES_FILE}")


if __name__ == "__main__":
    main()
