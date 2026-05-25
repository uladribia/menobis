# Archive legacy thesis folders

## TL;DR

The thesis-era C/Python folders were removed from the working tree. ODME now
uses only the Rust + Python implementation under `crates/`, `src/`, `tests/`,
and `docs/`.

## Context

The old folders were useful as scientific reference material during the rewrite:

| Legacy folder | Replacement area |
|---|---|
| `1. Network analysis/` | `odme.analysis`, `odme.data`, Rust graph/stat kernels |
| `2. Model Fitting/` | `odme.models.fitting`, Rust fitting kernels |
| `3. Model Generation/` | `odme.models.generation`, Rust sampling providers |

Keeping them in the active tree created ambiguity about supported APIs,
dependencies, and behavior. The project does not provide backward compatibility
with the thesis command-line tools or module layout.

## Decision

Remove the legacy folders instead of moving them to another package namespace.
The git history remains the archive. New work must extend the modern ODME API,
CLI, tests, and documentation directly.

## Coverage check

A throwaway script compared legacy feature groups with modern public symbols.
Required groups were present for analysis, fitting, generation, filtering,
partial constraints, custom probability sampling, and clustering.

Known intentional non-goals remain:

| Legacy feature | Modern status |
|---|---|
| Radiation model | not implemented yet |
| Sequential gravity model | not implemented yet |
| Byte-compatible `.hist`/`.list` report files | replaced by typed APIs and CLI tables |

## Consequences

- No dead legacy source remains in the repository root.
- Legacy examples are not supported entry points.
- Future comparisons should use thesis equations and generated feasible
  fixtures, not golden compatibility with the removed tools.
