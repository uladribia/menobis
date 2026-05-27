---
description: v2.0 readiness audit for MENoBiS.
---

# v2.0 readiness audit

> TL;DR: MENoBiS is scientifically well covered and passes current checks, but
> is **not ready for v2.0** until dense partial/normal solver unification,
> constraint-level abstraction, and wrapper/binding reuse are cleaned up.

## Verdict

| Area | Status | Reason |
|---|---|---|
| Scientific formulas | Ready | ME/B/W kernels and zero-inflated formulas have tests; degree-events fit/generate/filter APIs now use fitted positive-weight parameters. |
| Extensibility | Ready | `fit_model(ensemble, family, constraint, ...)` and `sample_model(...)` route with typed enums; unsupported cases raise `UnsupportedModelCaseError`. |
| Code reuse | Needs work | PyO3 exports, Python fitting wrappers, and filter wrappers repeat family/constraint plumbing. Constraint-level abstraction is missing. |
| Memory ecology | Needs work | Core generation/filtering stream pairs, but Rust partial masks still allocate `N x N`. |
| Solver architecture | Needs work | Partial and normal solver logic are separate code paths; should be unified via free-pair providers. |
| Constraint abstraction | Needs work | Sampling/fitting code does not yet factor constraints by linearity class. |
| Release packaging | Needs work | Rename is complete, but publish metadata, release docs, and generated artifacts need final review. |

## Blocking findings

| ID | Finding | Required fix |
|---|---|---|
| V2-1 | `src/menobis/models/partial.py` now delegates coordinate partial ME/B/W work to Rust. | Keep Python wrappers thin and prevent dense Python `N x N` regressions. |
| V2-2 | `crates/menobis-core/src/fitting/partial.rs` uses dense pair masks (`Vec<bool>` of `N*N`) and duplicates masked fitting logic. | Add reusable sparse free-pair provider traits and let partial solvers call full solvers. |
| V2-3 | Low-level degree-events PyO3 bindings still expose primitive arrays/rates. | Keep them internal and prefer `DegreeEventsFit` in public Python APIs. |
| V2-4 | No unified Python routing API for ensemble + family + constraint. | **Fixed:** `fit_model(...)` and `sample_model(...)` added with enums and custom `UnsupportedModelCaseError`. |
| V2-5 | Python wrappers for fitting/filtering/generation are large and repetitive. | Introduce wrapper factories or registries shared by public APIs and CLI. |
| V2-6 | Text formats for Matrix Market/Pajek use `read_text().splitlines()`. | Switch to streaming line iteration for large inputs. |
| V2-7 | `crates/menobis-python/src/lib.rs` is a large manual PyO3 surface. | Split bindings by domain and add a binding registry macro/helper. |
| V2-8 | Coordinate cost handling needs cleanup. | Keep cost-constrained APIs coordinate-based and ensure consistent edge-list handling. |
| V2-9 | Constraint-level abstraction is missing. | Factor code by constraint linearity: (a) linear on occupation numbers (strengths, strengths+cost, custom t_ij), (b) linear on binary occupation only (edges+total events), (c) linear on both (degrees+strengths, strengths+edges). Poisson/multinomial sampling is valid for any constraint linear on occupation numbers — code should reflect this. |
| V2-10 | Partial and normal solver logic are separate code paths. | Unify partial and full solvers: a full solve is just a partial solve with an empty known-pair set. |

## Non-blocking findings

| ID | Finding | Follow-up |
|---|---|---|
| N-1 | `src/menobis/analysis/ensemble.py` intentionally loops in Python over samples. | Keep as orchestration only; document that heavy statistics stay Rust-backed. |
| N-2 | CLI commands compute small derived vectors in Python. | Accept short-term, but prefer Rust helpers where input size scales with edges. |
| N-3 | Benchmark result fixtures are committed. | Keep only curated baselines; archive bulky generated outputs before release. |

## Immediate red/green order

1. Unify partial and full solver paths: full solve = partial with empty known set.
2. Factor constraint-level abstraction (linear on occupation, binary, or both).
3. Replace dense Rust partial masks with sparse free-pair providers.
4. Split or registry-factor the large PyO3 binding surface.
5. Clean sparse matrix/cost handling.
