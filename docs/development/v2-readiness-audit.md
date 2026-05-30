---
description: v2.0 readiness audit for MENoBiS.
---

# v2.0 readiness audit

> TL;DR: MENoBiS covers the core scientific scope, has a unified routing API,
> coordinate-only cost handling, calibrated filtering, a clean benchmark suite,
> and a well-characterised optimal benchmark regime (dense). ME and B converge
> at 100% across all constraints at N ≤ 1000. W remains broken for
> zero-inflated constraints. Outstanding closing steps: publish GitHub Pages
> site, create a use-case notebook, and update tests to use the dense regime
> by default.

## Verdict

| Area | Status | Reason |
|---|---|---|
| Scientific formulas | Ready | ME/B/W kernels, zero-inflated formulas, and conditional p-values tested. |
| Extensibility | Ready | `fit_model(ensemble, family, constraint, ...)` routes with typed enums. |
| Cost handling | Ready | Coordinate-only public APIs; `EuclideanCostProvider` trait in Rust. |
| Benchmark suite | Ready | Single modern E2E script; PA geographic generate → fit → sample → filter-FPR. Three regimes (sparse, dense, saturated). |
| Filtering calibration | Ready | Conditional p-values `P(X≥w|X>0)` give stable FPR ≈ 0.02 across N. |
| Code reuse | Acceptable | Python wrappers are repetitive but correct; PyO3 split by domain. |
| Memory ecology | Acceptable | Pair masks are sparse O(N+K); no dense N² allocations remain. ~215 MB RSS peak at N=1000 with 2% known pairs. |
| Solver architecture | Acceptable | Partial solvers reuse masked balance functions; full solve is not yet literally partial with empty known set. |
| Dense regime characterisation | Ready | Documented as optimal benchmark regime: `average_degree = N/5, events_per_edge = 8.0`. ME + B converge at 100% at N ≤ 1000 across all constraint types. |
| Release packaging | Needs work | Publish metadata, release docs, generated artifacts, and **GitHub Pages site** need final review. |

## Remaining findings (non-blocking for users)

| ID | Finding | Notes |
|---|---|---|
| V2-5 | Python fitting/filtering/generation wrappers repetitive | ~4600 lines (routing.py + models/*.py); correct but could use factory pattern. Low priority. |
| V2-6 | Matrix Market/Pajek `read_text().splitlines()` | 2 occurrences in `data/io.py`. Only matters for multi-GB files. |
| V2-7 | PyO3 binding surface is large (~3400 lines total) | Already split by domain (fitting/generation/filter/stats). Acceptable. |
| V2-9 | Constraint-level abstraction missing | Constraints enumerated but code doesn't factor by linearity class. Architectural wish. |
| V2-10 | Partial/full solver not literally unified | Partial uses masked balance; full uses unmasked. Same math, separate entry points. |

## Closing steps (not started)

| ID | Step | Priority |
|---|---|---|
| V2-13 | Publish GitHub Pages site | High — configure `mkdocs.yml`, add GitHub Actions workflow, deploy to `gh-pages`. |
| V2-14 | Create use-case notebook | High — a single `docs/examples/main-use-cases.ipynb` covering (a) null-model filtering and (b) ensemble statistics over sampled networks. |
| V2-15 | Update all E2E tests to dense regime by default | High — sparse and saturated are pathological; dense is the reliable benchmark. |

## Known solver limitations

| Issue | Status |
|---|---|
| W strength-cost `self_loops=False` slow at N≥500 | Confirmed: ~500s at N=500. Newton needs adaptive damping. |
| W strength-cost `self_loops=False` at N=1000 | Confirmed: ~137s, diverges for partial fits. |
| W strength-edges / strength-degree | Never converges at any N or regime. ZI W formula unstable. |
| B strength-cost saturated N≥1000 | IPF plateau at 10k iters with residual ~10⁻¹⁰. |
| B strength-degree sparse N=5000 | Overflow in ZI multipliers (z ~ 1e+262). |