---
description: v2.0 readiness audit for MENoBiS.
---

# v2.0 readiness audit

> TL;DR: MENoBiS covers the core scientific scope, has a unified routing API,
> coordinate-only cost handling, calibrated filtering, and a clean benchmark
> suite. Remaining work is internal quality: wrapper repetition, text I/O
> streaming, and constraint-level code factoring.

## Verdict

| Area | Status | Reason |
|---|---|---|
| Scientific formulas | Ready | ME/B/W kernels, zero-inflated formulas, and conditional p-values tested. |
| Extensibility | Ready | `fit_model(ensemble, family, constraint, ...)` routes with typed enums. |
| Cost handling | Ready | Coordinate-only public APIs; `EuclideanCostProvider` trait in Rust. |
| Benchmark suite | Ready | Single modern E2E script; PA geographic generate → fit → sample → filter-FPR. |
| Filtering calibration | Ready | Conditional p-values `P(X≥w|X>0)` give stable FPR ≈ 0.02 across N. |
| Code reuse | Acceptable | Python wrappers are repetitive but correct; PyO3 split by domain. |
| Memory ecology | Acceptable | Pair masks are sparse O(N+K); no dense N² allocations remain. |
| Solver architecture | Acceptable | Partial solvers reuse masked balance functions; full solve is not yet literally partial with empty known set. |
| Release packaging | Needs work | Publish metadata, release docs, and generated artifacts need final review. |

## Resolved findings

| ID | Finding | Resolution |
|---|---|---|
| V2-1 | Partial Python delegating to Rust | Done — coordinate partial ME/B/W use Rust kernels. |
| V2-2 | Dense `N×N` pair masks | Fixed — `PairMask` uses sparse `HashSet` with O(N+K) memory. |
| V2-3 | Degree-events PyO3 bindings expose primitives | Acceptable — internal; public API uses `DegreeEventsFit`. |
| V2-4 | No unified routing API | Fixed — `fit_model(...)`, `sample_model(...)`, `filter_model(...)` with enums. |
| V2-8 | Sparse/coordinate cost handling | Fixed — coordinate-only public APIs; sparse removed. |
| N-3 | Benchmark results committed | Fixed — results gitignored; old artifacts deleted. |

## Remaining findings (non-blocking for users)

| ID | Finding | Notes |
|---|---|---|
| V2-5 | Python fitting/filtering/generation wrappers repetitive | ~3300 lines; correct but could use factory pattern. Low priority. |
| V2-6 | Matrix Market/Pajek `read_text().splitlines()` | 2 occurrences in `data/io.py`. Only matters for multi-GB files. |
| V2-7 | PyO3 binding surface is large (~3100 lines total) | Already split by domain (fitting/generation/filter/stats). Acceptable. |
| V2-9 | Constraint-level abstraction missing | Constraints enumerated but code doesn't factor by linearity class. Architectural wish. |
| V2-10 | Partial/full solver not literally unified | Partial uses masked balance; full uses unmasked. Same math, separate entry points. |

## Known solver limitations (documented in AGENTS.md)

| Issue | Status |
|---|---|
| W strength-cost `self_loops=False` slow at N≥500 | Confirmed: ~500s at N=500. Newton needs adaptive damping. |
| W strength `self_loops=False` | Converges (22 iters at N=50) — issue resolved or overstated. |
| B `self_loops=False` at N≥200 | Resolved: converges in 22 iters at N=300. |
