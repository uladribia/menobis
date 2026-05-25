# ODME modernization plan

Scientific reference: <https://hdl.handle.net/10803/400560>.

## Completed

- [x] Rust + Python scaffold with PyO3/maturin
- [x] Edge-list data model, I/O, analysis
- [x] ME/W fitting for main constraint types; B strength and strength-cost fitting
- [x] Generation and filtering providers for ME/B/W
- [x] Partial known-weight fitting scaffold; full Rust/family conformance pending
- [x] Unified `coord_distance`, bisection gamma, O(K) sparse IPF
- [x] Removed Clarabel/cvxrust — all W uses Newton solver
- [x] MkDocs documentation site
- [x] E2E pipeline tests
- [x] B feasibility validation (`max_s <= M*(N-1)`)
- [x] W Newton solver rewrite (bisection + adaptive damping)
- [x] Degree-events boundary fix (clip to `n-2`)
- [x] W diagnostics bug fix (probability → log-space conversion)
- [x] Canonical PA geographic generator (`odme.synthetic`)
- [x] Benchmark CLI reorganized: generate → fit → sample → null-filter
- [x] Self-loop variant for PA generator and benchmark CLI
- [x] Saturation peeling for B strength and all degree-events families (P6)
- [x] B strength-edges and strength-degree use family-specific Rust kernels (P5)
- [x] Re-checked W/Wnb large-N PA and Pareto workflows at N=500; P1 not reproducible with current Newton solver
- [x] Re-checked B strength-cost at N=500; P2 not reproducible on current benchmark pipeline
- [x] Re-checked partial W/Wnb coordinate cost tests; P3 not reproducible after W solver rewrite
- [x] Mixed strength-degree degree-saturation handling for ME/W/Wnb
- [x] `uv run ty check`
- [x] Archived/removed legacy thesis-era folders after coverage audit

## Open problems (by priority)

### P1 — B fixed-strength no-self-loop solver is slower than legacy at N=500

`uv run python -m benchmarks.legacy_fit_compare --nodes 100,500 --families me,b`
shows B fixed-strength no-self-loop expectations match the archived fitter, but
modern inner solver time is slower at N=500. Investigate sparse IPF update costs
and stopping criteria.

### Informational: P4 — W single-sample errors are stochastic, not bugs

W/Wnb geometric variance = q/(1−q)² diverges as q→1. Single-sample strength
errors of 100–5000 are expected. Ensemble z-scores confirm correct mean
recovery. The benchmark pipeline uses ensemble checks (5 samples).

## Infrastructure status

### Benchmark CLI

```bash
uv run python -m benchmarks all --nodes 100,500
uv run python -m benchmarks all --nodes 100 --self-loops
uv run odme-bench fit --nodes 500 --families me,w
```

Pipeline: PA geographic generate → fit → ensemble sample-check → null-filter FPR.

### Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
mkdocs build --strict
```

### Known solver limitations (documented)

| Model | Limitation |
|---|---|
| ME/W/Wnb strength-degree | Degree-saturated nodes use fixed high occupation multipliers while strength multipliers remain active |
| ME/W/Wnb strength-edges | Rejects target_edges ≥ capacity (use strength-only) |
| B strength-edges / strength-degree | Uses B-specific Rust zero-inflated binomial kernels; benchmarked at N=500 |
| W/Wnb strength at large N | Re-checked at N=500 on PA and Pareto-like inputs; current Newton solver converges |

## Next steps (priority order)

1. Final rename decision: ODME → MENoBiS
2. Publish MkDocs site
3. Write tutorials and notebooks with real-world examples
