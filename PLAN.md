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

## Open problems (by priority)

### Critical: P5 — B strength-edges and B strength-degree use wrong kernel

**Impact:** Silent wrong results for any user of these functions.

`fit_strength_edges_binomial` calls `_odme.fit_strength_edges_poisson` and
`fit_strength_degree_binomial` calls `_odme.fit_strength_degree_poisson`.
No B-specific Rust kernel exists. The fit uses ME parameters but the B sampler
applies `Binomial(M, p/(1+p))` → nonsensical output.

**Fix:** Implement B-specific Rust kernels using bisection-over-lambda + B IPF,
mirroring the existing W implementations in `crates/odme-core/src/fitting/w.rs`.
These should include saturation peeling from day one. Until fixed, the benchmark
pipeline skips these cases.

---

### High: P1 — W/Wnb Newton solver fails at large N

**Symptom:** Coordinate-descent solver exhausts iterations:
- Strength-only: fails at N ≥ 500
- Strength-cost: fails at N ≥ 100 on high-heterogeneity inputs

**Note:** On the PA geographic benchmarks (moderate heterogeneity), W converges
at N=500 for all constraints. The failure manifests on Pareto-like heavy-tailed
inputs.

**Fix options (in order of promise):**
1. Anderson/SQUAREM acceleration on the fixed-point iteration
2. L-BFGS on the convex W dual
3. Block Newton with dense Jacobian (O(N³) per step, ~10 steps)

---

### Medium: P2 — B strength-cost coordinate fitting at N ≥ 50

**Symptom:** IPF oscillates when pairs have `x_i * y_j * exp(-γd)` near 1.

**Note:** On PA geographic benchmarks, B cost converges at N=500 (layers=10).
Failure requires high saturation ratios.

**Fix options:**
1. Anderson acceleration on the inner IPF
2. Log-space parameterization of B multipliers
3. Bisection-over-gamma (like the W implementation)

---

### Medium: P3 — Partial W/Wnb cost-coord never converges

**Symptom:** Excess problem is ill-conditioned after subtracting known pairs.

**Root cause:** Inherits P1 — the W solver cannot handle the resulting
poorly-conditioned residual.

**Fix:** Resolve P1 first; then add regularization for near-zero excess nodes.

---

### Low: strength-degree saturation peeling (future)

The coupled strength-degree model (ME/W/Wnb) does NOT support degree-saturated
nodes. When `k_i = capacity`, the solver may not converge because:
- Occupation must be 1 (z channel needs large multiplier)
- Per-pair weight still needs fitting (x channel must remain active)
- This requires a mixed-constraint solver not yet implemented

**Workaround:** Users should clip degrees to `capacity - 1` or use the
degree-events formulation instead.

---

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
| ME/W/Wnb strength-degree | No convergence when degree = capacity |
| ME/W/Wnb strength-edges | Rejects target_edges ≥ capacity (use strength-only) |
| B strength-edges / strength-degree | P5 bug: calls ME kernel (skipped in benchmarks) |
| W/Wnb strength at large N | P1: may not converge for heavy-tailed inputs at N ≥ 500 |

## Next steps (priority order)

1. **Fix P5** — Implement B strength-edges and B strength-degree Rust kernels
2. **Fix P1** — Anderson acceleration for W coordinate descent
3. **Fix P2** — B cost-coord Anderson or log-space IPF
4. Archive/remove legacy thesis-era folders (`1. Network analysis/`, etc.)
5. Final rename decision: ODME → MENoBiS
6. Publish MkDocs site
7. Write tutorials and notebooks with real-world examples
