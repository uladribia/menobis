# ODME modernization plan

Scientific reference: <https://hdl.handle.net/10803/400560>.

## Done

- [x] Rust + Python scaffold with PyO3/maturin
- [x] Edge-list data model, I/O, analysis
- [x] ME/B/W fitting for all constraint types
- [x] Generation and filtering for ME/B/W
- [x] Partial known-weight fitting with family-specific formulas
- [x] Unified `coord_distance`, bisection gamma, O(K) sparse IPF
- [x] Removed Clarabel/cvxrust — all W uses Newton solver
- [x] MkDocs documentation site

## Testing and benchmarking policy (mandatory)

All tests and benchmarks MUST follow this pipeline:

```
1. Generate a realistic weighted directed network with coordinates
2. Derive constraints from the generated network (strengths, degrees, costs, edges)
3. Fit the model using derived constraints
4. Sample from the fitted model
5. Verify: sampled network recovers original constraints within tolerance
```

Tests that do NOT follow this pipeline are justified ONLY for:
- Unit tests of pure mathematical functions (e.g., `w_mean` formula check)
- API contract tests (e.g., return type shape, field existence)
- CLI smoke tests (e.g., exit code, output format)

Any fitting/generation test that uses arbitrary hand-picked constraint values
without verifying they come from a realistic network MUST be rewritten or removed.

## Immediate next steps

### 1. ~~Fix E2E test script sampler calls~~ ✅ Done

Fixed in branch `test/fix-e2e-benchmark`:
- Created `tests/test_odme_e2e_pipeline.py` with 7 passing E2E tests covering
  ME/B strength, ME strength-cost, ME strength-edges, ME strength-degree,
  ME degree-events, plus an ensemble z-score test.
- Rewrote `benchmarks/bench_e2e.py` with correct API calls for all 18 cases
  (4 families × 4 constraint types + 2 strength-only families).
- Fixed: `sample_strength_poisson(fit.x, fit.y, self_loops=..., seed=...)`
  (not `sample_strength_poisson(fit, seed=...)`)
- Fixed: `sample_degree_events_poisson(fit, total_events=..., seed=..., self_loops=...)`
- Fixed: all samplers return `EdgeTable` (access `.source`, `.target`, `.weight`)
- Made B `layers` adaptive: `max(10, ceil(max_s / (n-1)) + 1)` for feasibility.
- Added proper sampler dispatch for all families in the benchmark.

### 2. ~~Diagnose W Newton `self_loops=False` convergence failure~~ ✅ Fixed

**Root cause:** Joint Newton coordinate-descent on (a, b, γ) with fixed
damping=0.8 oscillated indefinitely for heterogeneous networks (max_s >> avg_s).
The gamma=0 initialization created a basin trap that the joint solver couldn't
escape.

**Fix applied** (branch `test/fix-e2e-benchmark`):
1. **Bisection over gamma** (like the ME solver) instead of joint Newton.
   At each gamma, solve (a,b) independently via Newton.
2. **Adaptive damping with backtracking**: start at 0.5, increase on progress,
   revert and halve on stall (3 stalls trigger revert).
3. **Better initialization**: per-node `a_i` from individual strength targets
   (not global average), and `r_min` reduced from 0.01 to 1e-4.
4. **ME gamma bootstrap**: use the fast ME coordinate solver to find the
   correct gamma bracket before W bisection.

**Results:**
- Before: Failed at N≥15 for ALL realistic inputs (0% convergence)
- After: Converges at N=15 (1099 iters), N=25 (4510), N=50 (9824)
- W strength-only (no cost): converges in 10-12 iterations at all sizes
- N=100 needs more iterations (24k) but gamma is correct (~0.028)
- Still slow compared to ME (~10x iterations); acceptable for now

### 3. Diagnose B fixed-strength slow convergence `self_loops=False`

**Symptom:** `fit_strength_binomial(self_loops=False)` takes 28s at N=200.

**Hypothesis:** The B IPF (`balance_strength_binomial`) uses residual-based
convergence which converges slowly for large heterogeneous networks without
self-loops. The no-self-loop case removes the diagonal regularization.

**Investigation plan:**
- Profile the existing `balance_masked_strength_binomial` at N=200
- Check if the issue is iteration count (slow convergence) or per-iteration cost
- Compare with the B strength-cost IPF which converges fast

### 4. Diagnose degree-events N=500 infeasibility

**Symptom:** `fit_degree_events_*` rejects inputs at N=500 as infeasible.

**Hypothesis:** The degree sequences generated from a Poisson gravity model
have some nodes with `k_out` or `k_in` exceeding `n-1` (impossible without
self-loops), triggering the feasibility check.

**Investigation plan:**
- Print the rejected degree values
- Clip degrees to `n-1` before fitting
- Or generate degrees from the actual binary adjacency of the network

### 5. Rewrite existing tests to follow E2E pipeline

Audit all `tests/test_odme_*.py` files. For each:
- If it tests fitting + generation: rewrite to use the pipeline
- If it tests a pure function or API contract: keep as-is
- If it uses arbitrary infeasible inputs: remove or fix

Priority files to audit:
- `test_odme_fitting.py`
- `test_odme_generation.py`
- `test_odme_strength_cost.py`
- `test_odme_w_strength_cost_fitting.py`
- `test_odme_w_strength_fitting.py`
- `test_odme_benchmark_cases.py`

### 6. Rewrite benchmarks to follow E2E pipeline

The benchmark suite (`benchmarks/bench_fitting.py`) should:
- Generate networks with `generate_network(n, seed)`
- Derive ALL constraint types from the same network
- Fit each model
- Sample and verify

Remove any benchmark that uses synthetic constraint values not derived from
a generated network.

### 7. Further work (lower priority)

- Incremental benchmark saving
- Archive/remove legacy thesis-era folders
- Final rename decision: ODME → MENoBiS
- Publish MkDocs site

## Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
mkdocs build --strict
```
