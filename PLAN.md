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

### 1. Fix E2E test script sampler calls

The true E2E benchmark (`/tmp/bench_e2e_true.py`) has wrong sampler API calls:
- `sample_strength_poisson(fit, seed=)` needs `total_events` argument
- `sample_degree_events_poisson(fit, seed=)` needs explicit parameters
- Samplers return `EdgeTable`, not `(src, tgt, wt)` tuples

Fix these to complete the generate → fit → sample → check loop.

### 2. Diagnose W Newton `self_loops=False` convergence failure

**Symptom:** W strength-cost coordinate fitting fails to converge at N≥50
when `self_loops=False` with realistic gravity-model inputs.

**Hypothesis:** The feasibility projection `a_i >= r_min - min_j(b_j + γ·d_ij)`
is too conservative when diagonal pairs are excluded but nearby nodes have
small distances, causing the solver to hit the projection boundary repeatedly.

**Investigation plan:**
- Compare the `self_loops=True` and `self_loops=False` paths at N=50
- Log the number of projection activations per iteration
- Test with increased `r_min` (currently 0.01) or adaptive damping
- Compare Newton residual trajectory for both cases

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
