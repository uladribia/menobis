# ODME Modernization Plan

Scientific reference: <https://hdl.handle.net/10803/400560>.

## Completed milestones

| # | Milestone |
|---|-----------|
| 0 | Planning and conventions |
| 1 | Build scaffold |
| 2 | Core graph/data representation |
| 3 | Modern I/O |
| 4 | Analysis statistics |
| 5 | Fixed-strength fitting and generation |
| 6 | Remaining ME models (Cases 1–5 + cost + partial) |
| 7a | Distribution variants: geometric, binomial, negative binomial |
| 7b | Binomial family IPF fitting |
| 7b+ | Binomial sampling/filtering for all constraints (cost, edges, degree, degree-events) |
| 7b++ | Ensemble equivalence validation |
| 8 | Modern CLI (analyze, fit, generate, filter, convert) |
| 9 | Documentation site |
| 10 | Performance benchmarks with regression baselines |
| 11 | Statistical filtering (all models + absent edges) |
| — | Naming cleanup: `{operation}_{constraint}_{distribution}` |
| — | Provider unification: `WeightFamily` enum + `FixedStrengthProvider` |

Total: 157 Python tests, 34 Rust tests, all checks green.

## Remaining work

### Milestone 7c: Geometric / negative binomial fitting — NOT STARTED

The W-ensemble (geometric, negative binomial) fitting does NOT use IPF.
The legacy code uses bounded scipy optimization (TNC) with explicit
log-likelihood, gradient, and sparse Hessian.

Implementation plan:

1. Implement log-likelihood, gradient, and Hessian in Rust for the geometric
   family (`crates/odme-core/src/fitting.rs` or new file).
   Formulas from legacy `fitter_s.py` case `W`:
   - $\mathcal{L} = M \sum_{ij} \log(1 - x_i y_j) + \sum_i s^{out}_i \log x_i + \sum_j s^{in}_j \log y_j$
   - Domain constraint: $x_i y_j < 1$ for all pairs
2. Implement a bounded optimizer in Rust (L-BFGS-B or TNC).
   Options: `argmin` crate, or custom implementation.
3. The negative binomial case is geometric with $M > 1$ layers
   (same log-likelihood structure).
4. Add `fit_strength_geometric` and `fit_strength_neg_binomial` Python wrappers.
5. Add masked/partial variants.
6. TDD: convergence, domain-violation detection, comparison with legacy.

### Milestone 7d: Legacy mobility benchmarks — NOT STARTED

The modern ODME rewrite will **not** implement the thesis-era mobility models.
Keep them as legacy reference code only during modernization, and benchmark
them explicitly before removal so their behavior and performance are documented.
At the end of the process, preserve the legacy code on a separate branch rather
than in the main ODME rewrite branch.

| Legacy model | Required action | New implementation |
|--------------|-----------------|--------------------|
| Sequential gravity (Bernoulli + multinomial) | Benchmark legacy code | ❌ No |
| Radiation model (stochastic + multinomial) | Benchmark legacy code | ❌ No |

These are standalone mobility/null models, not ME distribution variants. Do not
add Rust/Python ODME APIs for them unless the project scope changes.

### Future: thesis-equation mapping

Create `docs/thesis-context.md` mapping thesis equations to ODME code paths.
Requires careful verification against the thesis PDF.

### Future: tutorials

1. `docs/tutorials/filter-workflow.md` — end-to-end walkthrough.
2. `docs/tutorials/partial-constraints.md` — partial-constraint workflow.

### Future: legacy code archiving and removal

Before deleting legacy directories from the main rewrite branch, create a
separate archival branch that stores the thesis-era code for reference.

**Phase 1 (safe now):** Remove `1. Network analysis/`. Fully replaced.

**Phase 2 (after Milestone 7c and legacy benchmark capture):** Remove
`2. Model Fitting/` and `3. Model Generation/`. Unported items:

| Legacy | What remains |
|--------|-------------|
| `fitter_s.py` cases `B`, `W` | W fitting (7c); B fitting done |
| `fitter_sk.py` `agg=True` | W fitting for strength-degree |
| `fitter_E.py` `agg=True` | W fitting for strength-edges |
| `ula_null_models.c` | geometric/binomial/NB samplers (done in 7a) |
| `others_null_models.c` | benchmark radiation and sequential gravity before removal; do not port |
