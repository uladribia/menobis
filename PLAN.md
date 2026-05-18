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
| — | Final project rename from ODME to MENoBiS |

Total: 157 Python tests, 34 Rust tests, all checks green.

## Remaining work

### Milestone 7c: Geometric / negative binomial fitting — NOT STARTED

The W-ensemble (geometric, negative binomial) fitter replaces legacy SciPy TNC
preconditioning plus CVXOPT interior points. Do **not** port CVXOPT. Implement a
Rust-native convex optimizer after reparameterizing $x_i=\exp(u_i)$ and
$y_j=\exp(v_j)$.

Key observation: in raw $(x,y)$ variables the domain $x_i y_j < 1$ is awkward.
In log variables it becomes linear: $u_i+v_j<0$. The negative log-likelihood is
convex on this domain:

$$
F(u,v) = -M \sum_{ij}\log(1-\exp(u_i+v_j))
         - \sum_i s^{out}_i u_i - \sum_j s^{in}_j v_j .
$$

Implementation plan:

1. Create a Rust module for W-family fitting with matrix-free objective,
   gradient, Hessian-vector product, and constraint residuals.
2. Use a feasible-interior optimizer: damped Newton-CG with backtracking line
   search, or L-BFGS with an explicit maximum feasible step to keep
   `max(u_i+v_j) <= -margin`.
3. Remove the scale nullspace by recentering after each step
   (`u += c`, `v -= c`) or by fixing one gauge variable.
4. Initialize from fixed-strength ME multipliers, globally scaled so all
   products are safely below one; fall back to uniform feasible starts.
5. Support `self_loops=False` and masked/partial variants by skipping forbidden
   pairs in objective, gradient, and Hessian-vector products.
6. Expose `fit_strength_geometric` (`M=1`) and
   `fit_strength_neg_binomial(layers=M)` in Python, returning convergence,
   iterations, final max product, and max strength residuals.
7. Defer W-family strength-edges and strength-degree fitting until the fixed
   strength W fitter is stable; legacy `fitter_E.py agg=True` and
   `fitter_sk.py agg=True` remain references, not direct ports.
8. TDD order: hand-solved tiny networks, domain-boundary rejection,
   finite-difference gradients, Hessian-vector checks, seeded legacy comparison,
   and property tests for recovered strengths within documented tolerances.

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

### Final milestone: project rename to MENoBiS — NOT STARTED

After the scientific refactor is complete, rename the project from ODME
(Origin-Destination Multi-Edge) to **MENoBiS**:
**Maximum Entropy Non-Binary Null Model Suite**.

The rename makes sense because the modern scope is broader and clearer than the
old ODME name: it covers maximum-entropy null models for non-binary weighted
networks, including multi-edge and binomial families. Treat this as the final
polish step, not an active refactor task, to avoid churn while APIs are still
settling.

Rename scope:

| Item | Target |
|------|--------|
| Python package | decide final import name near release |
| CLI | decide final command name near release |
| Docs/site title | MENoBiS |
| Repository/package metadata | MENoBiS |
| Historical references | keep ODME as legacy/thesis-era name |

### Future: thesis-equation mapping

Create `docs/thesis-context.md` mapping thesis equations to ODME/MENoBiS code
paths. Requires careful verification against the thesis PDF.

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
