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

Equation review: the legacy W log-likelihood maximizes
$M\sum_{ij}\log(1-x_i y_j)+\sum_i s^{out}_i\log x_i+
\sum_j s^{in}_j\log y_j$. With $x_i=\exp(u_i)$ and $y_j=\exp(v_j)$, define
$z_{ij}=u_i+v_j<0$ and minimize:

$$F(u,v)=-M\sum_{ij}\log(1-\exp(z_{ij}))
-\sum_i s^{out}_i u_i-\sum_j s^{in}_j v_j.$$

This is convex because $\phi(z)=-\log(1-e^z)$ has
$\phi''(z)=e^z/(1-e^z)^2>0$ for $z<0$. The stationarity equations recover the
strength constraints with expected weight $M e^{z_{ij}}/(1-e^{z_{ij}})$.
The log transform fixes the non-convex raw domain, but not ill-conditioning
when many $z_{ij}$ are close to zero.

Implementation plan:

1. Implement matrix-free objective, gradient, Hessian-vector product, and
   residuals from the equations above.
2. Use damped Newton-CG first; keep L-BFGS with feasible step control as a
   fallback if Newton-CG is too costly.
3. Enforce feasibility by computing the maximum step such that all allowed
   $u_i+v_j \le -\epsilon$; then use Armijo backtracking.
4. Remove the scale nullspace (`u += c`, `v -= c`) by recentering after each
   step or fixing one gauge variable.
5. Initialize from fixed-strength ME multipliers, scaled into the interior;
   fall back to uniform feasible starts.
6. Support `self_loops=False` and masks by skipping forbidden pairs.
7. Expose `fit_strength_geometric` and `fit_strength_neg_binomial(layers=M)`
   with convergence, iterations, max product, and strength residuals.
8. Defer W strength-edges and strength-degree fitting until fixed-strength W is
   stable; legacy `fitter_E.py agg=True` and `fitter_sk.py agg=True` remain
   references, not direct ports.
9. TDD: tiny hand cases, domain-boundary rejection, finite-difference gradient,
   Hessian-vector checks, legacy comparison, and property tests.

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
