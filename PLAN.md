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

The W-ensemble fitter replaces legacy SciPy TNC preconditioning plus CVXOPT
interior points. Do **not** port CVXOPT, and do **not** assume that a log
transform makes the loss numerically benign.

Equation review: the legacy W likelihood uses $q_{ij}=x_i y_j<1$ and expected
weight $M q_{ij}/(1-q_{ij})$. Prefer inverse-fitness variables
$x_i=\exp(-a_i)$ and $y_j=\exp(-b_j)$, with $r_{ij}=a_i+b_j>0$:

$$F(a,b)=\sum_i s^{out}_i a_i+\sum_j s^{in}_j b_j
-M\sum_{ij}\log(1-\exp(-r_{ij})).$$

This form is convex, but the barrier term still diverges as $r_{ij}\to0^+$.
That divergence is intrinsic to the W model, not solved by reparameterization.
Use the objective only with stable kernels and residual diagnostics.

Implementation plan:

1. Implement scalar kernels for Bose-Einstein terms: `neg_ln_1m_exp_neg(r)`,
   `mean=M/expm1(r)`, and curvature `mean*(1+mean/M)`, with small-`r` series.
2. Implement matrix-free residuals and Hessian-vector products from these
   kernels; avoid naive `log(1-exp(...))` and `1/(exp(...)-1)` calls.
3. Solve the stationarity equations with safeguarded Newton-CG using residual
   norm as the primary convergence signal and stable objective as a merit check.
4. Enforce feasibility with maximum-step control so all allowed $r_{ij}$ stay
   above a machine-safe margin; classify smaller required margins as boundary.
5. Remove the scale nullspace (`a += c`, `b -= c`) by recentering or fixing one
   gauge variable.
6. Initialize from fixed-strength ME multipliers translated to feasible
   inverse-fitness variables; fall back to uniform interior starts.
7. Support `self_loops=False` and masks by skipping forbidden pairs.
8. Expose `fit_strength_geometric` and `fit_strength_neg_binomial(layers=M)`
   with convergence, iterations, minimum margin, and strength residuals.
9. Defer W strength-edges and strength-degree fitting until fixed-strength W is
   stable; legacy `fitter_E.py agg=True` and `fitter_sk.py agg=True` remain
   references, not direct ports.
10. TDD: scalar kernels, tiny hand cases, boundary rejection, finite-difference
   gradient, Hessian-vector checks, legacy comparison, and property tests.

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
