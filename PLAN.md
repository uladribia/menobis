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
interior points. Do **not** port CVXOPT and do **not** write a custom solver.
Use `cvxrust` for convex modeling, then use Clarabel as the explicit conic
solver/backend if CVXRust cannot solve the model directly.

Current implementation facts:

- `odme-core::distribution::WeightFamily` already supports `Geometric` and
  `NegBinomial(M)` sampling with pair parameter `q_ij = x_i y_j`.
- `PairDistribution::expected()` implements `q/(1-q)` and `M q/(1-q)` and
  returns infinity if `q >= 1`.
- Rust and PyO3 expose samplers for geometric/NB strength models, but no fitter
  exists yet; Python package exports should be completed with the fitter.
- Existing ME/B fitters live in `crates/odme-core/src/fitting.rs`; W fitting
  should be added in a new module (for example `w_fitting.rs`) and re-exported.

Equations:

Let `P` be the allowed pair set after self-loop/mask filtering. For geometric
`M=1`; for negative binomial use integer layers `M > 1`. The legacy likelihood
uses

$$q_{ij}=x_i y_j,\quad 0 \le q_{ij}<1,\quad
\mathbb{E}[t_{ij}] = \frac{M q_{ij}}{1-q_{ij}}.$$

Up to constants, maximize

$$\ell(x,y)=M\sum_{(i,j)\in P}\log(1-x_i y_j)
+\sum_i s_i^{out}\log x_i+\sum_j s_j^{in}\log y_j.$$

Use inverse-fitness variables

$$x_i=e^{-a_i},\quad y_j=e^{-b_j},\quad r_{ij}=a_i+b_j>0.$$

Then minimize the convex objective

$$F(a,b)=\sum_i s_i^{out}a_i+\sum_j s_j^{in}b_j
-M\sum_{(i,j)\in P}\log(1-e^{-r_{ij}}).$$

The barrier term diverges as `r_ij -> 0+`; this is intrinsic, not a numerical
artifact. Stable residuals are

$$\hat{s}^{out}_i=M\sum_{j:(i,j)\in P}\frac{1}{\exp(r_{ij})-1},\quad
\hat{s}^{in}_j=M\sum_{i:(i,j)\in P}\frac{1}{\exp(r_{ij})-1}.$$

Stationarity is `s_out = ŝ_out` and `s_in = ŝ_in`. Curvature per pair is

$$h_{ij}=M\frac{\exp(r_{ij})}{(\exp(r_{ij})-1)^2}
=\mu_{ij}\left(1+\frac{\mu_{ij}}{M}\right),$$

where `mu_ij = M / expm1(r_ij)`.

Conic model for CVXRust/Clarabel:

Introduce variables `a_i`, `b_j`, and for every allowed pair `(i,j)` auxiliary
`t_ij,u_ij,v_ij`. Minimize the linear objective

$$\sum_i s_i^{out}a_i+\sum_j s_j^{in}b_j+M\sum_{(i,j)\in P}t_{ij}.$$

Represent `t_ij >= -log(1-exp(-r_ij))` as

$$\exp(-t_{ij})+\exp(-r_{ij}) \le 1,$$

using two exponential-cone epigraphs and one linear constraint:

- `(-t_ij, 1, u_ij) in K_exp`  implies `exp(-t_ij) <= u_ij`;
- `(-a_i-b_j, 1, v_ij) in K_exp` implies `exp(-r_ij) <= v_ij`;
- `u_ij + v_ij <= 1`.

Add one gauge constraint, for example `sum(a) - sum(b) = 0`, because
`a += c, b -= c` leaves all `r_ij` unchanged. After solving, recover
`x_i = exp(-a_i)` and `y_j = exp(-b_j)` with a gauge recentering that avoids
large individual multipliers while preserving products.

Implementation sequence:

1. Dependency spike: add `cvxrust` in Rust only on a feature branch; verify crate
   maturity, exponential-cone support, sparse assembly, solver status reporting,
   and whether Clarabel can be selected as backend.
2. If CVXRust lacks direct solve support, build the same conic problem and pass
   it to Clarabel explicitly. Do not change the model to fit a weaker solver.
3. Start with fixed-strength all-pairs W fitting only; masked and no-self-loop
   support are just pair-set changes once the all-pairs formulation works.
4. Implement scalar diagnostic kernels in Rust: `neg_ln_1m_exp_neg(r)`,
   `mean = M / expm1(r)`, and `curvature = mean * (1 + mean/M)` using small-`r`
   series and large-`r` branches. These are for validation/residual reporting,
   not a hand-written optimizer.
5. Validate inputs at the Python boundary: balanced non-negative strengths,
   positive layer count for NB, feasible support for nonzero strengths, and
   boundary warnings when solved `min(r_ij)` is near machine-safe limits.
6. Add core result type with `x`, `y`, `converged`, `iterations`, solver status,
   objective value, `min_margin = min(r_ij)`, `max_product`, and max/total
   strength residuals.
7. Add PyO3 functions `fit_strength_geometric` and
   `fit_strength_neg_binomial`; add Python dataclass/wrapper and export existing
   geometric/NB samplers from `odme.models` at the same time.
8. TDD order: scalar kernel tests; 1-node and homogeneous-network analytic
   cases; no-self-loop feasibility tests; CVXRust/Clarabel smoke solve; residual
   recovery tests; seeded comparison with legacy `fitter_s.py`; property tests
   for balanced strengths and `max(x_i y_j) < 1`.
9. Performance gate: benchmark lifted conic variable count and memory. The
   formulation uses O(|P|) cones and auxiliaries, so document practical N limits.
   Do not promise large all-pairs W fits until measured.
10. Defer W strength-edges and W strength-degree fitting until fixed-strength W
    is stable; legacy `fitter_E.py agg=True` and `fitter_sk.py agg=True` remain
    references, not direct ports.

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
