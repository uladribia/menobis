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
| — | Publish MkDocs site to dedicated GitHub Pages |

Total: 202 Python tests, 46 Rust tests, all checks green.

## Remaining work

### Milestone 7c: Complete W ensemble — IN PROGRESS

**Done:**

- `ZeroInflatedGeometric` and `ZeroInflatedNegativeBinomial` pair distributions with full
  `expected()`, `occupation_probability()`, `lower_pvalue()`, `upper_pvalue()`,
  and `sample()` implementations.
- All providers (`StrengthEdgesProvider`, `StrengthDegreeProvider`,
  `DegreeEventsProvider`, `StrengthCostProvider`) produce exact zero-inflated W
  distributions for geometric/negative binomial families (no Poisson fallback).
- Sampling for all 5 W constraints × 2 families = 10 new Rust samplers +
  PyO3 bindings + Python wrappers.
- Rust filter functions for all W zero-inflated constraint types (8 filter + 4 absent)
  with PyO3 bindings.
- Degree-events W fitting (geometric + negative binomial) fully in Rust: Brent bisection
  for `q` + reuse of `balance_degree_bernoulli`.
- Unified `fit_strength_poisson` in Rust (analytic self-loops + IPF
  no-self-loops in one function). All Python fitting code is now a thin
  validation/logging shell over Rust.
- Partial-constraint fitting pipeline moved entirely to Rust (`partial.rs`).
  Python `partial.py` is now validation + wrapping only.
- Result types extracted into per-submodule `types.py` files:
  `analysis/types.py`, `models/types.py`, `filtering_types.py`.
- Removed overly strict `s == k` boundary check; all 194 tests pass.
- Added stable Rust scalar kernels for W fitting validation/residuals in
  `crates/odme-core/src/fitting/w.rs`.
- Split Rust fitting code into `fitting/me.rs`, `fitting/b.rs`, `fitting/w.rs`,
  `fitting/partial.rs`, `fitting/types.rs`, and shared internal
  `fitting/support.rs`; ME strength-cost fitting now lives in `fitting/me.rs`,
  and the old top-level `cost.rs` module was removed.
- Added W fixed-strength result diagnostics, lifted problem metrics, and
  independent-strength residual helpers in Rust.
- Added optional Rust `w-conic` feature with `cvxrust` and Clarabel dependencies;
  fixed-strength W fit entry points now solve the independent geometric and
  negative-binomial strength model via Clarabel exponential cones, with PyO3 and
  Python wrappers.
- Added W strength-cost geometric and negative-binomial conic fitting in Rust,
  PyO3 bindings, Python wrappers, typed diagnostics, tests, and Python/concept
  documentation updates.
- Recorded decision `docs/decisions/0004-w-conic-solver-boundary.md`: keep
  Clarabel/CVXRust scoped to W conic fits; keep ME/B fitting on balancing plus
  scalar root searches unless benchmark evidence justifies changing that boundary.
- Fit-result API cleanup started: W fixed-strength wrappers now return the
  shared `FitResult`, W strength-cost wrappers now return the shared
  `StrengthCostFit`, and both carry `family`, optional `layers`, shared
  `OptimizationDiagnostics`, and nested W-only `ConicDiagnostics`. Public W
  aliases were removed from `odme.models` exports; internal compatibility
  aliases remain only during migration.
- Partial-fit API cleanup is still needed. `PartialFitResult` should not remain
  an unrelated sparse-rate-only type: partial constraints are the same fitted
  model types with a support mask / known-rate mask applied. Final API should
  represent partial strength, strength-cost, strength-edges, degree, and
  strength-degree fits as the corresponding constraint-oriented fit result plus
  mask/support metadata and diagnostics, not as a disconnected result family.
- General fitting failure policy started and documented in
  `docs/decisions/0005-fitting-failure-policy.md`: shared Python-boundary
  validation for shape, finite values, non-negativity, balance, and cheap
  infeasibility; solver failures should return typed diagnostics/warnings rather
  than model-specific ad-hoc behavior. Added shared strength-edges capacity,
  strength-cost cost-entry, and degree-events capacity/layer validation across
  relevant Poisson/W wrappers.

**Remaining:**

- W zero-inflated absent-edge wrappers for strength-edges, strength-degree, and
  degree-events geometric/NB: Rust native functions do not exist yet; Python
  wrappers explicitly reject with clear error. Implement or document as
  intentionally unsupported.
- API consistency fixes documented in `docs/decisions/0006-api-consistency-audit.md`:
  add `family`/`layers`/`self_loops`/`diagnostics` to `DegreeEventsFit`; add
  `self_loops` to `FitResult`; fix `fit_strength_binomial` family; fix
  `fit_degree_bernoulli` family; populate diagnostics in
  `fit_strength_degree_poisson`. Pending review.
- Benchmark script `bench_w_fitting.py` for all 10 W fitting APIs.
- Documentation updates for complete W ensemble coverage.
- Partial-fit API unification (use constraint-oriented types with mask metadata).

**Session handoff (2026-05-20):**

- Current branch: `refactor/w-ensemble`.
- Current red/green state: W strength-cost geometric recovery and negative-
  binomial `layers > 1` validation are green.
- Checks run: `uv run maturin develop -m crates/odme-python/Cargo.toml`, full
  `UV_EXCLUDE_NEWER=2026-05-08T22:43:59Z uv run pytest` (208 passed),
  `cargo test --workspace --features w-conic`, `cargo clippy --workspace
  --features w-conic --all-targets -- -D warnings`, `cargo fmt --all --
  --check`, and `UV_EXCLUDE_NEWER=2026-05-08T22:43:59Z uv run mkdocs build
  --strict`.
- Checks still recommended before commit: full `uv run ruff format --check .`,
  full `uv run ruff check .`, and `uv run ty check`.
- Latest API cleanup checks: added `tests/test_odme_fit_result_api.py`; focused
  pytest for unified fit result shape is green; `uv run ty check src/odme/models
  tests/test_odme_fit_result_api.py` is green.
- W strength-edges update: added Rust/PyO3/Python entry points plus homogeneous
  and non-homogeneous geometric recovery tests. The implementation is currently
  an exact monotone root/IPF routine over `lambda`, not a Clarabel conic model.
- Added harder W workflow tests on a Pareto-like `N=10` sequence with
  non-saturating expected binary support. The tests fit and simulate geometric
  and negative-binomial variants for fixed strengths, strength-cost,
  degree-events, and strength-edges, then check ensemble alignment of strengths,
  costs, degrees/events, and edge counts with statistical tolerances. Observed
  max z-scores were comfortably below the test limits: fixed-strength <= 1.68,
  strength-cost <= 1.68 with cost z <= 0.32, degree-events degree z <= 2.75
  with total-event z <= 0.90, and strength-edges strength z <= 3.29 with edge z
  <= 1.30.
- Ran an out-of-test `N=100` Pareto strength-edges stress check with total
  strength 600 and target edges 180 (density 0.018). Geometric solved in ~38.9s
  with max strength residual ~3e-9 and edge residual ~2.2e-9; negative binomial
  (`M=3`) solved in ~33.0s with max strength residual ~5.1e-10 and edge
  residual ~9e-9. Accuracy is strong, but runtime is high.
- Solver recommendation recorded for handoff: keep the W strength-edges
  root/IPF solver provisionally as a validated reference for the
  strengths+total-edges constraint only. Do not use this method for
  fixed-strength-only W fits. Optimize and benchmark it before deciding whether
  to replace it with the planned Clarabel `eta = log(lambda)` conic model.
- Risk note: the strength-edges coordinate-bisection routine is not yet proven
  globally convergent. It passed current Pareto `N=10` workflow tests, an
  out-of-test `N=100` stress check, and explicit convergence/failure-mode tests
  (near-capacity, very sparse, heterogeneous no-self-loops, NB high layers,
  forced non-convergence warning). Treat it as experimental until property tests,
  benchmark sweeps, and either a proof or a conic fallback are available.
- W **strength-degree fitting** — Rust/PyO3/Python public wrappers with
  monotone coordinate solver (same approach as strength-edges). Homogeneous
  recovery test passes. Same experimental/risk status as strength-edges.
- Benchmark results (Pareto strengths, self-loops, `tolerance=1e-7`,
  `max_iterations=500`):

  | N | edges_geo | edges_nb3 | degree_geo | degree_nb3 |
  |---:|---:|---:|---:|---:|
  | 25 | 0.62s | 0.56s | 0.20s | 0.42s |
  | 50 | 2.01s | 2.02s | 0.78s | 1.67s |
  | 100 | 9.71s | 8.78s | 3.31s | 6.42s |
  | 200 | 41.9s | 38.9s | 13.7s | 24.4s |
  | 500 | 281.9s | 276.6s | 91.2s | 137.4s |

  All converged with max strength residual ≤ 1e-5 (degree) or ≤ 1e-9 (edges).
- Clarabel conic comparison: current implementation uses dense matrix assembly
  and OOM-kills at N≥50 (~8.5 GB for N=50, ~17 GB for N=100). The dense builder
  must be rewritten to sparse CSC before Clarabel can scale beyond N≈25.
  Monotone coordinate solver is currently the only viable path for N≥50.
- Sparse Clarabel CSC assembly implemented for both W fixed-strength and W
  strength-cost fitting. Eliminates OOM: N=100 in ~5s for both (was OOM at
  N≥50). The `dense_to_csc` helper has been removed. Updated scaling limits.
- Assessment on Clarabel vs monotone coordinate for edges/degree constraints:
  Clarabel is not applicable because `log(1 + v*G_M(r))` terms in the
  zero-inflated objective do not reduce to simple exponential cones. The
  monotone coordinate solver is the correct and only practical approach for
  these constraints. Documented and closed.

**Session handoff (2026-05-20, session 2):**

- Current branch: `refactor/w-ensemble`.
- Full test suite: 236 passed (after fixing `WStrengthFit` import in CLI).
- Checks run: `cargo clippy --workspace --features w-conic --all-targets --
  -D warnings`, `cargo fmt --all`, `UV_EXCLUDE_NEWER=2026-05-08T22:43:59Z
  uv run pytest -q` (236 passed), `UV_EXCLUDE_NEWER=2026-05-08T22:43:59Z
  uv run mkdocs build --strict`.
- Checks not run: `uv run ty check` (full project), full `uv run ruff check .`.
- Next recommended steps:
  1. Add Pareto workflow tests for W strength-degree (fit → simulate → check
     strengths and degrees).
  2. Add `.pyi` stubs for new PyO3 W strength-degree functions.
  3. Run full `uv run ruff check .` and `uv run ty check` before commit.
  4. Commit all session work with `/skill:commit`.
  5. Consider release-mode benchmarks for the monotone coordinate solver to
     get realistic production timings (debug builds are 3–5x slower).
- All 5 W constraints now have fitting entry points. Remaining work: Pareto
  workflow tests for strength-degree, documentation updates, sparse Clarabel
  rewrite for fixed-strength/cost scaling, and benchmark coverage.

**Known technical debt:**

- None blocking.

The W ensemble covers geometric (`M = 1`) and negative binomial (`M > 1`)
weighted null models for every W-capable constraint in this package. The final
scope is **sampling + filtering + fitting** for fixed strengths, fixed degrees
plus total events, fixed strengths plus total edges, fixed degrees and
strengths, and fixed strengths plus cost.

Do **not** port CVXOPT and do **not** write a custom nonlinear optimizer. Use
`cvxrust` for convex modeling, then use Clarabel explicitly if CVXRust cannot
solve the conic model directly. Legacy SciPy/CVXOPT code is reference material
only.

#### Public W scope

| Constraint | Geometric API | Negative-binomial API | Sampling/filtering work |
|------------|---------------|-----------------------|-------------------------|
| strengths | `fit_strength_geometric` | `fit_strength_negative_binomial` | Audit existing `sample_*/filter_*` exports. |
| degrees + total events | `fit_degree_events_geometric` | `fit_degree_events_negative_binomial` | Add exact zero-inflated W samplers/filters if missing. |
| strengths + total edges | `fit_strength_edges_geometric` | `fit_strength_edges_negative_binomial` | Add exact zero-inflated W samplers/filters if missing. |
| strengths + degrees | `fit_strength_degree_geometric` | `fit_strength_degree_negative_binomial` | Add exact zero-inflated W samplers/filters if missing. |
| strengths + cost | `fit_strength_cost_geometric` | `fit_strength_cost_negative_binomial` | Add cost-modulated W samplers/filters if missing. |

There is no standalone W fixed-degree model without a total-event constraint:
degrees determine occupation probabilities, while `T` determines the positive
weight distribution.

Current implementation facts to verify before fitting:

- `WeightFamily::{Geometric, negative binomial(M)}` and independent fixed-strength
  samplers/filters already exist.
- `StrengthCostProvider` is family-parameterized, but Python/PyO3 W cost
  sampler/filter wrappers may still be missing.
- `WeightFamily::zip_distribution`, `StrengthEdgesProvider`,
  `StrengthDegreeProvider`, and `DegreeEventsProvider` currently need an audit:
  they must produce exact zero-inflated geometric/negative binomial distributions, not fall
  back to zero-inflated Poisson.
- Existing ME/B fitters live in `crates/odme-core/src/fitting/mod.rs`; add W fitting
  in `crates/odme-core/src/fitting/w.rs` and re-export it through the fitting module.

Validate feasibility at the Python boundary before invoking the solver:

- strengths are finite, non-negative, length-matched, and balanced;
- degrees are finite, non-negative, length-matched, balanced, and within support
  capacity (`N` or `N-1` depending on self-loops);
- strength-degree inputs satisfy `s_out >= k_out`, `s_in >= k_in`, and
  `sum(s_out) >= sum(k_out)`;
- strength-edges inputs satisfy `0 <= E <= |P|` and `E <= T`;
- degree-events inputs satisfy `T >= E = sum(k_out) = sum(k_in)`;
- cost inputs are finite, non-negative, aligned sparse triples with finite target
  `C`, and their missing-pair semantics match generation/filtering;
- negative binomial `layers` is a positive integer, and public negative binomial APIs should reject `M = 1`
  if geometric is the intended spelling.

#### Common W equations

Let `P` be the allowed pair set after self-loop and mask filtering. Use `M = 1`
for geometric and integer `M > 1` for negative binomial. Define

$$q_{ij}=e^{-r_{ij}},\quad 0 \le q_{ij}<1,$$

$$A_M(r)=(1-e^{-r})^{-M},\quad G_M(r)=A_M(r)-1.$$

Independent W pairs have partition `A_M(r)` and expectation

$$\mu_{ij}=\frac{M}{\exp(r_{ij})-1}.$$

Zero-inflated W pairs with occupation multiplier `v_ij > 0` have partition

$$Z_{ij}=1+v_{ij}G_M(r_{ij}),$$

occupation probability and expected weight

$$\pi_{ij}=\frac{v_{ij}G_M(r_{ij})}{Z_{ij}},\quad
\mu_{ij}=\frac{v_{ij}M e^{-r_{ij}}(1-e^{-r_{ij}})^{-M-1}}{Z_{ij}}.$$

The conditional positive-weight mean used by degree-events models is

$$m_+(q,M)=\frac{M q}{(1-q)(1-(1-q)^M)}.$$

#### Why naïve balancing fails for most W constraints

IPF/balancing converges for ME and B because expected weights are linear or
monotone-ratio functions of multipliers with unbounded domains. For W, the
independent fixed-strength expected weight `M xy/(1-xy)` has a pole at `xy = 1`,
and the naïve balancing update `x_i = s_i / sum(...)` is implicit (`x_i` appears
inside the denominator). Naïve iteration either diverges or requires aggressive
clamping that destroys convergence guarantees. The legacy code confirms this:
`fitter_s.py` case W uses TNC+CVXOPT rather than `balance_xy`. Therefore the
independent fixed-strength W case stays on the conic solver.

A restricted exception is currently being evaluated for W strengths + total
edges. With fixed `lambda`, each row/column strength equation is monotone in a
single coordinate if the opposite side and `lambda` are held fixed. The current
strength-edges implementation solves each coordinate by scalar bisection and
uses an outer bracketed solve for `lambda` to recover total expected edges. This
is **not** the naïve multiplicative IPF update and is **not** used for
fixed-strength-only W fitting. It is provisional until optimized, benchmarked,
and compared conceptually against the planned conic `eta = log(lambda)` model.
Because alternating coordinate bisection lacks a documented global convergence
proof here, future work must either prove/characterize convergence, add strong
failure detection plus conic fallback, or replace this solver with the conic
formulation.

The degrees+events case remains another exception because its `q` is a single
global scalar solved by root-finding, and the remaining occupation IPF has no
W strength barrier.

#### Convex fitting objectives

Use inverse/log variables and add gauge constraints for non-identifiable node
multipliers. Always recover finite `x`, `y`, `z`, `w`, and scalar multipliers by
recentering gauges after solving.

1. **Strengths:** `r_ij = a_i + b_j`, `x_i=e^{-a_i}`, `y_j=e^{-b_j}`.

   $$F=\sum_i s_i^{out}a_i+\sum_j s_j^{in}b_j
   -M\sum_{(i,j)\in P}\log(1-e^{-r_{ij}}).$$

   Gauge: `sum(a) - sum(b) = 0`.

2. **Strengths + cost:** `r_ij = a_i + b_j + gamma d_ij`.

   $$F=\sum_i s_i^{out}a_i+\sum_j s_j^{in}b_j+\gamma C
   -M\sum_{(i,j)\in P}\log(1-e^{-r_{ij}}).$$

   Validate non-negative finite costs, target cost `C`, and `r_ij > 0` for all
   pairs with missing costs treated exactly as the sampler/filter treats them.

3. **Strengths + total edges:** use `eta = log(lambda)` and
   `r_ij = a_i + b_j`.

   $$F=\sum_i s_i^{out}a_i+\sum_j s_j^{in}b_j-E\eta
   +\sum_{(i,j)\in P}\log(1+\exp(\eta)G_M(r_{ij})).$$

   Residuals must recover strengths and `sum(pi_ij) = E`.

4. **Strengths + degrees:** use `c_i = log(z_i)`, `d_j = log(w_j)`,
   `v_ij = exp(c_i+d_j)`, and `r_ij = a_i + b_j`.

   $$F=\sum_i s_i^{out}a_i+\sum_j s_j^{in}b_j
   -\sum_i k_i^{out}c_i-\sum_j k_j^{in}d_j
   +\sum_{(i,j)\in P}\log(1+\exp(c_i+d_j)G_M(r_{ij})).$$

   Gauges: `sum(a)-sum(b)=0` and `sum(c)-sum(d)=0`. Residuals must recover
   strengths and directed degrees.

5. **Degrees + total events:** use global `rho = -log(q) > 0` and
   `v_ij = exp(c_i+d_j)`.

   $$F=T\rho-\sum_i k_i^{out}c_i-\sum_j k_j^{in}d_j
   +\sum_{(i,j)\in P}\log(1+\exp(c_i+d_j)G_M(\rho)).$$

   Because `G_M(rho)` is common, implementation **must** solve `q` from
   `m_+(q,M)=T/E` via Brent's method, then reuse `balance_degree_bernoulli`
   with effective multipliers `z_i' = z_i sqrt(G_M(rho))`,
   `w_j' = w_j sqrt(G_M(rho))`. This is the only W constraint where IPF
   balancing works reliably, because the nonlinearity is isolated in a single
   global scalar and the remaining structure is identical to Bernoulli IPF. The
   exposed W result must include `q`, `positive_mean`, and total-event
   residuals. Reject `T < E` and infeasible degree supports.

#### Conic modeling details

- Implement stable scalar kernels first: `neg_ln_1m_exp_neg(r)`, `w_a(r,M)`,
  `w_g(r,M)`, `w_log_g(r,M)`, `w_mean(r,M)`, `w_occupation(v,r,M)`,
  `w_zip_mean(v,r,M)`, `w_positive_mean(q,M)`, and curvature/residual helpers.
- For independent strengths/cost, represent
  `t >= -log(1-exp(-r))` with two exponential-cone epigraphs and `u+v <= 1`.
- For zero-inflated models, model
  `log(1+exp(theta) G_M(r))` exactly. For `M=1`, use
  `log G_1(r) = -r - log(1-exp(-r))` and a log-sum-exp epigraph. For `M>1`, use
  exact CVXRust atoms or Clarabel exponential/power cones for
  `G_M(r)=(1-exp(-r))^{-M}-1`; do not approximate it with the geometric case.
- Track lifted problem size: number of original variables, auxiliary variables,
  exponential cones, power cones, linear constraints, and sparse nonzeros.

#### Implementation sequence

1. Create a dedicated feature branch and add `cvxrust`/Clarabel behind the Rust
   fitting implementation only. Verify sparse assembly, exponential cones,
   power-cone support, solver status, infeasibility reporting, and warm starts.
2. Complete W sampling/filtering coverage before declaring W fitting done:
   add `ZeroInflatedGeometric` and `ZeroInflatedNegativeBinomial` pair distributions; update providers;
   add Rust, PyO3, Python wrappers; and add p-value/absent-edge tests.
3. Implement `fitting/w.rs` result structs with solver status, objective,
   `iterations`, `min_margin = min(r_ij)`, `max_q`, fitted scalar multipliers,
   max/total residuals for every active constraint, and lifted problem metrics.
   Fixed-strength diagnostics and residual helpers are in place; extend the
   same typed diagnostics to cost, edges, and degree W fits as those solvers are
   added.
4. Fit strengths first, then strengths+cost, because both use independent W
   pairs and share the barrier term. Keep the same organization as the ME/B
   fitters: W strength-cost implementation belongs in `fitting/w.rs`, with
   shared result types in `fitting/types.rs` and only compatibility re-exports
   outside the fitting module. **Done for initial geometric/NB APIs; add more
   edge-case tests and benchmarks later.**
5. Fit degrees+events next. This is the only W constraint where balancing works:
   solve `q` from the scalar equation `m_+(q,M) = T/E` via Brent's method, then
   reuse the existing `balance_degree_bernoulli` IPF with effective multipliers
   scaled by `sqrt(G_M(rho))`. Do **not** use the conic solver for this case.
   Add a smoke test confirming that the IPF result matches the conic objective
   value on tiny graphs.
6. Fit strengths+edges next, then strengths+degrees. These require exact
   zero-inflated W partition terms and should reuse the same pair kernels as
   sampling/filtering.
7. Add PyO3 functions and Python dataclasses/wrappers for every API in the
   public W scope table; export them from `odme.models` and update stubs. Keep
   the final API clean: prefer one constraint-oriented fit type per constraint
   (`StrengthCostFit`, `StrengthEdgesFit`, etc.) with `family`, optional
   `layers`, common optimization diagnostics, and nested conic diagnostics over
   separate public types for each ME/B/W case.
8. Mirror that cleanup privately in Rust/PyO3: shared result structs should make
   common fields obvious (`x`, `y`, scalar multipliers, `converged/status`,
   `iterations`, residuals), while W-only lifted-problem metrics remain typed
   diagnostics instead of leaking into every ME/B/Binomial result.
9. Fold partial-constraint results into the same model-result ontology. Partial
   fits are not a different mathematical result type; they are fixed-strength,
   strength-cost, strength-edges, degree, or strength-degree fits on a masked
   support with known-rate contributions. Their public/private result shape
   should expose the corresponding constraint fit fields plus sparse support /
   mask metadata and diagnostics.
10. Update docs for public API and thesis terminology: concepts, API, CLI if W
   commands are exposed, and a decision record for the conic formulation.

#### TDD checklist

1. Scalar kernel tests for limiting cases (`r -> 0+`, large `r`, `M=1`, `M>1`).
2. Pair-distribution tests for means, occupations, CDF/SF, absent-edge p-values,
   seeded sampling, and non-negative integer weights.
3. Analytic 1-node and homogeneous-network fits for all five W constraints.
4. No-self-loop and mask feasibility tests for zero and nonzero constraints.
5. Residual recovery tests for strengths, degrees, edges, total events, and cost.
6. Property tests: balanced strengths, graphical degree supports,
   `max(q_ij) < 1`, non-negative multipliers after gauge recentering, and seeded
   reproducibility.
7. Legacy comparisons where meaningful: `fitter_s.py` W, `fitter_k.py`
   `rho_calculator(indist=True)`, and generation/filtering distributions from
   `ula_null_models.c`. Do not require compatibility with legacy convergence
   behavior.

#### Benchmark update

- Add W fitting benchmarks separate from streaming generation, for example
  `benchmarks/bench_w_fitting.py`, with `N={10, 25, 50, 100}` initially and a
  hard cap that prevents accidental dense all-pairs conic solves at large `N`.
- Benchmark all ten W fitting APIs: geometric and negative binomial for strengths,
  degree-events, strength-cost, strength-edges, and strength-degree.
- Record wall time, iterations, solver status, objective, max residuals,
  `min_margin`, `max_q`, variables, cones, linear constraints, sparse nonzeros,
  and max RSS.
- Extend `benchmarks/bench_quick.py` with tiny W smoke timings and extend
  `benchmarks/bench_streaming_generation.py` only for W sampling/filtering cases
  whose fit has already been produced at safe `N`.
- Update `benchmarks/regression_baselines.json` with conservative smoke
  thresholds, not large-scale promises. Document practical W fitting limits in
  `docs/development/benchmarking.md` after measurements.

### Refactor: ontology-driven sampling and filtering — DONE

Goal: remove repetitive boilerplate from sampling/filtering while preserving
explicit public APIs and keeping fitting algorithms readable.

Implemented:

- Rust sampling now has an internal `SamplingModel` dispatcher over provider-
  backed constraints and `WeightFamily`.
- Cost-map construction is centralized for Rust sampling.
- Public API names use `stub_matching` for the exact-strength stub sampler.
- Public API names now use `negative_binomial` instead of old abbreviated names.
- Python W zero-inflated filtering wrappers exist for strength-cost, strength-edges,
  strength-degree, and degree-events geometric/negative-binomial models.
- Tests cover the new sampling names and the W zero-inflated filtering wrappers.
- Benchmark scripts share common synthetic-network/cost/timing helpers in
  `benchmarks/common.py`; old generated benchmark result artifacts were
  removed from `benchmarks/results/`.

Deferred:

- A public `sample_model(...)` API remains intentionally private.
- Full Rust filtering dispatcher remains future cleanup; existing filtering
  already reuses `PairDistributionProvider`, and this pass focused on Python
  coverage and naming consistency.
- Degree-events samplers/filters still accept positive-weight parameters where
  the fit object does not fully encode all needed distribution parameters.

#### Scope

| Area | Refactor target | Public API impact |
|------|-----------------|-------------------|
| Rust sampling | One generic internal dispatcher over constraint + weight family | Keep existing named functions initially |
| Rust filtering | Same provider-dispatch pattern as sampling | Keep existing named functions initially |
| PyO3 wrappers | Thin wrappers route through shared helpers | Keep existing exported names initially |
| Python sampling | One internal factory/dispatcher; named functions become adapters | Keep existing named functions initially |
| Python filtering | Same internal factory/dispatcher as sampling | Add missing W zero-inflated convenience wrappers through the factory |
| Fitting | Only share validation, logging, result wrapping, and fixed-degree balancing reuse | No broad solver abstraction |
| Naming | Use consistent `stub_matching` and `negative_binomial` terminology | Breaking rename completed |

#### Ontology to encode explicitly

Introduce small typed enums/specs instead of encoding every combination only in
function names.

| Concept | Values |
|---------|--------|
| `Case` | `ME`, `W`, `B` |
| `Constraint` | `strength`, `strength_edges`, `strength_degree`, `degree_events`, `strength_cost`, `custom_probability` |
| `WeightFamily` | `poisson`, `geometric`, `negative_binomial`, `binomial` |
| `Statistic` | `independent`, `poisson_multinomial`, `multinomial`, `stub_matching` |
| `Inflation` | `none`, `zero_inflated` where pair occupation is constrained separately |

Notes:

- Keep `WeightFamily` in Rust as the canonical pair-family selector.
- Treat `negative_binomial(layers=M)` and `binomial(layers=M)` as parameterized
  families; validate `M` at API boundaries.
- Use `stub_matching` for the exact-strength sampler. It is a stub-matching
  algorithm over distinguishable events, not a generic uniform sampler over all
  integer matrices.
- Use zero-inflated model names for public APIs. Internally, pair samplers may
  describe the positive edge-weight draw as conditioned on edge existence.

#### Rust sampling design

Current good foundation:

- `WeightFamily` chooses the pair distribution family.
- `PairDistribution` handles expected value, occupation, p-values, and sampling.
- `PairDistributionProvider` abstracts candidate-pair support and per-pair
  distribution construction.
- `sample_provider()` already samples any provider.

Refactor plan:

1. Add an internal `SamplingSpec`/`ModelSpec` enum that combines constraint and
   family without replacing the existing public functions.
2. Add a single internal dispatcher, e.g. `sample_model(spec, seed)`, that
   builds the appropriate provider and calls `sample_provider()`.
3. Replace repeated `sample_strength_*_*` bodies with one-line adapters that
   construct a spec.
4. Factor repeated cost-map construction into a helper used by all cost
   sampling/filtering code.
5. Keep multinomial, Poisson-multinomial, and `stub_matching` as explicit
   non-provider samplers because they do not reduce to independent pair draws in
   the same way.
6. Do not hide mathematically different positive-weight-rate calculations. For
   example, degree-events Poisson still needs the positive-edge Poisson rate
   solve; W degree-events receives/uses the W positive-rate parameter. However, abstract away the zero inflated cases, since the mechanics are the same for sampling. First draw the edge binary probability according to model spec, then draw the other statistic according also to model spec. But the core can be reused.

#### Rust filtering design

Filtering should mirror sampling almost exactly:

1. Reuse `PairDistributionProvider` for all edge p-values and absent-edge
   filtering.
2. Add an internal filtering dispatcher over the same model spec/family enums.
3. Replace repeated `filter_*` and `filter_absent_*` functions with thin named
   adapters.
4. Add the missing W zero-inflated Python-level wrappers by routing through the generic
   filtering dispatcher.
5. Keep sparse custom-probability filtering separate if its support semantics do
   not match all-pairs constraints. It should not in principle.

#### PyO3 design

The native extension currently amplifies boilerplate because every public Python
function maps to a dedicated Rust wrapper.

Plan:

1. Keep named PyO3 functions for backward compatibility and clear `.pyi` stubs.
2. Internally route them to shared helper functions that accept:
   - constraint kind;
   - family kind + layers;
   - common arrays/multipliers;
   - self-loop flag;
   - seed or filtering thresholds.
3. Use small helper structs for parsed inputs instead of long repeated argument
   conversion blocks.
4. Generate or centralize `.pyi` declarations only after the internal shape is
   stable; do not mix stub churn with the first behavior-preserving refactor.

#### Python sampling/filtering design

Python should remain a thin, typed shell over Rust.

1. Add internal specs in `src/odme/models/generation.py` and
   `src/odme/filtering.py` only if they reduce wrapper duplication without
   making public docs harder to read.
2. Preserve explicit public functions such as
   `sample_strength_edges_geometric(...)`; each should delegate to a shared
   private helper.
3. Centralize conversions to lists/arrays and `EdgeTable` construction.
4. Use `sample_strength_stub_matching(...)` as the exact-strength sampler name.
   Breaking API changes are acceptable when they keep the module conceptually
   healthy.
5. Use the same family names in Python and Rust, spelled consistently as
   `negative_binomial`. Names in all modules should be consistent across Python,
   Rust interfaces, and CLI.

#### Fitting boundaries

Do not force the fitting module into the same factory abstraction.

Allowed cleanup:

- common feasibility checks for balanced strengths/degrees, support capacity,
  non-negative finite costs, target edge/event ranges, and `layers` validation;
- common logging/convergence warning wrapper;
- common result construction helpers for `FitResult`, `StrengthEdgesFit`,
  `StrengthDegreeFit`, `StrengthCostFit`, and `DegreeEventsFit`;
- shared fixed-degree Bernoulli balancing, including W degree-events where the
  plan already says the W scalar solve reduces to Bernoulli occupation IPF.

Avoid:

- a universal `fit_model()` solver abstraction for all constraints;
- hiding conic W fitting behind Poisson/binomial-like IPF names;
- over-generalizing fixed strength-cost fitting when Poisson/B/W objectives
  differ enough that explicit code is easier to audit scientifically.

#### Tests for the refactor

Use characterization tests before touching implementation:

1. For every existing public sampling function, compare old vs refactored output
   exactly for fixed seeds on small fixtures.
2. For filtering, compare p-values/decisions exactly for a small graph across
   all currently exported families and constraints.
3. Add coverage for the new `stub_matching` name.
4. Add enum/spec validation tests: invalid family/constraint combinations fail
   with clear errors.
5. Keep scientific invariant tests unchanged: non-negative integer weights,
   seeded reproducibility, expected occupied-edge semantics, and no-self-loop
   behavior.

#### Migration sequence

1. Write characterization tests around current sampling and filtering wrappers.
2. Introduce Rust helper/spec types without changing public functions.
3. Convert Rust sampling adapters to the internal dispatcher.
4. Convert Rust filtering adapters to the internal dispatcher.
5. Convert PyO3 wrapper bodies to shared helpers.
6. Convert Python sampling/filtering wrappers to shared private helpers.
7. Add `stub_matching` naming in Python/Rust/docs and remove old aliases during
   the transition.
8. Run the full Python/Rust test suites, then update API/concept docs.

#### Questions before implementation

1. Old exact-strength sampler aliases should not remain. Do not be afraid of
   breaking API changes if the entire suite is adapted.
2. Should a generic public API such as `sample_model(spec, seed=...)` be exposed,
   or should the spec/factory remain private while public named functions stay
   the main API? The factory should be private, but docs should be written to explain how to easily extend to new cases (only the independent family cases, with fixed W-B-ME types, but different constraints). This only affects the fitter module mostly, since it changes the saddle point equations, not the rest of the code (or it would be easy to adapt).
3. Should `negative_binomial` replace old abbreviated names everywhere in public names,
   or only inside the internal ontology? It should be consistent, so yes. Also, geometric and negative binomial with one layer are the same, so avoid unnecessary repetition, only keep the difference in the public facing methods.
4. Should `Case` values be public Python enums (`Case.ME`, `Case.W`, `Case.B`),
   or Rust/Python-private implementation details for now? Not now, they do not add value.
5. For W degree-events, should public wrappers accept `positive_weight_rate` as
   today, or should they accept the fitted `DegreeEventsFit` with `q` and derive
   the sampler parameter internally? In all cases, not only W, derive from the Fit. Add a Fit class for custom qij provided statistics across all cases to make it all conformant. "q" is not the positive weight rate, but the product of lagrange strength/event related multipliers, that map to the parameters of each distribution in the independent case. Be careful with this, not to mix both.
6. Should custom-probability Poisson/multinomial models be included in the same
   ontology as constraints, or kept as a separate utility family? Implement them as constraint.custom probability.
7. Should filtering and sampling share one exact `ModelSpec`, or should they use
   separate specs with a common lower-level family/constraint enum? Share model description, separate operation options.
8. What deprecation policy do you want for old names while the final MENoBiS
   rename is still pending? Rename immidiately. Perform all naming changes that belong in this refactor (not the MENOBIS renaming). Do not be afraid to break the API externally.

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

### Final milestone: publish MkDocs to GitHub Pages — NOT STARTED

After the scientific refactor and final naming decisions are stable, publish the
MkDocs Material documentation to a dedicated GitHub Pages site. This is a release
polish milestone, not part of the W fitting implementation.

Publishing scope:

| Item | Target |
|------|--------|
| Source docs | `docs/` + `mkdocs.yml` |
| Validation | `mkdocs build --strict` in CI |
| Deployment | GitHub Actions workflow for Pages |
| Hosting | Dedicated GitHub Pages environment/site |
| Branch policy | deploy only from protected default branch or release tags |

Required steps:

1. Decide final Pages URL after the MENoBiS rename decision.
2. Configure `site_url`, repository links, and canonical project name in
   `mkdocs.yml`.
3. Add a GitHub Actions workflow that installs with `uv`, runs
   `mkdocs build --strict`, uploads the generated `site/` artifact, and deploys
   to GitHub Pages.
4. Protect the Pages deployment environment and document who can publish.
5. Add docs release instructions under `docs/development/`.
6. Verify published navigation, search, API links, and thesis terminology links
   after deployment.

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
| `fitter_s.py` case `W` | W fixed-strength fitting (7c) |
| `fitter_k.py` `rho_calculator(indist=True)` | W degree-events fitting (7c) |
| `fitter_grav.py` + thesis equations | W strength-cost fitting (7c) |
| `fitter_E.py` / thesis equations | W strength-edges fitting (7c) |
| `fitter_sk.py` / thesis equations | W strength-degree fitting (7c) |
| `ula_null_models.c` | Audit/complete W samplers and filters for all constraints (7c) |
| `others_null_models.c` | Benchmark radiation and sequential gravity before removal; do not port |
