# Phase 5 Specification — Fixed Strengths with Expected Cost

**Repository:** `uladribia/menobis`  
**Planning baseline:** `master` after Phase 4 merge `aeb9483bf04e6cf04dbc19b8bbd10a2cfbab23ed`  
**Version:** August 2026

---

# 1. Purpose

This document specifies Phase 5 of the MENoBiS microcanonical roadmap:

\[
\boxed{\text{fixed directed strengths}+\text{expected cost}}
\]

for ME, B, and W.

Phase 5 must reuse the Phase 4 fixed-strength implementation. It must not introduce a new state, chain, move kernel, family abstraction, cost-provider abstraction, feasibility problem, or joint \(x,y,\gamma\) fitter.

The only new scientific ingredient is

\[
e^{-\gamma C(\mathbf t)}.
\]

Only the scalar parameter

\[
\boxed{\gamma}
\]

must be fitted.

---

# 2. Ensemble definition

The hard constraints are

\[
\sum_j t_{ij}=s_i^{\mathrm{out}},
\qquad
\sum_i t_{ij}=s_j^{\mathrm{in}}.
\]

Hence

\[
T=\sum_i s_i^{\mathrm{out}}=\sum_j s_j^{\mathrm{in}}
\]

is fixed automatically.

Define

\[
C(\mathbf t)=\sum_{ij}c_{ij}t_{ij}.
\]

Cost is expected, not hard. The target is

\[
P_F(\mathbf t\mid\mathbf s,\gamma)
=
\frac{1}{Z_F(\mathbf s,\gamma)}
D_F(\mathbf t)e^{-\gamma C(\mathbf t)}.
\]

The family degeneracies are

\[
D_{\mathrm{ME}}(\mathbf t)\propto\prod_{ij}\frac{1}{t_{ij}!},
\]

\[
D_{\mathrm B}(\mathbf t)=\prod_{ij}\binom{M}{t_{ij}},
\qquad 0\le t_{ij}\le M,
\]

and

\[
D_{\mathrm W}(\mathbf t)=
\prod_{ij}\binom{M+t_{ij}-1}{t_{ij}}.
\]

---

# 3. Relation to the grand-canonical model

The GC strength-cost law contains factors

\[
x_i y_j e^{-\gamma c_{ij}}.
\]

Schematically,

\[
P_{\mathrm{GC}}(\mathbf t)
\propto
D_F(\mathbf t)
\left(\prod_i x_i^{s_i^{\mathrm{out}}}\right)
\left(\prod_j y_j^{s_j^{\mathrm{in}}}\right)
e^{-\gamma C(\mathbf t)}.
\]

Conditioning on exact strengths makes the \(x_i,y_j\) terms constant, so

\[
P_{\mathrm{GC}}(\mathbf t\mid\mathbf s)
\propto
D_F(\mathbf t)e^{-\gamma C(\mathbf t)}.
\]

Therefore Phase 5 needs:

- the same cost definition;
- the same cost provider;
- the same sign convention;
- only \(\gamma\).

It does not need the fitted \(x_i,y_j\) values and must not reuse the full joint GC solver as the Phase 5 fitting algorithm.

---

# 4. Phase 4 components to reuse

Reuse unchanged wherever possible:

```text
fixed_strength/
    chain.rs
    domain.rs
    feasibility.rs
    initializer.rs
    move_cycle.rs
    problem.rs
    state.rs
    target.rs
```

Reuse:

- `StrengthState`;
- `FixedStrengthChain`;
- `cycle4_step`;
- `PairDomain`;
- transportation/max-flow feasibility;
- fixed-pair residualization;
- `McmcConfig`;
- `McmcCounters`;
- `OccupationFamily`;
- ME/B/W local degeneracy;
- exact strength preservation;
- masks and loop policies.

Phase 5 is an extension of the target and orchestration, not a new sampler.

---

# 5. Existing cost-ready target

The current target already has the correct shape:

```rust
pub struct StrengthTarget<'a> {
    pub family: OccupationFamily,
    pub gamma: f64,
    pub costs: Option<&'a dyn PairCostProvider>,
}
```

Its intended local ratio is

\[
\Delta\log\pi
=
\Delta\log D_F-\gamma\Delta C.
\]

Keep this target as the scientific core.

---

# 6. Reuse `PairCostProvider`

Reuse the existing shared trait:

```rust
pub trait PairCostProvider: Sync {
    fn cost(
        &self,
        source: usize,
        target: usize,
    ) -> Option<f64>;
}
```

Reuse existing providers such as `EuclideanCostProvider`.

Do not introduce a second cost API.

The same provider must be usable by:

- grand-canonical strength-cost;
- Phase 5 gamma fitting;
- Phase 5 MCMC;
- cost diagnostics.

---

# 7. Lazy cost evaluation

Do not scan or materialize a dense \(N^2\) cost matrix.

Query costs only for:

- positive cells in the initial state;
- cells touched by a proposal;
- positive cells in sampled states;
- fixed positive pairs;
- optional optimization routines that explicitly require cost access.

The functional provider design already supports this.

---

# 8. Cost error semantics

The Phase 5 domain is defined by:

- `PairDomain`;
- masks;
- self-loop policy;
- fixed cells.

A missing cost must not redefine the domain.

If the sampler requests the cost of an admissible pair and receives `None`, abort with an error.

Likewise reject:

- `NaN`;
- \(+\infty\);
- \(-\infty\).

Required errors:

```rust
MissingCost {
    source: u64,
    target: u64,
}

NonFiniteCost {
    source: u64,
    target: u64,
    value: f64,
}
```

A missing cost is not:

- a normal hold;
- a Metropolis rejection;
- an implicit structural zero.

Negative finite costs remain allowed unless a separate MENoBiS rule forbids them.

---

# 9. Target return contract

The current `Option<f64>` return type conflates invalid occupations and invalid cost-provider responses.

Use either:

```rust
pub enum TargetDelta {
    Valid(f64),
    InvalidOccupation,
}
```

with

```rust
Result<TargetDelta, FixedStrengthCostError>
```

or:

```rust
Result<Option<f64>, FixedStrengthCostError>
```

with semantics:

```text
Ok(Some(delta))  valid local move
Ok(None)         family occupation invalid
Err(...)         cost-provider/configuration error
```

Errors must propagate through the move and chain and terminate the run.

---

# 10. Acceptance rule

For a local change

\[
t_{ij}\to t'_{ij},
\]

compute

\[
\Delta\log\pi_{ij}
=
\log d_F(t'_{ij})
-
\log d_F(t_{ij})
-
\gamma c_{ij}(t'_{ij}-t_{ij}).
\]

For a cycle, sum over changed cells.

Accept with

\[
\alpha=\min(1,e^{\Delta\log\pi}).
\]

Use log space.

No cost-specific move kernel is needed.

---

# 11. Reuse the four-cycle move

Reuse the Phase 4 move exactly.

A positive orientation is

\[
t_{ab}'=t_{ab}+1,\qquad
t_{cd}'=t_{cd}+1,
\]

\[
t_{ad}'=t_{ad}-1,\qquad
t_{cb}'=t_{cb}-1.
\]

Strengths remain fixed.

The cost difference is

\[
\Delta C=c_{ab}+c_{cd}-c_{ad}-c_{cb},
\]

which follows automatically by summing local deltas.

Do not add `cost_cycle_step` or another move implementation.

---

# 12. Disable ME direct sampling for nontrivial cost

The Phase 4 ME stub matcher samples

\[
P(\mathbf t\mid\mathbf s)
\propto
\frac{1}{\prod t_{ij}!}.
\]

It does not sample

\[
\frac{e^{-\gamma C(\mathbf t)}}{\prod t_{ij}!}.
\]

Therefore the direct backend is allowed only when:

```text
family == ME
and
(cost provider absent or gamma == 0)
```

Add:

```rust
pub fn has_nontrivial_cost(&self) -> bool {
    self.costs.is_some() && self.gamma != 0.0
}
```

and include it in backend selection.

---

# 13. Refactor the orchestrator

Extract an internal target-driven path:

```rust
fn sample_strength_target<'a>(
    problem: ResidualStrengthProblem,
    target: StrengthTarget<'a>,
    config: McmcConfig,
    has_fixed_pairs: bool,
) -> Result<
    (SampledNetwork, StrengthBackend),
    FixedStrengthError,
>;
```

The Phase 4 wrapper constructs a zero-cost target.

The Phase 5 wrapper constructs:

```rust
StrengthTarget::with_costs(
    problem.family,
    gamma,
    costs,
)
```

Do not duplicate validation, initialization, chain creation, burn-in, thinning, or reconstruction.

---

# 14. Scalar fitting problem

Let

\[
\mu_C(\gamma)=\mathbb E_\gamma[C].
\]

Fit

\[
\boxed{\mu_C(\gamma)=C_{\mathrm{obs}}}.
\]

Do not fit \(x_i,y_j\).

The key derivative is

\[
\frac{d}{d\gamma}\mu_C(\gamma)
=
-\operatorname{Var}_\gamma(C)
\le0.
\]

Expected cost is monotone non-increasing in \(\gamma\).

---

# 15. Reuse from GC fitting

Reuse only compatible scalar infrastructure:

- tolerances;
- iteration-limit conventions;
- scalar bracketing helpers;
- fit-result conventions;
- logging;
- observed-cost calculation;
- error formatting;
- cost-provider construction.

Do not reuse:

- independent-pair expectation formulas;
- \(x,y\) coordinate updates;
- joint residual systems;
- multidimensional Jacobians;
- GC multiplier normalization.

The Phase 5 expected cost must be estimated under the constrained chain.

---

# 16. Fitting method

Use stochastic bisection first.

Suggested config:

```rust
pub struct FixedStrengthCostFitConfig {
    pub gamma_lower: f64,
    pub gamma_upper: f64,
    pub max_iterations: usize,

    pub adaptation_sweeps: usize,
    pub estimation_sweeps: usize,
    pub samples_per_iteration: usize,

    pub absolute_cost_tolerance: f64,
    pub relative_cost_tolerance: f64,
    pub confidence_multiplier: f64,

    pub seed: u64,
}
```

At each midpoint

\[
\gamma_m=\frac{\gamma_{\mathrm{low}}+\gamma_{\mathrm{high}}}{2},
\]

perform:

1. update gamma;
2. run adaptation sweeps;
3. collect cost samples;
4. estimate mean and standard error;
5. update the bracket.

Because the function decreases:

```text
mean cost > observed cost
    gamma too small
    move lower bound upward

mean cost < observed cost
    gamma too large
    move upper bound downward
```

---

# 17. Persistent warm-started fitting

Initialize the Phase 4 chain only once.

Reuse the same state while changing gamma.

Add:

```rust
pub fn set_gamma(&mut self, gamma: f64)
```

to `StrengthTarget` or replace the target in the chain.

After a gamma change, run adaptation sweeps before measuring cost.

Do not reconstruct the transportation table for each iteration.

---

# 18. Stopping criterion

Stop when

\[
|\hat\mu_C-C_{\mathrm{obs}}|
\le
\max(
\varepsilon_{\mathrm{abs}},
\varepsilon_{\mathrm{rel}}|C_{\mathrm{obs}}|
)
\]

and Monte Carlo uncertainty is sufficiently small:

\[
z\,\operatorname{SE}(\hat\mu_C)
\le
\max(
\varepsilon_{\mathrm{abs}},
\varepsilon_{\mathrm{rel}}|C_{\mathrm{obs}}|
).
\]

If the residual is small but uncertainty is too large, collect more samples at the same gamma.

---

# 19. Gamma bracket

Support user-supplied bounds.

Otherwise expand a bracket geometrically around an initial gamma.

A valid bracket should satisfy, within uncertainty,

\[
\mu_C(\gamma_{\mathrm{low}})
\ge C_{\mathrm{obs}},
\]

\[
\mu_C(\gamma_{\mathrm{high}})
\le C_{\mathrm{obs}}.
\]

Bound the number of expansions.

Return `BracketNotFound` if no bracket can be established.

---

# 20. Gamma sign

Do not force

\[
\gamma\ge0.
\]

Low observed cost generally requires positive gamma.

High observed cost may require negative gamma.

Because strengths fix a finite total occupation and the feasible state space is finite, finite negative gamma is valid for ME, B, and W.

---

# 21. Cost measurement

Add:

```rust
pub fn state_cost(
    state: &StrengthState,
    costs: &dyn PairCostProvider,
) -> Result<f64, FixedStrengthCostError>;
```

Iterate only over positive occupations:

\[
C(\mathbf t)=
\sum_{(i,j):t_{ij}>0}c_{ij}t_{ij}.
\]

Fail immediately on missing or non-finite cost.

Complexity is

\[
O(E_{\mathrm{occupied}}).
\]

---

# 22. Fixed-pair residual cost

Fixed occupations contribute

\[
C_{\mathrm{fixed}}
=
\sum_{(i,j)\in F}
c_{ij}t_{ij}^{\mathrm{fixed}}.
\]

Fit against

\[
C_{\mathrm{obs,res}}
=
C_{\mathrm{obs,total}}
-
C_{\mathrm{fixed}}.
\]

Validate costs lazily for the fixed positive pairs.

The residual MCMC target contains only variable cells.

Final reported cost is

\[
C_{\mathrm{total}}
=
C_{\mathrm{fixed}}
+
C_{\mathrm{residual}}.
\]

---

# 23. Public modes

## Explicit gamma

Allow expert use with a supplied gamma.

## Fit gamma

Allow standard use with:

- observed strengths;
- observed cost;
- a cost provider;
- fit configuration.

If a GC fit is supplied, its gamma may be used as:

- an initial guess;
- or an explicitly requested fixed gamma.

Do not silently rerun or reuse the full joint GC fit as the Phase 5 fitting method.

---

# 24. Fit result

```rust
pub struct FixedStrengthCostFitResult {
    pub gamma: f64,
    pub expected_cost_estimate: f64,
    pub expected_cost_standard_error: f64,
    pub observed_cost: f64,
    pub residual: f64,
    pub iterations: usize,
    pub converged: bool,
    pub bracket_lower: f64,
    pub bracket_upper: f64,
    pub mcmc_proposals: u64,
    pub mcmc_accepted: u64,
}
```

Also retain:

- seed;
- family;
- sample count;
- fixed cost;
- residual target cost.

---

# 25. Error types

Add:

```rust
pub enum FixedStrengthCostError {
    MissingCost {
        source: u64,
        target: u64,
    },

    NonFiniteCost {
        source: u64,
        target: u64,
        value: f64,
    },

    NonFiniteObservedCost {
        value: f64,
    },

    InvalidGamma {
        value: f64,
    },

    InvalidBracket {
        lower: f64,
        upper: f64,
    },

    BracketNotFound,

    FitDidNotConverge {
        iterations: usize,
        residual: f64,
    },

    CostNotIdentifiable,

    ResidualCostInconsistent {
        total: f64,
        fixed: f64,
        residual: f64,
    },

    FixedStrength(FixedStrengthError),
}
```

---

# 26. MCMC uncertainty

Do not treat correlated samples as IID.

Use batch means.

Split collected costs into contiguous batches, compute batch means, and estimate the standard error from their variance.

Require a minimum number of batches.

---

# 27. Cost identifiability

If cost is constant over the feasible state space, then

\[
\operatorname{Var}_\gamma(C)=0
\]

and gamma is not identifiable.

Examples:

- constant pair cost under fixed \(T\);
- a one-state feasible problem;
- constrained domains where every feasible table has equal cost.

Detect this and return `CostNotIdentifiable`.

---

# 28. Validation

Required tests:

1. Hand-checked local target deltas for ME, B, and W.
2. Missing cost aborts.
3. NaN and infinite cost abort.
4. Zero-gamma Phase 5 agrees with Phase 4.
5. Constant cost leaves probabilities unchanged.
6. Exact enumeration for small ME, B, and W cases.
7. Detailed balance with nonconstant cost.
8. Conditioned-GC identity.
9. Gamma recovery from an exactly enumerable target.
10. Fixed-pair residual cost correctness.
11. Positive and negative gamma cases.
12. Persistent-chain warm-start regression.

---

# 29. Performance rules

Phase 5 adds only:

- up to four cost lookups per four-cycle proposal;
- cost accumulation for sampled states;
- repeated sweeps during scalar fitting.

Do not:

- scan dense cost matrices;
- precompute \(N^2\) costs;
- rebuild the chain per gamma;
- rebuild max-flow initialization per gamma;
- add an unbounded complete-domain cost cache.

Optional caching is allowed only for explicit sparse domains and must be bounded.

---

# 30. Repository additions

Recommended:

```text
fixed_strength/
    cost.rs
    cost_fit.rs
```

## `cost.rs`

- lazy cost lookup;
- strict cost validation;
- state cost;
- fixed-pair cost;
- cost errors.

## `cost_fit.rs`

- fit config;
- fit result;
- bracket construction;
- stochastic bisection;
- batch-means uncertainty;
- gamma update.

Minimal modifications to:

```text
target.rs
move_cycle.rs
chain.rs
errors.rs
```

---

# 31. Commit sequence

1. `refactor(fixed-strength): inject target into shared orchestrator`
2. `feat(fixed-strength): add strict lazy cost error semantics`
3. `feat(fixed-strength): add expected-cost sampling wrapper`
4. `feat(fitting): add scalar gamma fit for fixed strengths`
5. `test(fixed-strength): validate cost-tilted ME B W by enumeration`
6. `test(fixed-strength): add conditioned-GC and gamma-recovery tests`
7. `feat(api): expose fixed-strength expected-cost sampling`
8. `feat(benchmarks): add strength-cost case to micro benchmark CLI`
9. `docs(benchmarks): report executed Phase 5 benchmark results`

---

# 32. Benchmark CLI integration

Phase 5 is not complete until the new case is available through the repository benchmark CLI.

Extend the existing microcanonical benchmark command with the strength-cost constraint, following the current CLI conventions.

Preferred form:

```text
python -m benchmarks micro --constraint strength-cost
```

If the repository uses canonical constraint identifiers in CLI values, use the established `STRENGTH_COST` spelling instead. Do not introduce a separate standalone benchmark executable unless the existing CLI cannot represent the case cleanly.

## 32.1 Required CLI arguments

The benchmark route should support:

- family: ME/B/W;
- node count or synthetic-network size;
- self-loop policy;
- layer count for B/W;
- fixed-pair fraction;
- burn-in sweeps;
- sweeps per sample;
- explicit gamma mode;
- fitted gamma mode;
- gamma-fit iteration and sampling controls;
- seed;
- coordinate-based Euclidean cost;
- output format consistent with existing benchmarks.

Reuse the current synthetic-network generation and strength extraction logic.

## 32.2 Required benchmark validation

Every run must verify:

\[
s_i^{\mathrm{out,sample}}
=
s_i^{\mathrm{out,target}},
\]

\[
s_j^{\mathrm{in,sample}}
=
s_j^{\mathrm{in,target}}.
\]

For B, also verify:

\[
0\le t_{ij}\le M.
\]

For fitted-gamma runs, verify that the reported expected-cost estimate satisfies the configured tolerance against the observed target cost, accounting for the reported Monte Carlo standard error.

For explicit-gamma runs, report the sampled mean cost but do not compare it to an observed-cost fitting target unless one was supplied.

Also verify:

- no forbidden self-loops;
- fixed-pair occupations;
- finite sampled cost;
- no missing-cost provider errors in successful runs.

## 32.3 Required benchmark matrix

At minimum run:

```text
ME:
    explicit gamma
    fitted gamma

B:
    explicit gamma
    fitted gamma

W:
    explicit gamma
    fitted gamma
```

Include at least:

- one small correctness-oriented case;
- one medium performance case;
- self-loops allowed;
- self-loops forbidden;
- one fixed-pair case;
- positive gamma;
- negative gamma where the target cost requires it.

The final gate does not require an exhaustive performance sweep, but all three families and both gamma modes must execute successfully.

## 32.4 Required metrics

Record:

- initialization time;
- gamma-fitting time;
- final sampling time;
- total wall time;
- proposals per second;
- acceptance rate;
- number of cost lookups if available;
- fitted or supplied gamma;
- observed target cost;
- estimated expected cost;
- expected-cost residual;
- Monte Carlo standard error;
- exact-strength validation result;
- peak memory if the benchmark framework already captures it;
- backend selected;
- seed and benchmark parameters.

## 32.5 Benchmark report

Commit or generate a concise benchmark report containing:

- exact command lines;
- repository commit SHA;
- machine/runtime information already used by the benchmark framework;
- one result table for ME/B/W;
- any failures or warnings;
- interpretation of cost residuals relative to uncertainty;
- known performance limitations.

Suggested location:

```text
docs/benchmarks/microcanonical_strength_cost.md
```

or the existing benchmark-results location used by the repository.

Do not report only that the command ran. Include actual timings, acceptance rates, gamma values, cost residuals, and validation results.

---

# 33. Final completion gate

Phase 5 is complete only when all items below pass.

## 33.1 Scientific and implementation gate

- Phase 4 remains the sampling core.
- No new state or move kernel is introduced.
- `PairCostProvider` is reused directly.
- Cost access is lazy.
- No dense cost scan occurs.
- Missing or non-finite cost aborts when touched.
- Only gamma is fitted.
- Fitting uses a persistent warm-started chain.
- Stochastic bisection is implemented.
- Monte Carlo uncertainty is reported.
- Fixed-pair cost is residualized correctly.
- ME/B/W exact enumeration passes.
- Zero-gamma regression passes.
- Conditioned-GC identity passes.
- Gamma-recovery tests pass.
- Public API and capability registry are updated.

## 33.2 Benchmark CLI gate

The implementation must add the new case to the benchmark CLI and run it.

Required command shape:

```text
python -m benchmarks micro --constraint strength-cost
```

or the repository-equivalent canonical spelling.

The final gate must execute successful benchmark runs for:

- ME explicit gamma;
- ME fitted gamma;
- B explicit gamma;
- B fitted gamma;
- W explicit gamma;
- W fitted gamma.

At least one run must include fixed pairs, and at least one must forbid self-loops.

Every successful run must validate exact strengths and family occupation bounds.

## 33.3 Required reported result

The implementing agent must include in the final delivery:

1. the exact benchmark commands executed;
2. the commit SHA tested;
3. a compact results table;
4. wall-clock timings;
5. acceptance rates;
6. supplied or fitted gamma;
7. observed target cost;
8. estimated expected cost;
9. cost residual;
10. Monte Carlo standard error;
11. exact-strength validation status;
12. any warnings, failures, or skipped cases with reasons.

A report file must be added to the repository benchmark documentation or results directory.

Phase 5 must not be marked complete merely because tests pass or the benchmark CLI route exists. The CLI route must actually be executed and its measured results reported.

---

# 34. Final principle

Phase 5 is:

\[
\boxed{
\text{Phase 4 fixed-strength chain}
+
\text{existing PairCostProvider}
+
\text{scalar }\gamma\text{ fitter}
}
\]

The acceptance rule is

\[
\boxed{
\Delta\log\pi
=
\Delta\log D_F
-
\gamma\Delta C
}
\]

and the fitting equation is

\[
\boxed{
\mathbb E_\gamma[C]
=
C_{\mathrm{obs}}.
}
\]
