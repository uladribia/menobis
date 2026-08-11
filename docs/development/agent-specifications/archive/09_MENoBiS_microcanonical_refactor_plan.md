# MENoBiS Microcanonical Refactor — Implementation Plan

**Repository:** `uladribia/menobis`  
**Scope:** Grand-canonical / microcanonical generation architecture, with primary focus on replacing the current microcanonical production implementations with scalable designs.  
**Target scale:** `N = 1,000–25,000` where the mathematical structure permits sparse scaling.  
**Primary implementation language:** Rust for all heavy computation.  
**Python role:** thin validation, orchestration, I/O, API wrapping, and benchmark invocation only.

---

# 1. Purpose

MENoBiS has reached a point where the current microcanonical implementation is becoming too complex and does not scale sufficiently.

The current repository contains several microcanonical subsystems organized primarily by named constraint case:

- fixed `(E,T)`,
- fixed `(k,T)`,
- fixed strengths,
- fixed strengths plus expected cost,
- related support samplers,
- exact occupation samplers,
- rejection samplers,
- dynamic-programming samplers,
- special-case direct samplers.

Several of these methods are mathematically exact for small systems, but they are unsuitable as long-term production algorithms because their practical range is often around `N≈100–500`, while MENoBiS grand-canonical workflows can reach much larger systems.

The purpose of this refactor is to replace constraint-specific production subsystems with a small number of reusable, scalable components.

The central rule is:

> **Decompose an ensemble as far as exact conditional factorization permits. Use MCMC only for the remaining coupled occupation-number problem.**

A second equally important rule is:

> **Only scalable algorithms belong in production code. Exact or combinatorial samplers that are useful only at marginal small N must live in test/reference infrastructure, never in the production generation path.**

This is a major refactor. It must be executed incrementally and statistically validated at each step.

---

# 2. Mandatory branching strategy

This refactor must not be developed directly from `master`.

Create one long-lived integration branch:

```text
microcanonical-refactor
```

This branch is the integration target for the entire project.

All implementation work must happen on short-lived subbranches created **from `microcanonical-refactor`**, never directly from `master`.

Example:

```text
master
  |
  +-- microcanonical-refactor
        |
        +-- refactor/model-primitives
        +-- refactor/fixed-total
        +-- refactor/fixed-et
        +-- refactor/binary-degree
        +-- refactor/fixed-kt
        +-- refactor/fixed-strength
        +-- refactor/strength-cost
        +-- refactor/fixed-es
        +-- refactor/test-oracles
        +-- refactor/final-cleanup
```

Workflow for every phase:

```text
checkout microcanonical-refactor
pull / update
create phase subbranch
implement one coherent phase
run phase gate
merge phase branch into microcanonical-refactor
delete phase branch
continue from updated microcanonical-refactor
```

Do not merge `microcanonical-refactor` back to `master` until the entire refactor is complete and the final global gate passes.

The integration branch must remain runnable throughout the project.

No phase is allowed to leave the integration branch with knowingly broken public functionality.

---

# 3. Scientific terminology

MENoBiS models **non-binary networks through integer occupation numbers**

\[
t_{ij}\in\mathbb N_0
\]

defined on admissible node pairs or states.

The binary support is derived from occupation:

\[
a_{ij}=\mathbf 1[t_{ij}>0].
\]

Use this terminology consistently.

Preferred terms:

- non-binary network,
- occupation number,
- occupation-number configuration,
- occupation state,
- occupied node pair,
- binary support,
- binary degree,
- strength.

Do **not** introduce `weighted` as the central implementation terminology for this architecture.

The word “weight” may appear in generic probability calculations such as `log_weight`, but network-state types and modules should use occupation terminology.

---

# 4. Constraint ontology

The implementation must distinguish **hard constraints** from **soft / expectation constraints**.

## 4.1 Hard constraints

Hard constraints define the admissible state space exactly.

Current hard quantities include:

- strength sequence \(s\),
- binary degree sequence \(k\),
- binary edge count \(E\),
- total occupation/events \(T\),
- fixed node-pair occupations,
- admissible-pair masks,
- self-loop policy.

Examples:

```text
fixed s
fixed E,T
fixed k,T
fixed E,s
```

## 4.2 Soft observables

Soft observables are matched in expectation through fitted multipliers.

Current important case:

```text
fixed s + expected cost <C>
```

This means:

\[
s=s^\ast
\]

exactly, but

\[
\mathbb E_\gamma[C]=C^\ast.
\]

Cost is continuous. It is therefore not treated as a hard exact equality constraint.

The MCMC target on the fixed-strength fiber is:

\[
\log \pi(\mathbf t)
=
\log g_F(\mathbf t)
-\gamma C(\mathbf t)
+\mathrm{const}.
\]

For a local move:

\[
\Delta\log\pi
=
\Delta\log g_F
-\gamma \Delta C.
\]

---

# 5. Fundamental architecture

The production architecture must be organized by **probabilistic structure**, not primarily by named constraint.

Conceptually:

```text
Prepared model
      |
      +-- Grand canonical
      |      |
      |      +-- fit parameters
      |      +-- direct factorized occupation sampling over pairs
      |
      +-- Microcanonical
             |
             +-- Factorized / conditional plan
             |      |
             |      +-- stage 1
             |      +-- stage 2
             |      +-- ...
             |
             +-- Coupled occupation-number MCMC
```

The current cases map as follows:

| Case | Production sampling structure |
|---|---|
| Grand canonical | direct pair/state factorization |
| fixed `(E,T)` | binary support + fixed-total occupation sampler |
| fixed `(k,T)` | fixed-degree binary support MCMC + same fixed-total occupation sampler |
| fixed `s` | occupation-number MCMC preserving strengths |
| fixed `(E,s)` | occupation-number MCMC preserving strengths and edge count |
| fixed `s,<C>` | fixed-strength occupation MCMC + fitted cost multiplier |

The implementation must preserve the possibility of adding more factorized cases later.

The router must therefore encode a **sampling plan**, not simply dispatch to one directory per constraint.

---

# 6. Grand canonical policy

Grand-canonical generation must **not** be replaced by MCMC merely for architectural symmetry.

Once parameters are fitted, GC models factorize over admissible node pairs.

The production path should remain:

```text
prepare problem
    ->
fit family/constraint parameters
    ->
iterate admissible pairs
    ->
sample pair occupation independently
    ->
emit sparse sampled network
```

Family- and constraint-specific fitting methods remain acceptable and expected.

Examples include:

- balancing / iterative proportional fitting,
- L-BFGS,
- scalar root searches,
- gamma fitting,
- specialized W-domain handling.

The architecture should unify model semantics and pair-domain logic, not force all numerical solvers into one algorithm.

---

# 7. Proposed production module structure

The exact final names may be adjusted to existing repository conventions, but responsibilities must follow this structure.

```text
crates/menobis-core/src/

    model/
        mod.rs
        family.rs
        hard_constraints.rs
        observables.rs
        problem.rs
        sampling_plan.rs

    occupation/
        mod.rs
        state.rs
        delta.rs
        target.rs

    binary/
        mod.rs
        domain.rs
        state.rs
        initializer.rs
        fixed_edges.rs
        degree_chain.rs
        moves.rs

    conditional/
        mod.rs
        fixed_total/
            mod.rs
            state.rs
            initializer.rs
            pair_conditional.rs
            chain.rs
            diagnostics.rs

    mcmc/
        mod.rs
        config.rs
        counters.rs
        outcome.rs
        acceptance.rs
        diagnostics.rs

    occupation_mcmc/
        mod.rs
        state.rs
        initializer.rs
        chain.rs
        moves/
            cycle.rs
            fixed_edge_count.rs

    fitting/
        ... existing scalable fitting kernels ...

    generation/
        mod.rs
        grandcanonical.rs
        microcanonical.rs
        output.rs
```

Do not create all modules immediately.

Create them incrementally as functionality migrates.

The final production tree must not contain large independent `fixed_et/`, `fixed_kt/`, `fixed_strength/` mini-frameworks each implementing its own state, chain, target, diagnostics, and occupation law.

---

# 8. Shared family model

Create one scientific source of truth for ME/B/W occupation-number degeneracy.

Conceptually:

```rust
pub enum OccupationFamily {
    ME,
    B { layers: u32 },
    W { layers: u32 },
}
```

The common family API should support operations equivalent to:

```rust
fn max_occupation(&self) -> Option<OccNum>;

fn log_base_measure(&self, t: OccNum) -> f64;

fn delta_log_base_measure(
    &self,
    old: OccNum,
    new: OccNum,
) -> f64;
```

The implementation must avoid repeatedly evaluating expensive gamma/factorial expressions when a local ratio has a cheaper closed form.

Example principle:

```text
old occupation = t
new occupation = t+1
```

should normally use a direct family ratio instead of:

```text
log_base_measure(t+1) - log_base_measure(t)
```

if a stable closed form is available.

This matters because local MCMC may execute millions of such operations.

Family-specific GC distributions may remain in specialized code, but microcanonical occupation degeneracy formulas must not be duplicated across constraint-specific samplers.

---

# 9. Prepared problem

Before generation, convert raw public constraints into a prepared immutable internal problem.

Preparation should perform:

- dimension validation,
- total consistency checks,
- fixed-pair residualization,
- mask validation,
- self-loop handling,
- removal of saturated/trivial rows or pairs where already supported,
- computation of residual \(E,T,k,s\),
- construction of efficient admissible-pair/domain structures.

The proposal loop must not repeatedly redo these operations.

Prepared problem state must avoid dense \(N\times N\) storage.

---

# 10. Sampling-plan routing

Introduce an internal routing representation.

A simple first form is sufficient:

```rust
enum SamplingPlan {
    GrandCanonical(GrandCanonicalPlan),

    FixedEdgesThenTotal(FixedEdgesTotalPlan),

    FixedDegreesThenTotal(FixedDegreesTotalPlan),

    OccupationMcmc(OccupationMcmcPlan),
}
```

A later generalized form may use factorized stages, but do not build a generic workflow engine before needed.

Document the conceptual future generalization:

\[
P(X_1,\dots,X_r)
=
P(X_1)
P(X_2|X_1)
\cdots
P(X_r|X_1,\dots,X_{r-1}).
\]

The immediate architecture must nevertheless keep factorized cases visibly separate from coupled occupation MCMC.

---

# 11. Fixed `(E,T)` — new production implementation

Fixed `(E,T)` should exploit factorization.

The binary support is sampled first:

\[
A\sim P(A\mid E).
\]

Then positive occupations are sampled conditional on support cardinality and total occupation:

\[
\mathbf t
\sim
P_F(\mathbf t\mid E,T).
\]

The second stage depends on:

- family \(F\),
- number of occupied pairs \(E\),
- total occupation \(T\),
- B layer bound \(M\),

but not on node-pair identities when no additional pair-dependent observable is present.

## 11.1 Binary support stage

After fixed-pair residualization, let:

```text
P = number of admissible free node pairs
E_free = required number of free occupied pairs
```

If all free pairs are equally weighted under the microcanonical support law:

```text
sample E_free distinct pairs uniformly without replacement
```

No MCMC is needed.

Production implementation should use a scalable sample-without-replacement strategy.

Do not materialize an \(N^2\) Boolean array.

Possible implementation strategies:

- sample indices from the admissible-pair iterator/rank space,
- reservoir sampling if the pair domain is streamed,
- partial Fisher-Yates over an indexed admissible domain if it is explicitly sparse,
- hash-based selection when `E_free << P`.

Choose based on the existing pair-domain abstraction and benchmark.

Requirements:

- exact support size,
- no duplicates,
- mask respected,
- self-loop policy respected,
- fixed occupied/free pairs respected.

## 11.2 Occupation stage

Use the common fixed-total occupation sampler described below.

## 11.3 Assembly

Associate the sampled occupation vector with the occupied pairs.

If the occupation law is exchangeable over support positions, randomize or assign positions consistently without introducing bias.

---

# 12. Fixed-total occupation sampler

This is one of the most important new production components.

It replaces:

- DP tables,
- exact composition enumeration,
- brute-force rejection conditioned on positivity,
- family-specific small-N occupation backends.

State:

\[
t_e\ge 1,
\qquad
\sum_{e=1}^{E}t_e=T.
\]

For B:

\[
t_e\le M.
\]

Memory must be:

\[
O(E).
\]

No production algorithm may allocate storage proportional to \(E\times T\).

---

# 13. Fixed-total initializer

The initializer only needs to produce a feasible state.

It does not need to be distributed according to the target because burn-in follows.

## 13.1 ME and W

Feasibility:

\[
T\ge E.
\]

Initialize:

\[
t_e=1.
\]

Residual:

\[
R=T-E.
\]

Balanced initialization:

\[
q=\left\lfloor \frac{R}{E}\right\rfloor,
\qquad
r=R\bmod E.
\]

Set:

```text
all cells += q
choose r distinct cells
those cells += 1
shuffle or randomize positions
```

Complexity:

\[
O(E).
\]

## 13.2 B

Feasibility:

\[
E\le T\le ME.
\]

Initialize one event per occupied pair.

Residual:

\[
R=T-E.
\]

Each pair has residual capacity:

\[
M-1.
\]

Distribute residual while respecting capacity.

Prefer a balanced randomized fill, not one-edge-at-a-time random retry.

The algorithm must have bounded runtime.

---

# 14. Fixed-total pair Gibbs kernel

One step:

1. select two distinct occupation positions \(a\neq b\);
2. compute
   \[
   q=t_a+t_b;
   \]
3. redraw their split from the exact two-cell conditional;
4. leave all other occupations unchanged.

Set:

\[
t_a'=k,
\qquad
t_b'=q-k.
\]

This preserves total occupation exactly.

Because the split is drawn from the exact conditional, the update is a Gibbs update and has acceptance probability one.

---

# 15. ME pair conditional

ME base measure:

\[
g_{\mathrm{ME}}(t)\propto \frac{1}{t!}.
\]

For a fixed two-cell sum \(q\),

\[
P(k|q)
\propto
\frac{1}{k!(q-k)!}.
\]

Thus:

\[
k\sim \mathrm{Binomial}(q,1/2)
\]

conditioned on:

\[
1\le k\le q-1.
\]

Implementation requirements:

- do not repeatedly sample unconstrained binomial until success when `q` is small and positivity rejection becomes noticeable;
- use a bounded truncated-binomial routine or endpoint correction;
- special case `q=2`: only split `(1,1)`.

---

# 16. B pair conditional

For B with \(M\) layers:

\[
g_B(t)=\binom{M}{t},
\qquad
0\le t\le M.
\]

Given two-cell sum \(q\),

\[
P(k|q)
\propto
\binom{M}{k}
\binom{M}{q-k}.
\]

This is hypergeometric:

\[
k\sim \mathrm{Hypergeometric}(2M,M,q)
\]

with support additionally restricted by positivity:

\[
\max(1,q-M)
\le k\le
\min(M,q-1).
\]

Requirements:

- never propose occupations above \(M\);
- handle near-saturation cases without rejection explosions;
- use an exact or numerically stable hypergeometric sampler.

---

# 17. W pair conditional

For W with \(M\) layers:

\[
g_W(t)=\binom{M+t-1}{t}.
\]

For two-cell sum \(q\):

\[
P(k|q)
\propto
\binom{M+k-1}{k}
\binom{M+q-k-1}{q-k}.
\]

This is a beta-binomial split:

\[
k\sim \mathrm{BetaBinomial}(q,M,M)
\]

conditioned on:

\[
1\le k\le q-1.
\]

One valid generation strategy is:

\[
p\sim \mathrm{Beta}(M,M),
\]

then:

\[
k\sim \mathrm{Binomial}(q,p),
\]

with an appropriate bounded treatment of forbidden endpoints.

For `M=1`, the conditional simplifies strongly and should be implemented explicitly if that is cheaper and numerically safer.

Do not construct an `O(q)` probability vector for every proposal.

---

# 18. Fixed-total chain semantics

Suggested production chain:

```rust
struct FixedTotalChain {
    state: FixedTotalState,
    family: OccupationFamily,
    config: McmcConfig,
    counters: McmcCounters,
}
```

A step should:

```text
choose two indices
sample exact conditional split
write two occupations
increment counters
```

No heap allocation in the step.

A sweep should scale with number of occupied pairs, e.g.:

```text
E pair updates per sweep
```

The final number must be calibrated using mixing benchmarks.

Persistent chains should be supported.

---

# 19. Fixed `(k,T)` — new production implementation

Fixed `(k,T)` should use:

```text
binary degree-constrained support sampler
+
same fixed-total occupation sampler
```

Never duplicate occupation logic.

The support stage samples:

\[
A\sim P(A\mid k).
\]

Then:

\[
\mathbf t\sim P_F(\mathbf t\mid E(A),T).
\]

Since the degree sequence determines the edge count for a directed network:

\[
E=\sum_i k_i^{out}=\sum_i k_i^{in}.
\]

---

# 20. Fixed-degree binary support state

Create a sparse binary support state.

It should support expected `O(1)` operations for:

- edge existence check,
- insert edge,
- remove edge,
- random occupied edge,
- degree lookup,
- mask/admissibility test.

Suggested components:

```text
Vec<Pair> occupied_edges
HashMap<Pair, slot> or equivalent sparse lookup
optional per-node adjacency indexes if justified
degree arrays
```

Do not allocate dense adjacency.

If hash lookup becomes a throughput bottleneck, benchmark alternatives before optimizing.

---

# 21. Fixed-degree initialization

The initializer must produce a feasible binary realization of the target degree sequences.

Reuse an existing scalable realization algorithm if one already exists and is correct.

Requirements:

- directed in/out sequence exact,
- no duplicate edge,
- no forbidden self-loop,
- mask respected,
- fixed pairs respected,
- bounded runtime,
- structured failure.

Do not use exhaustive search in production.

Do not use an unbounded retry loop.

Pathological feasible sequences may require specialized handling; if initialization cannot reliably solve them yet, return a clear error rather than hang.

---

# 22. Fixed-degree binary switch MCMC

Basic directed switch:

existing edges:

\[
(a,b),\quad(c,d)
\]

propose:

\[
(a,d),\quad(c,b).
\]

This preserves:

- out-degree of `a`,
- out-degree of `c`,
- in-degree of `b`,
- in-degree of `d`.

Reject/hold if:

- proposed edge already exists,
- proposed edge violates mask,
- proposed edge is an illegal self-loop,
- fixed pair would be removed or changed,
- move is a no-op.

Proposal probabilities must be proven symmetric or an explicit Hastings ratio must be included.

Do not simply assume “degree preserving” means unbiased.

---

# 23. Binary-chain mixing

Acceptance rate alone is insufficient.

During validation record autocorrelation or convergence of support observables such as:

- edge overlap with initial support,
- reciprocity,
- simple motif count,
- degree-degree endpoint correlation,
- any existing cheap support statistic.

Use multiple dispersed initial realizations when possible.

The production defaults for burn-in/thinning must be derived from these measurements.

---

# 24. Coupled occupation-number MCMC

The following cases should use one common sparse occupation-number MCMC architecture:

- fixed `s`,
- fixed `(E,s)`,
- fixed `s,<C>`.

They differ primarily in:

- hard invariant checks,
- proposal mixture,
- target terms,
- initialization.

They should not each define an independent chain framework.

---

# 25. Occupation MCMC state

The state must be sparse and incrementally maintained.

At minimum:

```text
node count
occupied pairs
occupation values
pair -> slot/value lookup
out-strength array
in-strength array
edge count E
total occupation T
```

For cost-aware runs, maintain cached total cost if useful.

A local proposal must update these cached statistics incrementally.

Never recompute full strengths after each move.

Never count occupied edges by scanning all pairs after each move.

---

# 26. Proposal representation

Current fixed-strength code should be refactored to avoid heap allocation in the hot loop.

Use a compact fixed-size proposal representation.

Conceptually:

```rust
struct LocalChange {
    pair: Pair,
    old: OccNum,
    new: OccNum,
}

struct Proposal<const K: usize> {
    changes: [LocalChange; K],
    used: usize,
    log_hastings: f64,
}
```

or equivalent.

For a 4-cycle there are at most four affected pairs before overlap merging.

Handle repeated coordinates explicitly with small stack logic.

Do not allocate a `HashMap` or `Vec` per proposal.

---

# 27. Fixed-strength move

The primary move remains a 4-cycle / rectangle update.

Select two source nodes:

\[
a\neq c
\]

and two target nodes:

\[
b\neq d.
\]

Choose sign:

\[
\sigma\in\{-1,+1\}.
\]

Apply:

\[
t_{ab}'=t_{ab}+\sigma,
\]

\[
t_{cd}'=t_{cd}+\sigma,
\]

\[
t_{ad}'=t_{ad}-\sigma,
\]

\[
t_{cb}'=t_{cb}-\sigma.
\]

Each source row and target column receives one `+sigma` and one `-sigma`.

Therefore out- and in-strengths are preserved.

Feasibility checks:

- no occupation below zero,
- B occupation never above \(M\),
- forbidden pair never becomes occupied,
- fixed-pair occupations unchanged,
- no invalid self-loop.

Target ratio is computed only from affected cells.

For no-cost fixed-strength:

\[
\Delta\log\pi
=
\sum_{\text{changed pairs}}
\Delta\log g_F.
\]

For expected cost:

\[
\Delta\log\pi
=
\sum
\Delta\log g_F
-
\gamma\Delta C.
\]

---

# 28. Improving fixed-strength proposal selection

The existing node-uniform rectangle proposal may generate many infeasible/held moves in sparse systems.

After the correctness-preserving refactor, benchmark alternative proposal strategies.

Possible scalable variants:

1. **node-uniform rectangle**
   - current conceptual method;
   - cheap;
   - many zero/invalid combinations possible.

2. **occupied-edge-seeded rectangle**
   - choose one or two occupied edges first;
   - derive candidate rectangle;
   - potentially much higher valid proposal rate.

3. **mixture kernel**
   - combine global node-uniform moves for ergodicity with occupied-edge-informed moves for local efficiency.

Any non-symmetric proposal must include the correct Hastings factor.

Do not optimize this before the clean fixed-size proposal representation exists.

---

# 29. Fixed `(E,s)` move design

Fixed `(E,s)` is genuinely coupled.

A naive strength-preserving 4-cycle can change support size when cells cross zero.

The final chain must preserve both:

\[
s=s^\ast
\]

and:

\[
E=E^\ast.
\]

Before implementation, create an internal design note containing the exact move equations.

Candidate move strategies include:

## 29.1 Support-preserving cycle moves

Only accept cycle changes where the number of zero-to-positive transitions equals the number of positive-to-zero transitions.

For each local proposal compute:

```text
delta_E =
count(old=0,new>0)
-
count(old>0,new=0)
```

Require:

```text
delta_E == 0
```

This is simple and should be the first implementation candidate.

However, this may mix slowly because many cycle proposals will be held.

## 29.2 Dedicated support-exchange cycle

Construct proposals deliberately containing one support activation and one support deletion while preserving strengths.

This may improve mixing but proposal probabilities are more complex.

Implement only after the simple constrained-cycle version is validated and benchmarked.

## 29.3 Kernel mixture

A future production version may mix:

- support-preserving occupation redistribution moves,
- support-changing but `E`-neutral moves.

The chain must document empirical evidence that tested fibers are connected/mixing.

Do not claim general ergodicity without justification.

---

# 30. Fixed `(E,s)` initializer

This is likely the hardest initializer in the current scope.

It must construct an occupation table satisfying:

\[
s^{out}=s^{out,\ast},
\]

\[
s^{in}=s^{in,\ast},
\]

\[
|\{(i,j):t_{ij}>0\}|=E^\ast.
\]

The production initializer must not depend on the original observed network.

Possible staged approach:

1. construct a feasible strength table with a transportation-style algorithm;
2. measure resulting support size;
3. apply deterministic/stochastic support-adjustment moves preserving strengths until target \(E\) is reached;
4. use bounded iteration count;
5. if unsuccessful, report structured initialization failure.

An annealed repair process is acceptable if:

- runtime is bounded,
- feasibility is always maintained after the first feasible strength table,
- diagnostic progress is measurable,
- failure is explicit.

Never use an infinite or uncontrolled stochastic repair loop.

This initializer must receive substantially more testing than the simpler cases.

---

# 31. Expected-cost gamma fitting

The gamma fitter should operate on a persistent fixed-strength chain.

Do not reconstruct the whole chain for every gamma evaluation.

Conceptual iteration:

```text
set gamma
run adaptation/burn-in sweeps
collect cost samples
estimate mean cost
estimate uncertainty / ESS
update gamma bracket
repeat
```

Preferred root strategy:

- bracket target,
- stochastic bisection or safeguarded secant,
- reuse previous chain state as warm state,
- stop based on both residual and Monte Carlo uncertainty.

Do not rely on a single noisy variance estimate to create a large Newton-like step.

The current large-N instability is likely due in part to variance estimation becoming poor relative to total cost scale.

Use:

- batch means,
- autocorrelation-aware ESS,
- multiple cost samples per gamma,
- bounded gamma steps if needed for numerical safety.

A clamp may be used as a safety device, but not as the main convergence strategy.

---

# 32. MCMC common engine

The common `mcmc` module should remain intentionally small.

It should contain generic concepts:

```text
McmcConfig
McmcCounters
McmcOutcome
accept/reject helper
diagnostics
```

It should not become a heavy generic trait framework.

Avoid runtime polymorphism in hot loops unless benchmarking proves it harmless.

Prefer concrete chains that reuse common helper types.

The common acceptance helper should accept:

```text
delta_log_target
log_hastings
random uniform
```

and implement:

\[
\log \alpha
=
\min(0,\Delta\log\pi + \log q(y\to x)-\log q(x\to y)).
\]

---

# 33. Production-code scalability policy

After this refactor, production code may contain only algorithms intended to scale beyond marginal small-N cases.

The following are **not allowed in production generation paths**:

- exact state enumeration,
- exact enumeration of positive compositions,
- dynamic programming with memory \(O(E T)\),
- transition-matrix construction,
- exhaustive binary graph enumeration,
- brute-force rejection whose acceptance decays exponentially with \(E\),
- small-N exact conditional samplers implemented through state-list enumeration,
- fallback exact methods selected automatically for small N.

Even if such methods are faster at `N=10`, they belong in reference tests, not production.

The production API must not select algorithms based on:

```text
if N < 50 use exact
else use MCMC
```

unless the exact algorithm is independently scalable in the relevant dimensions.

The only acceptable production special cases are algorithms that remain asymptotically sound and are genuinely useful at scale, e.g.:

- direct uniform fixed-E support sampling,
- efficient direct family pair distributions,
- trivial deterministic cases such as `T=E`,
- mathematically degenerate/saturated cases that can be solved in `O(E)`.

---

# 34. Test architecture

Rust tests must be separated into two categories.

## 34.1 Normal Rust tests

Fast tests belong in the normal crate test structure:

```text
crates/menobis-core/src/... #[cfg(test)]
crates/menobis-core/tests/
```

These tests should be suitable for normal CI.

Examples:

- unit formulas,
- proposal invariant checks,
- property-based tests,
- small deterministic state updates,
- API behavior,
- reproducibility,
- no-negative-occupation checks,
- B capacity,
- exact hard constraints.

Target runtime should remain reasonable.

## 34.2 Heavy reference / exact test crate

Heavy tests must live in a separate crate.

Create:

```text
crates/menobis-test-oracles/
```

This crate must be development/test infrastructure only.

It must **not** be a normal production dependency of `menobis-core`.

Recommended structure:

```text
crates/menobis-test-oracles/

    Cargo.toml

    src/
        lib.rs

        enumeration/
            mod.rs
            occupations.rs
            binary_graphs.rs
            strength_tables.rs

        exact/
            mod.rs
            fixed_et.rs
            fixed_kt.rs
            fixed_strength.rs

        transition/
            mod.rs
            detailed_balance.rs
            transition_matrix.rs

        statistics/
            mod.rs
            compare.rs

    tests/
        fixed_total_stationarity.rs
        fixed_et_distribution.rs
        fixed_kt_distribution.rs
        fixed_strength_distribution.rs
        fixed_es_distribution.rs
```

This crate may contain intentionally expensive code such as:

- exact state enumeration,
- exact fiber probabilities,
- DP reference algorithms,
- old exact samplers,
- transition matrices,
- long Monte Carlo comparison tests,
- multi-chain mixing validation,
- exact small-N graph enumeration.

These tests should run through explicit commands or dedicated CI jobs, not every fast local `cargo test` invocation if runtime becomes large.

Provide commands such as:

```bash
cargo test -p menobis-test-oracles
```

and, if useful, ignored heavy tests:

```bash
cargo test -p menobis-test-oracles -- --ignored
```

Document which tests are expected to be slow.

---

# 35. Legacy exact-code migration

Do not immediately delete current exact implementations.

Use this sequence:

```text
current exact production implementation
        |
        +-- validate new scalable backend against it
        |
        +-- move exact logic into menobis-test-oracles
        |
        +-- rerun distribution comparison
        |
        +-- remove exact implementation from production crate
```

Exact DP and reference algorithms move **permanently** into
`menobis-test-oracles`, where they serve as exact oracles for
validating the scalable backends at larger sizes.  No separate
archive branch or tag is created: git commit history already
preserves the old production code.

The final `menobis-core` generation modules must contain no exact
small-N sampler merely as a fallback.

---

# 36. Statistical correctness requirements

Constraint preservation alone is insufficient.

Every stochastic replacement must pass two distinct classes of validation.

## 36.1 Invariant validation

Examples:

```text
fixed E,T:
    E exact
    T exact

fixed k,T:
    k_out exact
    k_in exact
    T exact

fixed s:
    s_out exact
    s_in exact

fixed E,s:
    E exact
    s exact

fixed s,<C>:
    s exact
    expected sampled C statistically matches target
```

Also verify:

- nonnegative occupations,
- B capacity,
- masks,
- fixed pairs,
- self-loop policy.

## 36.2 Distribution validation

For small fibers use exact enumeration.

Compare:

- empirical state probabilities,
- total variation distance,
- scalar observable means,
- variances,
- quantiles,
- empirical CDFs.

Useful occupation observables:

\[
\max_e t_e,
\]

\[
\sum_e t_e^2,
\]

plus:

```text
occupation variance
fraction occupation=1
occupation entropy
family log degeneracy
support size
```

Useful support observables:

```text
reciprocity
edge overlap
motif counts
endpoint degree correlation
```

Do not use only KS p-values as a pass/fail criterion.

Use effect sizes and Monte Carlo error.

---

# 37. Detailed balance tests

For small explicitly enumerable states, verify detailed balance directly when possible.

For states \(x,y\):

\[
\pi(x)P(x\to y)
\approx
\pi(y)P(y\to x).
\]

This is especially important for:

- asymmetric support proposals,
- informed sparse fixed-strength proposals,
- future fixed `(E,s)` support-changing kernels.

Pair-Gibbs fixed-total updates should match the exact conditional by construction; validate the conditional sampler directly.

---

# 38. Burn-in and thinning

Do not choose universal values merely because they are convenient.

A sweep should represent an amount of work proportional to state size.

Initial definitions:

```text
fixed-total occupation:
    E pair-Gibbs updates per sweep

binary fixed-degree:
    E switch attempts per sweep

fixed-strength occupation:
    max(E, 2N) or similar local proposals per sweep
```

These are starting points only.

Calibrate burn-in and sampling intervals using:

- autocorrelation,
- ESS,
- between-chain comparison,
- convergence of key observables.

Final defaults must be justified in benchmark documentation.

---

# 39. Synthetic-data validation pipeline

Continue to use the repository synthetic generator.

For all nontrivial integration tests:

1. generate a realistic directed non-binary network;
2. derive constraints from the generated network;
3. construct the sampler from those constraints;
4. sample;
5. verify exact constraints or expected observables.

Do not invent arbitrary large constraint arrays unless the test is explicitly a mathematical unit test.

This guarantees feasibility.

---

# 40. Correctness matrix

Use a small-system correctness matrix.

Recommended:

```text
N = 10, 25, 50, 100

families:
    ME
    B
    W

regimes:
    sparse
    dense
    saturated where mathematically valid

fixed-pair fractions:
    0%
    5%

self loops:
    enabled
    disabled where supported
```

Not every Cartesian combination must run if mathematically redundant, but each implementation path must have broad coverage.

Small-N exact distribution comparison belongs in `menobis-test-oracles`.

---

# 41. Performance benchmark matrix

Performance benchmarks must distinguish correctness scale from production scale.

## 41.1 Medium scale

```text
N = 500
N = 1000
N = 5000
```

Run all production samplers expected to scale.

## 41.2 Large scale

```text
N = 10000
N = 25000
```

Use realistic sparse scenarios.

The objective is to prove:

- no hidden dense memory allocation,
- no DP wall,
- no exact-enumeration fallback,
- no exponential rejection wall,
- sparse states remain manageable,
- hot loops scale with `E` rather than `N^2` when mathematically possible.

Dense `N=25000` microcanonical networks are not required if the model itself entails prohibitively many pairs.

---

# 42. Benchmark metrics

Every benchmark record should contain:

```text
commit SHA
branch
timestamp
family
ensemble
constraint
sampling plan
N
E
T
T/E
layers M
self-loop setting
fixed-pair fraction
synthetic regime
seed
burn-in sweeps
sweeps/sample
sample count

initialization time
burn-in time
sampling time
total wall time
peak memory

proposal count
accepted count
held count
Metropolis rejected count
updates/sec
sweeps/sec

ESS for selected observables
ESS/sec

constraint recovery
```

For factorized cases record stage times separately:

```text
support sampling time
occupation initialization time
occupation burn-in time
occupation sampling time
```

For expected-cost:

```text
gamma
gamma iterations
gamma-fit wall time
target cost
mean sampled cost
cost standard error
cost ESS
```

---

# 43. Performance gates

Do not accept “it runs” as scalability evidence.

## Fixed-total gate

Demonstrate approximately linear memory in `E`.

Benchmark directly at:

```text
E = 1e3
E = 1e4
E = 1e5
```

for:

```text
T/E ~ 1.1
T/E ~ 2
T/E ~ 8
```

and B near saturation.

## Fixed `(E,T)` gate

At minimum:

```text
N = 1000 sparse
N = 5000 sparse
N = 10000 sparse
N = 25000 sparse representative case
```

must complete without dense allocation.

## Fixed `(k,T)` gate

At minimum:

```text
N = 1000
N = 5000
```

for realistic sparse degree sequences.

Attempt `10000` and `25000` after mixing and initializer behavior are stable.

## Fixed-strength gate

At minimum:

```text
N = 1000
N = 5000
```

must be demonstrated.

Continue to `10000` and `25000` if proposal throughput and mixing remain practical.

## Strength-cost gate

Required progression:

```text
N = 100
500
1000
5000
```

Do not claim large-N readiness until gamma fitting remains statistically stable.

---

# 44. Phase plan

The project must be executed in the following order.

---

# Phase 0 — create integration branch and freeze baseline

Branch:

```text
microcanonical-refactor
```

Tasks:

- record master SHA,
- run current tests,
- run representative existing benchmarks,
- record known limitations,
- inventory production microcanonical code,
- inventory exact/DP/rejection code,
- inventory current Python/Rust routing.

Create a baseline report under development docs or benchmark results.

## Gate 0

Required:

- integration branch exists,
- baseline SHA recorded,
- current benchmark output saved,
- current failures documented,
- no production behavior changed.

---

# Phase 1 — common family and problem primitives

Subbranch:

```text
refactor/model-primitives
```

Tasks:

- centralize family occupation degeneracy,
- centralize hard/soft terminology,
- define prepared problem,
- unify pair-domain semantics,
- migrate callers without algorithm changes.

Tests:

- ME/B/W local ratios,
- B occupation cap,
- W stability,
- prepared mask behavior,
- fixed-pair residualization.

## Gate 1

Run:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

No material performance regression in unchanged samplers.

Merge to `microcanonical-refactor`.

---

# Phase 2 — sampling-plan routing

Subbranch:

```text
refactor/sampling-plan
```

Introduce internal classification:

```text
GC
fixed E,T factorized
fixed k,T factorized
fixed s occupation-MCMC
fixed s,<C> occupation-MCMC + soft cost
fixed E,s occupation-MCMC
```

Keep existing backends temporarily.

## Gate 2

All current public supported routes still function.

No duplicate compatibility APIs.

Merge.

---

# Phase 3 — create heavy oracle crate

Subbranch:

```text
refactor/test-oracles
```

Create:

```text
crates/menobis-test-oracles/
```

Move or copy reusable exact algorithms into it.

At this stage production code may still contain old exact paths until migration is validated.

Implement reusable:

- positive-composition enumeration,
- exact ME/B/W probabilities,
- binary support enumeration,
- exact observable comparison helpers,
- transition/detailed-balance helpers.

## Gate 3

Oracle crate independently validates at least one current exact production sampler.

Normal core tests remain fast.

Merge.

---

# Phase 4 — scalable fixed-total occupation sampler

Subbranch:

```text
refactor/fixed-total
```

Implement:

```text
conditional/fixed_total/
```

including:

- state,
- O(E) initializer,
- ME pair conditional,
- B pair conditional,
- W pair conditional,
- pair-Gibbs chain,
- diagnostics.

Fast tests in core:

- exact total,
- positivity,
- B cap,
- reproducibility,
- pair conditional support.

Heavy tests in oracle crate:

- exact composition enumeration,
- empirical stationary comparison,
- total variation / observable comparison.

Performance benchmarks:

```text
E=1e3,1e4,1e5
```

## Gate 4

Must pass:

- exact invariants,
- small-fiber distribution,
- no `O(ET)` memory,
- no exponential rejection loop,
- near-linear memory scaling.

Merge.

---

# Phase 5 — migrate fixed `(E,T)`

Subbranch:

```text
refactor/fixed-et
```

Implement final factorized path:

```text
residualize
sample E-edge support
sample fixed-total occupations
assemble network
```

Use direct support sampling when uniform.

Heavy validation against old exact implementation:

```text
N=10,25,50,100
```

Benchmark:

```text
N=100,500,1000,5000,10000,25000
```

with sparse representative high-N cases.

## Gate 5

Required:

- exact E,T,
- family law validated,
- W scaling wall removed,
- sparse N=25000 representative run,
- new backend becomes production default.

After gate passes:

- migrate old fixed-ET exact code fully into oracle crate,
- remove it from `menobis-core`.

No legacy exact production backend remains.

Merge.

---

# Phase 6 — binary fixed-degree support sampler

Subbranch:

```text
refactor/binary-degree
```

Implement:

- sparse support state,
- scalable degree realization,
- directed switch chain,
- diagnostics,
- no dense adjacency.

Fast tests:

- exact degree preservation,
- support legality,
- duplicates impossible,
- mask,
- self loops,
- fixed pairs.

Heavy oracle tests:

- exact small graph enumeration,
- stationary support comparison.

Benchmark:

```text
N=100,500,1000,5000
```

and larger if stable.

## Gate 6

Required:

- exact degrees,
- validated stationary support law,
- O(N+E) memory,
- no proposal scan over all pairs.

Merge.

---

# Phase 7 — migrate fixed `(k,T)`

Subbranch:

```text
refactor/fixed-kt
```

Compose:

```text
fixed-degree binary support chain
+
existing new fixed-total occupation chain
```

No new occupation implementation allowed.

Validate support and occupation components separately and jointly.

Heavy comparison against old backend for small N.

Benchmark through large sparse systems.

## Gate 7

Required:

- exact k,T,
- distribution comparison,
- no DP/rejection production path,
- no duplicated occupation code.

Then remove old fixed-KT exact production algorithms from core and preserve only in oracle crate.

Merge.

---

# Phase 8 — refactor fixed-strength occupation chain

Subbranch:

```text
refactor/fixed-strength
```

Tasks:

- create common occupation MCMC state,
- migrate current cycle move,
- eliminate per-step allocations,
- centralize target ratio,
- separate initialization from chain,
- remove special backend routing from chain.

Evaluate whether the current ME direct stub sampler deserves to remain in production.

Decision rule:

> It remains only if it is demonstrably scalable, materially faster at production-relevant N, and does not complicate architecture.

If it is mostly a small-N exact convenience, move it to oracle crate.

Benchmark node-uniform vs optional occupied-edge-informed proposals.

## Gate 8

Required:

- exact strengths,
- ME/B/W distribution validated,
- proposal throughput not worse than current,
- allocation-free hot path,
- sparse N=1000 and N=5000 demonstrated.

Merge.

---

# Phase 9 — refactor fixed-strength expected cost

Subbranch:

```text
refactor/strength-cost
```

Compose:

```text
fixed-strength occupation chain
+
cost observable
+
gamma target
+
gamma fitter
```

Do not create a separate chain framework.

Improve gamma fitting using persistent chain reuse and uncertainty-aware root finding.

Validate with synthetic target cost.

## Gate 9

Required:

- strengths exact,
- sampled mean cost matches target within Monte Carlo uncertainty,
- gamma convergence diagnostics meaningful,
- stable progression to at least N=1000 and preferably N=5000.

Merge.

---

# Phase 10 — fixed `(E,s)`

Subbranch:

```text
refactor/fixed-es
```

Before code, create a design note with:

- exact move equations,
- proposal probabilities,
- hard invariant proof,
- initializer strategy,
- expected connectivity limitations.

First implementation should preferably use constrained strength-preserving cycle moves requiring:

```text
delta_E == 0
```

Then benchmark whether specialized support-exchange moves are necessary.

Heavy exact small-N validation is mandatory.

## Gate 10

Required:

- exact E and strengths,
- detailed-balance validation,
- bounded initializer,
- multi-chain convergence on tested fibers,
- meaningful scaling beyond small-N validation.

If the sampler remains only practical for tiny N, do not promote it to normal production support.

Merge only when it meets the scalable-production criterion.

---

# Phase 11 — production cleanup

Subbranch:

```text
refactor/production-cleanup
```

Remove from `menobis-core`:

- exact fixed-total samplers,
- DP occupation tables,
- brute-force rejection,
- exact graph enumeration,
- exact transition construction,
- compare/legacy production backends,
- small-N production fallbacks.

Ensure all retained exact functionality exists only in `menobis-test-oracles`.

Simplify directory structure.

## Gate 11

Inspect production tree manually.

No exact small-N generation backend should remain.

Run full tests and benchmarks.

Merge.

---

# Phase 12 — benchmark and documentation consolidation

Subbranch:

```text
refactor/final-benchmarks-docs
```

Update benchmark CLI/results to expose:

- sampling plan,
- stage timings,
- ESS/sec,
- memory,
- constraint recovery.

Run final matrix.

Update:

```text
docs/decisions/generation-factorization.md
docs/development/scalability.md
docs/concepts/microcanonical.md
docs/api/...
```

Document which samplers reach which tested scales.

Do not claim untested scale.

## Gate 12

Run full project checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
mkdocs build --strict
```

Also run:

```bash
cargo test -p menobis-test-oracles
```

and explicitly invoke ignored heavy oracle tests when required for release validation.

---

# 45. Final global gate before merge to master

`microcanonical-refactor` may merge into `master` only if all conditions below hold.

## Architecture

- GC remains direct/factorized.
- Fixed `(E,T)` uses binary support + shared fixed-total occupation.
- Fixed `(k,T)` uses fixed-degree support + the same shared occupation sampler.
- Fixed `s` uses common occupation MCMC.
- Fixed `s,<C>` composes cost fitting with that chain.
- Fixed `(E,s)` uses coupled occupation MCMC if enabled.
- no duplicated ME/B/W microcanonical degeneracy logic.
- no one-directory-per-constraint mini-framework architecture.

## Production cleanliness

`menobis-core` contains only scalable sampling algorithms.

No production code remains for:

- exact enumeration,
- small-N DP,
- exact composition enumeration,
- exponentially bad rejection,
- state-space transition matrices.

All such functionality lives in:

```text
crates/menobis-test-oracles/
```

## Correctness

Every production MCMC sampler has:

- invariant tests,
- distribution tests,
- small-N reference comparison,
- reproducibility tests,
- mask/fixed-pair tests.

## Scalability

At least representative sparse cases demonstrate:

```text
N >= 1000
```

comfortably.

At least selected production routes demonstrate:

```text
N = 10000
```

and representative sparse factorized cases should target:

```text
N = 25000.
```

No hidden dense-memory behavior appears.

## Benchmarking

Stored benchmark output includes:

- baseline old implementation,
- final new implementation,
- wall time,
- memory,
- ESS where applicable,
- stage diagnostics,
- exact constraint recovery.

---

# 46. Anti-patterns

The implementation agent must stop and reconsider if it starts doing any of the following:

```text
creating fixed_et_v2/, fixed_kt_v2/, fixed_strength_v2/ parallel frameworks

making grand canonical use MCMC

copying occupation formulas between constraint modules

duplicating fixed-total logic

keeping exact code in production because it is convenient for N=20

selecting exact-vs-MCMC based only on N threshold

allocating N x N matrices

allocating O(E*T) tables

retrying rejection until lucky

using Python loops for chain updates

recomputing global strengths every proposal

allocating HashMap/Vec for every local four-cell proposal

adding legacy API compatibility wrappers

leaving old production backends behind after migration
```

---

# 47. Decision framework for future constraint combinations

When a new ensemble is added, answer these questions in order:

1. **What are the hard invariants?**
2. **What observables are constrained only in expectation?**
3. **Can the probability law factorize conditionally?**
4. **If yes, what are the conditional stages?**
5. **Can any stage be sampled directly instead of with MCMC?**
6. **If a coupled occupation fiber remains, what local moves preserve its hard invariants?**
7. **Is the proposal symmetric?**
8. **If not, what is the Hastings ratio?**
9. **Can a feasible initial state be constructed in bounded scalable time?**
10. **How will the sampler be validated against an exact small-N oracle?**
11. **What benchmark demonstrates that the new production algorithm scales?**

Only after answering these should new sampler code be written.

---

# 48. Final target architecture

```text
                     constraint specification
                              |
                              v
                       prepare problem
                              |
                              v
                     choose sampling plan
                              |
          +-------------------+--------------------+
          |                   |                    |
          v                   v                    v
 grand canonical       factorized MC      occupation-number MCMC
          |                   |                    |
 family solver          conditional stage 1       initializer
          |                   |                    |
 pair distributions     conditional stage 2       local move kernel
          |                   |                    |
 direct sampling        ...                  MH / Gibbs acceptance
          |                   |                    |
          +-------------------+--------------------+
                              |
                              v
                       SampledNetwork
```

Current factorized cases:

```text
fixed E,T:
    uniform/direct fixed-E binary support
    ->
    fixed-total occupation Gibbs chain

fixed k,T:
    fixed-degree binary support MCMC
    ->
    same fixed-total occupation Gibbs chain
```

Current coupled cases:

```text
fixed s:
    strength-preserving occupation MCMC

fixed E,s:
    strength- and edge-count-preserving occupation MCMC

fixed s,<C>:
    strength-preserving occupation MCMC
    +
    fitted cost bias
```

---

# 49. Core design statement

The final implementation should make the following principle obvious from the code structure:

> **MENoBiS samples non-binary networks through occupation numbers. Binary support is a derived layer. Whenever the ensemble factorizes, sampling is decomposed into exact conditional stages. Whenever factorization fails, a sparse local MCMC is used on the coupled occupation-number fiber. Exact small-system algorithms are validation oracles, not production backends.**

This principle should guide every implementation decision in this refactor.
