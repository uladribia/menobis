# MENoBiS implementation plan: exact microcanonical fixed strengths + exact edge count `(s, E)`

## Purpose of this document

This document is the complete implementation specification for adding the **microcanonical fixed-strength + fixed-edge-count sampler** to MENoBiS.

It is written for an implementation agent starting with a **clean session** and access to the repository at `master`.

The agent must not need any prior conversation, prior plan, or previous feature branch to execute this work.

---

# 0. Non-negotiable instructions

## 0.1 Start point

Start from `master`.

```bash
git checkout master
git pull --ff-only
git checkout -b feature/microcanonical-fixed-strength-edges
```

Do **not** merge, rebase, or cherry-pick any previous `fixed-sE`, `fixed-strength-edges`, or similar feature branch.

Do not inspect an old implementation and copy its architecture. The implementation described here deliberately starts from the mathematically validated fixed-strength machinery already present on `master`.

If an old branch exists, ignore it.

## 0.2 Follow repository instructions

Read `AGENTS.md` before editing.

In particular:

- Rust contains the heavy algorithms.
- Python is only routing/orchestration/data preparation.
- Keep public API surface minimal.
- No backwards-compatibility shims.
- Keep changes small and targeted.
- Use TDD.
- Use the repository's required commit skill/workflow.
- Run the repository checks before final handoff.

## 0.3 Scope

Implement only:

> exact microcanonical sampling with fixed out-strength sequence, fixed in-strength sequence, and exact total number of occupied ordered pairs `E`.

Support the three existing occupation families:

- `ME`
- `B { layers: M }`
- `W { layers: M }`

Reuse all applicable fixed-strength behavior from `master`:

- fixed pairs;
- self-loop policy;
- family support/capacity;
- sparse state;
- compressed construction;
- structural repair;
- fixed-strength family degeneracy;
- occupied-cell 4-cycle proposal;
- exact Hastings correction;
- existing routing philosophy.

Do **not** implement:

- fixed degree sequence;
- fixed `(s, degree)`;
- cost + fixed `E`;
- a generic Markov-basis framework;
- arbitrary long-cycle proposal frameworks;
- a new family-specific sampler;
- a second independent 4-cycle Hastings implementation;
- a generic constraint trait hierarchy.

The implementation should be a **small extension of the existing fixed-strength MCMC**.

## 0.4 Mathematical standard

The implementation must not be exposed as supported until all exact small-state tests in this document pass.

Monte Carlo histograms are **not** sufficient proof of correctness.

The mandatory mathematical verification is an **exact enumerated transition-matrix oracle** on tiny fibers.

## 0.5 Scalability standard

The production implementation must remain sparse.

For complete domains with a small number of fixed pairs:

- no `N x N` matrix;
- no `HashSet` containing all admissible pairs;
- no enumeration of all ordered pairs in the normal production path;
- state memory must remain `O(N + E + F)`, where:
  - `E` is the number of occupied residual pairs;
  - `F` is the number of fixed/excluded pairs.

The required practical scalability gate is `N = 1000`.

A `N = 5000` sparse smoke benchmark is recommended after the required gate passes.

`N = 25000` is a useful stretch benchmark, not a merge blocker for this feature.

---

# 1. Read these files first

Before modifying code, inspect the current implementation on `master`.

At minimum read:

```text
AGENTS.md

crates/menobis-core/src/model/family.rs

crates/menobis-core/src/generation/microcanonical/occupation_mcmc/
    chain.rs
    compressed.rs
    domain.rs
    errors.rs
    initializer.rs
    move_cycle.rs
    problem.rs
    rectangle.rs
    repair.rs
    state.rs
    target.rs
    mod.rs

crates/menobis-python/src/generation.rs

src/menobis/
    capabilities.py
    routing.py
    models/spec.py
    models/types.py
```

Also search the repository for tests of:

- `occupied_cycle4_step`;
- fixed-strength residualization;
- fixed pairs;
- fixed-strength stationarity/Hastings;
- microcanonical routing;
- capability registry.

Do not assume file names beyond those listed above if the repository has changed. Search the actual code.

---

# 2. Scientific target

Let `t_ij` denote the integer occupation of ordered pair `(i,j)`.

Let:

```text
s_out[i] = sum_j t_ij
s_in[j]  = sum_i t_ij
E(t)     = number of admissible ordered pairs with t_ij > 0
```

The target state space is:

```text
F_(s,E) = {
    t:
      exact out-strengths = s_out,
      exact in-strengths  = s_in,
      exact occupied-pair count = E_target,
      every pair is admissible,
      family occupation support is respected
}
```

The probability of a state in that fiber is proportional to the family base measure already implemented in `model/family.rs`.

For the variable/residual pairs:

### ME

```text
d_ME(t) = 1 / t!
```

Therefore:

```text
pi_ME(t | s,E) proportional to product_(i,j) 1 / t_ij!
```

### B with `M` layers

```text
d_B(t) = C(M,t),  0 <= t <= M
```

Therefore:

```text
pi_B(t | s,E) proportional to product_(i,j) C(M,t_ij)
```

### W with `M` layers

```text
d_W(t) = C(M+t-1,t)
```

Therefore:

```text
pi_W(t | s,E) proportional to product_(i,j) C(M+t_ij-1,t_ij)
```

Do not duplicate these formulas in production code.

Production acceptance must continue to use the existing:

```rust
OccupationFamily
StrengthTarget
delta_log_weight(...)
```

The explicit formulas above are used only to define the mathematical target and to build an **independent test oracle**.

---

# 3. Fixed pairs and the residual target

Fixed-pair handling is part of the mathematical problem, not a post-processing convenience.

Suppose a full problem contains fixed pairs:

```text
(i, j, occupation)
```

A fixed pair with positive occupation contributes:

```text
1
```

to the full edge count.

A fixed pair with zero occupation contributes:

```text
0
```

to the full edge count, but that coordinate must still remain frozen at zero and therefore must be excluded from the residual domain.

Define:

```text
E_fixed = number of UNIQUE fixed pairs with occupation > 0
E_residual_target = E_target - E_fixed
```

The residual sampler must sample:

```text
F_(s_residual, E_residual_target)
```

on a domain from which **every fixed coordinate is excluded**, including zero-fixed coordinates.

The family-weight contribution of fixed pairs is constant across residual states, so it cancels from every probability ratio. Therefore using only the residual family base measure is mathematically correct.

### Mandatory validation

Reject:

- duplicate fixed coordinates;
- fixed coordinates outside the domain;
- fixed occupation above family capacity;
- fixed occupation above the associated strengths;
- `E_target < E_fixed`;
- an impossible residual target.

Do not subtract fixed pairs in Python and then sample over a complete domain.

The residualization must occur exactly once in Rust.

---

# 4. Overall architecture

The final sampler is:

```text
full problem
    |
    v
Rust residualization
    |
    v
random sparse fixed-strength construction
    |
    v
existing structural repair
    |
    v
edge-count repair: reach exact E once
    |
    v
state is now in F_(s,E)
    |
    +----------------------------------+
    | production fixed-(s,E) MCMC      |
    |                                  |
    | mostly exact-E local 4-cycles    |
    | +                                |
    | occasional exact bridge paths    |
    +----------------------------------+
    |
    v
sample
    |
    v
merge positive fixed pairs
    |
    v
verify exact full strengths + exact E
```

There are two separate uses of biased movement around the target edge count:

1. **Initialization repair**
   - only needs to find one feasible exact-`E` state;
   - does not need to preserve a stationary distribution;
   - may therefore use a biased repair acceptance rule.

2. **Bridge MCMC**
   - is part of the stationary sampler;
   - must obey an exactly specified reversible auxiliary law;
   - must never call the biased repair transition.

The two paths must reuse the same deterministic rectangle/candidate machinery.

They must **not** reuse the same stochastic acceptance policy.

Rule:

> Share mechanics. Do not share biased stochastic transition policy.

---

# 5. Why ordinary exact-E local 4-cycles are not enough

The existing `master` fixed-strength kernel performs a fixed-direction 4-cycle:

```text
(a,b) -= 1
(c,d) -= 1
(a,d) += 1
(c,b) += 1
```

with exact state-dependent Hastings correction.

This move always preserves strengths.

If we simply reject moves that change `E`, the resulting kernel is exactly stationary for the conditional `(s,E)` target.

However it is not always connected.

Mandatory counterexample:

```text
N = 2
self loops = allowed
s_out = [2,2]
s_in  = [2,2]
E = 2
```

Two target states are:

```text
A =
[2 0
 0 2]

B =
[0 2
 2 0]
```

The one-step 4-cycle between either state and the all-ones table is:

```text
C =
[1 1
 1 1]
```

and:

```text
E(A) = 2
E(B) = 2
E(C) = 4
```

Therefore a kernel that refuses every intermediate state outside `E=2` cannot connect `A` and `B`.

This is the reason for the bridge kernel.

Do not solve this by inventing a new composite proposal with a hand-derived path Hastings ratio.

Instead reuse the already-correct fixed-strength kernel as an **auxiliary reversible chain**.

---

# 6. Mathematical design of the production sampler

The production kernel is a state-independent mixture:

```text
P = (1 - rho) * P_local + rho * P_bridge
```

where `rho` is a small fixed constant.

Start with:

```text
rho = 0.05
```

Keep it internal to Rust initially.

Do not expose `rho` as a public Python option unless benchmarking later proves that users need it.

---

# 7. Local exact-E kernel

Let the current state satisfy:

```text
E(current) = E_target
```

Use exactly the existing occupied-cell proposal law and exact Hastings ratio from `move_cycle.rs`.

The only new condition is:

```text
if E(proposed) != E_target:
    hold
```

Otherwise evaluate the same Metropolis-Hastings acceptance that the fixed-strength sampler already uses.

## Proof

For any two states `x,y` inside the exact-`E` fiber:

- proposal probability is unchanged from the existing fixed-strength chain;
- target ratio is unchanged, because conditioning on `E` multiplies all allowed states by the same normalization constant;
- Hastings correction is unchanged.

Therefore:

```text
pi_(s,E)(x) P_local(x,y)
=
pi_(s,E)(y) P_local(y,x)
```

for every distinct `x,y` in the fiber.

All proposals leaving the fiber are converted to diagonal self-loop probability.

Therefore the local kernel is reversible with respect to the exact conditional target.

---

# 8. Auxiliary distribution for bridge paths

Define:

```text
distance_E(x) = abs(E(x) - E_target)
```

Define an auxiliary distribution over the full fixed-strength fiber:

```text
mu_lambda(x)
    proportional to pi_s(x) * exp(-lambda * distance_E(x))
```

where `pi_s` is the ordinary fixed-strength target already used by `master`.

Use:

```text
bridge_lambda = 1.0
```

as the initial default.

This value is a performance parameter, not a correctness parameter.

For every state in the target fiber:

```text
E(x) = E_target
```

so:

```text
distance_E(x) = 0
```

and therefore:

```text
mu_lambda(x | E = E_target)
=
pi_s(x | E = E_target)
=
pi_(s,E)(x)
```

This identity is the key mathematical reason the bridge is exact.

---

# 9. One auxiliary bridge substep

Use the **same occupied-cell proposal** and the **same Hastings proposal ratio** as the existing fixed-strength kernel.

The only difference from the ordinary fixed-strength MH log acceptance is the edge-distance potential.

If:

```text
E_old = current occupied count
E_new = proposed occupied count
```

then:

```text
delta_edge_potential
    =
    -lambda * (
        abs(E_new - E_target)
        -
        abs(E_old - E_target)
    )
```

The auxiliary log acceptance is:

```text
log_alpha_aux
    =
    delta_log_family_target
    +
    log_q_reverse
    -
    log_q_forward
    +
    delta_edge_potential
```

Do not change the proposal law.

Do not preferentially choose a rectangle that raises or lowers `E` in the stationary bridge kernel.

The bias belongs in the **target acceptance**, not in the proposal selection.

This avoids having to derive a new proposal probability.

---

# 10. Exact bridge construction

The bridge is not a single custom composite MH move.

It is a censored path of the exact reversible auxiliary chain.

Use a small finite maximum number of auxiliary steps:

```text
bridge_max_steps
```

Start implementation with:

```text
bridge_max_steps = 16
```

The final default must be selected by the exact connectivity gate in this document.

Permitted escalation:

```text
16 -> 32 -> 64
```

Do not increase beyond 64 without reporting the smallest failing connectivity case.

A hard cap does **not** invalidate stationarity.

It can only limit connectivity/mixing.

That distinction is important.

## Bridge algorithm

Input state `x` must satisfy exact `E_target`.

Save enough information to restore `x` without cloning the complete sparse state.

Then:

### Step 1: require departure from the target fiber

Run one auxiliary MH substep.

If the state still has:

```text
E = E_target
```

then abort the bridge and restore the original state if an accepted in-fiber move occurred.

Return a bridge self-loop.

This includes:

- held proposal;
- MH rejection;
- accepted proposal with `delta E = 0`.

A successful bridge must first leave the exact-`E` fiber.

### Steps 2 ... `bridge_max_steps`

Continue running auxiliary MH substeps.

After each substep:

- if still outside the target fiber, continue;
- on the **first** return to `E_target`, stop and keep the returned state;
- if no return occurs within the maximum number of substeps, undo every accepted substep and restore the original state.

No additional path-level acceptance probability is applied.

## Why in-fiber first steps are aborted

Do not allow arbitrary exact-E moves before the excursion begins.

A successful bridge path must have this form:

```text
x in A
z1 outside A
z2 outside A
...
z_(k-1) outside A
y in A
```

where:

```text
A = {state: E(state) = E_target}
```

This path class is symmetric under reversal.

That makes the detailed-balance proof direct.

---

# 11. Proof that the bridge is exact

Let:

```text
K_lambda
```

be the one-step auxiliary MH kernel targeting `mu_lambda`.

By construction:

```text
mu_lambda(u) K_lambda(u,v)
=
mu_lambda(v) K_lambda(v,u)
```

for all states `u,v` in the fixed-strength state space.

Consider a successful bridge path:

```text
gamma = (x, z1, ..., z_(k-1), y)
```

where:

- `x,y` are in the exact-E fiber;
- every interior state is outside the exact-E fiber;
- `2 <= k <= bridge_max_steps`.

The probability of that path under the bridge construction is the product:

```text
K_lambda(x,z1)
K_lambda(z1,z2)
...
K_lambda(z_(k-1),y)
```

Repeatedly applying detailed balance gives:

```text
mu_lambda(x) * Prob(gamma)
=
mu_lambda(y) * Prob(reverse(gamma))
```

Because `x` and `y` both have exact `E_target`:

```text
mu_lambda(x) proportional to pi_(s,E)(x)
mu_lambda(y) proportional to pi_(s,E)(y)
```

with the same proportionality constant.

Therefore:

```text
pi_(s,E)(x) * Prob(gamma)
=
pi_(s,E)(y) * Prob(reverse(gamma))
```

Summing over every qualifying path of length at most `bridge_max_steps` gives:

```text
pi_(s,E)(x) P_bridge(x,y)
=
pi_(s,E)(y) P_bridge(y,x)
```

Failed bridge attempts are returned to the origin and only add diagonal self-loop probability.

Therefore `P_bridge` is reversible for the exact fixed-`(s,E)` target.

Finally, a state-independent mixture of two reversible kernels with the same target is reversible:

```text
P = (1-rho) P_local + rho P_bridge
```

Therefore the complete production kernel has the exact fixed-`(s,E)` target as its stationary distribution.

This proof is the mathematical basis of the implementation.

Do not replace it with a different composite proposal unless a new proof and exact oracle are supplied.

---

# 12. Required refactor of the existing 4-cycle implementation

The current `move_cycle.rs` already contains the correct fixed-direction occupied-cell proposal and exact Hastings ratio.

Do not rewrite the mathematics.

Refactor it minimally so three consumers reuse the same proposal mechanics:

1. existing free fixed-strength MCMC;
2. exact-E local MCMC;
3. auxiliary bridge MCMC;
4. edge-count initialization repair may reuse the candidate mechanics.

## 12.1 Introduce one fixed-size proposal record

Use a small stack-friendly struct similar to:

```rust
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cycle4Proposal {
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub d: u64,

    pub old_ab: OccNum,
    pub old_cd: OccNum,
    pub old_ad: OccNum,
    pub old_cb: OccNum,

    pub new_ab: OccNum,
    pub new_cd: OccNum,
    pub new_ad: OccNum,
    pub new_cb: OccNum,

    pub occupied_before: usize,
    pub occupied_after: usize,

    pub log_q_forward: f64,
    pub log_q_reverse: f64,
}
```

Exact field names may follow existing style, but the information content must be equivalent.

The struct must not allocate.

It must provide:

```rust
delta_edges()
apply(state)
undo(state)
```

`undo` must restore the four old occupations exactly with `StrengthState::set`.

## 12.2 Extract candidate drawing from the existing step

Create one internal function whose selection law is exactly the current `master` law:

```rust
draw_cycle4_proposal(...)
```

It must:

1. choose first occupied cell uniformly;
2. compute the existing valid-partner count;
3. choose the second occupied cell using the existing unbounded rejection logic;
4. use the same fixed direction:
   - decrement selected diagonal;
   - increment cross cells;
5. use the existing shared rectangle validation;
6. compute proposed occupations;
7. compute `occupied_after` using the same zero/positive transitions currently used in `move_cycle.rs`;
8. compute exactly the same `q_forward`;
9. compute exactly the same `q_reverse`.

If any existing guard would have returned `Held`, return a no-proposal result that the caller maps to `Held`.

Do not:

- add random orientation;
- bound the partner rejection with a finite retry count;
- choose decrement cells by a different law;
- enumerate all partners;
- allocate candidate vectors.

The original free fixed-strength wrapper must reproduce the current mathematical transition law after the refactor.

## 12.3 One shared MH evaluator

Create a small internal function that evaluates:

```text
delta_log_family
+ log_q_reverse
- log_q_forward
+ extra_log_weight
```

For ordinary fixed strength:

```text
extra_log_weight = 0
```

For auxiliary bridge:

```text
extra_log_weight = delta_edge_potential
```

For exact-E local moves:

- reject/hold first if `occupied_after != E_target`;
- otherwise `extra_log_weight = 0`.

Do not put fixed-E logic inside `StrengthTarget`.

`StrengthTarget` represents family degeneracy and optional cost and should remain orthogonal.

## 12.4 Existing free fixed-strength behavior

After refactoring, the existing function:

```rust
occupied_cycle4_step(...)
```

must still exist or all its call sites must be updated consistently.

Its behavior must remain mathematically identical:

```text
draw same proposal
same family delta
same Hastings q ratio
same acceptance
```

Add a regression test comparing exact transition probabilities before/after conceptually through the independent oracle described later.

---

# 13. Edge-count initialization repair

Before production MCMC starts, the state must be brought to exact residual `E_target`.

This is an initialization problem.

It does not need a stationary distribution.

## 13.1 Reuse the same cycle candidate mechanics

The repair must call the same fixed-direction cycle candidate generator.

It must not implement another rectangle-selection routine unless absolutely required.

This gives:

- exact strength preservation;
- existing domain validation;
- existing B capacity validation;
- existing self-loop policy after structural repair;
- sparse updates;
- exact `delta E` from `occupied_before/after`.

## 13.2 Repair objective

Define:

```text
d_old = abs(E_old - E_target)
d_new = abs(E_new - E_target)
```

Use this biased initialization-only acceptance rule:

```text
if d_new < d_old:
    accept

else if d_new == d_old:
    accept with probability 0.10

else:
    accept with probability exp(-2.0 * (d_new - d_old))
```

This is deliberately biased.

It is allowed because this phase only finds a feasible starting state.

The accepted move is applied directly.

Do **not** call this acceptance rule from production sampling.

## 13.3 Termination

Stop immediately when:

```text
state.occupied_count() == E_target
```

Use:

```text
max_edge_repair_steps_per_restart = 1_000_000
max_edge_repair_restarts = 5
```

as initial internal defaults.

These are safety limits, not correctness constants.

On exhaustion:

- discard the state;
- construct a fresh randomized fixed-strength state;
- rerun structural repair;
- retry edge-count repair.

If all attempts fail, return a structured error.

The error must include at least:

```text
best/current E
target E
absolute best distance
number of restarts
total attempted repair steps
```

Do not silently return an inexact state.

## 13.4 Randomized reconstruction is mandatory

`master` currently initializes through a compressed randomized constructor, but the wrapper must use the caller's RNG rather than reseeding internally with a constant seed.

Change:

```rust
initialize_table(...)
```

so it receives:

```rust
rng: &mut impl Rng
```

and forwards that RNG to:

```rust
compressed_aggregated_matching(...)
```

Update every caller.

Requirements:

- same top-level seed remains reproducible;
- different top-level seeds can produce different initial states;
- each edge-repair restart continues from the same RNG stream and therefore reconstructs differently.

Do not create `StdRng::seed_from_u64(0)` inside `initialize_table`.

---

# 14. Edge-target feasibility checks

Add a focused helper in a new module, preferably:

```text
fixed_edges.rs
```

or another equally explicit name under:

```text
occupation_mcmc/
```

Do not build a generic constraint framework.

Implement:

```rust
validate_edge_target(problem, target_edges)
```

for the **residual** problem.

At minimum enforce the following necessary conditions.

Let:

```text
T = sum residual strengths
A = number of admissible residual coordinates
```

### Zero total

If:

```text
T == 0
```

then require:

```text
E_target == 0
```

### Positive total

If:

```text
T > 0
```

require:

```text
E_target >= 1
```

### Every occupied pair has at least one event

Require:

```text
E_target <= T
```

### Cannot occupy more coordinates than exist

Require:

```text
E_target <= A
```

`A` must be computable without materializing an `N x N` set for complete-like domains.

### Family-capacity lower bound

For `B { layers: M }`:

```text
T <= M * E_target
```

equivalently:

```text
E_target >= ceil(T / M)
```

Also require:

```text
E_target >= sum_i ceil(s_out[i] / M)
E_target >= sum_j ceil(s_in[j] / M)
```

because each row/column occupied coordinate carries at most `M`.

For `ME` and `W`, each positive row/column needs at least one occupied pair:

```text
E_target >= number of i with s_out[i] > 0
E_target >= number of j with s_in[j] > 0
```

These checks are necessary, not claimed sufficient for every sparse structural-zero domain.

If a target passes these checks but repair exhausts, return `EdgeRepairExhausted`, not `InvalidEdgeTarget`.

Do not implement a general exact max-flow feasibility solver in this feature unless a concrete failing test proves it necessary.

---

# 15. Fix the current fixed-pair `O(N^2)` residual-domain problem

This is required for scalability.

The current `master` residualization of a `Complete` domain with fixed pairs builds a `Sparse` domain containing every non-fixed coordinate.

That is `O(N^2)` memory.

Do not keep that behavior.

## 15.1 Add a complete-minus-exclusions domain variant

Add a domain variant equivalent to:

```rust
PairDomain::CompleteMinus {
    node_count: usize,
    self_loops: bool,
    excluded: HashSet<(u64, u64)>,
}
```

Naming may differ, but semantics must be exactly:

```text
all coordinates allowed by the complete/self-loop policy
minus a small explicit excluded set
```

Use this variant when residualizing fixed pairs from a complete domain.

Memory becomes:

```text
O(F)
```

instead of:

```text
O(N^2)
```

where `F` is the number of fixed coordinates.

## 15.2 Update domain methods

Update every exhaustive `match PairDomain` in the repository.

At minimum ensure correct implementations of:

```text
node_count()
self_loops_allowed()
is_admissible(i,j)
capacity(...)
iter_admissible(...)
```

Add:

```rust
admissible_pair_count()
```

with:

- `O(1)` arithmetic for `Complete`;
- `O(1)` arithmetic plus `excluded.len()` for validated `CompleteMinus`;
- `allowed.len()` for `Sparse`.

For `CompleteMinus.iter_admissible()`:

- a lazy `N x N` iterator is acceptable;
- materializing all pairs is not.

Production fixed-strength sampling should not invoke this iterator in the normal complete-domain hot path.

## 15.3 Admissibility repair

The compressed constructor intentionally ignores domain restrictions.

Therefore a `CompleteMinus` initial table may place mass on an excluded fixed coordinate.

The existing inadmissible-pair repair must also run for `CompleteMinus` when exclusions are non-empty.

Do not special-case fixed coordinates in the constructor.

Reuse structural repair.

Add an explicit domain predicate such as:

```rust
requires_admissibility_repair()
```

if that simplifies `chain.rs`.

Do not infer this only from `matches!(domain, PairDomain::Sparse { .. })`.

---

# 16. Improve fixed-pair residualization while touching it

In `FixedStrengthProblem::into_residual` or a focused helper:

### Reject duplicates

Maintain the fixed-coordinate set.

If insertion of a fixed coordinate reports it was already present:

```text
return InvalidResidual("duplicate fixed pair ...")
```

Do not subtract duplicate occupations twice.

### Exclude zero-fixed pairs

Every fixed coordinate enters the exclusion set even when:

```text
occupation == 0
```

### Positive edge count

For fixed-`(s,E)` preparation compute:

```text
E_fixed = count(unique fixed pairs with occupation > 0)
```

Then:

```text
E_residual = E_full - E_fixed
```

### Single residualization

Python must pass the full strengths, full target edge count, and fixed arrays to Rust.

Rust performs:

```text
fixed-pair validation
strength subtraction
domain exclusion
edge-target subtraction
residual validation
```

once.

---

# 17. Suggested Rust module layout

Keep the existing modules.

Prefer only one new focused module:

```text
occupation_mcmc/fixed_edges.rs
```

It should contain fixed-edge-specific code such as:

```text
FixedEdgeKernelConfig
FixedEdgeCounters / internal diagnostics if needed
validate_edge_target
repair_to_edge_target
bridge_step
fixed_edge_step / fixed_edge_sweep helpers
```

Do not move family math into this file.

Do not move general rectangle mechanics into this file.

Minimal changes by existing file:

### `domain.rs`

- add complete-minus-exclusions representation;
- add efficient admissible-pair count;
- add helper identifying explicit exclusions/admissibility repair need.

### `problem.rs`

- reject duplicate fixed coordinates;
- residualize complete domains without `O(N^2)` materialization;
- add or call a helper that computes residual edge target.

### `initializer.rs`

- accept caller RNG;
- remove internal constant reseeding.

### `move_cycle.rs`

- factor current proposal into reusable fixed-size proposal record;
- preserve exact original proposal law;
- support:
  - ordinary MH;
  - exact-E veto;
  - auxiliary edge-potential MH;
- return applied proposal record when bridge needs undo.

### `rectangle.rs`

Only add small deterministic helpers if genuinely shared.

Do not turn it into a framework.

### `chain.rs`

Add one-shot fixed-strength+edges orchestration.

Reuse existing:

```text
init
structural repair
family target
sweep convention
sample conversion
```

### `errors.rs`

Add focused fixed-edge errors.

### `mod.rs`

Expose internal modules/functions only as required.

Avoid public Rust exports not required by Python binding.

---

# 18. Fixed-edge production sweep

The outer fixed-edge sampler should follow the same sweep-size convention as the current fixed-strength chain.

For one sweep:

```text
proposals_per_sweep =
    config.proposals_per_sweep
    or max(current occupied pairs, 2*N, 1)
```

For each outer proposal:

```text
draw u from Uniform(0,1)

if u < bridge_probability:
    bridge_step(...)
else:
    exact_E_local_step(...)
```

Use:

```text
bridge_probability = 0.05
```

initially.

This mixture probability must not depend on:

- current state;
- current E distance;
- node;
- family;
- acceptance history.

A constant mixture gives the simple stationarity proof above.

The bridge internally may execute multiple ordinary 4-cycle substeps.

---

# 19. Undo mechanics for failed bridge attempts

Do not clone the complete `StrengthState` on every bridge attempt.

That would make a rare bridge `O(E)` rather than `O(bridge_max_steps)`.

Instead record each **accepted** auxiliary 4-cycle proposal in a small vector:

```rust
Vec<Cycle4Proposal>
```

Preallocate:

```rust
Vec::with_capacity(bridge_max_steps)
```

This allocation is acceptable because:

- bridge attempts are rare;
- maximum capacity is tiny (`<=64`);
- the ordinary local hot path remains allocation-free.

If a bridge aborts or times out:

```text
for proposal in accepted_proposals.reverse():
    proposal.undo(state)
```

After undo, in debug/test builds assert:

```text
E == E_target
strengths unchanged
state.debug_validate()
```

If the bridge succeeds, discard the records and keep the current state.

Do not attempt to reconstruct the origin by running reverse stochastic proposals.

Undo is deterministic state restoration, not an MCMC transition.

---

# 20. Bridge counters

Track enough internal counters to diagnose mobility.

At minimum:

```text
outer proposals
local proposals
local accepted
local held due to delta E
local MH rejected

bridge attempts
bridge departures
bridge successful returns
bridge timeouts
bridge auxiliary substeps
bridge auxiliary accepted
```

Do not make these all public API fields unless existing diagnostics make that trivial.

They must at least be available to Rust tests and benchmark instrumentation.

Do not classify a bridge timeout as a correctness error.

It is a self-loop in the exact chain.

---

# 21. No false "insufficient mobility" error for singleton fibers

Do not fail production sampling merely because there are few accepted moves.

A valid exact `(s,E)` fiber can contain only one state.

In that case a chain that always stays there is perfectly correct.

Mobility thresholds belong in benchmark/test cases where the test fiber is known to contain multiple states.

Production may report diagnostics but must not reject a mathematically valid singleton problem just because acceptance is zero.

---

# 22. Exact small-state enumeration oracle

This is mandatory before Python capability exposure.

Create test-only exhaustive enumeration for tiny problems.

Do not use this enumeration in production.

## 22.1 Enumerate states independently

For tiny:

```text
N <= 3
small total strength
```

enumerate every occupation table satisfying:

```text
exact out strengths
exact in strengths
domain admissibility
family capacity
```

For fixed-`(s,E)` tests filter to:

```text
occupied_count == E_target
```

For bridge verification you also need the surrounding fixed-strength state space, including states with other values of `E`.

## 22.2 Independent target weights

In the test oracle implement explicit reference formulas independently of production:

```text
ME:  1/t!
B:   C(M,t)
W:   C(M+t-1,t)
```

Do not call:

```rust
OccupationFamily::log_base_measure
```

to create the expected distribution.

The production and oracle must not share the same family-weight function.

Normalize the tiny-state weights.

## 22.3 Exact base proposal matrix

Construct the exact transition matrix of the existing occupied-cell proposal.

For state `x` with:

```text
m = number of occupied cells
```

and an unordered decrement diagonal:

```text
(a,b), (c,d)
```

the realized transaction proposal probability is the existing master law:

```text
q_forward
=
(1/m) * (1/v_ab + 1/v_cd)
```

where:

```text
v_ab = number of occupied cells with source != a and target != b
v_cd = number of occupied cells with source != c and target != d
```

Use the exact reverse proposal calculated in the proposed state.

For every valid `x -> y`, compute independently:

```text
alpha
=
min(
    1,
    pi(y) q(y->x) /
    (pi(x) q(x->y))
)
```

Accumulate all transition probability into the matrix.

Put all invalid/rejected probability on the diagonal.

Assert:

```text
every row sums to 1
detailed balance holds pairwise
pi * P == pi
```

Use tight tolerance, e.g.:

```text
1e-12
```

for small cases unless floating-point accumulation requires a documented slightly looser bound.

## 22.4 Exact local fixed-E matrix

From the base proposal:

- if proposed `E != E_target`, move probability goes to self-loop;
- otherwise use ordinary exact MH.

Check detailed balance against exact conditional target.

## 22.5 Exact auxiliary matrix

Build:

```text
mu_lambda(x)
proportional to
pi_s(x) exp(-lambda abs(E(x)-E_target))
```

over the full tiny fixed-strength state space.

Construct exact auxiliary matrix `K_lambda`.

Assert:

```text
mu_lambda(x) K(x,y)
=
mu_lambda(y) K(y,x)
```

for every pair.

## 22.6 Exact bridge matrix by dynamic propagation

Do not enumerate random simulations.

Build the bridge matrix exactly.

For each origin `x` in the exact-E fiber:

### First auxiliary step

For every possible `z`:

- if `z` remains in exact-E fiber:
  - bridge aborts;
  - add that probability to `B[x,x]`;
- if `z` is outside:
  - place probability mass on outside state `z`.

### Further steps

For step lengths up to `bridge_max_steps`:

From outside-state probability mass:

- transitions to another outside state remain active;
- transitions to any exact-E state `y` are accumulated into `B[x,y]` and no longer propagated.

At the maximum step:

- all remaining outside probability becomes `B[x,x]` because production would undo and restore the origin.

Then assert:

```text
row sums = 1
pi_(s,E)(x) B(x,y)
=
pi_(s,E)(y) B(y,x)
```

This test directly verifies the production bridge construction.

## 22.7 Exact full mixed kernel

Construct:

```text
P =
(1-rho) L
+
rho B
```

where:

- `L` is exact local fixed-E kernel;
- `B` is exact bridge matrix.

Assert:

```text
row sums = 1
pairwise detailed balance
pi * P == pi
```

This is the primary mathematical release gate.

---

# 23. Mandatory connectivity oracle

Stationarity and connectivity are separate.

For the exact full production matrix `P` on each tiny fixed-`(s,E)` fiber:

- add an undirected graph edge between states with positive transition probability in either direction;
- compute connected components.

Mandatory counterexample:

```text
N = 2
self loops = true
s_out = [2,2]
s_in = [2,2]
E = 2
```

The full kernel **must** connect the two states shown in section 5 through a bridge.

Test at least:

### Families

- ME;
- B with `M=1` where feasible;
- B with `M=2` or greater;
- W with `M=1`;
- W with `M>1`.

### Self-loop policy

- allowed;
- forbidden where feasible.

### Margins

- symmetric;
- heterogeneous.

### Edge targets

- low feasible boundary;
- interior;
- high feasible boundary.

### Fixed pairs

- no fixed pairs;
- one positive fixed pair;
- one zero fixed pair.

Only use tiny cases that are explicitly proven feasible by enumeration.

## Bridge cap selection

Run connectivity grid with:

```text
bridge_max_steps = 16
```

If every mandatory fiber is connected, keep 16.

If not:

```text
try 32
```

If still not:

```text
try 64
```

Use the **smallest** value that passes.

If 64 still fails:

- do not add a longer arbitrary composite move;
- do not mark the capability supported;
- report the smallest disconnected fiber and investigate whether the underlying fixed-strength 4-cycle chain itself is disconnected under that domain.

---

# 24. Edge-repair tests

The repair is not tested for stationarity.

It is tested for:

```text
termination on known feasible cases
exact strengths
exact E
domain validity
family capacity
reproducibility by seed
different reconstruction across different seeds/restarts
```

## 24.1 Exhaustive tiny feasibility-driven tests

For tiny enumerated fibers:

1. enumerate all feasible fixed-strength states;
2. derive every feasible `E` value from those states;
3. for each feasible `E`:
   - start from at least one constructor state;
   - run structural repair;
   - run edge repair;
   - require exact target.

Do not test repair against arbitrary `E` values that enumeration proves impossible.

## 24.2 Infeasible targets

Test each validation bound separately.

Examples:

```text
E > T
E > admissible coordinates
E < positive-row lower bound
B: M*E < T
B: E < sum ceil(row_strength/M)
```

Require structured `InvalidEdgeTarget`.

## 24.3 Exhaustion semantics

Create a test configuration with intentionally tiny repair step budget.

Require:

```text
EdgeRepairExhausted
```

and verify that no inexact sample is returned.

---

# 25. Fixed-pair tests

Mandatory tests:

## Positive fixed pair

Given a full feasible sample-derived problem:

- fix one occupied pair to its observed positive occupation;
- target full `E` stays the original full `E`;
- sample residual;
- merge fixed pair;
- require exact full strengths and full E.

## Zero fixed pair

Choose a coordinate known absent in the generating network.

Freeze it at zero.

Require:

```text
output occupation at that pair = 0
```

and:

```text
full E = target E
```

The zero fixed coordinate must never be used by residual MCMC.

## Duplicate fixed pair

Pass the same coordinate twice.

Require a validation error.

## Fixed-edge subtraction

If there are `k` unique positive fixed pairs:

```text
E_residual = E_full - k
```

Test explicitly.

## No duplicate output coordinates

After merging fixed pairs, every output `(src,tgt)` must be unique.

Do not rely on downstream aggregation to repair duplicate coordinates.

---

# 26. Runtime invariant checks

Before returning a Rust residual sample, require:

```text
state.occupied_count() == residual_target_edges
```

This should be a runtime assertion/error at the orchestrator boundary, not only a debug assertion.

Also preserve existing strength validation.

At the Python/Rust binding boundary after merging fixed positive pairs, test:

```text
full occupied-pair count == target_edges
full out strengths == requested out strengths
full in strengths == requested in strengths
```

The implementation may keep expensive full checks in tests/debug code if necessary, but exact edge count is cheap and should be verified in production orchestration.

---

# 27. Python binding

Add one focused pyo3 entry point for fixed strengths + edges.

Use a signature consistent with existing fixed-strength binding, conceptually:

```text
family
strength_out
strength_in
target_edges
self_loops
fixed_sources
fixed_targets
fixed_occnums
layers
burn_in_sweeps
sweeps_per_sample
seed
```

Do not expose:

```text
bridge_probability
bridge_lambda
bridge_max_steps
repair_temperature
repair retry budgets
```

in the first public API.

Those are implementation details.

Python must not implement residual edge arithmetic.

The binding should:

1. validate array lengths;
2. map family enum;
3. build the full `FixedStrengthProblem`;
4. call the Rust fixed-strength+edges orchestrator;
5. merge only positive fixed pairs if the core returns only the residual sample;
6. return the normal sampled network arrays.

Prefer moving enough orchestration into core Rust that the binding remains thin.

---

# 28. Python router

Add the microcanonical route for:

```text
Constraint.STRENGTH_EDGES
```

for:

```text
ME
B
W
```

Required arguments:

```text
strength_out
strength_in
target_edges
```

Plus:

```text
layers
```

for B/W as already used by the family model.

Optional:

```text
known_source
known_target
known_occnum
self_loops
seed
burn_in_sweeps
sweeps_per_sample
```

Follow existing routing conventions.

Do not add a new top-level public sampling function.

The existing unified `sample_model` / `sample_model_detailed` route remains the public API.

---

# 29. Capability registry: expose only at the end

During implementation, do not mark fixed `(s,E)` as supported merely because routing compiles.

Capability exposure is the final gate.

After all mathematical tests and required scalability tests pass, add:

```text
Verb.SAMPLE
Ensemble.MICROCANONICAL
Constraint.STRENGTH_EDGES
```

for all three families.

Backend name should be explicit, e.g.:

```text
microcanonical_fixed_strength_edges
```

`requires_fit` should be:

```text
False
```

because this is a direct microcanonical constraint sampler.

Set sampling exactness in `sample_model_detailed` to:

```text
EXACT_STATIONARY_MCMC
```

with method:

```text
microcanonical_fixed_strength_edges
```

Do not let the generic fixed-strength `else` branch silently label the new case.

Add an explicit `STRENGTH_EDGES` branch.

---

# 30. Exactness terminology

The implementation is not an exact direct sampler.

It is:

```text
EXACT_STATIONARY_MCMC
```

Meaning:

- the implemented transition kernel has the exact desired stationary distribution;
- no approximation is introduced into the target law;
- finite burn-in and mixing remain ordinary MCMC concerns.

Do not call it:

```text
EXACT_DIRECT
```

Do not claim a formal global mixing-time bound.

---

# 31. Required scalability properties

## State

`StrengthState` remains sparse:

```text
O(N + E)
```

Do not add a dense adjacency matrix.

## Domain

Complete domain:

```text
O(1)
```

additional storage.

Complete-minus-fixed:

```text
O(F)
```

additional storage.

Sparse explicit domain:

```text
O(A)
```

where `A` is the user-provided allowed-coordinate set.

## Ordinary local proposal

Expected constant-time sparse operations:

```text
O(1)
```

apart from the existing occupied-partner rejection behavior inherited from `master`.

No heap allocation in the ordinary local proposal.

## Bridge

At most:

```text
bridge_max_steps <= 64
```

auxiliary proposals per bridge attempt.

Expected cost per outer MCMC step is bounded by:

```text
(1-rho) * 1 + rho * bridge_cost
```

With:

```text
rho = 0.05
bridge_max_steps = 16
```

the hard worst-case number of rectangle proposals averaged by scheduling is bounded by the mixture, while typical bridge attempts terminate earlier.

The bridge may allocate one tiny vector with capacity at most 64.

Do not clone `StrengthState`.

## Sweep

One sweep remains proportional to:

```text
max(E, 2N, 1)
```

outer proposals.

Therefore for sparse networks the intended work is approximately linear in the sparse state size per sweep.

---

# 32. Required `N=1000` scalability gate

Create a heavy benchmark/test fixture from an actual feasible generated network.

Do not hand-pick arbitrary strengths.

## Scenario A: ME

Generate a sparse directed occupation table with approximately:

```text
N = 1000
E approximately 10*N
```

and positive occupations large enough that:

```text
T > E
```

Derive:

```text
s_out
s_in
E
```

from that table.

Run fixed-`(s,E)` sampling.

Require:

```text
exact output strengths
exact E
no self-loop violation when disabled
completion without O(N^2) memory construction
nonzero movement on this known non-singleton case
```

## Scenario B: W

Same sparse scale for a representative `W` layer count.

Require the same invariants.

## Scenario C: B

Generate occupations respecting a modest layer count such as:

```text
M = 5
```

Derive constraints from the actual generated network.

Require exact constraints and capacity.

## Fixed-pair scalability case

For at least one `N=1000` scenario:

- freeze a modest number of positive pairs;
- freeze a modest number of zero pairs;
- verify the residual domain is the complete-minus-exclusions representation rather than an `N^2` `HashSet`;
- sample successfully.

Do not enforce fragile wall-clock thresholds in ordinary CI.

Record benchmark timings separately.

The hard architectural requirement is absence of `O(N^2)` production materialization.

---

# 33. Optional larger smoke benchmarks

After `N=1000` passes:

```text
N = 5000
E = O(N)
```

should be tested manually/heavy.

If practical, also run:

```text
N = 25000
E = O(N)
```

as a memory/scalability smoke test.

Do not delay the feature solely because `N=25000` requires benchmark tuning if:

- the algorithm is structurally sparse;
- `N=1000` passes;
- `N=5000` is healthy;
- no quadratic path is present.

But document the measured result.

---

# 34. Repair performance diagnostics

For the required generated benchmark cases record:

```text
initial E
target E
initial absolute gap
repair steps
repair restarts
best gap
final E
```

A healthy implementation should normally reach exact E well before the million-step safety bound.

If ordinary generated feasible cases repeatedly consume the full step budget:

- do not increase the budget blindly;
- inspect candidate `delta E` distribution;
- inspect whether structural constraints leave the 4-cycle proposal trapped;
- inspect whether the target is near a support boundary.

Do not change the stationary bridge proposal law to fix an initialization-repair performance issue.

Repair and stationary bridge are separate.

---

# 35. MCMC mobility diagnostics for known nontrivial benchmarks

For generated benchmark fibers known to contain many states, record:

```text
local proposal count
local accepted
local exact-E holds
local MH rejects

bridge attempts
bridge departures
bridge successful returns
bridge timeouts
bridge auxiliary steps
```

Required for benchmark cases:

```text
at least some accepted local or returned bridge transitions
```

Do not impose the same requirement on arbitrary user problems because singleton fibers are valid.

If bridge departure is almost zero:

- lower `bridge_lambda` only after exact tests remain passing.

If departures occur but returns almost never occur:

- raise `bridge_lambda` moderately.

Any `lambda > 0` preserves the bridge proof.

Parameter tuning changes efficiency, not stationary correctness.

Keep tuning constants internal.

---

# 36. Parameter-tuning procedure

Do not tune before mathematical tests pass.

Order:

1. implement with:
   ```text
   bridge_probability = 0.05
   bridge_lambda = 1.0
   bridge_max_steps = 16
   ```
2. run exact transition tests;
3. run exact connectivity grid;
4. increase only `bridge_max_steps` if connectivity requires it:
   ```text
   16 -> 32 -> 64
   ```
5. freeze the smallest passing value;
6. run N=1000 mobility benchmarks;
7. if return behavior is poor, tune `bridge_lambda`;
8. rerun exact transition tests after any code change;
9. changing only numeric lambda does not change the proof, but the oracle should still run.

Do not tune `bridge_probability` to hide a broken local or bridge kernel.

---

# 37. Tests that must remain unchanged/passing

All pre-existing fixed-strength tests on `master` must continue to pass.

In particular the refactor of `move_cycle.rs` must not break:

```text
ME fixed-strength sampling
B fixed-strength sampling
W fixed-strength sampling
self-loop handling
capacity handling
fixed-pair handling
strength-cost behavior
existing fixed-strength benchmarks
reproducibility by seed
```

Changing initializer RNG plumbing may change exact sample realizations for a given historical seed.

If an existing test incorrectly assumes a hard-coded old initial realization rather than reproducibility/invariants, update the test to assert the correct invariant.

Do not weaken statistical or mathematical tests merely to make them pass.

---

# 38. Recommended implementation sequence

Execute exactly in this order.

## Phase 1 - branch + baseline

1. branch from `master`;
2. run relevant existing fixed-strength Rust tests;
3. record that baseline is green;
4. do not edit Python yet.

Gate:

```text
existing fixed-strength core tests pass
```

## Phase 2 - domain and RNG scalability fixes

Implement:

- `CompleteMinus` domain;
- efficient `admissible_pair_count`;
- duplicate fixed-pair rejection;
- residualization without `O(N^2)` materialization;
- caller RNG into `initialize_table`;
- structural inadmissibility repair for complete-minus exclusions.

Tests:

- fixed-pair zero/positive residualization;
- duplicate pair rejection;
- `N=1000` domain representation does not allocate all pairs;
- same seed reproducible;
- different seeds can construct differently.

Gate:

```text
all existing fixed-strength tests + new residualization tests pass
```

Commit.

## Phase 3 - factor existing cycle proposal

Refactor `move_cycle.rs` into:

```text
draw proposal
evaluate MH
apply/undo
```

with fixed-size proposal record.

Do not add fixed-E yet.

Run all fixed-strength tests.

Gate:

```text
existing fixed-strength mathematical behavior is unchanged
```

Add an exact tiny transition test for the ordinary fixed-strength kernel if one does not already exist.

Commit.

## Phase 4 - exact-E local kernel

Add:

```text
occupied_after == E_target
```

veto.

Add exact local transition-matrix test.

Gate:

```text
pairwise detailed balance passes exactly on tiny fibers
```

Commit.

## Phase 5 - edge target validation + initialization repair

Add:

- fixed-edge error type;
- residual target computation;
- feasibility checks;
- biased repair using shared cycle candidate;
- randomized restart orchestration.

Add exhaustive tiny feasible-E repair tests.

Gate:

```text
every mandatory tiny feasible target reaches exact E
every infeasible bound fails cleanly
```

Commit.

## Phase 6 - auxiliary MH and bridge

Add auxiliary edge-potential MH.

First test its exact full fixed-strength transition matrix against `mu_lambda`.

Then add censored outside-excursion bridge with undo.

Then build exact bridge matrix oracle.

Gate:

```text
auxiliary detailed balance passes
bridge detailed balance passes
row sums pass
```

Commit.

## Phase 7 - full fixed-(s,E) kernel + connectivity

Add constant mixture of local and bridge kernels.

Build exact full transition matrix.

Run mandatory connectivity grid.

Select:

```text
bridge_max_steps = 16, 32, or 64
```

using the smallest passing value.

Gate:

```text
exact full detailed balance
exact stationarity
mandatory connectivity grid
```

If connectivity fails at 64, STOP.

Do not expose capability.

Commit only diagnostics/tests and report failure.

## Phase 8 - one-shot core sampler

Add one-shot:

```text
construct
structural repair
edge repair
burn in mixed exact kernel
thin/sample
return exact-E residual network
```

Add core invariant tests.

Gate:

```text
all families exact strengths + exact E
```

Commit.

## Phase 9 - Python binding and routing

Only now:

- pyo3 binding;
- unified Python route;
- fixed pairs through Rust residualization;
- `sample_model_detailed` method branch.

Keep capability unsupported until E2E tests pass.

Gate:

```text
Python route works in tests but is not yet advertised supported
```

Commit.

## Phase 10 - E2E + scalability

Run:

- generated feasible E2E cases;
- all three families;
- loops on/off;
- fixed positive/zero pairs;
- `N=1000` heavy scalability cases.

Gate:

```text
correctness + architecture + scalability pass
```

## Phase 11 - capability exposure

Only now update capability registry to supported.

Run full repo checks.

Commit final capability/docs changes.

---

# 39. Required full checks before handoff

Run the repository-prescribed checks from `AGENTS.md`.

At minimum:

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

Run heavy tests relevant to this feature.

If documentation is modified:

```bash
mkdocs build --strict
```

Do not report completion while any relevant failure remains unexplained.

---

# 40. Documentation after implementation

Only after the implementation and tests are stable, update user/developer docs according to repository policy.

Document:

```text
microcanonical fixed strengths + exact occupied-pair count
families ME/B/W
exact stationary MCMC
fixed pairs
self-loop support
meaning of target_edges
```

Do not expose internal bridge mathematics in a basic user-facing API page unless useful.

A developer/concepts document should include the stationary-target argument at a concise level.

---

# 41. Things the agent must NOT do

This section is explicit because these are easy implementation traps.

## Do not start from an old feature branch

Start from `master`.

## Do not implement a new general MCMC framework

Extend the fixed-strength code.

## Do not introduce random rectangle orientation

Keep the current fixed direction.

## Do not bound the existing second occupied-cell rejection sampler

The exact current proposal probability assumes the existing selection law.

## Do not invent a new path-level Hastings ratio

The bridge is a path of an already reversible auxiliary kernel.

## Do not call biased repair inside stationary MCMC

Repair bias is initialization-only.

## Do not let Python residualize fixed pairs

Rust residualizes once.

## Do not materialize complete-minus-fixed domains

Use explicit exclusions.

## Do not use Monte Carlo frequency agreement as the main stationarity proof

Use exact transition matrices.

## Do not expose capability before gates pass

Routing compilation is not mathematical validation.

## Do not solve connectivity by adding arbitrary path lengths or move families

First raise the bridge cap only through:

```text
16 -> 32 -> 64
```

If still disconnected, report the smallest case.

## Do not silently return approximate E

Exact target means exact.

## Do not add public tuning knobs unless required

Keep bridge/repair tuning internal.

---

# 42. Mathematical acceptance summary for implementation comments

Put a concise version of the following near the production fixed-edge kernel.

## Ordinary fixed-strength target

```text
pi_s(t) proportional to product_ij d_F(t_ij)
```

within the exact-strength/domain fiber.

## Desired fixed-edge target

```text
pi_(s,E)(t)
=
pi_s(t | E(t)=E_target)
```

## Exact local kernel

Use existing fixed-strength MH and hold whenever:

```text
E(proposal) != E_target
```

## Auxiliary bridge target

```text
mu_lambda(t)
proportional to
pi_s(t) exp(-lambda |E(t)-E_target|)
```

and therefore on the target fiber:

```text
mu_lambda(. | E=E_target)
=
pi_(s,E)(.)
```

## Auxiliary acceptance

```text
log alpha
=
Delta log pi_s
+
log q_reverse
-
log q_forward
-
lambda(
    |E_new-E_target|
    -
    |E_old-E_target|
)
```

## Bridge

Use auxiliary reversible steps.

A successful bridge:

```text
starts in exact E
leaves exact E on first step
has only non-exact-E interior states
returns to exact E before a fixed state-independent cap
```

Failed attempts restore the origin.

Path reversal plus auxiliary detailed balance proves bridge detailed balance.

## Full kernel

```text
P
=
(1-rho) P_local
+
rho P_bridge
```

with constant `rho`.

Therefore the exact conditional target is stationary.

---

# 43. Definition of done

This feature is done only when every item below is true.

## Mathematical

- [ ] Target distribution is exactly the family degeneracy conditioned on strengths and `E`.
- [ ] Existing fixed-strength proposal law is preserved.
- [ ] Exact-E local kernel has exact enumerated detailed balance.
- [ ] Auxiliary edge-biased kernel has exact enumerated detailed balance.
- [ ] Bridge matrix has exact enumerated detailed balance.
- [ ] Full mixed kernel has exact enumerated detailed balance.
- [ ] Exact target vector is stationary under the full transition matrix.
- [ ] Mandatory tiny connectivity grid passes with bridge cap <= 64.

## Feasibility/repair

- [ ] Edge-target necessary bounds implemented.
- [ ] Known feasible tiny edge targets repair to exact E.
- [ ] Repair exhaustion returns structured error.
- [ ] Reconstruction is randomized from caller RNG.
- [ ] No inexact state enters production MCMC.

## Fixed pairs

- [ ] Duplicate fixed coordinates rejected.
- [ ] Positive fixed pairs subtract one from full edge target.
- [ ] Zero fixed pairs remain excluded.
- [ ] Rust performs residualization once.
- [ ] Residual sampler cannot reoccupy a fixed coordinate.
- [ ] Merged output has unique coordinates.
- [ ] Full output has exact strengths and exact E.

## Scalability

- [ ] No dense `N x N` occupation structure.
- [ ] No `N^2` complete-minus-fixed `HashSet`.
- [ ] Complete-minus exclusions use `O(F)` memory.
- [ ] Ordinary local hot path is allocation-free.
- [ ] Bridge work is bounded by the selected cap.
- [ ] `N=1000` generated feasible cases pass for ME, B, W.
- [ ] `N=1000` fixed-pair case preserves sparse domain representation.
- [ ] Larger sparse smoke benchmark run if practical.

## API

- [ ] Thin pyo3 binding.
- [ ] Unified Python router.
- [ ] `STRENGTH_EDGES` direct microcanonical route for ME/B/W.
- [ ] `sample_model_detailed` identifies the method explicitly.
- [ ] Exactness is `EXACT_STATIONARY_MCMC`.
- [ ] Capability marked supported only after all gates.

## Regression

- [ ] Existing fixed-strength sampling still passes.
- [ ] Existing strength-cost sampling still passes.
- [ ] Existing self-loop/capacity/fixed-pair tests still pass.
- [ ] Full Rust checks pass.
- [ ] Full fast Python checks pass.
- [ ] Relevant heavy tests pass.

---

# 44. Final handoff report required from the implementation agent

When finished, report exactly:

## Code

- branch name;
- commits;
- files added;
- files materially changed.

## Mathematics

- bridge cap selected;
- maximum exact detailed-balance residual from oracle;
- maximum stationarity residual;
- number of mandatory connectivity fibers checked;
- whether any disconnected fibers remain in the mandatory grid.

## Repair

- tiny feasible targets tested;
- any repair exhaustion cases;
- N=1000 repair steps/restarts by family.

## Performance

For each N=1000 benchmark:

```text
family
N
target E
total occupation T
fixed pair count
construction time
structural repair time
edge repair steps
edge repair time
burn-in/sampling time
local acceptance
bridge departures/returns/timeouts
```

Absolute timings are informational; the key blocking issue is any evidence of quadratic materialization or unusable mobility.

## Tests

List every command run and whether it passed.

## Limitations

State any case that is not covered.

Do not use phrases such as "should be correct" or "seems to work".

If a required gate is not proven, state that the feature is not ready.

---

# 45. Final design summary

The intended implementation is deliberately small:

```text
existing master fixed-strength MCMC
        |
        +-- reuse exact occupied-cell proposal
        |
        +-- exact-E local veto
        |
        +-- edge-distance auxiliary target
        |       |
        |       +-- exact reversible bridge
        |
        +-- shared cycle candidate mechanics
                |
                +-- biased initialization-only edge repair
```

The important separation is:

```text
repair:
    biased, only finds a feasible start

sampling:
    exact MH / exact bridge, preserves target
```

The key scalability rule is:

```text
never replace sparse complete-domain logic by an explicit N^2 domain
```

The key mathematical rule is:

```text
never derive a new composite proposal probability when an exact reversible
path of the existing validated kernel can do the job
```

Implement this design, validate it with exact small-state transition matrices, and only then expose the new capability.
