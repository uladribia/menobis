# MENoBiS final microcanonical fixed-`(s,k)` implementation plan

## Purpose

Implement the final coupled microcanonical constraint in MENoBiS:

> exact directed out-strengths and in-strengths **and** exact directed out-degrees and in-degrees.

Notation:

- `t_ij >= 0`: occupation number of ordered pair `(i,j)`.
- `s_out[i] = sum_j t_ij`.
- `s_in[j]  = sum_i t_ij`.
- `k_out[i] = sum_j 1[t_ij > 0]`.
- `k_in[j]  = sum_i 1[t_ij > 0]`.
- `E = sum_i k_out[i] = sum_j k_in[j]`.

This corresponds to the existing public constraint name:

```text
Constraint.STRENGTH_DEGREE
```

The implementation must support:

```text
ME
B { layers: M }
W { layers: M }
```

This document is intended to be sufficient for an implementation agent starting with a clean session and the current MENoBiS `master`.

The primary goals are:

1. mathematical correctness;
2. maximum reuse of the now-finished fixed-`(s,E)` implementation;
3. minimal new production machinery;
4. sparse scaling at relevant `N`, with `N=1000` as a hard gate;
5. no new proposal/Hastings derivation unless absolutely necessary.

---

# 0. Start from current `master`

Create a new branch from current `master`.

```bash
git checkout master
git pull --ff-only
git checkout -b feature/microcanonical-fixed-strength-degree
```

Do not start from an older `fixed-sE` branch.

The fixed-`(s,E)` implementation now on `master` is the foundation of this work.

Read `AGENTS.md` before editing and follow the repository workflow.

---

# 1. Read these files before changing anything

The code is the source of truth.

Read at least:

```text
AGENTS.md

crates/menobis-core/src/model/
    family.rs
    problem.rs
    sampling_plan.rs

crates/menobis-core/src/generation/microcanonical/
    route.rs

crates/menobis-core/src/generation/microcanonical/occupation_mcmc/
    chain.rs
    domain.rs
    errors.rs
    fixed_edges.rs
    initializer.rs
    move_cycle.rs
    problem.rs
    repair.rs
    state.rs
    target.rs
    mod.rs

crates/menobis-core/src/generation/microcanonical/binary/
    core.rs
    feasibility.rs
    initializer.rs
    sampler.rs
    state.rs
    switch.rs

crates/menobis-python/src/
    generation.rs
    lib.rs

src/menobis/
    capabilities.py
    routing.py
    models/generation.py
    models/spec.py
    models/types.py

crates/menobis-test-oracles/tests/
    occupied_cell_hastings_verification.rs
    fixed_strength_enumeration.rs
    fixed_strength_edges_enumeration.rs
    fixed_strength_edges_scalability.rs
    fixed_kt_exhaustive.rs
```

Before production changes, run the relevant current tests and establish a green baseline.

---

# 2. Do not use the existing fixed-`(k,T)` factorization for fixed-`(s,k)`

This is the most important architectural restriction.

Current fixed-`(k,T)` code does:

```text
sample support with exact degrees
        ↓
allocate positive occupations with fixed total T
```

That factorization works because after the support has `E` edges, the fixed-total occupation law depends on:

```text
family, E, T
```

and not on node-specific strength constraints.

That factorization is **not valid** for fixed-`(s,k)`.

For fixed strengths, the admissible occupation allocations depend on which support was chosen:

```text
sum_j t_ij = s_out[i]
sum_i t_ij = s_in[j]
```

Different supports can have:

- different numbers of feasible strength allocations;
- different family-weighted partition sums;
- no feasible allocation at all.

Therefore the marginal target over supports is not the uniform fixed-degree support law used by fixed-`(k,T)`.

Do **not** implement:

```text
fixed-degree support MCMC
+ independent occupation allocator
```

for this feature.

Do not directly use `directed_switch_step` as a production fixed-`(s,k)` move either. A support-only switch generally destroys the exact strengths unless occupations are changed jointly.

The correct structural plan is:

```text
OccupationMcmc
```

not:

```text
FactorizedMicrocanonical
```

---

# 3. Mathematical target

Let the base-measure factor already implemented for family `F` be:

```text
d_F(t)
```

with:

```text
ME:  d(t) = 1 / t!

B(M): d(t) = C(M,t), 0 <= t <= M

W(M): d(t) = C(M+t-1,t)
```

The fixed-`(s,k)` target is:

```text
pi_(s,k)(t)
    proportional to
    product_(i,j) d_F(t_ij)
```

over exactly the states satisfying:

```text
s_out(t) = s_out_target
s_in(t)  = s_in_target

k_out(t) = k_out_target
k_in(t)  = k_in_target

domain constraints
family occupation support
fixed-pair constraints
```

Do not duplicate family formulas in production code.

Production must continue to obtain family log-weight changes through the existing:

```text
StrengthTarget
OccupationFamily
```

The explicit formulas above are for the independent exact test oracle.

---

# 4. The central reuse result: fixed-`(s,k)` is an exact trace of fixed-`(s,E)`

The degree target implies a unique edge count:

```text
E_target = sum_i k_out_target[i] = sum_j k_in_target[j]
```

Define:

```text
Omega_E = states with exact strengths and exact E_target
A_k     = states in Omega_E with the exact target degree vectors
```

The finished fixed-`(s,E)` implementation on `master` already provides an exact reversible Markov kernel:

```text
K_E
```

with invariant distribution:

```text
pi_E = pi_(s,E)
```

The desired target is simply:

```text
pi_(s,k) = pi_E(. | A_k)
```

The new sampler must therefore **not** try to invent a degree-preserving 4-cycle proposal. Instead it must build a degree-biased reversible auxiliary chain whose proposal is one complete `K_E` transition, and then take the first-return trace of that auxiliary chain onto `A_k`.

This is exactly the same architecture already used one level lower by fixed-`(s,E)`:

```text
fixed s:
    raw fixed-strength MH
    + edge-distance bias
    + first return to exact E

fixed (s,k):
    finished exact K_E
    + degree-distance bias
    + first return to exact k
```

This is the main design decision.

Consequences:

- no new occupied-cell proposal law;
- no new family Hastings derivation;
- no degree-preserving-veto kernel as the primary sampler;
- no second support-state representation;
- temporary degree violations are allowed inside the auxiliary excursion;
- strengths and E remain exact at every `K_E` endpoint;
- the degree bias affects efficiency only, not the exact target.

---

# 5. Existing code that must be reused directly

## 5.1 `StrengthState`

Do not create a second graph/support state.

`StrengthState` already maintains:

```text
row_occ_count
col_occ_count
```

These are exactly:

```text
current k_out
current k_in
```

because they count positive occupied cells by source and target.

`StrengthState::set` already updates these values correctly on:

```text
0 -> positive
positive -> 0
positive -> positive
```

Therefore no extra degree vectors need to be updated manually in the state.

## 5.2 `Cycle4Proposal`

Reuse the current fixed-direction 4-cycle proposal.

It already stores:

```text
old/new occupations
occupied_before
occupied_after
log_q_forward
log_q_reverse
```

and the draw code already computes row/column occupancy changes.

Extend it minimally so the fixed-degree layer can compute changes in degree-distance without scanning all `N` nodes.

## 5.3 Finished fixed-`(s,E)` kernel

Reuse:

```text
exact_e_local_step
auxiliary_substep
bridge_step
fixed_edge_step
fixed_edge_sweep
BridgeConfig
FixedEdgeCounters
validate_edge_target
repair_to_edge_target
edge_repair_rebuild
```

The new fixed-degree layer should call the finished fixed-edge outer transition as its reversible base transition.

## 5.4 Fixed-strength residualization

Reuse:

```text
FixedStrengthProblem::into_residual()
PairDomain::CompleteMinus
merge_fixed_pairs(...)
```

Do not create a parallel fixed-pair residualization implementation.

## 5.5 Sparse domain policy

Preserve:

```text
Complete
CompleteMinus
Sparse
```

and the existing no-`N^2` policy.

---

# 6. Full intended pipeline

The final sampler is:

```text
full strengths + full degrees + fixed pairs
        |
        v
FixedStrengthProblem::into_residual()
        |
        v
residualize degree vectors using the same fixed pairs
        |
        v
validate combined residual (s,k)
        |
        v
E_residual = sum residual k_out
        |
        v
reuse fixed-(s,E) preparation:
    randomized fixed-s construction
    structural repair
    edge-count repair to exact E_residual
        |
        v
degree repair to exact k_residual
    using the same degree-biased K_E auxiliary step
        |
        v
state is exactly in A_k
        |
        v
production fixed-(s,k) trace step:
    repeatedly propose one exact K_E transition
    apply degree-distance MH bias
    stop on first return to exact k
    timeout => undo whole excursion => self-loop
        |
        v
burn in / thin / sample
        |
        v
merge positive fixed pairs with existing helper
        |
        v
runtime verify exact full strengths + exact full degrees
```

There is **no initial implementation gate that tries a pure degree-preserving-veto kernel first**. That path risks severe rejection in sparse large graphs because a raw fixed-E transition usually changes one or more node degrees.

A one-step return to the exact degree fiber is naturally accepted by the trace kernel; no separate local kernel is needed for correctness.

---

# 7. Fixed-pair residualization

Fixed pairs must be handled entirely in Rust.

Input fixed triples are:

```text
(src, tgt, occupation)
```

The existing `FixedStrengthProblem::into_residual()` already:

- rejects duplicate fixed coordinates;
- validates admissibility;
- validates B capacity;
- subtracts fixed occupation from strengths;
- excludes every fixed coordinate from the residual domain;
- excludes zero fixed coordinates too;
- keeps complete-minus-fixed domains sparse via `CompleteMinus`.

Reuse that unchanged.

## 7.1 Degree contribution of a fixed pair

For a fixed pair with:

```text
occupation > 0
```

subtract:

```text
1 from degree_out[src]
1 from degree_in[tgt]
```

For a fixed pair with:

```text
occupation == 0
```

subtract nothing from degrees.

The zero fixed coordinate is still excluded from the residual domain by the existing strength residualization.

## 7.2 Correct ordering

In the core fixed-`(s,k)` orchestrator:

1. clone/store the fixed-pair list;
2. call existing `FixedStrengthProblem::into_residual()`;
3. only if that succeeds, residualize the full degree arrays from the stored fixed-pair list.

This ordering matters because `into_residual()` is already the authoritative duplicate/admissibility validation.

Do not degree-subtract duplicated fixed pairs before the existing duplicate check runs.

## 7.3 Residual edge count

After residualizing degrees:

```text
E_residual = sum residual_degree_out
```

and require:

```text
sum residual_degree_out
==
sum residual_degree_in
```

No extra user `target_edges` parameter is needed.

The exact degree sequence already fixes the edge count.

---

# 8. New focused production module

Add exactly one focused sibling module:

```text
crates/menobis-core/src/generation/microcanonical/occupation_mcmc/fixed_degrees.rs
```

Do not build a generic constraint framework.

This module should contain only fixed-strength + fixed-degree-specific code:

```text
ResidualDegreeTarget
DegreeTraceConfig
DegreeRepairConfig
DegreeRepairOutcome
DegreeTraceCounters

validate_degree_target(...)
residualize_degree_target(...)

degree_distance(...)
degree_distance_delta(...)

degree_auxiliary_step(...)
repair_to_degree_target(...)
degree_trace_step(...)
degree_trace_sweep(...)
```

`degree_auxiliary_step` is the central reusable primitive for both initialization repair and stationary sampling.

Do not move family mathematics into this file.

Do not duplicate `fixed_edges.rs`.

---

# 9. Residual degree target type

Use one compact internal type, conceptually:

```rust
pub(crate) struct ResidualDegreeTarget {
    pub out: Vec<u32>,
    pub in_: Vec<u32>,
    pub edge_count: usize,
}
```

It should be created once after fixed-pair residualization.

Do not make this a Python-visible/public API type.

---

# 10. Combined `(s,k)` feasibility checks

Create:

```text
validate_degree_target(
    residual_strength_problem,
    residual_degree_target
)
```

These checks are necessary and cheap.

They must not enumerate `N^2` pairs.

## 10.1 Shape

Require:

```text
len(k_out) == N
len(k_in) == N
```

## 10.2 Degree sums

Require:

```text
sum(k_out) == sum(k_in)
```

Define the common sum as `E`.

## 10.3 Strength/degree zero compatibility

For every source node:

```text
s_out[i] == 0 iff k_out[i] == 0
```

For every target node:

```text
s_in[i] == 0 iff k_in[i] == 0
```

## 10.4 Minimum one event per occupied pair

For every source:

```text
k_out[i] <= s_out[i]
```

For every target:

```text
k_in[i] <= s_in[i]
```

## 10.5 B capacity

For `B { layers: M }` require:

```text
s_out[i] <= M * k_out[i]
s_in[i]  <= M * k_in[i]
```

Use widened/checked arithmetic.

ME and W have no analogous finite upper occupation bound.

## 10.6 Domain slot capacity

Add one efficient domain helper, preferably:

```rust
PairDomain::admissible_degree_caps()
    -> (Vec<usize>, Vec<usize>)
```

Semantics:

```text
out_caps[i] = admissible residual coordinates (i,j)
in_caps[j]  = admissible residual coordinates (i,j)
```

Complexity:

```text
Complete:      O(N)
CompleteMinus: O(N + F)
Sparse:        O(N + A)
```

where `F` is excluded fixed coordinates and `A` explicit allowed pairs.

Do not call `iter_admissible()` over a complete `N x N` domain for this.

Require:

```text
k_out[i] <= out_caps[i]
k_in[i]  <= in_caps[i]
```

## 10.7 Reuse fixed-edge feasibility

Call existing:

```text
validate_edge_target(residual_strength_problem, E)
```

This reuses global edge/strength/capacity bounds.

## 10.8 Do not use `DirectedDegreeSequence::new` as a hard validator

The existing fixed-`(k,T)` validator includes a constructive greedy graphicality attempt and may classify a sequence as infeasible because that particular constructor failed.

Do not use it as an exact rejection criterion for fixed-`(s,k)`.

If cheap necessary checks pass but combined repair exhausts, return a repair-exhaustion error rather than claiming mathematical infeasibility.

---

# 11. Extend `Cycle4Proposal` for O(1) degree-distance changes

The current draw already computes row/column occupancy before/after values while computing reverse Hastings availability.

Store the necessary values in `Cycle4Proposal`, e.g.:

```rust
out_a_before: usize
out_a_after: usize

out_c_before: usize
out_c_after: usize

in_b_before: usize
in_b_after: usize

in_d_before: usize
in_d_after: usize
```

Equivalent compact fields are acceptable.

Do not recompute them by scanning `StrengthState`.

In `fixed_degrees.rs`, implement a pure helper for one proposal:

```text
Delta L1 =
  |out_a_after - k_out*[a]| - |out_a_before - k_out*[a]|
+ |out_c_after - k_out*[c]| - |out_c_before - k_out*[c]|
+ |in_b_after  - k_in*[b]|  - |in_b_before  - k_in*[b]|
+ |in_d_after  - k_in*[d]|  - |in_d_before  - k_in*[d]|
```

Because `a != c` and `b != d`, no out-row or in-column term is duplicated.

For a sequence of proposals, sum these deltas. They telescope to the full endpoint change.

This keeps hot-path degree checking proportional to underlying cycle count, not `N`.

---

# 12. Degree-distance definition

Define:

```text
D_raw(t)
    =
    sum_i |k_out(t)[i] - k_out_target[i]|
    +
    sum_j |k_in(t)[j] - k_in_target[j]|
```

Every endpoint of `K_E` has the same total E as the target, so `D_raw` is even.

For repair/tuning use:

```text
D(t) = D_raw(t) / 2
```

Compute the full value by O(N) scan only:

- when degree repair begins;
- in debug/test verification when needed.

During production update it from recorded proposal deltas.

Never scan all N degree entries after every proposal.

---

# 13. Make the finished fixed-`(s,E)` outer transition recordable

The new layer must be able to:

1. execute one exact production `K_E` transition;
2. determine the endpoint degree-distance change;
3. undo the complete `K_E` transition if an outer degree-potential MH decision rejects it;
4. retain a sequence of accepted `K_E` transitions so a capped degree excursion can be undone on timeout.

Do this with a minimal recorder.

Do **not** duplicate the fixed-edge kernel.

## 13.1 Required internal API

Add a recorded internal variant conceptually like:

```rust
pub(crate) fn fixed_edge_step_recorded(
    ...,
    recorder: &mut Vec<Cycle4Proposal>,
    ...
) -> FixedEdgeStepOutcome
```

or one private implementation accepting:

```text
Option<&mut Vec<Cycle4Proposal>>
```

The existing `fixed_edge_step(...)` and `fixed_edge_sweep(...)` behavior must remain unchanged for current callers.

Do not introduce a trait framework.

## 13.2 Recording semantics

For one outer `K_E` transition:

- if the local exact-E subkernel accepts one cycle, append that `Cycle4Proposal`;
- if the local subkernel holds/rejects, append nothing;
- if the fixed-E bridge succeeds, append every cycle that remains applied in the successful bridge path, including the final return cycle;
- if the fixed-E bridge aborts or times out and restores its origin, append nothing externally.

A caller must therefore be able to mark:

```text
start_index = recorder.len()
```

run one recorded `K_E` transition, and know that:

```text
recorder[start_index..]
```

is exactly the deterministic undo log for the net state change of that one `K_E` transition.

## 13.3 Mandatory fixed-sE regression

The unrecorded fixed-`(s,E)` path must:

- consume the same RNG sequence as before;
- have the same transition law;
- keep the same exact stationary proof;
- preserve current hot-path behavior for ordinary fixed-sE calls.

Add direct tests:

```text
same seed + same initial state:
    recorded K_E endpoint == ordinary K_E endpoint
```

and:

```text
execute recorded K_E
undo recorder[start_index..] in reverse
=> exactly original state, occupied coordinates, strengths and degree caches
```

All existing fixed-sE exact oracle and scalability tests are mandatory regression gates after this change.

---

# 14. Initialization: reuse complete fixed-`(s,E)` preparation

After validation:

```text
E_residual = sum residual k_out
```

First construct a valid exact-`(s,E_residual)` state.

Use existing:

```text
edge_repair_rebuild(...)
repair_to_edge_target(...)
```

or a tiny shared helper factored from them.

Do not invent a new constructor.

At the end require exact strengths, exact E, valid domain and valid family capacity.

Only then start degree repair.

---

# 15. Degree repair: reuse the same degree-biased `K_E` auxiliary step

The exact-E state produced by section 14 will generally not have the requested degree vectors.

Do not implement a separate stochastic degree-repair proposal/acceptance law.

Reuse the exact same `degree_auxiliary_step` that production sampling uses.

## 15.1 Repair state space

Every proposal is one complete exact `K_E` transition, so every repair endpoint automatically preserves:

```text
exact residual strengths
exact residual E
valid domain
valid family support
```

Only the degree vector is allowed to vary.

## 15.2 Degree-biased repair transition

Let:

```text
D(x) = normalized degree distance
```

as defined in section 12.

One repair substep:

1. execute one recorded `K_E` transition `x -> y`;
2. compute `D(y)-D(x)` from the recorded cycle metadata;
3. accept the endpoint with:

```text
alpha = min(1, exp(-lambda_repair * (D(y)-D(x))))
```

4. on rejection, undo the recorded `K_E` transition in reverse;
5. on acceptance, update the cached current distance.

Use initial internal:

```text
lambda_repair = 1.0
```

The same helper should implement production degree bias; only the supplied lambda/config may differ.

This acceptance is exact for the auxiliary distribution, although repair does not need stationarity. Reusing it removes one whole extra stochastic policy from the code.

## 15.3 Success

Stop immediately when:

```text
D == 0
```

Then require:

```text
state.row_occ_count == residual k_out target
state.col_occ_count == residual k_in target
```

## 15.4 Restart policy

Add:

```rust
DegreeRepairConfig
```

with initial safety defaults:

```text
max_steps_per_restart = 1_000_000
max_restarts = 5
lambda_repair = 1.0
```

A restart must rebuild the whole exact-E start:

1. discard state;
2. randomized fixed-strength construction;
3. structural repair;
4. existing edge repair to exact residual E;
5. retry degree repair.

Do not restart from an arbitrary inexact-E state.

## 15.5 Error

Add `DegreeRepairExhausted` carrying at least:

```text
best_degree_distance
restarts
total_steps
target_edges
```

Optionally record best outgoing/incoming L1 components.

Never return an inexact sample.

---

# 16. Primary degree-biased auxiliary kernel over `K_E`

This is the central production transition below the trace wrapper.

Let `K_E(x,y)` be the complete finished fixed-`(s,E)` transition kernel.

Define the degree-distance target potential:

```text
D(x) = 1/2 * [
    sum_i |k_out(x)[i] - k_out_target[i]|
  + sum_j |k_in(x)[j]  - k_in_target[j]|
]
```

Define an auxiliary distribution over the exact fixed-`(s,E)` fiber:

```text
mu_lambda(x)
    proportional to
    pi_E(x) * exp(-lambda * D(x))
```

Start with internal:

```text
degree_trace_lambda = 1.0
```

This is a performance parameter, not a correctness parameter.

## 16.1 Proposal

Propose one **complete production fixed-E transition**:

```text
y ~ K_E(x, .)
```

using the recorded variant from section 13.

Do not inspect or rederive its internal local/bridge proposal probability.

## 16.2 Outer MH acceptance

Because `K_E` is reversible for `pi_E`:

```text
pi_E(x) K_E(x,y) = pi_E(y) K_E(y,x)
```

Therefore the MH ratio for `mu_lambda` simplifies exactly to:

```text
alpha_degree(x,y)
=
min(
    1,
    exp(-lambda * (D(y)-D(x)))
)
```

This cancellation is the main code-reuse benefit.

Do not recompute:

- family degeneracy change;
- occupied-cell Hastings q;
- fixed-E bridge path probability;
- edge-distance bias used internally by K_E.

Those are already part of the exact `K_E` transition and cancel at this outer level.

## 16.3 Rejection

If the outer degree-potential MH rejects:

```text
undo every Cycle4Proposal produced by that K_E transition in reverse
truncate recorder to its pre-transition length
state and cached D return exactly to x
```

If `K_E` itself is a self-loop, then `D(y)=D(x)` and the outer acceptance is 1; the state remains unchanged.

## 16.4 Complexity

Computing `D(y)-D(x)` must use the O(1)-per-cycle proposal metadata from section 11.

Do not scan all N node degrees after a K_E proposal.

---

# 17. Primary production kernel: capped first-return trace to the exact degree fiber

The production fixed-`(s,k)` kernel is the trace of the degree-biased auxiliary chain from section 16 onto:

```text
A_k = {x in Omega_E : D(x)=0}
```

This is the **default production sampler**, not an optional bridge.

## 17.1 Algorithm

Precondition:

```text
D(current) = 0
```

Save:

```text
origin log index
current D = 0
```

Then for at most:

```text
degree_trace_max_steps
```

repeat:

1. run one `degree_auxiliary_step` using proposal `K_E`;
2. if it rejects, state remains at the previous auxiliary state;
3. if after the auxiliary step `D == 0`, this is the **first return** to `A_k`; stop and keep the state;
4. otherwise continue outside `A_k`.

Important: a return at step 1 is valid.

That includes:

- a degree-preserving accepted `K_E` transition;
- a K_E/outer-MH self-loop at the origin;
- any other one-step endpoint in `A_k`.

Do **not** force the first step to leave the fiber. The forced-departure rule in the existing fixed-E bridge exists because that bridge is mixed with a separate local kernel. Here the trace itself is the primary kernel, so the mathematical first return is at time `n >= 1`.

## 17.2 Timeout

If no return occurs by the cap:

```text
undo the complete excursion log in reverse
restore the exact origin
return a top-level self-loop
```

No path-level MH correction is applied.

## 17.3 Initial cap

Start with:

```text
degree_trace_max_steps = 16
```

Permitted deterministic escalation during oracle/performance validation:

```text
16 -> 32 -> 64
```

Use the smallest value passing the required exact connectivity and N=1000 mobility gates.

If 64 is inadequate, STOP and report the smallest/representative failing case rather than inventing a new move family in the same feature.

---

# 18. Proof that the primary trace kernel is exact

Let:

```text
Q_lambda
```

be the degree-biased auxiliary kernel from section 16.

By MH construction it is reversible for:

```text
mu_lambda(x) proportional to pi_E(x) exp(-lambda D(x))
```

On the exact degree fiber `A_k`:

```text
D(x)=0
```

so:

```text
mu_lambda(x | A_k)
=
pi_E(x | A_k)
=
pi_(s,k)(x)
```

Consider a successful first-return path:

```text
x, z1, ..., z_(n-1), y
```

with:

```text
x,y in A_k
all interior z_r outside A_k
1 <= n <= degree_trace_max_steps
```

For `n=1` there are no interior states; direct in-fiber transitions are therefore naturally included.

Reversibility of `Q_lambda` gives pathwise:

```text
mu_lambda(x) * Prob(path)
=
mu_lambda(y) * Prob(reverse(path))
```

The reverse path has the same first-return structure because all interior states are outside `A_k`.

Since the endpoint restriction of `mu_lambda` is proportional to `pi_(s,k)`, summing over every first-return path of length up to the cap yields:

```text
pi_(s,k)(x) R(x,y)
=
pi_(s,k)(y) R(y,x)
```

Timeout probability is assigned to the origin diagonal and cannot violate detailed balance.

Therefore the capped first-return trace kernel is exactly reversible for the desired fixed-`(s,k)` target.

The cap and lambda affect only:

```text
mobility
mixing
runtime
```

not the stationary law.

---

# 19. Recorder design for the nested trace

Use one flat:

```rust
Vec<Cycle4Proposal>
```

for the complete top-level degree excursion.

For each auxiliary proposal:

```text
start_index = log.len()
run one recorded K_E transition
```

If the outer degree-potential MH rejects:

```text
undo log[start_index..] in reverse
truncate log to start_index
```

If it accepts:

```text
keep the appended records
```

If the top-level degree trace returns to `D=0`:

```text
discard the log without undo
```

If the degree trace times out:

```text
undo the entire excursion log in reverse
restore the origin
```

Do not clone `StrengthState`.

The existing fixed-E bridge remains internally bounded, so each nested `K_E` transition has bounded recorded path length.

Preallocate only a modest vector capacity; never reserve memory proportional to N^2.

---

# 20. Reuse the existing exact oracle infrastructure

The fixed-sE oracle already provides the hard components:

```text
independent family weights
tiny fixed-strength enumeration
exact occupied-cell MH matrix
exact fixed-E local matrix
exact edge auxiliary matrix
exact fixed-E bridge matrix
exact full K_E matrix
detailed-balance checks
stationarity checks
connectivity checks
```

Do not duplicate these.

Preferred test refactor:

```text
crates/menobis-test-oracles/tests/common/occupation_matrix.rs
```

or another test-only common module containing reusable exact matrix helpers.

Then both:

```text
fixed_strength_edges_enumeration.rs
fixed_strength_degree_enumeration.rs
```

consume the same exact `K_E` builder.

If moving helpers risks destabilizing the existing oracle, keeping the degree oracle in the same integration-test module is acceptable.

Copying a second giant implementation of `K_E` is not.

---

# 21. Independent fixed-`(s,k)` target oracle

Enumerate tiny fixed-strength states exactly as the fixed-sE oracle already does.

For every enumerated state compute support degrees independently from its positive cells.

Filter target states to:

```text
k_out == k_out_target
k_in  == k_in_target
```

Expected state weights must use independent family formulas:

```text
ME:  1/t!
B:   C(M,t)
W:   C(M+t-1,t)
```

Do not call production `OccupationFamily::log_base_measure` to construct expected probabilities.

Normalize only over the exact fixed-`(s,k)` fiber.

---

# 22. Exact auxiliary and trace transition-matrix oracle

This is the primary mathematical release gate.

## 22.1 Exact degree-biased auxiliary matrix

Start from the exact full production fixed-E matrix:

```text
K_E
```

over the tiny exact-E state space.

For `x != y` define:

```text
alpha(x,y)
=
min(1, exp(-lambda * (D[y]-D[x])))
```

and:

```text
Q[x,y] = K_E[x,y] * alpha(x,y)
```

Put all outer-MH rejection probability on `Q[x,x]`.

Expected auxiliary weights:

```text
mu[x] proportional to pi_E[x] * exp(-lambda * D[x])
```

Assert:

```text
every Q row sums to 1
pairwise detailed balance for mu
mu * Q == mu
```

Do not inspect the internal path probabilities of `K_E` in this calculation.

## 22.2 Exact capped first-return matrix

For every origin `x` in the exact degree fiber:

```text
active[x] = 1
```

For steps `1..=L`:

1. propagate active probability through `Q`;
2. any destination with `D=0` is absorbed into trace row `R[x,destination]` and is no longer propagated;
3. only destinations with `D>0` remain active.

After step `L`:

```text
all remaining active outside-fiber mass -> R[x,x]
```

because production timeout undoes the excursion and restores the origin.

Assert:

```text
every R row sums to 1
pairwise detailed balance against pi_(s,k)
pi_(s,k) * R == pi_(s,k)
```

This exact trace matrix, not a histogram, is the scientific proof of the final kernel.

---

# 23. Connectivity, cap selection, and efficiency gate

The trace architecture is mandatory; connectivity testing now chooses/tunes its finite cap rather than deciding whether the trace exists.

## 23.1 Mandatory automatic tiny-fiber search

At minimum search feasible cases with:

```text
N = 2 and 3
small strength totals
loops allowed/forbidden where feasible

ME
B(M=1)
B(M=2 or 3)
W(M=1)
W(M=2 or 3)
```

For each exact-strength state space:

1. derive feasible degree targets from enumerated states;
2. for every fixed-`(s,k)` fiber with at least two states;
3. build exact trace matrix `R`;
4. build the positive-probability undirected transition graph;
5. compute connected components.

Try:

```text
L = 16
```

then only if necessary:

```text
L = 32
L = 64
```

Use the smallest cap connecting every mandatory tiny fiber.

If 64 still leaves mandatory fibers disconnected, STOP and report the smallest case.

## 23.2 Lambda tuning does not affect correctness

Start with:

```text
lambda = 1.0
```

At N=1000 record separately:

```text
fraction of top-level traces returning at step 1
fraction leaving D=0
fraction successfully returning after leaving
fraction timing out
mean auxiliary K_E steps per trace
mean maximum D reached
fraction ending at a different exact-k state
```

Interpretation:

- too little departure from `D=0`: lambda may be too large;
- many departures but many timeouts: lambda may be too small or cap too short;
- almost all returns are identical states: poor effective movement.

If tuning is needed, test a small internal grid such as:

```text
lambda in {0.5, 1.0, 2.0}
```

and keep one universal default unless evidence strongly requires family-specific tuning.

Any finite lambda preserves exactness; rerun the exact oracle after implementation changes.

---

# 24. Degree repair and production deliberately share the same auxiliary transition

Initialization repair and production have different stopping rules but should reuse the same stochastic primitive:

```text
degree_auxiliary_step = one recorded K_E proposal + degree-distance MH
```

Repair:

```text
start from arbitrary exact-(s,E) state
run degree_auxiliary_step repeatedly
stop when D=0
restart if safety budget exhausted
```

Production:

```text
start from D=0
run degree_auxiliary_step repeatedly
stop at first return to D=0
undo to origin if trace cap expires
```

This gives maximum code reuse and removes an unnecessary second biased move policy.

Do not call the production trace before repair has reached exact target degrees.

---

# 25. New one-shot core orchestrator

Add in `chain.rs` or a small call-through into `fixed_degrees.rs`:

```rust
sample_fixed_strength_degree(...)
```

Prefer a signature conceptually like:

```rust
pub fn sample_fixed_strength_degree(
    problem: FixedStrengthProblem,
    full_degree_out: Vec<u32>,
    full_degree_in: Vec<u32>,
    config: McmcConfig,
    degree_config: DegreeTraceConfig,
) -> Result<SampledNetwork, FixedStrengthError>
```

Keep degree tuning internal at the pyo3/Python boundary.

Pipeline:

```text
clone fixed pairs
problem.into_residual()
residualize full degree targets
validate residual (s,k)

E_res = sum residual degree_out

build exact-(s,E_res) state:
    existing construction
    structural repair
    existing edge repair

repair_to_degree_target using degree_auxiliary_step
assert D=0

burn in with degree_trace_sweep
thin/sample with degree_trace_sweep

state -> SampledNetwork
merge_fixed_pairs using existing helper
runtime validate full strengths and full degrees
return
```

Do not call the independent fixed-`(k,T)` core.

---

# 26. Runtime final invariant

Before returning, compute once:

```text
actual full out strengths
actual full in strengths
actual full out degrees
actual full in degrees
```

Require exact equality to full requested vectors.

Also require unique output coordinates, family capacity, self-loop policy.

This O(E) boundary validation is acceptable.

---

# 27. Counters and benchmark diagnostics

Track enough diagnostics to evaluate the nested trace at scale.

## Degree repair

At minimum:

```text
initial degree distance
best degree distance
repair auxiliary K_E steps
repair accepted endpoints
repair restarts
```

## Production trace

At minimum:

```text
top-level trace proposals
trace returns at step 1
trace departures from D=0
trace successful returns after departure
trace timeouts
trace auxiliary K_E steps
trace outer degree-MH acceptances/rejections
trace different-state returns
maximum/mean excursion degree distance
```

Also retain/aggregate the underlying `FixedEdgeCounters` so benchmark output can distinguish whether poor performance originates in `K_E` itself or in the outer degree trace.

Do not fail arbitrary production requests solely because different-state movement is zero; singleton fibers are valid.

Movement thresholds belong only to generated benchmark fibers known to contain multiple states.

---

# 28. Sampling-plan ontology correction

Current classifier gives degree/edge constraints priority over strengths.

Change it to:

```text
1. if exact strengths are present:
       OccupationMcmc

2. else if exact degrees or edge count are present:
       FactorizedMicrocanonical

3. else:
       GrandCanonical
```

Why:

```text
strength + edge
strength + degree
```

are coupled occupation-number problems.

Pure:

```text
degree + total
edge + total
```

factorize.

Update `model/sampling_plan.rs` docs/tests.

Add tests:

```text
strength + degree -> OccupationMcmc
strength + edges  -> OccupationMcmc
degree + total    -> FactorizedMicrocanonical
edges + total     -> FactorizedMicrocanonical
```

Do not add a new SamplingPlan variant.

---

# 29. Rust microcanonical router

Update:

```text
generation/microcanonical/route.rs
```

so `route_occupation_mcmc` handles:

```text
if residual degrees present:
    fixed-(s,k)
else if residual edges present:
    existing fixed-(s,E)
else:
    existing fixed-s
```

For the no-fixed `PreparedProblem` path, construct the same complete PairDomain/FixedStrengthProblem pattern already used by fixed strength.

Keep `route_factorized` for pure `(k,T)` and `(E,T)` only.

---

# 30. pyo3 binding

Add a thin binding analogous to:

```text
sample_fixed_strength_edges
```

Name preferably:

```text
sample_fixed_strength_degree
```

Inputs:

```text
family
strength_out
strength_in
degree_out
degree_in
self_loops
fixed_sources
fixed_targets
fixed_occnums
layers
burn_in_sweeps
sweeps_per_sample
seed
```

No target_edges or total_events.

Validate lengths, build `PairDomain::Complete`, `FixedStrengthProblem`, `McmcConfig`, then call core.

Rust performs residualization.

Register in `crates/menobis-python/src/lib.rs`.

Do not expose repair/bridge tuning.

---

# 31. Unified native router for no-fixed calls

Extend `sample_model_microcanonical(...)` with:

```text
constraint = "strength_degree"
```

Require all four strength/degree arrays.

Construct one `PreparedProblem` containing both strength and degree vectors.

After SamplingPlan priority correction it routes through OccupationMcmc.

This lets no-fixed Python calls reuse the unified native router.

Fixed-pair calls can use the dedicated binding because PreparedProblem does not carry fixed coordinate lists.

Do not add fixed-pair fields to PreparedProblem in this feature.

---

# 32. Python generation wrapper

Add to:

```text
src/menobis/models/generation.py
```

a thin:

```text
_sample_fixed_strength_degree_mcmc
```

analogous to `_sample_fixed_strength_edges_mcmc`.

Only:

- normalize numpy dtypes;
- map optional fixed arrays to lists;
- call native function;
- construct EdgeTable.

No residualization or feasibility logic.

---

# 33. Python routing

In microcanonical `src/menobis/routing.py`, add `Constraint.STRENGTH_DEGREE`.

Require:

```text
strength_out
strength_in
degree_out
degree_in
```

Recommended:

```text
if no fixed pairs:
    _sample_model_router(
        ensemble="microcanonical",
        family=...,
        constraint="strength_degree",
        strength_out=...,
        strength_in=...,
        degree_out=...,
        degree_in=...,
        ...
    )
else:
    _sample_fixed_strength_degree_mcmc(...)
```

Do not copy the Python fixed-kT fixed-pair residualization.

Fixed-(s,k) follows fixed-(s,E): residualize in Rust.

Update stale supported-constraint error messages.

---

# 34. Capability exposure is last

Only after all gates add ME/B/W microcanonical `STRENGTH_DEGREE`:

```text
supported=True
requires_fit=False
backend="microcanonical_fixed_strength_degree"
```

Required:

```text
strength_out
strength_in
degree_out
degree_in
```

Optional:

```text
seed
self_loops
burn_in_sweeps
sweeps_per_sample
layers  # B/W
```

Update `unsupported_cases()`.

---

# 35. Sampling exactness metadata

In `sample_model_detailed` add explicit branch:

```text
method = "microcanonical_fixed_strength_degree"
exactness = EXACT_STATIONARY_MCMC
```

Do not fall through to generic fixed-strength naming.

---

# 36. Unit tests: residualization and validation

## Positive fixed pair

Generate a feasible full table.

Freeze one occupied coordinate at its **full observed occupation**.

Verify:

```text
strength residual subtracts occupation
out degree residual subtracts 1
in degree residual subtracts 1
coordinate excluded
```

## Zero fixed pair

Freeze an actually absent coordinate at zero.

Verify strengths/degrees unchanged but coordinate excluded.

## Duplicate fixed coordinate

Existing FixedStrengthProblem must reject before degree double-subtraction.

## Positive fixed pair exceeds degree target

Require InvalidDegreeTarget.

## B capacity

`strength > M * degree` must fail.

## Degree greater than strength

Must fail.

## Domain cap

Use CompleteMinus exclusions to make row/column have insufficient free slots; fail without N^2 enumeration.

---

# 37. Unit tests: proposal degree delta

Test cached proposal before/after degree counts against independent full recomputation.

Cover:

```text
no support change
two decrement cells disappear + two cross cells appear
one decrement disappears
one cross enters
mixed cases
```

Production delta must exactly equal independent scan.

---

# 38. Exact primary trace-kernel oracle gate

The release gate is the exact matrix of the **actual primary degree-biased first-return trace**, not a censored local kernel.

Mandatory assertions:

```text
auxiliary row stochasticity
auxiliary detailed balance
auxiliary stationarity
trace row stochasticity
trace detailed balance
trace stationarity
trace connectivity components
```

Cover at least:

```text
ME
B(M=1)
B(M>1)
W(M=1)
W(M>1)

loops on/off
symmetric and heterogeneous strengths
multiple degree targets
positive fixed pair
zero fixed pair
```

Only use enumerated feasible fibers.

A Monte Carlo histogram can be a secondary smoke test but is not a correctness proof.

---

# 39. Production-vs-oracle correspondence test

In addition to the exact mathematical matrix, verify that the Rust production transition implements the same kernel.

For selected tiny origin states:

1. reset production state to the same origin;
2. run one top-level degree trace many times with independent seeds;
3. estimate endpoint frequencies;
4. compare to the exact trace row from section 22.

Use this only as an implementation-correspondence test; the exact matrix remains the proof.

Also directly test:

```text
lambda = 0
```

where the outer auxiliary MH always accepts every non-held `K_E` endpoint. The resulting trace must equal the first-return trace of unmodified `K_E` onto the exact degree fiber.

---

# 40. Trace timeout and undo tests

Mandatory deterministic tests:

## Immediate return

Construct a case where one `K_E` transition lands in the degree fiber in one step.

Require the trace to stop immediately and keep that endpoint.

## Immediate self-loop

If the first auxiliary transition leaves state unchanged, require a valid one-step trace self-loop.

Do not force departure.

## Timeout restoration

Configure a deliberately tiny cap on a case that can leave the degree fiber.

Require:

```text
state after timeout == exact origin
strengths exact
degrees exact
E exact
occupied-pair map exact
row_occ_count/col_occ_count exact
```

## Nested rejection restoration

Force an outer degree-potential MH rejection after an accepted recorded `K_E` endpoint and require exact restoration to the pre-K_E auxiliary state.

---

# 41. Degree-repair tests

Degree repair uses the same degree-biased auxiliary step as production, but it is tested for reaching a feasible start rather than stationarity.

Require:

```text
known feasible target reaches D=0
strengths remain exact
E remains exact at every K_E endpoint
domain remains valid
family capacity remains valid
same seed reproducible
different seeds can produce different paths
exhaustion returns structured error
```

For tiny tests:

1. enumerate exact-strength states;
2. derive feasible degree targets from actual states;
3. start from other exact-E states where possible;
4. require repair success under normal budgets.

Do not invent arbitrary degree targets and assume feasibility.

---

# 42. Hard scalability gate: N=1000

Reuse the existing fixed-sE scalability fixture as much as possible.

Generate sparse feasible directed occupation tables and derive all constraints from the actual table:

```text
s_out
s_in
k_out
k_in
E
```

The benchmark target is therefore known feasible.

## 42.1 ME

Use approximately:

```text
N = 1000
E ~= 10*N
T > E
```

Require:

```text
exact output strengths
exact output degrees
no forbidden loops
successful degree repair
no O(N^2) state/domain object
nonzero different-state trace returns on this known nontrivial case
```

## 42.2 W

Run the same scale for representative W layers.

## 42.3 B

Generate occupancies satisfying a moderate cap such as `M=5`, derive targets, and require exact constraints/capacity.

## 42.4 Fixed-pair case

Freeze a modest number of occupied pairs at their full observed occupation and absent pairs at zero.

Require:

```text
CompleteMinus residual domain
O(F) exclusion storage
exact merged full strengths
exact merged full degrees
```

## 42.5 Trace-efficiency measurements

For every N=1000 family record:

```text
repair initial/final D
repair steps/restarts

production traces
step-1 returns
trace departures
successful post-departure returns
timeouts
different-state returns
mean K_E substeps per top-level trace
mean/max D during excursions
underlying K_E local/bridge counters
```

The test must show usable movement; exactness alone is not an adequate scalability result.

---

# 43. N=1000 performance failure and tuning policy

If generated feasible N=1000 cases show poor mobility, diagnose before adding new architecture.

## 43.1 Degree repair failure

If repair repeatedly reaches its full safety budget:

```text
STOP
```

Do not blindly increase the budget.

Inspect:

```text
distance trajectory
accepted K_E endpoint distance deltas
underlying K_E mobility
lambda_repair
```

## 43.2 Trace failure

If most traces never leave `D=0`, lower the degree lambda modestly.

If traces leave but time out frequently, either:

- increase lambda modestly to bias return more strongly; or
- escalate cap `16 -> 32 -> 64`.

Use a small documented tuning grid, not open-ended parameter search.

If different-state return rate remains effectively zero at cap 64 across generated nontrivial cases, STOP and report it as an algorithmic limitation.

Do not add in this task:

```text
dense max-flow construction
generic Markov basis
N^2 support enumeration
new support-only production chain
```

---

# 44. Optional N=5000 smoke

After all N=1000 gates pass, run a sparse:

```text
N = 5000
E = O(N)
```

ME smoke test.

Require:

```text
exact strengths
exact degrees
sparse memory behavior
bounded trace memory
no O(N) degree scan inside auxiliary steps
no obvious superlinear blow-up caused by the outer trace
```

Record trace return/timeout statistics in addition to wall times.

`N=25000` remains a useful stretch test, not a blocking requirement if N=1000/5000 are healthy and the implementation is structurally sparse.

---

# 45. Complexity requirements

## State

Reuse:

```text
StrengthState = O(N + E)
```

No second support adjacency matrix.

## Fixed-pair domain

Reuse:

```text
CompleteMinus = O(F)
```

No complete-minus-fixed free-pair materialization.

## Degree target

```text
O(N)
```

## One raw cycle

Degree-distance delta:

```text
O(1)
```

using cached proposal before/after degree counts.

## One base `K_E` transition

Degree update work:

```text
O(number of Cycle4Proposal objects actually applied inside K_E)
```

with existing fixed-E bridge cap.

## One top-level fixed-`(s,k)` trace

At most:

```text
degree_trace_max_steps
```

base `K_E` transitions before timeout restoration.

No full `O(N)` degree scan per auxiliary step.

## Trace memory

A flat undo log proportional only to the bounded nested excursion length.

Do not clone `StrengthState`.

## Sweep

Use the existing sparse outer convention unless benchmarks justify another fixed rule:

```text
max(E, 2*N, 1)
```

top-level degree trace attempts per sweep.

---

# 46. Do not over-generalize production code

Do not create:

```text
GenericConstraint
SupportStatistic trait
ConstraintPotential trait
GenericBridge
MarkovBasis
GenericRepairObjective
```

in this feature.

Use concrete hierarchy:

```text
move_cycle.rs
    ↓
fixed_edges.rs
    ↓
fixed_degrees.rs
```

That is enough.

---

# 47. Do not modify fixed-`(k,T)` except harmless shared tests

The `binary/` implementation solves a different factorization.

Do not rewrite it.

Do not make fixed-(s,k) inherit from its support sampler.

At most reuse terminology/test helpers.

---

# 48. Recommended implementation sequence

Follow this order.

## Phase 1 — baseline

Branch from current master and run fixed-s, fixed-sE and fixed-kT relevant tests.

Gate:

```text
baseline green
```

## Phase 2 — reuse exact oracle infrastructure

Before production code:

- factor/reuse the exact `K_E` matrix builder;
- add independent degree-vector extraction and fixed-(s,k) target grouping;
- implement exact degree-biased auxiliary matrix and capped trace matrix in the oracle.

This phase establishes the mathematical object the production code must match.

Gate:

```text
exact auxiliary + trace mathematics pass on initial tiny cases
```

## Phase 3 — residual degree target and feasibility

Implement:

```text
admissible_degree_caps
ResidualDegreeTarget
fixed-pair degree subtraction
validate_degree_target
errors
```

Gate: new validation tests + all existing tests green.

Commit.

## Phase 4 — proposal degree metadata

Extend `Cycle4Proposal` with O(1) before/after degree information.

Add independent delta tests.

Run existing occupied-cell and fixed-sE exact oracles.

Gate:

```text
no fixed-s/fixed-sE regression
```

Commit.

## Phase 5 — recordable `K_E`

Add the minimal recorder to the finished fixed-edge transition.

Require:

```text
recorded/unrecorded K_E same endpoint for same RNG
undo exact
successful fixed-E bridge records final return cycle
failed fixed-E bridge leaks no external record
```

Run the full fixed-sE exact oracle and scalability regression.

Commit.

## Phase 6 — degree auxiliary production primitive

Implement exactly:

```text
one recorded K_E proposal
+ scalar degree-distance MH
```

Compare production one-step empirical behavior against exact auxiliary oracle on tiny cases.

Gate:

```text
auxiliary kernel exact
```

Commit.

## Phase 7 — exact-E preparation + degree repair

Reuse existing fixed-sE construction/edge repair.

Implement degree repair by repeatedly calling the same `degree_auxiliary_step` until `D=0`.

Gate:

```text
every mandatory enumerated feasible tiny target reaches D=0
```

Commit.

## Phase 8 — primary capped degree trace

Implement first-return trace with:

```text
step-1 return allowed
timeout undo
initial cap 16
```

Run exact trace matrix oracle, connectivity search and timeout/undo tests.

Escalate cap only:

```text
16 -> 32 -> 64
```

Gate:

```text
exact trace detailed balance + stationarity + mandatory tiny connectivity
```

If 64 fails mandatory connectivity, STOP.

Commit.

## Phase 9 — one-shot core sampler + N=1000 tuning

Implement:

```text
sample_fixed_strength_degree
sample_fixed_strength_degree_bench
```

Run N=1000 ME/B/W/fixed-pair benchmarks.

Tune only internal lambda/cap if needed, using the documented small grid.

Gate:

```text
exact constraints + sparse architecture + usable different-state trace movement
```

Commit.

## Phase 10 — Rust route and sampling-plan ontology

- strengths take priority in `SamplingPlan`;
- route coupled s,k through `OccupationMcmc`;
- keep pure k,T factorized.

Commit.

## Phase 11 — pyo3 + Python wrapper/routing

Add binding, wrapper and high-level `STRENGTH_DEGREE` route.

Keep capability unsupported until E2E passes.

Commit.

## Phase 12 — E2E

Generated feasible E2E:

```text
ME/B/W
loops on/off
positive/zero fixed pairs
same-seed reproducibility
exact full strengths
exact full degrees
```

Correct the obvious typo `aexact` to `exact` when implementing this document.

## Phase 13 — capability exposure

Only now enable microcanonical `STRENGTH_DEGREE` with:

```text
EXACT_STATIONARY_MCMC
```

Run full checks.

---

# 49. E2E construction

For every E2E:

1. generate a valid directed non-binary network;
2. derive strengths and degrees;
3. sample with:
   ```text
   ensemble=MICROCANONICAL
   constraint=STRENGTH_DEGREE
   ```
4. verify all four target vectors exactly.

Do not use arbitrary hand-picked high-dimensional constraints unless feasibility is explicitly proven.

---

# 50. Full regression requirements

Must remain green:

```text
fixed s
fixed s + cost
fixed s,E
fixed E,T
fixed k,T
ME/B/W
fixed pairs
CompleteMinus
self-loop policies
B capacity
```

Any change to move_cycle.rs or fixed_edges.rs requires the existing fixed-sE exact oracle again.

---

# 51. Full commands before handoff

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

Also explicitly run relevant ignored/heavy:

```text
fixed-sE exact oracle
fixed-s,k exact oracle
fixed-sE N=1000 regression
fixed-s,k N=1000 ME/B/W
fixed-s,k fixed-pair scalability
```

If docs change:

```bash
mkdocs build --strict
```

---

# 52. Things the agent must NOT do

- Do not factorize support and occupations for fixed-`(s,k)`.
- Do not write another occupied-cell Hastings formula.
- Do not rederive the internal fixed-E bridge probability.
- Do not make degree-preserving censorship of one K_E step the primary sampler.
- Do not force the first trace step to leave the exact degree fiber.
- Do not scan all N degrees after every proposal.
- Do not clone `StrengthState` for veto/undo.
- Do not implement a second stochastic degree-repair policy; reuse `degree_auxiliary_step`.
- Do not residualize fixed pairs in Python.
- Do not use fixed-kT greedy graphicality as an exact infeasibility proof.
- Do not build N^2 admissible-pair lists.
- Do not expose lambda/cap/repair tuning publicly in the first version.
- Do not weaken fixed-sE oracle tests to accommodate recorder changes.
- Do not add a new move family merely because cap 16 is inefficient; follow 16 -> 32 -> 64 and diagnose.
- Do not enable capability before mathematical and scalability gates pass.

---

# 53. Definition of done

## Mathematics

- [ ] Target is family base measure conditioned on exact strengths/degrees.
- [ ] Degree-implied E is handled exactly.
- [ ] `K_E` is reused as the proposal kernel of the degree auxiliary chain.
- [ ] Outer degree MH ratio is exactly `exp(-lambda * Delta D)`.
- [ ] Exact auxiliary matrix satisfies detailed balance/stationarity.
- [ ] Primary production kernel is the capped first-return trace to `D=0`.
- [ ] Step-1 return is handled correctly.
- [ ] Exact trace matrix satisfies detailed balance/stationarity.
- [ ] Mandatory tiny trace connectivity grid passes with cap <= 64.
- [ ] No new unproven path-Hastings formula exists.

## Reuse

- [ ] No new graph state.
- [ ] `StrengthState` degree caches reused.
- [ ] `Cycle4Proposal` reused and extended minimally.
- [ ] finished fixed-sE `K_E` reused whole.
- [ ] same `degree_auxiliary_step` reused by repair and production.
- [ ] existing fixed-sE construction/edge repair reused.
- [ ] fixed-strength fixed-pair residualization reused.
- [ ] `CompleteMinus` reused.
- [ ] `merge_fixed_pairs` reused.

## Feasibility/repair

- [ ] Positive/zero fixed degree residualization correct.
- [ ] Combined node strength/degree bounds validated.
- [ ] Domain row/column degree caps computed sparsely.
- [ ] Tiny feasible degree repair succeeds.
- [ ] Degree repair exhaustion structured.
- [ ] No inexact-degree state enters burn-in.

## Scalability

- [ ] No N^2 state/domain allocation.
- [ ] Per-cycle degree delta O(1).
- [ ] No O(N) degree scan in auxiliary hot path.
- [ ] Trace undo uses bounded flat log, not state clone.
- [ ] N=1000 ME passes.
- [ ] N=1000 W passes.
- [ ] N=1000 B passes.
- [ ] N=1000 fixed-pair CompleteMinus passes.
- [ ] Known nontrivial N=1000 cases show different-state trace returns.
- [ ] Trace timeout/return diagnostics are reported.
- [ ] N=5000 smoke run if practical.

## API

- [ ] pyo3 binding exists.
- [ ] unified no-fixed native route exists.
- [ ] Python microcanonical `STRENGTH_DEGREE` branch exists.
- [ ] fixed pairs remain Rust-residualized.
- [ ] capability is no-fit.
- [ ] exactness is `EXACT_STATIONARY_MCMC`.
- [ ] method is `microcanonical_fixed_strength_degree`.

## Regression

- [ ] fixed-s tests pass.
- [ ] fixed-sE exact tests pass.
- [ ] fixed-sE scalability passes.
- [ ] fixed-kT tests pass.
- [ ] full Rust checks pass.
- [ ] full Python checks pass.
- [ ] relevant heavy tests pass.

---

# 54. Required implementation-agent handoff report

Report exactly:

## Branch/code

```text
branch
commits
files added
files materially changed
```

## Exact oracle

```text
number of fixed-(s,k) fibers enumerated
lambda used in oracle
trace cap selected
max auxiliary detailed-balance residual
max auxiliary stationarity residual
max trace detailed-balance residual
max trace stationarity residual
number of disconnected mandatory trace fibers
smallest disconnected case if any
```

## Degree repair by family

```text
initial degree distance
final/best distance
repair K_E steps
repair accepted endpoints
repair restarts
repair time
```

## N=1000 trace behavior

For each family:

```text
N
E
T
fixed pair count
exact-E preparation time
degree repair time
MCMC time

trace top-level attempts
step-1 returns
departures
successful returns after departure
timeouts
different-state returns
mean K_E steps per trace
mean/max degree distance
underlying K_E counters
```

## Tuning

State:

```text
final degree_trace_lambda
final degree_trace_max_steps
why those values were selected
```

## Tests

List every command and status.

## Limitations

State any unproven or poorly mixing case.

Do not say only `seems correct` or `looks fine`.

---

# 55. Final architecture

```text
family base-measure target
        |
        v
existing exact fixed-strength machinery
        |
        v
existing exact fixed-(s,E) production kernel K_E
        |
        +---- target degrees imply E
        |
        v
new degree auxiliary transition
        |
        | proposal: one complete recorded K_E transition
        |
        | outer acceptance:
        | min(1, exp[-lambda * Delta degree_distance])
        |
        v
+-----------------------------------------------+
| initialization                                |
| repeat same degree auxiliary transition       |
| until exact degree vector D=0                 |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| production fixed-(s,k)                        |
| start at D=0                                  |
| repeatedly run degree auxiliary transition    |
| stop on FIRST RETURN to D=0                   |
| cap timeout => undo excursion => self-loop    |
+-----------------------------------------------+
        |
        v
exact fixed-(s,k) sample
```

The key mathematical simplification is:

```text
pi_(s,k) = pi_(s,E)( . | k = k_target )
```

and the key reversible-kernel cancellation is:

```text
K_E reversible for pi_(s,E)
=> outer degree MH only needs exp(-lambda * Delta D)
```

The key engineering simplification is:

```text
reuse the whole finished fixed-sE transition as a black-box reversible proposal
```

rather than attempting to make each raw fixed-E move preserve every node degree.

The key performance principle is:

```text
temporary degree violations are useful, not wasted proposals;
only strengths and E remain exact during the excursion,
and exact degrees are enforced at first-return endpoints.
```

The key implementation-reuse principle is:

```text
one degree_auxiliary_step
used both for initialization repair and for the production trace.
```

---

