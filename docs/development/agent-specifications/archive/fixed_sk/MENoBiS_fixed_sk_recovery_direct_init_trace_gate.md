# MENoBiS fixed-(s,k) recovery plan
## Direct exact initialization + large-N trace mobility gate

**Repository:** `uladribia/menobis`  
**Starting branch:** `feature/microcanonical-fixed-strength-degree`  
**Recovery branch:** `fix/fixed-sk-direct-init-trace-gate`  
**Status at start:** STOPPED at the N=1000 degree-repair gate  
**Primary status document:** `docs/development/agent-specifications/STATUS.md`  
**Decision records:**  
- `docs/decisions/microcanonical-fixed-sk-stop.md` — the original STOP (old repair)  
- `docs/decisions/microcanonical-fixed-sk-trace-mobility.md` — **Gate A outcome: `TRACE_FROM_EXACT_STATE_VIABLE = true`**  
- `docs/decisions/microcanonical-fixed-sk-direct-init.md` — **Gate B outcome: N=1000 heterogeneous construction blocked (co-joint extras transport)**  
**Old implementation plan:** `MENoBiS_fixed_sk_implementation_plan_v2.md` (same folder)

---

# 0. Read this first

This document **supersedes the initialization/degree-repair part** of
`MENoBiS_fixed_sk_implementation_plan_v2.md`.

Do **not** restart the old degree-repair approach.

The current branch already contains valuable and mathematically validated work:

- residual fixed-degree targets;
- fixed-pair degree residualization;
- exact `(s,k)` target validation;
- O(1)-per-cycle degree-delta metadata;
- recordable exact fixed-`(s,E)` kernel `K_E`;
- the degree-biased auxiliary kernel;
- the capped first-return degree trace;
- exact tiny-state transition-matrix oracles for the auxiliary and trace kernels;
- a one-shot fixed-`(s,k)` core sampler;
- a reproducible N=1000 STOP artifact showing that the old **initialization repair** does not scale.

Do **not** throw those pieces away.

The failure observed on the branch is:

```text
random exact-(s,E) state
    -> degree-biased MCMC repair
    -> cannot reach exact k at scale
```

The replacement architecture is:

```text
construct exact k support directly
    -> put occupation 1 on every support edge
    -> allocate remaining strengths on that support
    -> obtain an exact (s,k) state directly
    -> start the already-implemented exact first-return trace
```

However, before investing in the new constructor, first answer a more important question:

> **Is the existing stationary degree trace practically mobile at N=1000 when it starts from an already exact `(s,k)` state?**

The current N=1000 test never answers this because it dies during initialization before the trace runs.

Therefore this recovery task has **two sequential gates**:

1. **Gate A — stationary trace mobility from an exact witness.**
2. **Gate B — direct exact `(s,k)` constructor.**

If Gate A is clearly red, STOP before implementing Gate B and report the trace problem.
If Gate A is viable, implement Gate B and integrate it.

---

# 1. Mathematical target — do not change it in this task

The target remains the fixed-strength + fixed-directed-degree microcanonical law

\[
\pi_{s,k}(t) \propto \prod_{ij} d_F(t_{ij})
\]

over occupation tables satisfying exactly

\[
\sum_j t_{ij}=s_i^{out}, \qquad \sum_i t_{ij}=s_j^{in},
\]

and

\[
\sum_j \mathbf 1[t_{ij}>0]=k_i^{out}, \qquad
\sum_i \mathbf 1[t_{ij}>0]=k_j^{in}.
\]

The degree vectors imply

\[
E^*=\sum_i k_i^{out}=\sum_j k_j^{in}.
\]

The identity used by the current implementation is still

\[
\pi_{s,k}=\pi_{s,E}(\cdot \mid k=k^*).
\]

The current degree-distance auxiliary target is

\[
\mu_\lambda(x)\propto \pi_{s,E}(x)\exp[-\lambda D(x)]
\]

with

\[
D(x)=\frac12\left[
\sum_i |k_i^{out}(x)-k_i^{out,*}|
+
\sum_j |k_j^{in}(x)-k_j^{in,*}|
\right].
\]

One complete fixed-`(s,E)` transition `K_E` is used as the proposal, and because
`K_E` is reversible for `pi_(s,E)`, the outer degree-potential MH acceptance is

\[
\alpha(x,y)=\min\left(1,\exp[-\lambda(D(y)-D(x))]\right).
\]

The production fixed-`(s,k)` kernel is the capped first-return trace of this
auxiliary chain onto

\[
A_k=\{x:D(x)=0\}.
\]

The exact tiny-state oracle on the current branch already verifies the relevant
row sums, detailed balance, stationarity and tiny-fiber connectivity.

**Do not rederive or rewrite this stationary kernel during Gate A or Gate B.**

---

# 2. The key correction to the old plan

The old plan incorrectly required initialization to reuse the same reversible
degree-biased MCMC used by the stationary sampler.

That requirement must be removed.

Initialization does **not** need:

- detailed balance;
- reversibility;
- the target distribution;
- an MH acceptance rule;
- a first-return argument.

Initialization only needs to return **one valid state** satisfying all hard
constraints.

Therefore the new direct initializer is allowed to use:

- greedy construction;
- deterministic allocation;
- randomized retries;
- max-flow / augmenting paths;
- support reconstruction;
- support-preserving combinatorial heuristics.

The starting-state distribution may be biased.

Correctness of final sampling comes from the stationary MCMC kernel plus
burn-in, not from drawing the starting state from `pi_(s,k)`.

---

# 3. Branch policy

Create a new branch from the current stopped feature branch.

Suggested name:

```text
fix/fixed-sk-direct-init-trace-gate
```

Base it on:

```text
feature/microcanonical-fixed-strength-degree
```

Do not branch from `master`, because the exact degree trace and its oracles live
only on the feature branch.

Before modifying code:

```bash
git status
git branch --show-current
git log --oneline -12
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Expected baseline:

- the ordinary workspace suite is green;
- exact fixed-`(s,k)` tiny oracle is green;
- fixed-`(s,E)` regressions are green;
- the ignored N=1000 fixed-`(s,k)` STOP artifact still fails by
  `DegreeRepairExhausted` as documented.

Commit nothing until the baseline is recorded.

---

# 4. Files that matter

Read these files before editing:

```text
docs/development/agent-specifications/STATUS.md
docs/decisions/microcanonical-fixed-sk-stop.md

MENoBiS_fixed_sk_implementation_plan_v2.md

crates/menobis-core/src/generation/microcanonical/occupation_mcmc/fixed_degrees.rs
crates/menobis-core/src/generation/microcanonical/occupation_mcmc/fixed_edges.rs
crates/menobis-core/src/generation/microcanonical/occupation_mcmc/chain.rs
crates/menobis-core/src/generation/microcanonical/occupation_mcmc/domain.rs
crates/menobis-core/src/generation/microcanonical/occupation_mcmc/state.rs
crates/menobis-core/src/generation/microcanonical/occupation_mcmc/problem.rs

crates/menobis-core/src/generation/microcanonical/binary/initializer.rs
crates/menobis-core/src/generation/microcanonical/binary/state.rs
crates/menobis-core/src/generation/microcanonical/binary/switch.rs
crates/menobis-core/src/generation/microcanonical/binary/sampler.rs

crates/menobis-test-oracles/tests/fixed_strength_degree_enumeration.rs
crates/menobis-test-oracles/tests/fixed_strength_degree_scalability.rs
```

Do not start by editing Python, pyo3, routing, capability tables, or documentation
for users.

Those remain blocked until the Rust scalability gates pass.

---

# PART A — GATE A: TEST THE EXISTING TRACE FROM D=0

# 5. Goal of Gate A

The current N=1000 STOP test answers:

```text
Can the old degree-repair MCMC reach D=0 from a random exact-E state?
```

Answer: no.

It does **not** answer:

```text
If we are already at D=0, is the exact degree trace mobile enough to sample?
```

Gate A must answer the second question before any sampler redesign.

Use a known feasible table as the starting state.

The scalability test already has a helper that generates a feasible table and
derives from it:

```text
s_out
s_in
k_out
k_in
E
```

That table itself is an exact witness on `A_k`.

Therefore:

```text
DO NOT run:
compressed fixed-s construction
edge repair
degree repair
```

for Gate A.

Start the trace directly from the witness.

---

# 6. Minimal benchmark entry point

Do not expose a user-facing API.

Add one small Rust diagnostic helper that accepts an already exact table and
runs the existing trace.

Preferred location:

```text
crates/menobis-core/src/generation/microcanonical/occupation_mcmc/chain.rs
```

Suggested diagnostic signature:

```rust
#[doc(hidden)]
pub fn benchmark_fixed_sk_trace_from_exact_table(
    problem: FixedStrengthProblem,
    full_degree_out: Vec<u32>,
    full_degree_in: Vec<u32>,
    exact_table: Vec<((u64, u64), OccNum)>,
    trace_attempts: usize,
    seed: u64,
    degree_config: DegreeTraceConfig,
) -> Result<FixedSkTraceBenchmark, FixedStrengthError>
```

The exact signature may vary if current types make a nearby shape cleaner.

The helper must:

1. store the full targets before residualization;
2. for Gate A initially require **no fixed pairs**;
3. residualize/validate using the same degree target logic as normal code;
4. construct `StrengthState` directly from `exact_table`;
5. verify, before running the trace:
   - exact residual strengths;
   - exact residual out-degrees;
   - exact residual in-degrees;
   - exact residual E;
   - family capacity;
   - domain admissibility;
6. compute `D` independently by a full scan;
7. require `D == 0`;
8. create the existing:
   - `StrengthTarget`;
   - `FixedEdgeCounters`;
   - `DegreeTraceCounters`;
   - flat `Vec<Cycle4Proposal>`;
9. run exactly `trace_attempts` top-level `degree_trace_step` calls, or an
   equivalent number through `degree_trace_sweep`;
10. require `D == 0` after every top-level trace;
11. return counters and wall time.

Do not duplicate the trace algorithm.

---

# 7. Add one missing diagnostic: support movement

The existing counters already distinguish:

```text
trace_attempts
timeouts
step1_returns
departures
successful_returns
different_state_returns
auxiliary_steps
outer_accepts
outer_rejects
max_excursion_distance
```

These are useful but `different_state_returns` includes occupation-only changes.

For fixed degrees we need to know whether the **support topology** actually
moves.

Add:

```rust
pub support_changed_returns: u64,
```

to `DegreeTraceCounters`, but only if this can be done without adding a new
O(E) cost to every production trace.

Definition:

```text
increment only when:
    the top-level trace returns to D=0
    AND the endpoint support set differs from the origin support set
```

Do not define it as "some intermediate proposal changed support".

If exact endpoint support comparison would add a new expensive hot-path scan,
keep this metric benchmark-only and compare support at periodic checkpoints.

Never add an O(N²) support comparison.

---

# 8. Gate A benchmark cases

Start with:

```text
family = ME
N = 1000
mean out-degree ≈ 8
self_loops = false
```

Use a known feasible support and derive all constraints from the witness.

Vary occupation patterns to vary `T/E`.

Mandatory ME cases:

```text
A1: all occupied pairs occupation 1        -> T/E = 1
A2: occupations 1 or 2, roughly balanced  -> T/E ≈ 1.5
A3: all occupied pairs occupation 2        -> T/E = 2
A4: all occupied pairs occupation 3        -> T/E = 3
A5: all occupied pairs occupation 5        -> T/E = 5
A6: all occupied pairs occupation 10       -> T/E = 10
```

For each case initially run:

```text
lambda = 1.0
trace cap = 16
trace attempts = 100_000
fixed reported seed
```

A 10,000-attempt smoke test is allowed first if needed, but the final decision
requires 100,000 attempts on the key cases.

Record:

```text
N
E
T
T/E
fraction of occupied pairs with occupation 1

trace_attempts
step1_returns
departures
successful_returns
different_state_returns
support_changed_returns
timeouts
auxiliary_steps
outer_accepts
outer_rejects
max_excursion_distance

wall_time
auxiliary_steps / trace_attempt
auxiliary_steps / different_state_return
auxiliary_steps / support_changed_return
different_state_return_rate
support_changed_return_rate
timeout_rate
```

Handle division by zero explicitly.

---

# 9. Gate A decision policy

These are **engineering diagnostics**, not mathematical correctness thresholds.

## GREEN

Initially healthy:

```text
different_state_returns / trace_attempts >= 1e-2
```

with clear nonzero support movement.

If key ME ratios up to `T/E = 5` are green, continue to family checks.

## YELLOW

If:

```text
1e-4 <= different_state_return_rate < 1e-2
```

or support movement is much rarer than occupation-only movement:

run the tuning grid in Section 10 before redesigning anything.

## RED

STOP the current stationary-kernel architecture if a representative case has:

```text
different_state_return_rate < 1e-4
```

after 100,000 attempts,

or:

```text
different_state_returns == 0
```

or:

```text
support_changed_returns == 0
```

when the known fiber has multiple supports,

or:

```text
timeout_rate > 0.95
```

and cap tuning does not fix it,

or:

```text
auxiliary_steps / different_state_return > 100_000
```

for ordinary sparse N=1000 cases.

If RED:

```text
STOP.
Do not implement the direct constructor yet.
Do not change routing.
Do not expose the capability.
Write a sampler-mobility decision record.
```

---

# 10. Gate A tuning grid for YELLOW cases

Test fixed configurations:

```text
lambda ∈ {0.25, 0.5, 1.0, 2.0}
cap    ∈ {16, 32, 64}
```

Suggested sequence:

```text
(1.0,16)
(0.5,16)
(2.0,16)
(1.0,32)
(0.5,32)
(1.0,64)
```

Only fill the rest if necessary.

Choose performance using:

```text
support_changed_returns per auxiliary K_E step
```

not only raw top-level return rate.

Do not introduce state-dependent tuning.

After any trace change:

```bash
cargo test -p menobis-test-oracles --test fixed_strength_degree_enumeration
```

must remain green.

---

# 11. Family checks after ME

If ME is not RED, repeat representative cases for W and B.

Suggested W:

```text
T/E ∈ {1, 2, 5}
```

Suggested B:

```text
M = 5
T/E ∈ {1, 2, 3, 5}
```

For B require every witness occupation:

\[
1 \le t_{ij}\le M.
\]

The `T/E = M` case is intentionally stressful.

---

# 12. Gate A deliverable

Write:

```text
docs/decisions/microcanonical-fixed-sk-trace-mobility.md
```

containing:

```text
branch / commit
benchmark generator
N / E / T / T/E
family
lambda
cap
attempts
all trace counters
derived rates
wall time
GREEN/YELLOW/RED
```

State exactly one:

```text
TRACE_FROM_EXACT_STATE_VIABLE = true
```

or:

```text
TRACE_FROM_EXACT_STATE_VIABLE = false
```

Only if true continue to Part B.

Suggested commit:

```text
test(microcanonical): benchmark fixed-(s,k) trace from exact N=1000 witnesses
```

---

# PART B — DIRECT EXACT (s,k) INITIALIZATION

# 13. Direct construction identity

Construct a binary directed support

\[
A_{ij}\in\{0,1\}
\]

with

\[
\sum_j A_{ij}=k_i^{out}, \qquad
\sum_i A_{ij}=k_j^{in}.
\]

Set one occupation on every support edge:

\[
t_{ij}=A_{ij}.
\]

Define residual strengths:

\[
r_i=s_i^{out}-k_i^{out}, \qquad
c_j=s_j^{in}-k_j^{in}.
\]

Find nonnegative integer extras `y_ij` only on support edges:

\[
\sum_j y_{ij}=r_i, \qquad
\sum_i y_{ij}=c_j.
\]

Then:

\[
t_{ij}=A_{ij}(1+y_{ij})
\]

has exact strengths and degrees.

For ME/W:

```text
y_ij >= 0
```

For B with `M` layers:

\[
0\le y_{ij}\le M-1.
\]

---

# 14. Not every exact-k support is strength-compatible

Do not assume:

```text
exact k support => residual strengths can always be allocated
```

The correct initializer is:

```text
construct exact-k support
    -> try residual-strength allocation
    -> feasible: success
    -> infeasible support: construct another exact-k support
```

Failure of one support is **not proof that the target is infeasible**.

---

# 15. Do not use the full fixed-(k,T) sampler for initialization

Reuse the binary support-construction machinery, not the full support MCMC.

Do **not** use:

```text
sample_fixed_degree_support(...)
```

just to obtain a starting state.

Reasons:

1. initial support need not be uniform;
2. final fixed-`(s,k)` trace supplies burn-in;
3. binary complement mode may construct O(N²) output;
4. this is unnecessary work.

Use the exact-degree initializer.

---

# 16. Make the binary initializer reusable with PairDomain

The fixed-strength code has:

```text
PairDomain::Complete
PairDomain::CompleteMinus
PairDomain::Sparse
PairDomain::is_admissible(src,tgt)
```

Do not turn Complete or CompleteMinus into an explicit pair list.

Minimally factor `binary/initializer.rs` to allow a caller-supplied
admissibility predicate and caller RNG.

Conceptual helper:

```rust
pub(crate) fn greedy_directed_initialize_with_admissibility<F>(
    out_degrees: &[u32],
    in_degrees: &[u32],
    self_loops: bool,
    rng: &mut impl Rng,
    is_admissible: F,
) -> Result<DegreeSupportState, FixedKTError>
where
    F: Fn(u64, u64) -> bool;
```

The exact signature may vary.

The old public:

```rust
greedy_directed_initialize(...)
```

must keep existing behavior and tests.

Fixed-`(s,k)` passes:

```rust
|src, tgt| residual.domain.is_admissible(src, tgt)
```

## Randomization

Repeated support attempts must be capable of producing different supports.

The fixed-`(s,k)` helper should consume the caller RNG and randomize tie
breaking from the first attempt.

Same seed must be reproducible.

Suggested commit:

```text
refactor(binary): expose domain-aware exact-degree support initialization
```

---

# 17. New direct initializer module

Create:

```text
crates/menobis-core/src/generation/microcanonical/occupation_mcmc/fixed_degree_init.rs
```

Register it in `occupation_mcmc/mod.rs`.

Suggested types:

```rust
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExactSkInitConfig {
    pub max_support_attempts: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ExactSkInitDiagnostics {
    pub support_attempts: usize,
    pub greedy_allocation_successes: usize,
    pub flow_fallback_attempts: usize,
    pub incompatible_supports: usize,
    pub residual_total: OccNum,
}

pub(crate) fn initialize_exact_sk(
    problem: &ResidualStrengthProblem,
    degree: &ResidualDegreeTarget,
    rng: &mut impl Rng,
    config: &ExactSkInitConfig,
) -> Result<(StrengthState, ExactSkInitDiagnostics), FixedStrengthError>
```

Initial internal default:

```text
max_support_attempts = 32
```

Do not expose it through Python.

---

# 18. ExactSk initializer top-level algorithm

Implement exactly:

```text
validate preconditions

for attempt in 0 .. max_support_attempts:
    support = construct exact residual-k support
    allocation = allocate residual strengths on support

    if allocation succeeds:
        build StrengthState
        validate every invariant
        return success

return ExactSkInitializationExhausted
```

Do not insert MH or first-return logic here.

---

# 19. Step 1 — construct exact residual-k support

Use:

```text
degree.out
degree.in_
problem.domain.self_loops_allowed()
problem.domain.is_admissible(...)
```

After construction require:

```text
edge count == degree.edge_count
out degrees == degree.out
in degrees == degree.in_
all pairs admissible
no duplicates
loop policy respected
```

Construction failure may trigger another attempt.

Do not convert construction exhaustion into a mathematical
`InvalidDegreeTarget`.

Do not add the binary graphicality heuristic as a hard infeasibility oracle.

---

# 20. Step 2 — put one on every support edge

Set:

```text
occupation(edge) = 1
```

Compute with checked subtraction:

```rust
r[i] = strength_out[i] - degree.out[i]
c[j] = strength_in[j]  - degree.in_[j]
```

Assert:

```text
sum(r) == sum(c)
sum(r) == total_strength - E
```

If residual total is zero, return the all-ones state after validation.

---

# 21. Step 3 — fast greedy residual allocation

The greedy is a speed optimization, not the correctness fallback.

Work from fresh mutable:

```text
r
c
per-edge extra capacity
```

Capacity:

```text
ME/W: residual_total
B(M): M - 1
```

Recommended heuristic:

1. choose a positive-residual row with the fewest currently usable support
   neighbors;
2. among its usable neighbors prefer a column with larger positive `c_j`;
3. allocate

\[
x=\min(r_i,c_j,\text{edge remaining capacity});
\]

4. update;
5. repeat.

Success iff all row and column residuals reach zero.

If greedy gets stuck:

```text
discard the partial greedy allocation
run sparse max-flow from the original residual vectors
```

Do not implement greedy backtracking.

---

# 22. Step 4 — sparse max-flow fallback

Implement a small private integer max-flow, preferably Dinic.

No new general graph framework.

Flow graph:

```text
source -> row i       capacity r_i
row i -> column j     only for support edges
column j -> sink      capacity c_j
```

Row-column capacity:

```text
ME/W: residual_total
B(M): M - 1
```

Memory must be:

\[
O(N+E).
\]

Never enumerate non-support pairs.

Let:

\[
R=\sum_i r_i.
\]

If:

```text
max_flow == R
```

extract `y_ij` and set:

```text
occupation_ij = 1 + y_ij
```

If:

```text
max_flow < R
```

the **support** is incompatible.

Retry another exact-k support.

Do not call the whole target infeasible.

---

# 23. Mandatory max-flow tests

## 23.1 trivial

One row, one column, residual 5, one support edge.

Expected max flow 5.

## 23.2 greedy trap but flow succeeds

Support:

```text
A -> X
A -> Y
B -> X
```

Residuals:

```text
r_A = 1
r_B = 1
c_X = 1
c_Y = 1
```

Flow must find:

```text
A -> Y = 1
B -> X = 1
```

even if a naive greedy could get stuck.

## 23.3 incompatible support

Positive residual needs a column unreachable from the supplying rows.

Require max flow < total residual.

## 23.4 B capacity

Construct a case feasible without edge capacity but infeasible with
`M-1`.

Require incompatible support.

## 23.5 extraction

Extracted edge flows reproduce row/column residuals exactly.

---

# 24. Exact initializer invariants

Before returning, independently verify:

\[
\sum_j t_{ij}=s_i^{out},
\qquad
\sum_i t_{ij}=s_j^{in}.
\]

And:

\[
\sum_j 1[t_{ij}>0]=k_i^{out},
\qquad
\sum_i 1[t_{ij}>0]=k_j^{in}.
\]

Also:

```text
occupied_count == degree.edge_count
all occupations > 0
all pairs unique
all pairs admissible
self-loop policy exact
B: occupation <= M
```

After `StrengthState::new` require:

```text
state.out_strengths == residual s_out
state.in_strengths == residual s_in
state.row_occ_count == residual k_out
state.col_occ_count == residual k_in
degree_distance(...) == 0
state.occupied_count() == residual E
```

Only then may burn-in start.

---

# 25. Fixed pairs

Do not special-case them in the allocator.

Reuse current residualization.

Positive fixed pair:

```text
subtract full occupation from strengths
subtract 1 from degree
remove coordinate from residual domain
```

Zero fixed pair:

```text
subtract nothing from strengths/degrees
keep coordinate forbidden
```

The direct initializer works only on residual `(s,k,domain)`.

After sampling use existing merge and full validation.

Mandatory tests:

```text
positive fixed pair
zero fixed pair
multiple fixed pairs
CompleteMinus
```

Never materialize the CompleteMinus domain.

---

# 26. Integrate into one-shot fixed-(s,k)

Current active path:

```text
fixed-s construction
-> structural repair
-> edge repair
-> degree repair
-> trace
```

Replace with:

```text
residualize strengths/domain
-> residualize degrees
-> validate
-> initialize_exact_sk()
-> assert D=0
-> trace burn-in
-> trace thinning
-> merge fixed pairs
-> full validation
```

Do not delete fixed-s or fixed-sE construction/repair code.

The old `repair_to_degree_target` may remain temporarily for historical tests,
but production fixed-`(s,k)` must no longer call it.

Add diagnostics:

```text
direct_init_time_s
support_attempts
greedy_allocation_success
flow_fallback_attempts
incompatible_supports
```

Suggested commit:

```text
feat(microcanonical): construct exact fixed-(s,k) starting states directly
```

---

# 27. Preserve the old failure evidence

Do not erase the documented fact that the old repair fails.

Keep the decision record.

The old ignored STOP artifact may remain as a historical regression.

Add a new ignored test:

```text
n1000_direct_sk_initialization
```

with:

```text
N = 1000
mean degree ≈ 8
self_loops = false
known feasible witness-derived s,k
ME first
```

Require exact `D=0` and report support attempts + wall time.

---

# 28. Constructor stress grid

After basic N=1000 success:

## ME

```text
N=1000
d ∈ {4,8,16}
T/E ∈ {1,2,5,10}
```

## W

```text
N=1000
d=8
T/E ∈ {1,2,5}
```

## B

```text
M=5
N=1000
d=8
T/E ∈ {1,2,3,5}
```

Structural variants:

```text
loops off/on
Complete
CompleteMinus positive fixed pairs
CompleteMinus zero fixed pairs
```

Report:

```text
support attempts
greedy outcome
flow fallbacks
incompatible supports
construction time
```

---

# 29. Combined N=1000 gate

Only after Gate A and direct initialization pass:

```text
direct exact (s,k) init
-> trace burn-in
-> trace thinning
-> exact final output
```

For ME/B/W require:

```text
full s_out exact
full s_in exact
full k_out exact
full k_in exact
domain exact
family capacity exact
fixed pairs exact
```

Report initialization + trace diagnostics + wall time.

---

# 30. Correctness vs mixing

Keep these separate in every report:

```text
EXACTNESS
    tiny exact Q/R transition matrix oracle

PRACTICAL MOBILITY
    N=1000 trace from exact D=0 state

INITIALIZATION
    direct combinatorial exact-sk constructor
```

Do not claim the N=1000 chain is mathematically proven to mix rapidly.

If benchmarks are good, say:

```text
the exact kernel shows acceptable empirical mobility on the mandatory N=1000 suite
```

---

# 31. Exact oracle regression

After changes touching trace/kernel/state:

```bash
cargo test -p menobis-test-oracles --test fixed_strength_degree_enumeration
```

If fixed-E machinery changes:

```bash
cargo test -p menobis-test-oracles --test fixed_strength_edges_enumeration
```

The direct initializer is not a stationary kernel and does not need a
detailed-balance proof.

---

# 32. Resume public routing only after Rust gates

Only if all are true:

```text
TRACE_FROM_EXACT_STATE_VIABLE = true
DIRECT_EXACT_SK_INITIALIZATION = pass at N=1000
combined N=1000 ME/B/W = pass
```

resume old phases:

1. SamplingPlan priority correction;
2. guard against silently ignored strengths;
3. pyo3;
4. Python wrapper;
5. STRENGTH_DEGREE routing;
6. E2E;
7. capability exposure last.

---

# 33. Error handling

Add a structured error such as:

```rust
ExactSkInitializationExhausted {
    support_attempts: usize,
    best_flow: OccNum,
    residual_total: OccNum,
}
```

Do not label the target infeasible merely because the retry budget is exhausted.

Keep `DegreeRepairExhausted` while the historical repair tests exist.

---

# 34. Reproducibility

All randomization uses the existing sampler RNG or an explicitly derived seeded
RNG.

Same problem + target + seed + config must reproduce:

```text
support attempts
initial exact state
final sample
```

No hidden `thread_rng()`.

Max-flow should be deterministic for fixed support/edge order.

---

# 35. Complexity requirements

Hard:

```text
state memory        O(E)
flow graph memory   O(N+E)
fixed-pair exclusion O(F)
degree delta        O(1) per cycle
```

Forbidden:

```text
N x N matrices
explicit complete admissible-pair vectors
O(T) per-event stub expansion
full-state clone per auxiliary substep
O(N) degree scan per K_E step
```

The existing binary initializer may still have nonideal large-N time because it
scans candidate nodes. Do not redesign that before N=1000 functionality is
proven.

Measure N=1000 first; optionally N=5000 later.

---

# 36. TDD order — follow exactly

## Phase 0
Baseline only. No commit.

## Phase 1
Trace-from-exact-state diagnostic helper.

Commit.

## Phase 2
Support-mobility diagnostic if cheap.

Run exact oracle.

Commit.

## Phase 3
N=1000 trace mobility grid.

GREEN/YELLOW/RED decision.

If RED: STOP.

Write decision record.

Commit.

## Phase 4
Domain-aware randomized binary support initializer.

Run binary + workspace tests.

Commit.

## Phase 5
Residual allocation tests first.

Implement greedy + sparse flow.

Commit.

## Phase 6
Direct `initialize_exact_sk`.

Tiny ME/B/W + fixed-pair tests.

Commit.

## Phase 7
N=1000 direct-initializer gate.

If retries/failures pathological: STOP and report.

Commit.

## Phase 8
Integrate direct initializer into one-shot sampler.

Old degree repair no longer in production path.

Commit.

## Phase 9
Full N=1000 ME/B/W + fixed-pair E2E.

Commit.

## Phase 10
Cleanup old degree-repair production dead code only after Phase 9.

Commit.

## Phase 11
Routing/pyo3/Python only after all gates.

---

# 37. Mandatory tiny initializer cases

1. **All ones:** `s=k`, residual total zero.
2. **Simple feasible extras:** exact support plus nonzero residual allocation.
3. **Greedy trap:** greedy can fail but max-flow succeeds.
4. **Incompatible support:** allocation fails cleanly, no invalid output.
5. **B capacity:** `M-1` extra capacity enforced.
6. **Positive fixed pair.**
7. **Zero fixed pair.**

At least one test must show:

```text
one exact-k support incompatible with s
another exact-k support compatible
```

or test those two supports separately if deterministic retry control is awkward.

---

# 38. Trace report questions

Explicitly answer:

1. How often does the trace return to a different exact state?
2. How often does the endpoint support change?
3. How many `K_E` calls per effective return?
4. How does this depend on `T/E`?
5. How does lambda/cap affect it?
6. Are losses due to self-loops, outer rejection, timeout, or same-state return?
7. Does support mobility collapse when occupation-1 edges are rare?

Do not report only an acceptance rate.

---

# 39. Things the agent must NOT do

- Do not continue production initialization with `repair_to_degree_target`.
- Do not increase the old repair budget.
- Do not infer stationary trace failure from old initialization failure.
- Do not redesign trace before Gate A.
- Do not change target mathematics.
- Do not add path-level Hastings terms.
- Do not use fixed-(k,T) factorization as final fixed-(s,k) sampling.
- Do not use full binary support MCMC merely for initialization.
- Do not materialize Complete/CompleteMinus as N² pair lists.
- Do not assume every exact-k support supports the strengths.
- Do not treat one incompatible support as global infeasibility.
- Do not let greedy failure equal allocation failure.
- Do not return approximate k.
- Do not enable routing/Python capability before N=1000 gates.
- Do not weaken the exact Q/R oracle.
- Do not alter fixed-(s,E) behavior without necessity.
- Do not build a generic constraint or graph framework.
- Do not optimize N=25k before N=1000 is demonstrated.

---

# 40. Commands before handoff

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Exact fixed-sk:

```bash
cargo test -p menobis-test-oracles --test fixed_strength_degree_enumeration
```

Fixed-sE regression:

```bash
cargo test -p menobis-test-oracles --test fixed_strength_edges_enumeration
```

Heavy:

```bash
cargo test -p menobis-test-oracles   --test fixed_strength_degree_scalability   -- --ignored --nocapture
```

Python commands only after Rust/public-integration phases:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

---

# 41. Required handoff report

## Branch and commits

```text
branch
base
commits
files added
files changed
```

## Trace decision

Exactly:

```text
TRACE_FROM_EXACT_STATE_VIABLE = true/false
```

Table:

```text
family
N
E
T/E
lambda
cap
attempts
different returns
support changed returns
timeouts
aux steps
aux/different return
wall time
classification
```

## Direct initialization

Exactly:

```text
DIRECT_EXACT_SK_INITIALIZATION = pass/fail
```

Table:

```text
family
N
E
T/E
support attempts
greedy success
flow fallbacks
incompatible supports
init time
```

## Exactness

```text
Q oracle
R oracle
max DB residual
max stationarity residual
tiny connectivity
```

## N=1000 E2E

```text
ME
W
B
fixed pairs
```

For each: exact constraints, init time, MCMC time, trace mobility.

## Regression commands

Every command with PASS/FAIL.

## Remaining risks

```text
constructor risks
trace mixing risks
large-N runtime risks
unproven global mixing/connectivity
```

---

# 42. Definition of done

## Trace

- [ ] Runs from exact N=1000 witness without repair.
- [ ] Top-level boundaries always D=0.
- [ ] Different-state return measured.
- [ ] Support movement measured.
- [ ] K_E cost per effective return measured.
- [ ] T/E sensitivity measured.
- [ ] Decision recorded.
- [ ] Exact Q/R oracle green.

## Direct initializer

- [ ] Direct exact-k support construction.
- [ ] One occupation per support edge.
- [ ] Residual strengths `s-k`.
- [ ] Greedy allocator.
- [ ] Sparse exact flow fallback.
- [ ] B capacity `M-1`.
- [ ] Incompatible support retries.
- [ ] Fixed pairs via residualization.
- [ ] No N² domain materialization.
- [ ] N=1000 pass.

## Integrated sampler

- [ ] Production path no longer calls degree repair.
- [ ] Burn-in starts exactly at D=0.
- [ ] N=1000 ME/W/B pass.
- [ ] Fixed-pair case passes.
- [ ] Full output validation passes.
- [ ] Fixed-s, fixed-sE, fixed-kT regressions green.

## Public integration

- [ ] Routing only after all above.
- [ ] pyo3.
- [ ] Python.
- [ ] capability enabled last.

---

# 43. Final intended architecture

```text
full (s,k) target + fixed pairs
             |
             v
existing residualization
             |
             +--> residual s
             +--> residual k
             +--> residual PairDomain
             |
             v
exact-k binary support constructor
             |
             v
put 1 on every support edge
             |
             v
r = s_out - k_out
c = s_in  - k_in
             |
             v
fast greedy residual allocation
             |
       success? ---- yes -------------------+
             |                              |
             no                             |
             v                              |
sparse integer max-flow on support          |
             |                              |
       feasible? --- yes -------------------+
             |
             no
             v
construct another exact-k support
             |
             v
exact residual (s,k) StrengthState
             |
             | D=0 by construction
             v
existing exact degree-biased first-return trace
             |
             v
burn-in + thinning
             |
             v
merge fixed pairs
             |
             v
full exact output validation
```

The direct constructor is combinatorial and does not need stationarity.

The sampler remains the already-oracle-validated exact trace kernel **unless
Gate A demonstrates unacceptable practical mobility**.

That separation is the core rule of this recovery.
