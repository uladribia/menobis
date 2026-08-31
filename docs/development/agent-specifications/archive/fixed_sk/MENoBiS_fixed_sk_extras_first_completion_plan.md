# MENoBiS fixed-(s,k) — extras-first constructor and completion plan

**Repository:** `uladribia/menobis`  
**Required branch:** `fix/fixed-sk-direct-init-trace-gate`  
**Purpose:** replace the blocked support-first fixed-`(s,k)` initializer with an
extras-first constructor, integrate it with the already-validated stationary
trace, finish public routing only after N=1000 gates pass, and clean up the
accumulated fixed-`(s,k)` agent documentation.

Suggested live path while this task is active:

```text
docs/development/agent-specifications/MENoBiS_fixed_sk_extras_first_completion_plan.md
```

---

# 0. Read this first

Do not repeat either failed initialization approach already documented on this
branch.

Current branch evidence establishes:

1. the original degree-biased MCMC **initialization repair** does not scale;
2. the stationary fixed-`(s,k)` first-return trace **does** work from realistic
   exact `(s,k)` states at N=1000;
3. the second initializer, which first creates a generic exact-`k` support and
   then tries to fit the residual strengths `s-k` inside it, fails
   systematically for heterogeneous realistic instances;
4. the same heterogeneous residual-strength transport is easy if it can choose
   its own sparse support;
5. that extras support is much smaller than the requested final edge count, so
   the missing degree slots can potentially be completed afterwards with
   occupation-1 edges.

The new order is therefore:

```text
NOT:
    exact-k support
    -> try to fit strengths inside it

NEW:
    allocate strength extras first
    while using at most k support slots
    -> complete missing k slots with occupation-1 filler edges
    -> exact (s,k) state
    -> existing stationary trace
```

Core design rule:

> **The extras determine the hard row/column co-joint structure.  
> Exact degrees are completed afterwards.**

Do not redesign the stationary sampler in this task.

---

# 1. Required context

Read these files in this exact order before coding:

```text
docs/development/agent-specifications/STATUS.md
docs/development/agent-specifications/MENoBiS_fixed_sk_recovery_direct_init_trace_gate.md

docs/decisions/microcanonical-fixed-sk-trace-mobility.md
docs/decisions/microcanonical-fixed-sk-direct-init.md
docs/decisions/microcanonical-fixed-sk-stop.md
```

Then inspect:

```text
crates/menobis-core/src/generation/microcanonical/occupation_mcmc/
    fixed_degree_init.rs
    fixed_degrees.rs
    fixed_edges.rs
    chain.rs
    domain.rs
    problem.rs
    state.rs

crates/menobis-core/src/generation/microcanonical/binary/
    initializer.rs
    state.rs
    switch.rs
    sampler.rs

crates/menobis-test-oracles/tests/
    fixed_strength_degree_trace_gate.rs
    fixed_strength_degree_direct_init.rs

crates/menobis-test-oracles/src/
    pa_geographic.rs
```

Do not begin with Python, pyo3, public routing, or capability tables.

---

# 2. Facts that must not be re-litigated

## 2.1 Stationary target and kernel

The target remains:

\[
\pi_{s,k}(t)\propto\prod_{ij}d_F(t_{ij})
\]

on exact strengths and exact directed degrees.

The current sampler uses:

\[
\pi_{s,k}=\pi_{s,E}(\cdot\mid k=k^*)
\]

with the degree-biased auxiliary target:

\[
\mu_\lambda(x)\propto\pi_{s,E}(x)e^{-\lambda D(x)}
\]

and:

\[
D(x)=\frac12\left[
\sum_i|k_i^{out}(x)-k_i^{out,*}|
+
\sum_j|k_j^{in}(x)-k_j^{in,*}|
\right].
\]

Because the complete fixed-`(s,E)` transition `K_E` is reversible for
`pi_(s,E)`, the outer MH term is:

\[
\alpha(x,y)=\min\{1,e^{-\lambda(D(y)-D(x))}\}.
\]

The production kernel is the capped first-return trace onto `D=0`.

The tiny exact Q/R transition-matrix oracle already validates row sums,
detailed balance and stationarity.

**Do not rewrite this mathematics.**

## 2.2 Gate A passed

Realistic N=1000 PA-geographic witnesses show roughly:

```text
E ≈ 8000
mean degree ≈ 8
T/E ≈ 8
occupation-1 fraction ≈ 0.18

different exact-state return rate ≈ 3%
support-changing return rate      ≈ 3%
K_E calls/effective return        ≈ 34
timeout rate                      ≈ 0.2%
```

So the sampler is usable on the target class we care about.

Uniform corners with no occupation-1 edges can be nearly immobile. The new
constructor must therefore report the final occupation-1 fraction, but must not
reject a mathematically valid target solely because this number is small.

## 2.3 Support-first construction failed

The current blocked initializer does:

```text
generic exact-k support
-> one unit per edge
-> residual extras r=s_out-k_out, c=s_in-k_in
-> greedy / exact max-flow on that fixed support
```

On heterogeneous N=1000 cases, hundreds of such supports fail. The witness
support succeeds. Therefore `k` marginals alone do not determine the necessary
row-column correlation.

Do not add more retries or more variants of the same support-first idea.

## 2.4 Positive evidence for extras-first

The current investigation found a full-domain extras transport that is:

- feasible;
- sparse;
- about 1752 positive extras edges on the representative N=1000 case;
- within every row `k_out` cap;
- only one column over its `k_in` cap, and only by one edge.

This is the evidence motivating the new constructor.

---

# 3. Mathematical decomposition

We need a final occupation table `t` satisfying:

\[
\sum_jt_{ij}=s_i^{out},\qquad
\sum_it_{ij}=s_j^{in},
\]

and:

\[
\sum_j\mathbf1[t_{ij}>0]=k_i^{out},\qquad
\sum_i\mathbf1[t_{ij}>0]=k_j^{in}.
\]

Define residual extras:

\[
r_i=s_i^{out}-k_i^{out},\qquad
c_j=s_j^{in}-k_j^{in}.
\]

Construct nonnegative integer extras `y` with:

\[
\sum_jy_{ij}=r_i,\qquad
\sum_iy_{ij}=c_j.
\]

Let:

\[
B=\{(i,j):y_{ij}>0\}.
\]

The only support condition required at this stage is:

\[
d_i^{out}(B)\le k_i^{out},\qquad
d_j^{in}(B)\le k_j^{in}.
\]

Then compute missing degree slots:

\[
\delta_i^{out}=k_i^{out}-d_i^{out}(B),
\]

\[
\delta_j^{in}=k_j^{in}-d_j^{in}(B).
\]

Construct a filler support `C`, disjoint from `B`, with exact degree vectors
`delta_out`, `delta_in`.

Final occupations are:

\[
t_{ij}=
\begin{cases}
1+y_{ij},&(i,j)\in B,\\
1,&(i,j)\in C,\\
0,&\text{otherwise}.
\end{cases}
\]

Then:

\[
\sum_jt_{ij}
=
d_i^{out}(B)+\delta_i^{out}+r_i
=
k_i^{out}+s_i^{out}-k_i^{out}
=
s_i^{out}.
\]

The same holds for columns, and support degrees are exactly `k`.

This algebra is the implementation contract.

---

# 4. Do not solve a generic cardinality-constrained flow problem

Do not introduce:

```text
MILP
fixed-charge flow
generic b-matching framework
dense N x N max-flow
new external optimization dependency
```

Implement two simple stages:

```text
Stage 1:
    slot-aware compressed extras transport
    support degrees <= k

Stage 2:
    binary exact-degree completion
    for the remaining support slots
```

Stage 2 reuses the binary initializer already present on the branch.

---

# 5. Baseline and commits

Stay on:

```text
fix/fixed-sk-direct-init-trace-gate
```

Before edits:

```bash
git status
git branch --show-current
git log --oneline -15

cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

cargo test -p menobis-test-oracles --test fixed_strength_degree_enumeration
cargo test -p menobis-test-oracles --test fixed_strength_edges_enumeration
```

Require a clean working tree and green baseline.

Implement one phase at a time. After every phase:

```text
tests
fmt
clippy
commit
```

Do not implement later phases early.

---

# PART A — EXTRAS-FIRST PROTOTYPE

# 6. Keep the prototype in `fixed_degree_init.rs`

Use:

```text
crates/menobis-core/src/generation/microcanonical/
    occupation_mcmc/fixed_degree_init.rs
```

Refactor conceptually around:

```rust
fn residual_extras(...)
fn construct_extras_slot_aware(...)
fn complete_support_to_exact_k(...)
fn build_state_from_extras_and_fillers(...)
fn validate_exact_sk_state(...)
```

Do not put constructor logic into `fixed_degrees.rs`.

# 7. Preserve old support-first code temporarily

Until the new N=1000 gate passes, do not delete:

```text
old exact-k-first support construction
allocate_residual_greedy
allocate_residual_flow
private Dinic
```

Add a separate prototype entry point such as:

```rust
pub(crate) fn initialize_exact_sk_extras_first(...)
```

Only replace the active `initialize_exact_sk` after the new gate passes.

# 8. Inputs

Inputs are already residualized:

```text
ResidualStrengthProblem
degree_out
degree_in
PairDomain
OccupationFamily
caller RNG
```

Compute with checked subtraction:

```rust
r[i] = strength_out[i] - degree_out[i]
c[j] = strength_in[j]  - degree_in[j]
```

Require:

```text
sum(r) == sum(c)
sum(k_out) == sum(k_in)
```

Fixed pairs remain the responsibility of existing residualization.

# 9. Per-edge extra capacity

For ME/W:

```text
edge_extra_cap = residual_total
```

For B with `M` layers:

\[
0\le y_{ij}\le M-1.
\]

Therefore:

```text
edge_extra_cap = M - 1
```

If `M=1`, valid residual extras must be zero.

---

# 10. Slot-aware compressed extras heuristic

Maintain:

```rust
row_mass[i]       // remaining r_i
col_mass[j]       // remaining c_j

row_slots[i]      // maximum new positive extras edges still allowed
col_slots[j]

extras_edges      // positive y edges only
extras_set        // duplicate protection
```

Initialize:

```text
row_mass = r
col_mass = c

row_slots = k_out
col_slots = k_in
```

Every new positive extras edge consumes exactly one row support slot and one
column support slot.

# 11. Hard slot invariant

Whenever a new extras edge `(i,j)` is created:

```text
row_slots[i] -= 1
col_slots[j] -= 1
```

If:

```text
row_mass[i] > 0 && row_slots[i] == 0
```

the current attempt fails.

Likewise for columns.

Abort the attempt and restart from original residuals.

Do not return a partial state.

# 12. Pressure score

Use an integer pressure:

```text
pressure(mass, slots) = ceil(mass / slots)
```

for positive mass and positive slots.

Interpretation:

```text
high pressure = much residual mass must pass through few remaining edges
```

Use integer arithmetic only.

Implement a small checked/widened `ceil_div`.

# 13. Select the row

Choose an active row with:

```text
row_mass > 0
row_slots > 0
```

Ordering:

```text
1. highest pressure(row_mass,row_slots)
2. larger row_mass
3. smaller row_slots
4. node index on deterministic attempt 0
```

Do not choose rows uniformly at random.

# 14. Select the column

For the chosen row, candidates satisfy:

```text
col_mass[j] > 0
col_slots[j] > 0
domain.is_admissible(i,j)
(i,j) not already in extras_set
```

Ordering:

```text
1. highest pressure(col_mass,col_slots)
2. larger col_mass
3. smaller col_slots
4. node index / bounded randomized top-window
```

This deliberately couples strength-heavy rows to strength-heavy columns.

# 15. Allocate a block

For chosen `(i,j)`:

\[
x=\min(row\_mass_i,col\_mass_j,edge\_extra\_cap).
\]

Require `x>0`.

Append:

```text
((i,j), x)
```

Update:

```text
row_mass[i] -= x
col_mass[j] -= x
row_slots[i] -= 1
col_slots[j] -= 1
```

For ME/W the edge cap cannot bind before at least one endpoint is exhausted, so
the extras support is naturally compressed.

For B, the edge cap can bind and more extras edges may be needed.

# 16. Never reuse an extras coordinate

If an extras coordinate already exists, do not select it again.

For B, if both endpoint masses remain after an allocation, the edge must have
hit `M-1` capacity and is saturated.

Use `extras_set`.

# 17. Domain rules

Use:

```rust
problem.domain.is_admissible(src,tgt)
```

directly.

Do not materialize all possible pairs.

Must work for:

```text
Complete
CompleteMinus
Sparse
```

# 18. First version may use O(N) candidate scans

For the first N=1000 correctness spike, O(N) scans for row/column selection are
acceptable.

Forbidden:

```text
O(N^2) memory
```

Record candidate selection as the expected one-time constructor hotspot.

Do not optimize it before correctness is demonstrated at N=1000.

# 19. Deterministic attempt 0

Attempt 0 uses exact pressure ordering and node-index tie breaks.

Same inputs must produce the same extras table.

# 20. Randomized retries

Add bounded retries.

Suggested internal config:

```rust
pub struct ExactSkInitConfig {
    pub max_extras_attempts: usize,      // 64
    pub max_completion_attempts: usize,  // 16 per extras table
    pub randomized_top_window: usize,    // 8
}
```

Do not expose in Python yet.

For attempts >0:

1. rank columns with the same pressure rule;
2. retain the best `min(window, candidates.len())`;
3. choose one with the caller RNG.

Same seed must remain reproducible.

Do not create hidden RNGs.

# 21. Prototype failure

An extras attempt either:

```text
succeeds with all row/column residual masses zero
```

or:

```text
fails because positive mass became stranded by slot/domain choices
```

On failure restart from original `r,c,k`.

Do not run old support-first max-flow as fallback.

---

# 22. Unit tests for extras construction

Write before the N=1000 test.

Mandatory:

```text
1. zero extras: s==k
2. one extras edge
3. high-pressure row paired with high-pressure column
4. row support cap never exceeds k_out
5. column support cap never exceeds k_in
6. B: extra <= M-1
7. loopless domain
8. CompleteMinus exclusion
9. reproducibility
10. retry diversity on a nontrivial fixture
```

Tests must check extras row/column sums independently.

---

# PART B — FILL MISSING DEGREE SLOTS

# 23. Compute extras support degrees

After extras success compute:

```rust
extras_k_out
extras_k_in
```

Require:

```text
extras_k_out <= target k_out
extras_k_in  <= target k_in
```

Any violation is a bug.

# 24. Compute missing degrees

Checked subtraction:

```rust
delta_out[i] = k_out[i] - extras_k_out[i]
delta_in[j]  = k_in[j]  - extras_k_in[j]
```

Require:

```text
sum(delta_out) == sum(delta_in)
sum(delta_out) == target_E - extras_edges.len()
```

# 25. Fillers must be new coordinates

Filler admissibility:

```rust
domain.is_admissible(src,tgt)
    && !extras_set.contains(&(src,tgt))
```

Never add a filler unit to an extras edge. It would change strength but not
degree.

# 26. Reuse the binary exact-degree initializer

Reuse the existing domain-aware binary helper, conceptually:

```rust
greedy_directed_initialize_with_admissibility(...)
```

Inputs:

```text
delta_out
delta_in
loop policy
caller RNG
closure excluding extras support
```

Do not invoke the full fixed-`(k,T)` MCMC just to get one filler support.

# 27. Completion retries

For one extras table:

```text
retry filler construction max_completion_attempts times
```

If every completion attempt fails:

```text
discard the extras table
start a new extras attempt
```

Do not report global infeasibility.

# 28. Zero filler case

If all deltas are zero:

```text
fillers = empty
```

This is valid.

# 29. Filler tests

Mandatory:

```text
basic exact delta completion
extras/filler disjointness
CompleteMinus exclusion
zero filler case
reproducible retry behavior
```

---

# PART C — BUILD AND VALIDATE FINAL STATE

# 30. Final table

Extras coordinate:

```text
occupation = 1 + extra
```

Filler coordinate:

```text
occupation = 1
```

No duplicates.

# 31. Independent pre-state validation

Before constructing `StrengthState`, recompute from the table:

```text
out strengths
in strengths
out degrees
in degrees
E
```

Require exact residual targets.

Also require:

```text
occupation > 0
domain admissible
loop policy
B capacity
no duplicates
```

# 32. State validation

After `StrengthState::new` require:

```text
state.out_strengths == residual s_out
state.in_strengths  == residual s_in
state.row_occ_count == residual k_out
state.col_occ_count == residual k_in
state.occupied_count() == residual E
degree_distance(...) == 0
```

Only then return success.

# 33. Occupation-1 diagnostics

Report:

```text
occupation_one_edges
occupation_one_fraction
extras_edges
filler_edges
```

Every filler is occupation 1.

Do not reject a mathematically valid target only because the one fraction is
small.

---

# 34. New diagnostics

Replace/extend `ExactSkInitDiagnostics` after the new architecture is selected:

```rust
pub struct ExactSkInitDiagnostics {
    pub extras_attempts: usize,
    pub extras_failed_attempts: usize,

    pub extras_edges: usize,
    pub filler_edges: usize,

    pub completion_attempts: usize,
    pub completion_failed_attempts: usize,

    pub residual_total: OccNum,

    pub occupation_one_edges: usize,
    pub occupation_one_fraction: f64,
}
```

Optional:

```text
max extras out-degree ratio
max extras in-degree ratio
```

Once old support-first code is removed, do not retain misleading production
fields such as `best_flow`.

# 35. Update the exhaustion error

The old error describes support-first max-flow.

After extras-first becomes active, use a structure such as:

```rust
ExactSkInitializationExhausted {
    extras_attempts: usize,
    extras_failures: usize,
    completion_failures: usize,
    residual_total: OccNum,
}
```

Retry exhaustion is not mathematical infeasibility.

---

# PART D — GATE C: N=1000 CONSTRUCTOR

# 36. Gate C1 — realistic ME

Use existing PA generator:

```text
N=1000
mean degree=8
self_loops=false
family=ME
PaGeographic { events_per_edge: 8.0 }
```

Derive constraints from witness, but **do not pass witness table or support into
the constructor**.

Require:

```text
constructor success
exact s
exact k
D=0
E exact
domain exact
```

Report:

```text
extras attempts
extras edges
filler edges
completion attempts
occupation-one fraction
wall time
```

**STOP condition:** if this fails after 64 extras attempts, stop. Do not invent
transport-cycle repair or a new optimizer.

# 37. Gate C2 — Balanced12

Use current:

```text
OccupationPattern::Balanced12
```

This heterogeneous case failed support-first.

Require success and the same diagnostics.

# 38. Gate C3 — uniform regression grid

Retain/update current grid:

```text
ME d={4,8,16}, uniform {1,2,5,10}
W d=8, uniform {1,2,5}
B M=5 d=8, uniform {1,2,3,5}
```

Must remain green.

# 39. Gate C4 — structural variants

At N=1000 include:

```text
loops enabled
CompleteMinus positive fixed pairs
CompleteMinus zero fixed pairs
```

Require exact construction.

# 40. Heterogeneous B and W

Add:

```text
B M=5 + Balanced12
representative heterogeneous W PA case
```

Do not use a B witness whose occupations exceed `M`.

# 41. Gate C decision record

Create:

```text
docs/decisions/microcanonical-fixed-sk-extras-first-init.md
```

State exactly:

```text
EXTRAS_FIRST_INITIALIZATION = pass
```

or:

```text
EXTRAS_FIRST_INITIALIZATION = fail
```

Include:

```text
commit
test command
family
N
E
T/E
extras attempts
extras edges
filler edges
completion attempts
one fraction
wall time
exactness
```

If fail, stop the task.

---

# PART E — RECHECK TRACE FROM THE ACTUAL CONSTRUCTOR OUTPUT

# 42. Do not assume witness mobility transfers automatically

Gate A started from witness states. The new constructor may generate a very
different occupation pattern.

Run the existing trace benchmark from the **constructed state**.

# 43. Gate D

At least:

```text
ME realistic PA N=1000
W realistic PA N=1000
B Balanced12 N=1000
```

Initial trace config:

```text
lambda=1
cap=16
100,000 attempts for final evidence
```

Report:

```text
one fraction
different-state rate
support-change rate
timeouts
auxiliary steps/effective return
wall time
```

# 44. Gate D interpretation

For realistic ME/W require approximately the existing engineering gate:

```text
different-state rate >= 1e-2
clear nonzero support movement
```

Do not claim rapid mixing is proven.

Degenerate targets that force no occupation-1 edges may remain mobility
warnings rather than constructor errors.

If realistic constructed states are essentially immobile, stop before public
exposure.

---

# PART F — INTEGRATE INTO THE ONE-SHOT SAMPLER

# 45. Switch `initialize_exact_sk` only after Gates C/D

Once proven, make extras-first the sole active implementation.

Production fixed-`(s,k)` path becomes:

```text
residualize strengths/domain
residualize degrees
validate combined target
initialize_exact_sk extras-first
assert D=0
trace burn-in
trace thinning
merge fixed pairs
full output validation
```

# 46. Remove old initialization stages from fixed-(s,k) path

The active fixed-`(s,k)` path must not run:

```text
compressed fixed-s initializer
structural repair
edge repair
degree repair
```

Do not alter fixed-`s` or fixed-`(s,E)` paths.

# 47. No active degree repair

Production fixed-`(s,k)` must no longer call:

```text
repair_to_degree_target
DegreeRepairConfig
```

Keep:

```text
degree_distance
degree_auxiliary_step
degree_trace_step
degree_trace_sweep
```

They belong to the stationary sampler.

# 48. Bench diagnostics

Update `FixedStrengthDegreeBench` to report:

```text
direct_init_time_s
extras_attempts
extras_edges
filler_edges
completion_attempts
occupation_one_fraction

mcmc_time_s
degree_trace counters
fixed_edge counters
```

Remove obsolete fixed-`(s,k)` degree-repair timing fields when safe.

---

# PART G — N=1000 END-TO-END GATE

# 49. ME

Run the actual one-shot sampler on realistic PA N=1000.

Require exact:

```text
s_out
s_in
k_out
k_in
E
domain
```

Report init and MCMC time separately.

# 50. W

Same for heterogeneous W.

# 51. B

Use heterogeneous B-valid case, e.g. Balanced12 with M=5.

Keep capacity corner regressions.

# 52. Fixed pairs

End-to-end:

```text
positive fixed pairs
zero fixed pairs
CompleteMinus
```

Merged output must reproduce full strengths/degrees and fixed occupations.

# 53. Exact oracle regression

Always run:

```bash
cargo test -p menobis-test-oracles --test fixed_strength_degree_enumeration
cargo test -p menobis-test-oracles --test fixed_strength_edges_enumeration
```

Do not weaken tolerances.

---

# PART H — PUBLIC ROUTING / PYTHON

Only continue if:

```text
Gate C PASS
Gate D PASS
N=1000 ME/W/B E2E PASS
fixed pairs PASS
exact oracles PASS
```

# 54. Sampling-plan priority

Strength constraints must win routing priority:

```text
if strengths present:
    occupation MCMC
else if degrees/edges present:
    factorized microcanonical
```

Within occupation MCMC:

```text
s+k
s+E
s
```

in that order.

Add a regression test proving `s+k` cannot silently route as only fixed-`k,T`.

# 55. Rust route

Wire fixed-`(s,k)` to the new core sampler.

No new generic constraint framework.

# 56. pyo3

Add thin binding following fixed-`(s,E)` style.

Do not duplicate math/validation in Python.

# 57. Python routing

Wire:

```text
Constraint.STRENGTH_DEGREE
```

No fit step.

# 58. Capability

Enable only after Python E2E passes and the routing guard lands atomically.

---

# PART I — CODE CLEANUP AFTER SUCCESS

# 59. Cleanup is last

Do not delete failed implementations while debugging the new constructor.

After all gates pass, use git history as the archive and remove confusing dead
production paths.

# 60. Old support-first code

Search:

```bash
rg "allocate_residual_flow|allocate_residual_greedy|Dinic|best_flow"
```

If these are used only by the obsolete support-first initializer/tests:

```text
remove old greedy allocator
remove old Dinic
remove support-first retry logic
remove tests specific only to the abandoned policy
```

Keep generic binary support initializer code.

# 61. Old degree-repair initialization

Search:

```bash
rg "repair_to_degree_target|DegreeRepairConfig|DegreeRepairOutcome|DegreeRepairExhausted"
```

If no longer required except historical tests:

```text
remove initialization degree-repair code
remove obsolete error variants/tests
```

Do **not** remove stationary auxiliary/trace code.

# 62. Rewrite failure-pinning scalability tests

Current N=1000 direct-init tests intentionally expect heterogeneous failure.

After success:

```text
rewrite them to require extras-first success
remove "unexpectedly succeeded" / "documented blocker" live comments
```

Historical failure stays in decision records, not live tests.

---

# PART J — DOCUMENTATION CLEANUP

# 63. One current summary only

The current fixed-`(s,k)` agent-spec directory has no separate authoritative
`summary.md`.

Treat:

```text
docs/development/agent-specifications/STATUS.md
```

as the authoritative **summary / entry point**.

If the implementation process creates temporary:

```text
summary.md
SUMMARY.md
handoff.md
notes.md
results.md
```

merge unique current facts into `STATUS.md` or the final decision record, then
delete/archive the temporary file.

Do not leave competing summaries.

# 64. Rewrite `STATUS.md`

Current title still reflects the old degree-repair STOP.

After success use one of:

```text
# STATUS — Microcanonical fixed-(s,k): implemented and N=1000 validated
```

or, if public integration remains:

```text
# STATUS — Microcanonical fixed-(s,k): core complete, public integration pending
```

Final major sections:

```text
## Current verdict
## Final architecture
## Exactness evidence
## N=1000 initialization evidence
## N=1000 trace/mobility evidence
## Public integration status
## How to verify
## Historical decisions
```

Keep it concise. Link history instead of reproducing pages of STOP narrative.

# 65. Final architecture in STATUS

State:

```text
residualize s,k,fixed pairs
-> extras-first slot-aware compressed transport
-> binary completion of missing support degree slots with occupation-1 edges
-> exact D=0 state
-> exact first-return trace
-> merge fixed pairs
-> full validation
```

Mention:

```text
initialization is combinatorial, no detailed balance required
trace exactness is independently checked by Q/R oracle
```

# 66. Preserve decision records

Do not delete:

```text
docs/decisions/microcanonical-fixed-sk-stop.md
docs/decisions/microcanonical-fixed-sk-trace-mobility.md
docs/decisions/microcanonical-fixed-sk-direct-init.md
```

They document real evidence.

# 67. Add supersession banners

At top of `microcanonical-fixed-sk-stop.md`:

```text
> Superseded for current implementation status.
> This record documents the abandoned degree-repair initializer.
> See STATUS.md and microcanonical-fixed-sk-extras-first-init.md.
```

At top of `microcanonical-fixed-sk-direct-init.md`:

```text
> Superseded for current implementation status.
> This record documents the failed support-first direct initializer.
> The replacement is the extras-first constructor.
```

Do not rewrite old evidence tables.

Gate A remains valid; optionally add a note that final constructed-start
mobility is recorded in the extras-first/final decision.

# 68. Final current decision record

Expand/finalize:

```text
docs/decisions/microcanonical-fixed-sk-extras-first-init.md
```

Sections:

```text
## Verdict
## Constructor architecture
## Why support-first failed
## Why extras-first works
## N=1000 evidence
## Constructed-start trace mobility
## Exactness evidence
## Complexity
## Known limitations
## Final routing status
```

# 69. Archive generated instruction documents

Once implementation is complete, create:

```text
docs/development/agent-specifications/archive/fixed_sk/
```

Move with `git mv`:

```text
MENoBiS_fixed_sk_implementation_plan_v2.md
MENoBiS_fixed_sk_recovery_direct_init_trace_gate.md
MENoBiS_fixed_sk_extras_first_completion_plan.md
```

from the live agent-specifications root into `archive/fixed_sk/`.

Do this only after the work described by the plans is complete.

# 70. Fixed-sk archive README

Create:

```text
docs/development/agent-specifications/archive/fixed_sk/README.md
```

Chronology:

```text
1. implementation_plan_v2
   exact trace design
   failed degree-repair initializer

2. recovery_direct_init_trace_gate
   Gate A validated trace
   support-first direct init failed

3. extras_first_completion_plan
   replacement constructor and final integration
```

State clearly:

```text
Historical implementation instructions only.
Read ../../STATUS.md and current code for live requirements.
```

# 71. Update agent-specifications README

Update:

```text
docs/development/agent-specifications/README.md
```

Remove live wording such as:

```text
Current fixed-(s,k) work (STOPPED / recovery)
```

Final current section should primarily link:

```text
STATUS.md
archive/fixed_sk/README.md
```

Do not list archived plans as live requirements.

# 72. Update archive README

Update:

```text
docs/development/agent-specifications/archive/README.md
```

Add `fixed_sk/README.md` to the archive index.

# 73. Fix links after moves

Run:

```bash
rg "MENoBiS_fixed_sk_implementation_plan_v2"
rg "MENoBiS_fixed_sk_recovery_direct_init_trace_gate"
rg "MENoBiS_fixed_sk_extras_first_completion_plan"
```

Update live Markdown links to archived paths.

Historical prose can remain historical, but links must resolve.

# 74. Remove temporary generated docs

Inspect:

```bash
git status
find docs -maxdepth 4 -type f | sort
```

Do not leave scratch instructions or temporary summaries in live doc roots.

# 75. Search stale STOP wording

Run:

```bash
rg "STOPPED|STOP artifact|DegreeRepairExhausted|co-joint blocker|not shipped" \
  docs crates/menobis-test-oracles crates/menobis-core
```

Manual rule:

```text
historical decision record:
    preserve evidence + supersession banner

live STATUS / README / code comment / current test:
    update to final status
```

---

# PART K — FINAL VERIFICATION

# 76. Rust

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

All green.

# 77. Exact oracles

```bash
cargo test -p menobis-test-oracles --test fixed_strength_degree_enumeration
cargo test -p menobis-test-oracles --test fixed_strength_edges_enumeration
```

All green.

# 78. Heavy N=1000 gates

Run final release-mode ignored tests with `--nocapture`.

Must include:

```text
realistic ME direct init
Balanced12 direct init
heterogeneous W
heterogeneous B
structural variants
constructed-state trace mobility
full E2E
```

Record actual commands and timings.

# 79. Python after public integration

Use repository-standard commands, at minimum:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

# 80. Routing release blocker

Add a test proving:

```text
microcanonical strengths + degrees
cannot silently route to fixed-(k,T)
```

Capability cannot be enabled without this test.

---

# 81. Mandatory STOP conditions

## STOP A — realistic extras construction fails

If realistic ME N=1000 still fails after:

```text
64 extras attempts
```

stop. Do not invent augmenting-cycle repair.

## STOP B — filler completion systematically fails

If extras satisfy support-degree upper bounds but `delta k` cannot be completed
after configured retries, report:

```text
extras support degrees
delta degree vectors
domain exclusions
attempt counts
```

and stop.

## STOP C — constructed-state sampler mobility collapses

If realistic ME/W constructed states have effectively zero support-changing
returns, stop before public integration.

## STOP D — exact oracle failure

Any Q/R detailed-balance or stationarity regression is immediate STOP.

## STOP E — routing still ignores strengths

Do not expose the capability.

---

# 82. Things the agent must NOT do

- Do not return to degree-repair initialization.
- Do not increase old repair budgets.
- Do not generate arbitrary exact-k supports then retry residual flow.
- Do not use witness support as production input.
- Do not require a witness table from users.
- Do not loosen exact s or exact k.
- Do not use approximate degrees.
- Do not allocate N×N structures.
- Do not introduce MILP or a generic fixed-charge flow framework.
- Do not write a new stationary sampler.
- Do not change `K_E` mathematics.
- Do not weaken the Q/R exact oracle.
- Do not enable Python capability before N=1000 gates.
- Do not delete historical decisions.
- Do not leave generated implementation plans in the live docs root after
  completion.
- Do not leave multiple current summaries.
- Do not claim mathematically proven rapid mixing.

---

# 83. Expected final architecture

```text
full target (s,k) + fixed pairs
        |
        v
existing residualization
        |
        +--> residual s
        +--> residual k
        +--> residual PairDomain
        |
        v
r = s_out - k_out
c = s_in  - k_in
        |
        v
slot-aware compressed extras transport
(high mass-per-slot rows paired with
 high mass-per-slot columns)
        |
        | exact extras margins
        | support degree <= k
        v
extras support B
        |
        v
delta_k = k - degree(B)
        |
        v
existing binary exact-degree constructor
on domain minus B
        |
        v
occupation-1 filler support C
        |
        v
t = 1+y on B
t = 1   on C
        |
        v
exact residual (s,k) state
D=0
        |
        v
existing exact degree-biased
first-return trace
        |
        v
burn-in / thinning
        |
        v
merge fixed pairs
        |
        v
full exact validation
        |
        v
Rust route / pyo3 / Python
```

---

# 84. Definition of done

## Constructor

- [ ] extras are constructed before exact support completion;
- [ ] extras row sums equal `s_out-k_out`;
- [ ] extras column sums equal `s_in-k_in`;
- [ ] extras support degrees never exceed k;
- [ ] B extras never exceed `M-1`;
- [ ] missing degree vectors are exact;
- [ ] fillers are disjoint from extras;
- [ ] filler degrees exactly equal missing degrees;
- [ ] final state has exact s and exact k;
- [ ] fixed pairs work through residualization;
- [ ] no N×N memory materialization.

## N=1000

- [ ] realistic ME succeeds;
- [ ] Balanced12 succeeds;
- [ ] heterogeneous W succeeds;
- [ ] heterogeneous B succeeds;
- [ ] uniform stress grid remains green;
- [ ] structural variants succeed;
- [ ] constructor diagnostics recorded.

## Sampler

- [ ] constructed state starts at D=0;
- [ ] exact Q/R oracle remains green;
- [ ] realistic constructed states have useful support mobility;
- [ ] end-to-end sampler preserves exact constraints.

## Public integration

- [ ] `s+k` routes to occupation MCMC;
- [ ] Rust route complete;
- [ ] pyo3 complete;
- [ ] Python routing complete;
- [ ] capability enabled last.

## Cleanup

- [ ] obsolete support-first production code removed or isolated;
- [ ] obsolete degree-repair initialization removed if unused;
- [ ] failure-pinning live tests rewritten as success gates;
- [ ] `STATUS.md` rewritten as the single current summary;
- [ ] old decisions have supersession banners;
- [ ] final extras-first decision exists;
- [ ] all generated fixed-sk plans archived under `archive/fixed_sk/`;
- [ ] archive index exists;
- [ ] agent-specifications README updated;
- [ ] no competing `summary.md`/handoff scratch remains;
- [ ] stale live STOP wording removed;
- [ ] final verification commands recorded.

---

# 85. Final instruction to the agent

Keep these three concepts separate:

```text
INITIALIZATION
    combinatorial
    no detailed balance needed
    this task changes it

STATIONARY SAMPLER
    exact first-return trace
    Gate A already validated it
    preserve it

HISTORICAL FAILED METHODS
    degree-repair initialization
    support-first exact-k initialization
    preserve their decision records
    remove them from live architecture
```

If you are unsure which one you are modifying, stop and re-read `STATUS.md` and
this plan before coding.
