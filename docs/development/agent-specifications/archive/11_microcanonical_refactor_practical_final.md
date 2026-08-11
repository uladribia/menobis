# MENoBiS Microcanonical Refactor — Practical Sparse Architecture

**Status:** implementation and migration specification  
**Target:** current MENoBiS `master`, August 2026  
**Scope:** currently implemented microcanonical cases  
**Deferred:** fixed \((\mathbf s,E)\) and fixed \((\mathbf s,\mathbf k)\)  
**Primary goals:** simpler production code, sparse scalability, maximum useful reuse, preservation of heavy exact methods as Rust test oracles.

---

# 1. Goal

MENoBiS already has working microcanonical implementations, but parts of the current architecture are becoming too heavy for production use.

The main issues are:

- some initialization paths operate over \(O(N^2)\) candidate pairs;
- fixed-strength loopless initialization is more complicated than necessary;
- fixed-strength MCMC wastes many proposals on sparse occupation states;
- fixed-strength + expected-cost inherits those mobility problems;
- some exact fixed-total allocation algorithms scale poorly;
- correct but expensive algorithms are mixed into production even though their best long-term role is validation.

The refactor should make production code follow the simplest algorithm appropriate to each constraint.

The goal is **not** to force every ensemble through one universal algorithm.

The goal is:

\[
\boxed{
\text{simple sparse production algorithms}
+
\text{heavy exact Rust validation oracles}
}
\]

Production should remain viable at

\[
N\sim10^3-10^4
\]

when the actual sampled network is sparse enough to fit in memory.

---

# 2. Scope

The currently supported microcanonical cases in scope are:

1. fixed \((E,T)\);
2. fixed \((\mathbf k,T)\);
3. fixed strengths \(\mathbf s\);
4. fixed strengths + expected cost.

The following remain deferred:

- fixed \((\mathbf s,E)\);
- fixed \((\mathbf s,\mathbf k)\).

Do not implement infrastructure specifically for the deferred cases during this refactor.

The new architecture should merely avoid blocking them later.

---

# 3. Practical architectural rule

Use the natural decomposition of each constraint.

## Fixed \((E,T)\)

```text
sample binary support
    ↓
sample positive occupations with total T
```

## Fixed \((\mathbf k,T)\)

```text
sample fixed-degree support
    ↓
use the same positive fixed-total occupation sampler
```

## Fixed strengths

```text
construct any exact-strength sparse table
    ↓
repair only the remaining structural violations
    ↓
run the correct fixed-strength MCMC
```

## Fixed strengths + expected cost

```text
same fixed-strength construction
    ↓
same structural repair
    ↓
fit scalar gamma
    ↓
same fixed-strength MCMC with cost tilt
```

This is the final mental model.

Do not create a generic `RepairFramework` merely because some cases use repair.

---

# 4. Production versus Rust oracle code

Separate MENoBiS into two algorithmic roles.

## Production

Production implementations must be:

- sparse;
- simple;
- scalable;
- understandable;
- free of unnecessary exact global optimization.

## Heavy Rust test oracles

The existing heavy Rust suite should retain correct algorithms that are useful for validation but unsuitable for production scaling.

Examples:

- exact enumeration;
- current exact rejection allocation;
- dynamic programming;
- max-flow feasibility/construction;
- explicit stub matching;
- legacy fixed-strength MCMC kernels;
- other exact or expensive reference implementations.

Heavy code is allowed to be slow.

Its purpose is to answer:

> Is the new scalable implementation correct?

It is not part of the public generation path.

---

# 5. Migration rule for heavy algorithms

For every production algorithm to be replaced:

```text
CURRENT PRODUCTION
       |
       +-------------------+
       |                   |
       v                   v
RUST HEAVY ORACLE     NEW PRODUCTION
       |                   |
       +---------+---------+
                 |
                 v
              COMPARE
                 |
                 v
              VALIDATE
                 |
                 v
      REMOVE OLD PRODUCTION PATH
```

Do not delete scientifically useful code before the replacement is validated.

Do not keep duplicate production routing after validation.

Once migrated:

- Python must expose only the new production path;
- normal Rust generation must expose only the new production path;
- heavy comparison methods remain test-only.

Use the repository's existing heavy-test conventions rather than inventing a second validation framework.

---

# 6. Sparse pair domains

Complete pair domains must remain implicit.

For a complete directed loopless network:

\[
(i,j)\text{ admissible}
\iff
i\ne j.
\]

This check is \(O(1)\).

Do not construct all

\[
N(N-1)
\]

possible pairs merely to represent the missing diagonal.

Sparse masks may use sparse lookup structures.

Ordinary complete or loopless production paths must never require \(O(N^2)\) domain storage.

---

# 7. Fixed \((E,T)\): keep the current architecture, replace the heavy occupation backend

Fixed \((E,T)\) already has the correct conceptual factorization:

```text
uniform E-edge support
    +
positive occupations summing to T
```

Keep the existing support sampler.

Do not introduce repair here.

The scalability refactor concerns only the occupation layer.

---

# 8. Shared scalable fixed-total occupation sampler

Fixed \((E,T)\) and fixed \((\mathbf k,T)\) should share one occupation sampler.

State:

\[
\mathbf t=(t_1,\ldots,t_E)
\]

with

\[
t_e\ge1,
\qquad
\sum_e t_e=T.
\]

For B:

\[
t_e\le M.
\]

Use the planned two-cell Gibbs update.

Choose two distinct support indices \(a,b\).

Let

\[
q=t_a+t_b.
\]

Replace:

\[
(t_a,t_b)
\]

with

\[
(k,q-k)
\]

drawn from the exact two-cell conditional family law.

This gives:

- \(O(E)\) state;
- no large DP table;
- no full-composition rejection;
- one shared implementation for ME/W/B;
- one shared occupation layer for fixed \((E,T)\) and fixed \((\mathbf k,T)\).

The existing exact fixed-total implementations become Rust test oracles after validation.

---

# 9. Fixed \((\mathbf k,T)\): preserve the working support sampler

Do not redesign the fixed-degree backbone sampler if it already scales acceptably.

Its architecture remains:

```text
fixed-degree support sampler
    ↓
shared fixed-total occupation sampler
```

The only migration required is replacing the old occupation backend with the shared Gibbs implementation.

No repair phase is needed.

---

# 10. Fixed-strength production state

Use one sparse fixed-strength state.

Conceptually:

```rust
struct StrengthState {
    occupations: HashMap<PairId, OccNum>,
    occupied_pairs: Vec<PairId>,
    occupied_positions: HashMap<PairId, usize>,
}
```

Target strength vectors belong to the problem definition.

Strength-preserving moves do not need to recompute full row and column sums after every proposal.

The state must support:

- \(O(1)\) expected occupation lookup;
- uniform random occupied-pair selection;
- \(O(1)\) expected insertion;
- \(O(1)\) expected removal by swap-remove.

Memory should scale as:

\[
O(N+E_{\mathrm{occ}}).
\]

---

# 11. Fixed-strength constructor: compressed aggregated matching

Do **not** use explicit event stubs in the scalable production initializer.

Explicit stub matching costs:

\[
O(T)
\]

memory/work and can become wasteful when

\[
T\gg E_{\mathrm{occ}}.
\]

Instead construct an aggregated table from residual strengths.

Given:

\[
\mathbf s^{out},
\qquad
\mathbf s^{in},
\]

with:

\[
\sum_i s_i^{out}
=
\sum_j s_j^{in}
=
T,
\]

maintain:

```rust
remaining_out[i]
remaining_in[j]
```

and active source/target index collections.

Repeatedly:

1. choose an active source \(i\);
2. choose an active target \(j\);
3. choose a block
   \[
   x\le\min(r_i,c_j);
   \]
4. set or increment
   \[
   t_{ij}\leftarrow t_{ij}+x;
   \]
5. update the residuals;
6. remove exhausted entries.

The constructor should randomize source and target order.

It should not try to solve every structural restriction.

Its primary invariant is:

\[
\boxed{\text{exact strengths}}
\]

with sparse aggregated storage.

---

# 12. Keep the constructor simple

Do not optimize the constructor into another difficult mathematical problem.

Avoid:

- max flow;
- minimum-cost flow;
- dense pair search;
- rejection over all possible pairs;
- exact loopless construction;
- family-weight calculations.

The constructor is only a starting-state generator.

If its output contains self-loops or other currently supported structural violations, repair them afterwards.

---

# 13. Complete fixed-strength networks with self-loops allowed

For ME/W on a complete domain with self-loops allowed:

```text
compressed constructor
    ↓
production MCMC
```

No structural repair is needed.

For B, capacity violations may require repair if the constructor produces occupations above \(M\).

Do not complicate construction merely to prevent them.

---

# 14. Complete loopless fixed-strength ME/W: exact feasibility test

For complete directed ME/W without self-loops, define:

\[
T=
\sum_i s_i^{out}
=
\sum_i s_i^{in}.
\]

A loopless realization exists iff:

\[
\boxed{
s_i^{out}+s_i^{in}\le T
\qquad
\forall i.
}
\]

Check this in:

\[
O(N).
\]

If any node violates it, return a clear infeasibility error before construction/repair.

No max-flow feasibility test is needed for this case.

---

# 15. Complete loopless ME/W: guaranteed loop repair

For this case, do **not** use generic annealing.

Suppose:

\[
t_{ii}>0.
\]

Choose a positive donor cell:

\[
t_{cd}>0
\]

with:

\[
c\ne i,
\qquad
d\ne i.
\]

Then choose:

\[
\delta
=
\min(t_{ii},t_{cd}).
\]

Apply:

\[
t_{ii}'=t_{ii}-\delta,
\]

\[
t_{cd}'=t_{cd}-\delta,
\]

\[
t_{id}'=t_{id}+\delta,
\]

\[
t_{ci}'=t_{ci}+\delta.
\]

This preserves every source and target strength exactly.

The increment cells are non-diagonal because:

\[
d\ne i,
\qquad
c\ne i.
\]

Therefore no new loop is introduced.

---

# 16. Why the loop repair must progress

Define total loop occupation:

\[
L(\mathbf t)=\sum_i t_{ii}.
\]

Every repair step satisfies:

\[
L(\mathbf t')<L(\mathbf t).
\]

A valid donor must exist whenever:

- \(t_{ii}>0\);
- the loopless feasibility condition holds.

If no positive cell existed outside row \(i\) and column \(i\), then:

\[
T=s_i^{out}+s_i^{in}-t_{ii}.
\]

But feasibility requires:

\[
s_i^{out}+s_i^{in}\le T.
\]

Hence:

\[
s_i^{out}+s_i^{in}
\le
s_i^{out}+s_i^{in}-t_{ii},
\]

which is impossible for:

\[
t_{ii}>0.
\]

Therefore loop repair cannot get stuck in the complete unbounded ME/W case.

This is a deterministic constructive repair problem, not an annealing problem.

---

# 17. Efficient donor selection

Do not scan the entire occupation state for every loop.

The sparse state already maintains:

```rust
occupied_pairs
```

Select donors by randomized attempts from this vector.

Reject candidates where:

```text
source == i
or
target == i
```

For normal sparse networks this should be cheap.

If repeated randomized attempts fail, fall back to one bounded linear scan over occupied pairs.

Because a valid donor is guaranteed to exist in the feasible ME/W case, the bounded scan must find one.

This avoids \(O(N^2)\) pair traversal.

---

# 18. B repair

B adds the hard local bound:

\[
0\le t_{ij}\le M.
\]

The same rectangle repair remains the first mechanism.

However, increment cells must satisfy enough residual capacity:

\[
t_{id}+\delta\le M,
\]

\[
t_{ci}+\delta\le M.
\]

The simple ME/W termination proof therefore does not automatically apply.

Keep the production solution simple:

```text
compressed randomized construction
    ↓
capacity/loop-aware rectangle repair
    ↓
if stuck:
    discard initial state
    randomize constructor again
    retry
```

Use a bounded number of restarts.

Do not add a generic flow solver back into production.

Do not add general simulated annealing unless heavy-oracle validation demonstrates a real family of feasible B cases that simple repair + restart consistently misses.

---

# 19. Masks and arbitrary structural zeros

Arbitrary masks are harder than simply forbidding the diagonal.

A feasible state may require longer alternating cycles rather than a single rectangle.

Use the same conservative strategy:

1. attempt local rectangle repair;
2. use randomized reconstruction/restart;
3. stop after a bounded number of failures;
4. return `RepairDidNotConverge`.

During migration, compare against the heavy Rust feasibility oracle.

Do not introduce general alternating-cycle repair until benchmark/oracle data proves it is necessary.

This keeps production architecture small.

---

# 20. Repair should remain targeted, not generic

The first production repair module should contain only repairs required by currently supported cases.

Conceptually:

```rust
repair_self_loops(...)
repair_capacity(...)
repair_forbidden_pairs(...)
```

These may share the same rectangle update helper.

Do not create:

```text
ViolationTrait
RepairConstraintSet
RepairObjectiveGraph
GenericAnnealer
```

unless later constraints genuinely need them.

Targeted functions are easier to reason about and easier to delete or improve.

---

# 21. Repair failure semantics

Repair must be bounded.

For cases without a proof of termination, use:

```rust
RepairConfig {
    max_steps,
    max_restarts,
}
```

Keep configuration small.

On failure return:

```text
RepairDidNotConverge
```

with:

```text
remaining loops
remaining capacity violations
remaining forbidden occupation
restart count
steps
```

Do not silently invoke heavy Rust oracle code from production.

---

# 22. Fixed-strength production MCMC: replace sparse-hostile proposal

The current uniform-coordinate four-cycle kernel should be replaced.

Do not randomly search the entire pair domain for decrement cells.

Choose decrement cells from the occupied state.

Select two distinct occupied pairs:

\[
(a,b),
\qquad
(c,d).
\]

Require:

\[
a\ne c,
\qquad
b\ne d.
\]

Propose:

\[
t_{ab}'=t_{ab}-1,
\]

\[
t_{cd}'=t_{cd}-1,
\]

\[
t_{ad}'=t_{ad}+1,
\]

\[
t_{cb}'=t_{cb}+1.
\]

Validate:

- pair domain;
- self-loop policy;
- B capacity;
- family support.

This removes the sparse-state factor associated with randomly discovering two positive decrement cells.

---

# 23. Hastings correction

Occupied-cell proposal selection is state-dependent.

Use exact Metropolis-Hastings.

If:

\[
m(\mathbf t)
\]

is the number of occupied cells, compute the actual forward and reverse proposal probabilities from the implemented selection rule.

The acceptance log ratio is:

\[
\Delta\log A
=
\Delta\log D_F
-
\gamma\Delta C
+
\log q(\mathbf t'\to\mathbf t)
-
\log q(\mathbf t\to\mathbf t').
\]

Use the existing family degeneracy and cost-provider abstractions.

Do not duplicate ME/W/B formulas in the move module.

---

# 24. Remove per-proposal allocation

The normal four changed cells are distinct.

Represent them in a stack-local fixed-size object.

Do not allocate:

- temporary `HashMap`s;
- temporary heap `Vec`s;
- generic move graphs.

The proposal hot path should remain approximately:

\[
O(1).
\]

---

# 25. Fixed strengths + expected cost

This is not a separate sampler.

Reuse exactly:

- compressed construction;
- structural repair;
- `StrengthState`;
- occupied-cell proposal;
- family target;
- MCMC lifecycle.

Cost changes only the target:

\[
\Delta\log\pi
=
\Delta\log D_F
-
\gamma\Delta C.
\]

Structural repair must ignore cost.

Once the state is feasible, fit gamma and sample on the correct constrained manifold.

---

# 26. Simplify gamma fitting after MCMC repair

Delete the fragile variance-based warm start once the new kernel is validated.

Start at:

\[
\gamma=0.
\]

Estimate:

\[
\mu_C(0).
\]

If:

\[
\mu_C(0)>C_{\mathrm{obs}},
\]

search positive gamma.

If:

\[
\mu_C(0)<C_{\mathrm{obs}},
\]

search negative gamma.

Expand geometrically:

```text
0
±g
±2g
±4g
±8g
...
```

until the observed cost is bracketed.

Then use stochastic bisection.

Do not add more clamps around unstable warm-start formulas.

Remove those formulas instead.

---

# 27. Gamma evaluations must require real movement

Record:

```text
proposal count
structurally valid count
accepted transitions
cost samples
effective sample information
```

Do not treat a large number of rejected proposals as equilibration.

If a gamma evaluation cannot obtain enough accepted transitions or effective samples, return:

```text
InsufficientMobility
```

and fail the fit.

A nonconverged gamma fit is not a successful production result by default.

---

# 28. Heavy Rust oracle migration: fixed strength

Before removing any current fixed-strength production machinery, move or expose it to the heavy Rust suite.

Keep test-only access to:

- current max-flow constructor;
- current uniform-coordinate cycle kernel;
- explicit stub matcher where useful;
- exact enumeration for tiny systems.

Use them as separate oracles.

## Feasibility oracle

For small/mid systems:

```text
heavy solver says feasible?
```

Compare against:

```text
new constructor + repair succeeds?
```

## Distribution oracle

Compare:

```text
legacy MCMC
vs
new occupied-cell MH
vs
exact enumeration where possible
```

Do not require the heavy constructor to remain callable from production.

---

# 29. Heavy Rust oracle migration: fixed total

Retain current exact fixed-total allocation methods in the heavy suite.

Validate new Gibbs production sampling against:

- exact enumeration;
- legacy rejection;
- legacy DP;
- deterministic special cases.

After validation, remove legacy backend routing from production.

The heavy implementation may remain expensive because test sizes are bounded.

---

# 30. Heavy Rust oracle organization

Follow the repository's existing heavy Rust test structure.

Conceptually:

```text
tests/
    heavy/
        microcanonical/
            fixed_et.rs
            fixed_kt.rs
            fixed_strength.rs
            strength_cost.rs

        oracle/
            legacy_fixed_total.rs
            legacy_strength_flow.rs
            legacy_strength_kernel.rs
            enumeration.rs
```

Use the actual existing test structure if names differ.

Avoid creating a second test framework.

Legacy source that exists only for heavy validation should be test-only or feature-gated.

Ordinary users should not compile or route through it where practical.

---

# 31. Oracle validation bounds

Heavy algorithms must have explicit test limits.

Examples:

```text
enumeration:
    tiny state spaces

legacy DP/rejection:
    small and medium E,T

max flow:
    small/mid N

legacy MCMC comparisons:
    sizes where the old chain actually mixes
```

Do not use a slow oracle outside the range where it is scientifically useful.

---

# 32. Validation for complete loopless ME/W repair

This repair has a mathematical guarantee, so test it aggressively.

Generate random feasible strength sequences satisfying:

\[
s_i^{out}+s_i^{in}\le T.
\]

For each:

1. construct an exact-strength table allowing loops;
2. record initial loop mass;
3. run loop repair;
4. verify:
   \[
   t_{ii}=0
   \]
   for every \(i\);
5. verify exact strengths;
6. verify loop mass decreases monotonically;
7. verify no \(O(N^2)\) domain allocation.

Also test adversarial cases close to:

\[
s_i^{out}+s_i^{in}=T.
\]

---

# 33. Validation for B and masks

For B and arbitrary masks, repair is initially heuristic.

Use the heavy oracle to measure failure rather than assuming success.

For thousands of bounded test instances:

```text
oracle infeasible:
    production may fail immediately or fail repair

oracle feasible:
    production repair should succeed at very high rate
```

Record failures with deterministic seeds.

If a reproducible feasible family systematically defeats simple repair + restart, then design the smallest targeted improvement.

Do not preemptively add complex general repair machinery.

---

# 34. Performance targets

Production must avoid:

\[
O(N^2)
\]

memory for ordinary complete or loopless domains.

Fixed-strength constructor must avoid:

\[
O(T)
\]

explicit stub expansion.

Expected storage:

\[
O(N+E_{\mathrm{occ}}).
\]

Normal MCMC proposals:

\[
O(1)
\]

expected local work.

Repair should operate on occupied pairs and violation lists rather than scanning all possible pairs.

---

# 35. Benchmark matrix

At minimum:

\[
N=
100,\;
500,\;
1000.
\]

For sparse stress testing add, where practical:

\[
N=5000
\]

and:

\[
N=10000.
\]

Test:

- self-loops allowed;
- complete loopless;
- ME/W/B;
- sparse and more occupied regimes;
- fixed strength;
- fixed strength + cost.

Report separately:

```text
construction time
repair time
repair steps
repair restarts
occupied pairs
MCMC structurally valid proposals/sec
accepted transitions/sec
gamma fitting time
cost ESS
final sampling time
peak memory
```

Do not hide construction/repair cost inside one total runtime.

---

# 36. Migration sequence from current `master`

## Phase A — Freeze oracle baselines

Before altering behavior:

- identify current exact/heavy implementations;
- make them callable from heavy Rust tests;
- record deterministic baseline cases;
- ensure current exact fixed-total and max-flow behavior is covered.

No production deletion yet.

---

## Phase B — Replace fixed-strength MCMC kernel

Implement occupied-cell proposal + exact Hastings correction.

Remove per-proposal heap allocation.

Temporarily retain old kernel only in heavy tests.

Validate:

- exact enumeration;
- old/new comparison;
- sparse and dense benchmarks.

Once validated, remove old kernel from production.

---

## Phase C — Add compressed fixed-strength constructor

Implement aggregated randomized matching.

It must:

- satisfy strengths exactly;
- avoid stubs;
- avoid \(N^2\) pair enumeration;
- produce sparse output.

Initially keep old constructor available only for comparison.

---

## Phase D — Add targeted loop repair

Implement guaranteed complete-loopless ME/W repair.

Route complete loopless ME/W through:

```text
compressed construction
    ↓
loop repair
    ↓
new production MCMC
```

Validate against heavy max flow on small/mid systems.

Then remove max-flow routing from this production path.

---

## Phase E — Add simple B/mask repair

Implement capacity-aware and forbidden-pair rectangle repair with bounded randomized restarts.

Do not add annealing initially.

Validate against heavy feasibility tests.

Only add extra machinery if measured failures justify it.

---

## Phase F — Simplify strength-cost fitting

After the new fixed-strength chain is mobile:

- remove variance warm start;
- remove associated gamma clamps;
- implement zero-centered bracket expansion;
- retain stochastic bisection;
- require accepted transitions / ESS.

Run ME/W/B cost benchmarks again.

---

## Phase G — Finish fixed-total Gibbs migration

Replace the heavy production occupation backend for:

- fixed \((E,T)\);
- fixed \((\mathbf k,T)\).

Validate against old exact methods.

Move old methods fully into heavy Rust oracle scope.

Remove temporary backend selectors.

---

## Phase H — Cleanup

Remove production code no longer required:

- max-flow production routing;
- complete-domain pair materialization;
- explicit stub production initializer;
- old uniform-coordinate strength kernel;
- exact DP/rejection production routing;
- obsolete gamma warm-start code;
- migration flags;
- duplicated errors/configuration.

Run:

- normal Rust tests;
- Python tests;
- benchmark CLI;
- heavy Rust oracle suite.

---

# 37. What should remain shared

Reuse only stable concepts.

## Shared family mathematics

```text
OccupationFamily
```

## Shared cost mathematics

```text
PairCostProvider
```

## Shared MCMC control

```text
McmcConfig
McmcCounters
RNG conventions
```

## Shared fixed-total occupation layer

Used by:

- fixed \((E,T)\);
- fixed \((\mathbf k,T)\).

## Shared fixed-strength state and rectangle transaction

Used by:

- fixed strengths;
- fixed strengths + expected cost;
- structural repair.

Do not force initializers or production kernels for mathematically different constraints into one trait hierarchy.

---

# 38. What should remain separate

Keep separate:

- fixed-E support sampling;
- fixed-degree support sampling;
- fixed-strength construction;
- fixed-total Gibbs;
- fixed-strength MCMC;
- gamma fitting.

These solve different mathematical problems.

Reuse should remove duplication, not hide those differences.

---

# 39. Deferred future work

Do not implement now:

- fixed \((\mathbf s,E)\);
- fixed \((\mathbf s,\mathbf k)\);
- generic annealed repair;
- general alternating-cycle mask repair;
- grand-canonical warm starts;
- universal MCMC kernel framework.

The current refactor should leave clean extension points, not speculative code.

---

# 40. Completion gate

The refactor is complete when:

### Fixed total

- fixed \((E,T)\) and fixed \((\mathbf k,T)\) share the scalable Gibbs occupation layer;
- heavy DP/rejection methods are test oracles only.

### Fixed strength

- scalable constructor uses aggregated residual matching;
- no explicit \(O(T)\) stub expansion in production;
- complete loopless ME/W uses the guaranteed targeted repair;
- no \(O(N^2)\) pair-domain construction;
- occupied-cell MH is the production kernel;
- old kernel is oracle-only or removed.

### Strength + expected cost

- same fixed-strength engine is reused;
- gamma fitting is simplified;
- mobility is measured using actual transitions;
- nonconverged fits fail by default.

### Heavy validation

- old exact/heavy methods are retained where scientifically useful;
- production-versus-oracle comparisons are automated;
- oracle tests use bounded sizes and deterministic seeds.

### Code quality

- no permanent migration flags;
- no duplicate family or cost formulas;
- no unnecessary generic framework;
- production code is smaller and easier to explain than before the refactor.

---

# 41. Final architecture

```text
MICROCANONICAL

fixed (E,T)
    |
    +-- uniform support
    |
    +-- shared fixed-total Gibbs


fixed (k,T)
    |
    +-- fixed-degree support
    |
    +-- shared fixed-total Gibbs


fixed strengths
    |
    +-- compressed aggregated construction
    |
    +-- targeted repair if needed
    |
    +-- occupied-cell MH


fixed strengths + expected cost
    |
    +-- same construction
    |
    +-- same repair
    |
    +-- scalar gamma fit
    |
    +-- same occupied-cell MH + cost tilt
```

Alongside:

```text
HEAVY RUST ORACLES

exact enumeration
legacy DP
legacy rejection
max flow
stub matching
legacy MCMC
```

The final design principle is:

\[
\boxed{
\text{use proofs and simple targeted algorithms where available;}
}
\]

\[
\boxed{
\text{use heavy exact algorithms to validate production, not to burden it.}
}
\]

For the currently painful loopless fixed-strength ME/W case in particular, the production path should become:

\[
\boxed{
\text{aggregated exact-strength construction}
\rightarrow
\text{guaranteed loop repair}
\rightarrow
\text{occupied-cell MH}
}
\]

with no max flow, no explicit stubs, no dense domain, and no generic annealing machinery.
