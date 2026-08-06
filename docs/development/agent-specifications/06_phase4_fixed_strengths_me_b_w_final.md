# Phase 4 Specification — Fixed Strengths for ME, B, and W

**Repository:** `uladribia/menobis`  
**Planning baseline:** `master` at commit `590f7002d4bf84989289e7e86bb99bcd09d6dd62`  
**Version:** August 2026

---

# 1. Purpose

This document specifies the next MENoBiS microcanonical phase:

\[
\boxed{\text{exact directed strength sequences for ME, B, and W}}
\]

The implementation order is:

1. ME fixed strengths;
2. B fixed strengths;
3. W fixed strengths;
4. prepare, but do not yet expose, the pair-potential abstraction required by the next phase:

\[
\boxed{\text{fixed strengths + expected cost}}.
\]

The implementation must preserve the repository philosophy:

- family laws are shared;
- constraints are separate from moves;
- preprocessing and masks are shared;
- direct exact algorithms are preferred when available;
- MCMC is used only where direct sampling is not practical;
- sparse memory is mandatory;
- no implementation may depend on conversation history.

---

# 2. Current repository baseline

The current microcanonical tree contains:

```text
generation/microcanonical/
    fixed_et/
    fixed_kt/
    support/
    mod.rs
```

The repository already provides:

- fixed \((E,T)\) for ME, B, and W;
- fixed \((\mathbf k,T)\) for ME, B, and W;
- shared fixed-pair preprocessing;
- sparse support state and switch-and-hold MCMC;
- family-independent support generation;
- family-specific positive-occupation allocation;
- exact enumeration tests;
- conditioned grand-canonical validation;
- capability routing;
- benchmark integration.

The current ME fixed-strength implementation is still a standalone function in `microcanonical/mod.rs`:

```rust
sample_strength_stub_matching(
    strength_out,
    strength_in,
    seed,
)
```

It:

1. materializes all outgoing stubs;
2. materializes all incoming stubs;
3. shuffles incoming stubs;
4. aggregates the resulting directed pairs.

This method is exact for:

- ME;
- directed networks;
- complete pair domain;
- self-loops allowed;
- no fixed-pair restrictions beyond trivial preprocessing.

It is not yet a complete Phase 4 implementation.

---

# 3. Scientific target

For a directed occupation matrix

\[
\mathbf t=\{t_{ij}\},
\]

the hard constraints are

\[
\sum_j t_{ij}=s_i^{\mathrm{out}}
\qquad \forall i,
\]

and

\[
\sum_i t_{ij}=s_j^{\mathrm{in}}
\qquad \forall j.
\]

The total occupation is implied:

\[
T
=
\sum_i s_i^{\mathrm{out}}
=
\sum_j s_j^{\mathrm{in}}.
\]

The target distribution for family \(F\) is

\[
P_F(\mathbf t\mid\mathbf s)
=
\frac{1}{Z_F(\mathbf s)}
\prod_{ij} d_F(t_{ij}),
\]

over all states satisfying:

- exact out-strengths;
- exact in-strengths;
- admissible-pair restrictions;
- self-loop policy;
- family occupation bounds;
- fixed occupations.

The local degeneracies are:

## ME

\[
d_{\mathrm{ME}}(t)=\frac{1}{t!}.
\]

Since \(T!\) is constant under fixed strengths, this is equivalent to the usual global ME degeneracy

\[
\frac{T!}{\prod_{ij}t_{ij}!}.
\]

## B

\[
d_{\mathrm B}(t)=\binom{M}{t},
\qquad
0\le t\le M.
\]

## W

\[
d_{\mathrm W}(t)=\binom{M+t-1}{t},
\qquad
t\ge0.
\]

The repository already exposes these laws through:

```rust
OccupationFamily::log_local_degeneracy(t)
OccupationFamily::delta_log_local_degeneracy(old, new)
OccupationFamily::validate_occnum(t)
OccupationFamily::occupation_support()
```

Phase 4 must reuse these methods rather than introducing another independent family formula layer.

---

# 4. Why fixed strengths require a joint occupation state

Fixed \((E,T)\) and fixed \((\mathbf k,T)\) separate into:

```text
support
    +
positive occupation allocation
```

Fixed strengths do not generally separate this way.

The support and occupations are coupled because the hard constraints are:

\[
\sum_j t_{ij}=s_i^{\mathrm{out}},
\qquad
\sum_i t_{ij}=s_j^{\mathrm{in}}.
\]

Different supports may:

- be infeasible;
- admit different numbers of occupation matrices;
- carry different total family degeneracy.

Therefore the sampler state must be the complete sparse occupation matrix.

The generic Phase 4 backend should operate directly on \(\mathbf t\).

---

# 5. Overall implementation strategy

Use a hybrid backend.

## ME fast path

For the simple domain:

- complete directed pair set;
- self-loops allowed;
- no immutable fixed cells;
- no pair mask;
- no expected cost;

use the existing exact stub-matching construction.

This gives one exact independent sample.

## Generic family backend

For:

- B;
- W;
- ME with self-loops forbidden;
- ME with masks or fixed cells;
- future fixed-strength-plus-cost sampling;

use a sparse Metropolis-Hastings chain on integer contingency tables.

The chain uses strength-preserving alternating-cycle moves.

The first production kernel is a \(2\times2\) unit cycle.

The implementation architecture must allow longer alternating cycles later for masked domains and difficult connectivity cases.

---

# 6. Architecture for Phase 4 and Phase 5

The implementation must reuse two existing repository subsystems instead of creating parallel machinery:

1. the fixed-\((\mathbf k,T)\) MCMC control layer;
2. the grand-canonical pair-cost provider layer.

The new code should add only the pieces that are genuinely different:

- an integer occupation state rather than a binary support state;
- strength-preserving occupation-cycle proposals rather than binary edge switches;
- Metropolis acceptance using family degeneracy and optional cost.

The target is

\[
\pi(\mathbf t)
\propto
\prod_{ij}d_F(t_{ij})
\exp\left[-\gamma\sum_{ij}c_{ij}t_{ij}\right].
\]

For Phase 4, set

\[
\gamma=0.
\]

For Phase 5, use the fitted cost multiplier and the existing cost provider.

The dependency structure should be:

```text
existing MCMC control
    config
    persistent chain lifecycle
    sweep / burn-in / thinning
    diagnostics conventions
    switch-and-hold semantics
    RNG handling
            |
            v
new integer occupation state + cycle proposal
            |
            v
existing OccupationFamily degeneracy
            +
existing PairCostProvider
```

Phase 5 must not introduce a second cost abstraction unless a thin adapter is required by Rust ownership or generic typing.

---

# 7. Reuse-oriented repository layout

Do not copy `fixed_kt/sampler.rs`, `fixed_kt/diagnostics.rs`, or the cost providers.

First extract the parts of fixed-\((\mathbf k,T)\) that are genuinely generic:

```text
generation/microcanonical/
    mcmc/
        mod.rs
        config.rs
        diagnostics.rs
        chain.rs
        outcome.rs

    fixed_kt/
        state.rs
        switch.rs
        initializer.rs
        feasibility.rs
        core.rs

    fixed_strength/
        mod.rs
        core.rs
        residual.rs
        feasibility.rs
        initializer.rs
        state.rs
        target.rs
        move_cycle.rs
        chain.rs
        me_direct.rs
```

The shared `mcmc/` layer should contain only state-agnostic control logic:

- burn-in and thinning configuration;
- proposal counters;
- accepted/held/rejected counters;
- persistent-chain lifecycle;
- sweep execution;
- deterministic RNG seeding;
- common exactness wording.

It must not know about edges, occupations, degrees, or strengths.

The fixed-\((\mathbf k,T)\) sampler should be migrated to this layer with no change in scientific behavior.

The fixed-strength chain then reuses the same control layer with a different state and proposal kernel.

Cost providers remain in the existing shared pair-provider module:

```text
crates/menobis-core/src/pairs.rs
    PairCostProvider
    EuclideanCostProvider
    future sparse cost providers
```

Phase 4 uses no cost provider at runtime.

Phase 5 injects an existing `PairCostProvider` into the fixed-strength target.

---

# 8. Core types

## 8.1 Reuse `OccupationFamily`

Do not create another family trait.

Use the existing methods:

```rust
OccupationFamily::validate_occnum
OccupationFamily::occupation_support
OccupationFamily::log_local_degeneracy
OccupationFamily::delta_log_local_degeneracy
```

These already encode ME, B, and W.

## 8.2 Reuse `PairCostProvider`

The repository already defines:

```rust
pub trait PairCostProvider: Sync {
    fn cost(&self, source: usize, target: usize) -> Option<f64>;
}
```

and:

```rust
EuclideanCostProvider
```

The fixed-strength target should use that trait directly.

A `None` cost must have the same semantic meaning as in the grand-canonical path:

> the pair is excluded from the cost-constrained domain.

This avoids different admissibility semantics between grand-canonical and fixed-strength-cost generation.

## 8.3 Target type

Use a small target object:

```rust
pub struct StrengthTarget<'a> {
    pub family: OccupationFamily,
    pub gamma: f64,
    pub costs: Option<&'a dyn PairCostProvider>,
}
```

For Phase 4:

```rust
StrengthTarget {
    family,
    gamma: 0.0,
    costs: None,
}
```

For Phase 5:

```rust
StrengthTarget {
    family,
    gamma,
    costs: Some(cost_provider),
}
```

The local log-ratio is:

```rust
fn delta_log_weight(
    &self,
    pair: PairId,
    old_occ: OccNum,
    new_occ: OccNum,
) -> Option<f64>
```

with:

\[
\Delta\log\pi_{ij}
=
\log d_F(t'_{ij})-\log d_F(t_{ij})
-\gamma c_{ij}(t'_{ij}-t_{ij}).
\]

Return `None` if the pair is excluded by the cost provider.

This should be the only scientific difference between Phase 4 and Phase 5.

## 8.4 Reuse the fixed-\((\mathbf k,T)\) MCMC lifecycle

Extract or generalize these existing concepts:

```rust
FixedDegreeMcmcConfig
FixedDegreeChain::step
FixedDegreeChain::sweep
FixedDegreeChain::burn_in
FixedDegreeChain::sample_support
SwitchOutcome
```

into generic equivalents such as:

```rust
McmcConfig
McmcDiagnostics
McmcOutcome
run_sweep
run_burn_in
```

Do not force a complicated trait hierarchy.

A small closure- or kernel-based driver is sufficient:

```rust
pub trait McmcKernel<S> {
    fn step(
        &mut self,
        state: &mut S,
        rng: &mut impl Rng,
    ) -> McmcOutcome;
}
```

Both fixed-degree and fixed-strength chains can use it.

The state and proposal remain constraint-specific.

---

# 9. Residual preprocessing

Fixed cells must be processed before sampling.

For every fixed occupation

\[
t_{ij}^{\mathrm{fix}},
\]

subtract it from the target margins:

\[
s_i^{\mathrm{out,res}}
=
s_i^{\mathrm{out,target}}
-
\sum_jt_{ij}^{\mathrm{fix}},
\]

\[
s_j^{\mathrm{in,res}}
=
s_j^{\mathrm{in,target}}
-
\sum_it_{ij}^{\mathrm{fix}}.
\]

A fixed cell is immutable.

It must be removed from the residual variable-cell domain.

The residual backend receives:

- residual out-strengths;
- residual in-strengths;
- family;
- layer count through `OccupationFamily` where relevant;
- admissible variable-pair domain;
- self-loop policy;
- seed and MCMC configuration.

The final result is:

```text
fixed occupations
    +
sampled residual occupations
```

Final validation must check the original target strengths.

## Required preprocessing errors

Reject:

- unequal source and target vector lengths;
- negative residual margins;
- unbalanced residual totals;
- fixed B occupation above \(M\);
- fixed occupation on a forbidden pair;
- fixed self-loop when loops are forbidden;
- duplicate fixed-pair declarations;
- checked arithmetic overflow.

---

# 10. Feasibility as a bounded transportation problem

Strength feasibility is independent of family degeneracy.

It depends on:

- row sums;
- column sums;
- allowed cells;
- cell capacities.

Represent the residual problem as a bipartite flow network.

## Nodes

```text
source
out-nodes i
in-nodes j
sink
```

## Capacities

\[
\text{source}\to i:
s_i^{\mathrm{out,res}},
\]

\[
j\to\text{sink}:
s_j^{\mathrm{in,res}}.
\]

For each admissible variable cell \((i,j)\):

### ME and W

\[
i\to j:
U_{ij}
=
\min(s_i^{\mathrm{out,res}},s_j^{\mathrm{in,res}}).
\]

### B

\[
i\to j:
U_{ij}
=
\min(M,s_i^{\mathrm{out,res}},s_j^{\mathrm{in,res}}).
\]

An integral max flow of value

\[
T_{\mathrm{res}}
\]

is a feasible residual occupation matrix.

Because capacities are integers, standard integral max-flow algorithms return integer occupations.

---

# 11. Feasibility backend hierarchy

Use the following order.

## 11.1 Trivial and deterministic cases

Handle immediately:

- \(T_{\mathrm{res}}=0\);
- one nonzero row;
- one nonzero column;
- one admissible cell;
- fully forced B capacities.

## 11.2 Complete-domain fast constructor

For complete directed domains with self-loops allowed, construct a transportation table greedily without building all \(N^2\) flow edges.

A northwest-corner-style algorithm is sufficient:

```text
i = first positive row
j = first positive column

while rows and columns remain:
    x = min(row_remaining[i], col_remaining[j], capacity(i,j))
    assign x to (i,j)
    reduce row and column
    advance exhausted indices
```

For ME/W the capacity is effectively unbounded.

For B, greedy construction may hit capacity dead ends; use it only with a feasibility-preserving policy or fall back to max flow.

## 11.3 Sparse or restricted-domain max flow

Use max flow when:

- self-loops are forbidden;
- a mask is present;
- fixed cells remove variable capacity;
- B capacities matter;
- greedy construction fails.

Memory should be:

\[
O(N+L),
\]

where \(L\) is the number of admissible variable cells represented explicitly.

## 11.4 Failure semantics

If max flow cannot send \(T_{\mathrm{res}}\), the residual problem is infeasible.

A failed heuristic constructor is not proof of infeasibility.

A failed exact max-flow feasibility test is.

---

# 12. Initial state versus equilibrium sample

The max-flow or greedy table is only an initial state.

It is generally biased.

The production chain must run after construction for B, W, and constrained ME cases.

The only Phase 4 direct independent sample is the ME stub-matching fast path in its explicitly supported domain.

---

# 13. Sparse occupation state

The state should avoid \(O(N^2)\) storage.

Recommended structure:

```rust
pub struct StrengthState {
    node_count: usize,

    // Only positive variable occupations.
    occupations: HashMap<PairId, OccNum>,

    // Optional dense O(N) margin caches for validation and diagnostics.
    out_strengths: Vec<OccNum>,
    in_strengths: Vec<OccNum>,

    // Positive variable pairs for fast iteration and diagnostics.
    occupied_pairs: Vec<PairId>,
    occupied_positions: HashMap<PairId, usize>,
}
```

Since the moves preserve strengths structurally, the margin arrays do not need to be updated on every move if kept only for validation.

A more minimal state can store only:

```text
occupation map
occupied-pair vector
occupied-pair positions
```

The target margins belong to the problem object.

## Required operations

- `get(pair) -> OccNum`, returning zero if absent;
- `set(pair, new_occ)`;
- insert positive pair in expected \(O(1)\);
- remove zeroed pair by swap-remove in expected \(O(1)\);
- apply a local delta atomically;
- compute current occupied-pair count;
- convert to `SampledNetwork`.

---

# 13.1 What is reused from fixed \((\mathbf k,T)\), and what is not

Reuse directly or through extraction:

- MCMC configuration;
- persistent-chain lifecycle;
- sweep/burn-in/thinning loops;
- RNG seeding;
- proposal/acceptance diagnostics;
- switch-and-hold policy;
- one-shot and persistent APIs;
- exact-stationary-MCMC capability wording;
- fixed-pair residualization conventions;
- mask lookup representation.

Do not reuse directly:

- `DegreeSupportState`, because it stores binary edges only;
- the directed double-edge switch, because fixed strengths require integer occupation transfers;
- the fixed-degree greedy initializer, because strength feasibility is a capacitated transportation problem;
- complement mode, because the complement of an integer occupation matrix does not preserve the target margins or family law.

The correct design is therefore:

```text
reuse MCMC engine
replace state
replace initializer
replace proposal
reuse diagnostics and API patterns
```

---

# 14. Admissible-pair abstraction

Do not pass masks as slices and call linear `.contains()`.

Use an interface such as:

```rust
pub trait PairDomain {
    fn node_count(&self) -> usize;
    fn is_admissible(&self, source: u64, target: u64) -> bool;
    fn capacity(
        &self,
        source: u64,
        target: u64,
        family: OccupationFamily,
    ) -> OccNum;
}
```

Implementations:

- complete domain;
- complete domain without diagonal;
- sparse hash-backed mask;
- fixed-cell-excluding domain;
- future coordinate/cost-aware domain wrapper.

The hot-path membership test must be expected \(O(1)\) or better.

---

# 15. The basic \(2\times2\) cycle move

Choose distinct source nodes

\[
a\ne c
\]

and distinct target nodes

\[
b\ne d.
\]

Choose one of the two signs uniformly.

For the positive orientation:

\[
t_{ab}'=t_{ab}+1,
\]

\[
t_{cd}'=t_{cd}+1,
\]

\[
t_{ad}'=t_{ad}-1,
\]

\[
t_{cb}'=t_{cb}-1.
\]

For the negative orientation, reverse all signs.

Every row and column receives one \(+1\) and one \(-1\).

Therefore:

\[
\mathbf s^{\mathrm{out}\prime}
=
\mathbf s^{\mathrm{out}},
\]

\[
\mathbf s^{\mathrm{in}\prime}
=
\mathbf s^{\mathrm{in}}.
\]

---

# 16. Proposal rule

Use a proposal that is symmetric independently of the current occupations:

1. choose two distinct source indices uniformly;
2. choose two distinct target indices uniformly;
3. choose sign \(+1\) or \(-1\) with probability \(1/2\);
4. construct the four-cell delta;
5. validate;
6. accept by Metropolis-Hastings or hold.

Do not choose decrement cells only from occupied pairs in the baseline kernel.

That would make the proposal probability state-dependent and require a nontrivial Hastings correction.

Uniform node selection makes:

\[
q(x\to y)=q(y\to x).
\]

The acceptance probability is then:

\[
\alpha(x\to y)
=
\min\left(1,\frac{\pi(y)}{\pi(x)}\right).
\]

---

# 17. Local validation

Before computing expensive log weights:

1. merge coincident pair deltas into a stack-local map;
2. reject if any new occupation is negative;
3. reject if any changed pair is outside the variable admissible domain;
4. reject if any B occupation exceeds \(M\);
5. reject arithmetic overflow;
6. reject exact no-op moves.

For the normal \(a\ne c\), \(b\ne d\) proposal, the four cells are distinct.

Still use a general local-delta representation so longer cycles can reuse it.

---

# 18. Metropolis acceptance

For a local move affecting cells \(e\),

\[
\Delta\log\pi
=
\sum_e
\left[
\log d_F(t_e')
-
\log d_F(t_e)
\right]
+
\sum_e
\Delta\log w_{\mathrm{potential},e}.
\]

For Phase 4:

\[
\Delta\log\pi
=
\sum_e
\left[
\log d_F(t_e')
-
\log d_F(t_e)
\right].
\]

Accept if:

\[
\log U < \min(0,\Delta\log\pi),
\qquad U\sim\operatorname{Uniform}(0,1).
\]

Use log space.

Do not compute products of large degeneracies.

The implementation should call:

```rust
target.delta_log_weight(pair, old, new)
```

for each distinct changed cell.

---

# 19. Family-specific local ratios

These formulas are useful for tests and optional fast paths.

## ME

For an increment:

\[
\frac{d_{\mathrm{ME}}(t+1)}
{d_{\mathrm{ME}}(t)}
=
\frac{1}{t+1}.
\]

For a decrement:

\[
\frac{d_{\mathrm{ME}}(t-1)}
{d_{\mathrm{ME}}(t)}
=
t.
\]

## B

For an increment:

\[
\frac{d_{\mathrm B}(t+1)}
{d_{\mathrm B}(t)}
=
\frac{M-t}{t+1}.
\]

For a decrement:

\[
\frac{d_{\mathrm B}(t-1)}
{d_{\mathrm B}(t)}
=
\frac{t}{M-t+1}.
\]

## W

For an increment:

\[
\frac{d_{\mathrm W}(t+1)}
{d_{\mathrm W}(t)}
=
\frac{M+t}{t+1}.
\]

For a decrement:

\[
\frac{d_{\mathrm W}(t-1)}
{d_{\mathrm W}(t)}
=
\frac{t}{M+t-1}.
\]

The first implementation should use the shared log-degeneracy API for correctness.

A later optimization may use these \(O(1)\) rational ratios to avoid repeated `lgamma` calls.

---

# 20. Direct ME fast path

Retain a direct exact method for the easy ME domain.

## Conditions

Use stub matching only when:

- family is ME;
- self-loops are allowed;
- every pair is admissible;
- there are no immutable fixed cells in the residual problem;
- potential is `NoPotential`.

## Algorithm

1. build outgoing stubs of total length \(T\);
2. build incoming stubs of total length \(T\);
3. shuffle one side uniformly;
4. aggregate pairs.

## Exactness

Uniform labelled matchings induce:

\[
P(\mathbf t\mid\mathbf s)
\propto
\frac{1}{\prod_{ij}t_{ij}!}.
\]

## Memory boundary

The method uses:

\[
O(T+E)
\]

memory.

For very large \(T\), explicit stubs can be too expensive.

Add a configurable or internal work limit:

```rust
const DEFAULT_MAX_EXPLICIT_STUBS: u64 = ...;
```

If the limit is exceeded, use the generic MCMC backend initialized by a transportation table.

Do not fail solely because the direct fast path is too large.

---

# 21. Efficient ME direct implementation improvements

The existing function stores both outgoing and incoming stub vectors.

Only one shuffled side is strictly necessary if the deterministic side is streamed by source blocks.

Possible implementation:

1. materialize incoming stubs;
2. shuffle incoming stubs;
3. iterate source strengths without materializing outgoing stubs;
4. consume the shuffled targets sequentially;
5. aggregate pairs.

This reduces direct-path stub memory from approximately:

\[
2T
\]

node identifiers to:

\[
T
\]

node identifiers.

Reserve the occupation hash map using an estimate bounded by:

\[
\min(T,L).
\]

---

# 22. Chain configuration

Reuse the generalized configuration extracted from fixed-\((\mathbf k,T)\):

```rust
pub struct McmcConfig {
    pub burn_in_sweeps: usize,
    pub sweeps_per_sample: usize,
    pub proposals_per_sweep: Option<usize>,
    pub seed: u64,
}
```

`FixedKTConfig` and `FixedStrengthConfig` should embed this same type.

Do not maintain separate burn-in, thinning, and seed structs for each constraint.

Do not define one sweep as \(T\) proposals.

For high occupations, \(T\) can be enormous while the effective dimension is controlled more closely by:

- node count;
- admissible-cell count;
- occupied-cell count.

Suggested default:

\[
P_{\mathrm{sweep}}
=
\max(4E_{\mathrm{current}},\,2N,\,1),
\]

capped by checked arithmetic and a configurable work limit.

This is a heuristic, not a mixing theorem.

For repeated sampling, expose a persistent chain.

---

# 23. Persistent chain

```rust
pub struct FixedStrengthChain<P> {
    pub state: StrengthState,
    pub target: StrengthTarget<P>,
    pub domain: ...,
    pub config: FixedStrengthMcmcConfig,
    pub diagnostics: FixedStrengthDiagnostics,
}
```

Methods:

```rust
step()
sweep()
burn_in()
sample()
current_network()
```

The one-shot API can construct and burn in a temporary chain.

Repeated public sampling should eventually use a persistent Python-facing object to avoid repeated:

- max-flow initialization;
- burn-in;
- memory allocation.

---

# 24. Connectivity and move completeness

Detailed balance is necessary but not sufficient.

## Complete unbounded tables

For a complete bipartite cell domain with no upper bounds, \(2\times2\) integer cycle moves form the standard local connectivity mechanism for tables with fixed margins.

## Structural zeros and masks

With forbidden cells, four-cycles may not connect the full feasible state space.

The difference between two feasible flows is a circulation on the bipartite admissibility graph and decomposes into alternating even cycles, which may be longer than four.

Therefore the architecture must allow a general alternating-cycle move.

## B upper bounds

Capacity constraints can also block simple four-cycle paths.

Longer feasible cycle augmentations may be required.

## Phase 4 boundary

The first implementation may use:

- \(2\times2\) cycles as the production fast kernel;
- small-system connectivity enumeration;
- a general alternating-cycle fallback for masked or disconnected cases.

Do not advertise arbitrary-mask universality with only four-cycle moves unless connectivity tests support it.

---

# 25. General alternating-cycle move

On the bipartite graph of source and target nodes, choose an even cycle:

\[
i_1,j_1,i_2,j_2,\ldots,i_r,j_r,i_1.
\]

Apply alternating deltas:

\[
+1,-1,+1,-1,\ldots
\]

around the cycle.

Each source and target receives one increment and one decrement.

Strengths remain fixed.

The reverse move is obtained by reversing the sign.

For symmetric cycle and sign selection, the proposal is symmetric.

A practical first fallback can:

1. choose a start source;
2. perform a bounded random walk alternating source/target nodes through admissible cells;
3. stop when a simple even cycle closes;
4. reject if no cycle is found within a bounded length;
5. choose sign uniformly;
6. apply ordinary Metropolis acceptance.

If cycle-generation probabilities are not symmetric, include the Hastings ratio.

The simplest safe implementation is to store the generated oriented cycle and ensure the same generation probability under reversal.

---

# 26. Cost-ready target

Phase 5 should add expected cost:

\[
C(\mathbf t)
=
\sum_{ij}c_{ij}t_{ij}.
\]

The target becomes:

\[
P_F(\mathbf t\mid\mathbf s,\gamma)
\propto
\prod_{ij}d_F(t_{ij})
e^{-\gamma C(\mathbf t)}.
\]

For a local move:

\[
\Delta\log\pi
=
\Delta\log D_F
-
\gamma\Delta C.
\]

For a four-cycle:

\[
\Delta C
=
c_{ab}+c_{cd}-c_{ad}-c_{cb}
\]

for the corresponding \(+1,+1,-1,-1\) orientation.

Thus Phase 5 should require no new state representation and no new move.

Only these pieces change:

- a cost provider;
- fitted or supplied \(\gamma\);
- local potential delta;
- cost diagnostics and validation.

---

# 27. Cost-provider reuse

Do not add a new `PairCost` trait.

Reuse:

```rust
crate::pairs::PairCostProvider
crate::pairs::EuclideanCostProvider
```

The same provider instance or provider type used by grand-canonical `StrengthCostProvider` should be usable by the fixed-strength-cost target.

Required consistency tests:

1. `None` excludes the same pair in both ensemble paths.
2. Euclidean costs are numerically identical.
3. The sign convention is identical:
   \[
   e^{-\gamma c_{ij}t_{ij}}.
   \]
4. Self-loop handling is controlled by the common pair domain, not independently by the cost provider.
5. Sparse cost providers added later work in both paths.

This is a direct scientific and engineering reuse point, not merely a similar interface.

---

# 28. Diagnostics

```rust
pub struct FixedStrengthDiagnostics {
    pub backend: FixedStrengthBackend,
    pub initializer: StrengthInitializer,
    pub proposals: u64,
    pub accepted: u64,
    pub infeasible_holds: u64,
    pub capacity_holds: u64,
    pub mask_holds: u64,
    pub metropolis_rejects: u64,
    pub occupied_pairs_initial: usize,
    pub occupied_pairs_final: usize,
    pub max_flow_edges: usize,
}
```

Backend enum:

```rust
pub enum FixedStrengthBackend {
    MeDirectStubMatching,
    CycleMcmc,
}
```

Track:

- acceptance rate;
- support turnover;
- occupation autocorrelation;
- family log-weight trajectory in debug/benchmark mode;
- cost trajectory in Phase 5.

Diagnostics should be available to Rust tests and benchmarks even if the first public Python API returns only a network.

---

# 29. Performance requirements

## Memory

### ME direct

\[
O(T+E)
\]

initially, with the recommended streamed-source optimization reducing explicit stub storage.

### Generic MCMC

\[
O(N+E+L_{\mathrm{flow}})
\]

during initialization, and:

\[
O(N+E)
\]

during the chain for implicit complete domains.

For sparse masks:

\[
O(N+E+L).
\]

No dense occupation matrix.

## Move cost

A \(2\times2\) proposal should be expected:

\[
O(1)
\]

for:

- cell lookups;
- domain checks;
- support insertions/removals;
- family delta;
- cost delta.

## Avoid repeated `lgamma`

The existing log-degeneracy API is the correctness baseline.

After validation, add specialized one-step delta methods to `OccupationFamily`:

```rust
delta_log_increment(t)
delta_log_decrement(t)
```

using `ln` of the rational formulas.

A four-cycle then needs only a few logarithms, not repeated gamma evaluations.

## Initialization

The complete-domain greedy constructor should avoid building \(N^2\) cell edges.

Max flow should be reserved for restricted domains or bounded-capacity failures.

---

# 30. Public API

Conceptually:

```python
sample_model(
    family="ME" | "B" | "W",
    constraint="STRENGTH",
    ensemble="microcanonical",
    strength_out=...,
    strength_in=...,
    layers=...,                 # B/W
    self_loops=True,
    fixed_pairs=...,
    seed=...,
    burn_in_sweeps=...,
    sweeps_per_sample=...,
)
```

For ME direct-path calls, MCMC parameters are accepted but ignored or reported as unused.

The capability registry should mark:

- ME fixed strengths: exact direct in the simple domain, stationary MCMC otherwise;
- B fixed strengths: stationary MCMC;
- W fixed strengths: stationary MCMC.

Because one route can select multiple backends, the registry may need a more general backend label:

```text
microcanonical_fixed_strength
```

and runtime diagnostics should report the selected backend.

---

# 31. Exactness language

Use precise categories.

## ME direct

- exact independent sample.

## B/W and constrained ME MCMC

- exact stationary target;
- finite-chain sample is correlated and subject to burn-in/mixing error.

Do not describe a finite MCMC run as an exact independent sample.

The hard strengths and family support are exact in every emitted state.

---

# 32. Validation hierarchy

Every family must pass:

1. feasibility validation;
2. exact hard-strength validation;
3. family support validation;
4. fixed-pair validation;
5. local detailed balance;
6. exact small-state enumeration;
7. connectivity enumeration for supported small domains;
8. conditioned grand-canonical identity;
9. repeated-chain agreement;
10. benchmark integration.

---

# 33. Exact enumeration

For small \(N\), enumerate all occupation matrices satisfying the margins and family support.

Compute:

\[
w_F(\mathbf t)
=
\prod_{ij}d_F(t_{ij}).
\]

Normalize exactly or in high-precision log space.

Compare empirical state probabilities.

Required enumerated cases:

- ME complete with loops;
- ME loopless;
- B with small \(M\);
- W with small \(M\);
- fixed cells;
- sparse masks;
- zero margins;
- saturated B rows or columns.

For ME complete with loops, compare both:

- direct stub matching;
- generic cycle MCMC.

They must agree.

---

# 34. Detailed-balance tests

Enumerate small feasible states \(x,y\) linked by a move.

Verify numerically:

\[
\pi(x)q(x,y)\alpha(x,y)
=
\pi(y)q(y,x)\alpha(y,x).
\]

For the symmetric four-cycle kernel:

\[
q(x,y)=q(y,x).
\]

Test each family independently.

For Phase 5, repeat with a nonconstant cost provider.

---

# 35. Conditioned grand-canonical identity

For each family:

\[
P_{\mathrm{GC},F}
\left(
\mathbf t
\mid
\mathbf s^{\mathrm{out}},
\mathbf s^{\mathrm{in}}
\right)
=
P_{\mathrm{MC},F}
\left(
\mathbf t
\mid
\mathbf s^{\mathrm{out}},
\mathbf s^{\mathrm{in}}
\right).
\]

For small systems:

1. generate or enumerate grand-canonical probabilities;
2. retain exact strength states;
3. normalize;
4. compare to direct enumeration and MCMC.

For Phase 5, the analogous identity retains the cost exponential tilt while conditioning only on strengths.

---

# 36. Connectivity tests

For each small residual domain:

1. enumerate all feasible states;
2. enumerate all valid four-cycle moves;
3. construct the state graph;
4. count connected components.

Repeat after enabling general alternating cycles.

Document supported domains where four-cycles are sufficient.

If arbitrary masks remain disconnected, either:

- enable longer cycles;
- restrict public support;
- or add an expanded/flow-based kernel.

Do not hide disconnected-state results.

---

# 37. Benchmark plan

Add:

```text
python -m benchmarks micro-strength
```

or:

```text
python -m benchmarks micro --constraint strength
```

Sweep:

- family;
- \(N\);
- \(T\);
- support density;
- strength heterogeneity;
- loops allowed/forbidden;
- mask density;
- fixed-cell fraction;
- B layer count \(M\);
- W layer count \(M\).

Record:

- initialization time;
- max-flow time;
- chain time;
- peak memory;
- proposals per second;
- acceptance rate;
- effective sample size;
- support turnover;
- selected backend;
- exact constraint errors;
- repeated-chain convergence.

For Phase 5, additionally record:

- cost-evaluation time;
- cost autocorrelation;
- mean sampled cost.

---

# 38. Implementation sequence

## Step 0 — Extract the reusable MCMC control layer

Refactor fixed-\((\mathbf k,T)\) without changing behavior:

- move common config to `microcanonical/mcmc/config.rs`;
- move common diagnostics counters to `microcanonical/mcmc/diagnostics.rs`;
- extract sweep, burn-in, thinning, and persistent lifecycle;
- retain fixed-degree state and switch inside `fixed_kt/`;
- run all Phase 3 tests unchanged.

Do not begin fixed-strength MCMC until this refactor is green.

## Step 1 — Extract and formalize ME direct sampling

Create `fixed_strength/me_direct.rs`.

- replace assertions with `Result`;
- stream outgoing stubs instead of storing both sides;
- add checked \(T\to usize\) conversion;
- add explicit work limit and MCMC fallback;
- add fixed-strength tests;
- preserve current public behavior.

## Step 2 — Add residual problem type

Create:

```rust
FixedStrengthProblem
ResidualStrengthProblem
```

Integrate shared `FixedPairs` and mask abstractions.

## Step 3 — Add family target and no-potential abstraction

Use `OccupationFamily`.

Implement:

```text
StrengthTarget<NoPotential>
```

and unit tests for ME/B/W local deltas.

## Step 4 — Add sparse occupation state

Implement atomic local updates and output conversion.

## Step 5 — Add feasibility and initialization

- complete-domain greedy transportation;
- sparse/bounded max-flow fallback;
- exact infeasibility errors.

## Step 6 — Add symmetric four-cycle kernel

- uniform source pair;
- uniform target pair;
- uniform sign;
- switch-and-hold;
- log-space Metropolis.

## Step 7 — Validate ME MCMC against direct ME

This is the first correctness oracle.

Test:

- exact enumeration;
- empirical agreement;
- detailed balance;
- constraint preservation.

## Step 8 — Route constrained ME cases through MCMC

Support:

- no self-loops;
- masks;
- fixed cells;
- large \(T\) beyond direct work limit.

## Step 9 — Add B

Use the same state, initializer, and moves.

Only family validation and local degeneracy change.

Ensure:

\[
0\le t_{ij}\le M.
\]

## Step 10 — Add W

Use the same state, initializer, and moves.

Only local degeneracy changes.

## Step 11 — Add alternating-cycle fallback

Required before claiming robust arbitrary-mask support.

## Step 12 — Python routing and registry

Expose all three families under microcanonical `STRENGTH`.

## Step 13 — Benchmarks and documentation

Add exactness wording, MCMC controls, and limitations.

## Step 14 — Prepare Phase 5 without exposing it

Land:

- `PairPotential`;
- `NoPotential`;
- test-only `LinearCostPotential`;
- cost-delta detailed-balance tests.

Do not expose fixed-strength-cost publicly until fitting/routing and validation are complete.

---

# 39. Suggested commit sequence

## Commit 1

```text
refactor(microcanonical): extract shared MCMC control from fixed kt
```

## Commit 2

```text
refactor(microcanonical): formalize ME fixed-strength direct sampler
```

## Commit 3

```text
feat(microcanonical): add residual strength problem and transportation initializer
```

## Commit 4

```text
feat(microcanonical): add generic strength-constrained cycle MCMC
```

## Commit 5

```text
test(microcanonical): validate ME direct and cycle samplers by enumeration
```

## Commit 6

```text
feat(microcanonical): add B and W fixed-strength targets
```

## Commit 7

```text
feat(microcanonical): add alternating-cycle moves for restricted domains
```

## Commit 8

```text
feat(api): expose microcanonical fixed-strength ME B W
```

## Commit 9

```text
perf(microcanonical): benchmark and optimize strength samplers
```

---

# 40. Explicit boundaries for the first release

The first complete Phase 4 release should guarantee:

- directed networks;
- exact in-strengths and out-strengths;
- ME/B/W family targets;
- self-loops allowed or forbidden;
- fixed cells;
- complete and supported sparse domains;
- exact family occupation bounds;
- sparse state;
- direct ME fast path;
- stationary-correct MCMC for generic cases.

It should not claim:

- independent samples from MCMC;
- universal rapid mixing;
- arbitrary-mask connectivity without validation;
- efficient explicit-stub ME sampling for unlimited \(T\);
- cost-constrained sampling before Phase 5 is exposed;
- fixed \(E\) or fixed degree simultaneously with strengths.

---

# 41. Completion criteria

Phase 4 is complete when:

- the old free stub-matching function is replaced by a structured module;
- ME direct sampling remains exact and tested;
- residual fixed-cell preprocessing works;
- exact feasibility is implemented through transportation/max flow;
- sparse joint occupation state is implemented;
- symmetric strength-preserving moves are implemented;
- `OccupationFamily::log_local_degeneracy` is reused;
- ME, B, and W pass exact enumeration;
- local detailed balance passes;
- connectivity is measured for supported domains;
- no-self-loop ME is supported through the MCMC backend;
- B capacities are enforced;
- W sampling is stable in log space;
- conditioned grand-canonical identities pass;
- Python capabilities are updated;
- benchmarks report time, memory, acceptance, and mixing diagnostics;
- no \(O(N^2)\) occupation matrix is allocated;
- the optional-potential abstraction is ready for Phase 5.

---

# 42. Strategic result

After Phase 4, the microcanonical architecture will contain three progressively more general patterns:

```text
fixed (E,T)
    direct support
    direct occupations

fixed (k,T)
    support MCMC
    direct occupations

fixed strengths
    joint occupation-state MCMC
```

The reusable core introduced here is:

\[
\boxed{
\text{hard-margin state}
+
\text{alternating-cycle moves}
+
\text{family local degeneracy}
+
\text{optional local potential}
}
\]

Phase 5 then becomes:

\[
\boxed{
\text{Phase 4 target}
-
\gamma\Delta C
}
\]

rather than a new sampler.

The immediate implementation priority is:

\[
\boxed{
\text{Extract the exact ME fast path, then build the generic cycle-MCMC core around }
\texttt{OccupationFamily::delta\_log\_local\_degeneracy}.
}
\]

---

# 43. Final architecture review — maximum reuse without overengineering

This section is normative. It resolves the balance between reuse and code complexity.

## 43.1 Reuse rule

A component should be extracted as shared infrastructure only when all three conditions hold:

1. it is already used by fixed \((\mathbf k,T)\);
2. fixed strengths needs the same semantics, not merely a similar implementation;
3. the extraction does not require the state or kernel to implement a large generic interface.

Under this rule, share:

- RNG seeding;
- burn-in and thinning configuration;
- sweep loops;
- proposal and acceptance counters;
- persistent-chain lifecycle conventions;
- exactness and diagnostics vocabulary;
- `OccupationFamily`;
- `PairCostProvider`;
- fixed-pair and mask preprocessing conventions.

Do not share:

- binary support state;
- occupation state;
- fixed-degree initialization;
- transportation initialization;
- binary edge switches;
- integer cycle moves;
- complement representation.

## 43.2 Keep the shared MCMC layer deliberately small

Do not build a general-purpose MCMC framework with:

- dynamic dispatch over arbitrary state types;
- boxed kernels;
- generic transition graphs;
- generic acceptance-policy objects;
- pluggable observer systems;
- deeply nested traits.

The shared layer should consist of plain data and small helper functions.

Recommended minimum:

```rust
pub struct McmcConfig {
    pub burn_in_sweeps: usize,
    pub sweeps_per_sample: usize,
    pub proposals_per_sweep: Option<usize>,
    pub seed: u64,
}

pub struct McmcCounters {
    pub proposals: u64,
    pub accepted: u64,
    pub held: u64,
    pub metropolis_rejected: u64,
}

pub enum McmcOutcome {
    Accepted,
    Held,
    Rejected,
}
```

Optional helpers:

```rust
run_sweeps(...)
run_burn_in(...)
```

The fixed-degree and fixed-strength chains should remain concrete structs.

This gives reuse without forcing every sampler through a complex trait hierarchy.

## 43.3 Prefer concrete chains over a universal generic chain

Keep:

```rust
FixedDegreeChain
FixedStrengthChain
```

Both may embed:

```rust
McmcConfig
McmcCounters
```

and call the same helper functions.

Do not replace them with a single highly generic:

```rust
Chain<State, Kernel, Target, Domain, Observer, ...>
```

unless a third production sampler proves that such a type is genuinely simpler.

At Phase 4, concrete chain types are easier to:

- read;
- debug;
- benchmark;
- expose through PyO3;
- specialize for performance.

## 43.4 Reuse the cost provider directly, but not the grand-canonical distribution provider

Reuse:

```rust
PairCostProvider
EuclideanCostProvider
```

Do not reuse:

```rust
PairDistributionProvider
StrengthCostProvider
```

inside the microcanonical chain.

Reason:

- `PairCostProvider` expresses pair-local cost data and admissibility;
- `StrengthCostProvider` constructs independent grand-canonical pair distributions using fitted multipliers;
- the microcanonical chain needs only local cost differences.

The fixed-strength-cost target should therefore call:

```rust
costs.cost(source, target)
```

directly.

This is maximum scientific reuse with minimum coupling.

## 43.5 Reuse `OccupationFamily` as the only family abstraction

Do not add:

```rust
StrengthFamily
MicrocanonicalFamily
LocalDegeneracy
JointStateFamily
```

The existing `OccupationFamily` already provides:

- family identity;
- layer count;
- occupation support;
- occupation validation;
- local log-degeneracy;
- local degeneracy delta.

Add only narrowly useful performance helpers to that existing enum, for example:

```rust
delta_log_increment(t)
delta_log_decrement(t)
```

These should be implemented alongside `delta_log_local_degeneracy`, not in the fixed-strength module.

## 43.6 Use one domain abstraction across feasibility and MCMC

A single pair-domain representation should serve:

- fixed-pair residualization;
- max-flow feasibility;
- initializer construction;
- cycle validity checks;
- Phase 5 cost exclusion.

Avoid separate:

```text
Mask
AdmissiblePairs
FlowDomain
CycleDomain
CostDomain
```

with duplicated semantics.

Recommended concrete enum:

```rust
pub enum PairDomain<'a> {
    Complete {
        node_count: usize,
        self_loops: bool,
    },
    Sparse {
        node_count: usize,
        allowed: &'a HashSet<PairId>,
    },
}
```

Provide:

```rust
is_admissible(pair)
iter_admissible()
```

Only add a trait if multiple external domain providers become necessary.

## 43.7 Do not over-unify initializers

The following initializers solve different mathematical problems:

- fixed-degree support realization;
- fixed-strength transportation realization;
- ME labelled-stub matching.

They should not implement a common initializer trait solely for symmetry.

Use separate concrete functions:

```rust
greedy_directed_initialize(...)
initialize_transportation_table(...)
sample_me_stub_matching(...)
```

Share only error conventions and residual preprocessing.

## 43.8 Keep the first move kernel simple

The production baseline should be one concrete four-cell cycle function:

```rust
cycle4_step(
    state,
    target,
    domain,
    rng,
    counters,
)
```

Do not introduce an abstract move registry in the first implementation.

Add longer alternating-cycle moves only after connectivity tests demonstrate they are required.

When they are added, use a small enum or explicit mixture:

```rust
enum StrengthMoveKind {
    Cycle4,
    AlternatingCycle,
}
```

rather than a boxed plugin system.

## 43.9 Separate correctness path from optimization path

First implementation:

- use `delta_log_local_degeneracy`;
- use hash-backed sparse occupations;
- use exact max flow for restricted feasibility;
- use symmetric node-based cycle proposals;
- use log-space Metropolis.

Only after exact validation:

- add rational one-step degeneracy ratios;
- add reusable scratch buffers;
- optimize hash maps;
- tune proposal mixtures;
- cache cost values when profitable.

Do not make the first code harder to audit for speculative speedups.

## 43.10 Recommended final dependency graph

```text
crate::distribution::OccupationFamily
crate::pairs::PairCostProvider
crate::constraints::{FixedPairs, PairMask}
generation::microcanonical::mcmc::{
    McmcConfig,
    McmcCounters,
    McmcOutcome,
    sweep helpers,
}
                |
                +----------------------------+
                |                            |
                v                            v
        fixed_kt concrete chain      fixed_strength concrete chain
        binary support state         integer occupation state
        edge-switch kernel           cycle kernel
        degree initializer           transportation initializer
```

Phase 5 reuses the fixed-strength chain exactly:

```text
fixed strengths:
    gamma = 0
    costs = None

fixed strengths + expected cost:
    gamma = fitted value
    costs = Some(existing PairCostProvider)
```

## 43.11 Final implementation boundary

The Phase 4 implementation should introduce only these new major components:

1. structured ME direct sampler;
2. residual fixed-strength problem;
3. transportation feasibility and initialization;
4. sparse integer occupation state;
5. four-cell cycle kernel;
6. fixed-strength concrete chain;
7. ME/B/W routing;
8. small shared MCMC config/counter extraction;
9. direct reuse of `OccupationFamily`;
10. direct reuse of `PairCostProvider` for Phase 5 readiness.

Everything else should remain deferred until a demonstrated need exists.

## 43.12 Final review verdict

The preferred balance is:

\[
\boxed{
\text{share control, mathematics, providers, and preprocessing;}
\quad
\text{keep states, moves, and initializers constraint-specific.}
}
\]

This maximizes useful reuse while preserving:

- local reasoning;
- performance specialization;
- simple Rust types;
- testability;
- a clear migration path to fixed strengths plus expected cost.
