# Phase 3 Plan — Microcanonical Fixed \((\mathbf k,T)\)

**Repository:** `uladribia/menobis`  
**Planning baseline:** commit `52426033f4b56bdef220f55a85b47b82135fabe8`  
**Version:** August 2026

---

# 1. Purpose

This document defines the next implementation step for the MENoBiS microcanonical project.

The immediate target is:

\[
\boxed{\text{fixed out-degree sequence } \mathbf k^{\mathrm{out}} \text{, fixed in-degree sequence } \mathbf k^{\mathrm{in}} \text{, and fixed total occupation } T}
\]

for non-binary network families.  MENoBiS models directed graphs — every admissible pair
\((i,j)\) is an ordered pair, and degree constraints come as separate out- and
in-degree vectors.

The first implementation should be the **ME family**, followed by B and W after the common support machinery is validated.

This phase should not yet implement the full joint occupation-state MCMC framework required by fixed \((\mathbf s,E)\). Instead, it should introduce the smallest new abstraction needed after fixed \((E,T)\):

> a constrained support sampler that produces a simple directed graph with exactly the requested out- and in-degree sequences.

Once the support is sampled, the existing fixed-\((E,T)\) occupation allocators can be reused unchanged.

---

# 2. Current repository state

The repository already contains a complete fixed-\((E,T)\) implementation for ME, B, and W.

The current implementation provides:

- a shared `FixedETOccupancy` trait;
- one generic fixed-\((E,T)\) orchestrator;
- uniform support sampling;
- family-specific positive-occupation allocation;
- rejection and exact fallback backends;
- fixed-pair residualization;
- sparse output construction;
- exact enumeration validation;
- conditioned grand-canonical validation;
- benchmark integration;
- public documentation.

The current fixed-\((E,T)\) layout is:

```text
crates/menobis-core/src/generation/microcanonical/
    mod.rs
    fixed_et/
        mod.rs
        core.rs
        support.rs
        pairs.rs
        errors.rs
        me.rs
        b.rs
        w.rs
```

The shared orchestrator currently performs:

```text
validation
    |
    v
uniform support sampling
    |
    v
family-specific positive occupation allocation
    |
    v
SampledNetwork construction
```

This is the correct base for Phase 3.

The scientific backlog already identifies fixed \((\mathbf k,T)\) as the natural next phase.

---

# 3. Overall microcanonical architecture

The long-term architecture should distinguish three layers.

## 3.1 Problem layer

Defines:

- family;
- residual admissible-pair domain;
- hard constraints;
- fixed-pair contribution;
- graph directionality;
- self-loop policy.

Examples:

```text
FixedETProblem
FixedKTProblem
FixedStrengthProblem
FixedSEProblem
```

## 3.2 Support layer

Generates the occupied-pair support when the support constraint can be handled independently from positive occupations.

Examples:

- fixed \(E\): uniform subset of admissible ordered pairs;
- fixed \(\mathbf k\): uniform constrained simple directed graph;
- fixed support: deterministic support;
- fixed \((\mathbf s,E)\): not separable in general.

## 3.3 Occupation layer

Given a support of size \(E\), generates positive occupations satisfying the remaining occupation constraints.

For fixed total occupation \(T\), the existing family allocators already solve this problem:

```text
ME:
    multinomial rejection
    + Stirling fallback

B:
    binary-cell subset rejection
    + bounded DP fallback

W:
    weak-composition rejection
    + unbounded DP fallback
```

This separation should become explicit:

```text
Microcanonical sampler
    |
    +-- support sampler
    |
    +-- occupation allocator
    |
    +-- sparse output builder
```

Fixed \((E,T)\) and fixed \((\mathbf k,T)\) should share the same occupation layer.

---

# 4. Why fixed \((\mathbf k,T)\) is the correct next phase

Fixed \((\mathbf k,T)\) adds one major new difficulty while preserving the successful factorization from fixed \((E,T)\).

MENoBiS models **directed** graphs, so the degree constraint is a pair of
vectors.  For a directed simple support graph,

\[
E=\sum_i k_i^{\mathrm{out}}
=
\sum_i k_i^{\mathrm{in}}.
\]

Once a support graph \(G\) realizing \((\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}})\) is fixed, the family-specific occupation distribution depends only on:

- the support size \(E\);
- the total occupation \(T\);
- the family parameters.

It does not depend on the identities of the support pairs.

Therefore the exact distribution factorizes as:

\[
P(\mathbf t\mid \mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}},T)
=
P(G\mid\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}})
P(\mathbf t\mid G,T).
\]

The implementation becomes:

```text
sample support uniformly subject to out- and in-degree sequences
    |
    v
compute E from k (E = sum out-degrees = sum in-degrees)
    |
    v
reuse existing positive occupation allocator for (E,T)
    |
    v
assign occupations to the sampled support
```

This is a much smaller architectural jump than fixed \((\mathbf s,E)\), where support and occupation numbers are coupled.

---

# 5. Mathematical factorization

Let \(G\) be a support directed graph with out-degree sequence
\(\mathbf k^{\mathrm{out}}\) and in-degree sequence \(\mathbf k^{\mathrm{in}}\).

Let its \(E\) occupied ordered pairs carry positive occupations

\[
t_1,\ldots,t_E,
\]

with

\[
t_e\ge1,
\qquad
\sum_{e=1}^{E}t_e=T.
\]

For each family:

## ME

\[
D_{\mathrm{ME}}(\mathbf t)
=
\frac{T!}{\prod_{e=1}^{E}t_e!}.
\]

## B

\[
D_{\mathrm B}(\mathbf t)
=
\prod_{e=1}^{E}\binom{M}{t_e}.
\]

## W

\[
D_{\mathrm W}(\mathbf t)
=
\prod_{e=1}^{E}
\binom{M+t_e-1}{t_e}.
\]

For fixed \(E,T\), and \(M\) where applicable, the positive-occupation partition function is identical for every support graph with \(E\) pairs.

Therefore:

\[
P(G\mid\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}},T,F)
=
P(G\mid\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}}),
\]

and, under the intended unbiased support ensemble,

\[
P(G\mid\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}})
=
\frac{1}{|\mathcal G(\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}})|}.
\]

Thus:

> fixed \((\mathbf k,T)\) requires a uniform sampler over feasible support directed graphs with degree sequences \((\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}})\), followed by the existing family occupation allocator.

No family-specific support sampler is required.

---

# 6. Scope of the first implementation

The first implementation should be deliberately restricted.

## Required first target

- **directed** network (ordered pairs, no symmetric-pair assumption);
- no self-loops;
- simple support (no parallel directed edges);
- ME family;
- fixed out-degree sequence \(\mathbf k^{\mathrm{out}}\);
- fixed in-degree sequence \(\mathbf k^{\mathrm{in}}\);
- fixed total occupation \(T\);
- no fixed positive pairs in the first internal milestone;
- no arbitrary sparse mask in the first internal milestone;
- exact degree preservation;
- support MCMC based on directed double-edge switches;
- existing ME fixed-\((E,T)\) occupation allocator.

## Required before public completion

- self-loop policy consistent with MENoBiS;
- fixed positive-pair residualization;
- explicit admissible-pair path;
- B and W family reuse;
- exact small-state validation;
- conditioned grand-canonical identity;
- benchmark integration;
- Python routing and capability registry update.

## Deferred

- joint support/occupation MCMC;
- fixed strengths;
- fixed \((\mathbf s,E)\);
- advanced Coolen mobility corrections;
- pseudo-marginal support backends;
- large constrained-mask optimization;
- guaranteed polynomial-time mixing claims.

---

# 7. Constraint naming and public model identity

The existing project terminology indicates this constraint should be represented as:

```text
DEGREE_EVENTS
```

with:

- out-degree sequence \(\mathbf k^{\mathrm{out}}\);
- in-degree sequence \(\mathbf k^{\mathrm{in}}\);
- total occupation \(T\).

The public API should remain consistent with existing constraint identifiers.

Conceptually:

```python
sample_model(
    family="ME",
    constraints="DEGREE_EVENTS",
    degree_out=...,
    degree_in=...,
    total_events=...,
    ensemble="microcanonical",
    ...
)
```

Exact parameter names must follow the existing Python API conventions rather than introducing a parallel naming style.

The capability registry should indicate:

```text
sample
microcanonical
DEGREE_EVENTS
ME/B/W
```

only when the corresponding family backend is complete and tested.

---

# 8. Residualization with fixed pairs

Fixed-pair preprocessing must remain outside the sampler.

Suppose fixed positive pairs are given by occupations

\[
t_{ij}^{\mathrm{fix}}>0.
\]

They contribute fixed support out- and in-degrees:

\[
k_i^{\mathrm{out,fix}}
=
\sum_j
\mathbf 1(t_{ij}^{\mathrm{fix}}>0),
\qquad
k_j^{\mathrm{in,fix}}
=
\sum_i
\mathbf 1(t_{ij}^{\mathrm{fix}}>0)
\]

for directed networks.

They also contribute:

\[
E_{\mathrm{fix}}
=
\sum_i k_i^{\mathrm{out,fix}}
=
\sum_i k_i^{\mathrm{in,fix}},
\]

and

\[
T_{\mathrm{fix}}
=
\sum_{i,j}t_{ij}^{\mathrm{fix}}.
\]

The residual constraints are:

\[
k_i^{\mathrm{out,res}}
=
k_i^{\mathrm{out,target}}
-
k_i^{\mathrm{out,fix}},
\qquad
k_i^{\mathrm{in,res}}
=
k_i^{\mathrm{in,target}}
-
k_i^{\mathrm{in,fix}},
\]

\[
T_{\mathrm{res}}
=
T_{\mathrm{target}}
-
T_{\mathrm{fix}}.
\]

Then:

\[
E_{\mathrm{res}}
=
\sum_i k_i^{\mathrm{out,res}}
=
\sum_i k_i^{\mathrm{in,res}}.
\]

Fixed positive pairs must be excluded from the residual support domain because selecting them again would not create a new occupied pair.

Fixed zero pairs remain forbidden through the residual admissible-pair representation.

The reconstructed graph is:

```text
fixed positive occupations
    +
sampled residual positive occupations
```

Final validation must check the original global out-degree sequence, in-degree sequence, and total occupation.

---

# 9. Residual feasibility

For a directed loopless simple support graph, basic checks include:

\[
k_i^{\mathrm{out,res}}\ge0,\qquad
k_i^{\mathrm{in,res}}\ge0,
\]

\[
\sum_i k_i^{\mathrm{out,res}}
=
\sum_i k_i^{\mathrm{in,res}}
(=
E_{\mathrm{res}}),
\]

\[
T_{\mathrm{res}}\ge E_{\mathrm{res}}.
\]

Also:

\[
k_i^{\mathrm{out,res}}
\le a_i^{\mathrm{out}},
\qquad
k_i^{\mathrm{in,res}}
\le a_i^{\mathrm{in}},
\]

where \(a_i^{\mathrm{out}}\) is the number of residual admissible out-neighbours
of node \(i\) (i.e., distinct \(j\neq i\) that are not fixed-occupied) and
\(a_i^{\mathrm{in}}\) is the analogous count for in-neighbours.

Without an arbitrary mask, use a full directed graphicality test:

- **Fulkerson–Chen–Anstee** theorem (necessary and sufficient conditions for a
  simple directed graph with given out- and in-degree sequences); or
- equivalently, the bipartite realization problem: treat each node as a left
  vertex with \(k_i^{\mathrm{out}}\) stubs and a right vertex with
  \(k_i^{\mathrm{in}}\) stubs, and check whether a simple bipartite graph exists
  (Gale–Ryser theorem for 0–1 matrices with prescribed row and column sums),
  with the additional constraint that no edge connects left-\(i\) to right-\(i\)
  when self-loops are forbidden.

With an admissibility mask, ordinary graphicality is insufficient.

The masked problem becomes a degree-constrained subgraph or \(f\)-factor feasibility problem.

The implementation should use a hierarchy:

1. cheap necessary checks;
2. deterministic unmasked graphicality;
3. constructive masked search;
4. exact small-instance oracle.

A failed heuristic constructor must not be reported as a proof of infeasibility.

---

# 10. Support state representation

Use a sparse binary support state.

Recommended fields:

```rust
struct DegreeSupportState {
    node_count: usize,
    edges: Vec<PairId>,                // ordered (src,tgt) pairs
    edge_positions: HashMap<PairId, usize>,
    out_adjacency: Vec<HashSet<u32>>,  // outgoing neighbours per node
}
```

The exact container types should follow repository conventions.

Required operations:

- sample one support edge uniformly;
- sample two distinct support edges uniformly;
- test whether a candidate ordered pair is occupied;
- test whether a candidate ordered pair is admissible;
- insert and remove ordered pairs;
- preserve ordered (source, target) representation;
- verify out-degrees (and optionally in-degrees) in debug/test builds.

Memory must remain:

\[
O(N+E).
\]

No dense adjacency matrix should be introduced.

---

# 11. Initial support construction

The MCMC requires one feasible support directed graph.

## 11.1 Unmasked directed case

Use a greedy directed constructor.

Implementation outline:

1. Maintain a max-heap of nodes keyed by residual out-degree.
2. For each node \(u\) (popped in order of decreasing out-degree \(d_u^{\mathrm{out}}\)):
   a. Select the \(d_u^{\mathrm{out}}\) distinct target nodes with the largest
      residual in-degrees, skipping \(u\) (if self-loops are forbidden) and
      any already-occupied pair \((u, v)\).
   b. Add a directed edge \(u \to v\) for each selected target \(v\).
   c. Decrement the residual in-degree of each selected target.
3. Repeat until all residual out-degrees are zero.

The resulting directed graph is feasible if the sequence pair passed
directed graphicality validation.

This constructor is:

- bounded;
- deterministic;
- robust for hub-dominated sequences.

Expected complexity:

\[
O(E \log N).
\]

## 11.2 Self-loops allowed

If the support is still simple and loop occupancy means a support pair \((i,i)\),
the graphicality conditions and degree convention must be defined explicitly.

In a directed graph with self-loops allowed, a loop \((i,i)\) contributes:

- one to \(k_i^{\mathrm{out}}\);
- one to \(k_i^{\mathrm{in}}\).

This follows the existing MENoBiS pair representation (ordered pairs; a loop is
simply \((i,i)\)).  The greedy constructor must skip self-loops when they are
forbidden, and may include them when permitted.

Do not silently reuse the loopless initializer for sequences that require
self-loops.

## 11.3 Masked case

A mask-aware greedy directed heuristic may be attempted with bounded
backtracking.

However, it is not guaranteed to find a feasible realization.

Use one of:

- bounded recursive backtracking for small instances;
- max-flow or \(b\)-matching formulation on the bipartite representation;
- MILP feasibility oracle;
- annealed support repair with bounded restarts.

The first public milestone may restrict arbitrary masks if necessary, but fixed-pair residualization should still be designed correctly.

---

# 12. Uniform support MCMC

The first support backend should use the standard **directed** double-edge switch.

Select two distinct occupied ordered pairs:

\[
a\to b,
\qquad
c\to d.
\]

Always propose the reconnection:

\[
a\to d,
\qquad
c\to b.
\]

Each node appears with the same out- and in-degree multiplicity before and after.

Therefore the out- and in-degree sequences are preserved exactly.

This is the only symmetric degree-preserving move for directed simple graphs.
The undirected double-edge switch (random choice between two cross-pairings) is
not used — directed edges have a fixed orientation, so the reconnection is
deterministic given the two selected edges.

---

# 13. Switch-and-hold validity rules

A proposed switch is invalid if it creates:

- a forbidden self-loop;
- a pair outside the residual admissible set;
- a duplicate occupied ordered pair;
- an unchanged no-op transition;
- a malformed pair caused by repeated endpoints;
- removal and reinsertion logic that does not leave exactly \(E\) support pairs.

Invalid proposals must result in a hold.

The state must not resample until it finds a valid move, because rejection-free resampling would alter the transition probabilities.

Correct logic:

```text
sample proposal once
    |
    +-- valid -> apply
    |
    +-- invalid -> retain current state
```

This is the switch-and-hold principle.

---

# 14. Proposal symmetry

If:

1. the two old support edges are chosen uniformly without replacement;
2. the reconnection \(a\to d, c\to b\) is deterministic (no orientation
   coin-flip needed);
3. invalid proposals hold;

then valid forward and reverse switches have equal proposal probability under the basic edge-pair proposal.

Every valid move can be accepted with probability one.

The chain then has a uniform stationary law over its connected support-state component.

If future optimization changes proposal selection, such as:

- selecting only swappable edge pairs;
- degree-weighted endpoint selection;
- cached valid moves;
- mobility-biased proposals;

then proposal asymmetry must be corrected explicitly.

Do not introduce mobility-biased optimization in the first implementation.

---

# 15. Switch pseudocode

```text
directed_switch_step(state, admissible_domain, self_loops, rng):

    if state.edge_count < 2:
        return Hold(TooFewEdges)

    (a, b), (c, d) =
        sample_two_distinct_edges_uniformly(state.edges, rng)

    // Directed: fixed orientation, always reconnect a→d, c→b
    candidate_1 = (a, d)
    candidate_2 = (c, b)

    if candidate_1 == candidate_2:
        return Hold(DuplicateCandidate)

    if loops forbidden:
        if a == d or c == b:
            return Hold(SelfLoop)

    if either candidate is not admissible:
        return Hold(ForbiddenPair)

    old_set = {(a,b), (c,d)}
    new_set = {candidate_1, candidate_2}

    if new_set == old_set:
        return Hold(NoOp)

    if candidate_1 occupied outside old_set:
        return Hold(ParallelSupportEdge)

    if candidate_2 occupied outside old_set:
        return Hold(ParallelSupportEdge)

    remove (a,b) and (c,d)
    insert candidate_1 and candidate_2

    return Switched
```

All pairs are ordered \((source, target)\) — no min/max canonicalization is needed.

---

# 16. Connectivity

The directed double-edge switch is the natural first kernel, but connectivity must be validated rather than assumed for every masked domain and self-loop policy.

For small cases:

1. enumerate every feasible support directed graph realizing \((\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}})\);
2. apply every possible directed double-edge switch;
3. construct the support transition graph;
4. count connected components.

Perform this for:

- loopless complete admissible domain;
- self-loop policy variants;
- fixed-pair residual cases;
- small masked domains.

If disconnected examples occur under supported policies, add a larger support move or constrain the advertised backend scope.

Potential later kernels include:

- three-edge alternating cycle switches;
- alternating-cycle moves;
- Coolen mobility-aware kernels;
- expanded support ensembles.

Do not claim universal ergodicity until tested or proven for the supported domain.

---

# 17. Burn-in and sampling cadence

The support sampler is MCMC, so unlike fixed \((E,T)\), it produces correlated support samples.

Define one support sweep as approximately:

\[
E
\]

directed double-edge-switch attempts.

Configuration should include:

```rust
struct SupportMcmcConfig {
    burn_in_sweeps: usize,
    sweeps_per_sample: usize,
    seed: u64,
}
```

The first public API may emit one graph per call after burn-in.

For repeated sampling, prefer a persistent chain interface rather than reconstructing and burning in from scratch for every sample.

Defaults must be documented as heuristics.

They are not universal mixing guarantees.

---

# 18. Reusing the existing occupation layer

The current fixed-\((E,T)\) code combines support and occupation allocation inside `sample_fixed_et_core`.

Phase 3 should refactor this carefully.

Extract a reusable internal positive-occupation function:

```rust
pub(crate) fn sample_positive_occupations<F: FixedETOccupancy>(
    family: &F,
    total: OccNum,
    edge_count: usize,
    rng: &mut StdRng,
) -> Result<Vec<OccNum>, FixedETError>;
```

Currently this logic is private to `core.rs`.

The refactor should:

- preserve fixed-\((E,T)\) behavior exactly;
- preserve RNG determinism where practical;
- avoid duplicating family logic;
- keep special cases in one place;
- keep fallback thresholds and limits centralized.

Then fixed \((\mathbf k,T)\) can call:

```text
occupations =
    sample_positive_occupations(
        family,
        residual_total,
        residual_edge_count,
        rng
    )
```

after the degree support has been sampled.

The family files `me.rs`, `b.rs`, and `w.rs` should not gain degree-specific code.

---

# 19. Proposed module layout

Recommended layout:

```text
crates/menobis-core/src/generation/microcanonical/
    mod.rs

    occupation/
        mod.rs
        fixed_total.rs

    support/
        mod.rs
        uniform_edges.rs

    fixed_et/
        mod.rs
        core.rs
        me.rs
        b.rs
        w.rs
        pairs.rs
        errors.rs

    fixed_kt/
        mod.rs
        core.rs
        state.rs
        feasibility.rs
        initializer.rs
        switch.rs
        sampler.rs
        diagnostics.rs
        errors.rs
```

The `fixed_kt/` directory owns the directed fixed-degree support kernel.
No `support/fixed_degree/` subdirectory is needed — the directed kernel lives
directly under `fixed_kt/`.  The `support/uniform_edges.rs` module (moved from
`fixed_et/support.rs`) provides the uniform flat support sampler shared by both
fixed-\((E,T)\) and any future ensemble that needs it.

The essential requirement is that occupation allocation be reusable without importing the fixed-\((E,T)\) support orchestrator.

---

# 20. Proposed core interfaces

## 20.1 Support sampler trait

```rust
pub trait SupportSampler {
    type Error;

    fn edge_count(&self) -> usize;

    fn sample_support(
        &mut self,
        rng: &mut StdRng,
    ) -> Result<Vec<(u64, u64)>, Self::Error>;
}
```

For persistent chains, a stateful interface may be preferable:

```rust
pub trait SupportChain {
    fn step(&mut self, rng: &mut StdRng) -> SupportStep;
    fn sweep(&mut self, rng: &mut StdRng);
    fn current_support(&self) -> &[(u64, u64)];
}
```

Avoid over-generalizing before the first implementation.

## 20.2 Fixed-\((\mathbf k,T)\) orchestrator

```rust
pub fn sample_fixed_kt_core<F>(
    family: &F,
    residual_out_egrees: &[usize],
    residual_in_egrees: &[usize],
    residual_total: OccNum,
    domain: &AdmissiblePairDomain,
    config: &FixedKTConfig,
) -> Result<SampledNetwork, FixedKTError>
where
    F: FixedETOccupancy;
```

The function should:

1. validate residual constraints;
2. compute residual \(E\);
3. initialize support;
4. burn in support chain;
5. obtain one support;
6. allocate positive occupations;
7. pair occupations with support;
8. build sparse output;
9. validate result.

---

# 21. ME first, then B and W

The support distribution is family-independent.

Therefore the implementation sequence should be:

## Milestone 1 — ME only

- fixed-degree support state;
- feasibility (directed);
- greedy directed initializer;
- directed double-edge switch;
- support-chain tests;
- ME occupation reuse;
- end-to-end exact validation.

## Milestone 2 — B reuse

Add B by routing the sampled support through `BFamily`.

Validate:

\[
E\le T\le ME.
\]

No support changes are needed.

## Milestone 3 — W reuse

Add W by routing through `WFamily`.

Validate:

\[
T\ge E.
\]

No support changes are needed.

This sequence tests the architectural claim that family affects occupation allocation but not degree-support generation.

---

# 22. Directed networks are the default

MENoBiS models **directed** graphs throughout.  The fixed-\((\mathbf k,T)\)
implementation is directed from the start — there is no undirected path.

The degree constraint is always a pair of vectors:

\[
\mathbf k^{\mathrm{out}},\qquad \mathbf k^{\mathrm{in}}.
\]

The support is a simple directed graph (ordered pairs, no parallel edges).

The directed switch selects:

\[
a\to b,\qquad c\to d,
\]

and proposes:

\[
a\to d,\qquad c\to b.
\]

This preserves all in- and out-degrees.

Directed initialization uses the greedy directed constructor (section 11.1)
or, for the masked case, a bipartite realization algorithm.

Public types must never assume a single degree vector.  The
`UndirectedDegreeSequence` type should not exist; the project uses
pairs of out/in sequences.

---

# 23. Exact validation

## 23.1 Support enumeration

For small \(N\), enumerate every feasible binary support directed graph with
degree sequences \((\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}})\).

Verify that the support MCMC visits each support uniformly.

Test:

- support frequencies;
- edge marginals;
- transition detailed balance;
- connected components.

## 23.2 Full ME state enumeration

For each support, enumerate positive occupations summing to \(T\).

The exact ME weight is:

\[
w(\mathbf t)
=
\frac{1}{\prod_e t_e!}.
\]

Because every support has the same occupation partition function, the support marginal should be uniform.

Compare the complete sampled graph distribution with exact probabilities.

## 23.3 Factorization test

Test support and occupation components independently:

1. force a fixed support and validate occupation allocation;
2. ignore occupations and validate support uniformity;
3. validate the full composed sampler.

This makes failures easier to localize.

---

# 24. Conditioned grand-canonical identity

Use the existing grand-canonical `DEGREE_EVENTS` model when available.

Generate grand-canonical samples and condition on:

\[
\mathbf k^{\mathrm{out}}(\mathbf t)=\mathbf k^{\mathrm{out}},
\qquad
\mathbf k^{\mathrm{in}}(\mathbf t)=\mathbf k^{\mathrm{in}},
\]

and

\[
T(\mathbf t)=T.
\]

Then verify:

\[
P_{\mathrm{GC}}
\left(
\mathbf t
\mid
\mathbf k^{\mathrm{out}}(\mathbf t)=\mathbf k^{\mathrm{out}},
\mathbf k^{\mathrm{in}}(\mathbf t)=\mathbf k^{\mathrm{in}},
T(\mathbf t)=T
\right)
=
P_{\mathrm{MC}}
(\mathbf t\mid\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}},T).
\]

This is an exact finite-size identity.

For small systems, compare:

- complete state probabilities;
- support probabilities;
- occupation marginals;
- network observables.

---

# 25. Required tests

## Feasibility tests

- graphical directed sequence pair;
- non-graphical sequence pair;
- out-degree sum != in-degree sum;
- negative residual degree;
- degree exceeding admissible-neighbour count;
- \(T<E\);
- B case with \(T>ME\);
- fixed pairs exceeding target degrees;
- fixed occupation exceeding B capacity.

## Initializer tests

- greedy directed constructor produces exact out- and in-degrees;
- deterministic reproducibility;
- empty support;
- complete directed graph;
- star-like out-sequence, uniform in-sequence;
- directed cycle;
- disconnected realizations where permitted.

## Switch tests

- out-degrees preserved;
- in-degrees preserved;
- support remains simple (no parallel edges);
- no forbidden self-loop;
- invalid proposal holds;
- reverse move exists;
- edge count preserved;
- state containers remain synchronized.

## Statistical tests

- uniform support enumeration;
- ME full-state enumeration;
- B full-state enumeration;
- W full-state enumeration;
- conditioned-GC identity.

## Regression tests

- fixed-\((E,T)\) outputs unchanged;
- existing benchmark command still works;
- existing fixed-pair tests still pass.

---

# 26. Diagnostics

Expose internal diagnostics at least to tests and benchmarks:

```rust
pub struct FixedKTDiagnostics {
    pub initialization_method: InitializationMethod,
    pub burn_in_attempts: u64,
    pub proposal_attempts: u64,
    pub accepted_switches: u64,
    pub self_loop_holds: u64,
    pub duplicate_edge_holds: u64,
    pub forbidden_pair_holds: u64,
    pub no_op_holds: u64,
}
```

Also record:

- observed switch acceptance rate;
- support edge turnover;
- support overlap autocorrelation;
- per-edge occupancy frequency;
- selected occupation backend.

The current fixed-\((E,T)\) backlog already notes that backend diagnostics are hidden. Phase 3 is a good point to introduce a private/internal diagnostics path that can later be surfaced consistently.

---

# 27. Benchmarks

Add a fixed-\((\mathbf k,T)\) benchmark mode.

Suggested command:

```text
python -m benchmarks micro-degree
```

or extend the existing microcanonical command with a constraint selector:

```text
python -m benchmarks micro --constraint degree-events
```

Benchmark dimensions:

- family: ME/B/W;
- node count;
- out-degree / in-degree distribution;
- support density;
- total occupation;
- self-loop policy;
- fixed pairs;
- mask presence;
- one-shot versus persistent-chain sampling.

Degree regimes:

- directed regular (same out/in per node);
- sparse heterogeneous;
- hub out-star (one node high out-degree, uniform in-degree);
- near-complete (most ordered pairs occupied);
- disconnected feasible support;
- high-degree boundary.

Record:

- initialization time;
- burn-in time;
- sampling time;
- memory;
- switch acceptance;
- effective support sample size;
- exact out-degree recovery;
- exact in-degree recovery;
- exact total recovery;
- occupation backend.

---

# 28. Documentation updates

When Phase 3 is complete:

- add fixed \((\mathbf k,T)\) to `docs/concepts/microcanonical.md`;
- update the Python API reference;
- update the model capability table;
- add examples for ME/B/W;
- document MCMC burn-in and correlation;
- distinguish exact target distribution from finite-chain approximation;
- update `docs/development/todos.md`;
- add this specification to the agent-specifications README;
- document unsupported masks or loop conventions honestly.

---

# 29. Immediate next coding step

The next commit should not attempt the complete public feature.

It should introduce the reusable **fixed-degree support kernel** and tests.

## Commit objective

```text
feat(microcanonical): add directed fixed-degree support state and double-edge switch
```

## Files to add

```text
crates/menobis-core/src/generation/microcanonical/
    fixed_kt/
        mod.rs
        state.rs
        feasibility.rs
        initializer.rs
        switch.rs
        errors.rs
```

Also move `fixed_et/support.rs` → `support/uniform_edges.rs` as part of the
module reorganisation (see section 19).

## Files to modify

```text
crates/menobis-core/src/generation/microcanonical/mod.rs
```

Potentially:

```text
crates/menobis-core/src/constraints/
```

only if a genuinely reusable degree-residual abstraction is added.

## Deliverables

1. `DirectedDegreeSequence` validation type (holds both out_degrees and
   in_degrees).
2. Loopless directed graphicality check (Fulkerson–Chen–Anstee or bipartite
   Gale–Ryser).
3. Greedy directed initializer.
4. Sparse support state with ordered pair representation.
5. Uniform two-edge proposal for directed edges.
6. Directed double-edge switch-and-hold implementation.
7. Exact out- and in-degree preservation tests.
8. Exhaustive small-state transition tests.
9. No public Python routing yet.
10. No occupation allocation yet.

This keeps the first commit narrow and reviewable.

---

# 30. Second coding step

After the support kernel is validated:

```text
feat(microcanonical): compose directed fixed-degree support with ME occupation sampler
```

Required work:

1. expose the existing positive-occupation allocator internally;
2. add `fixed_kt/core.rs`;
3. compute

   \[
   E = \sum_i k_i^{\mathrm{out}} = \sum_i k_i^{\mathrm{in}};
   \]

4. sample one degree-constrained support;
5. reuse `MeFamily`;
6. construct `SampledNetwork`;
7. validate exact \((\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}})\) and \(T\);
8. add exact small-system ME tests.

Still defer public Python routing until the Rust core is stable.

---

# 31. Third coding step

Generalize family routing:

```text
feat(microcanonical): add B and W fixed-degree-event samplers
```

Because the support sampler is family-independent, this should mainly add:

- B/W validation;
- family dispatch;
- enumeration tests;
- boundary tests.

If significant support code changes are needed at this stage, the abstraction is wrong and should be corrected before continuing.

---

# 32. Fourth coding step

Integrate repository-wide:

```text
feat(api): expose microcanonical DEGREE_EVENTS sampling
```

Include:

- Rust exports;
- Python bindings;
- public routing;
- capability registry;
- fixed-pair residualization;
- documentation;
- benchmarks;
- conditioned-GC tests.

---

# 33. Completion criteria

Phase 3 is complete when:

- fixed-degree support initialization is bounded and validated;
- the support chain preserves out- and in-degrees exactly;
- switch-and-hold is implemented correctly;
- support uniformity passes exact enumeration;
- connectivity is tested for supported domains;
- fixed-pair residual out- and in-degrees are correct;
- ME/B/W reuse the existing occupation allocators;
- sampled outputs satisfy exact \(\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}}\) and \(T\);
- sparse memory is preserved;
- no \(O(N^2)\) allocation is introduced;
- conditioned-GC identity tests pass;
- benchmarks include support-chain diagnostics;
- public capability routing is complete;
- documentation explains MCMC correlation and burn-in;
- existing fixed-\((E,T)\) behavior remains unchanged.

---

# 34. Strategic outcome

After Phase 3, MENoBiS will have two complementary microcanonical branches:

```text
fixed (E,T)
    |
    +-- direct support
    +-- direct occupation allocation

fixed (k,T)
    |
    +-- MCMC constrained support (directed)
    +-- direct occupation allocation
```

This is the correct bridge toward the later joint-state ensembles:

```text
fixed strengths
fixed (s,E)
fixed (s,k)
```

Phase 3 should establish:

- sparse support state (directed);
- move abstraction;
- switch-and-hold;
- constrained initialization (directed);
- connectivity validation;
- MCMC diagnostics.

Those components will later be reused, but Phase 3 avoids prematurely introducing event-level joint occupation dynamics.

The immediate priority is therefore:

\[
\boxed{
\text{Implement and validate the directed fixed-degree support sampler first.}
}
\]


---

# 35. Fixed-\(\mathbf k\) generator — concrete implementation design

This section refines the immediate implementation into a simple, bounded, and efficient production design.

The first supported domain is:

- **directed** (ordered pairs);
- loopless;
- simple support (no parallel directed edges);
- no arbitrary mask;
- optional fixed positive pairs after residualization;
- one sampled support per call;
- exact out- and in-degree preservation;
- uniform stationary support law;
- no claim of independent sampling after a finite number of switches.

The implementation should avoid generic abstractions until the core algorithm is stable.

---

# 36. Generator pipeline

The fixed-\(\mathbf k\) support generator should follow this exact pipeline:

```text
validate residual out- and in-degree sequences
    |
    v
choose sparse or complement representation
    |
    v
construct one feasible support deterministically
    |
    v
run bounded switch-and-hold burn-in
    |
    v
optionally run additional decorrelation sweeps
    |
    v
return current support + diagnostics
```

No unbounded retry loop is allowed.

Naive stub matching is not the default constructor.

---

# 37. Degree sequence input

Use a compact validated type:

```rust
pub struct DirectedDegreeSequence {
    out_degrees: Vec<u32>,
    in_degrees: Vec<u32>,
    edge_count: usize,
    max_out_degree: u32,
    max_in_degree: u32,
}
```

Validation should compute once:

\[
E=\sum_i k_i^{\mathrm{out}}
=
\sum_i k_i^{\mathrm{in}},
\]

\[
k_{\max}^{\mathrm{out}}=\max_i k_i^{\mathrm{out}},
\qquad
k_{\max}^{\mathrm{in}}=\max_i k_i^{\mathrm{in}}.
\]

Reject immediately if:

- any residual degree is negative before conversion;
- the out-degree sum does not equal the in-degree sum;
- any \(k_i^{\mathrm{out}}\ge N\) or \(k_i^{\mathrm{in}}\ge N\) in the loopless
  case;
- \(E=0\) but some degree is nonzero;
- \(T<E\);
- the sequence pair is not graphical.

Use the **Fulkerson–Chen–Anstee** theorem for directed graphicality
validation (or equivalently the Gale–Ryser theorem on the bipartite 0–1 matrix
and a self-loop avoidance check).  For the unmasked case, the necessary and
sufficient conditions are:

1. \(\sum_i k_i^{\mathrm{out}} = \sum_i k_i^{\mathrm{in}}\).
2. For all subsets \(S,T\subseteq\{1,\dots,N\}\),

   \[
   \sum_{i\in S} k_i^{\mathrm{out}}
   \le
   \sum_{j=1}^N \min(k_j^{\mathrm{in}}, |S|)
   - \sum_{j\notin T} \max(0, k_j^{\mathrm{in}} - |S|)
   \quad\text{(simplified form)},
   \]

   with the self-loop constraint \(k_i^{\mathrm{out}} + k_i^{\mathrm{in}} \le N\)
   (or \(N-1\) when self-loops are forbidden).

In practice, implement the equivalent bipartite check via max-flow or the
Gale–Ryser constructive test.

---

# 38. Sparse versus complement representation

For a directed loopless support, the complement out- and in-degree sequences are

\[
\bar k_i^{\mathrm{out}} = N-1 - k_i^{\mathrm{out}},
\qquad
\bar k_i^{\mathrm{in}} = N-1 - k_i^{\mathrm{in}}.
\]

If

\[
\sum_i \bar k_i^{\mathrm{out}} < \sum_i k_i^{\mathrm{out}},
\]

construct and sample the complement directed graph instead.

This changes the working edge count from

\[
E
\]

to

\[
\bar E = N(N-1) - E.
\]

The output is complemented only once at the end.

Use complement mode when it reduces working edges substantially:

```text
if complement_edge_count < edge_count:
    use complement mode
else:
    use direct mode
```

This keeps memory and switch cost proportional to

\[
O(N+\min(E,\bar E)).
\]

The public result must always use the original support convention.

---

# 39. Initial support construction

## 39.1 Default constructor

Use the greedy directed constructor (section 11.1).

Implementation outline:

1. Store all nodes in a max-heap keyed by residual out-degree.
2. Pop the node \(u\) with highest residual out-degree \(d_u^{\mathrm{out}}\).
3. Collect the \(d_u^{\mathrm{out}}\) distinct targets with the highest residual
   in-degrees, skipping \(u\) (if self-loops are forbidden) and any target \(v\)
   for which \((u,v)\) is already occupied.
4. Add a directed edge \((u\to v)\) for each selected target \(v\).
5. Decrement the residual in-degree of each selected target.
6. If \(u\) still has residual out-degree, push it back into the heap.
7. Repeat until all residual out-degrees are zero.

The resulting graph is feasible if the sequence pair passed directed
graphicality validation.

This constructor is:

- bounded;
- deterministic;
- robust for hub-dominated sequences.

Expected complexity:

\[
O(E\log N).
\]

## 39.2 Optional random initializer

A configuraion-model attempt may be added later as an optional fast
initializer for light-tailed sequences (sample out-stubs and in-stubs uniformly,
rejecting parallel edges and self-loops).

It must be:

- bounded by a small attempt count;
- disabled by default initially;
- never required for correctness.

Do not reject entire random pairings indefinitely.

---

# 40. Heavy-tail classification

Compute the diagnostics separately for out- and in-degrees:

\[
\rho_{\max}^{\mathrm{out}}
=
\frac{(k_{\max}^{\mathrm{out}})^2}{2E},
\qquad
\rho_{\max}^{\mathrm{in}}
=
\frac{(k_{\max}^{\mathrm{in}})^2}{2E},
\]

and

\[
\rho_2^{\mathrm{out}}
=
\frac{\sum_i k_i^{\mathrm{out}}(k_i^{\mathrm{out}}-1)}{2E},
\qquad
\rho_2^{\mathrm{in}}=
\frac{\sum_i k_i^{\mathrm{in}}(k_i^{\mathrm{in}}-1)}{2E}.
\]

Use the maximum of the out- and in-degree heterogeneity as the working
classifier:

\[
\rho_{\max} = \max(\rho_{\max}^{\mathrm{out}},\rho_{\max}^{\mathrm{in}}),
\qquad
\rho_2 = \max(\rho_2^{\mathrm{out}},\rho_2^{\mathrm{in}}).
\]

Suggested classification:

```text
if rho_max < 0.1:
    Light
elif rho_max < 1.0:
    Heterogeneous
else:
    HubDominated
```

The exact thresholds are heuristic and should remain internal configuration values.

For `HubDominated` sequences:

- skip configuration-model initialization;
- increase default burn-in;
- enable stranger diagnostics;
- report that generic mixing guarantees are unavailable.

The generator must still run with the greedy directed constructor.

---

# 41. Support state

Use a sparse state with three synchronized structures:

```rust
pub struct DegreeSupportState {
    node_count: usize,
    edges: Vec<(u64, u64)>,         // ordered (src, tgt) pairs
    edge_positions: HashMap<(u64, u64), usize>,
    out_adjacency: Vec<HashSet<u32>>,  // outgoing neighbours per node
}
```

Responsibilities:

## `edges`

- uniform random edge selection;
- \(O(1)\) edge access;
- compact iteration.

## `edge_positions`

- \(O(1)\) expected occupancy lookup;
- \(O(1)\) expected swap-remove update.

## `out_adjacency`

- outgoing-neighbour lookup (for quick duplicate detection during switch);
- out-degree validation;
- future Curveball-style moves.

For memory efficiency, use the repository's preferred fast hash implementation if one already exists.

Avoid storing both `edge_positions` and a second full `edge_set`; the map already provides occupancy.

Pairs are stored as-is \( (src, tgt) \) — no min/max canonicalization is needed for directed graphs.

---

# 42. Constant-time edge updates

Removing an edge should use swap-remove:

```text
idx = edge_positions.remove(edge)
last = edges.pop()

if idx < edges.len():
    edges[idx] = last
    edge_positions[last] = idx
```

Insertion should append:

```text
idx = edges.len()
edges.push(edge)
edge_positions[edge] = idx
```

Update out_adjacency simultaneously.

A valid switch removes two edges and inserts two edges in expected \(O(1)\) time.

Debug builds should periodically verify:

- edge vector and map agreement;
- out-degree sequence correctness;
- in-degree sequence correctness (can be computed from edges);
- absence of loops;
- absence of duplicate edges.

Do not run full validation after every move in release mode.

---

# 43. Directed double-edge proposal

Select two distinct edge indices uniformly:

```rust
let i = rng.random_range(0..m);
let mut j = rng.random_range(0..m - 1);
if j >= i {
    j += 1;
}
```

This avoids allocation and rejection.

Let the selected ordered edges be

\[
(a,b),\qquad(c,d).
\]

Always propose the directed reconnection:

\[
(a,d),\qquad(c,b).
\]

No random orientation flip is needed — directed edges have fixed orientation.

Proposal steps:

1. select two distinct edge indices;
2. build the two candidate ordered pairs \( (a,d)\) and \( (c,b)\);
3. validate;
4. switch or hold.

No temporary heap allocation should occur.

Use a small stack-local structure for candidate pairs.

---

# 44. Fast validity checks

Check in this order:

1. repeated endpoints that force a self-loop (\(a=d\) or \(c=b\));
2. candidate self-loops (when self-loops are forbidden);
3. candidate pairs equal to each other;
4. exact no-op (new set == old set);
5. candidate already occupied outside the two removed edges;
6. mask validity, when masks are later supported.

This ordering rejects cheap failures first.

The occupancy test must account for the fact that an existing candidate may equal one of the two old edges.

Conceptually:

```text
occupied_after_removal(candidate) =
    state.contains(candidate)
    and candidate != old_1
    and candidate != old_2
```

If either candidate is occupied after removal, hold.

Do not physically remove the old edges before all checks pass.

---

# 45. Switch-and-hold correctness

The implementation must perform exactly one proposal per MCMC step.

It must not repeatedly sample until finding a valid switch.

Correct:

```text
propose once
if invalid:
    hold
else:
    apply
```

Incorrect:

```text
repeat:
    propose
until valid
apply
```

The second procedure biases transition probabilities toward states with greater switch mobility.

The first procedure preserves proposal symmetry and yields the uniform stationary law over the connected component.

---

# 46. Burn-in defaults

Define one sweep as

\[
E_{\mathrm{work}}
\]

proposal attempts, where \(E_{\mathrm{work}}\) is the number of edges in the direct or complement working representation.

Use conservative initial defaults:

```rust
pub struct FixedDegreeMcmcConfig {
    pub burn_in_sweeps: usize,
    pub sweeps_per_sample: usize,
}
```

Suggested defaults:

```text
Light:
    burn_in_sweeps = 20
    sweeps_per_sample = 5

Heterogeneous:
    burn_in_sweeps = 50
    sweeps_per_sample = 10

HubDominated:
    burn_in_sweeps = 100
    sweeps_per_sample = 20
```

These values are implementation defaults, not theoretical guarantees.

Cap total proposal attempts using checked arithmetic.

Expose configuration overrides.

---

# 47. Persistent chain interface

For one sample, a convenience function is sufficient:

```rust
sample_fixed_degree_support(...)
```

For repeated samples, create a persistent chain:

```rust
pub struct FixedDegreeChain {
    state: DegreeSupportState,
    diagnostics: FixedDegreeDiagnostics,
    config: FixedDegreeMcmcConfig,
}
```

Methods:

```rust
impl FixedDegreeChain {
    pub fn step(&mut self, rng: &mut StdRng) -> SwitchOutcome;
    pub fn sweep(&mut self, rng: &mut StdRng);
    pub fn sample_support(&mut self, rng: &mut StdRng) -> &[(u64, u64)];
}
```

This avoids reconstructing and re-burning-in the support for every sample.

The one-shot public API can build a temporary chain internally.

---

# 48. Efficiency boundaries

The first implementation should explicitly support:

- \(N\) up to the point where \(O(N+E)\) sparse storage is practical;
- graphical loopless directed degree sequence pairs;
- heavy-tailed sequences, including hub-dominated cases;
- near-complete directed graphs through complement mode.

The first implementation should explicitly not promise:

- rapid mixing for every graphical sequence;
- arbitrary masked-degree realization;
- exact independent samples;
- uniform direct generation without MCMC;
- efficient operation when the support-state graph has extremely low mobility.

Return diagnostics rather than silently hiding these limitations.

---

# 49. Mobility and failure diagnostics

Track:

```rust
pub struct FixedDegreeDiagnostics {
    pub representation: RepresentationMode,
    pub heterogeneity: DegreeHeterogeneity,
    pub proposals: u64,
    pub accepted: u64,
    pub self_loop_holds: u64,
    pub duplicate_holds: u64,
    pub no_op_holds: u64,
}
```

Compute:

\[
\text{acceptance rate}
=
\frac{\text{accepted}}{\text{proposals}}.
\]

Also track edge turnover during burn-in:

\[
\text{turnover}
=
1-
\frac{|S_0\cap S_t|}{E}.
\]

A low acceptance rate should not abort automatically.

However, emit an internal warning when both:

- acceptance is very low;
- turnover remains very low after burn-in.

Suggested diagnostic boundary:

```text
accepatance < 0.01
and
turnover < 0.1
```

This indicates a likely low-mobility regime.

---

# 50. Exact small-system validation

For small \(N\), enumerate all simple directed supports with degree sequences
\((\mathbf k^{\mathrm{out}},\mathbf k^{\mathrm{in}})\).

Tests must verify:

1. The greedy directed initializer returns a valid realization.
2. Every directed double-edge switch preserves out-degrees.
3. Every directed double-edge switch preserves in-degrees.
4. Every valid switch has a valid reverse.
5. Invalid proposals hold.
6. Transition graph connectivity is measured.
7. Empirical support probabilities are uniform within tolerance.
8. Complement mode gives the same support law after inversion.

Use sequences including:

- **Complete directed graph**: all \(N(N-1)\) ordered pairs occupied.
- **Directed cycle**: \(k_i^{\mathrm{out}} = k_i^{\mathrm{in}} = 1\) for all \(i\).
- **Out-star**: node 0 has \(k_0^{\mathrm{out}} = N-1\), all others 0 out;
  in-degree = 1 for nodes \(1\ldots N-1\), 0 for node 0.
- **Bipartite-like**: half the nodes have high out-degree, the other half have
  high in-degree.
- **Near-complete**: \(E\) close to \(N(N-1)\), exercising complement mode.
- **Heterogeneous graphical sequence pairs**.

---

# 51. Integration with fixed-\((\mathbf k,T)\)

After support generation:

1. compute

   \[
   E=\sum_i k_i^{\mathrm{out}}
   =
   \sum_i k_i^{\mathrm{in}};
   \]

2. validate family-specific total constraints;
3. call the existing positive occupation allocator;
4. optionally shuffle occupation values before pairing with support;
5. construct `SampledNetwork`.

The support and occupation RNG streams may share one `StdRng`, but the call order should remain stable for reproducibility.

The occupation allocator must be extracted from `fixed_et/core.rs` rather than duplicated.

---

# 52. Immediate implementation checklist

The next implementation commit should include only:

- `DirectedDegreeSequence` (out_degrees + in_degrees);
- directed graphicality validation (Fulkerson–Chen–Anstee or bipartite
   Gale–Ryser);
- greedy directed construction;
- sparse `DegreeSupportState` with ordered pairs;
- complement mode for directed graphs;
- one symmetric directed double-edge switch;
- switch-and-hold;
- bounded burn-in;
- diagnostics;
- exhaustive small-state tests.

It should not yet include:

- Python bindings;
- B/W routing;
- arbitrary masks;
- Curveball moves;
- configuration-model initialization;
- advanced mobility correction.

This boundary keeps the implementation simple enough to audit and efficient enough to handle highly heterogeneous degree sequences without relying on fragile stub-matching rejection.