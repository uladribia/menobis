# MENoBiS Microcanonical Sampling Framework

**Version:** August 2026

---

# 1. Purpose

This document defines the general architecture for microcanonical sampling in MENoBiS.

The framework must support multiple:

- non-binary families;
- hard-constraint sets;
- preprocessing rules;
- sampling backends;
- move families;
- validation strategies.

The goal is not to create one independent sampler per ensemble.

The goal is to expose a common microcanonical problem abstraction and dispatch each problem to the simplest exact backend available.

The first move-based implementation governed by this framework will be:

\[
\text{ME family with fixed strengths } \mathbf s \text{ and fixed occupied-pair count } E.
\]

The already designed fixed-\((E,T)\) samplers also belong to this framework, but they use exact direct or conditional backends rather than MCMC.

---

# 2. Core design principle

A microcanonical problem should be defined independently from the algorithm used to solve it.

Conceptually,

```text
MicrocanonicalProblem
    =
Family
    +
ResidualDomain
    +
HardConstraints
    +
ConditioningConstraints
    +
ExpectedConstraints
```

The sampling backend is selected afterward.

```text
MicrocanonicalProblem
    |
    v
Backend selection
    |
    +-- exact direct sampler
    +-- exact rejection with exact fallback
    +-- exact dynamic programming
    +-- constrained hard-state MCMC
    +-- expanded-ensemble MCMC
    +-- approximate backend
```

The backend must not redefine the family law or duplicate constraint logic.

---

# 3. Scientific objects

The central state is an occupation-number graph

\[
\mathbf t=\{t_{ij}\},
\]

with integer occupations

\[
t_{ij}\in\mathbb N_0
\]

or family-specific bounded support.

An occupied pair is defined by

\[
t_{ij}>0.
\]

The implementation must use the terminology:

- occupation number;
- occupied pair;
- occupation distribution;
- occupation histogram.

Observed occupations should not be called weights.

---

# 4. Family layer

A family is defined mathematically by:

1. local support;
2. degeneracy.

For a configuration \(\mathbf t\),

\[
P(\mathbf t)\propto D_F(\mathbf t)
\]

on the hard-constraint manifold.

The supported families are:

## ME

\[
D_{\mathrm{ME}}(\mathbf t)
=
\frac{T!}{\prod_{ij}t_{ij}!}.
\]

## B

\[
D_{\mathrm B}(\mathbf t)
=
\prod_{ij}\binom{M}{t_{ij}},
\qquad
0\le t_{ij}\le M.
\]

## W

\[
D_{\mathrm W}(\mathbf t)
=
\prod_{ij}
\binom{M+t_{ij}-1}{t_{ij}}.
\]

The family layer should expose:

- local support bounds;
- local log-degeneracy;
- local degeneracy ratios;
- generating-function coefficients where available;
- family-specific occupation validation.

The family layer must not implement samplers.

---

# 5. Constraint taxonomy

The framework should distinguish four roles.

## 5.1 Always-hard constraints

These are preserved in every state visited by a backend.

Examples:

- residual strengths during event-rewiring MCMC;
- total occupation during occupation-transfer MCMC;
- directed in-strengths and out-strengths.

These constraints define the base state space.

## 5.2 Target hard constraints

These define the final microcanonical ensemble but may be temporarily softened in an expanded backend.

Examples:

- occupied-pair count \(E\);
- degree sequence \(\mathbf k\);
- looplessness;
- admissibility masks.

A strict backend preserves them at every step.

An expanded backend permits temporary violations and conditions on their exact target values.

## 5.3 Expected constraints

These remain canonical or grand-canonical.

Examples:

- expected transportation cost;
- expected distance;
- expected energy.

They enter the target through exponential factors rather than exact state rejection.

## 5.4 Search-only penalties

These exist only during feasible-state construction.

Examples:

- self-loop occupation;
- forbidden-pair occupation;
- deviation from target \(E\);
- temporary concentration penalties.

They must not alter the final target law.

---

# 6. Existing preprocessing and residual problems

All microcanonical samplers must reuse the common generation pipeline:

```text
user parameters
    |
    v
validation
    |
    v
fixed-mask and forbidden-pair prefilter
    |
    v
residual problem
    |
    v
backend
    |
    v
residual graph
    |
    v
reconstruction
    |
    v
final validation
```

The backend should not manipulate user masks directly.

The prefilter is responsible for:

- fixed zero occupations;
- fixed positive occupations;
- forbidden pairs;
- deterministic assignments;
- fixed contribution to observables;
- residual constraint calculation;
- basic feasibility checks.

For fixed strengths and fixed \(E\),

\[
s_i^{\mathrm{res}}
=
s_i^{\mathrm{target}}
-
\sum_j t_{ij}^{\mathrm{fix}},
\]

and

\[
E_{\mathrm{res}}
=
E_{\mathrm{target}}
-
E_{\mathrm{fix}}.
\]

The backend solves only the residual problem.

---

# 7. Backend hierarchy

MENoBiS should always select the simplest exact backend.

Priority order:

1. exact direct sampler;
2. exact conditional sampler;
3. exact rejection with bounded retries and exact fallback;
4. exact dynamic programming;
5. hard-state constrained MCMC;
6. expanded-ensemble MCMC;
7. approximate methods.

MCMC is a fallback for constraint sets that do not admit a practical exact direct factorization.

---

# 8. Where fixed \((E,T)\) fits

Fixed \((E,T)\) belongs to the same framework.

Its residual constraints are

\[
E(\mathbf t)=E,
\qquad
T(\mathbf t)=T.
\]

Its backend is selected by family:

## ME

- uniform support sampling;
- multinomial rejection when predicted rejection is acceptable;
- Stirling-recursion fallback.

## B

- uniform support sampling;
- binary-cell subset rejection;
- bounded weighted-composition DP fallback.

## W

- uniform support sampling;
- weak-composition rejection;
- unbounded weighted-composition DP fallback.

These samplers do not need the move layer.

The framework should therefore distinguish:

```text
sampling backends
    |
    +-- direct/conditional
    |
    +-- move-based
```

Fixed \((E,T)\) exercises the direct branch.

Fixed \((\mathbf s,E)\) will exercise the move-based branch.

---

# 9. First move-based implementation

The first constrained MCMC implementation will target:

- family: ME;
- residual strengths: fixed exactly;
- occupied-pair count: fixed exactly in the final ensemble;
- masks: respected;
- self-loops: forbidden in the final ensemble;
- graph type: undirected first;
- backend: annealed feasible-state construction plus exact production sampling.

The target law is

\[
P_{\mathrm{ME}}(\mathbf t\mid \mathbf s,E)
\propto
\frac{1}{\prod_{i<j}t_{ij}!},
\]

because

\[
T=\frac12\sum_i s_i
\]

is fixed by the strength sequence.

---

# 10. ME event-level representation

The preferred ME representation uses distinguishable event instances.

Node \(i\) contributes

\[
s_i
\]

labelled stubs.

A labelled state is a pairing of stubs.

Aggregating the paired stubs gives occupations \(t_{ij}\).

For a fixed occupation graph, the number of labelled pairings producing it is proportional to

\[
\frac{\prod_i s_i!}{\prod_{i<j}t_{ij}!}.
\]

Since \(\mathbf s\) is fixed, the numerator is constant.

Therefore uniform sampling over valid labelled event pairings induces

\[
P(\mathbf t)
\propto
\frac{1}{\prod_{i<j}t_{ij}!}.
\]

This is exactly the ME microcanonical target.

The event-level representation should therefore be treated as a family-specific auxiliary state that makes the ME degeneracy automatic.

---

# 11. State layers

The framework should support multiple synchronized state views.

## 11.1 Sparse occupation graph

```text
PairId -> OccNum
```

Used for:

- support size;
- mask checks;
- local occupation updates;
- output;
- family validation.

## 11.2 Event-instance state

For ME fixed strengths:

```text
Event {
    endpoint_a,
    endpoint_b
}
```

There are

\[
T=\frac12\sum_i s_i
\]

event instances.

Used for:

- uniform event selection;
- endpoint rewiring;
- labelled-state detailed balance.

## 11.3 Constraint cache

Maintain incrementally:

- occupied-pair count;
- loop occupation;
- forbidden-pair occupation;
- any penalty energy;
- diagnostics.

The state must remain sparse.

No \(O(N^2)\) storage is permitted.

---

# 12. Feasibility

Feasibility must be treated separately from sampling.

For an undirected loopless multigraph without masks, necessary and sufficient strength feasibility is:

\[
\sum_i s_i
\text{ is even},
\]

and

\[
\max_i s_i
\le
\frac12\sum_i s_i.
\]

With masks and fixed \(E\), feasibility becomes a constrained integer realization problem.

The implementation should use a hierarchy:

1. cheap necessary checks;
2. annealed constructive search;
3. exact feasibility fallback for small or difficult instances.

Cheap checks include:

\[
E\le T,
\]

\[
E\le L,
\]

and every positive-strength node must have at least one admissible neighbour.

The constructor must never loop indefinitely.

---

# 13. Feasible-state construction

The initializer should preserve strengths from the beginning but temporarily allow violations of:

- looplessness;
- admissibility;
- target occupied-pair count.

This avoids the pathological failure mode of trying to generate a loopless masked stub pairing directly.

## 13.1 Initial enlarged state

Create all labelled stubs and pair them arbitrarily.

This guarantees the target strengths.

The resulting state may contain:

- self-loops;
- forbidden pairs;
- the wrong support size.

These are handled by a search energy.

## 13.2 Search energy

Use

\[
H(\mathbf t)
=
\lambda_{\mathrm{loop}}
\sum_i t_{ii}
+
\lambda_{\mathrm{mask}}
\sum_{(i,j)\notin\mathcal A}t_{ij}
+
\lambda_E
\left(
E(\mathbf t)-E_\star
\right)^2.
\]

Optional search-only terms may be added, such as

\[
\lambda_C\sum_{i<j}t_{ij}^2,
\]

but every search-only coefficient must be reduced to zero before a state is accepted as feasible.

A feasible state satisfies

\[
H=0.
\]

---

# 14. Annealed initializer

Use strength-preserving event reconnections.

At inverse temperature \(\beta\), accept a proposal with

\[
\alpha
=
\min
\left(
1,
e^{-\beta\Delta H}
\right).
\]

Increase \(\beta\) according to a bounded schedule.

A staged schedule is recommended:

1. remove forbidden occupations;
2. remove self-loops;
3. approach target \(E\);
4. activate all penalties together;
5. stop only at \(H=0\).

The initializer must use:

- bounded sweeps;
- bounded restarts;
- diagnostics;
- exact fallback where available.

Failure to reach \(H=0\) means:

- the instance may be infeasible;
- the schedule may be insufficient;
- the move family may be insufficient.

It must not be silently treated as proof of infeasibility.

---

# 15. Move abstraction

A move should define:

- affected event instances;
- endpoint reconnection;
- preserved always-hard constraints;
- reverse operation;
- local occupation delta;
- proposal probability;
- validity checks.

The move should not contain family acceptance logic unless it is intrinsically family-specific.

The first move set should include:

- two-event switches;
- three-event cycles.

Larger moves can be added after connectivity analysis.

---

# 16. Two-event switch

Select two distinct event instances:

\[
(a,b),
\qquad
(c,d).
\]

Randomly orient the undirected event pairs.

Choose one cross-reconnection:

\[
(a,c),(b,d),
\]

or

\[
(a,d),(b,c).
\]

Every endpoint appears once before and once after.

Therefore all strengths remain fixed.

The local occupation update is computed through a small map over affected pairs.

---

# 17. Three-event cycle

Select three event instances:

\[
(a,b),
\qquad
(c,d),
\qquad
(e,f).
\]

Apply a symmetric cyclic reconnection, for example:

\[
(a,d),
\qquad
(c,f),
\qquad
(e,b).
\]

Strengths remain fixed.

The three-event kernel is included to:

- improve connectivity;
- cross barriers not reachable by two-event switches;
- assist annealed construction;
- reduce support isolation.

The exact set of symmetric cycle orientations must be enumerated explicitly in the implementation.

---

# 18. Local support update

For every candidate move:

1. collect all distinct old and new pair identifiers;
2. accumulate occupation changes \(\Delta t_e\);
3. verify

   \[
   t_e+\Delta t_e\ge0;
   \]

4. compute

   \[
   \Delta E
   =
   \sum_e
   \left[
   \mathbf 1(t_e+\Delta t_e>0)
   -
   \mathbf 1(t_e>0)
   \right].
   \]

This local map must handle:

- repeated old pairs;
- repeated new pairs;
- old and new pair coincidences;
- simultaneous increments and decrements;
- no-op proposals.

Do not implement \(\Delta E\) as a brittle list of special cases.

---

# 19. Production sampling modes

The framework should support two exact production modes.

## 19.1 Hard-state MCMC

Every visited state satisfies:

- fixed strengths;
- no self-loops;
- mask validity;
- fixed \(E\).

Invalid proposals are handled by switch-and-hold.

At the labelled ME state level, valid symmetric event reconnections are accepted with probability one.

This is the preferred production backend when the valid-state move graph is connected and mixes adequately.

## 19.2 Expanded-ensemble MCMC

The chain always preserves strengths but may temporarily violate:

- loops;
- masks;
- target \(E\).

Define

\[
\Pi_\beta(\mathbf t)
\propto
\frac{1}{\prod_{ij}t_{ij}!}
e^{-\beta H(\mathbf t)}.
\]

At the labelled event level, this becomes a penalty-weighted uniform pairing law.

Samples are emitted only when

\[
H(\mathbf t)=0.
\]

Conditioning gives

\[
\Pi_\beta(\mathbf t\mid H=0)
\propto
\frac{1}{\prod_{ij}t_{ij}!},
\]

on the desired fixed-\((\mathbf s,E)\) manifold.

This mode can connect hard-state components through temporary violations.

Its efficiency depends on the frequency of visits to \(H=0\).

---

# 20. Annealing versus equilibrium tempering

A one-time annealed initializer does not prove production-chain ergodicity.

These must remain distinct:

## Initialization annealing

Purpose:

- find one feasible state.

The schedule is nonstationary.

Its output is not a sample.

## Equilibrium expanded ensemble

Purpose:

- sample across hard-state components.

The chain must have a stationary target.

Suitable methods include:

- fixed-\(\beta\) soft constraints with exact conditioning;
- simulated tempering;
- parallel tempering;
- umbrella sampling in \(E\);
- multicanonical penalties.

Only the equilibrium methods can address production-state isolation rigorously.

---

# 21. Backend selection for fixed \((\mathbf s,E)\)

A practical first implementation should use:

```text
attempt annealed feasible-state construction
    |
    v
run hard-state MCMC
    |
    v
validate connectivity and mixing on small systems
```

If the hard-state graph is disconnected or mixing is poor:

```text
switch to expanded-ensemble production
    |
    v
emit only H = 0 states
```

The backend decision should eventually be represented explicitly:

```rust
enum FixedSEBackend {
    HardState,
    ExpandedEnsemble,
}
```

The first release may expose hard-state MCMC as experimental until connectivity evidence is sufficient.

---

# 22. Acceptance rules

## 22.1 Hard-state ME event moves

If the proposal:

- preserves strengths;
- respects masks;
- creates no forbidden self-loop;
- preserves \(E\);

then accept with probability

\[
1.
\]

Otherwise hold.

## 22.2 Annealed or expanded moves

For a symmetric event-level proposal,

\[
\alpha
=
\min
\left(
1,
e^{-\beta\Delta H}
\right).
\]

If asymmetric move selection is later introduced, include the Hastings correction:

\[
\alpha
=
\min
\left(
1,
e^{-\beta\Delta H}
\frac{q(x\mid x')}{q(x'\mid x)}
\right).
\]

All proposal asymmetries must be explicit.

---

# 23. Directed extension

For directed networks, represent:

- out-stubs;
- in-stubs.

A directed event connects one out-stub to one in-stub.

Select two events:

\[
a\to b,
\qquad
c\to d.
\]

Swap destinations:

\[
a\to d,
\qquad
c\to b.
\]

This preserves:

- all out-strengths;
- all in-strengths.

The same framework applies to:

- masks;
- loop penalties;
- fixed \(E\);
- hard-state and expanded backends.

The undirected implementation should be completed first.

---

# 24. Validation framework

Every move-based backend must satisfy:

1. exact hard constraints on emitted states;
2. exact mask correctness;
3. detailed balance;
4. move reversibility;
5. small-state connectivity analysis;
6. exact enumeration agreement;
7. conditioned grand-canonical agreement;
8. repeated-chain agreement;
9. benchmark integration.

---

# 25. Exact enumeration

For small residual systems, enumerate

\[
\Omega(\mathbf s,E).
\]

The ME state weight is

\[
w(\mathbf t)
=
\frac{1}{\prod_{i<j}t_{ij}!}.
\]

Normalize exactly:

\[
P(\mathbf t)
=
\frac{
w(\mathbf t)
}{
\sum_{\mathbf u\in\Omega(\mathbf s,E)}
w(\mathbf u)
}.
\]

Compare empirical frequencies from:

- hard-state MCMC;
- expanded-ensemble conditioning;
- multiple initial states.

---

# 26. Connectivity validation

For small systems:

1. enumerate every feasible state;
2. apply every implemented hard-valid move;
3. construct the transition graph;
4. compute connected components.

Perform this separately for:

- two-event moves;
- two-event plus three-event moves;
- any larger move family.

If the graph is disconnected, the hard-state backend is not universal.

That result must drive backend design rather than being ignored.

---

# 27. Conditioned grand-canonical validation

Generate from the ME grand-canonical model.

Retain states satisfying:

\[
\mathbf s(\mathbf t)=\mathbf s,
\]

and

\[
E(\mathbf t)=E.
\]

Then verify

\[
P_{\mathrm{GC}}
\left(
\mathbf t
\mid
\mathbf s(\mathbf t)=\mathbf s,\,
E(\mathbf t)=E
\right)
=
P_{\mathrm{MC}}
(\mathbf t\mid\mathbf s,E).
\]

This is an exact finite-size identity.

It is practical only for small systems but is one of the strongest validation tests.

---

# 28. Diagnostics

Record:

- initialization sweeps;
- initialization restarts;
- final initialization energy;
- two-event proposals;
- three-event proposals;
- accepted moves;
- mask holds;
- loop holds;
- fixed-\(E\) holds;
- no-op holds;
- support-changing moves;
- support-preserving moves;
- visits to \(H=0\);
- occupation autocorrelation;
- support autocorrelation;
- edge turnover;
- backend mode.

A low acceptance rate is not by itself a correctness failure.

The important question is whether the chain explores both occupation and support space.

---

# 29. Sampling cadence

Define one sweep as approximately

\[
T
\]

event-level proposal attempts.

Expose:

- burn-in sweeps;
- sweeps per emitted sample;
- number of samples;
- move-mixture probabilities;
- backend choice;
- annealing schedule;
- tempering configuration where applicable.

Defaults must be documented as heuristics, not universal guarantees.

---

# 30. Performance principles

Memory must remain

\[
O(T+E+N)
\]

or similarly sparse.

Do not allocate dense \(N\times N\) arrays.

Local move evaluation should be constant-time or logarithmic in sparse data structures.

The principal performance costs are expected to be:

- event-instance selection;
- sparse pair lookup;
- local occupation updates;
- penalty updates;
- connectivity-enhancing move proposals.

Benchmark both:

- wall-clock speed;
- effective sample size.

Raw proposal throughput is not sufficient.

---

# 31. Proposed repository layout

```text
generation/
    microcanonical/
        problem/
            mod.rs
            residual.rs
            constraints.rs
            diagnostics.rs

        direct/
            fixed_et/
                me/
                b/
                w/

        mcmc/
            state/
                occupation.rs
                event.rs
                caches.rs

            moves/
                two_event.rs
                three_event.rs
                cycle.rs

            penalties/
                loops.rs
                mask.rs
                occupied_pairs.rs

            annealing/
                schedule.rs
                initializer.rs

            expanded/
                fixed_beta.rs
                tempering.rs

            fixed_se/
                me/
                    sampler.rs
                    feasibility.rs
                    config.rs
                    validation.rs
```

Exact naming should follow existing MENoBiS conventions.

---

# 32. Implementation phases

## Phase A — Framework extraction

- define residual microcanonical problem types;
- define backend dispatch;
- formalize family and constraint interfaces;
- preserve existing fixed-\((E,T)\) implementations.

## Phase B — ME event state

- implement labelled event representation;
- synchronize sparse occupation map;
- validate aggregation and updates.

## Phase C — Move layer

- implement two-event switches;
- implement three-event cycles;
- implement local delta maps;
- test reversibility.

## Phase D — Annealed initializer

- implement penalties;
- implement bounded schedules;
- implement restart policy;
- add feasibility diagnostics.

## Phase E — Hard-state fixed-\((\mathbf s,E)\)

- enforce loops, masks, and \(E\);
- implement switch-and-hold;
- validate detailed balance;
- test exact enumeration.

## Phase F — Connectivity analysis

- enumerate small state graphs;
- identify disconnected cases;
- decide whether larger moves suffice.

## Phase G — Expanded ensemble

- implement soft-constraint equilibrium backend;
- emit only \(H=0\) states;
- validate conditioned exactness.

## Phase H — Benchmarks and API

- benchmark initialization;
- benchmark effective sample size;
- expose diagnostics;
- document experimental guarantees.

---

# 33. First implementation recommendation

The first production-capable fixed-\((\mathbf s,E)\) implementation should include:

1. residual prefilter;
2. cheap feasibility checks;
3. arbitrary strength-preserving labelled pairing;
4. annealed feasible-state constructor;
5. two-event and three-event moves;
6. hard-state switch-and-hold sampler;
7. small-system connectivity tests;
8. exact enumeration validation;
9. conditioned-GC validation;
10. expanded-ensemble fallback behind an experimental flag.

This is the smallest implementation that is scientifically defensible without pretending that two-event hard-state switching is universally ergodic.

---

# 34. Completion criteria

The general framework is ready for the first fixed-\((\mathbf s,E)\) release when:

- fixed-\((E,T)\) direct samplers are represented as framework backends;
- residual preprocessing is shared;
- family degeneracies are not duplicated;
- ME labelled-event states reproduce factorial degeneracy;
- annealed initialization is bounded;
- initialization failure is reported honestly;
- hard-state detailed balance is verified;
- small-system connectivity is measured;
- emitted states satisfy exact strengths and exact \(E\);
- masks and loop rules are exact on emitted states;
- exact enumeration tests pass;
- conditioned-GC tests pass;
- expanded conditioning is available when hard-state connectivity fails;
- sparse memory requirements are preserved;
- diagnostics and benchmark results are documented.

---

# 35. Long-term reuse

Once implemented, the framework should support later cases by replacing family acceptance logic and constraint definitions.

Examples:

## B or W with fixed \((\mathbf s,E)\)

Reuse:

- residual strengths;
- move layer;
- penalties;
- annealing;
- expanded ensemble.

Replace:

- family target ratios;
- local support checks;
- acceptance probabilities.

## Fixed \((\mathbf s,\mathbf k)\)

Reuse the same framework, with degree-sequence deviation added as a target constraint.

## Fixed strengths plus expected cost

Keep strengths hard and add

\[
e^{-\theta C(\mathbf t)}
\]

to the equilibrium target.

## Fixed \((\mathbf k,T)\)

Use a different always-hard move invariant, but reuse the backend architecture.

The framework should therefore be implemented once and specialized, rather than rebuilt per phase.
