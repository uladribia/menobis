# Phase 2 — Exact Microcanonical W Sampler with Fixed \((E,T)\)

**Version:** August 2026

---

# 1. Scope

This document specifies the implementation of the **W-family microcanonical ensemble with fixed occupied-pair count \(E\) and fixed total occupation \(T\)** in MENoBiS.

It builds directly on the common fixed-\((E,T)\) architecture established for the ME implementation and reused by the B implementation.

The objectives are:

- preserve the fixed-mask prefilter and residual-problem architecture;
- reuse uniform support sampling unchanged;
- implement the W-family positive-occupation law exactly;
- use a fast bounded rejection sampler whenever its predicted rejection is acceptable;
- use an exact dynamic-programming sampler as the guaranteed fallback;
- select between the two backends using the rejection threshold

\[
r=0.8;
\]

- validate the implementation through exact enumeration, backend agreement, and the conditioned grand-canonical identity.

This is not a separate generation architecture. It is a W-specific occupation allocator inside the common microcanonical fixed-\((E,T)\) framework.

---

# 2. Scientific definition

The W family represents indistinguishable events distributed over \(M\) internal layers or states per network pair.

For each admissible pair \((i,j)\), the occupation number satisfies

\[
t_{ij}\in\mathbb N_0.
\]

Unlike the B family, there is no upper bound on \(t_{ij}\).

The local W degeneracy is

\[
d_W(t)
=
\binom{M+t-1}{t}.
\]

The global degeneracy is

\[
D_W(\mathbf t)
=
\prod_{ij}
\binom{M+t_{ij}-1}{t_{ij}}.
\]

Under fixed occupied-pair count \(E\) and fixed total occupation \(T\), the target distribution is

\[
P_W(\mathbf t\mid E,T,M)
=
\frac{1}{Z_W(E,T,M)}
\prod_{ij}
\binom{M+t_{ij}-1}{t_{ij}},
\]

over configurations satisfying

\[
\sum_{ij}\mathbf 1(t_{ij}>0)=E,
\]

\[
\sum_{ij}t_{ij}=T,
\]

and

\[
t_{ij}\ge0.
\]

The implementation must sample this finite conditional distribution exactly, apart from ordinary floating-point rounding in branch probabilities.

---

# 3. Relationship to the common fixed-\((E,T)\) framework

The W implementation uses the same generation pipeline as ME and B:

```text
user constraints
    |
    v
validation
    |
    v
fixed-mask and forbidden-pair prefilter
    |
    v
residual fixed-(E,T) problem
    |
    v
uniform support sampling
    |
    v
W-specific positive occupation allocation
    |
    v
sparse residual graph construction
    |
    v
reconstruction with fixed occupations
    |
    v
final validation
```

The following components should be reused unchanged:

- fixed-mask preprocessing;
- forbidden-pair handling;
- residual-problem construction;
- admissible-pair indexing;
- uniform support selection;
- sparse graph construction;
- reconstruction;
- random-number-generator handling;
- hard-constraint validation;
- backend diagnostics;
- benchmark integration;
- conditioned grand-canonical validation infrastructure.

The new scientific component is the W positive-occupation allocator.

---

# 4. Residual problem

The W sampler never receives the original user problem directly.

The prefilter removes or resolves:

- forbidden pairs;
- fixed zero occupations;
- fixed positive occupations;
- deterministic assignments;
- unavailable pair locations;
- all mask-specific logic.

The residual sampler receives:

- \(L\): number of residual admissible pairs;
- \(E\): residual occupied-pair count;
- \(T\): residual total occupation;
- \(M\): W layer multiplicity;
- a residual admissible-pair collection.

If fixed positive occupations contribute

\[
E_{\mathrm{fix}}
=
\sum_{ij}\mathbf 1(t_{ij}^{\mathrm{fix}}>0),
\]

and

\[
T_{\mathrm{fix}}
=
\sum_{ij}t_{ij}^{\mathrm{fix}},
\]

then preprocessing computes

\[
E_{\mathrm{res}}
=
E_{\mathrm{user}}-E_{\mathrm{fix}},
\]

\[
T_{\mathrm{res}}
=
T_{\mathrm{user}}-T_{\mathrm{fix}}.
\]

The W sampler solves only the residual problem

\[
(E_{\mathrm{res}},T_{\mathrm{res}}).
\]

The fixed occupations are restored during reconstruction.

The sampler must not contain direct mask logic.

---

# 5. Feasibility

For the residual problem,

\[
0\le E\le L.
\]

If \(E=0\), then necessarily

\[
T=0.
\]

If \(E>0\), every occupied pair must contain at least one event, so

\[
T\ge E.
\]

There is no W upper bound analogous to

\[
T\le ME
\]

in the B family.

Therefore the complete feasibility conditions are:

```text
0 <= E <= L

if E == 0:
    T must equal 0

if E > 0:
    T >= E

M must be positive
```

Special deterministic cases are:

- \(E=0,T=0\): empty residual graph;
- \(E=1\): the selected pair receives occupation \(T\);
- \(T=E\): every selected pair receives occupation \(1\);
- \(M=1\): the W degeneracy is one for every occupation, reducing the allocator to a uniform positive composition of \(T\) into \(E\) parts.

All arithmetic must use checked integer operations.

---

# 6. Uniform support factorization

Let \(S\) be a support containing exactly \(E\) residual admissible pair locations.

For a fixed support, write the positive occupations as

\[
(t_1,\ldots,t_E),
\]

with

\[
t_i\ge1,
\qquad
\sum_{i=1}^{E}t_i=T.
\]

The support-specific degeneracy is

\[
D_W(t_1,\ldots,t_E)
=
\prod_{i=1}^{E}
\binom{M+t_i-1}{t_i}.
\]

The corresponding positive partition function is

\[
Z_W^+(E,T,M)
=
\sum_{\substack{t_i\ge1\\\sum_i t_i=T}}
\prod_{i=1}^{E}
\binom{M+t_i-1}{t_i}.
\]

This quantity depends only on \(E\), \(T\), and \(M\), not on the identities of the selected pair locations.

Therefore every support of size \(E\) has the same total weight:

\[
P(S\mid L,E,T,M)
=
\frac{1}{\binom{L}{E}}.
\]

Thus the W fixed-\((E,T)\) sampler factorizes exactly into:

1. uniform support sampling;
2. W-specific positive occupation allocation.

The common support sampler must be reused unchanged.

---

# 7. Positive occupation law

Conditioned on the selected support, the W occupation vector satisfies

\[
t_i\ge1,
\]

\[
\sum_{i=1}^{E}t_i=T,
\]

with probability

\[
P_W(t_1,\ldots,t_E\mid E,T,M)
=
\frac{1}{Z_W^+(E,T,M)}
\prod_{i=1}^{E}
\binom{M+t_i-1}{t_i}.
\]

This is an unbounded weighted positive-composition problem.

The occupation allocator should remain independent of graph structure.

---

# 8. Microscopic layer-box interpretation

The W occupation law has a direct microscopic construction.

For each selected network pair, introduce \(M\) internal layer boxes.

Across \(E\) selected pairs, there are

\[
K=ME
\]

layer boxes.

Distribute \(T\) indistinguishable events among these \(K\) boxes.

A microscopic state is therefore a weak composition

\[
(x_1,\ldots,x_K),
\]

with

\[
x_a\ge0,
\qquad
\sum_{a=1}^{K}x_a=T.
\]

Assume every weak composition is equally likely.

Group the \(K\) boxes into \(E\) consecutive groups of size \(M\), one group per selected network pair.

For pair \(i\), define

\[
t_i
=
\sum_{m=1}^{M}x_{i,m}.
\]

For a fixed pair occupation \(t_i\), the number of weak compositions into its \(M\) layer boxes is

\[
\binom{M+t_i-1}{t_i}.
\]

Therefore the number of microscopic states producing a macro occupation vector

\[
(t_1,\ldots,t_E)
\]

is

\[
\prod_{i=1}^{E}
\binom{M+t_i-1}{t_i}.
\]

Since all microscopic weak compositions are uniform,

\[
P(t_1,\ldots,t_E)
=
\frac{
\prod_i\binom{M+t_i-1}{t_i}
}{
\binom{ME+T-1}{T}
}.
\]

Conditioning on every pair being occupied,

\[
t_i>0
\qquad
\forall i,
\]

gives exactly

\[
P(t_1,\ldots,t_E\mid t_i>0\;\forall i)
\propto
\prod_i\binom{M+t_i-1}{t_i}.
\]

This is the exact W microcanonical positive-occupation law.

---

# 9. Uniform weak-composition sampling

The fast W proposal requires drawing a uniform weak composition of \(T\) into

\[
K=ME
\]

parts.

The number of such weak compositions is

\[
\binom{T+K-1}{K-1}
=
\binom{T+K-1}{T}.
\]

Use the standard stars-and-bars representation.

Represent the composition by:

- \(T\) stars;
- \(K-1\) separators.

There are

\[
T+K-1
\]

positions.

Choose the \(K-1\) separator positions uniformly without replacement.

The counts of stars before, between, and after separators give the weak composition.

For sorted separator positions

\[
0\le b_1<b_2<\cdots<b_{K-1}<T+K-1,
\]

the components are

\[
x_1=b_1,
\]

\[
x_j=b_j-b_{j-1}-1
\qquad
(2\le j\le K-1),
\]

and

\[
x_K=T+K-2-b_{K-1}.
\]

Equivalent indexing conventions are acceptable, but must be tested carefully.

A production implementation does not need to materialize all \(K\) microscopic counts if it can aggregate stars directly into the \(E\) groups.

---

# 10. Hybrid backend strategy

The W occupation allocator uses two exact backends:

1. **bounded weak-composition rejection**, used as the fast path;
2. **dynamic-programming sequential sampling**, used as the guaranteed fallback.

Backend selection uses the rejection threshold

\[
r=0.8.
\]

The policy is:

```text
if estimated_rejection <= 0.8:
    try bounded weak-composition rejection
else:
    use direct DP sampling
```

If bounded rejection exhausts its retry limit, switch to the DP sampler.

Retry exhaustion is not a user-visible sampling error.

Both backends produce the same exact target law.

---

# 11. Fast weak-composition rejection backend

## 11.1 Algorithm

Given \(E\), \(T\), and \(M\):

1. Compute

   \[
   K=ME.
   \]

2. Draw a uniform weak composition of \(T\) into \(K\) layer-box counts.

3. Aggregate each consecutive group of \(M\) counts:

   \[
   t_i
   =
   \sum_{m=1}^{M}x_{i,m}.
   \]

4. Accept if

   \[
   t_i>0
   \qquad
   \forall i.
   \]

5. Otherwise retry.

Pseudocode:

```text
try_w_weak_composition_rejection(E, T, M, max_attempts, rng):

    K = checked_mul(E, M)
    occupations = [0; E]

    for attempt in 1..=max_attempts:

        reset occupations to zero

        micro_counts =
            sample_uniform_weak_composition(T, K, rng)

        for box in 0..K:
            pair = box / M
            occupations[pair] += micro_counts[box]

        if every occupation > 0:
            return Success(occupations, attempt)

    return Exhausted
```

A more memory-efficient implementation should aggregate directly during stars-and-bars decoding.

---

## 11.2 Correctness

Every weak composition of \(T\) into \(ME\) microscopic layer boxes is selected with probability

\[
\frac{1}{
\binom{ME+T-1}{T}
}.
\]

For a fixed macro occupation vector \(\mathbf t\), the number of microscopic weak compositions that map to it is

\[
\prod_i
\binom{M+t_i-1}{t_i}.
\]

Therefore

\[
P(\mathbf t)
=
\frac{
\prod_i\binom{M+t_i-1}{t_i}
}{
\binom{ME+T-1}{T}
}.
\]

Conditioning on every macro pair being positive gives the target W law exactly.

---

# 12. Rejection probability

Let

\[
A_W(E,T,M)
\]

be the number of weak compositions of \(T\) into \(ME\) microscopic boxes such that every one of the \(E\) groups has positive total occupation.

Then

\[
p_{\mathrm{acc}}^W(E,T,M)
=
\frac{
A_W(E,T,M)
}{
\binom{ME+T-1}{T}
},
\]

and

\[
p_{\mathrm{rej}}^W(E,T,M)
=
1-p_{\mathrm{acc}}^W(E,T,M).
\]

By inclusion-exclusion over empty groups,

\[
A_W(E,T,M)
=
\sum_{j=0}^{E}
(-1)^j
\binom{E}{j}
\binom{M(E-j)+T-1}{T}.
\]

Terms with zero remaining microscopic boxes are interpreted carefully:

- if \(E-j=0\) and \(T>0\), the term is zero;
- if \(E-j=0\) and \(T=0\), the term is one.

The same quantity is the coefficient

\[
A_W(E,T,M)
=
[x^T]
\left(
(1-x)^{-M}-1
\right)^E.
\]

The exact inclusion-exclusion expression is useful for tests and diagnostics, but it is not the preferred production backend selector.

---

# 13. Cheap rejection estimate

For backend selection, estimate the probability that a specified group is empty.

A particular pair-group is empty when all \(T\) indistinguishable events are distributed among the remaining

\[
M(E-1)
\]

microscopic boxes.

Therefore

\[
q_0
=
P(\text{one specified pair is empty})
=
\frac{
\binom{M(E-1)+T-1}{T}
}{
\binom{ME+T-1}{T}
}.
\]

Approximate the all-positive probability by

\[
\widehat p_{\mathrm{acc}}
=
(1-q_0)^E.
\]

Then

\[
\widehat p_{\mathrm{rej}}
=
1-\widehat p_{\mathrm{acc}}.
\]

The empty-group events are dependent. This estimate is used only for backend selection.

An inaccurate estimate cannot bias the output because:

- the rejection sampler remains exact conditional on acceptance;
- the DP sampler remains exact;
- retries are bounded;
- exhaustion triggers the DP fallback.

Compute \(q_0\) in log space:

\[
\log q_0
=
\log\binom{M(E-1)+T-1}{T}
-
\log\binom{ME+T-1}{T}.
\]

For stable evaluation:

```text
log_q0 =
    log_binomial(M * (E - 1) + T - 1, T)
    - log_binomial(M * E + T - 1, T)

q0 = exp(log_q0)

log_p_acc_est =
    E * log1p(-q0)

p_rej_est =
    -expm1(log_p_acc_est)
```

All additions and multiplications in combinatorial arguments must use checked integer arithmetic.

---

# 14. Backend-selection rule

The W implementation uses

\[
r=0.8.
\]

Therefore:

```text
if estimated_rejection <= 0.8:
    use bounded weak-composition rejection first
else:
    use direct DP sampling
```

Equivalently, rejection is attempted when

\[
\widehat p_{\mathrm{acc}}\ge0.2.
\]

The threshold must be a named constant or configuration value:

```rust
const DEFAULT_REJECTION_THRESHOLD: f64 = 0.8;
```

---

# 15. Retry bound

Use a bounded number of rejection attempts.

A suitable initial default is:

```rust
const DEFAULT_MAX_REJECTION_ATTEMPTS: usize = 20;
```

The logic is:

```text
attempt rejection at most 20 times

if accepted:
    return the sample

if exhausted:
    build or retrieve the DP table
    return a DP sample
```

Retry exhaustion does not imply invalid parameters.

It is a normal fallback condition.

---

# 16. Memory-efficient weak-composition generation

A naive stars-and-bars implementation materializes:

- \(K-1\) separator positions;
- \(K\) microscopic counts.

This is acceptable for a reference implementation, but can be improved.

Only the \(E\) macro occupations are needed.

After sorting separators, decode the weak composition sequentially and add each microscopic component directly to its group:

```text
pair = micro_box_index / M
occupations[pair] += component
```

This avoids storing the full microscopic vector.

The required temporary memory becomes:

\[
O(K)
\]

for separator positions plus

\[
O(E)
\]

for macro occupations.

If the subset sampler returns sorted separator positions directly, no extra sort is needed.

---

# 17. Separator-side optimization

The stars-and-bars representation chooses

\[
K-1
\]

separator positions from

\[
T+K-1
\]

positions.

When \(K-1\) is large relative to \(T\), it is cheaper to choose the complementary set of \(T\) star positions.

Thus the uniform subset routine should sample

\[
\min(K-1,T)
\]

positions.

Two decoding modes are possible:

## Separator mode

Use when

\[
K-1\le T.
\]

Sample separator positions and derive component lengths.

## Star mode

Use when

\[
T<K-1.
\]

Sample star positions and count how many stars fall into each interval between separators.

A simpler Phase 2 implementation may always sample separators, but the production version should use the smaller side to avoid unnecessary memory and work.

The optimization must preserve exact uniformity over stars-and-bars strings.

---

# 18. Exact direct fallback

The guaranteed fallback is a weighted positive-composition dynamic program.

Define

\[
Z_W(k,s)
\]

as the total W degeneracy weight of assigning total occupation \(s\) to \(k\) occupied pairs:

\[
Z_W(k,s)
=
\sum_{\substack{t_i\ge1\\\sum_i t_i=s}}
\prod_{i=1}^{k}
\binom{M+t_i-1}{t_i}.
\]

The base cases are

\[
Z_W(0,0)=1,
\]

and

\[
Z_W(0,s)=0
\qquad
\text{for }s\ne0.
\]

For \(k>0\),

\[
Z_W(k,s)
=
\sum_{t=1}^{s}
\binom{M+t-1}{t}
Z_W(k-1,s-t).
\]

The remaining \(k-1\) occupations must each be at least one, so

\[
s-t\ge k-1.
\]

Therefore

\[
1\le t\le s-(k-1).
\]

The recurrence is

\[
Z_W(k,s)
=
\sum_{t=1}^{s-k+1}
\binom{M+t-1}{t}
Z_W(k-1,s-t).
\]

Only states satisfying

\[
s\ge k
\]

are feasible.

---

# 19. Generating-function interpretation

The positive single-pair generating function is

\[
g_W(x)
=
\sum_{t=1}^{\infty}
\binom{M+t-1}{t}x^t.
\]

Using the negative-binomial generating function,

\[
\sum_{t=0}^{\infty}
\binom{M+t-1}{t}x^t
=
(1-x)^{-M},
\]

so

\[
g_W(x)
=
(1-x)^{-M}-1.
\]

For \(E\) occupied pairs,

\[
g_W(x)^E
=
\left(
(1-x)^{-M}-1
\right)^E.
\]

Therefore

\[
Z_W(E,T)
=
[x^T]
\left(
(1-x)^{-M}-1
\right)^E.
\]

This is the same coefficient that counts microscopic weak compositions with every group positive.

The DP is therefore a coefficient recursion derived from the family generating function.

---

# 20. Log-space DP

The values \(Z_W(k,s)\) grow rapidly.

Production code should store

\[
\log Z_W(k,s).
\]

The recurrence becomes

\[
\log Z_W(k,s)
=
\operatorname{logsumexp}_{t=1}^{s-k+1}
\left[
\log\binom{M+t-1}{t}
+
\log Z_W(k-1,s-t)
\right].
\]

Impossible states are represented by

\[
-\infty.
\]

Boundary conditions are

\[
\log Z_W(0,0)=0,
\]

\[
\log Z_W(0,s)=-\infty
\quad
(s\ne0).
\]

Precompute

\[
\log d_W(t)
=
\log\binom{M+t-1}{t}
\]

for

\[
0\le t\le T.
\]

Prefer reusing the W family mathematical layer instead of duplicating the formula inside the sampler.

---

# 21. Sequential DP sampling

After constructing the table, sample occupations sequentially.

At state \((k,s)\), the probability that the next pair receives occupation \(t\) is

\[
P(t\mid k,s)
=
\frac{
\binom{M+t-1}{t}
Z_W(k-1,s-t)
}{
Z_W(k,s)
},
\]

for

\[
1\le t\le s-k+1.
\]

The log branch weight is

\[
\ell_t
=
\log\binom{M+t-1}{t}
+
\log Z_W(k-1,s-t).
\]

Pseudocode:

```text
sample_w_occupations_dp(E, T, M, rng):

    log_Z =
        build_w_log_partition_table(E, T, M)

    occupations =
        Vec::with_capacity(E)

    k = E
    remaining = T

    while k > 0:

        t_min = 1
        t_max = remaining - (k - 1)

        branches = empty

        for t in t_min..=t_max:

            log_weight =
                log_w_degeneracy[t]
                + log_Z[k - 1][remaining - t]

            branches.push((t, log_weight))

        selected_t =
            sample_log_categorical(branches, rng)

        occupations.push(selected_t)

        remaining -= selected_t
        k -= 1

    assert remaining == 0

    return occupations
```

A final shuffle is not mathematically necessary if the DP conditionals are correct, but may be retained for consistency or defensive exchangeability testing.

---

# 22. DP storage

The conceptual table has indices

\[
0\le k\le E,
\qquad
0\le s\le T.
\]

Only states satisfying

\[
s\ge k
\]

are feasible.

Recommended storage:

```text
row k stores s from k to T
```

This is triangular.

The table must remain available during sequential backward sampling.

Rolling rows alone are insufficient unless the implementation recomputes or checkpoints earlier rows.

For the first implementation, store the full triangular log table.

---

# 23. Complexity

## 23.1 Rejection path

A stars-and-bars proposal requires choosing a subset of size

\[
\min(K-1,T),
\qquad
K=ME,
\]

from

\[
T+K-1
\]

positions, followed by decoding and aggregation.

A straightforward cost is

\[
O(E+M E+\min(T,ME-1))
\]

or more simply

\[
O(ME+T)
\]

per attempt.

With direct aggregation and smaller-side subset sampling, the practical cost can be reduced.

The expected total cost is divided by the acceptance probability:

\[
O\left(
\frac{\text{proposal cost}}
{p_{\mathrm{acc}}^W}
\right).
\]

Because rejection is selected only when estimated rejection is at most \(0.8\), the estimated attempt count is at most five.

## 23.2 DP fallback

The naive recurrence evaluates up to \(O(T)\) branches per state.

There are \(O(ET)\) feasible states.

Worst-case time is therefore

\[
O(ET^2).
\]

Memory is

\[
O(ET).
\]

Sequential sampling costs

\[
O(ET)
\]

in the worst case if every step scans a large range.

This is acceptable as a correctness-oriented fallback for moderate \(E,T\), but it is more expensive than the B fallback.

Later optimization possibilities include:

- convolution acceleration;
- recurrence relations specific to negative-binomial coefficients;
- alias tables for repeated states;
- saddle-point proposals;
- recursive coefficient splitting;
- cached partition tables;
- FFT-based coefficient construction for large parameter grids.

These optimizations are outside the initial implementation.

---

# 24. Full end-to-end algorithm

```text
sample_w_fixed_et(residual_problem, M, rng):

    L = residual_problem.admissible_pair_count
    E = residual_problem.residual_edges
    T = residual_problem.residual_total

    validate:
        0 <= E <= L
        if E == 0 then T == 0
        if E > 0 then T >= E
        M > 0

    if E == 0:
        return empty residual graph

    support =
        sample_uniform_support(
            residual_problem.admissible_pairs,
            E,
            rng
        )

    if E == 1:
        occupations = [T]
        backend = Deterministic

    else if T == E:
        occupations = [1; E]
        backend = Deterministic

    else:
        estimated_rejection =
            estimate_w_rejection(E, T, M)

        if estimated_rejection <= 0.8:

            rejection_result =
                try_w_weak_composition_rejection(
                    E,
                    T,
                    M,
                    max_attempts = 20,
                    rng
                )

            if rejection_result succeeded:
                occupations = rejection_result.counts
                backend = BoundedWeakCompositionRejection
            else:
                occupations =
                    sample_w_occupations_dp(E, T, M, rng)
                backend = DynamicProgrammingFallback

        else:
            occupations =
                sample_w_occupations_dp(E, T, M, rng)
            backend = DynamicProgramming

    graph_builder =
        SparseGraphBuilder::with_capacity(E)

    for index in 0..E:

        pair = support[index]
        occupation = occupations[index]

        assert occupation >= 1

        graph_builder.insert(pair, occupation)

    residual_graph = graph_builder.finish()

    validate:
        residual_graph.occupied_pair_count == E
        residual_graph.total_occupation == T
        every occupation >= 1
        every pair belongs to the residual admissible set

    return residual_graph
```

The calling pipeline merges the residual graph with fixed occupations and performs final global validation.

---

# 25. Architecture

A possible repository layout is:

```text
generation/
    microcanonical/
        fixed_et/
            mod.rs
            residual.rs
            support.rs
            diagnostics.rs

            me/
                ...

            b/
                ...

            w/
                mod.rs
                weak_composition.rs
                rejection.rs
                rejection_estimate.rs
                partition.rs
                sampler.rs
```

The scientific separation is:

```text
fixed-E,T orchestration
    shared

support selection
    shared

uniform weak-composition proposal
    reusable combinatorial utility

W aggregation and positivity rejection
    W-specific

W partition function
    W-specific but derived from shared family laws

sparse reconstruction
    shared
```

---

# 26. Suggested internal interfaces

Conceptually:

```rust
pub struct WFixedETConfig {
    pub layers: OccNum,
    pub rejection_threshold: f64,
    pub max_rejection_attempts: usize,
}
```

Backend diagnostics:

```rust
pub enum WFixedETBackend {
    Deterministic,
    BoundedWeakCompositionRejection,
    DynamicProgramming,
    DynamicProgrammingFallback,
}
```

Sampler result:

```rust
pub struct WFixedETSample {
    pub graph: SparseOccGraph,
    pub backend: WFixedETBackend,
    pub rejection_attempts: usize,
    pub estimated_rejection: Option<f64>,
}
```

The public API may omit diagnostics while tests and benchmarks expose them.

---

# 27. Family-law reuse

The sampler must not duplicate W combinatorics already present elsewhere.

The W family abstraction should provide or expose:

- local support \(t\ge0\);
- positive support \(t\ge1\);
- \(\log\binom{M+t-1}{t}\);
- generating-function coefficients if already implemented;
- occupation validation.

Filtering and generation must share the same family law.

The sampler should not maintain an independent conflicting W degeneracy implementation.

---

# 28. Exact enumeration validation

For small \(L,E,T,M\), enumerate every feasible residual graph.

The unnormalized weight is

\[
w(\mathbf t)
=
\prod_{i:t_i>0}
\binom{M+t_i-1}{t_i}.
\]

Normalize:

\[
P(\mathbf t)
=
\frac{
w(\mathbf t)
}{
\sum_{\mathbf u}w(\mathbf u)
}.
\]

Compare empirical frequencies from:

- forced weak-composition rejection;
- forced DP sampling;
- automatic hybrid selection.

---

# 29. Occupation-vector validation

Test the occupation allocator independently from support selection.

Enumerate all positive compositions

\[
t_i\ge1,
\qquad
\sum_i t_i=T.
\]

The exact probability is

\[
P(t_1,\ldots,t_E)
=
\frac{
\prod_i\binom{M+t_i-1}{t_i}
}{
Z_W^+(E,T,M)
}.
\]

Compare exact and empirical:

- vector frequencies;
- marginal occupations;
- occupation histograms;
- symmetry under pair permutation.

This isolates the family-specific allocator.

---

# 30. Conditioned grand-canonical identity

Generate samples from the existing W grand-canonical model using:

- the same \(M\);
- the same residual admissible-pair set;
- the same masks;
- the same W family law.

Retain samples satisfying the target \((E,T)\).

Then verify

\[
P_{\mathrm{GC},W}
\left(
\mathbf t
\mid
E(\mathbf t)=E,\,
T(\mathbf t)=T
\right)
=
P_{\mathrm{MC},W}
(\mathbf t\mid E,T).
\]

This equality is exact at finite size.

Care is required near the W grand-canonical convergence boundary. Poor conditioned-sample yield is a validation-efficiency problem, not evidence against the identity.

---

# 31. Backend agreement

Force both backends at the same parameter points.

Compare:

- full occupation-vector frequencies;
- marginal pair occupations;
- occupation histograms;
- support frequencies;
- graph-level observables.

Include regimes with:

- low rejection;
- rejection near \(0.8\);
- high rejection;
- \(T\) close to \(E\);
- large \(T/E\);
- \(M=1\);
- large \(M\).

The two backends must agree within Monte Carlo uncertainty.

---

# 32. Required boundary tests

At minimum, test:

## \(E=0,T=0\)

Return an empty residual graph.

## \(E=1\)

Return occupation \([T]\).

## \(T=E\)

Return \([1,\ldots,1]\).

## \(M=1\)

Since

\[
\binom{1+t-1}{t}=1,
\]

all positive compositions of \(T\) into \(E\) parts are equally likely.

The DP and rejection samplers must reduce to the uniform positive-composition law.

## \(E=L\)

Every residual admissible pair is selected.

## Near lower bound

Test

\[
T=E+1,
\]

where positivity conditioning may strongly affect rejection.

## Large mean occupation

Test large

\[
T/E,
\]

where rejection should become favorable.

## Threshold region

Choose points where estimated rejection is close to \(0.8\).

## Fallback activation

Force retry exhaustion and verify exact DP fallback.

## Fixed-mask reconstruction

Include fixed positive and fixed zero occupations and verify the final global constraints.

---

# 33. Statistical tests

For enumerated systems, use:

- multinomial goodness-of-fit;
- total variation distance;
- maximum absolute probability error;
- KL divergence with careful zero-count handling;
- repeated fixed seed sets.

Avoid fragile tests based only on p-values.

Continuous-integration tests should have:

- deterministic seed sets;
- bounded runtime;
- variance-aware tolerances;
- no unbounded Monte Carlo loops.

---

# 34. Benchmarks

Benchmark separately:

1. rejection estimation;
2. uniform subset sampling for stars and bars;
3. weak-composition decoding;
4. group aggregation;
5. rejection attempts;
6. W DP table construction;
7. W DP sequential sampling;
8. complete residual graph generation;
9. fixed-mask reconstruction.

Sweep over:

- \(E\);
- \(T/E\);
- \(M\);
- residual admissible count \(L\);
- distance from the positivity boundary \(T-E\).

Record:

- runtime;
- allocations;
- peak memory;
- rejection attempts;
- estimated rejection;
- observed rejection;
- selected backend;
- DP cache hit or miss.

Compare estimated and observed rejection to determine whether the selector should later be improved.

---

# 35. Caching

Repeated sampling with the same \((E,T,M)\) may reuse:

- \(\log\binom{M+t-1}{t}\) for \(0\le t\le T\);
- the W DP table;
- rejection estimates;
- subset-sampling scratch buffers.

Cache keys must include

\[
(E,T,M).
\]

Caching should be optional and bounded.

Correct ownership and reproducibility take priority over global caching complexity.

---

# 36. Error handling

Return explicit errors for:

- \(E>L\);
- \(E=0,T\ne0\);
- \(E>0,T<E\);
- \(M=0\);
- checked arithmetic overflow;
- invalid residual mask state;
- impossible fixed occupations;
- DP allocation failure;
- categorical normalization failure caused by numerical corruption.

Do not return an error when rejection retries are exhausted.

That condition triggers the exact fallback.

---

# 37. Correctness of the complete sampler

The complete sampler:

1. chooses a support \(S\) uniformly among all size-\(E\) subsets of the \(L\) residual admissible pairs;
2. samples a positive occupation vector according to

   \[
   P(\mathbf t\mid S)
   =
   \frac{
   \prod_i\binom{M+t_i-1}{t_i}
   }{
   Z_W^+(E,T,M)
   };
   \]

3. assigns occupations to the selected support.

Therefore

\[
P(S,\mathbf t)
=
\frac{1}{\binom{L}{E}}
\frac{
\prod_i\binom{M+t_i-1}{t_i}
}{
Z_W^+(E,T,M)
}.
\]

The prefactor depends only on \(L,E,T,M\), not on the particular feasible state.

Hence

\[
P(S,\mathbf t)
\propto
\prod_i\binom{M+t_i-1}{t_i},
\]

which is exactly the W microcanonical fixed-\((E,T)\) target.

Both occupation backends implement the same conditional law:

- rejection by uniform microscopic weak compositions conditioned on nonempty groups;
- DP by sequential factorization of the exact partition function.

Automatic switching and fallback therefore preserve correctness.

---

# 38. Recommended implementation order

Implement in the following order:

1. Extend or confirm residual fixed-\((E,T)\) feasibility for unbounded positive occupations.
2. Reuse the common support sampler unchanged.
3. Implement W local log-degeneracy reuse.
4. Implement the W log-partition DP.
5. Implement sequential DP sampling.
6. Validate the DP allocator against exact positive-composition enumeration.
7. Integrate DP allocation with support and graph construction.
8. Validate complete residual graphs.
9. Implement uniform weak-composition sampling by stars and bars.
10. Implement direct macro aggregation.
11. Implement bounded positivity rejection.
12. Validate the rejection allocator independently.
13. Implement the cheap rejection estimate.
14. Add automatic selection with

    \[
    r=0.8.
    \]

15. Add retry fallback to DP.
16. Add forced-backend and threshold tests.
17. Validate against conditioned W grand-canonical samples.
18. Add benchmarks and diagnostics.

The DP sampler should be implemented first because it provides the exact debugging oracle for the rejection backend.

---

# 39. Completion criteria

The W fixed-\((E,T)\) phase is complete when:

- residual feasibility enforces \(T\ge E\);
- fixed-mask preprocessing is reused;
- support sampling is reused unchanged;
- W local degeneracy is reused from the family layer;
- the DP allocator passes exact enumeration tests;
- the weak-composition rejection allocator passes exact enumeration tests;
- automatic backend selection uses

  \[
  r=0.8;
  \]

- rejection retries are bounded;
- retry exhaustion falls back to DP;
- deterministic boundary cases are implemented;
- sparse graph memory is preserved;
- no \(O(N^2)\) graph allocation is introduced;
- conditioned grand-canonical validation passes;
- backend agreement tests pass;
- benchmark integration is complete;
- APIs and diagnostics are documented.

---

# 40. Conceptual correspondence across families

| Component | ME fixed \((E,T)\) | B fixed \((E,T)\) | W fixed \((E,T)\) |
|---|---|---|---|
| Microscopic object | distinguishable event label | binary layer-pair cell | indistinguishable event in a layer box |
| Microscopic sampling | labels with replacement | subset without replacement | uniform weak composition |
| Local support | \(t\ge0\) | \(0\le t\le M\) | \(t\ge0\) |
| Positive support | \(t\ge1\) | \(1\le t\le M\) | \(t\ge1\) |
| Degeneracy | \(T!/\prod_i t_i!\) | \(\prod_i\binom{M}{t_i}\) | \(\prod_i\binom{M+t_i-1}{t_i}\) |
| Support law | uniform | uniform | uniform |
| Acceptance condition | no empty box | no empty row | no empty group |
| Fast proposal | multinomial labels | uniform cell subset | uniform weak composition |
| Direct fallback | Stirling surjection recursion | bounded weighted DP | unbounded weighted DP |
| Positive generating function | \(e^x-1\) | \((1+x)^M-1\) | \((1-x)^{-M}-1\) |
| Rejection threshold | \(r=0.8\) | \(r=0.8\) | \(r=0.8\) |

The W implementation is therefore the third realization of the same fixed-\((E,T)\) architecture, with only the microscopic proposal and family-specific partition recurrence changed.
