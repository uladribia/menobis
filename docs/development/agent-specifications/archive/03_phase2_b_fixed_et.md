# Phase 2 — Exact Microcanonical B Sampler with Fixed \((E,T)\)

**Version:** August 2026

---

# 1. Scope

This document specifies the implementation of the **B-family microcanonical ensemble with fixed occupied-pair count \(E\) and fixed total occupation \(T\)** in MENoBiS.

It builds directly on the architecture established for the ME fixed-\((E,T)\) implementation.

The goals are:

- preserve the residual-problem and prefilter architecture;
- reuse the existing support sampler unchanged;
- replace only the family-specific positive-occupation allocator;
- implement an exact direct fallback;
- implement a fast bounded rejection path;
- select the backend automatically using a rejection threshold of

\[
r=0.8;
\]

- validate the implementation against exact enumeration and conditioned grand-canonical sampling.

This is not a new generation architecture. It is a new family-specific occupation law inside the same fixed-\((E,T)\) framework.

---

# 2. Scientific definition

The B family represents the aggregation of \(M\) binary layers.

For each admissible pair \((i,j)\), the occupation number satisfies

\[
0\le t_{ij}\le M.
\]

The family degeneracy is

\[
D_B(\mathbf t)
=
\prod_{ij}\binom{M}{t_{ij}}.
\]

Under fixed occupied-pair count \(E\) and fixed total occupation \(T\), the target distribution is

\[
P_B(\mathbf t\mid E,T,M)
=
\frac{1}{Z_{B}(E,T,M)}
\prod_{ij}\binom{M}{t_{ij}},
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
0\le t_{ij}\le M.
\]

The implementation must sample this finite distribution exactly, up to normal floating-point rounding in categorical branch probabilities.

---

# 3. Relationship to the ME implementation

The B implementation reuses the Phase 1 fixed-\((E,T)\) pipeline:

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
family-specific positive occupation allocation
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

The following components remain unchanged:

- fixed-mask preprocessing;
- forbidden-pair handling;
- residual-problem construction;
- admissible-pair indexing;
- uniform support selection;
- sparse graph construction;
- reconstruction;
- random-number-generator handling;
- hard-constraint validation;
- benchmark integration;
- conditioned grand-canonical validation framework.

Only the positive-occupation allocator and B-specific feasibility checks are new.

---

# 4. Residual problem

The sampler never receives the original user problem directly.

The prefilter must first remove or account for:

- forbidden pairs;
- fixed zero occupations;
- fixed positive occupations;
- deterministic assignments;
- unavailable pair locations;
- any existing mask logic.

The sampler receives the residual problem:

- \(L\): number of residual admissible pairs;
- \(E\): residual number of occupied pairs;
- \(T\): residual total occupation;
- \(M\): number of binary layers;
- an indexed or iterable collection of residual admissible pairs.

If the user requests total constraints \((E_{\mathrm{user}},T_{\mathrm{user}})\), and the fixed positive occupations contribute

\[
E_{\mathrm{fix}}
=
\sum_{ij}\mathbf 1(t_{ij}^{\mathrm{fix}}>0),
\]

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

The residual sampler solves only

\[
(E_{\mathrm{res}},T_{\mathrm{res}}).
\]

The fixed occupations are reintroduced after residual generation.

The sampler must not manipulate user masks directly.

---

# 5. Feasibility

For the residual problem, feasibility requires

\[
0\le E\le L.
\]

If \(E=0\), then necessarily

\[
T=0.
\]

If \(E>0\), every occupied pair must contain at least one occupied layer and at most \(M\) occupied layers. Therefore

\[
E\le T\le ME.
\]

The complete residual feasibility conditions are:

```text
0 <= E <= L

if E == 0:
    T must equal 0

if E > 0:
    E <= T <= M * E
```

All products such as \(ME\) must use checked integer arithmetic.

The prefilter must also verify every fixed occupation:

\[
0\le t_{ij}^{\mathrm{fix}}\le M.
\]

Special deterministic cases are:

- \(E=0,T=0\): return an empty residual graph;
- \(E=1\): assign the single support pair occupation \(T\);
- \(T=E\): every selected pair receives occupation \(1\);
- \(T=ME\): every selected pair receives occupation \(M\);
- \(M=1\): feasibility requires \(T=E\), reducing the problem to uniform binary support sampling.

These cases should bypass the generic occupation allocator.

---

# 6. Uniform support factorization

Let \(S\) be a support containing exactly \(E\) residual admissible pairs.

For a fixed support, write the positive occupations as

\[
(t_1,\ldots,t_E),
\]

with

\[
1\le t_i\le M,
\qquad
\sum_{i=1}^{E}t_i=T.
\]

The conditional degeneracy is

\[
D_B(t_1,\ldots,t_E)
=
\prod_{i=1}^{E}\binom{M}{t_i}.
\]

The corresponding support-specific partition function is

\[
Z_B^+(E,T,M)
=
\sum_{\substack{1\le t_i\le M\\\sum_i t_i=T}}
\prod_{i=1}^{E}\binom{M}{t_i}.
\]

This expression depends only on \(E\), \(T\), and \(M\). It does not depend on the identities of the selected pair locations.

Therefore every support of size \(E\) has the same total probability, and

\[
P(S\mid L,E,T,M)
=
\frac{1}{\binom{L}{E}}.
\]

Thus the B fixed-\((E,T)\) sampler factorizes exactly into:

1. uniform support sampling;
2. B-specific positive occupation allocation.

The support sampler from the ME implementation must be reused unchanged.

---

# 7. Positive occupation law

Conditioned on a sampled support, the B occupation vector must satisfy

\[
1\le t_i\le M,
\]

\[
\sum_{i=1}^{E}t_i=T,
\]

with probability

\[
P_B(t_1,\ldots,t_E\mid E,T,M)
=
\frac{1}{Z_B^+(E,T,M)}
\prod_{i=1}^{E}\binom{M}{t_i}.
\]

The occupation allocator is therefore a bounded weighted-composition sampler.

This is the only family-specific mathematical component of the fixed-\((E,T)\) generator.

---

# 8. Microscopic binary-cell representation

The B occupation law admits a direct microscopic interpretation.

Once \(E\) pair locations have been selected, construct conceptually an \(E\times M\) binary array:

- each row corresponds to one selected network pair;
- each column corresponds to one binary layer;
- each cell indicates whether that pair is occupied in that layer.

There are

\[
EM
\]

binary cells.

Choose exactly \(T\) cells uniformly without replacement.

For a row-occupation vector

\[
(t_1,\ldots,t_E),
\]

the number of cell subsets that produce this vector is

\[
\prod_{i=1}^{E}\binom{M}{t_i}.
\]

Before conditioning on nonempty rows,

\[
P(t_1,\ldots,t_E)
=
\frac{
\prod_i\binom{M}{t_i}
}{
\binom{EM}{T}
}.
\]

Conditioning on

\[
t_i>0
\qquad
\forall i
\]

gives

\[
P(t_1,\ldots,t_E\mid t_i>0\;\forall i)
\propto
\prod_i\binom{M}{t_i},
\]

which is exactly the target B microcanonical law.

This leads to a natural rejection sampler:

1. choose \(T\) cells uniformly from \(EM\);
2. count selected cells per row;
3. accept only if all \(E\) rows are nonempty.

---

# 9. Hybrid backend strategy

The occupation allocator uses two exact backends:

1. **bounded binary-cell subset rejection**, used as the fast path;
2. **dynamic-programming sequential sampling**, used as the guaranteed fallback.

Backend selection uses a predicted rejection probability and the threshold

\[
r=0.8.
\]

The policy is:

```text
if estimated_rejection <= 0.8:
    try bounded subset rejection
else:
    use the direct DP sampler
```

If bounded rejection unexpectedly exhausts its retry limit, the implementation must switch to the direct DP sampler.

Retry exhaustion is not a sampling error.

Both backends target the same exact distribution, so backend misclassification affects performance only, never correctness.

---

# 10. Fast rejection backend

## 10.1 Algorithm

Given \(E\), \(T\), and \(M\):

1. Let

   \[
   C=EM.
   \]

2. Sample a uniform subset of size \(T\) from the integer range

   \[
   \{0,\ldots,C-1\}.
   \]

3. Map each selected cell index \(c\) to row

   \[
   i=\left\lfloor\frac{c}{M}\right\rfloor.
   \]

4. Increment the occupation count for row \(i\).

5. Accept if every row count is positive.

6. Otherwise reset and retry.

Pseudocode:

```text
try_b_subset_rejection(E, T, M, max_attempts, rng):

    total_cells = checked_mul(E, M)
    counts = [0; E]

    for attempt in 1..=max_attempts:

        reset counts to zero
        occupied_rows = 0

        selected_cells =
            sample_uniform_subset(total_cells, T, rng)

        for cell in selected_cells:

            row = cell / M

            if counts[row] == 0:
                occupied_rows += 1

            counts[row] += 1

        if occupied_rows == E:
            return Success(counts, attempt)

    return Exhausted
```

The layer coordinate

\[
\text{layer}=c\bmod M
\]

does not need to be stored for aggregate B-network generation.

Only row counts are required.

---

## 10.2 Correctness

Every \(T\)-subset of the \(EM\) cells is selected with probability

\[
\frac{1}{\binom{EM}{T}}.
\]

For a fixed occupation vector \(\mathbf t\), the number of subsets yielding that vector is

\[
\prod_i\binom{M}{t_i}.
\]

Therefore the unconditional probability is

\[
P(\mathbf t)
=
\frac{
\prod_i\binom{M}{t_i}
}{
\binom{EM}{T}
}.
\]

Conditioning on all rows being nonempty yields

\[
P(\mathbf t\mid t_i>0\;\forall i)
=
\frac{
\prod_i\binom{M}{t_i}
}{
A_B(E,T,M)
},
\]

where \(A_B(E,T,M)\) is the number of valid nonempty-row subsets.

This is exactly the target B occupation law.

---

# 11. Rejection probability

Let

\[
A_B(E,T,M)
\]

denote the number of \(T\)-cell subsets touching all \(E\) rows.

Then

\[
p_{\mathrm{acc}}^B(E,T,M)
=
\frac{A_B(E,T,M)}{\binom{EM}{T}},
\]

and

\[
p_{\mathrm{rej}}^B(E,T,M)
=
1-p_{\mathrm{acc}}^B(E,T,M).
\]

By inclusion-exclusion,

\[
A_B(E,T,M)
=
\sum_{j=0}^{E}
(-1)^j
\binom{E}{j}
\binom{M(E-j)}{T}.
\]

Equivalently,

\[
A_B(E,T,M)
=
[x^T]
\left((1+x)^M-1\right)^E.
\]

The exact inclusion-exclusion expression is useful for small-system tests, but it is not the preferred production method for backend selection because alternating sums can be numerically delicate.

---

# 12. Cheap rejection estimate

For backend selection, use a cheap approximation based on the probability that a particular row is empty.

A given row is empty when all \(T\) selected cells come from the other \(E-1\) rows. Therefore

\[
q_0
=
P(\text{one specified row is empty})
=
\frac{
\binom{M(E-1)}{T}
}{
\binom{ME}{T}
},
\]

provided

\[
T\le M(E-1).
\]

If

\[
T>M(E-1),
\]

then \(q_0=0\), because it is impossible to place all selected cells outside that row.

Approximate the probability that all rows are nonempty by

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

The row-emptiness events are dependent, so this is only a backend-selection estimate. That is acceptable because:

- the rejection sampler itself remains exact;
- the DP sampler remains exact;
- a poor estimate changes only which exact backend runs first;
- retries are bounded;
- fallback guarantees termination.

Compute \(q_0\) in log space:

\[
\log q_0
=
\log\binom{M(E-1)}{T}
-
\log\binom{ME}{T}.
\]

Use

\[
\log\binom{n}{k}
=
\log\Gamma(n+1)
-
\log\Gamma(k+1)
-
\log\Gamma(n-k+1).
\]

For numerical stability:

```text
log_q0 = log_binomial(M * (E - 1), T)
       - log_binomial(M * E, T)

q0 = exp(log_q0)

log_p_acc_est = E * log1p(-q0)

p_rej_est = -expm1(log_p_acc_est)
```

The implementation should use existing combinatorial or log-gamma utilities already present in MENoBiS whenever possible.

---

# 13. Backend-selection rule

The production rule is fixed for this phase:

\[
r=0.8.
\]

Therefore:

```text
if estimated_rejection <= 0.8:
    use bounded rejection first
else:
    use direct DP sampling
```

Equivalently, the rejection backend is attempted when

\[
\widehat p_{\mathrm{acc}}\ge0.2.
\]

The estimated expected number of attempts is then at most approximately

\[
\frac{1}{0.2}=5.
\]

The threshold should be represented by a named constant or configuration field:

```rust
const DEFAULT_REJECTION_THRESHOLD: f64 = 0.8;
```

It should not be hidden as an unexplained literal.

---

# 14. Retry bound

Even with backend selection, rejection attempts must be bounded.

A suitable Phase 2 default is:

```rust
const DEFAULT_MAX_REJECTION_ATTEMPTS: usize = 20;
```

The logic is:

```text
attempt rejection up to 20 times

if accepted:
    return the exact sample

if exhausted:
    construct or retrieve the DP table
    return an exact DP sample
```

The retry bound protects against:

- poor approximation near the decision boundary;
- numerical error in the estimated rejection probability;
- unlucky random sequences;
- pathological parameter points.

Exhaustion must not return an approximate result.

Exhaustion must not be exposed as a user-visible sampling failure unless the DP fallback itself fails.

---

# 15. Efficient cell-subset sampling

The rejection backend must not allocate a dense vector of length \(EM\) merely to choose \(T\) cells.

Use a uniform integer-subset sampler.

Recommended methods:

- Floyd sampling when \(T\ll EM\);
- complement sampling when \(EM-T\ll EM\);
- partial Fisher-Yates only if an owned compact index buffer already exists;
- reservoir sampling only if the cell domain is streamed.

The implementation should choose between occupied-cell and hole-cell sampling.

## 15.1 Direct selected-cell mode

When

\[
T\le\frac{EM}{2},
\]

sample \(T\) selected cells.

Initialize

```text
counts = [0; E]
```

and increment each selected row.

## 15.2 Complement mode

When

\[
T>\frac{EM}{2},
\]

sample the

\[
H=EM-T
\]

unselected cells instead.

Initialize

```text
counts = [M; E]
```

and decrement each row for every sampled hole.

Accept if every count remains positive.

Pseudocode:

```text
sample_candidate_counts(E, T, M, rng):

    total_cells = E * M

    if T <= total_cells / 2:

        counts = [0; E]

        for cell in sample_uniform_subset(total_cells, T, rng):
            counts[cell / M] += 1

    else:

        holes = total_cells - T
        counts = [M; E]

        for cell in sample_uniform_subset(total_cells, holes, rng):
            counts[cell / M] -= 1

    return counts
```

This reduces each attempt to roughly

\[
O\left(E+\min(T,EM-T)\right).
\]

---

# 16. Exact direct fallback

The guaranteed fallback is a weighted bounded-composition dynamic program.

Define

\[
Z_B(k,s)
\]

as the total B degeneracy weight of assigning total occupation \(s\) to \(k\) occupied pairs:

\[
Z_B(k,s)
=
\sum_{\substack{1\le t_i\le M\\\sum_i t_i=s}}
\prod_{i=1}^{k}\binom{M}{t_i}.
\]

The base cases are

\[
Z_B(0,0)=1,
\]

and

\[
Z_B(0,s)=0
\qquad
\text{for }s\ne0.
\]

For \(k>0\),

\[
Z_B(k,s)
=
\sum_{t=1}^{M}
\binom{M}{t}
Z_B(k-1,s-t),
\]

with impossible terms excluded.

The remaining \(k-1\) occupations must satisfy

\[
k-1\le s-t\le M(k-1).
\]

Therefore the feasible range is

\[
t_{\min}
=
\max\left(1,\;s-M(k-1)\right),
\]

\[
t_{\max}
=
\min\left(M,\;s-(k-1)\right).
\]

The recurrence becomes

\[
Z_B(k,s)
=
\sum_{t=t_{\min}}^{t_{\max}}
\binom{M}{t}
Z_B(k-1,s-t).
\]

---

# 17. Generating-function interpretation

The positive single-pair generating function is

\[
g_B(x)
=
\sum_{t=1}^{M}\binom{M}{t}x^t.
\]

Using the binomial theorem,

\[
g_B(x)
=
(1+x)^M-1.
\]

For \(E\) occupied pairs,

\[
g_B(x)^E
=
\left((1+x)^M-1\right)^E.
\]

Therefore

\[
Z_B(E,T)
=
[x^T]
\left((1+x)^M-1\right)^E.
\]

This is exactly the number of \(T\)-cell subsets touching all \(E\) rows.

The direct DP is therefore a coefficient computation for the family generating function, not an ad hoc recurrence.

This observation should guide future W-family reuse.

---

# 18. Log-space DP

The values \(Z_B(k,s)\) can become extremely large.

Production code should store

\[
\log Z_B(k,s).
\]

The recurrence becomes

\[
\log Z_B(k,s)
=
\operatorname{logsumexp}_{t=t_{\min}}^{t_{\max}}
\left[
\log\binom{M}{t}
+
\log Z_B(k-1,s-t)
\right].
\]

Impossible states are represented by

\[
-\infty.
\]

Boundary conditions:

\[
\log Z_B(0,0)=0,
\]

\[
\log Z_B(0,s)=-\infty
\quad
\text{for }s\ne0.
\]

The implementation should precompute

\[
\log\binom{M}{t}
\]

for

\[
0\le t\le M
\]

once per \(M\), preferably through the existing B-family mathematical layer.

---

# 19. Sequential sampling from the DP

After building the table, occupations are sampled sequentially.

At state \((k,s)\), the probability that the next pair receives occupation \(t\) is

\[
P(t\mid k,s)
=
\frac{
\binom{M}{t}Z_B(k-1,s-t)
}{
Z_B(k,s)
}.
\]

In log space, the unnormalized branch weight is

\[
\ell_t
=
\log\binom{M}{t}
+
\log Z_B(k-1,s-t).
\]

Normalize the feasible branch weights using a stable categorical sampler.

Then update

\[
k\leftarrow k-1,
\]

\[
s\leftarrow s-t.
\]

Pseudocode:

```text
sample_b_occupations_dp(E, T, M, rng):

    log_Z = build_log_partition_table(E, T, M)

    occupations = Vec::with_capacity(E)

    k = E
    remaining = T

    while k > 0:

        t_min = max(1, remaining - M * (k - 1))
        t_max = min(M, remaining - (k - 1))

        branches = empty

        for t in t_min..=t_max:

            log_weight =
                log_binomial[M][t]
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

The output is an ordered exchangeable occupation vector.

A final random shuffle is not mathematically required if the sequential sampler is implemented correctly, because the target law is symmetric and the DP conditionals preserve that law. A shuffle may still be retained defensively or for consistency with other allocators.

---

# 20. DP storage

The full table is conceptually indexed by

\[
0\le k\le E,
\qquad
0\le s\le T.
\]

However, only states satisfying

\[
k\le s\le Mk
\]

are feasible.

The implementation should use one of:

- a dense rectangular table for simplicity;
- a banded table storing only feasible totals;
- per-row vectors with row-specific offsets.

For the first implementation, per-row feasible bands are recommended:

```text
row k stores s from k to min(T, M*k)
```

This reduces memory without making indexing excessively complex.

The table must remain available during backward sequential sampling. Rolling rows alone are insufficient unless rows are recomputed or checkpointed.

For Phase 2, retain the full feasible band table.

---

# 21. Complexity

## 21.1 Rejection path

Each attempt costs approximately

\[
O\left(E+\min(T,EM-T)\right)
\]

time.

The expected cost is

\[
O\left(
\frac{
E+\min(T,EM-T)
}{
p_{\mathrm{acc}}^B
}
\right).
\]

Because rejection is selected only when estimated rejection is at most \(0.8\), the estimated expected attempt count is at most five.

Temporary memory is approximately

\[
O\left(E+\min(T,EM-T)\right),
\]

depending on the subset-sampling implementation.

## 21.2 DP fallback

The naive recurrence requires up to \(M\) branches per state.

Worst-case time is

\[
O(ETM).
\]

Memory is

\[
O(ET),
\]

reduced in practice by storing only feasible bands.

Sequential generation after table construction costs up to

\[
O(EM)
\]

in the naive branch loop, usually less due to feasibility bounds.

Repeated samples with the same \((E,T,M)\) should reuse cached tables where appropriate.

---

# 22. Full end-to-end algorithm

```text
sample_b_fixed_et(residual_problem, M, rng):

    L = residual_problem.admissible_pair_count
    E = residual_problem.residual_edges
    T = residual_problem.residual_total

    validate:
        0 <= E <= L
        if E == 0 then T == 0
        if E > 0 then E <= T <= M * E

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

    else if T == E:
        occupations = [1; E]

    else if T == M * E:
        occupations = [M; E]

    else:
        estimated_rejection =
            estimate_b_rejection(E, T, M)

        if estimated_rejection <= 0.8:

            rejection_result =
                try_b_subset_rejection(
                    E,
                    T,
                    M,
                    max_attempts = 20,
                    rng
                )

            if rejection_result succeeded:
                occupations = rejection_result.counts
                backend = BoundedSubsetRejection
            else:
                occupations =
                    sample_b_occupations_dp(E, T, M, rng)
                backend = DynamicProgrammingFallback

        else:
            occupations =
                sample_b_occupations_dp(E, T, M, rng)
            backend = DynamicProgramming

    graph_builder =
        SparseGraphBuilder::with_capacity(E)

    for index in 0..E:

        pair = support[index]
        occupation = occupations[index]

        assert 1 <= occupation <= M

        graph_builder.insert(pair, occupation)

    residual_graph = graph_builder.finish()

    validate:
        residual_graph.occupied_pair_count == E
        residual_graph.total_occupation == T
        every occupation <= M
        every pair belongs to residual admissible set

    return residual_graph
```

The calling generation pipeline then merges the residual graph with fixed occupations and performs final global validation.

---

# 23. Architecture

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
                mod.rs
                rejection.rs
                stirling.rs

            b/
                mod.rs
                rejection.rs
                rejection_estimate.rs
                partition.rs
                sampler.rs
```

The exact file names should follow the existing MENoBiS conventions.

The scientific separation should remain:

```text
fixed-E,T orchestration
    shared

support selection
    shared

B rejection proposal
    B-specific

B partition function
    B-specific but built from shared family laws

sparse reconstruction
    shared
```

---

# 24. Suggested internal interfaces

Conceptually:

```rust
pub struct FixedETResidualProblem<'a> {
    pub admissible_pairs: &'a AdmissiblePairs,
    pub occupied_pairs: usize,
    pub total_occupation: OccNum,
}
```

B-specific configuration:

```rust
pub struct BFixedETConfig {
    pub layers: OccNum,
    pub rejection_threshold: f64,
    pub max_rejection_attempts: usize,
}
```

Backend diagnostics:

```rust
pub enum BFixedETBackend {
    Deterministic,
    BoundedSubsetRejection,
    DynamicProgramming,
    DynamicProgrammingFallback,
}
```

Sampler result:

```rust
pub struct BFixedETSample {
    pub graph: SparseOccGraph,
    pub backend: BFixedETBackend,
    pub rejection_attempts: usize,
    pub estimated_rejection: Option<f64>,
}
```

Public APIs may hide diagnostics by default while exposing them to tests and benchmarks.

---

# 25. Family-law reuse

The sampler must not duplicate B-family combinatorics already present elsewhere.

The B family abstraction should supply or expose:

- support bound \(0\le t\le M\);
- positive support bound \(1\le t\le M\);
- \(\log\binom{M}{t}\);
- any existing local generating-function coefficients;
- occupation validation.

Filtering and generation should share the same mathematical implementation.

The sampler should not maintain a separate independent implementation of the B degeneracy formula.

---

# 26. Validation strategy

Validation must proceed in layers.

## 26.1 Deterministic invariants

For every generated residual sample, verify:

\[
\sum_i\mathbf 1(t_i>0)=E,
\]

\[
\sum_i t_i=T,
\]

\[
1\le t_i\le M
\]

on selected support pairs.

Also verify:

- no duplicate support pairs;
- no forbidden pair appears;
- fixed occupations remain unchanged after reconstruction;
- RNG reproducibility for fixed seeds.

## 26.2 Exact enumeration

For small \(L,E,T,M\), enumerate every feasible residual graph.

The unnormalized state weight is

\[
w(\mathbf t)
=
\prod_{i:t_i>0}\binom{M}{t_i}.
\]

Normalize:

\[
P(\mathbf t)
=
\frac{w(\mathbf t)}
{\sum_{\mathbf u}w(\mathbf u)}.
\]

Compare empirical frequencies from:

- forced rejection backend;
- forced DP backend;
- automatic hybrid backend.

Use several parameter points, including both low- and high-rejection regimes.

## 26.3 Occupation-vector enumeration

Before testing complete graphs, test the family-specific allocator alone.

Enumerate all vectors satisfying

\[
1\le t_i\le M,
\qquad
\sum_i t_i=T.
\]

Compare empirical probabilities with

\[
P(t_1,\ldots,t_E)
\propto
\prod_i\binom{M}{t_i}.
\]

This isolates the allocator from support sampling and graph construction.

## 26.4 Conditioned grand-canonical identity

Generate samples from the existing B grand-canonical ensemble using:

- the same \(M\);
- the same residual admissible-pair set;
- the same masks;
- the same family law.

Retain samples satisfying the target \((E,T)\).

The conditioned empirical distribution must agree with the microcanonical distribution:

\[
P_{\mathrm{GC},B}
\left(
\mathbf t
\mid E(\mathbf t)=E,\,
T(\mathbf t)=T
\right)
=
P_{\mathrm{MC},B}(\mathbf t\mid E,T).
\]

This is an exact finite-size identity.

## 26.5 Backend agreement

Force each backend independently at the same parameter points.

Compare:

- occupation histograms;
- marginal pair occupations;
- complete occupation-vector frequencies;
- support frequencies;
- summary observables.

The rejection and DP backends must agree within Monte Carlo uncertainty.

---

# 27. Required boundary tests

At minimum, include:

## \(M=1\)

Then

\[
t_i=1
\]

for every occupied pair, and feasibility requires

\[
T=E.
\]

The sampler must reduce to uniform support sampling.

## \(T=E\)

All positive occupations equal \(1\).

## \(T=ME\)

All occupations equal \(M\).

## \(E=1\)

The only occupation equals \(T\).

## \(E=L\)

Every residual admissible pair is selected.

## Near lower bound

Test

\[
T=E+1,
\]

where rejection is often poor.

## Near saturation

Test

\[
T=ME-1.
\]

Complement cell sampling should be efficient.

## Threshold region

Choose parameter points where the estimated rejection probability is close to \(0.8\).

## Fallback activation

Force or simulate retry exhaustion and verify transparent transition to the DP sampler.

## Fixed-mask reconstruction

Include both fixed positive and fixed zero occupations and verify final global constraints.

---

# 28. Statistical tests

For small enumerated systems, use:

- multinomial goodness-of-fit;
- total variation distance;
- maximum absolute probability error;
- KL divergence, with care around zero empirical counts;
- confidence intervals over repeated seeds.

Avoid relying on a single large chi-square test when expected counts are too small.

Tests should be deterministic enough for continuous integration:

- fixed seed sets;
- bounded sample counts;
- tolerances justified by expected Monte Carlo variance;
- no fragile p-value-only assertions.

---

# 29. Benchmarks

Benchmark separately:

1. rejection-probability estimation;
2. uniform cell-subset sampling;
3. rejection attempts;
4. DP table construction;
5. DP sequential sampling;
6. complete residual graph generation;
7. reconstruction with fixed occupations.

Sweep over:

- \(E\);
- \(T/E\);
- \(M\);
- distance from lower bound \(T-E\);
- distance from saturation \(ME-T\);
- residual admissible count \(L\).

Record:

- runtime;
- allocations;
- peak memory;
- rejection attempts;
- estimated rejection;
- observed rejection;
- selected backend;
- DP cache hit or miss.

The observed rejection rate should also be compared against the estimate to evaluate whether a better selector is later warranted.

---

# 30. Caching

Repeated sampling with the same \((E,T,M)\) can reuse:

- \(\log\binom{M}{t}\) values;
- the DP partition table;
- backend-selection estimates.

Cache keys must include all parameters affecting the table:

\[
(E,T,M).
\]

Do not cache support samples.

Caching must remain optional and bounded.

The first implementation should prioritize correctness and clear ownership over elaborate global caches.

---

# 31. Error handling

Return explicit errors for:

- \(E>L\);
- \(E=0,T\ne0\);
- \(E>0,T<E\);
- \(T>ME\);
- \(M=0\) with nonzero constraints;
- fixed occupation exceeding \(M\);
- checked multiplication overflow;
- invalid residual mask state;
- DP allocation overflow or memory failure.

Do not return an error merely because rejection attempts were exhausted.

That condition triggers the exact fallback.

---

# 32. Correctness argument for the complete sampler

The complete sampler proceeds as follows:

1. choose a support \(S\) uniformly among all size-\(E\) subsets of the \(L\) residual admissible pairs;
2. sample a positive bounded occupation vector with probability

   \[
   P(\mathbf t\mid S)
   =
   \frac{
   \prod_i\binom{M}{t_i}
   }{
   Z_B^+(E,T,M)
   };
   \]

3. assign the occupations to the selected support.

Therefore

\[
P(S,\mathbf t)
=
\frac{1}{\binom{L}{E}}
\frac{
\prod_i\binom{M}{t_i}
}{
Z_B^+(E,T,M)
}.
\]

The prefactor depends only on \(L,E,T,M\), not on the particular feasible state.

Hence

\[
P(S,\mathbf t)
\propto
\prod_i\binom{M}{t_i},
\]

which is precisely the B microcanonical target distribution with fixed \((E,T)\).

Both occupation backends implement the same conditional law:

- rejection through uniform binary-cell subsets conditioned on nonempty rows;
- DP through exact sequential factorization of the partition function.

Therefore automatic switching and fallback preserve correctness.

---

# 33. Recommended implementation order

Implement in the following order:

1. Extend residual fixed-\((E,T)\) validation with the B upper bound

   \[
   T\le ME.
   \]

2. Reuse the ME support sampler unchanged.

3. Implement the B log-partition DP.

4. Implement sequential DP sampling.

5. Validate the DP allocator against exact occupation-vector enumeration.

6. Integrate DP allocation with support and sparse graph construction.

7. Validate complete graphs against enumeration.

8. Implement uniform binary-cell subset sampling.

9. Add direct and complement cell modes.

10. Implement bounded rejection.

11. Validate rejection independently against enumeration.

12. Implement the cheap hypergeometric rejection estimate.

13. Add automatic selection with

    \[
    r=0.8.
    \]

14. Add retry fallback to DP.

15. Add forced-backend and threshold tests.

16. Validate against conditioned B grand-canonical samples.

17. Add benchmarks and diagnostics.

The DP sampler should be implemented first even though rejection is the intended common fast path, because the DP implementation is the exact debugging oracle.

---

# 34. Completion criteria

The B fixed-\((E,T)\) phase is complete when:

- residual feasibility includes \(E\le T\le ME\);
- fixed-mask preprocessing is reused;
- support sampling is reused unchanged;
- the DP occupation allocator passes exact enumeration tests;
- the binary-cell rejection allocator passes exact enumeration tests;
- automatic selection uses

  \[
  r=0.8;
  \]

- rejection retries are bounded;
- retry exhaustion falls back to DP;
- deterministic boundary cases are implemented;
- sparse memory is preserved;
- no \(O(N^2)\) graph allocation is introduced;
- conditioned grand-canonical validation passes;
- backend agreement tests pass;
- benchmark integration is complete;
- public and internal APIs are documented;
- family degeneracy logic is reused rather than duplicated.

---

# 35. Conceptual correspondence with ME

| Component | ME fixed \((E,T)\) | B fixed \((E,T)\) |
|---|---|---|
| Microscopic object | distinguishable event | binary layer-pair cell |
| Local support | \(t\ge0\) | \(0\le t\le M\) |
| Positive support | \(t\ge1\) | \(1\le t\le M\) |
| Degeneracy | \(T!/\prod_i t_i!\) | \(\prod_i\binom{M}{t_i}\) |
| Support law | uniform | uniform |
| Fast proposal | labels with replacement | cells without replacement |
| Acceptance condition | no empty box | no empty row |
| Direct fallback | Stirling surjection recursion | bounded weighted-composition DP |
| Local positive generating function | \(e^x-1\) | \((1+x)^M-1\) |
| Production selector | rejection estimate | rejection estimate |
| Rejection threshold | \(r=0.8\) | \(r=0.8\) |

The B implementation is therefore a direct extension of the Phase 1 architecture rather than a separate subsystem.
