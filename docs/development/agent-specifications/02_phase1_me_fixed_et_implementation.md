# Phase 1 Implementation — ME Fixed \((E,T)\)

**Version:** August 2026

## 1. Objective

This document translates the Phase 1 design into an implementation plan for an exact sparse ME microcanonical generator with fixed occupied-pair count \(E\) and total occupation \(T\).

The chosen occupation strategy is:

1. estimate rejection probability cheaply;
2. if estimated rejection is at most \(0.8\), use bounded multinomial rejection;
3. otherwise use direct Stirling-recursion sampling;
4. if bounded rejection unexpectedly exhausts its attempts, fall back to Stirling recursion;
5. never return an approximate sample.

## 2. Suggested module responsibilities

Exact file and module names must follow the current MENoBiS repository conventions. The implementation should nevertheless separate the following responsibilities:

```text
generation/microcanonical/
    fixed_et/
        problem.rs          residual fixed-(E,T) problem contract
        support.rs          uniform support selection
        occupation.rs       shared occupation-sampler interface
        rejection.rs        multinomial rejection backend
        stirling.rs         Stirling table and direct surjection backend
        selector.rs         cheap estimate and backend choice
        sampler.rs          end-to-end ME fixed-(E,T) orchestration
        validation.rs       internal invariant checks
        tests/
```

If the repository already has equivalent shared modules, extend them instead of creating parallel code.

## 3. Core internal types

Conceptual types are shown below. Names and generics should be adapted to existing code.

```rust
struct ResidualFixedETProblem<'a, PairIndex> {
    admissible_pairs: &'a [PairIndex],
    residual_edges: usize,
    residual_total: OccNum,
}
```

For very large \(T\), `OccNum` must match the existing MENoBiS occupation type. Conversions to `usize` for loops or vector indexing must be checked explicitly.

Backend diagnostics may use:

```rust
enum OccupationBackend {
    Rejection,
    Stirling,
    StirlingFallback,
}
```

This enum should remain internal unless benchmark instrumentation already exposes similar metadata.

The occupation result is:

```rust
struct PositiveOccupations {
    counts: Vec<OccNum>,
    backend: OccupationBackend,
    attempts: usize,
}
```

Production builds may omit diagnostics from the returned type and report them through benchmark hooks.

## 4. Validation before allocation

The residual problem validator must check:

```text
L = number of residual admissible pairs
E = residual occupied-pair count
T = residual total occupation
```

Conditions:

```text
E <= L
T >= 0
(E == 0) == (T == 0)
E == 0 or T >= E
```

Checked conversions are required when comparing `OccNum` with `usize`.

Errors should use the existing MENoBiS error hierarchy and distinguish:

- invalid user constraints;
- inconsistent fixed-mask contributions;
- residual infeasibility;
- numeric-size limitations;
- internal invariant violations.

## 5. End-to-end sampler pseudocode

```text
sample_me_fixed_et(residual_problem, rng):

    validate residual_problem

    L <- number of admissible residual pairs
    E <- residual edge count
    T <- residual total occupation

    if E == 0:
        return empty residual graph

    support <- sample_uniform_support(L, E, rng)

    occupations <- sample_positive_me_occupations(T, E, rng)

    optionally shuffle occupations if required by fallback representation

    builder <- sparse graph builder with capacity E

    for q in 0..E:
        pair <- admissible_pairs[support[q]]
        occ  <- occupations[q]
        assert occ > 0
        builder.insert(pair, occ)

    residual_graph <- builder.finish()

    validate residual graph invariants

    return residual_graph
```

Reconstruction with fixed occupations belongs to the preprocessing/generation pipeline outside this core sampler unless existing architecture combines them in one generator entry point.

## 6. Uniform support sampler

### 6.1 Required distribution

Sample a subset

\[
S\subseteq\{0,\dots,L-1\},
\qquad |S|=E,
\]

uniformly.

### 6.2 Preferred indexed algorithm

When admissible pairs support random access, use Floyd's algorithm or an existing equivalent utility.

Floyd's algorithm:

```text
sample_uniform_indices(L, E, rng):

    selected <- empty hash/set structure with capacity E

    for j in (L-E)..L:
        t <- uniform_integer(0, j)

        if t is already selected:
            insert j
        else:
            insert t

    output selected as vector
```

Every size-\(E\) subset is equally likely. Expected time and memory are \(O(E)\).

If deterministic iteration order of the set could interact with occupation assignment, shuffle the resulting support vector uniformly or use a randomized extraction order.

### 6.3 Partial shuffle alternative

If the residual admissible index vector is owned and may be mutated, a partial Fisher-Yates shuffle is simple:

```text
for i in 0..E:
    j <- uniform_integer(i, L-1)
    swap(indices[i], indices[j])
return indices[0..E]
```

Do not clone an enormous admissible-pair vector solely to use this method.

### 6.4 Streaming alternative

If admissible pairs are exposed only as an iterator, use reservoir sampling of size \(E\). This costs \(O(L)\) time and \(O(E)\) memory.

The support algorithm should be chosen based on existing infrastructure, not by introducing a new dense representation.

## 7. Cheap backend-selection estimate

For \(E>1\) and \(T>E\), define

\[
\lambda=\frac{T}{E}.
\]

Use

\[
\widehat{p}_{\mathrm{acc}}
=
\left(1-e^{-\lambda}\right)^E,
\qquad
\widehat{r}=1-\widehat{p}_{\mathrm{acc}}.
\]

Compute stably:

```text
x = -T / E
one_minus_exp = -expm1(x)
log_p_acc = E * ln(one_minus_exp)
```

Then:

```text
if log_p_acc is sufficiently negative:
    estimated_rejection = 1
else:
    estimated_rejection = -expm1(log_p_acc)
```

Special cases must bypass the estimate:

- \(E=1\): deterministic `[T]`;
- \(E=T\): deterministic all ones;
- \(E=0,T=0\): empty;
- \(T<E\): infeasible.

The threshold is:

```rust
const REJECTION_THRESHOLD: f64 = 0.8;
```

Phase 1 should keep this constant internal. Benchmarks may later justify configuration, but a public tuning knob is unnecessary initially.

## 8. Multinomial rejection backend

### 8.1 Algorithm

For each attempt:

1. set all \(E\) counts to zero;
2. draw \(T\) independent uniform integers in \([0,E)\);
3. increment the selected count;
4. track the number of distinct nonzero boxes;
5. accept if all \(E\) boxes are occupied.

Pseudocode:

```text
try_multinomial_rejection(T, E, max_attempts, rng):

    counts <- vector of E zeros

    for attempt in 1..=max_attempts:
        fill counts with zero
        occupied <- 0

        repeat T times:
            j <- uniform_integer(0, E-1)

            if counts[j] == 0:
                occupied += 1

            counts[j] += 1

        if occupied == E:
            return success(counts, attempt)

    return exhausted
```

### 8.2 Exactness

Before conditioning, the count vector has multinomial probability

\[
P(\mathbf{n})
=
\frac{T!}{E^T\prod_i n_i!}.
\]

Conditioning on \(n_i>0\) for every \(i\) removes only a constant normalization factor, giving the target law

\[
P(\mathbf{n}\mid n_i>0\ \forall i)
\propto
\frac{1}{\prod_i n_i!}.
\]

### 8.3 Attempt bound

Use:

```rust
const MAX_REJECTION_ATTEMPTS: usize = 20;
```

If the estimate selects rejection but 20 attempts fail, return `Exhausted` to the selector, which immediately invokes the Stirling backend.

Exhaustion is not a user-visible error.

### 8.4 Buffer reuse

The counts vector must be allocated once and reset between attempts. Use the fastest safe reset pattern supported by the existing occupation type.

An optional touched-index list can avoid clearing all \(E\) entries when \(T\ll E\), but in the admissible rejection regime \(T\) is generally not extremely small relative to \(E\). Implement the simple version first and optimize only after profiling.

### 8.5 Overflow

Every increment must be valid for `OccNum`. Since the sum is \(T\), preprocessing should ensure \(T\) is representable. Debug assertions may verify no local count overflows.

## 9. Stirling-recursion backend

### 9.1 Mathematical recurrence

The Stirling numbers of the second kind satisfy

\[
S(n,k)=kS(n-1,k)+S(n-1,k-1),
\]

with

\[
S(0,0)=1,
\]

and zero for impossible states.

A uniform partition of \(n\) labelled events into \(k\) nonempty unlabeled blocks can be built recursively. For event \(n\):

- it starts a new block with probability
  \[
  p_{\mathrm{new}}(n,k)
  =
  \frac{S(n-1,k-1)}{S(n,k)};
  \]
- it joins one of the existing \(k\) blocks with total probability
  \[
  p_{\mathrm{join}}(n,k)
  =
  \frac{kS(n-1,k)}{S(n,k)}.
  \]

When joining, choose one of the \(k\) existing blocks uniformly.

Finally, label the \(E\) blocks by assigning their sizes to the \(E\) selected support positions exchangeably.

### 9.2 Log-space dynamic program

Ordinary Stirling numbers overflow rapidly. Store

\[
\ell(n,k)=\log S(n,k).
\]

Use

\[
\ell(n,k)
=
\operatorname{logaddexp}
\left(
\log k+\ell(n-1,k),
\ell(n-1,k-1)
\right).
\]

Boundary values:

```text
logS[0][0] = 0
all impossible states = -infinity
```

A helper is:

```text
logaddexp(a, b):
    if a == -inf: return b
    if b == -inf: return a
    m = max(a, b)
    return m + log(exp(a-m) + exp(b-m))
```

### 9.3 Table shape

Only states satisfying

\[
0\le k\le\min(n,E)
\]

are required.

The simplest Phase 1 implementation stores a triangular table up to \((T,E)\), giving:

- time: \(O(TE)\);
- memory: \(O(TE)\).

This is acceptable as an exact fallback for moderate residual sizes but must be benchmarked. If Phase 1 targets very large \(T E\), use checkpointing or a more memory-efficient reverse-sampling representation. Do not silently allocate an unbounded table.

The implementation should perform checked size multiplication before allocation and return a clear resource-limit error if a table would exceed an explicit safe bound. Such a bound must be documented and benchmarked.

### 9.4 Iterative reverse construction

Avoid deep recursion. Traverse from state \((T,E)\) down to a base case while recording decisions.

A direct size-only algorithm can be implemented as follows:

```text
sample_stirling_sizes(T, E, logS, rng):

    n <- T
    k <- E
    decisions <- empty vector

    while k > 1 and n > k:
        log_new  <- logS[n-1][k-1]
        log_join <- ln(k) + logS[n-1][k]
        log_den  <- logaddexp(log_new, log_join)
        p_new    <- exp(log_new - log_den)

        if uniform_01(rng) < p_new:
            decisions.push(NewBlock)
            n <- n - 1
            k <- k - 1
        else:
            decisions.push(JoinExisting)
            n <- n - 1

    initialize base partition sizes:
        if k == 1:
            sizes = [n]
        else:
            assert n == k
            sizes = [1; k]

    replay decisions in reverse:
        if decision == NewBlock:
            sizes.push(1)
        else:
            j <- uniform_integer(0, sizes.len()-1)
            sizes[j] += 1

    shuffle sizes uniformly if construction order is retained

    return sizes
```

Care is required: when replaying a `JoinExisting` decision, the number of blocks must equal the \(k\) at that forward state. Reversing the recorded path naturally restores the correct number of current blocks if `NewBlock` appends a block before earlier joins are replayed. Unit tests must verify this invariant exhaustively for small \((T,E)\).

An alternative recursive implementation may be easier to audit but risks stack depth \(O(T)\). Prefer iterative code unless repository constraints guarantee small \(T\).

### 9.5 Branch-probability stability

Compute the new-block probability with a two-term log softmax:

```text
log_new  = logS[n-1][k-1]
log_join = ln(k) + logS[n-1][k]
log_den  = logaddexp(log_new, log_join)
p_new    = exp(log_new - log_den)
```

Clamp only tiny floating-point excursions:

```text
p_new = min(1.0, max(0.0, p_new))
```

Do not replace the branch probability with an approximation.

### 9.6 Caching

If multiple samples use the same \((T,E)\), cache the log-Stirling table through the existing generator object or benchmark setup.

Cache ownership must be explicit and thread-safe according to current MENoBiS conventions. Avoid a global unbounded cache.

## 10. Occupation selector

Pseudocode:

```text
sample_positive_me_occupations(T, E, rng):

    if E == 0:
        assert T == 0
        return []

    if E == 1:
        return [T]

    if T == E:
        return [1; E]

    estimated_rejection <- estimate_rejection(T, E)

    if estimated_rejection <= 0.8:
        result <- try_multinomial_rejection(
            T,
            E,
            MAX_REJECTION_ATTEMPTS,
            rng
        )

        if result succeeded:
            return result.counts

        return sample_stirling_sizes(T, E, rng)

    return sample_stirling_sizes(T, E, rng)
```

The selector must not alter RNG seeding or reproducibility conventions. Given the same seed, parameters, and version, the backend choice must be deterministic.

## 11. Sparse graph construction

Preallocate capacity for exactly \(E\) residual occupied pairs.

For each selected support index and positive count:

1. map the residual index to the existing pair representation;
2. insert the occupation into the sparse graph builder;
3. reject duplicate support indices as an internal bug;
4. never insert zero occupations.

After construction, assert:

\[
|\mathrm{support}|=E,
\qquad
\sum_i n_i=T,
\qquad
n_i>0.
\]

No dense adjacency or occupation matrix may be created.

## 12. Reconstruction with fixed occupations

The outer generation pipeline merges:

- fixed-positive occupations from preprocessing;
- the sampled residual sparse graph.

The pair sets must be disjoint by construction.

Final graph checks are:

\[
\sum_{ij}\mathbf{1}(t_{ij}>0)=E_{\mathrm{user}},
\]

\[
\sum_{ij}t_{ij}=T_{\mathrm{user}},
\]

and every mask/fixed-value condition is respected.

## 13. Unit tests

### 13.1 Feasibility

Test:

- \((E,T)=(0,0)\);
- \(E=0,T>0\) rejected;
- \(E>0,T=0\) rejected;
- \(T<E\) rejected;
- \(E>L\) rejected;
- fixed occupations making residual values negative;
- residual \((0,0)\) with nonempty fixed graph.

### 13.2 Support sampler

For small \(L,E\), enumerate all \(\binom{L}{E}\) supports and verify approximate uniform frequencies over many samples.

Also test:

- no duplicates;
- exact support size;
- valid index range;
- reproducibility.

### 13.3 Rejection backend

Test:

- every returned count is positive;
- sum equals \(T\);
- deterministic special cases;
- forced exhaustion using a mock RNG or tiny attempt bound;
- no allocation per attempt after initialization;
- empirical distribution against exact composition probabilities.

### 13.4 Stirling recurrence

For small values, compare computed \(S(n,k)\) or \(\log S(n,k)\) with exact integer values.

Test boundary states:

- \(S(0,0)=1\);
- \(S(n,0)=0\) for \(n>0\);
- \(S(n,n)=1\);
- \(S(n,1)=1\).

### 13.5 Stirling sampler

For small \((T,E)\), enumerate every positive composition and verify empirical probabilities proportional to

\[
\frac{1}{\prod_i n_i!}.
\]

Test:

- positivity;
- exact sum;
- exact length;
- exchangeability under coordinate permutation;
- reproducibility;
- no recursive stack overflow.

### 13.6 Backend selector

Test that:

- estimate below or equal to \(0.8\) selects rejection first;
- estimate above \(0.8\) selects Stirling directly;
- rejection exhaustion triggers Stirling fallback;
- all three routes satisfy identical invariants.

The threshold boundary must be deterministic.

## 14. Exact enumeration validation

For small \(L,E,T\), enumerate all residual configurations.

For each configuration \(\mathbf{t}\), compute

\[
w(\mathbf{t})
=
\frac{1}{\prod_{\ell:t_\ell>0}t_\ell!}.
\]

Normalize:

\[
P(\mathbf{t})
=
\frac{w(\mathbf{t})}{\sum_{\mathbf{u}\in\Omega_{L,E,T}}w(\mathbf{u})}.
\]

Draw samples from:

- rejection backend forced on;
- Stirling backend forced on;
- automatic hybrid selector.

Compare empirical and exact probabilities with statistically justified tolerances. Do not rely on a single chi-square test alone when expected cell counts are small. Aggregate states or use total variation and confidence intervals where necessary.

## 15. Conditioned grand-canonical validation

Use the existing ME grand-canonical generator.

For a small system with sufficiently likely \((E,T)\):

1. generate a large grand-canonical sample;
2. retain only configurations with the target \((E,T)\);
3. compare the conditioned empirical distribution with the Phase 1 direct sampler;
4. compare both with exact enumeration when feasible.

The identity being tested is

\[
P_{\mathrm{GC}}(\mathbf{t}\mid E,T)
=
P_{\mathrm{MC}}(\mathbf{t}\mid E,T).
\]

This validation should be marked statistically expensive and kept separate from fast unit tests.

## 16. Mask and prefilter tests

Construct cases containing:

- forbidden pairs;
- fixed zeros;
- fixed positive occupations;
- mixtures of fixed and free pairs;
- all residual pairs removed;
- residual deterministic cases;
- fixed contributions that exactly consume \(E\) and \(T\);
- fixed contributions that make the request infeasible.

Verify that the sampler sees only the residual domain and that reconstruction recovers the original constraints exactly.

## 17. Property tests

For randomly generated feasible residual problems, assert:

```text
number of sampled residual pairs == E
sum of sampled residual occupations == T
all sampled occupations > 0
all sampled pairs belong to residual admissible set
no sampled pair is duplicated
```

For reconstructed graphs, assert original constraints and mask rules.

Property tests should include edge cases near:

- \(E=1\);
- \(E=T\);
- \(E=L\);
- high \(T/E\);
- the estimated-rejection threshold.

## 18. Benchmarks

Benchmark the following components independently:

1. support sampling versus \(L\) and \(E\);
2. rejection attempt cost versus \(T\) and \(E\);
3. mean rejection attempts in accepted regimes;
4. Stirling table construction versus \(TE\);
5. Stirling draw cost after table construction;
6. end-to-end residual generation;
7. reconstruction with fixed occupations;
8. repeated samples with cached Stirling tables.

Parameter grids should include:

- \(T=E\);
- \(T\) slightly above \(E\);
- threshold-region cases;
- high-acceptance cases with \(T/E\gg1\);
- large sparse support domains with \(E\ll L\).

Report:

- wall-clock time;
- allocations;
- peak memory where available;
- selected backend;
- rejection attempts;
- table build versus draw time.

## 19. Complexity

Let \(L\) be the residual admissible-pair count.

### Support

With indexed \(O(E)\) sampling:

- expected time: \(O(E)\);
- auxiliary memory: \(O(E)\).

With reservoir sampling:

- time: \(O(L)\);
- memory: \(O(E)\).

### Rejection backend

Per attempt:

- time: \(O(T+E)\);
- memory: \(O(E)\).

Expected time:

\[
O\left(\frac{T+E}{p_{\mathrm{acc}}}\right),
\]

bounded by 20 attempts before fallback.

### Stirling backend

Simple table implementation:

- preprocessing time: \(O(TE)\);
- preprocessing memory: \(O(TE)\);
- one draw after preprocessing: \(O(T)\) time and \(O(T+E)\) temporary decision/state memory, reducible with a refined implementation.

### Graph construction

- time: \(O(E)\);
- sparse output memory: \(O(E)\).

No component may allocate \(O(N^2)\) memory.

## 20. Documentation requirements

Public documentation should state:

- the sampler is exact for the ME family under fixed \((E,T)\);
- fixed masks and occupations are handled by preprocessing;
- output constraints are satisfied exactly;
- RNG seeding follows existing MENoBiS conventions.

Internal developer documentation should explain:

- support/occupation factorization;
- surjection interpretation;
- rejection threshold \(0.8\);
- bounded fallback behavior;
- Stirling recurrence;
- cache ownership and resource limits.

## 21. Completion criteria

Phase 1 is complete only when all of the following hold:

- residual fixed-\((E,T)\) validation is integrated;
- fixed-mask prefilter and reconstruction are reused, not duplicated;
- uniform sparse support sampling is implemented;
- multinomial rejection backend is implemented;
- rejection estimate and threshold \(r=0.8\) are implemented;
- rejection retries are bounded at 20 attempts;
- exact Stirling fallback is implemented;
- automatic fallback cannot return an approximate sample;
- exact enumeration tests pass;
- forced rejection and forced Stirling distributions agree;
- conditioned grand-canonical validation passes on small systems;
- mask and fixed-occupation tests pass;
- no \(O(N^2)\) allocation is introduced;
- benchmark integration is complete;
- public and internal APIs are documented;
- existing grand-canonical benchmark behavior is preserved.

Only after these criteria are met should Phase 2 begin.
