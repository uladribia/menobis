# Scalability Refactor Specification — Fixed (E,T) and Fixed (k,T)

**Repository:** `uladribia/menobis`  
**Version:** August 2026  
**Objective:** Replace the large rejection/DP occupation backends with a scalable shared pair-Gibbs sampler, validate it against the current exact implementations, then move the exact DP/reference algorithms into `menobis-test-oracles` and remove them from production `menobis-core`.

---

# 1. Decision

Yes, this migration strategy makes sense.

The current exact backends still have value as:

1. correctness oracles;
2. reproducibility references.

After the new Gibbs backend is validated:

- Gibbs replaces the production role;
- exact enumeration retains the small-system oracle role;
- the exact DP implementations move to `menobis-test-oracles` as reference
  science and medium-scale validation oracles;
- git commit history preserves the original production code (no separate
  archive branch or tag is created).

Keeping the large DP and low-acceptance rejection machinery indefinitely in
production `menobis-core` would create unnecessary maintenance, testing,
documentation, and API burden.

---

# 2. Target architecture

Keep the current decomposition:

```text
fixed (E,T)
    existing uniform backbone sampler
    +
    shared fixed-total occupation sampler

fixed (k,T)
    existing fixed-degree backbone sampler
    +
    shared fixed-total occupation sampler
```

Replace the current production occupation backend with a two-edge Gibbs chain.

For a fixed backbone containing `E` pairs, sample positive occupations

\[
t_e \ge 1,
\qquad
\sum_{e=1}^{E} t_e = T.
\]

For B also require

\[
t_e \le M.
\]

The family targets are

\[
\pi_{\mathrm{ME}}(\mathbf t)
\propto
\prod_e \frac{1}{t_e!},
\]

\[
\pi_{\mathrm B}(\mathbf t)
\propto
\prod_e \binom{M}{t_e},
\]

and

\[
\pi_{\mathrm W}(\mathbf t)
\propto
\prod_e \binom{M+t_e-1}{t_e}.
\]

---

# 3. New shared module

Add:

```text
generation/microcanonical/
    fixed_total/
        mod.rs
        state.rs
        initializer.rs
        pair_conditional.rs
        chain.rs
        diagnostics.rs
        errors.rs
```

Reuse:

- `OccupationFamily`;
- `McmcConfig`;
- `McmcCounters`;
- RNG conventions;
- burn-in and thinning conventions;
- persistent-chain patterns.

Do not introduce a new generic MCMC framework.

Use a concrete state:

```rust
pub struct FixedTotalState {
    occupations: Vec<OccNum>,
    total: OccNum,
}
```

Memory is

\[
O(E).
\]

---

# 4. Feasible initializer

Do not generate zeros and repair them.

Create a valid positive vector directly.

## ME and W

Start from

\[
t_e=1.
\]

Let

\[
R=T-E.
\]

Use a balanced linear initializer:

\[
q=\left\lfloor \frac{R}{E}
\right
floor,
\qquad
r=R \bmod E.
\]

Set every cell to

\[
1+q,
\]

add one extra unit to `r` randomly selected cells, then shuffle.

## B

Start from one per edge and distribute `R=T-E` while respecting residual capacity `M-1`.

Use a random permutation of edge indices and fill cells in bounded chunks.

The initializer is allowed to be biased because it is never emitted before burn-in.

---

# 5. Pair-Gibbs update

Choose two distinct edge indices uniformly:

\[
a
e b.
\]

Let

\[
q=t_a+t_b.
\]

Resample

\[
t_a'=k,
\qquad
t_b'=q-k
\]

from

\[
P_F(k\mid q)
\propto
d_F(k)d_F(q-k),
\]

subject to positivity and family bounds.

This update:

- preserves total `T`;
- preserves all `E` occupied pairs;
- preserves B capacity;
- has acceptance probability one;
- satisfies detailed balance exactly;
- is aperiodic because the current split may be redrawn.

## ME

\[
k \sim \operatorname{Binomial}(q,1/2)
\]

conditioned on

\[
1\le k\le q-1.
\]

## B

\[
k \sim \operatorname{Hypergeometric}(2M,M,q)
\]

conditioned on

\[
\max(1,q-M)
\le k\le
\min(M,q-1).
\]

## W

\[
k \sim \operatorname{BetaBinomial}(q,M,M)
\]

conditioned on

\[
1\le k\le q-1.
\]

Generate W through

\[
p\sim\operatorname{Beta}(M,M),
\qquad
k\sim\operatorname{Binomial}(q,p).
\]

For `M=1`, sample uniformly from `1..q-1`.

---

# 6. Integration

## Fixed (E,T)

Keep:

- validation;
- fixed-pair residualization;
- admissible-pair enumeration;
- uniform backbone sampling;
- output assembly.

Replace only the positive-occupation backend.

## Fixed (k,T)

Keep:

- degree validation;
- fixed-degree support MCMC;
- masks;
- fixed-pair residualization;
- output assembly.

Continue calling the same shared fixed-total occupation entry point.

No occupation logic should remain duplicated in `fixed_kt`.

---

# 7. Transitional backend modes

During migration expose:

```rust
pub enum FixedTotalBackend {
    ExactLegacy,
    Gibbs,
    Compare,
}
```

## `ExactLegacy`

Runs the current rejection/DP implementation.

## `Gibbs`

Runs the new scalable backend.

## `Compare`

Runs both backends for validation and reports diagnostics.

`Compare` must be restricted to tests and benchmark tooling.

---

# 8. Validation strategy

The new implementation must be validated against the current exact methods before the legacy implementation is removed from production `menobis-core`.

Validation has three levels.

---

# 9. Level 1 — Exact enumeration

For very small fibers, enumerate every valid occupation vector and normalize the exact family weights.

Recommended limits:

- `E <= 8`;
- `T <= 20`;
- small B/W layer counts.

Compare Gibbs empirical state probabilities using:

- total variation distance;
- maximum absolute probability error;
- chi-square where expected counts are adequate;
- KL divergence with safe zero handling.

Required cases:

- ME, B, and W;
- `T=E`;
- `E=1`;
- B near saturation;
- W with `M=1`;
- highly imbalanced valid compositions.

Suggested gate:

\[
\mathrm{TV}<0.02
\]

or a tighter threshold justified by sample size.

---

# 10. Level 2 — Comparison against current exact methods

For larger cases where exact enumeration is impossible but the current rejection/DP backend still runs, compare distributions of observables.

The validation target is synthetic network sizes up to

\[
N=100.
\]

Since occupation difficulty depends on `E`, `T`, and `M`, not only `N`, every report must include all of them.

For each case:

1. produce independent samples with the current exact backend;
2. produce Gibbs samples after burn-in and thinning;
3. compare identical observables.

Required observables:

\[
\max_e t_e,
\]

\[
\sum_e t_e^2,
\]

occupation variance,

occupation histogram,

fraction with occupation one,

quantiles,

occupation entropy,

family log-degeneracy.

For fixed `(k,T)`, verify that support statistics remain unchanged because both methods use the same support sampler.

---

# 11. Validation matrix up to N=100

Test approximately:

```text
N = 10, 25, 50, 100
```

## Fixed (E,T)

Use sparse and moderate-density backbones, for example:

```text
E ≈ 2N
E ≈ 5N
one moderate-density case
```

Occupation regimes:

```text
T/E ≈ 1.1
T/E ≈ 2
T/E ≈ 10
```

For B additionally test:

```text
T/(ME) ≈ 0.2
T/(ME) ≈ 0.5
T/(ME) ≈ 0.9
```

For W test:

```text
M = 1, 2, 5, 20
```

## Fixed (k,T)

Use:

- regular degrees;
- heterogeneous degrees;
- power-law-like degrees;
- hub-dominated degrees;
- loopless cases;
- masked cases where the exact backend remains practical.

Where possible, compare both occupation backends on the same sampled support to isolate occupation-law differences.

---

# 12. Statistical comparison protocol

Use:

- at least 10,000 exact samples for small and medium cases where practical;
- enough Gibbs samples to obtain comparable Monte Carlo error;
- at least three independent Gibbs chains;
- dispersed initial states.

Compare:

- means;
- variances;
- empirical CDFs;
- quantiles;
- confidence intervals;
- effect sizes.

A two-sample KS test may be included for scalar observables, but its p-value must not be the sole acceptance criterion.

Primary observable confidence intervals must overlap within the configured tolerance.

---

# 13. Mixing validation

Run Gibbs chains initialized from:

1. balanced allocation;
2. maximally concentrated valid allocation;
3. randomized valid allocation.

Track:

\[
\sum_e t_e^2,
\]

maximum occupation,

fraction of occupation-one cells,

family log weight.

Measure:

- integrated autocorrelation time;
- effective sample size;
- between-chain variance;
- overlap of stationary histograms.

Use these measurements to set default burn-in and thinning.

Do not choose defaults only from runtime.

---

# 14. Correctness gate

The Gibbs backend is ready only when:

1. hard positivity and total are always exact;
2. B capacity is always exact;
3. pair-conditional PMFs match their closed forms;
4. detailed balance tests pass;
5. exact enumeration agrees;
6. exact-backend observable comparisons agree through `N=100`;
7. dispersed chains converge to overlapping distributions;
8. fixed `(k,T)` support statistics are unchanged;
9. fixed-pair and mask regressions pass;
10. reproducibility tests pass.

---

# 15. Performance gate

Measure both backends.

Required metrics:

- wall-clock time;
- peak memory;
- initialization time;
- burn-in time;
- time per sample;
- pair updates per second;
- effective samples per second;
- DP allocation avoided;
- rejection attempts avoided.

Required outcome:

- normal production uses `O(E)` occupation memory;
- no `O(ET)` table is allocated;
- no `O(ET^2)` fallback is used;
- W cases that currently fail or hang finish predictably;
- fixed `(E,T)` and fixed `(k,T)` improve materially at `N=100`.

Performance must be evaluated using effective sample size, not raw proposal speed alone.

---

# 16. Benchmark CLI

During migration support:

```text
python -m benchmarks micro     --constraint edges-events     --occupation-backend legacy|gibbs|compare
```

and:

```text
python -m benchmarks micro     --constraint degree-events     --occupation-backend legacy|gibbs|compare
```

Include options for:

- family;
- `N`;
- `E`;
- `T`;
- layers;
- seed;
- burn-in;
- thinning;
- sample count;
- report output.

The final report must contain exact commands, commit SHA, parameters, timings, ESS, memory, and statistical comparison results.

---

# 17. Migration phases

## A. Parallel implementation

Add Gibbs while keeping legacy exact code in production.

Default remains legacy initially.

## B. Validation

Run:

- unit tests;
- detailed balance;
- exact enumeration;
- exact-backend comparisons;
- validation matrix through `N=100`;
- mixing diagnostics;
- fixed `(k,T)` regressions;
- performance benchmarks.

## C. Switch default

After all correctness gates pass:

```text
Auto -> Gibbs
```

Keep explicit `ExactLegacy` temporarily and rerun CI plus benchmarks.

## D. Move exact implementation to the oracle crate

The exact DP/rejection implementations are **not deleted** — they move
permanently into `menobis-test-oracles` as reference algorithms and
medium-scale validation oracles:

- Stirling-table ME sampler;
- bounded/unbounded DP samplers for B and W;
- their tests;
- the migration validation report.

No separate archive branch or tag is created: git commit history already
preserves the old production code for historical reproducibility.

## E. Remove legacy code from production

After Gibbs has been the default and passed the final gate, remove both the legacy sampler and all migration-only comparison plumbing.

Remove:

- DP table implementations;
- scaled rejection fallback;
- rejection-estimation routing used only by legacy code;
- legacy-only constants;
- obsolete error variants;
- the migration-only legacy/compare benchmark harness and flags;
- duplicated fallback tests;
- outdated documentation.

Keep trivial deterministic exact cases:

- `E=0, T=0`;
- `E=1`;
- `T=E`;
- B saturation:
  \[
  T=ME.
  \]

These are simple and should remain in production.

---

# 18. Final production layout

```text
fixed_et/
    validation
    support sampling
    deterministic cases
    shared fixed-total Gibbs call
    output assembly

fixed_kt/
    degree-support MCMC
    shared fixed-total Gibbs call
    output assembly

fixed_total/
    state
    initializer
    pair conditionals
    Gibbs chain
    diagnostics

mcmc/
    config
    counters
    outcomes
```

No large DP tables.

No whole-allocation rejection fallback.

No duplicated family occupation samplers.

---

# 19. Final completion gate

The migration is complete only when:

- shared pair Gibbs is implemented;
- all family conditionals are tested;
- detailed balance passes;
- exact enumeration passes;
- comparison against current exact methods passes through `N=100`;
- multiple-chain mixing checks pass;
- fixed `(E,T)` integration passes;
- fixed `(k,T)` integration passes;
- the temporary migration benchmark harness is executed;
- measured scaling improvement is reported;
- Gibbs becomes the default;
- Gibbs is the only production backend;
- exact DP/reference code is moved to `menobis-test-oracles`;
- dead constants, errors, tests, and docs are cleaned up;
- deterministic special cases remain;
- final CI and benchmark reports are committed.

---

# 20. Recommended sequence

```text
implement Gibbs
    ↓
validate against exact methods
    ↓
switch default to Gibbs
    ↓
move exact DP/reference code to oracle crate
    ↓
remove fallback machinery from production
```

This preserves scientific confidence while keeping the production code scalable and maintainable.
