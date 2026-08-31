---
description: Exact microcanonical fixed-strength + fixed-degree sampling via extras-first initialization and a capped first-return degree trace.
---

# Fixed strengths + exact degrees (s, k)

**TL;DR** — Sampling with exact out/in strength sequences **and** exact
out/in binary degree sequences uses a two-part construction: (1) a
**combinatorial, extras-first initializer** that builds an exact
starting state directly on the degree fiber (`D = 0`, no MCMC, no
detailed balance needed), and (2) the existing **capped first-return
degree trace** as the exact stationary sampler — a degree-distance-biased
auxiliary chain whose proposal is the whole finished fixed-(s,E) kernel
`K_E`, traced onto the degree fiber.  Both parts are separately
validated: the initializer by N=1000 construction gates, the stationary
kernel by a tiny exact `Q`/`R` transition-matrix oracle.

## Target

With `t_ij` the integer occupation of ordered pair `(i,j)`, define

```text
s_out[i] = Σ_j t_ij        s_in[j] = Σ_i t_ij
k_out[i] = #{ j : t_ij > 0 }      k_in[j] = #{ i : t_ij > 0 }
```

The desired law conditions the family base measure `d_F` on exact
strengths and exact degrees:

```text
π_(s,k)(t) ∝ Π_ij d_F(t_ij)     subject to s_out, s_in, k_out, k_in
```

with `E_target = Σ k_out = Σ k_in`, `d_ME = 1/t!`, `d_B = C(M,t)`,
`d_W = C(M+t−1,t)` — the same degeneracies used everywhere else.

## Stationary kernel: the capped first-return degree trace

The sampler uses the identity

```text
π_(s,k) = π_(s,E)(· | k = k_target)
```

(conditioning the fixed-(s,E) law on the degree fiber).  The
**degree-distance** auxiliary target is

```text
μ_λ(x) ∝ π_(s,E)(x) exp(−λ D(x))
D(x) = ½ ( Σ_i |k_out(x)−k*_out| + Σ_j |k_in(x)−k*_in| )
```

One complete fixed-(s,E) transition `K_E` is the proposal; because `K_E`
is reversible for `π_(s,E)`, the outer degree-potential acceptance
collapses to `min(1, exp(−λ(D(y)−D(x))))` — no internal Hastings or
bridge recomputation.  The production kernel is the **capped first-return
trace** of this auxiliary chain onto the fiber `A_k = { D = 0 }`;
timeouts deterministically restore the origin (an exact self-loop).
The tiny exact `Q`/`R` oracle verifies row sums, detailed balance and
stationarity on every enumerated tiny fiber — the kernel law is exact.

## Initialization: extras-first, combinatorial and exact

Initialization needs no stationarity, reversibility, or MH rule — one
valid `D = 0` state suffices (burn-in provides the rest).  Residual
extras are `r = s_out − k_out`, `c = s_in − k_in`.

```text
slot-aware compressed extras transport
  row_slots = k_out, col_slots = k_in       (every positive extras
                                              edge costs one slot)
  pressure(mass, slots) = ceil(mass/slots)  rows/columns,
  block x = min(row_mass, col_mass, cap)    per coordinate,
  coordinate never reused;
  deterministic attempt 0 + bounded randomized retries
        |
        v
extras support B        (support degrees ≤ k elementwise)
        |
        v
delta_k = k − degree(B)                      (checked subtraction)
        |
        v
occupation-1 filler support C on domain minus B
  (reuses the domain-aware binary exact-degree initializer)
        |
        v
t = 1 + y on B, t = 1 on C      -> exact (s,k) state, D = 0
```

The extras **determine the hard row/column co-joint structure**; exact
degrees are completed afterwards.  Complexity: `O(N)` candidate scans
per extras edge (constructor hotspot at large N), `O(N + B)` memory;
never an `N×N` matrix.  Retry exhaustion is **not** mathematical
infeasibility (`ExactSkExtrasFirstExhausted`).

## Why extras-first (and not support-first)

The residual-strength transport is co-joint: it couples strength-heavy
rows to strength-heavy columns the way the observed support does.
Building an exact-`k` support *from the degree marginals only* loses that
correlation, so the residual Hall condition failed systematically at
N=1000 (see `microcanonical-fixed-sk-direct-init.md`).  Extras-first
routes the transport over the full residual domain — where it is sparse
and feasible — while enforcing the `k` support caps during the transport
via slot accounting.  The legacy degree-repair initializer (walking a
random exact-E state down to `D=0`) also floored at `D ≈ O(N)` and was
replaced; both failures are archived in decision records, not live code.

## Validation (N=1000, release)

- **Gate C (constructor)**: realistic ME, Balanced12, the uniform stress
  grid, structural variants (loops, positive/zero fixed pairs), and
  heterogeneous B/W all construct exactly — usually on the first extras
  attempt, in 0.06–1.7 s (`EXTRAS_FIRST_INITIALIZATION = pass`).
- **Gate D (mobility)**: the trace started from the *constructed* state
  (occupation-1 fraction ≈ 0.83) returns a different/support-changed
  exact state in ~61% of top-level attempts at ~1.7 `K_E` per effective
  return — far inside the engineering gate and much more mobile than the
  witness start (3% at 34 `K_E`).
- **E2E**: the one-shot sampler reproduces exact full strengths, degrees,
  and `E` for ME/W/B at N=1000 (init 0.1–1.6 s, then trace sweeps).
- **Scale**: init 0.16 s at N=1000 → 3.7 s at N=5000; memory flat
  (6 → 19 MiB, `O(E)` state).  Full tables in
  `microcanonical-fixed-sk-performance.md`.

## Fixed pairs and the B M=1 invariant

Fixed pairs are residualized once in Rust: positive fixed pairs subtract
from strengths, degrees, and the domain; zero fixed pairs only forbid the
coordinate (`CompleteMinus`, `O(F)`).  After sampling, fixed pairs are
merged and the full output validated.  A B family with `M = 1`
(Bernoulli) forces per-pair occupations in `{0,1}`, so **strength must
equal degree per node** — an ensemble-independent invariant rejected
early in shared target validation, before any constructor or solver
logic activates.

## Routing

`Constraint.STRENGTH_DEGREE` with `Ensemble.MICROCANONICAL` routes to
this backend (no fit step).  Strengths win routing priority — a
strengths+degrees problem can never silently degrade to fixed-(k,T)
(the §80 routing release blocker is tested).  Exposed through
`routing.sample_model`, `menobis.python`'s `sample_model`/`sample_model_detailed`,
the capability registry (ME/B/W), and the CLI
(`strength-degree-mcmc`) / benchmark CLI (`benchmarks micro
--constraint strength-degree`).  See
[Python API](../api/python.md) and [Generate CLI](../cli/generate.md).