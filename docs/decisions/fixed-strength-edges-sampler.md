# Decision — Exact Fixed-(s,E) Sampler: Local 4-Cycles + Censored Bridge

**Status:** implemented (feature `microcanonical-fixed-strength-edges`)
**Policy:** the conditional `(s,E)` target is sampled with an exact
stationary MCMC built entirely from the existing fixed-strength kernel —
no new composite proposal, no new proposal probabilities.

## 1. Decision

Fixed out/in strengths **plus an exact occupied-pair count** `E` are
sampled by a constant, state-independent mixture

```text
P = (1 − ρ) P_local + ρ P_bridge        ρ = 0.05
```

where `P_local` is the existing occupied-cell 4-cycle Metropolis kernel
restricted to the exact-`E` fiber (hold any proposal that would change
`E`), and `P_bridge` is a censored excursion of an auxiliary edge-biased
chain.

Rationale: restricting the existing kernel to the fiber is immediately
exact but not connected (e.g. N=2, `s_out = s_in = [2,2]`, `E = 2` has
two states reachable only through an `E = 4` intermediate).  Rather than
derive a new composite proposal with a hand-made path Hastings ratio, the
bridge reuses the already-proved auxiliary kernel `μ_λ(t) ∝ π_s(t)
exp(−λ|E(t)−E_target|)` and obtains detailed balance by path reversal
plus auxiliary reversibility.  The bias lives in the auxiliary **target**,
never in the proposal selection, so no new proposal probability is ever
derived.

## 2. Key constants (internal, not public API)

| Constant | Value | Meaning |
|---|---|---|
| `bridge_probability` ρ | 0.05 | mixture weight per outer proposal |
| `bridge_lambda` λ | 1.0 | edge-distance potential strength (performance only) |
| `bridge_max_steps` | 16 | smallest value passing the mandatory tiny-fiber connectivity grid |
| edge-repair steps per restart | 1,000,000 | safety limit (spec §13.3) |
| edge-repair restarts | 5 | randomized reconstruction budget (same RNG stream) |

Any `λ > 0` and any finite cap preserve the stationarity proof; the
values above tune efficiency only and were frozen by the exact transition
oracles.

## 3. Repair vs stationary sampling

- **Edge-count initialization repair** is biased (strict gain, 10%
  equal-distance, `exp(−2·d)` worsening) and only finds one feasible
  exact-`E` start; it is never part of the stationary kernel.
- **Sampling** uses only exact MH / exact bridge transitions.
- An inexact-`E` state never enters MCMC: repair exhaustion is a
  structured error.

## 4. Correctness evidence

- Tiny-fiber enumeration oracles (independent ME/B/W reference weights)
  assert pairwise detailed balance and stationarity for the local,
  auxiliary, bridge, and full mixed matrices (tolerance 1e-9/1e-10).
- Mandatory `N=2 s=[2,2] E=2` counterexample is connected by the bridge
  matrix and not by the local kernel.
- E2E recovery on generated networks; N=1000 ME/B/W + fixed-pair cases
  and an N=5000 smoke run keep `O(E + F)` memory (fixed-pair residuals
  use the `CompleteMinus` domain, never an `N²` set).