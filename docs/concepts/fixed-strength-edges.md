---
description: Stationary-target argument for exact microcanonical fixed-strength + fixed-edge-count sampling.
---

# Fixed strengths + exact edge count (s, E)

**TL;DR** — Sampling with exact out/in strength sequences and an exact
number of occupied ordered pairs `E` uses an **exact stationary MCMC**: a
mixture of (1) the existing occupied-cell 4-cycle kernel restricted to the
exact-`E` fiber and (2) a censored excursion bridge that restores
connectivity.  Both sub-kernels are reversible for the same conditional
target, so the mixture is exact — no approximation enters the target law.

## Target

Let `t_ij` be the integer occupation of ordered pair `(i,j)`, and

```text
s_out[i] = Σ_j t_ij      s_in[j] = Σ_i t_ij      E(t) = #{ (i,j) : t_ij > 0 }
```

The desired distribution conditions the family base measure `d_F` on the
exact strengths and the edge count:

```text
π_(s,E)(t) ∝ Π_ij d_F(t_ij)        subject to s_out, s_in, E(t) = E_target
```

with `d_ME = 1/t!`, `d_B = C(M,t)`, `d_W = C(M+t−1,t)` (the same family
degeneracy used everywhere else; acceptance keeps reusing
`StrengthTarget::delta_log_weight`).

## Local exact-E kernel

The ordinary fixed-strength 4-cycle proposal (decrement `(a,b)`,`(c,d)`;
increment `(a,d)`,`(c,b)`) preserves strengths by construction.  The local
kernel **holds** any proposal whose destination leaves the fiber:

```text
if E(proposal) != E_target:  hold
else:  ordinary Metropolis–Hastings acceptance
```

Conditioning multiplies all allowed states by one constant, so the local
kernel is reversible for `π_(s,E)`.  It is not always connected: the
N=2, `s_out = s_in = [2,2]`, `E = 2` case has two states connected only
through an `E = 4` intermediate.

## Auxiliary bridge

Define an auxiliary target over the full fixed-strength fiber

```text
μ_λ(t) ∝ π_s(t) exp(−λ |E(t) − E_target|)      λ = 1.0 internally
```

On the exact-`E` fiber, `μ_λ` equals `π_(s,E)`.  The same occupied-cell
proposal and the same Hastings proposal ratio run against this target (only
the acceptance gains the edge-distance potential `−λ(|E_new−E_t| − |E_old−E_t|)`),
giving an exactly reversible auxiliary chain `K_λ`.

A **bridge attempt** is a censored excursion of `K_λ`:

```text
x ∈ A → z1 ∉ A → … → z_(k−1) ∉ A → y ∈ A      A = { E = E_target }
```

- the first substep must depart the fiber (held/rejected/in-fiber moves
  abort and restore the origin);
- the first return to the fiber keeps the returned state;
- if no return occurs within the cap (`bridge_max_steps`, selected by the
  tiny-fiber connectivity oracle as 16), every accepted substep is undone
  deterministically and the origin is restored — an exact self-loop.

Path reversal plus auxiliary detailed balance make the bridge reversible
for `π_(s,E)`; summing over all capped paths gives pairwise detailed
balance.  Failed attempts only add diagonal mass.

## Full kernel

```text
P = (1 − ρ) P_local + ρ P_bridge      ρ = 0.05 internally
```

A constant, state-independent mixture of two reversible kernels with the
same target is reversible, so `π_(s,E)` is the exact stationary
distribution.  This is `EXACT_STATIONARY_MCMC`: the kernel law is exact;
burn-in and mixing remain ordinary MCMC concerns.

## Validation

- **Tiny-fiber enumeration oracles** (independent ME/B/W reference
  weights, loops on/off, symmetric/heterogeneous margins, fixed-pair
  residuals) assert row sums, pairwise detailed balance, stationarity
  (`π·P = π`), and that the mandatory N=2 `E=2` counterexample is
  connected — all with tolerance `1e-9`/`1e-10`.
- The bridge cap is the **smallest passing value** from the connectivity
  grid: `16`.
- E2E recovery on generated networks checks exact strengths and exact `E`.
- N=1000 sparse cases (ME/B/W, fixed pairs) and an N=5000 smoke run pass
  with `O(E + F)` memory (no dense `N×N` structures; fixed-pair residuals
  use the `CompleteMinus` domain).

## Fixed pairs and repair

Fixed pairs are residualized once in Rust (strengths, domain exclusion,
and edge-target subtraction); `E_residual = E_target − #positive-fixed`.
Before MCMC, a **biased, initialization-only** repair drives a constructed
state to the exact residual `E` (strict-gain / 10% equal-distance /
`exp(−2·d)` worsening acceptance), with randomized reconstruction restarts.
An inexact-`E` state never enters sampling — repair exhaustion is a
structured error.  The repair bias is never part of the stationary kernel.