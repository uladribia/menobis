---
description: Contributor reference — exact stationary MCMC for fixed strengths + exact occupied-pair count (s,E): local kernel, censored bridge, cap, mixture, and validation.
---

> **Implementation / proof detail.** This page documents the *algorithm and
> proof* of the fixed `(s,E)` microcanonical route. The model definition and
> user-facing behaviour live in
> [Microcanonical sampling](../guide/microcanonical.md); this page is for
> contributors. Historical design/recovery records are linked at the bottom.

# Microcanonical fixed strengths + exact edge count (s, E)

## Target

Let \(t_{ij}\) be the integer occupation of ordered pair \((i,j)\), and

\[
s_i^{\mathrm{out}}=\sum_j t_{ij},
\qquad
s_j^{\mathrm{in}}=\sum_i t_{ij},
\qquad
E(t)=\#\{(i,j): t_{ij}>0\}.
\]

The desired distribution conditions the family base measure \(d_F\) on the
exact strengths and edge count:

\[
\pi_{(s,E)}(t)\propto \prod_{ij}d_F(t_{ij})
\quad\text{subject to}\quad
s^{\mathrm{out}}, s^{\mathrm{in}}, E(t)=E^{\mathrm{target}},
\]

with \(d_{\mathrm{ME}}=1/t!\), \(d_{\mathrm B}=\binom Mt\),
\(d_{\mathrm W}=\binom{M+t-1}{t}\) — the same family degeneracies used
everywhere else. Exactness: **exact stationary MCMC**.

## Local exact-\(E\) kernel

The ordinary fixed-strength 4-cycle proposal (decrement \((a,b),(c,d)\);
increment \((a,d),(c,b)\)) preserves strengths by construction. The local
kernel **holds** any proposal whose destination leaves the fiber:

\[
\text{if } E(\text{proposal})\ne E^{\mathrm{target}}:\ \text{hold},
\qquad
\text{else: ordinary Metropolis–Hastings acceptance}.
\]

Conditioning multiplies all allowed states by one constant, so the local
kernel is reversible for \(\pi_{(s,E)}\). It is not always connected: the
\(N=2\), \(s^{\mathrm{out}}=s^{\mathrm{in}}=[2,2]\), \(E=2\) case has two
states connected only through an \(E=4\) intermediate.

## Auxiliary target and censored bridge

Define an auxiliary target over the full fixed-strength fiber

\[
\mu_\lambda(t)\propto \pi_s(t)\exp(-\lambda |E(t)-E^{\mathrm{target}}|)
\qquad(\lambda=1.0 \text{ internally}).
\]

On the exact-\(E\) fiber \(\mu_\lambda=\pi_{(s,E)}\). The same occupied-cell
proposal with the edge-distance potential
\(-\lambda(|E_{\mathrm{new}}-E^{\mathrm{target}}|-|E_{\mathrm{old}}-E^{\mathrm{target}}|)\)
gives an exactly reversible auxiliary chain \(K_\lambda\).

A **bridge attempt** is a censored excursion of \(K_\lambda\):

\[
x\in A \to z_1\notin A \to \cdots \to z_{k-1}\notin A \to y\in A,
\qquad
A=\{E=E^{\mathrm{target}}\}:
\]

- the first substep must depart the fiber (in-fiber moves abort and restore
  the origin);
- the first return to the fiber keeps the returned state;
- if no return occurs within the cap (`bridge_max_steps`, selected by the
  tiny-fiber connectivity oracle as `16`), every accepted substep is undone
  deterministically and the origin is restored — an exact self-loop.

## Path reversal and cap

Path reversal plus auxiliary detailed balance make the bridge reversible
for \(\pi_{(s,E)}\); summing over all capped paths gives pairwise detailed
balance. Failed attempts only add diagonal mass.

## Mixture

\[
P=(1-\rho)P_{\mathrm{local}}+\rho P_{\mathrm{bridge}}
\qquad(\rho=0.05 \text{ internally}).
\]

A constant, state-independent mixture of two reversible kernels with the
same target is reversible, so \(\pi_{(s,E)}\) is the exact stationary
distribution: the kernel law is exact; burn-in and mixing remain ordinary
MCMC concerns ([MCMC diagnostics](../performance/mcmc-diagnostics.md)).

## Validation

- **Tiny-fiber enumeration oracles** (independent ME/B/W reference
  weights, loops on/off, symmetric/heterogeneous margins, fixed-pair
  residuals) assert row sums, pairwise detailed balance, stationarity
  (\(\pi P=\pi\)), and that the mandatory \(N=2, E=2\) counterexample is
  connected — tolerances \(10^{-9}\)/\(10^{-10}\).
- The bridge cap is the **smallest passing value** from the connectivity
  grid: `16`.
- E2E recovery on generated networks checks exact strengths and exact \(E\).
- N=1000 sparse cases (ME/B/W, fixed pairs) and an N=5000 smoke run pass
  with \(O(E+F)\) memory (no dense \(N\times N\) structures; fixed-pair
  residuals use a complete-minus-fixed domain).

## Initialization repair (fixed pairs)

Fixed pairs are residualized once in Rust (strengths, domain exclusion, and
edge-target subtraction); \(E_{\mathrm{residual}}=E^{\mathrm{target}}-\#\text{positive-fixed}\).
Before MCMC, a **biased, initialization-only** repair drives a constructed
state to the exact residual \(E\) (strict-gain / 10% equal-distance /
\(\exp(-2d)\) worsening acceptance), with randomized reconstruction
restarts. An inexact-\(E\) state never enters sampling — repair exhaustion
is a structured error. The repair bias is never part of the stationary
kernel.

## Benchmark evidence

See the committed benchmark matrix
([Benchmarks](../performance/benchmarks.md)) and
`microcanonical-fixed-sk-performance.md` decision record for measured
timings and memory.

## Historical design/recovery records

- `../decisions/fixed-strength-edges-sampler.md`
- `../decisions/exact-fixed-total-v1-migration.md`
- `../decisions/microcanonical-fixed-sk-performance.md`