---
description: Contributor reference — exact microcanonical fixed-strength + fixed-degree (s,k) sampling via extras-first initialization and a capped first-return degree trace.
---

> **Implementation / proof detail.** This page documents the *algorithm and
> proof* of the fixed `(s,k)` microcanonical route. The model definition and
> user-facing behaviour live in
> [Microcanonical sampling](../guide/microcanonical.md); this page is for
> contributors. Historical design/recovery records are linked at the bottom.

# Microcanonical fixed strengths + exact degrees (s, k)

## Target

With \(t_{ij}\) the integer occupation of ordered pair \((i,j)\),

\[
s_i^{\mathrm{out}}=\sum_j t_{ij},
\qquad
s_j^{\mathrm{in}}=\sum_i t_{ij},
\qquad
k_i^{\mathrm{out}}=\#\{j:t_{ij}>0\},
\qquad
k_j^{\mathrm{in}}=\#\{i:t_{ij}>0\}.
\]

The desired law conditions the family base measure \(d_F\) on exact
strengths and exact degrees:

\[
\pi_{(s,k)}(t)\propto \prod_{ij}d_F(t_{ij})
\quad\text{subject to}\quad
s^{\mathrm{out}}, s^{\mathrm{in}}, k^{\mathrm{out}}, k^{\mathrm{in}},
\]

with \(E^{\mathrm{target}}=\sum k^{\mathrm{out}}=\sum k^{\mathrm{in}}\),
\(d_{\mathrm{ME}}=1/t!\), \(d_{\mathrm B}=\binom Mt\),
\(d_{\mathrm W}=\binom{M+t-1}{t}\) — the same degeneracies used elsewhere.

## Stationary kernel: capped first-return degree trace

The sampler uses the identity

\[
\pi_{(s,k)}=\pi_{(s,E)}(\cdot \mid k=k^{\mathrm{target}}),
\]

conditioning the fixed-\((s,E)\) law on the degree fiber. The
**degree-distance** auxiliary target is

\[
\mu_\lambda(x)\propto \pi_{(s,E)}(x)\exp(-\lambda D(x)),
\]

\[
D(x)=\tfrac12\left(\sum_i|k_i^{\mathrm{out}}(x)-k_i^{\mathrm{out},\star}|
+\sum_j|k_j^{\mathrm{in}}(x)-k_j^{\mathrm{in},\star}|\right).
\]

One complete fixed-\((s,E)\) transition \(K_E\) is the proposal; because
\(K_E\) is reversible for \(\pi_{(s,E)}\), the outer degree-potential
acceptance collapses to

\[
\min\left(1,\exp(-\lambda(D(y)-D(x)))\right),
\]

with no internal Hastings or bridge recomputation. The production kernel is
the **capped first-return trace** of this auxiliary chain onto the fiber
\(A_k=\{D=0\}\); timeouts deterministically restore the origin (an exact
self-loop). The tiny exact \(Q/R\) transition-matrix oracle verifies row
sums, detailed balance and stationarity on every enumerated tiny fiber —
the kernel law is exact.

## Initialization: extras-first, combinatorial and exact

Initialization needs no stationarity, reversibility, or Metropolis rule —
one valid \(D=0\) state suffices (burn-in provides the rest). Residual
extras are \(r=s^{\mathrm{out}}-k^{\mathrm{out}}\), \(c=s^{\mathrm{in}}-k^{\mathrm{in}}\).

```text
slot-aware compressed extras transport
  row_slots = k_out, col_slots = k_in   (every positive extras
                                         edge costs one slot)
  pressure(mass, slots) = ceil(mass/slots) rows/columns,
  block x = min(row_mass, col_mass, cap) per coordinate,
  coordinate never reused;
  deterministic attempt 0 + bounded randomized retries
        |
        v
extras support B        (support degrees <= k elementwise)
        |
        v
delta_k = k - degree(B)                  (checked subtraction)
        |
        v
occupation-1 filler support C on domain minus B
        |
        v
t = 1 + y on B,  t = 1 on C  ->  exact (s,k) state, D = 0
```

The extras **determine the hard row/column co-joint structure**; exact
degrees are completed afterwards. Complexity: \(O(N)\) candidate scans per
extras edge (constructor hotspot at large \(N\)), \(O(N+B)\) memory; never
an \(N\times N\) matrix. Retry exhaustion is **not** mathematical
infeasibility.

## Why extras-first (and not support-first)

The residual-strength transport is co-joint: it couples strength-heavy rows
to strength-heavy columns the way the observed support does. Building an
exact-\(k\) support *from the degree marginals only* loses that correlation,
so the residual Hall condition failed systematically at N=1000. Extras-first
routes the transport over the full residual domain — where it is sparse and
feasible — while enforcing the \(k\) support caps during the transport via
slot accounting. The legacy degree-repair initializer (walking a random
exact-\(E\) state down to \(D=0\)) also floored at \(D\approx O(N)\) and was
replaced; both failures are archived in decision records, not live code.

## Validation (N=1000, release)

- **Constructor gate**: realistic ME, balanced, uniform stress grid,
  structural variants (loops, positive/zero fixed pairs), and heterogeneous
  B/W all construct exactly — usually on the first extras attempt, in
  0.06–1.7 s.
- **Mobility gate**: the trace started from the *constructed* state
  (occupation-1 fraction ≈ 0.83) returns a different/support-changed exact
  state in ≈ 61% of top-level attempts at ≈ 1.7 \(K_E\) per effective
  return.
- **E2E**: the one-shot sampler reproduces exact full strengths, degrees,
  and \(E\) for ME/W/B at N=1000.
- **Scale**: init 0.16 s at N=1000 → 3.7 s at N=5000; memory flat
  (6 → 19 MiB, \(O(E)\) state). Full tables in
  `microcanonical-fixed-sk-performance.md`.

## Fixed pairs and the B \(M=1\) invariant

Fixed pairs are residualized once in Rust: positive fixed pairs subtract
from strengths, degrees, and the domain; zero fixed pairs only forbid the
coordinate. After sampling, fixed pairs are merged and the full output
validated. A B family with \(M=1\) (Bernoulli) forces per-pair occupations
in \(\{0,1\}\), so **strength must equal degree per node** — an
ensemble-independent mathematical invariant rejected early in shared target
validation.

## Routing

`Constraint.STRENGTH_DEGREE` with `Ensemble.MICROCANONICAL` routes to this
backend (no fit step). Strengths win routing priority — a strengths+degrees
problem can never silently degrade to fixed-\((k,T)\). Exposed through
`sample_model`/`sample_model_detailed`, the capability registry (ME/B/W),
and the CLI (`strength-degree-mcmc`) / benchmark CLI.

## Historical design/recovery records

- `../decisions/microcanonical-fixed-sk-direct-init.md`
- `../decisions/microcanonical-fixed-sk-extras-first-init.md`
- `../decisions/microcanonical-fixed-sk-trace-mobility.md`
- `../decisions/microcanonical-fixed-sk-performance.md`
- `../decisions/microcanonical-fixed-sk-stop.md`