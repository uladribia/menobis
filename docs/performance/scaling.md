---
description: Practical scaling of MENoBiS — asymptotic complexity, measured wall times with provenance, and the sparsity regimes used by the benchmark presets.
---

# Practical scaling

## TL;DR

Grand-canonical all-pairs fitting costs per-iteration \(O(N^2)\) work over
\(I\) solver iterations; sparse edge-list operations are \(O(E)\); the
microcanonical sparse state is \(O(E_{\mathrm{occ}})\). Measured wall times
are hardware- and configuration-specific — read them as "tested at", not
"supports up to".

## Asymptotic complexity

| Component | Complexity | Notes |
|---|---|---|
| Edge-list statistics | \(O(E)\) | single-pass Rust kernels |
| All-pairs GC fitting | \(O(N^2 I)\) | \(N^2\) admissible pairs per IPF sweep, \(I\) solver iterations |
| GС sampling (all pairs) | \(O(N^2)\) per draw | independent per-pair draws |
| MC sparse-state memory | \(O(E_{\mathrm{occ}})\) | occupied pairs only |
| MC strength-cost route | gamma-search MCMC | fitted \(\gamma\) loop dominates the cost |
| Filtering observed pairs | \(O(E)\) | per observed pair; absent-edge scan is optional |

Route-specific constructors and kernels follow the code; consult the
[capability registry](../guide/supported-models.md) and the
[contributor algorithm index](../development/microcanonical-algorithms.md)
for backend details.

## Sparsity regimes

Two separate axes describe a network's sparsity:

**Support sparsity** — how many of the admissible pairs are occupied.
For directed networks,

\[
\bar k = \frac{E}{N},
\qquad
\rho = \frac{E}{L},
\]

where \(L=N(N-1)\) without self loops and \(L=N^2\) with self loops.

**Occupation intensity conditional on support** —

\[
\bar t_+ = \frac{T}{E}.
\]

Do not call a network "dense" merely because \(T/E\) is high: the two axes
are independent.

The benchmark presets use:

| Preset | `average_degree` | `events_per_edge` | Character |
|---|---|---|---|
| sparse | \(3.0\) | \(3.0\) | low support density, low occupation intensity |
| dense | \(N/5\) | \(8.0\) | moderate support density, unsaturated, high occupation intensity |
| saturated | \(0.85\,(N-1)\) | \(8.0\) | support near the no-self-loop bound |

These are the generator parameters used by `benchmarks/cli.py`; when a
legacy label "sparse"/"dense" appears in benchmark tables, these parameters
are what it means.

## Empirical wall times

Every timing table must include provenance:

- date and git SHA;
- CPU model, physical/logical cores, RAM, OS;
- build profile (release), Python version;
- self-loop policy, sparsity regime, \(N\), family, constraint;
- sampling settings (burn-in sweeps, sweeps per sample, sample count);
- seed, wall time, peak RSS.

Use "**tested at**" wording — measured on stated hardware for a stated
commit; not universal limits. See
[Benchmarks](benchmarks.md) for the metadata schema and how results are
collected.