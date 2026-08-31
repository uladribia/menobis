# Decision — Fixed-(s,k) scale and memory benchmark (N=100…5000)

**Status:** recorded — end-to-end performance evidence for the thesis
experiments.
**Branch:** `fix/fixed-sk-direct-init-trace-gate`
**Benchmark:** `crates/menobis-test-oracles/tests/fixed_strength_degree_benchmark.rs`
**Sampler:** production one-shot `sample_fixed_strength_degree` (extras-first
constructor + exact capped first-return degree trace).
**Related:** `microcanonical-fixed-sk-extras-first-init.md` (Gate C/D/E2E
correctness evidence).

## 1. Purpose and method

Measure the **actual production route** (not a witness bench) on realistic
PA-geographic instances (d=8, T/E=8) over N ∈ {100, 500, 1000, 2000, 5000},
plus W and B M=5 at N=1000. Constraints are derived from the witness
generator but the witness table is never passed to the sampler (same
protocol as Gate C/D).

Budget: `burn_in_sweeps = 3`, `sweeps_per_sample = 1`,
`proposals_per_sweep = E` → **4 full E-sweeps** per case (a top-level
`degree_trace_step` ≈ one proposal). Per-sweep mobility and `K_E`/support
are budget-independent; wall times scale linearly with the sweep count.
Environment: single measurement, `cargo test --release -- --ignored`,
seed 7, Linux.  Peak RSS via `VmHWM`.  All cases: exact s/k/E verified.

## 2. ME scale sweep (T/E = 8, loopless, seed 7)

| N | E | init (s) | extras attempts/edges | fillers | occ-1 | mcmc (s, 4 sweeps) | support/sweep | aux/support | ms/trace | RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 800 | 0.002 | 1 / 148 | 652 | 0.815 | 0.18 | 298 | 4.05 | 0.06 | 6 MiB |
| 500 | 4000 | 0.024 | 1 / 675 | 3325 | 0.831 | 6.19 | 2367 | 1.87 | 0.39 | 7 MiB |
| 1000 | 8000 | 0.162 | 1 / 1366 | 6634 | 0.829 | 79.5 | 4907 | 1.72 | 2.48 | 8 MiB |
| 2000 | 16000 | 0.546 | 1 / 2575 | 13425 | 0.839 | 285.7 | 10384 | 1.58 | 4.46 | 10 MiB |
| 5000 | 40000 | 3.739 | 1 / 6953 | 33047 | 0.826 | 1911 | 25604 | 1.58 | 11.9 | 19 MiB |

## 3. Family coverage at N=1000

| family | init (s) | extras attempts | occ-1 | mcmc (s) | support/sweep | aux/support | RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| ME | 0.162 | 1 | 0.829 | 79.5 | 4907 | 1.72 | 8 MiB |
| W M=1 | 0.156 | 1 | 0.829 | 34.8 | 4895 | 1.73 | 7 MiB |
| B M=5 (Balanced12) | 0.456 | 3 | 0.824 | 75.2 | 4865 | 1.74 | 10 MiB |

## 4. Interpretation

1. **Initialization is cheap and exact.** 0.16 s at N=1000, 3.7 s at
   N=5000, always on the first extras attempt (B retries twice).  The
   constructor hotspot is the `O(N)` candidate scans per extras edge
   (documented in §18 of the plan as the expected one-time cost): the
   measured init growth is ≈ quadratic in N (0.002 → 3.7 s over 50× in
   N), consistent with `O(N × E)`-ish scan cost at `E = 8N`.
2. **Mobility stays strong at scale.** Support-changing returns per
   sweep *grow* with N (298 @ N=100 → 25 604 @ N=5000) and `K_E` per
   support change *falls* (4.05 → 1.58): the constructed states'
   occupation-1 fraction ~0.83 keeps the trace in its most mobile
   regime (Gate D evidence).  No mixing collapse at N=5000.
3. **MCMC wall ≈ proposals × per-step cost, and per-step cost grows
   with N** (0.06 ms @ N=100 → 11.9 ms @ N=5000 per top-level trace).
   A full burn-in (10–50 sweeps) at N=5000 is therefore expensive
   (≈ 1–8 h at the measured per-step cost); the numbers above are for
   4 sweeps (≈ 32 min at N=5000).  Practical guidance for thesis
   experiments: N ≤ 2000 with a few sweeps is comfortable; N=5000 use
   the minimum sweep budget that the exactness/mobility target needs
   (exactness requires 0 sweeps; mixing scales with sweeps).
4. **Memory is flat.** Peak RSS 6 → 19 MiB from N=100 to N=5000
   (E = 40 000): the sparse `O(E)` state and `O(N+B)` constructor
   (never an `N²` matrix) are confirmed at scale.

## 5. Design choices this validates

- Extras-first constructor: sub-second at N=1000, minutes only for the
  MCMC, never a blocker.
- Bounded retry config (§20 defaults) untouched: realistic cases need
  attempt 1; B capacity corners retry ≤ 5 (Gate C).
- The trace is the cost center, not initialization — future
  performance work should target per-`K_E` step cost (recordable
  cycle undo, degree-delta metadata) rather than the constructor.

## 6. Reproduction

```bash
cargo test --release -p menobis-test-oracles \
  --test fixed_strength_degree_benchmark -- --ignored --nocapture
#   fixed_sk_scale_sweep_me        (ME, N = 100..5000)
#   fixed_sk_family_sweep_n1000    (W, B M=5)
```

Timings are single-run (not criterion-median); absolute values vary
with machine load, scaling ratios and the mobility diagnostics are the
stable evidence.