# Decision — Fixed-(s,k) Trace Mobility from Exact N=1000 Witnesses (Gate A)

**Status:** complete — decision recorded, awaiting user review before Gate B
**Branch:** `fix/fixed-sk-direct-init-trace-gate`
**Base:** `feature/microcanonical-fixed-strength-degree`
**Recovery plan:** `../development/agent-specifications/MENoBiS_fixed_sk_recovery_direct_init_trace_gate.md` (Part A, §5–§12)
**Orca/diagnostics:** `menobis-test-oracles/tests/fixed_strength_degree_trace_gate.rs`

## 1. Verdict

```text
TRACE_FROM_EXACT_STATE_VIABLE = true
```

with one documented qualification: the stationary trace kernel is
practically mobile at N=1000 **when the exact start state contains a
non-trivial fraction of occupation-1 edges** (as every realistic
PA-geographic state does).  It is **not** mobile from contrived
uniform-corner witnesses with *no* occupation-1 edges (all-occupation
`c ≥ 3` patterns, all-level B): there `K_E` has no E-preserving local
moves and the trace is dead.  This is a **start-state pathology**, not a
kernel defect (the exact Q/R oracle stays green), but it constrains
Gate B: the direct constructor must not hand the trace a zero-occupation-1
corner.

## 2. Evidence summary

Witnesses: Rust port of the PA geographic generator (test-oracles
`pa_geographic.rs`; port of `src/menobis/utilities/synthetic.py`) —
N=1000, homogeneous-ish node set, **heterogeneous**
preferential-attachment support (mean degree 8, E=8000, no self-loops),
plan-mandated occupation patterns.  `λ = 1.0`, cap = 16, 100,000
top-level trace attempts, `--release`, seed: support 42 / trace 7.

| family | pattern | T/E | frac occ=1 | diff rate | support rate | timeouts | aux/diff | wall | class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ME | A1 all-1 | 1.0 | 1.000 | 9.05e-1 | 9.05e-1 | 0 | 1.1 | 65.9 s | GREEN |
| ME | A2 balanced 1/2 | 1.5 | 0.500 | 2.25e-1 | 2.25e-1 | 701 | 4.9 | 58.8 s | GREEN |
| ME | realistic PA-geo | 8.0 | 0.179 | 3.03e-2 | 3.00e-2 | 222 | 34.1 | 56.2 s | GREEN |
| W M=1 | realistic PA-geo | 8.0 | 0.179 | 3.10e-2 | 3.05e-2 | 334 | 33.9 | 55.6 s | GREEN |
| ME | A3 all-2 | 2.0 | 0.000 | 3.30e-4 | 4.0e-5 | 14 | 3037 | 48.7 s | YELLOW |
| W M=1 | all-2 | 2.0 | 0.000 | 6.90e-4 | 5.0e-5 | 46 | — | 49.9 s | YELLOW |
| ME | A4 all-3 | 3.0 | 0.000 | 4.00e-4 | 0 | 0 | 2500 | 22.5 s | RED (support=0) |
| ME | A5 all-5 | 5.0 | 0.000 | 3.80e-4 | 0 | 0 | 2632 | 22.4 s | RED (support=0) |
| ME | A6 all-10 | 10.0 | 0.000 | 5.20e-4 | 0 | 0 | 1923 | 22.4 s | RED (support=0) |
| W M=1 | all-5 | 5.0 | 0.000 | 6.60e-4 | 0 | 0 | 1515 | 22.8 s | RED (support=0) |
| B M=5 | all-1 | 1.0 | 1.000 | 9.05e-1 | 9.05e-1 | 0 | 1.1 | 66.6 s | GREEN |
| B M=5 | all-2 | 2.0 | 0.000 | 1.40e-4 | 0 | 0 | 7143 | 46.0 s | RED |
| B M=5 | all-3 | 3.0 | 0.000 | 1.30e-4 | 0 | 0 | 7692 | 22.4 s | RED |
| B M=5 | all-5 (at capacity) | 5.0 | 0.000 | 0 | 0 | 0 | ∞ | 22.4 s | RED |

Full counter detail for the key ME cases (100k attempts):

```text
A1: step1=100000 departures=0 successful=0 diff=90546 support=90546 timeouts=0
    aux=100000 accepts=100000 rejects=0 max_exc=0            aux/trace=1.00
A2: step1=99299  departures=0 successful=0 diff=22460 support=22457 timeouts=701
    aux=110515 accepts=109191 rejects=1324 max_exc=3         aux/trace=1.11
A3: step1=99986  departures=0 successful=0 diff=33    support=4     timeouts=14
    aux=100210 accepts=100183 rejects=27 max_exc=1           aux/trace=1.00
A4: step1=100000 departures=0 successful=0 diff=40    support=0     timeouts=0
    aux=100000 accepts=100000 rejects=0 max_exc=0            aux/trace=1.00
A5: step1=100000 departures=0 successful=0 diff=38    support=0     timeouts=0
    aux=100000 accepts=100000 rejects=0 max_exc=0            aux/trace=1.00
A6: step1=100000 departures=0 successful=0 diff=52    support=0     timeouts=0
    aux=100000 accepts=100000 rejects=0 max_exc=0            aux/trace=1.00
ME realistic: step1=99778 departures=0 successful=0 diff=3031 support=2999 timeouts=222
    aux=103330 accepts=102925 rejects=405 max_exc=2          aux/trace=1.03
```

## 3. What the numbers say

1. **The trace is mobile when occupation-1 edges exist.**  The realistic
   PA-geographic ME/W cases (T/E=8, frac occ=1 ≈ 0.18) return a different
   exact `(s,k)` state in ~3% of top-level attempts and a support-changed
   state in ~3%, at 34 `K_E` steps per effective return — far inside the
   §9 GREEN bands (`≥ 1e-2`, `aux/diff ≤ 100_000`).  99.8% of traces
   return at step 1 (the kernel is essentially in-fiber: `K_E` endpoints
   stay on the degree fiber), so `burin` sweeps of E=8000 proposals give
   ≈ 235 support changes per sweep.
2. **All-1 fibers are trivially mobile** (90.5% different/support returns,
   zero timeouts) for every family: every 4-cycle is a pure support
   shuffle with no occupation change.
3. **Support mobility collapses as occupation-1 edges vanish.**  From an
   all-`c` corner (`c ≥ 2`) every single-unit 4-cycle transfer keeps the
   source cells occupied (`c → c−1 > 0`) while occupying two empty
   diagonals, so the occupied count grows `m → m + 2` and the exact-E
   local kernel vetoes it (`HeldEdgeVeto`).  E-preserving local moves
   require **two occupation-1 source cells**; none exist in an all-`c`
   corner.  The residual ~4e-4 different-state rate (and the rare support
   change) comes entirely from the 5%-probability auxiliary bridge.
4. **Tuning does not help.**  The §10 grid on the A3 (all-2, YELLOW)
   case — (λ, cap) ∈ {(1,16),(0.5,16),(2,16),(1,32),(0.5,32),(1,64)} at
   50k attempts — produced **identical** rates (3.60e-4 diff, 0 support,
   same timeouts) on every configuration.  The absorption is structural
   (no E-preserving proposals), not an acceptance/temperature artefact.
5. **B is the worst case.**  B M=5 at capacity (all-5) has literally zero
   different-state returns in 100k attempts (no valid move at all); the
   all-2/all-3 fibers are ~1.4e-4 diff, zero support movement.
6. **max_excursion_distance** is small everywhere (< 5): the trace almost
   never wanders deep off the fiber before returning (or timing out at
   short depth).  The counter was found unwired on the feature branch and
   is now correctly aggregated (§8 record field).

## 4. Answers to the §38 report questions

1. *How often does the trace return to a different exact state?*
   Realistic: ~3% per attempt (ME/W).  All-1: 90.5% (trivial).  All-`c`
   corners `c ≥ 2`: ~0.04% (bridge-mediated only).
2. *How often does the endpoint support change?*  ~3% realistic; 90.5%
   all-1; ≈ 0 in all-`c` corners (A3: 4/100k; A4–A6, B: 0/100k).
3. *How many `K_E` calls per effective return?*  34 (realistic), 1.1
   (all-1), ~2e3–3e3 (corners, for the rare bridge returns).
4. *Dependence on T/E?*  Monotone in `frac(occ=1)`: 90.5% (T/E=1) →
   22.5% (1.5) → ~3% (realistic 8, occ-1 fraction 0.18) → ~0.03–0.04%
   (T/E ≥ 2 uniform).
5. *λ/cap effect?*  None measurable (§3.4) — structural absorption.
6. *Loss attribution?*  No self-loop losses (all-1/A1); no departures
   beyond the fiber in any case (`successful_returns = departures = 0`);
   losses are (a) step-1 self-loop returns for the corners (the chain
   proposes only in-fiber vetoed moves) and (b) 0.2–0.7% timeouts at
   shallow depth.
7. *Does support mobility collapse when occupation-1 edges are rare?*
   **Yes — sharply.**  At `frac(occ=1) ≈ 0` support-changed returns drop
   to 0/100k while different-state returns fall to ~1e-4.  The
   construction implication is mandatory (§5).

## 5. Consequence for Gate B (direct constructor)

The direct constructor builds `t_ij = A_ij · (1 + y_ij)` with residual
extras `y`.  Gate A shows the trace burn-in is only useful if the
constructed start has occupation-1 edges.  Therefore Gate B must:

- allocate residual extras heterogeneously so that a non-trivial fraction
  of support edges keep `y = 0` (occupation 1) — the realistic PA-geo
  allocation reaches `~0.18`; even 0.5 (balanced 1/2) gives 22% mobility;
- treat `frac(occ=1) = 0` starts as construction failures for the *burn-in
  path* (or add a preliminary mixing step), rather than assuming the
  trace will mix them;
- note that all-1 starts (T/E = 1) are the most mobile possible (90.5%),
  so low T/E is not a concern.

## 6. Exactness / regressions (unchanged)

| gate | result |
|---|---|
| exact Q/R trace-matrix oracle (`fixed_strength_degree_enumeration`) | ✅ green |
| fixed-(s,E) enumeration oracle | ✅ green |
| `cargo test --workspace` (fast) | ✅ 457 tests, 0 failures |
| clippy `-D warnings` / fmt `--check` | ✅ clean |

## 7. Commits on this branch

| commit | content |
|---|---|
| `87620ea` | `benchmark_fixed_sk_trace_from_exact_table` diagnostic + witness validation + unit tests |
| `c3dac63` | `DegreeTraceCounters.support_changed_returns` (support-topology metric) |
| `94d2b65` | PA-geographic generator + N=1000 mobility grid + `max_excursion_distance` wiring |

## 8. Remaining risks

- **Constructor risk:** a greedy/flow start with `frac(occ=1) ≈ 0` would
  be dead for the trace (§5).  Mitigation: heterogeneous allocation +
  constructor diagnostic reporting `frac(occ=1)`.
- **Trace mixing risk:** even realistic mobility (~3% per step, 34 K_E
  per support change) needs burn-in sweeps; no claim of rapid mixing at
  N=1000 (§30 wording applies).
- **Large-N runtime:** 100k attempts ≈ 60 s (release) at N=1000/E=8000;
  a full sample at N=1000 with burn-in+thinning is comfortably within
  reach; N=5000 not assessed.
- **Unproven global mixing/connectivity** on the N=1000 fiber: the exact
  oracle covers tiny fibers only.