# Decision — Fixed-(s,k) Extras-First Constructor: N=1000 Gate C PASS

**Status:** complete — gate C passed; Gate D (trace mobility from the
constructed state, plan §42–§44) and production integration (§45–§48)
remain before public exposure.
**Branch:** `fix/fixed-sk-direct-init-trace-gate`
**Plan:** `../development/agent-specifications/MENoBiS_fixed_sk_extras_first_completion_plan.md`
(Part D, §36–§41)
**Prior decisions:** `microcanonical-fixed-sk-stop.md` (degree-repair
STOP), `microcanonical-fixed-sk-trace-mobility.md` (Gate A),
`microcanonical-fixed-sk-direct-init.md` (Gate B support-first blocker)
**Gate tests:** `menobis-test-oracles/tests/fixed_strength_degree_direct_init.rs`

## 1. Verdict

```text
EXTRAS_FIRST_INITIALIZATION = pass
```

The extras-first constructor resolves the Gate B co-joint blocker on the
very first attempt for realistic heterogeneous instances: it allocates
the strength extras first (slot-aware compressed transport with per-node
`k` caps) and completes the missing degree slots with occupation-1
fillers afterwards. All N=1000 constructor gates are green (§3).

```text
commit:        e612851  (plus the Gate C test rewrite in the next commit)
test command:  cargo test --release -p menobis-test-oracles \
                 --test fixed_strength_degree_direct_init -- --ignored --nocapture
family:        ME (C1/C2/C3/C4), W, B M=5 (§40)
N:             1000
E:             8000 (mean degree 8) / 4000 (d=4) / 16000 (d=16)
T/E:           1…10 (uniform grid), 1.5 (Balanced12), 8 (realistic)
exactness:     exact s_out, s_in, k_out, k_in, E, domain, D=0,
               B capacity, fixed-pair exclusion (independent verification)
```

## 2. Constructor architecture (as implemented)

```text
r = s_out − k_out, c = s_in − k_in            (checked; B M=1 invariant)
   -> slot-aware compressed extras transport:
        row_mass/col_mass + row_slots=k_out/col_slots=k_in
        pressure(mass,slots) = ceil(mass/slots)
        rows: pressure↓, mass↓, slots↑, index
        cols: same + bounded randomized top-window on attempts > 0
        block x = min(row_mass, col_mass, edge_cap) per coordinate
        coordinate never reused; stranded mass fails the attempt (§11)
   -> extras support degrees ≤ k
   -> delta = k − extras support degrees
   -> occupation-1 binary completion on domain − extras support
        (reuses greedy_directed_initialize_with_admissibility)
   -> t = 1 + y on extras, t = 1 on fillers
   -> independent table validation + StrengthState post-validation
```

Retry config (§20): `max_extras_attempts = 64`,
`max_completion_attempts = 16`, `randomized_top_window = 8`.
Per-edge extra capacity: ME/W `residual_total` (never binds before an
endpoint exhausts); B `M−1`.

## 3. N=1000 evidence

### C1/C2 — heterogeneous ME (d=8, loopless)

| case | extras attempts | extras edges | filler edges | completion attempts | occ-1 fraction | wall |
|---|---:|---:|---:|---:|---:|---:|
| realistic PA-geo (T/E=8, residual 56000) | 1 | 1366 | 6634 | 1 | 0.829 | 0.10 s |
| Balanced12 (T/E=1.5, residual 4000) | 1 | 904 | 7096 | 1 | 0.887 | 0.06 s |

The gate-b evidence (0/400 k-greedy supports, best partial flow
55 980/56 000) is superseded: the slot-aware transport found a feasible
extras table on attempt 0.

### C3 — uniform stress grid (all green)

| family | cases | worst extras attempts | worst wall |
|---|---|---|---|
| ME d∈{4,8,16} × T/E∈{1,2,5,10} | 12 | 1 | 0.10 s |
| W d=8 × {1,2,5} | 3 | 1 | 0.08 s |
| B M=5 d=8 × {1,2,3,5} | 4 | 5 (T/E=3) | 1.68 s (T/E=3) |

B capacity corners retry more (T/E=3: 5 attempts, 4 failures; at-capacity
T/E=5: 3 attempts, occ-1 fraction 0.0) but all succeed exactly. `occ-1 =
0` at full capacity is the documented Gate A start-state mobility
pathology (§33: not a constructor rejection criterion).

### C4 — structural variants (d=8, realistic T/E)

| variant | extras attempts | extras edges | filler edges | wall |
|---|---|---:|---:|---:|---:|
| loops enabled | 1 | 1341 | 6659 | 0.15 s |
| CompleteMinus + 1333 positive fixed pairs | 1 | 1295 | 5372 | 0.13 s |
| CompleteMinus + 1000 zero fixed pairs | 2 | 1320 | 6680 | 0.21 s |

### §40 — heterogeneous B/W

| family | extras attempts | extras edges | filler edges | occ-1 | wall |
|---|---|---:|---:|---:|---:|---:|
| B M=5 Balanced12 (T/E=1.5) | 3 | 1406 | 6594 | 0.824 | 0.41 s |
| W realistic (T/E=8) | 1 | 1366 | 6634 | 0.829 | 0.17 s |

## 4. Why it works (vs. support-first)

- The extras transport is **co-joint**: it needs the row–column strength
  correlation that only the witness-like support carries.  Support-first
  construction draws a support from the `k` marginals only and loses that
  correlation, so the residual Hall condition fails systematically
  (Gate B record).
- Extras-first keeps the transport on the full residual domain — where it
  is sparse and feasible (§2.4 evidence: ~1752 positive edges, one column
  over by 1 on the *unconstrained* version) — and enforces the `k` support
  caps **during** the transport via the slot invariant rather than after
  the support is fixed.
- The dominant occupation-1 fraction of the constructed state (~0.83)
  is *higher* than the witness (0.18), which the Gate A record identifies
  as the mobility-friendly regime for the stationary trace.

## 5. Complexity

- Extras transport: `O(N)` candidate scans per extras edge; `O(N+B)`
  memory; never an `N×N` structure (§4, §18, §35).
- Completion: reuses the binary initializer (`O(N·d)` per attempt).
- Observed N=1000 wall times: 0.06–1.7 s per construction (release).

## 6. Remaining work before public exposure

- **Gate D** (plan §42–§44): trace mobility benchmark started from the
  *constructed* state (not a witness) for ME/W/B at N=1000.
- **Part F**: make extras-first the sole active `initialize_exact_sk`;
  remove the old construction stages from the fixed-(s,k) production path
  in `chain.rs`.
- **Part G**: N=1000 end-to-end gates + fixed pairs + exact oracles.
- **Part H**: Rust route / pyo3 / Python wiring + capability (only after
  C/D/E2E pass).  Per user decision, this session stops after Part G.
- **Parts I–J**: cleanup of the legacy support-first machinery and
  documentation/status rewrite (after all gates).

## 7. Reproduction

```bash
cargo test -p menobis-core fixed_degree_init               # tiny gates: green
cargo test --release -p menobis-test-oracles \
  --test fixed_strength_degree_direct_init -- --ignored --nocapture
  #   n1000_extras_first_initialization      -> C1/C2 green
  #   n1000_constructor_stress_grid          -> C3 green
  #   n1000_structural_variants              -> C4 green
  #   n1000_extras_first_heterogeneous_b_w   -> §40 green
cargo test --workspace                                 # green (ignored skipped)
```