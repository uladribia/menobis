# STATUS — Microcanonical fixed-(s,k): core complete, public integration pending

> **Review entry point for fixed-(s,k) work.** This is the single current
> summary.  Read this, then the current decision record and the code.
> Historical plans are archived under `archive/fixed_sk/` and are not
> live requirements.

| | |
|---|---|
| Branch | `fix/fixed-sk-direct-init-trace-gate` |
| Current decision | `docs/decisions/microcanonical-fixed-sk-extras-first-init.md` |
| Constructor | extras-first (`occupation_mcmc/fixed_degree_init.rs`) |
| Stationary kernel | capped first-return degree trace (`occupation_mcmc/fixed_degrees.rs`) |
| Exact-math oracle | `test-oracles/tests/fixed_strength_degree_enumeration.rs` |
| N=1000 gates | `test-oracles/tests/fixed_strength_degree_direct_init.rs`, `fixed_strength_degree_trace_constructed.rs`, `fixed_strength_degree_scalability.rs`, `fixed_strength_degree_e2e.rs` |
| Routing | `Constraint.STRENGTH_DEGREE` → occupation MCMC (Rust route + pyo3 + Python) |
| Feature status | **core complete and N=1000 validated; public integration implemented and tested** |

## Current verdict

The fixed-strength + fixed-directed-degree microcanonical law

\[
\pi_{s,k}(t)\propto\prod_{ij}d_F(t_{ij})
\]

(ME/B/W) is implemented end-to-end:

1. **Initialization** is combinatorial and exact: *extras-first*
   slot-aware compressed extras transport (§10–§21 of the plan), then
   occupation-1 binary completion of the missing degree slots.  This
   resolves the Gate B co-joint blocker — realistic N=1000 targets
   construct exactly, usually on the first extras attempt (0.1–1.7 s).
2. **Sampling** is the exact capped first-return trace of the
   degree-distance-biased auxiliary chain onto the degree fiber
   (unchanged since Gate A; Q/R oracle green).
3. **Public integration** is wired: Rust router (s+k/s+E/s priority),
   pyo3 binding, Python `Constraint.STRENGTH_DEGREE` route (no fit), and
   the capability registry.  The §80 routing release blocker test is in
   place (strengths can never silently route as fixed-(k,T)).

The two failed initializer families (degree-repair MCMC; support-first
exact-k + residual allocation) are **removed from live code**; their
evidence lives in the historical decision records.

## Final architecture

```text
full (s,k) target + fixed pairs
        |
        v
existing residualization (strengths/degrees/domain; B M=1 invariant)
        |
        v
r = s_out - k_out, c = s_in - k_in
        |
        v
slot-aware compressed extras transport
  (pressure(mass,slots) rows/cols, block allocation, per-node k caps,
   deterministic attempt 0 + bounded randomized retries)
        |
        v
extras support B  (support degrees <= k)
        |
        v
delta_k = k - degree(B)
        |
        v
occupation-1 filler support C on domain minus B
  (reuses the domain-aware binary exact-degree initializer)
        |
        v
t = 1 + y on B, t = 1 on C   -> exact (s,k) state, D = 0
        |
        v
exact capped first-return degree trace (burn-in / thinning)
        |
        v
merge fixed pairs
        |
        v
full exact validation
        |
        v
Rust route / pyo3 / Python (Constraint.STRENGTH_DEGREE)
```

Initialization is combinatorial and needs no detailed balance; trace
exactness is independently checked by the Q/R oracle.

## Exactness evidence

- Tiny exact `Q`/`R` transition-matrix oracle: row sums, detailed
  balance, stationarity, tiny-fiber connectivity — green.
- Production-vs-exact correspondence, fixed-(s,E) regressions — green.
- The B M=1 (Bernoulli) invariant (strength == degree per node) is
  rejected early in shared target validation, before any constructor or
  solver logic activates (independent of the sampling ensemble).

## N=1000 initialization evidence (Gate C)

`EXTRAS_FIRST_INITIALIZATION = pass` — see the decision record for the
full table.  Highlights:

| case | extras attempts | extras edges | fillers | occ-1 | wall |
|---|---:|---:|---:|---:|---:|
| ME realistic T/E=8 | 1 | 1366 | 6634 | 0.829 | 0.10 s |
| Balanced12 | 1 | 904 | 7096 | 0.887 | 0.06 s |
| B M=5 Balanced12 | 3 | 1406 | 6594 | 0.824 | 0.41 s |
| W realistic | 1 | 1366 | 6634 | 0.829 | 0.17 s |
| uniform stress grid (ME d=4/8/16, W, B) | 1–5 | — | — | — | ≤ 1.7 s |
| structural variants (loops, pos/zero fixed pairs) | 1–2 | — | — | — | ≤ 0.21 s |

## N=1000 trace/mobility evidence (Gate D)

`CONSTRUCTED_TRACE_MOBILITY = green` — the trace starts from the actual
constructor output (occ-1 ≈ 0.83), not the witness: ~61% different /
support-changing returns per top-level trace at ~1.7 `K_E` per effective
return (witness start: 3% at 34).  ME/W/B all GREEN at 100k attempts.

Degenerate at-capacity corners (B T/E=M, occ-1 = 0) remain immobile —
a documented start-state pathology (Gate A record), not an error; no
rapid mixing is claimed anywhere.

## Public integration status

- Rust: `SamplingPlan::classify` gives strengths routing priority;
  `route_occupation_mcmc` dispatches s+k → s+E → s.  §80 blocker test
  green.
- pyo3: `sample_fixed_strength_degree` (fixed pairs residualized and
  merged in Rust).
- Python: `Constraint.STRENGTH_DEGREE` microcanonical route (no fit),
  capability registry entries for ME/B/W, `sample_model_detailed`
  method/exactness mapping.
- Fast Python suite green (408 passed / 24 skipped / 0 failed); the
  new `tests/test_menobis_microcanonical_fixed_strength_degree.py`
  covers exact s+k recovery, the §80 blocker, determinism, self-loops,
  fixed pairs, B M=1 rejection, and an N=100 heavy case.

## How to verify

```bash
# fast suite (tiny extras-first gates + exact Q/R oracle + trace tests)
cargo test --workspace

# exact (s,k) oracle — the mathematical release gate
cargo test -p menobis-test-oracles --test fixed_strength_degree_enumeration
# fixed-sE regression
cargo test -p menobis-test-oracles --test fixed_strength_edges_enumeration

# N=1000 gates (release)
cargo test --release -p menobis-test-oracles \
  --test fixed_strength_degree_direct_init -- --ignored --nocapture   # Gate C
cargo test --release -p menobis-test-oracles \
  --test fixed_strength_degree_trace_constructed -- --ignored         # Gate D
cargo test --release -p menobis-test-oracles \
  --test fixed_strength_degree_scalability -- --ignored               # one-shot
cargo test --release -p menobis-test-oracles \
  --test fixed_strength_degree_e2e -- --ignored                       # E2E

# Python (after maturin develop)
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest

# static gates
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Historical decisions

- `docs/decisions/microcanonical-fixed-sk-stop.md` — the original
  degree-repair STOP (superseded; preserved evidence).
- `docs/decisions/microcanonical-fixed-sk-trace-mobility.md` — Gate A
  (still valid evidence).
- `docs/decisions/microcanonical-fixed-sk-direct-init.md` — Gate B
  support-first blocker (superseded; preserved evidence).
- `docs/decisions/microcanonical-fixed-sk-extras-first-init.md` — the
  current extras-first decision (Gate C/D + Part F/G/H evidence).

Generated fixed-(s,k) plan documents are archived under
`docs/development/agent-specifications/archive/fixed_sk/` and are
historical implementation instructions only.