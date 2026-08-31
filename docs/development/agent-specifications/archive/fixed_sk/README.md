# Archived fixed-(s,k) implementation plans

> **Historical implementation instructions only.**
> Read `../../STATUS.md` and the current code for live requirements.

These three documents trace the fixed-strength + fixed-degree
(`(s,k)`, microcanonical ME/B/W) implementation effort. They are frozen
historical artifacts: the work they describe is complete, and the failed
approaches they prescribed are no longer in the live architecture.

## Chronology

| # | Document | What it prescribed | Outcome |
|---|---|---|---|
| 1 | `MENoBiS_fixed_sk_implementation_plan_v2.md` | Exact degree trace design; **degree-repair initializer** (reuse the degree-biased auxiliary step to walk a randomized exact-E state down to D=0) | Trace kernel exact (oracle-proven), but the repair initialization **does not scale**: D floors at ~O(N). STOP — see `docs/decisions/microcanonical-fixed-sk-stop.md` |
| 2 | `MENoBiS_fixed_sk_recovery_direct_init_trace_gate.md` | **Gate A** (trace mobility from exact witnesses — viable) + **Gate B** (support-first constructor: exact-k support → occupation 1 → residual-strength allocation) | Gate A passed. Gate B **failed at N=1000** for heterogeneous residuals — the extras transport is co-joint with the witness support and no generic k-draw carries it. See `docs/decisions/microcanonical-fixed-sk-trace-mobility.md` and `microcanonical-fixed-sk-direct-init.md` |
| 3 | `MENoBiS_fixed_sk_extras_first_completion_plan.md` | **Extras-first constructor**: allocate strength extras first (slot-aware compressed transport with per-node k caps), then complete the missing degree slots with occupation-1 fillers | **Implemented and validated** — Gate C/D/E2E pass at N=1000; routed publicly as `Constraint.STRENGTH_DEGREE`. See `docs/decisions/microcanonical-fixed-sk-extras-first-init.md` |

## What to read instead

| Need | Live document |
|---|---|
| Current fixed-(s,k) status and architecture | `../../STATUS.md` |
| Constructor design (extras-first) | `crates/menobis-core/src/generation/microcanonical/occupation_mcmc/fixed_degree_init.rs` |
| Stationary kernel | `.../occupation_mcmc/fixed_degrees.rs` (capped first-return trace) |
| Exactness oracle | `crates/menobis-test-oracles/tests/fixed_strength_degree_enumeration.rs` |
| Decision records | `docs/decisions/microcanonical-fixed-sk-*.md` |