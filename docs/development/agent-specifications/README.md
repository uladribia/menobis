# Agent specifications

This directory contains implementation specifications for autonomous coding agents working on MENoBiS.

> ⚠️ **Historical.** These documents specify the microcanonical refactor,
> which is now **COMPLETE** (phases A–H + §35 benchmark matrix merged into
> `microcanonical-refactor`). The per-phase design/implementation specs are
> archived under `archive/` for reference; the authoritative live spec is
> `microcanonical_implementation/11_microcanonical_refactor_practical_final.md`.
> Docs might be obsolete — always refer to code in case of doubt!

These documents are intentionally more detailed than end-user documentation. They provide:

- mathematical context and scientific invariants;
- precise code organization requirements;
- symbol-by-symbol rename tables;
- memory and performance constraints;
- benchmark and test protocols;
- exit gates and completion checklists.

## Documents

### Current fixed-(s,k) work (STOPPED / recovery)

| Document | Status | Description |
|---|---|---|
| `STATUS.md` | **Entry point** | Fixed-(s,k) review / recovery status: feature branch STOP at degree repair, recovery branch `fix/fixed-sk-direct-init-trace-gate`, current Gate A/B outcomes |
| `MENoBiS_fixed_sk_implementation_plan_v2.md` | Historical spec | Master implementation plan for fixed strengths + fixed degrees (the old degree-repair initialization that does not scale) |
| `MENoBiS_fixed_sk_recovery_direct_init_trace_gate.md` | **Recovery plan** | Supersedes the initialization part: Gate A (trace mobility from exact witnesses) + Gate B (direct exact-(s,k) constructor). Outcomes in `docs/decisions/microcanonical-fixed-sk-trace-mobility.md` (viable) and `docs/decisions/microcanonical-fixed-sk-direct-init.md` (N=1000 heterogeneous blocked) |

### Microcanonical refactor (COMPLETE)

| Document | Status | Description |
|---|---|---|
| `microcanonical_implementation/11_microcanonical_refactor_practical_final.md` | **Authoritative** | Practical final specification for the microcanonical refactor (phases A–H, §35 benchmark matrix, §40 completion gate) |
| `archive/00_intro.md` | Archived | Microcanonical framework overview: ontology, notation, and project conventions |
| `archive/01_phase1_me_fixed_et_design.md` | Archived | Design specification for the ME fixed-(E,T) microcanonical sampler |
| `archive/02_phase1_me_fixed_et_implementation.md` | Archived | Implementation details and test protocol for the ME fixed-(E,T) sampler |
| `archive/03_phase2_b_fixed_et.md` | Archived | B (BinaryLayers) family fixed-(E,T) design and implementation |
| `archive/04_phase2_w_fixed_et.md` | Archived | W (Weighted/NegativeBinomial) family fixed-(E,T) design and implementation |
| `archive/05_phase3_fixed_kt_plan.md` | Archived | Plan for fixed-degree-sequence microcanonical sampling (ME/B/W, directed) |
| `archive/05_microcanonical_sampling_framework_fixed_se_plan.md` | Archived | Plan for the microcanonical sampling framework with fixed strength sequences |
| `archive/06_phase4_fixed_strengths_me_b_w_final.md` | Archived | Design for fixed-strength microcanonical sampling of ME, B, and W (implemented) |
| `archive/07_phase5_fixed_strength_expected_cost_final.md` | Archived | Design for fixed strengths + expected cost (gamma fitted, ME/B/W; implemented) |
| `archive/08_scalable_fixed_total_gibbs_migration_final.md` | Archived | Scalable fixed-(E,T)/fixed-(k,T) pair-Gibbs migration design (implemented) |
| `archive/09_MENoBiS_microcanonical_refactor_plan.md` | Archived | Overall refactor plan (superseded by the practical final spec) |
| `archive/microcanonical-phase-0.md` | Archived | Foundation refactor: terminology migration, shared abstractions, generation split, filtering adaptation, and benchmark baseline |
| `archive/12_microcanonical_fixed_strength_edges_master_implementation_plan.md` | Archived | Master implementation plan for exact microcanonical fixed strengths + exact edge count (ME/B/W; completed on `feature/microcanonical-fixed-strength-edges`) |
| `archive/README.md` | Index | Explains why the historical specs are archived and what to read instead |

## Benchmark results

- `docs/benchmarks/microcanonical_strength_cost.md` — Phase 5 strength-cost
  benchmark runs (ME/B/W, exact-strength and cost validation).
- §35 benchmark matrix results are recorded in `_workspace/` evidence files
  and `docs/benchmarks/`.

## Conventions

- Specifications follow the thesis ontology: non-binary networks, occupation numbers, ME/W/B families, grand-canonical/canonical/microcanonical ensembles.
- Each specification is self-contained and lists files to inspect, symbols to rename, invariants to preserve, and tests to add.
- Agents must read the relevant specification before making changes.
