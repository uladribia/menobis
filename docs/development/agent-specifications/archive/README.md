# Archived microcanonical specifications

> TL;DR — Historical design and implementation specifications from the
> **completed** microcanonical refactor (phases A–H + §35 benchmark matrix,
> all merged into `microcanonical-refactor`). Retained for git history and
> reference only. Do not treat these as live requirements — refer to the
> authoritative spec instead.

## Why these are archived

These documents are the historical design/implementation specs produced and
consumed **during** the microcanonical refactor. The refactor is complete;
the documents describe work that has been merged and superseded. They are
kept under version control so that:

- the design rationale and symbol tables remain traceable in git history;
- future contributors can reconstruct *why* the current architecture looks
  the way it does;
- the per-phase plans (Phase 1–5, scalability, refactor plan) document the
  evolution of the sampling backends.

## What to read instead

| Need | Live document |
|---|---|
| Current microcanonical architecture and requirements | `../microcanonical_implementation/11_microcanonical_refactor_practical_final.md` (authoritative) |
| Implementation status | `_workspace/refactor-state.md` (repo root), `../baseline-sha.txt` |
| Backlog and deferred work | `../../todos.md` |

## Archived documents

| Document | Phase | Summary |
|---|---|---|
| `00_intro.md` | Framework | Microcanonical framework overview: ontology, notation, project conventions |
| `01_phase1_me_fixed_et_design.md` | Phase 1 | Design for the ME fixed-(E,T) microcanonical sampler |
| `02_phase1_me_fixed_et_implementation.md` | Phase 1 | Implementation details and test protocol for the ME fixed-(E,T) sampler |
| `03_phase2_b_fixed_et.md` | Phase 2 | B (BinaryLayers) family fixed-(E,T) design and implementation |
| `04_phase2_w_fixed_et.md` | Phase 2 | W (Weighted/NegativeBinomial) family fixed-(E,T) design and implementation |
| `05_phase3_fixed_kt_plan.md` | Phase 3 | Plan for fixed-degree-sequence microcanonical sampling (ME/B/W, directed) |
| `05_microcanonical_sampling_framework_fixed_se_plan.md` | Phase 3 | Plan for the microcanonical sampling framework with fixed strength sequences |
| `06_phase4_fixed_strengths_me_b_w_final.md` | Phase 4 | Design for fixed-strength microcanonical sampling of ME, B, and W |
| `07_phase5_fixed_strength_expected_cost_final.md` | Phase 5 | Design for fixed strengths + expected cost (gamma fitted, ME/B/W) |
| `08_scalable_fixed_total_gibbs_migration_final.md` | Phase 6 | Scalable fixed-(E,T)/fixed-(k,T) pair-Gibbs migration design |
| `09_MENoBiS_microcanonical_refactor_plan.md` | Pre-refactor | Overall refactor plan (superseded by the practical final spec) |
| `microcanonical-phase-0.md` | Phase 0 | Foundation refactor: terminology migration, shared abstractions, generation split, filtering adaptation, benchmark baseline |
| `12_microcanonical_fixed_strength_edges_master_implementation_plan.md` | Fixed (s,E) | Master implementation plan for exact microcanonical fixed strengths + exact edge count (ME/B/W; local 4-cycle + censored-bridge MCMC) — completed on `feature/microcanonical-fixed-strength-edges` |

## Notes

- The documents are intentionally **not** updated in place; they are frozen
  historical artifacts.
- If a section still matches current code, treat it as incidental — the code
  is the source of truth.
