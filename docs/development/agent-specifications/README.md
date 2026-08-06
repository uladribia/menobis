# Agent specifications

This directory contains implementation specifications for autonomous coding agents working on MENoBiS.

These documents are intentionally more detailed than end-user documentation. They provide:

- mathematical context and scientific invariants;
- precise code organization requirements;
- symbol-by-symbol rename tables;
- memory and performance constraints;
- benchmark and test protocols;
- exit gates and completion checklists.

## Documents

| Document | Status | Description |
|---|---|---|
| `00_intro.md` | Final | Microcanonical framework overview: ontology, notation, and project conventions |
| `01_phase1_me_fixed_et_design.md` | Final | Design specification for the ME fixed-(E,T) microcanonical sampler |
| `02_phase1_me_fixed_et_implementation.md` | Final | Implementation details and test protocol for the ME fixed-(E,T) sampler |
| `03_phase2_b_fixed_et.md` | Final | B (BinaryLayers) family fixed-(E,T) design and implementation |
| `04_phase2_w_fixed_et.md` | Final | W (Weighted/NegativeBinomial) family fixed-(E,T) design and implementation |
| `05_phase3_fixed_kt_plan.md` | Draft | Plan for fixed-degree-sequence microcanonical sampling (ME/B/W, directed) |
| `05_microcanonical_sampling_framework_fixed_se_plan.md` | Draft | Plan for the microcanonical sampling framework with fixed strength sequences |
| `06_phase4_fixed_strengths_me_b_w_final.md` | Final | Design for fixed-strength microcanonical sampling of ME, B, and W (implemented) |
| `07_phase5_fixed_strength_expected_cost_final.md` | Final | Design for fixed strengths + expected cost (gamma fitted, ME/B/W; implemented) |
| `microcanonical-phase-0.md` | Final | Foundation refactor: terminology migration, shared abstractions, generation split, filtering adaptation, and benchmark baseline before the general microcanonical engine |

## Benchmark results

- `docs/benchmarks/microcanonical_strength_cost.md` — Phase 5 strength-cost
  benchmark runs (ME/B/W, exact-strength and cost validation).

## Conventions

- Specifications follow the thesis ontology: non-binary networks, occupation numbers, ME/W/B families, grand-canonical/canonical/microcanonical ensembles.
- Each specification is self-contained and lists files to inspect, symbols to rename, invariants to preserve, and tests to add.
- Agents must read the relevant specification before making changes.