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
| `microcanonical-phase-0.md` | Final | Foundation refactor: terminology migration, shared abstractions, generation split, filtering adaptation, and benchmark baseline before the general microcanonical engine |

## Conventions

- Specifications follow the thesis ontology: non-binary networks, occupation numbers, ME/W/B families, grand-canonical/canonical/microcanonical ensembles.
- Each specification is self-contained and lists files to inspect, symbols to rename, invariants to preserve, and tests to add.
- Agents must read the relevant specification before making changes.