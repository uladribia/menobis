---
description: Decision to keep external graph libraries outside MENoBiS dependencies.
---

# 0009. External graph libraries are optional adapters

## TL;DR

MENoBiS does not depend on NetworkX, rustworkx, or similar graph libraries. Core
network statistics stay in Rust; users may convert `EdgeTable` data to external
libraries in their own projects.

## Context

MENoBiS models weighted origin-destination multi-edge networks. The supported
statistics required by fitting, filtering, and validation are already available
through Rust kernels exposed to Python.

Adding a general graph library as a runtime dependency would expand the supply
chain and couple MENoBiS's installability to APIs that are not part of the model
implementation.

## Decision

- Keep MENoBiS runtime dependencies limited to libraries needed by MENoBiS itself.
- Keep graph-statistics kernels in `menobis-core` when they are part of MENoBiS's
  supported scientific workflow.
- Do not ship built-in adapters for optional graph libraries.
- Document loose recipes for users who want NetworkX, rustworkx, or Rust graph
  crates downstream.

## Consequences

| Area | Consequence |
|---|---|
| Runtime install | Smaller dependency surface |
| Python API | No graph-library-specific adapter functions |
| Rust API | `menobis-core` remains the supported metric implementation boundary |
| User extension | External graph tooling is possible through local conversion code |

## Future change policy

A graph library may be reconsidered only if MENoBiS adopts a metric that cannot be
maintained well in `menobis-core`. That change must include benchmarks, tests,
documentation, and a new decision record.
