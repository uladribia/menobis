---
description: Split PyO3 binding code by Python-facing domain.
---

# 0017. PyO3 bindings are split by domain

## TL;DR

The `_menobis` PyO3 extension keeps one Python module but its Rust binding code
is split into domain files: `stats.rs`, `fitting.rs`, `generation.rs`, and
`filter.rs`.

## Context

`crates/menobis-python/src/lib.rs` had grown past 3500 lines. It mixed graph
statistics, fitting, generation, filtering, helper tuple conversion, and module
registration. That made review and follow-up changes risky.

## Decision

- Keep the public Python extension name `_menobis` unchanged.
- Keep shared imports, type aliases, `rust_core_version`, and edge validation in
  `lib.rs`.
- Move PyO3 functions into domain modules:
  - `stats.rs` for graph/statistics/clustering helpers;
  - `fitting.rs` for full and partial fitting wrappers;
  - `generation.rs` for sampling wrappers;
  - `filter.rs` for observed/absent filter wrappers.
- Use a local `add_pyfunction!` macro to reduce registration boilerplate.

## Consequences

The registration surface stays centralized while implementation code follows the
same domains as the Python package. Future changes can target a smaller file and
avoid unrelated PyO3 churn.
