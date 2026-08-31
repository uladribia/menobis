---
description: Code organization, crate split, and testing responsibilities (summary; see Architecture for the module map).
---

# Development responsibilities

## TL;DR

- **Rust `menobis-core`** — all heavy computation. Minimal public API.
- **Rust `menobis-python`** — thin pyo3 bindings; no logic beyond argument
  transformation; type stubs in `_menobis.pyi`.
- **Python `src/menobis/`** — validation, orchestration, CLI, I/O.
- **Rust `menobis-test-oracles`** — exact (slow) backends for
  cross-validation. **Never imported by production code.**

The full Rust module map lives in
[Architecture](architecture.md#module-map-rust-menobis-core).

## `menobis-test-oracles` (validation)

NOT a production dependency. Contains legacy exact backends (DP, rejection,
enumeration, max-flow, greedy constructor, stub matching) and
cross-validation tests. Validates that the fast production code matches
exact results for small N. Seeds are deterministic; test sizes are bounded.

## Test separation

| Where | What | Speed |
|---|---|---|
| `menobis-core` `#[cfg(test)]` | per-function unit tests | < 1 s |
| `menobis-test-oracles/tests/` | production-vs-oracle cross-validation | moderate |
| `tests/` (no mark) | Python API contract, routing, data flow, docs contract | seconds |
| `tests/` `@pytest.mark.heavy` | E2E recovery, strength-cost, deep oracle | minutes |

`uv run pytest` for the fast suite; `uv run pytest --run-heavy` for the full
suite; `cargo test --workspace` for all Rust including oracles.

## How to extend

1. Implement the kernel in `menobis-core`.
2. Add a pyo3 binding in `menobis-python` + type stub in `_menobis.pyi`.
3. Wire into the Python routing layer (`routing.py`).
4. Test at Python level (light: API, heavy: E2E recovery).
5. Validate against the oracle crate for small N.

See [Extending MENoBiS](extending-thesis-cases.md) for the full guide,
including the contributor documentation policy.