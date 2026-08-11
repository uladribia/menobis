---
description: Code organization, crate split, and testing responsibilities.
---

# Development responsibilities

## TL;DR

- **Rust `menobis-core`** — all heavy computation. Minimal public API.
- **Rust `menobis-python`** — thin pyo3 bindings (~100 functions).
- **Python `src/menobis/`** — validation, orchestration, CLI, I/O.
- **Rust `menobis-test-oracles`** — exact (slow) backends for cross‑validation.
  Never imported by production code.

## Crate map

### `menobis-core` (production)

| Module | Responsibility |
|---|---|
| `generation/microcanonical/occupation_mcmc/` | Fixed‑strength MCMC + compressed constructor + repair |
| `generation/microcanonical/conditional/` | Fixed‑total pair‑Gibbs chain (shared by E,T and k,T) |
| `generation/microcanonical/binary/` | Binary degree‑support sampling |
| `generation/microcanonical/support/` | Uniform edge‑support sampling |
| `generation/microcanonical/mcmc/` | Shared MCMC config, counters |
| `generation/grandcanonical/` | Independent‑pair sampling (fitted multipliers) |
| `fitting/` | Lagrange‑multiplier solvers (IPF, L‑BFGS, bisection) |
| `pairs.rs` | Pair‑cost and pair‑distribution providers |

### `menobis-test-oracles` (validation)

NOT a production dependency. Contains legacy exact backends (DP, rejection,
enumeration, max‑flow, greedy constructor, stub matching) and cross‑validation
tests. Validates that the fast production code matches exact results for
small N. Seeds are deterministic. Test sizes are bounded.

### `menobis-python` (bindings)

~100 pyo3 functions bridging Python to Rust. Each thin wrapper: unwrap
arguments → call `menobis-core` → wrap result. No logic beyond argument
transformation. Type stubs in `_menobis.pyi`.

## Test separation

| Where | What | Speed |
|---|---|---|
| `menobis-core` `#[cfg(test)]` | Per‑function unit tests | < 1 s |
| `menobis-test-oracles/tests/` | Production‑vs‑oracle cross‑validation | Moderate (5–10 s) |
| `tests/` (no mark) | Python API contract, routing, data flow | ~4 s |
| `tests/` `@pytest.mark.heavy` | E2E recovery, strength‑cost, deep oracle | 30–300 s |

`uv run pytest` for fast suite; `uv run pytest --run-heavy` for full.
`cargo test --workspace` for all Rust including oracles.

## How to extend

1. Implement the kernel in `menobis-core`.
2. Add a pyo3 binding in `menobis-python` + type stub in `_menobis.pyi`.
3. Wire into the Python routing layer (`routing.py`).
4. Test at Python level (light: API, heavy: E2E recovery).
5. Validate against the oracle crate for small N.

See `docs/development/extending-thesis-cases.md` for the full guide.