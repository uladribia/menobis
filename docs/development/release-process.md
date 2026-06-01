---
description: Release process for MENoBiS.
---

# Release process

## TL;DR

MENoBiS is not yet published to PyPI or crates.io. The current release process is
local development builds with maturin.

## Development build

```bash
uv run maturin develop          # debug build
uv run maturin develop --release  # optimized build
```

## Pre-release checklist

Before any future release:

| Check | Command |
|-------|---------|
| Rust format | `cargo fmt --all -- --check` |
| Rust lint | `cargo clippy --workspace --all-targets -- -D warnings` |
| Rust tests | `cargo test --workspace` |
| Python lint | `uv run ruff check .` |
| Python format | `uv run ruff format --check .` |
| Type check | `uv run ty check` |
| Python tests | `uv run pytest` |
| Docs build | `uv run mkdocs build --strict` |

## Versioning

MENoBiS follows semantic versioning. The current public documentation release is `1.0.0`.

## Future plans

- Publish to PyPI as `menobis` with maturin-built wheels.
- Publish `menobis-core` to crates.io for Rust-only users.
- Add CI/CD pipeline with GitHub Actions.
- Add benchmarking regression thresholds to CI.
