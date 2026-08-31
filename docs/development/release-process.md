---
description: Release process for MENoBiS.
---

# Release process

## TL;DR

MENoBiS is not yet published to PyPI or crates.io. The current release
process is local development builds with maturin, plus the existing
documentation GitHub Actions workflow.

## Development build

```bash
uv run maturin develop                # debug build
uv run maturin develop --release      # optimized build
```

## Pre-release checklist

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
| Capability tables | `uv run python scripts/docs/generate_capabilities.py --check` |

## CI/CD

A GitHub Actions workflow (`docs.yml`) builds and deploys the documentation
site to GitHub Pages on pushes to `master`. It installs dependencies, builds
the Python extension, checks `mkdocs build --strict`, and deploys.

Broader automation (tests, lint, type checks, releases) is not yet wired
into CI beyond that documentation workflow. If you add it, reuse the
existing workflow's setup steps (uv sync, maturin develop) rather than
duplicating environment configuration.

## Versioning

MENoBiS follows semantic versioning. The current public documentation
release is `1.0.1`.

## Future plans

- Publish to PyPI as `menobis` with maturin-built wheels.
- Publish `menobis-core` to crates.io for Rust-only users.
- Add benchmark regression thresholds to CI.