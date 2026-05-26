# 0001. Rust and Python architecture

## Status

Accepted.

## Context

MENoBiS needs fast numerical and graph kernels while remaining convenient for research workflows.

## Decision

MENoBiS will expose public Rust and Python endpoints.

- Rust crates contain performance-sensitive kernels and typed model logic.
- Python exposes ergonomic APIs, Polars data handling, Typer CLIs, logging, and documentation workflows.
- PyO3 and Maturin connect the two layers.

## Consequences

- Tests must cover both Rust and Python APIs.
- Public behavior should be documented in Markdown.
- Rust code should not print from library paths; Python entry points configure Loguru logging.
