---
description: Decision to organize Python modules by user-facing domains.
---

# 0010. Python modules follow user-facing domains

## TL;DR

Python package layout should mirror MENoBiS's public workflows: analyze, fit,
generate, filter, and data. Repository-only tools stay outside `src/menobis`.

## Context

MENoBiS's CLI is organized by workflow, but some Python modules were flat at the
package root. This made implementation details look like public domains and put
development-only benchmark entry points inside the installable package.

## Decision

- Keep `src/menobis/cli/` as thin Typer command wiring.
- Place library code behind matching domains where it improves clarity.
- Move ensemble averaging under `menobis.analysis`.
- Keep generation cohesive unless formula/provider code needs smaller units.
- Move statistical filtering into `menobis.filtering` as a package.
- Keep benchmarks in the repository `benchmarks/` package, invoked with
  `python -m benchmarks`, not as an installed `menobis` entry point.
- Do not ship graph-library interop packages for optional dependencies.

## Current Python domain map

| Domain | Package |
|---|---|
| analyze | `menobis.analysis` |
| fit/generate/partial | `menobis.models` |
| filter | `menobis.filtering` |
| data/convert | `menobis.data` |
| utilities (synthetic, logging) | `menobis.utilities` |
| CLI wiring | `menobis.cli` |
| repository benchmarks | `benchmarks` |

## Consequences

The package root stays small. Public imports remain explicit, while large
domains such as filtering can grow internal modules without changing the main
CLI shape.
