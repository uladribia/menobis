---
description: Decision to organize Python modules by user-facing domains.
---

# 0010. Python modules follow user-facing domains

## TL;DR

Python package layout should mirror ODME's public workflows: analyze, fit,
generate, filter, and data. Repository-only tools stay outside `src/odme`.

## Context

ODME's CLI is organized by workflow, but some Python modules were flat at the
package root. This made implementation details look like public domains and put
development-only benchmark entry points inside the installable package.

## Decision

- Keep `src/odme/cli/` as thin Typer command wiring.
- Place library code behind matching domains where it improves clarity.
- Move ensemble averaging under `odme.analysis`.
- Keep generation cohesive unless formula/provider code needs smaller units.
- Move statistical filtering into `odme.filtering` as a package.
- Keep benchmarks in the repository `benchmarks/` package, invoked with
  `python -m benchmarks`, not as an installed `odme` entry point.
- Do not ship graph-library interop packages for optional dependencies.

## Current Python domain map

| Domain | Package |
|---|---|
| analyze | `odme.analysis` |
| fit/generate/partial | `odme.models` |
| filter | `odme.filtering` |
| data/convert | `odme.data` |
| utilities (synthetic, logging) | `odme.utilities` |
| CLI wiring | `odme.cli` |
| repository benchmarks | `benchmarks` |

## Consequences

The package root stays small. Public imports remain explicit, while large
domains such as filtering can grow internal modules without changing the main
CLI shape.
