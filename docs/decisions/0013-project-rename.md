---
description: Decision to rename the project to MENoBiS.
---

# 0013 Project rename to MENoBiS

> TL;DR: The project name is MENoBiS. Public Python imports, CLI commands,
> Rust crates, docs, and tests use `menobis`/`MENoBiS` names without
> compatibility aliases.

## Context

The modernization plan selected a final project name before a v2.0 readiness
review. MENoBiS stands for **Max Entropy NOn Binary Suite** for null
modeling. The old name was tied to the thesis-era package identity and appeared in
Python imports, Rust crate names, CLI examples, documentation, tests, and
benchmark utilities.

## Decision

Rename the codebase consistently:

| Surface | Name |
|---|---|
| Project suite name | `MENoBiS` |
| Python package | `menobis` |
| Native extension | `menobis._menobis` |
| CLI command | `menobis` |
| Rust core crate | `menobis-core` / `menobis_core` |
| Rust Python crate | `menobis-python` / `_menobis` |

No backward-compatible aliases, re-export packages, or deprecated commands are
introduced. Existing call sites, tests, and documentation move to the new names
in the same change.

## Consequences

Downstream users must update imports and commands directly. Build artifacts
from the old package name should be removed before local testing so the native
extension is rebuilt under the new module name.
