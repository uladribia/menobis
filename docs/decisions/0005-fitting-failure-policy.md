# 0005 General fitting failure policy

## TL;DR

All fitting APIs should expose failures consistently. Feasibility errors are
rejected at the Python boundary; solver failures return typed diagnostics or
raise only through an explicit policy, not through one-off custom behavior.

## Context

ODME has several fitting families: ME, W, B, and partial constraints. Some use
IPF, some use scalar root finding, and W strength-family fits use Newton or
coordinate/root solves. Each solver can fail differently, but users need one
mental model.

## Decision

Use a shared failure policy for all fitting wrappers:

| Stage | Behavior |
|-------|----------|
| invalid shape/type | raise `ValueError` before native code |
| non-finite constraints | raise `ValueError` before native code |
| negative constraints | raise `ValueError` before native code |
| unbalanced in/out totals | raise `ValueError` before native code |
| explicit infeasibility | raise `ValueError` when cheaply known |
| iteration/solver failure | return result with diagnostics and warn |
| future strict mode | optionally raise on non-convergence |

Fit result types should carry common diagnostics: `converged`, `status`,
`iterations`, residuals where available, and solver-specific nested diagnostics.
W solver metrics belong under nested diagnostics rather than bespoke result
classes.

## Consequences

- Failure handling is not custom per model except for model-specific feasibility
  predicates.
- CLI commands can map all fitting errors to consistent messages and exit codes.
- Tests should cover shared failure classes across at least two families when a
  validation rule applies broadly.
- Partial fits should eventually follow the same policy as masked versions of
  their corresponding full constraints.

## Validation

Add tests for common policy rules, for example non-finite inputs and
strength-edges capacity failures across Poisson and W families. Solver-specific
stress tests should assert explicit non-convergence diagnostics rather than
silent bad fits.
