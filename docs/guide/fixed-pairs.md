---
description: Fixing or excluding known node pairs — residual-domain partial constraints via known_source, known_target, known_occnum.
---

# Fixed / known pairs

## TL;DR

Partial constraints freeze specific node-pair occupations before fitting or
sampling. They are a **residual-domain transformation**, not a separate
`Constraint` enum. Pass:

```python
known_source=...
known_target=...
known_occnum=...
```

to `fit_model` (partial fitting) or `sample_model` (partial microcanonical
sampling).

## What is frozen?

Each triple `(known_source[k], known_target[k], known_occnum[k])` fixes the
occupation of one ordered pair:

- `known_occnum == 0`: the pair is **excluded** from the free domain and
  contributes nothing;
- `known_occnum > 0`: the pair is **frozen with that occupation** and
  contributes to the constrained totals.

## Residual-domain transformation

The remaining (non-frozen) pairs are fitted/sampled as a residual problem.
For known occupation \(r_{ij}\) on frozen pairs \((i,j)\in Q\), the residual
out-strength of node \(i\) is

\[
s_i^{\mathrm{out,res}}
=
s_i^{\mathrm{out}}
-
\sum_{j:(i,j)\in Q}r_{ij}.
\]

Apply the equivalent residual logic to in-strength, degree, edge count,
total occupation, and cost where relevant:

- a positive fixed occupation \(r_{ij}\) contributes \(r_{ij}\) to
  strengths, \(1\) to support counts, \(r_{ij}\) to \(T\), and
  \(r_{ij}d_{ij}\) to cost;
- a zero fixed occupation contributes nothing and removes the pair from the
  admissible domain.

The frozen positive pairs are merged back into the sampled result.

## Fitting example

```python
from menobis.models import Constraint, ModelFamily, fit_model

fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=strength_out,
    strength_in=strength_in,
    known_source=known_source,   # int64/uint64 arrays
    known_target=known_target,
    known_occnum=known_occnum,
)
if not fit.converged:
    raise RuntimeError(fit.status)
```

The result is a `PartialFitResult`; the free-pair multipliers are fitted on
the residual problem. Fixed pairs whose occupation exceeds the requested
totals raise a validation error.

## Sampling example

The same keyword set drives partial microcanonical sampling:

```python
from menobis.models import Constraint, Ensemble, ModelFamily
from menobis.routing import sample_model

sample = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=strength_out,
    strength_in=strength_in,
    known_source=known_source,
    known_target=known_target,
    known_occnum=known_occnum,
)
```

The residual problem is sampled over the admissible pairs and the frozen
positive pairs are merged back, preserving the exact constrained totals.

## Feasibility

The residual constraints must remain feasible: after subtracting fixed
contributions, residual strengths/degrees must be non-negative and fit the
reduced admissible domain. The residual occupied-pair count cannot exceed
the number of remaining admissible pairs.