---
description: Filter observed node pairs against a fitted null model — workflow, convergence guard, tails, and multiple-testing corrections.
---

# Filter node pairs

## TL;DR

Filtering computes, for each observed pair, how surprising its occupation is
under the fitted null, then marks pairs whose p-value is below a threshold,
applying a multiple-testing correction when chosen.

Always guard on fit convergence before filtering:

```python
fit = fit_model(...)
if not fit.converged:
    raise RuntimeError(fit.status)

result = filter_model(
    observed_edges,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    correction="fdr",
)
```

## Workflow

1. **Choose a model** — [Choose a model](choose-model.md).
2. **Fit** — `fit_model(...)` on feasible constraints.
3. **Inspect `fit.converged`** — raise if unconverged (never filter on an
   unconverged null).
4. **Choose a tail** — upper (surprisingly many events), lower (surprisingly
   few), or two-sided; the default is two-sided.
5. **Choose a correction** — `none`, `bonferroni`, or `fdr`
   (Benjamini–Hochberg). FDR is *a* sensible default for screening many
   pairs, not universally the right correction; justify the choice for your
   analysis.
6. **Filter** — `filter_model(...)`.
7. **Interpret relative to the null** — significant pairs are the ones whose
   occupation is unusual *given the null*; the null determines what "unusual"
   means.

## Why a correction matters

Filtering tests one hypothesis per observed pair — thousands of pairs means
thousands of tests. Without a correction, the expected number of false
positives at \(\alpha=0.05\) over \(m\) pairs is roughly \(0.05m\). Use
`bonferroni` to control the family-wise error rate, or `fdr` to control the
false discovery rate; document which one you used.

## Example

```python
from menobis.filtering import filter_model
from menobis.models import Constraint, ModelFamily, fit_model

fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=strength_out,
    strength_in=strength_in,
)
if not fit.converged:
    raise RuntimeError(fit.status)

result = filter_model(
    observed_edges,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    correction="fdr",
)

significant = result.upper.edges  # pairs with surprisingly many events
```

## Result structure

`FilterResult` exposes four edge buckets:

- `result.upper` — pairs significant in the upper tail;
- `result.lower` — pairs significant in the lower tail;
- `result.compatible` — pairs compatible with the null;
- `result.absent_lower` — absent-edge detections (see below).

Each bucket is a `FilteredEdges` with the edges plus per-pair
`upper_pvalue`, `lower_pvalue`, `expected`, and `occupation` under the null.

## Absent-edge detection

With `detect_absent=True`, MENoBiS also tests *absent* pairs (observed
zero-occupation pairs) whose expected occupation under the null is
surprisingly large — a missing link that should statistically be present.
Control `min_occupation`, `min_expected`, and `max_absent` to bound the
search. The mathematical definition is in
[Filtering statistics](../science/filtering.md).