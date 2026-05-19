# Scalability

## TL;DR

ODME generation defaults to **streaming pair rates** and uses Rayon parallel
chunks for large candidate supports. It does not materialize an $N^2$
probability matrix for independent grand-canonical models, zero-inflated Poisson models,
or canonical multinomial sampling.

## Default execution model

Generation is organized as:

```text
model parameters -> pair-rate provider -> sampler/stat sink -> sparse edges
```

The provider computes candidate pairs on demand. For large supports, ODME
splits rows or sparse entries into deterministic chunks, samples chunks on all
available Rayon worker threads, and merges chunk outputs in order. The sink
stores only non-zero sampled edges or accumulates statistics.

## Memory complexity by operation

| Operation | Default memory | Notes |
|-----------|----------------|-------|
| Poisson generation | $O(E_s)$ | parallel chunks for large supports |
| zero-inflated Poisson generation | $O(E_s)$ | parallel occupation and weight draws |
| Factorized multinomial | $O(N + E_s)$ | row totals, then parallel non-empty rows |
| Sparse custom $p_{ij}$ Poisson | $O(E_p + E_s)$ | parallel sparse chunks |
| Sparse custom $p_{ij}$ multinomial | $O(E_p + E_s)$ | chunk totals, then parallel chunks |
| Dense matrix export | $O(N^2)$ | explicit opt-in only |
| Fitting multipliers | model-dependent | usually $O(N)$ plus sparse inputs |

## Existing model coverage

| ODME model | Generation strategy |
|------------|---------------------|
| Fixed-strength ME Poisson | stream $x_i y_j$ |
| Fixed-strength ME multinomial | sample source totals, then targets |
| Custom probability ME | stream supplied sparse $p_{ij}$ entries |
| Degree-events ME | stream Bernoulli occupation + conditional weight |
| Strength-cost ME | stream $x_i y_j e^{-\gamma d_{ij}}$ |
| Strength-edges ME | stream zero-inflated occupation + positive-edge Poisson |
| Strength-degree ME | stream zero-inflated occupation + positive-edge Poisson |
| Partial constraints | stream free pairs plus known rates |

## Parallelism and reproducibility

Independent Poisson and zero-inflated Poisson models are parallelized by deterministic
row chunks. Sparse custom probability inputs are parallelized by deterministic
entry chunks. Each chunk gets a seed derived from the global seed and chunk id,
so execution avoids shared RNG locks and is independent of thread scheduling.

Canonical multinomial sampling is coupled by the fixed total $T$. ODME first
samples totals for rows or sparse chunks, then samples each non-empty row/chunk
in parallel.

## Canonical multinomial without dense probabilities

A multinomial with total $T$ can be sampled by repeated binomial draws:

```text
remaining_T = T
remaining_mass = sum(rate)
for pair or group:
    count ~ Binomial(remaining_T, rate / remaining_mass)
    remaining_T -= count
    remaining_mass -= rate
```

For factorized fixed-strength models, ODME uses this idea hierarchically:

1. sample source-node event totals from row masses;
2. sample target counts within each non-empty source row.

This avoids storing all pair probabilities and still preserves exact total
$T$.

## When $N^2$ is still unavoidable

Computation over all node pairs is still $O(N^2)$ for all-pairs models. The
streaming design removes $O(N^2)$ **memory**, not necessarily $O(N^2)$ time.
Dense memory is still needed only when a user explicitly requests a full matrix
or supplies a full dense cost/probability table.

## Cost and probability inputs

Sparse custom $p_{ij}$ inputs are treated as the candidate support: only
supplied entries are sampled. Strength-cost fitting and sampling currently use
sparse cost entries with missing costs interpreted as $d_{ij}=0$; pass complete
costs unless zero-cost missing pairs are intentional.

## Large-network regression

The test suite includes generation smoke tests at `N = 1000` for:

- factorized Poisson generation;
- factorized canonical multinomial generation;
- strength-cost generation.

Run them with:

```bash
uv run pytest tests/test_odme_streaming_generation.py -q
```
