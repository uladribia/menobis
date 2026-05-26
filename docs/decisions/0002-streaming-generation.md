# 0002 Streaming and parallel generation by default

## TL;DR

MENoBiS generation streams candidate pair distributions and parallelizes large
supports with deterministic Rayon chunks. Dense $N^2$ probability matrices are
to be avoided at all costs.

## Context

Maximum-entropy MENoBiS models define a distribution per candidate pair. Sampling
only needs one pair distribution at a time, so materializing all rates wastes
memory and blocks large networks. After streaming was introduced, all-pairs
work became easy to split across CPU cores.

## Decision

Use reusable generation and filtering components:

| Component | Responsibility |
|-----------|----------------|
| `PairDistribution` | Encapsulate sampling, expectation, occupation probability, and p-values |
| `PairDistributionProvider` | Compute a pair distribution on demand for a null model |
| Sampler sink | Draw weights and keep non-zero edges |
| Filter sink | Compute p-values and classify observed/absent pairs |
| Parallel chunk runner | Process row or sparse-entry chunks on Rayon |
| Canonical allocator | Allocate fixed totals by binomial splits |
| Future stats sink | Accumulate expected/sample stats without edges |

Python APIs call Rust kernels. The default path is streaming; dense matrix
construction is not a prerequisite for sampling or filtering. Custom sparse
Poisson rates and degree-events zero-inflated also use providers, so new independent null
models should implement one provider and reuse generation/filtering/stat sinks.

## Parallel strategy

| Model family | Parallel unit | Coupling handled by |
|--------------|---------------|---------------------|
| Poisson all-pairs | source-row chunks | independent draws |
| zero-inflated Poisson all-pairs | source-row chunks | independent occupation/weight draws |
| sparse custom Poisson | sparse-entry chunks | independent draws |
| factorized multinomial | non-empty source rows | source totals sampled first |
| sparse custom multinomial | sparse-entry chunks | chunk totals sampled first |
| stub_matching | not parallelized yet | stub shuffle dominates |

All-pairs generation and absent-edge filtering switch to row chunks for large
supports. Sparse custom probability generation and sparse-support absent
filtering switch to entry chunks for large sparse supports. Chunks are merged in
deterministic chunk order.

## Reproducibility

Each parallel chunk uses a seed derived from the global seed and chunk id. This
avoids shared RNG locks and makes results independent of thread scheduling.
Parallel and serial paths may produce different samples for the same seed, but
each path is reproducible for fixed code, thresholds, and Rayon configuration.


## Consequences

- Independent generation memory is $O(E_s)$ sampled non-zero edges.
- Sparse custom $p_{ij}$ inputs stay sparse and do not imply $N^2$ storage.
- All-pairs models still require $O(N^2)$ time.
- Large generators use all Rayon worker threads by default.
- Future generators and filters should implement a provider, not a dense matrix builder.


