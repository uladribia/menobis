---
description: Maximum-entropy model constraints implemented by ODME.
---

# Maximum entropy models

## TL;DR

ODME currently implements directed fixed-strength multi-edge fitting and directed
fixed-degree binary fitting. Both return Lagrange multipliers `x` and `y`.

## Implemented models

| Model | Constraint | Expectation/probability | Python API |
|-------|------------|-------------------------|------------|
| Fixed-strength ME | `s_out`, `s_in` | `E[t_ij] = x_i y_j` | `fit_fixed_strength_me` |
| Fixed-degree binary | `k_out`, `k_in` | `p_ij = x_i y_j / (1 + x_i y_j)` | `fit_fixed_degree_binary` |
| Strength-degree ZIP | `s_out`, `s_in`, `k_out`, `k_in` | `E[t_ij] = p_ij(1 + λ_ij)` | `fit_strength_degree_zip` |

## Strength-degree invariant

For weighted integer networks, every node must satisfy:

\[
s_i^{out} \ge k_i^{out}, \quad s_i^{in} \ge k_i^{in}.
\]

Each positive edge contributes at least one unit of strength, so ODME rejects
fractional weights at the data boundary and validates this constraint before
future coupled strength-degree model fitting.

```python
from odme.models import validate_strength_degree_constraints

validate_strength_degree_constraints(s_out, s_in, k_out, k_in)
```

## Fixed-strength multi-edge model

For directed multi-edge networks with self loops, the analytical solution is:

\[
E[t_{ij}] = \frac{s_i^{out} s_j^{in}}{T}.
\]

ODME returns multipliers where `x_i = s_i^out / sqrt(T)` and
`y_j = s_j^in / sqrt(T)`, so `E[t_ij] = x_i y_j`.

```python
import numpy as np
from odme.models import fit_fixed_strength_me

s_out = np.array([10, 20, 30])
s_in = np.array([15, 25, 20])
fit = fit_fixed_strength_me(s_out, s_in)
```

## Fixed-degree binary model

For directed binary networks, ODME fits expected degrees using:

\[
p_{ij} = \frac{x_i y_j}{1 + x_i y_j}.
\]

```python
import numpy as np
from odme.models import fit_fixed_degree_binary

k_out = np.array([0.8, 1.2, 1.0])
k_in = np.array([1.1, 0.9, 1.0])
fit = fit_fixed_degree_binary(k_out, k_in)
```

Set `self_loops=False` to constrain only `i != j` probabilities.

## Strength-degree zero-inflated shifted-Poisson model

For the grand-canonical fixed-strength-and-degree case currently implemented,
ODME uses a hurdle/zero-inflated shifted-Poisson distribution:

\[
P(t_{ij} > 0) = p_{ij}, \quad t_{ij} | t_{ij}>0 = 1 + Pois(\lambda_{ij}).
\]

The expected weight is:

\[
E[t_{ij}] = p_{ij}(1 + \lambda_{ij}), \quad
\lambda_{ij} = a_i b_j.
\]

The binary part fits degrees. The excess part fits `s - k`, which enforces the
integer-weight invariant `s_i >= k_i`.

```python
import numpy as np
from odme.models import fit_strength_degree_zip, sample_strength_degree_zip

s_out = np.array([2.0, 3.5, 2.5])
s_in = np.array([2.7, 2.4, 2.9])
k_out = np.array([0.8, 1.2, 1.0])
k_in = np.array([1.1, 0.9, 1.0])

fit = fit_strength_degree_zip(s_out, s_in, k_out, k_in)
sample = sample_strength_degree_zip(fit, seed=42)
```
