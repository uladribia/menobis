# Maximum entropy models

Maximum-entropy models generate ensembles of networks subject to selected constraints, such as strengths, degrees, or spatial costs.

## Fixed-strength multi-edge expectation

For the directed multi-edge fixed-strength model with self loops, ODME exposes the expected occupation

\[
\mathbb{E}[t_{ij}] = \frac{s_i^{out} s_j^{in}}{T}.
\]

The Python helper is:

```python
from odme.models import expected_multi_edge_weights
```
