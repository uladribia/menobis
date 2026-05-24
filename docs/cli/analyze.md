# `odme analyze`

Network analysis workflows for ODME edge tables. Metrics exposed by ODME use
Rust-native kernels and do not require NetworkX or rustworkx.

## Directed strengths

```bash
odme analyze strengths edges.csv --output strengths.csv --node-count 100
```

The command reads a canonical weighted edge table with columns `source`, `target`, and `weight`, then writes a CSV with:

- `node`
- `strength_out`
- `strength_in`

Directed degree output is available through the Python API and will be added to the CLI when the analysis command surface stabilizes.
