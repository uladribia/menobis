# PLAN.md — Outstanding Work

## 1. Solver Convergence Issues

Documented in `docs/decisions/convergence-issues.md`. Summary:

| Case | Status | Proposed fix |
|---|---|---|
| W strength-edges | never converges | barrier + adaptive damping |
| W strength-degree | never converges | barrier + adaptive damping |
| W strength-cost partial | diverges at all N | conic solver boundary with q→1 |
| B strength-cost saturated N≥1000 | fails to converge within 10k iters | IPF plateau — needs convergence relaxation |
| B strength-degree sparse N=5000 | overflow in z multipliers (z~1e+262) | numerical scaling for ZI solver |

These are algorithm improvements, not missing plumbing.

## 2. Benchmark Results

Comprehensive benchmarks completed for families ME, B, and W (see
`docs/development/benchmark-results.md`).

### Cell coverage

| N | ME | B | W |
|---:|---:|---:|---:|
| 10 | ✅ all | ✅ all | ✅ all (but ✗ for edges/degree) |
| 100 | ✅ all | ✅ all | ⚠️ edges/degree fail |
| 1000 | ✅ all | ❌ cost-sat doesn't converge | ❌ only strength works |
| 5000 | ✅ all (deg sparse 28min) | ⚠️ only strength/cost/edges sparse | ❌ not run |

### B gaps at N=5000

| Cell | Problem |
|---|---|
| strength-degree sparse | Overflow in ZI multiplier (z ~ 1e+262) |
| saturated (all except strength) | Unfinished (timed out) |
| strength-cost saturated 0%/5% | Did not converge (10k iters) |

### W gaps at N=1000+
- strength-edges and strength-degree never converge at any N
- strength-cost partial fits diverge
- N=5000 not run (no point until solvers are fixed)

### Key observations

- **IPF solvers** (strength, strength-cost): single-threaded O(N²), but
  strength is trivial (<1s at N=5000). Strength-cost is the bottleneck at
  large N (45s at N=5000).
- **L-BFGS ZI solvers** (strength-edges, strength-degree): 12× parallel,
  but sparse regime can be 10× slower than saturated.
- **ME works perfectly.** B works well except for known edge cases. W needs
  solver redesign for ZI constraints.
