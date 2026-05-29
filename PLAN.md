# MENoBiS modernization plan

Scientific reference: <https://hdl.handle.net/10803/400560>.


## Benchmark CLI

```bash
uv run python -m benchmarks all --nodes 100,500
uv run python -m benchmarks all --nodes 100 --self-loops
uv run python -m benchmarks fit --nodes 500 --families me,w
```

Pipeline: PA geographic generate → fit → ensemble sample-check → null-filter FPR.

## Next steps (priority order)
6. Implement fixed strength + edges follwoing L-BFGS inspired ME and B cases fixed strength degrees. Only for ME and B.
7. Ensure benchmark includes known pairs 5%-20% + masked partial fits are exposed in python via fit_model / sample_model / filter_model also. Also, benchmarks should be unique, either in python or rust, clarify what is happening there!
8. Write tutorials and notebooks with real-world examples + HOWTO on two+one cases: (1) filter a network with a null model (2) assess if some network statistic is relevant under a nullmodel (3) just fit and generate null model instances to do whatever the user needs. Use the PA synth model as realistic example. Do a complete audit of the docs to minimize overlap, maximize clarity and alignment, and refer to the thesis when needed. The docs should read cleary in the README using a mermaid diagram as follows (the notebook should follow these ideas).
    1. Choose a case based on your data:
        - Aggregated binary networks: Binomial. E.g. airlines connecting airports.
        - Integer weights are distinguishable events: ME case. E.g. trips.
        - Integer wegihts are undistinguishable events: W/NB case. E.g. total euros in milions of trade networks, global or per commodity (different layers).
    2. Select if you want to support self loops or not.
    3. Choose a constraint you want to check against. The constraint must be expressed as a linear combination of occupatin numbers or binary occupation events.
        - Is it supported by the package? (fixed strength, fixed strength plus cost, fixed total events and degrees, fixed total binary edges and strengths, fixed degrees and strengthsl)? Use appropriate fitter.
        - Is it not?
            -- Check the thesis, it might be solved there analytically or computationally. If so, implement it as a new fitter.
    4. Choose an ensemble:
        - Microcanonical is only supported for ifxed strengths and ME with self loops.
        - Canonical is only supported for ME (all cases).
        - Grandcanonical covers the rest of cases.
    Now your case is set: family, constraint, self-loop, ensemble.
    5. Choose use-case:
        1. Filter network: use the filter CLI, after having fitted the model.
        2. Generate networks: use the sampling CLI, after having fitted the model.
        3. Assess statistical relevance of high order network features: Use the sampling CLI to generate null models and ensemble the expectations. You might need to impemnet the magnitude you are interested in on the analysis package (or connect with existing packages like networkx or rustworkx).
9. Publish MkDocs site with updated docs.
10. **Investigate strength-edges and strength-degree solver performance** —
    The benchmark CLI now supports all constraint types (`strength`, `strength-cost`,
    `strength-edges`, `strength-degree`) for all families (ME, B, W). However initial
    runs reveal severe performance/convergence issues that must be diagnosed:

    **Strength-edges (total binary edges constraint):**
    - ME: Very slow even at N=100 with `self_loops=False` (~1.8s for N=100).
      At N=1000 it does not complete in reasonable time. The zero-inflated
      Poisson coupled solver (IPF + lambda search) needs profiling.
    - B: Known slow at N≥300 (~30s). Zero-inflated binomial IPF needs acceleration.
    - W: Status unknown — needs benchmarking.

    **Strength-degree (full degree sequence constraint):**
    - ME: Does not complete at N=50 within 60s with `self_loops=True`.
      The coupled zero-inflated Poisson strength-degree solver appears to have
      convergence issues or extremely slow per-iteration cost.
    - B: Known non-convergent at N≥300.
    - W: Status unknown — needs benchmarking.

    **Action items:**
    1. Profile the ME strength-edges solver at N=100 to identify the bottleneck
       (is it iteration count, per-iteration cost, or lambda line search?).
    2. Profile the ME strength-degree solver at N=50 — same questions.
    3. Check if the Rust implementation uses dense N×N operations where sparse
       would suffice (the mask is only diagonal exclusion for no-self-loops).
    4. Consider adaptive damping, Anderson acceleration, or SQUAREM for the
       coupled IPF loops.
    5. Benchmark W strength-edges and W strength-degree to complete the picture.
    6. Document acceptable N ranges per family×constraint in `docs/decisions/`.
    7. Add timeout guards to the benchmark CLI to avoid hanging.
    8. **ME strength-degree EM convergence fix**: The bisection-based solver was
       replaced with multiplicative ratio-correction (EM-style IPF). This is
       ~100× cheaper per iteration but does not converge for the degree (z/w)
       multipliers because the sigmoid occupation function saturates. The ratio
       correction gets stuck when `z*w*G(q) >> 1`. Options:
       - Newton step in log-space for z/w (uses gradient `sum p*(1-p)`) — fast
         convergence but breaks the pure-balancing philosophy.
       - Better initialization of z/w to the correct scale (~0.3–0.6 for
         typical inputs) so the sigmoid stays in the linear regime.
       - Hybrid: ratio-correct x/y (works well), Newton for z/w.
    9. **Implement proper peeling for strength-degree (all families)**:
       Currently only B-strength and B-degree use `peel_degree_saturation`.
       All strength-degree solvers (ME, B, W) use a crude "pin z/w=1e30" hack
       instead. Proper peeling should:
       - Deduct 1 per saturated neighbor from partner degree targets (constant).
       - Keep saturated pairs in the strength balance with simplified formula
         (no zero-inflation: `E[t_ij|v→∞] = q/(1-exp(-q))` for ME).
       - Remove z/w for saturated nodes from the iteration.
       Note: this improves robustness for edge cases but does NOT fix the
       non-convergence at N=50 (no nodes are saturated there). Priority: medium.

11. **Draft TODO.md** — Consolidate all detected issues into a single prioritized
    TODO list for future improvements. Erase the audit docs and merge their
    content into the TODO. Known issues to include:
    - W strength-cost `self_loops=False` very slow at N≥500 (~500s). Newton
      solver needs adaptive damping or better feasibility projection.
    - B strength-edges slow at N≥300 (~30s). IPF convergence for zero-inflated
      binomial coupled constraints needs acceleration.
    - B strength-degree does not converge at N≥300 (any self-loop setting).
      The coupled zero-inflated binomial strength-degree solver needs
      investigation — possibly requires a different algorithm.
    - Python wrappers repetitive (~3300 lines). Could use factory pattern.
    - Text I/O not streaming for Matrix Market/Pajek (2 occurrences).
    - Constraint-level code factoring (architectural wish).
    - Partial/full solver unification (cosmetic).
    - Release packaging: publish metadata, CI artifacts.

