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

6. ~~**Sparse matrix/cost handling cleanup**~~ — Done.
7. ~~**Benchmark cleaning**~~ — Done. Single modern E2E script.
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
10. **Draft TODO.md** — Consolidate all detected issues into a single prioritized
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

