"""Quick benchmark to N=1000 with verbose fitting."""

import time

import numpy as np

from odme.analysis import directed_degrees, directed_strengths
from odme.models import (
    fit_degree_bernoulli,
    fit_strength_poisson,
    fit_strength_cost_poisson,
    fit_strength_degree_poisson,
    fit_strength_edges_poisson,
    sample_strength_poisson,
)

N_VALUES = [50, 100, 200, 500, 1000]
AVG_S = 100


def _make_network(n: int):
    rng = np.random.default_rng(42)
    raw = rng.pareto(1.5, size=n) + 1.0
    p_out = raw / raw.sum()
    raw = rng.pareto(1.5, size=n) + 1.0
    p_in = raw / raw.sum()
    total = n * AVG_S
    s_out = np.round(p_out * total).astype(np.uint64)
    s_in = np.round(p_in * total).astype(np.uint64)
    diff = int(s_out.sum()) - int(s_in.sum())
    if diff > 0:
        s_in[np.argmax(s_in)] += abs(diff)
    elif diff < 0:
        s_out[np.argmax(s_out)] += abs(diff)
    fit = fit_strength_poisson(s_out, s_in)
    return sample_strength_poisson(fit.x, fit.y, seed=42)


def _check_strength(name, fit, s_out, s_in, n, rate_fn):
    """Check that fitted model recovers strengths."""
    err_out = 0.0
    err_in = 0.0
    for i in range(n):
        row = sum(rate_fn(i, j) for j in range(n))
        err_out = max(err_out, abs(row - s_out[i]))
    for j in range(n):
        col = sum(rate_fn(i, j) for i in range(n))
        err_in = max(err_in, abs(col - s_in[j]))
    total_s = s_out.sum()
    print(f"    {name}: max_err_out={err_out:.2f} max_err_in={err_in:.2f} (T={total_s:.0f})")


for n in N_VALUES:
    print(f"\n{'='*60}")
    print(f"N={n}")
    edges = _make_network(n)
    s = directed_strengths(edges)
    k = directed_degrees(edges)
    s_out = s.out.astype(float)
    s_in = s.incoming.astype(float)
    k_out = k.out.astype(float)
    k_in = k.incoming.astype(float)
    print(f"  E={edges.num_edges}, T={edges.total_events}")

    # Fixed strength ME
    t0 = time.perf_counter()
    fit_s = fit_strength_poisson(s_out.astype(np.uint64), s_in.astype(np.uint64), verbose=2)
    dt = time.perf_counter() - t0
    print(f"  fit_strength_poisson: {dt:.4f}s  conv={fit_s.converged} iters={fit_s.iterations}")

    # Fixed degree binary
    t0 = time.perf_counter()
    fit_k = fit_degree_bernoulli(k_out, k_in, verbose=2)
    dt = time.perf_counter() - t0
    print(f"  fit_degree_bernoulli: {dt:.4f}s  conv={fit_k.converged} iters={fit_k.iterations}")
    _check_strength("degree", fit_k, k_out, k_in, n,
                     lambda i, j: fit_k.x[i]*fit_k.y[j]/(1+fit_k.x[i]*fit_k.y[j]))

    # Fixed strength + edges
    if n <= 500:
        t0 = time.perf_counter()
        fit_se = fit_strength_edges_poisson(s_out, s_in, float(edges.num_edges), verbose=2)
        dt = time.perf_counter() - t0
        print(f"  fit_strength_edges_poisson: {dt:.4f}s  conv={fit_se.converged} iters={fit_se.iterations} lam={fit_se.lam:.6f}")

    # Fixed strength + degree
    if n <= 200:
        t0 = time.perf_counter()
        fit_sd = fit_strength_degree_poisson(s_out, s_in, k_out, k_in, verbose=2)
        dt = time.perf_counter() - t0
        print(f"  fit_strength_degree_poisson: {dt:.4f}s  conv={fit_sd.converged} iters={fit_sd.iterations}")

    # Fixed strength + cost
    if n <= 500:
        rng = np.random.default_rng(99)
        pos = rng.uniform(0, 10, size=(n, 2))
        c_src, c_tgt, c_val = [], [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                c_src.append(i)
                c_tgt.append(j)
                c_val.append(float(np.linalg.norm(pos[i] - pos[j])))
        cost_src = np.array(c_src)
        cost_tgt = np.array(c_tgt)
        cost_val = np.array(c_val)
        cost_map = {(int(a), int(b)): float(c) for a, b, c in zip(cost_src, cost_tgt, cost_val)}
        target_cost = sum(
            float(w) * cost_map.get((int(sv), int(tv)), 0.0)
            for sv, tv, w in zip(edges.source, edges.target, edges.weight)
        )
        t0 = time.perf_counter()
        fit_sc = fit_strength_cost_poisson(
            s_out, s_in, cost_src, cost_tgt, cost_val, target_cost, verbose=2
        )
        dt = time.perf_counter() - t0
        print(f"  fit_strength_cost_poisson: {dt:.4f}s  conv={fit_sc.converged} iters={fit_sc.iterations} gamma={fit_sc.gamma:.6f}")
