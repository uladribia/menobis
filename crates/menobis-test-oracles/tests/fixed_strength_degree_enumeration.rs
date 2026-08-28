//! Heavy: exact enumerated transition-matrix oracle for the fixed-(s,k)
//! kernels (§21–§22 of the fixed-(s,k) implementation plan).
//!
//! Tiny exact-strength fibers are enumerated independently (families
//! ME/B/W, self-loop policies, symmetric/heterogeneous margins), the
//! exact fixed-(s,E) kernel `K_E` is built as an explicit matrix from
//! the finished fixed-sE machinery, and the new fixed-degree objects are
//! verified exactly:
//!
//! - the degree-biased auxiliary matrix `Q` (§22.1):
//!   `Q[x,y] = K_E[x,y]·min(1, exp(−λ·(D[y]−D[x])))` with the outer-MH
//!   rejection mass on the diagonal, targeting
//!   `mu(x) ∝ pi_E(x)·exp(−λ·D(x))`;
//! - the capped first-return trace matrix `R` (§22.2): probability that
//!   the auxiliary chain first returns to the exact degree fiber within
//!   `L` steps (timeout mass restores the origin, `R[x,x]`);
//! - detailed balance, stationarity, row sums, and connectivity of both
//!   matrices against the independent target
//!   `pi_(s,k)(t) ∝ Π d_F(t_ij)` conditioned on exact degrees.
//!
//! **Independence guarantee (§21):** family target weights use explicit
//! reference formulas — ME `1/t!`, B `C(M,t)`, W `C(M+t−1,t)` — never
//! production `OccupationFamily::log_base_measure`.  The degree vectors
//! are extracted from the enumerated positive cells, never from
//! production state caches.
//!
//! This is intentionally a **separate, self-contained** integration test
//! file (review decision): it duplicates the exact `K_E` builder instead
//! of refactoring the finished fixed-sE oracle.

use std::collections::HashMap;

use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;

/// A sorted list of positive-occupation pairs: `((src, tgt), occ)`.
type OccupiedState = Vec<((u64, u64), OccNum)>;

/// An enumerated occupation table together with its log family weight.
type WeightedState = (OccupiedState, f64);

/// Degree vectors of a state: `(k_out, k_in)`.
type Degrees = (Vec<u32>, Vec<u32>);

// ---------------------------------------------------------------------------
// Independent family weights and capacity (§21)
// ---------------------------------------------------------------------------

/// Independent log local degeneracy `log d_F(t)`.
///
/// - ME: `-ln Γ(t+1)`
/// - B:  `ln C(M, t)`
/// - W:  `ln C(M+t-1, t)`
fn log_d_independent(family: OccupationFamily, t: OccNum) -> f64 {
    match family {
        OccupationFamily::ME => -libm::lgamma(t as f64 + 1.0),
        OccupationFamily::B { layers } => {
            if t > layers as OccNum {
                return f64::NEG_INFINITY;
            }
            let m = layers as f64;
            libm::lgamma(m + 1.0) - libm::lgamma(t as f64 + 1.0) - libm::lgamma(m - t as f64 + 1.0)
        }
        OccupationFamily::W { layers } => {
            let m = layers as f64;
            libm::lgamma(m + t as f64) - libm::lgamma(t as f64 + 1.0) - libm::lgamma(m)
        }
    }
}

/// Independent per-cell capacity (`B` layers, unbounded otherwise).
fn family_capacity(family: OccupationFamily) -> OccNum {
    match family {
        OccupationFamily::B { layers } => layers as OccNum,
        _ => OccNum::MAX,
    }
}

// ---------------------------------------------------------------------------
// Tiny-fiber enumeration
// ---------------------------------------------------------------------------

/// Enumerate every occupation table satisfying exact strengths, domain
/// admissibility (self-loop policy), and family capacity.
#[allow(dead_code)] // convenience wrapper; the grid uses the excluding variant
fn enumerate_fiber(
    family: OccupationFamily,
    s_out: &[OccNum],
    s_in: &[OccNum],
    self_loops: bool,
) -> Vec<WeightedState> {
    let empty = std::collections::HashSet::new();
    enumerate_fiber_excluding(family, s_out, s_in, self_loops, &empty)
}

fn enumerate_fiber_excluding(
    family: OccupationFamily,
    s_out: &[OccNum],
    s_in: &[OccNum],
    self_loops: bool,
    excluded: &std::collections::HashSet<(u64, u64)>,
) -> Vec<WeightedState> {
    let n = s_out.len();
    let cap = family_capacity(family);
    let cells: Vec<(u64, u64)> = (0..n as u64)
        .flat_map(|i| (0..n as u64).map(move |j| (i, j)))
        .filter(|&(i, j)| (self_loops || i != j) && !excluded.contains(&(i, j)))
        .collect();

    let mut results = Vec::new();

    #[allow(clippy::too_many_arguments)]
    fn recurse(
        idx: usize,
        cells: &[(u64, u64)],
        cap: OccNum,
        family: OccupationFamily,
        remaining_out: &mut [OccNum],
        remaining_in: &mut [OccNum],
        current: &mut OccupiedState,
        results: &mut Vec<WeightedState>,
    ) {
        if idx == cells.len() {
            if remaining_out.iter().all(|&s| s == 0) && remaining_in.iter().all(|&s| s == 0) {
                let log_weight: f64 = current
                    .iter()
                    .map(|&(_, occ)| log_d_independent(family, occ))
                    .sum();
                results.push((current.clone(), log_weight));
            }
            return;
        }
        let (src, tgt) = cells[idx];
        let max_possible = remaining_out[src as usize]
            .min(remaining_in[tgt as usize])
            .min(cap);
        for occ in 0..=max_possible {
            remaining_out[src as usize] -= occ;
            remaining_in[tgt as usize] -= occ;
            if occ > 0 {
                current.push(((src, tgt), occ));
            }
            recurse(
                idx + 1,
                cells,
                cap,
                family,
                remaining_out,
                remaining_in,
                current,
                results,
            );
            if occ > 0 {
                current.pop();
            }
            remaining_out[src as usize] += occ;
            remaining_in[tgt as usize] += occ;
        }
    }

    let mut remaining_out = s_out.to_vec();
    let mut remaining_in = s_in.to_vec();
    let mut current: OccupiedState = Vec::new();
    recurse(
        0,
        &cells,
        cap,
        family,
        &mut remaining_out,
        &mut remaining_in,
        &mut current,
        &mut results,
    );
    results
}

// ---------------------------------------------------------------------------
// Exact K_E builder (identical law to the finished fixed-(s,E) kernel)
// ---------------------------------------------------------------------------

struct OccupancyCounts {
    row: Vec<usize>,
    col: Vec<usize>,
}

fn occupancy_counts(state: &OccupiedState, n: usize) -> OccupancyCounts {
    let mut row = vec![0usize; n];
    let mut col = vec![0usize; n];
    for &((src, tgt), _) in state {
        row[src as usize] += 1;
        col[tgt as usize] += 1;
    }
    OccupancyCounts { row, col }
}

fn apply_proposal(state: &OccupiedState, a: u64, b: u64, c: u64, d: u64) -> OccupiedState {
    let mut map: HashMap<(u64, u64), i64> = state.iter().map(|&(p, o)| (p, o as i64)).collect();
    *map.entry((a, b)).or_insert(0) -= 1;
    *map.entry((c, d)).or_insert(0) -= 1;
    *map.entry((a, d)).or_insert(0) += 1;
    *map.entry((c, b)).or_insert(0) += 1;
    let mut result: OccupiedState = map
        .into_iter()
        .filter(|&(_, v)| v > 0)
        .map(|(k, v)| (k, v as OccNum))
        .collect();
    result.sort_unstable();
    result
}

fn state_index_map(states: &[WeightedState]) -> HashMap<OccupiedState, usize> {
    states
        .iter()
        .enumerate()
        .map(|(i, (s, _))| (s.clone(), i))
        .collect()
}

/// Exact Metropolis–Hastings matrix of the ordinary fixed-strength
/// proposal (identical law to production `draw_cycle4_proposal` +
/// `log_alpha_with_extra`), targeting `pi ∝ exp(log_weight)`.
#[allow(clippy::too_many_arguments)]
fn mh_matrix(
    states: &[WeightedState],
    n: usize,
    family: OccupationFamily,
    self_loops: bool,
    log_weight: &[f64],
    excluded: &std::collections::HashSet<(u64, u64)>,
) -> Vec<Vec<f64>> {
    let k = states.len();
    debug_assert_eq!(log_weight.len(), k);
    let cap = family_capacity(family);
    let index = state_index_map(states);
    let mut p = vec![vec![0.0_f64; k]; k];

    for si in 0..k {
        let (state, _) = &states[si];
        let lw = log_weight[si];
        let m = state.len();
        if m < 2 {
            p[si][si] = 1.0;
            continue;
        }
        let occ = occupancy_counts(state, n);
        let map: HashMap<(u64, u64), OccNum> = state.iter().copied().collect();

        for i in 0..m {
            let ((a, b), occ_ab) = state[i];
            for j in (i + 1)..m {
                let ((c, d), occ_cd) = state[j];
                if a == c || b == d {
                    continue;
                }

                let v_ab = m as i64 - occ.row[a as usize] as i64 - occ.col[b as usize] as i64 + 1;
                let v_cd = m as i64 - occ.row[c as usize] as i64 - occ.col[d as usize] as i64 + 1;
                if v_ab <= 0 || v_cd <= 0 {
                    continue;
                }

                let occ_ad = map.get(&(a, d)).copied().unwrap_or(0);
                let occ_cb = map.get(&(c, b)).copied().unwrap_or(0);
                let new_ad = occ_ad + 1;
                let new_cb = occ_cb + 1;

                if !self_loops && (a == d || c == b) {
                    continue;
                }
                if excluded.contains(&(a, d)) || excluded.contains(&(c, b)) {
                    continue;
                }
                if new_ad > cap || new_cb > cap {
                    continue;
                }

                let leaves_ab = i64::from(occ_ab == 1);
                let leaves_cd = i64::from(occ_cd == 1);
                let enters_ad = i64::from(occ_ad == 0);
                let enters_cb = i64::from(occ_cb == 0);
                let m_prime = m as i64 + enters_ad + enters_cb - leaves_ab - leaves_cd;
                if m_prime <= 0 {
                    continue;
                }
                let row_a_prime = occ.row[a as usize] as i64 + enters_ad - leaves_ab;
                let row_c_prime = occ.row[c as usize] as i64 + enters_cb - leaves_cd;
                let col_b_prime = occ.col[b as usize] as i64 + enters_cb - leaves_ab;
                let col_d_prime = occ.col[d as usize] as i64 + enters_ad - leaves_cd;
                let v_ad_prime = m_prime - row_a_prime - col_d_prime + 1;
                let v_cb_prime = m_prime - row_c_prime - col_b_prime + 1;
                if v_ad_prime <= 0 || v_cb_prime <= 0 {
                    continue;
                }

                let q_fwd = (1.0 / v_ab as f64 + 1.0 / v_cd as f64) / m as f64;
                let q_rev = (1.0 / v_ad_prime as f64 + 1.0 / v_cb_prime as f64) / m_prime as f64;

                let dest = apply_proposal(state, a, b, c, d);
                let di = index
                    .get(&dest)
                    .copied()
                    .unwrap_or_else(|| panic!("destination {dest:?} not enumerated"));

                let log_alpha = (log_weight[di] - lw) + q_rev.ln() - q_fwd.ln();
                let acceptance = if log_alpha >= 0.0 {
                    1.0
                } else {
                    log_alpha.exp()
                };
                p[si][di] += q_fwd * acceptance;
            }
        }

        let sum: f64 = p[si].iter().sum();
        debug_assert!(sum <= 1.0 + 1e-12, "row {si} transition sum {sum} > 1");
        p[si][si] = 1.0 - sum;
    }
    p
}

/// Exact-E local kernel: destinations leaving the exact-`E` fiber
/// collapse onto the diagonal.
fn local_fixed_e_matrix(
    base: &[Vec<f64>],
    states: &[WeightedState],
    edge_target: usize,
) -> Vec<Vec<f64>> {
    let k = states.len();
    let mut p = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        if states[i].0.len() != edge_target {
            continue;
        }
        let mut inside: f64 = 0.0;
        for j in 0..k {
            if states[j].0.len() == edge_target {
                p[i][j] = base[i][j];
                inside += base[i][j];
            }
        }
        p[i][i] = base[i][i] + (1.0 - inside);
    }
    p
}

/// Auxiliary bridge target weights `mu_lambda(t) ∝ pi_s(t) exp(−λ
/// |E(t) − E_target|)`.
fn aux_log_weights(states: &[WeightedState], lambda: f64, edge_target: usize) -> Vec<f64> {
    states
        .iter()
        .map(|(s, w)| w - lambda * s.len().abs_diff(edge_target) as f64)
        .collect()
}

/// Exact fixed-E bridge matrix (identical law to `bridge_step_impl`):
/// first substep in-fiber aborts into the diagonal; first return stops;
/// cap exhaustion restores the origin.
#[allow(clippy::too_many_arguments)]
fn bridge_matrix(
    states: &[WeightedState],
    n: usize,
    family: OccupationFamily,
    self_loops: bool,
    edge_target: usize,
    lambda: f64,
    max_steps: usize,
    excluded: &std::collections::HashSet<(u64, u64)>,
) -> Vec<Vec<f64>> {
    let k = states.len();
    let aux_lw = aux_log_weights(states, lambda, edge_target);
    let aux = mh_matrix(states, n, family, self_loops, &aux_lw, excluded);
    let in_fiber = |i: usize| states[i].0.len() == edge_target;

    let mut b = vec![vec![0.0_f64; k]; k];
    for x in 0..k {
        if !in_fiber(x) {
            continue;
        }
        let mut mass = vec![0.0_f64; k];
        for z in 0..k {
            let p = aux[x][z];
            if p == 0.0 {
                continue;
            }
            if in_fiber(z) {
                b[x][x] += p; // abort (held / rejected / in-fiber move)
            } else {
                mass[z] += p;
            }
        }
        for _ in 1..max_steps {
            let mut next = vec![0.0_f64; k];
            for z in 0..k {
                let mz = mass[z];
                if mz == 0.0 {
                    continue;
                }
                for w in 0..k {
                    let p = aux[z][w];
                    if p == 0.0 {
                        continue;
                    }
                    if in_fiber(w) {
                        b[x][w] += mz * p;
                    } else {
                        next[w] += mz * p;
                    }
                }
            }
            mass = next;
        }
        let remaining: f64 = mass.iter().sum();
        b[x][x] += remaining; // cap: undo -> origin
    }
    b
}

/// Exact full mixed fixed-(s,E) kernel `K_E` (identical law to the
/// production `fixed_edge_step` constant mixture with `rho`).
#[allow(clippy::too_many_arguments)]
fn full_mixed_matrix(
    states: &[WeightedState],
    n: usize,
    family: OccupationFamily,
    self_loops: bool,
    edge_target: usize,
    lambda: f64,
    max_steps: usize,
    rho: f64,
    excluded: &std::collections::HashSet<(u64, u64)>,
) -> Vec<Vec<f64>> {
    let k = states.len();
    let base = mh_matrix(
        states,
        n,
        family,
        self_loops,
        &states.iter().map(|(_, w)| *w).collect::<Vec<_>>(),
        excluded,
    );
    let local = local_fixed_e_matrix(&base, states, edge_target);
    let bridge = bridge_matrix(
        states,
        n,
        family,
        self_loops,
        edge_target,
        lambda,
        max_steps,
        excluded,
    );
    let mut p = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        if states[i].0.len() != edge_target {
            continue;
        }
        for j in 0..k {
            p[i][j] = (1.0 - rho) * local[i][j] + rho * bridge[i][j];
        }
    }
    p
}

// ---------------------------------------------------------------------------
// Degree machinery (§12, §21): all independent of production caches
// ---------------------------------------------------------------------------

/// Extract the directed degree vectors `(k_out, k_in)` of a state from
/// its positive cells only.
fn degrees_of(state: &OccupiedState, n: usize) -> Degrees {
    let mut out = vec![0u32; n];
    let mut inp = vec![0u32; n];
    for &((s, t), _) in state {
        out[s as usize] += 1;
        inp[t as usize] += 1;
    }
    (out, inp)
}

/// Raw degree distance `D_raw = Σ_i |k_out[i]−k*| + Σ_j |k_in[j]−k*|`
/// (§12).  Every exact-E state has the same total edge count as any
/// feasible degree target, so `D_raw` is even; the half-normalized
/// distance used by the potential is `D = D_raw/2`.
fn degree_distance_raw(state: &OccupiedState, n: usize, target: &Degrees) -> u64 {
    let (out, inp) = degrees_of(state, n);
    let mut d = 0u64;
    for i in 0..n {
        d += (out[i] as u64).abs_diff(target.0[i] as u64);
        d += (inp[i] as u64).abs_diff(target.1[i] as u64);
    }
    d
}

/// Index of the exact degree fiber: states with `D == 0` for `target`.
fn degree_fiber_indices(states: &[WeightedState], n: usize, target: &Degrees) -> Vec<usize> {
    states
        .iter()
        .enumerate()
        .filter(|(_, (s, _))| degree_distance_raw(s, n, target) == 0)
        .map(|(i, _)| i)
        .collect()
}

/// Exact degree-biased auxiliary matrix `Q` (§22.1) over the exact-E
/// state space: for `x != y`,
///
/// ```text
/// Q[x,y] = K_E[x,y] · min(1, exp(−λ·(D[y]−D[x])))
/// ```
///
/// with all outer-MH rejection mass on the diagonal:
///
/// ```text
/// Q[x,x] = K_E[x,x] + Σ_{y≠x} K_E[x,y]·(1 − alpha(x,y))
/// ```
fn degree_auxiliary_matrix(
    k_e: &[Vec<f64>],
    states: &[WeightedState],
    n: usize,
    lambda: f64,
    e_space: &[usize],
    target: &Degrees,
) -> Vec<Vec<f64>> {
    let k = states.len();
    let d: Vec<i64> = e_space
        .iter()
        .map(|&i| (degree_distance_raw(&states[i].0, n, target) / 2) as i64)
        .collect();
    let mut q = vec![vec![0.0_f64; k]; k];
    for (r, &x) in e_space.iter().enumerate() {
        let mut diag = k_e[x][x];
        for (c, &y) in e_space.iter().enumerate() {
            if r == c {
                continue;
            }
            let delta = d[c] - d[r];
            let alpha = if delta >= 0 {
                (-lambda * delta as f64).exp()
            } else {
                1.0
            };
            q[x][y] = k_e[x][y] * alpha;
            diag += k_e[x][y] * (1.0 - alpha);
        }
        q[x][x] = diag;
    }
    q
}

/// Exact capped first-return trace matrix `R` (§22.2): from every origin
/// in the exact degree fiber, propagate active probability through `Q`
/// for at most `L` steps; any destination with `D == 0` is absorbed into
/// `R[x,·]` immediately (including step-1 returns and self-loops); after
/// `L` steps all remaining outside-fiber mass restores the origin
/// (`R[x,x]` — production timeout undoes the excursion).
fn first_return_trace_matrix(
    q: &[Vec<f64>],
    states: &[WeightedState],
    n: usize,
    e_space: &[usize],
    target: &Degrees,
    cap: usize,
) -> Vec<Vec<f64>> {
    let k = states.len();
    let in_fiber = |i: usize| degree_distance_raw(&states[i].0, n, target) == 0;
    let mut r = vec![vec![0.0_f64; k]; k];
    for &x in e_space {
        if !in_fiber(x) {
            continue;
        }
        let mut mass = vec![0.0_f64; k];
        mass[x] = 1.0;
        for _ in 0..cap {
            let mut next = vec![0.0_f64; k];
            for z in 0..k {
                let mz = mass[z];
                if mz == 0.0 {
                    continue;
                }
                for w in 0..k {
                    let p = q[z][w];
                    if p == 0.0 {
                        continue;
                    }
                    if in_fiber(w) {
                        r[x][w] += mz * p; // absorbed, not propagated
                    } else {
                        next[w] += mz * p;
                    }
                }
            }
            mass = next;
        }
        let remaining: f64 = mass.iter().sum();
        r[x][x] += remaining; // cap timeout -> origin
    }
    r
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

fn assert_row_sums(matrix: &[Vec<f64>], rows: &[usize], label: &str) {
    for &i in rows {
        let s: f64 = matrix[i].iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-9,
            "{label}: row {i} sums to {s:.3e} (expected 1)"
        );
    }
}

/// Pairwise detailed balance on `fiber` with unnormalized weights
/// `w(x) = exp(log_weight[x])`: `w(x) P(x,y) == w(y) P(y,x)`.
fn assert_pairwise_detailed_balance(
    states: &[WeightedState],
    matrix: &[Vec<f64>],
    fiber: &[usize],
    label: &str,
) {
    let mut max_rel = 0.0f64;
    for &i in fiber {
        let wi = states[i].1.exp();
        for &j in fiber {
            let wj = states[j].1.exp();
            let lhs = wi * matrix[i][j];
            let rhs = wj * matrix[j][i];
            if lhs == 0.0 && rhs == 0.0 {
                continue;
            }
            let denom = lhs.abs().max(rhs.abs());
            let rel = (lhs - rhs).abs() / denom;
            max_rel = max_rel.max(rel);
        }
    }
    assert!(
        max_rel < 1e-9,
        "{label}: max detailed-balance residual {max_rel:.3e} >= 1e-9"
    );
}

/// Stationarity: normalized `pi` over `fiber` is a fixed point of the
/// matrix restricted to the fiber.
fn assert_stationarity(
    states: &[WeightedState],
    matrix: &[Vec<f64>],
    fiber: &[usize],
    label: &str,
) {
    let z: f64 = fiber.iter().map(|&i| states[i].1.exp()).sum();
    let k = states.len();
    let pi: Vec<f64> = (0..k)
        .map(|i| {
            if fiber.contains(&i) {
                states[i].1.exp() / z
            } else {
                0.0
            }
        })
        .collect();
    let mut next = vec![0.0_f64; k];
    for &i in fiber {
        for j in 0..k {
            next[j] += pi[i] * matrix[i][j];
        }
    }
    let mut max_abs = 0.0f64;
    for &j in fiber {
        max_abs = max_abs.max((next[j] - pi[j]).abs());
    }
    assert!(
        max_abs < 1e-10,
        "{label}: max stationarity residual {max_abs:.3e} >= 1e-10"
    );
}

/// Connected components of the undirected graph induced by `matrix` over
/// the fiber (edge where either direction has positive probability).
fn fiber_components(matrix: &[Vec<f64>], fiber: &[usize]) -> Vec<usize> {
    let mut comp = vec![usize::MAX; fiber.len()];
    let mut next = 0usize;
    for fi in 0..fiber.len() {
        if comp[fi] != usize::MAX {
            continue;
        }
        let mut stack = vec![fi];
        comp[fi] = next;
        while let Some(fu) = stack.pop() {
            let u = fiber[fu];
            for (fv, &v) in fiber.iter().enumerate() {
                if comp[fv] == usize::MAX && (matrix[u][v] > 0.0 || matrix[v][u] > 0.0) {
                    comp[fv] = next;
                    stack.push(fv);
                }
            }
        }
        next += 1;
    }
    comp
}

// ---------------------------------------------------------------------------
// Mandatory tiny grid (§23.1)
// ---------------------------------------------------------------------------

/// Mandatory tiny cases: N = 2 and 3, small strength totals, loops
/// allowed/forbidden where feasible, ME / B(M=1) / B(M=2) / W(M=1) /
/// W(M=2) where the strength totals stay enumerable.  The 6th element is
/// the residual-domain exclusion set (empty for no fixed pairs).
#[allow(clippy::type_complexity)]
fn mandatory_cases() -> Vec<(
    String,
    OccupationFamily,
    Vec<OccNum>,
    Vec<OccNum>,
    bool,
    std::collections::HashSet<(u64, u64)>,
)> {
    let mut out = Vec::new();
    let empty = std::collections::HashSet::new();
    // ME
    for sl in [true, false] {
        out.push((
            "me-n2".into(),
            OccupationFamily::ME,
            vec![2, 2],
            vec![2, 2],
            sl,
            empty.clone(),
        ));
        out.push((
            "me-n2-hetero".into(),
            OccupationFamily::ME,
            vec![3, 1],
            vec![1, 3],
            sl,
            empty.clone(),
        ));
        out.push((
            "me-n3".into(),
            OccupationFamily::ME,
            vec![2, 2, 2],
            vec![2, 2, 2],
            sl,
            empty.clone(),
        ));
        out.push((
            "me-n3-hetero".into(),
            OccupationFamily::ME,
            vec![3, 2, 1],
            vec![1, 2, 3],
            sl,
            empty.clone(),
        ));
    }
    out.push((
        "me-n3-loops-symmetric".into(),
        OccupationFamily::ME,
        vec![3, 3, 3],
        vec![3, 3, 3],
        true,
        empty.clone(),
    ));
    // B(M=1): binary occupations.
    out.push((
        "b1-n2".into(),
        OccupationFamily::B { layers: 1 },
        vec![1, 1],
        vec![1, 1],
        true,
        empty.clone(),
    ));
    out.push((
        "b1-n2-loopless".into(),
        OccupationFamily::B { layers: 1 },
        vec![1, 1],
        vec![1, 1],
        false,
        empty.clone(),
    ));
    out.push((
        "b1-n3".into(),
        OccupationFamily::B { layers: 1 },
        vec![2, 1, 1],
        vec![1, 1, 2],
        true,
        empty.clone(),
    ));
    // B(M=2).
    out.push((
        "b2-n2".into(),
        OccupationFamily::B { layers: 2 },
        vec![2, 2],
        vec![2, 2],
        true,
        empty.clone(),
    ));
    out.push((
        "b2-n3".into(),
        OccupationFamily::B { layers: 2 },
        vec![3, 3, 3],
        vec![3, 3, 3],
        true,
        empty.clone(),
    ));
    out.push((
        "b2-n3-loopless".into(),
        OccupationFamily::B { layers: 2 },
        vec![2, 2, 2],
        vec![2, 2, 2],
        false,
        empty.clone(),
    ));
    // W(M=1) geometric and W(M=2).
    out.push((
        "w1-n2".into(),
        OccupationFamily::W { layers: 1 },
        vec![2, 2],
        vec![2, 2],
        true,
        empty.clone(),
    ));
    out.push((
        "w1-n3".into(),
        OccupationFamily::W { layers: 1 },
        vec![2, 2, 2],
        vec![2, 2, 2],
        false,
        empty.clone(),
    ));
    out.push((
        "w2-n2".into(),
        OccupationFamily::W { layers: 2 },
        vec![3, 3],
        vec![3, 3],
        true,
        empty.clone(),
    ));
    // Fixed-pair residuals: one positive and one zero fixed coordinate.
    let mut ex_pos = std::collections::HashSet::new();
    ex_pos.insert((0, 1));
    out.push((
        "me-n3-positive-fixed".into(),
        OccupationFamily::ME,
        vec![2, 3, 3],
        vec![3, 2, 3],
        true,
        ex_pos,
    ));
    let mut ex_zero = std::collections::HashSet::new();
    ex_zero.insert((0, 1));
    out.push((
        "me-n3-zero-fixed".into(),
        OccupationFamily::ME,
        vec![2, 2, 2],
        vec![2, 2, 2],
        true,
        ex_zero,
    ));
    out
}

/// Whether every state of the degree fiber lies in a single component of
/// the undirected support graph of `matrix` over `e_space`.  The trace
/// kernel can never connect states the underlying `K_E` (here the exact
/// 4-cycle chain) cannot reach at all — e.g. the two loopless directed
/// 3-cycles have **no valid 4-cycle** (every pair of occupied cells has a
/// diagonal cross cell), so `K_E` is the identity there and no cap can
/// help.  This mirrors the fixed-sE oracle's `underlying_connected`
/// guard (§23.1 note).
fn underlying_connects_fiber(matrix: &[Vec<f64>], e_space: &[usize], fiber: &[usize]) -> bool {
    let mut comp = vec![usize::MAX; matrix.len()];
    let mut stack = vec![fiber[0]];
    comp[fiber[0]] = 0;
    while let Some(u) = stack.pop() {
        for &v in e_space {
            if comp[v] == usize::MAX && (matrix[u][v] > 0.0 || matrix[v][u] > 0.0) {
                comp[v] = 0;
                stack.push(v);
            }
        }
    }
    fiber.iter().all(|&i| comp[i] == 0)
}
/// Build the exact Q and R matrices for one case and assert the full §38
/// gate on **every degree fiber with at least two states**.  Returns
/// (failures, number of non-singleton fibers checked).  Connectivity is
/// asserted only when the underlying `K_E` support graph connects the
/// fiber (a trace can never move between states the 4-cycle chain cannot
/// reach at all); underlying-disconnected fibers are genuine support
/// limitations, not trace defects.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn check_degree_kernel(
    label: &str,
    family: OccupationFamily,
    so: &[OccNum],
    si: &[OccNum],
    sl: bool,
    excluded: &std::collections::HashSet<(u64, u64)>,
    lambda: f64,
    bridge_steps: usize,
    rho: f64,
    cap: usize,
) -> (Vec<String>, usize) {
    let states = enumerate_fiber_excluding(family, so, si, sl, excluded);
    assert!(!states.is_empty(), "{label}: empty enumeration");
    let n = so.len();
    let k = states.len();

    // Group states by exact edge count.
    let mut edges: Vec<usize> = states.iter().map(|(s, _)| s.len()).collect();
    edges.sort_unstable();
    edges.dedup();

    let mut failures = Vec::new();
    let mut fibers_checked = 0usize;

    for &e in &edges {
        let e_space: Vec<usize> = states
            .iter()
            .enumerate()
            .filter(|(_, (s, _))| s.len() == e)
            .map(|(i, _)| i)
            .collect();
        if e_space.len() < 2 {
            continue;
        }

        // Exact K_E over the exact-E space (production device constants).
        let k_e = full_mixed_matrix(
            &states,
            n,
            family,
            sl,
            e,
            lambda,
            bridge_steps,
            rho,
            excluded,
        );

        // Feasible degree targets: distinct degree vectors among the
        // enumerated exact-E states (feasibility is guaranteed by
        // construction, §38).
        let mut seen = std::collections::HashSet::new();
        let mut targets: Vec<Degrees> = Vec::new();
        for &i in &e_space {
            let deg = degrees_of(&states[i].0, n);
            if seen.insert(deg.clone()) {
                targets.push(deg);
            }
        }

        for target in targets {
            let fiber = degree_fiber_indices(&states, n, &target);
            if fiber.len() < 2 {
                continue;
            }
            fibers_checked += 1;
            let flabel = format!(
                "{label} family={family:?} sl={sl} E={e} |A_k|={} cap={cap}",
                fiber.len()
            );

            // Q: exact degree-biased auxiliary matrix (§22.1).
            let q = degree_auxiliary_matrix(&k_e, &states, n, lambda, &e_space, &target);
            assert_row_sums(&q, &e_space, &format!("{flabel} Q"));
            // Auxiliary weights mu(x) ∝ pi_E(x)·exp(−λ D(x)).
            let mut mu_log = Vec::with_capacity(k);
            for state in &states {
                let d = degree_distance_raw(&state.0, n, &target) / 2;
                mu_log.push(state.1 - lambda * d as f64);
            }
            let mu_states: Vec<WeightedState> =
                (0..k).map(|i| (states[i].0.clone(), mu_log[i])).collect();
            assert_pairwise_detailed_balance(
                &mu_states,
                &q,
                &e_space,
                &format!("{flabel} Q aux DB"),
            );
            // Exact mu stationarity for Q: normalize over the E-space.
            let z: f64 = e_space.iter().map(|&i| mu_states[i].1.exp()).sum();
            let pi_mu: Vec<f64> = (0..k)
                .map(|i| {
                    if e_space.contains(&i) {
                        mu_states[i].1.exp() / z
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut next = vec![0.0_f64; k];
            for &i in &e_space {
                for j in 0..k {
                    next[j] += pi_mu[i] * q[i][j];
                }
            }
            let mut max_abs = 0.0f64;
            for &j in &e_space {
                max_abs = max_abs.max((next[j] - pi_mu[j]).abs());
            }
            assert!(
                max_abs < 1e-10,
                "{flabel}: Q stationarity residual {max_abs:.3e} >= 1e-10"
            );

            // R: exact capped first-return trace (§22.2).
            let r = first_return_trace_matrix(&q, &states, n, &e_space, &target, cap);
            assert_row_sums(&r, &fiber, &format!("{flabel} R"));
            assert_pairwise_detailed_balance(&states, &r, &fiber, &format!("{flabel} R trace DB"));
            assert_stationarity(
                &states,
                &r,
                &fiber,
                &format!("{flabel} R trace stationarity"),
            );

            // Connectivity of the trace over the degree fiber — only when
            // the underlying 4-cycle chain can reach all fiber states.
            if underlying_connects_fiber(&k_e, &e_space, &fiber) {
                let comps = fiber_components(&r, &fiber);
                let connected = comps.iter().collect::<std::collections::HashSet<_>>().len() == 1;
                if !connected {
                    failures.push(format!(
                        "{flabel}: trace DISCONNECTED ({} components)",
                        comps.iter().max().unwrap() + 1
                    ));
                }
            }
        }
    }
    (failures, fibers_checked)
}

/// Mandatory connectivity grid with cap selection (§23.1).  Try
/// `L = 16` (the initial default), then 32, then 64; use the smallest
/// passing value.  If 64 still leaves mandatory fibers disconnected,
/// report the smallest failing case (the feature must STOP).
#[test]
fn degree_trace_exact_gate_and_connectivity_grid() {
    let lambda = 1.0;
    let bridge_steps = 16usize;
    let rho = 0.05;

    let mut total_fibers = 0usize;
    let mut total_failures: Vec<String> = Vec::new();
    for (label, family, so, si, sl, excluded) in mandatory_cases() {
        let (fails16, fibers16) = check_degree_kernel(
            &label,
            family,
            &so,
            &si,
            sl,
            &excluded,
            lambda,
            bridge_steps,
            rho,
            16,
        );
        total_fibers += fibers16;
        if fails16.is_empty() {
            continue; // cap 16 already connects everything
        }
        let mut cap_failures = fails16.clone();
        let mut chosen_cap = 16usize;
        for cap in [32usize, 64usize] {
            let (fails, _) = check_degree_kernel(
                &label,
                family,
                &so,
                &si,
                sl,
                &excluded,
                lambda,
                bridge_steps,
                rho,
                cap,
            );
            if fails.is_empty() {
                cap_failures = Vec::new();
                chosen_cap = cap;
                break;
            }
        }
        if !cap_failures.is_empty() {
            total_failures.extend(cap_failures);
        }
        assert!(
            chosen_cap <= 64,
            "{label}: mandatory fiber disconnected even at cap 64; STOP per §23.1"
        );
    }
    assert!(
        total_failures.is_empty(),
        "degree-trace gate failures: {}",
        total_failures.join("; ")
    );
    assert!(
        total_fibers >= 8,
        "expected at least 8 non-singleton degree fibers in the mandatory grid, got {total_fibers}"
    );
}
