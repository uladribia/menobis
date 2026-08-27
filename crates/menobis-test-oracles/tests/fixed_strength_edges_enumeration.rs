//! Heavy: exact enumerated transition-matrix oracle for the fixed-(s,E)
//! kernels (§22–§23 of the fixed-sE implementation plan).
//!
//! Tiny fibers are enumerated independently (families ME/B/W, self-loop
//! policies, symmetric/heterogeneous margins), and the **exact** kernels
//! are built as explicit matrices:
//!
//! - base ordinary fixed-strength MH matrix [`mh_matrix`];
//! - exact-E local kernel [`local_fixed_e_matrix`] (veto on E change);
//! - (later phases) auxiliary edge-biased kernel, bridge matrix, and the
//!   full mixed kernel with mandatory connectivity grid.
//!
//! **Independence guarantee (§22.2):** the family target weights are
//! computed from explicit reference formulas — ME `1/t!`, B `C(M,t)`,
//! W `C(M+t-1,t)` — never by calling the production
//! `OccupationFamily::log_base_measure`.  Tolerances are documented per
//! assertion.
//!
//! This lives in the oracle crate so the exact-enumeration cost does not
//! slow down `cargo test -p menobis-core`.

use std::collections::HashMap;

use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;

/// A sorted list of positive-occupation pairs: `((src, tgt), occ)`.
type OccupiedState = Vec<((u64, u64), OccNum)>;

/// An enumerated occupation table together with its log family weight
/// `Σ log d_F(t_ij)` from the independent reference formulas.
type WeightedState = (OccupiedState, f64);

// ---------------------------------------------------------------------------
// Independent family weights and capacity (§22.2)
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
// Tiny-fiber enumeration (§22.1)
// ---------------------------------------------------------------------------

/// Enumerate every occupation table satisfying exact strengths, domain
/// admissibility (self-loop policy), and family capacity.
fn enumerate_fiber(
    family: OccupationFamily,
    s_out: &[OccNum],
    s_in: &[OccNum],
    self_loops: bool,
) -> Vec<WeightedState> {
    let n = s_out.len();
    let cap = family_capacity(family);
    let cells: Vec<(u64, u64)> = (0..n as u64)
        .flat_map(|i| (0..n as u64).map(move |j| (i, j)))
        .filter(|&(i, j)| self_loops || i != j)
        .collect();

    let mut results = Vec::new();

    /// Recursive enumeration over the tiny cell grid.
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
// Exact MH transition matrix over the ordinary fixed-strength target (§22.3)
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

/// Destination state after the fixed-direction 4-cycle (decrement
/// `(a,b)`,`(c,d)`; increment `(a,d)`,`(c,b)`), zero cells removed.
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

/// Exact Metropolis–Hastings transition matrix targeting weights
/// `pi(x) ∝ exp(log_weight[x])` under the ordinary occupied-cell proposal
/// (`q_fwd = (1/m)(1/v_ab + 1/v_cd)`), with the independent target and
/// the capacity/self-loop validation the production kernel applies.
///
/// `log_weight` is the log family degeneracy of each state (the
/// fixed-strength target).  Passing modified weights yields the
/// auxiliary bridge target in later phases.
fn mh_matrix(
    states: &[WeightedState],
    n: usize,
    family: OccupationFamily,
    self_loops: bool,
    log_weight: &[f64],
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

                // Availability counts (both orderings of P1).
                let v_ab = m as i64 - occ.row[a as usize] as i64 - occ.col[b as usize] as i64 + 1;
                let v_cd = m as i64 - occ.row[c as usize] as i64 - occ.col[d as usize] as i64 + 1;
                if v_ab <= 0 || v_cd <= 0 {
                    continue;
                }

                let occ_ad = map.get(&(a, d)).copied().unwrap_or(0);
                let occ_cb = map.get(&(c, b)).copied().unwrap_or(0);
                let new_ad = occ_ad + 1;
                let new_cb = occ_cb + 1;

                // Production `validate_four_cell`: domain admissibility
                // (self-loops) and family capacity on the cross cells.
                if !self_loops && (a == d || c == b) {
                    continue;
                }
                if new_ad > cap || new_cb > cap {
                    continue;
                }

                // Reverse availability in the destination state.
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
                let di = index[&dest];

                // Independent Hastings acceptance: pi(y) q_rev / (pi(x) q_fwd).
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

/// Connected components of the undirected graph induced by the transition
/// matrix (edges where either direction has positive probability).  Used
/// to identify genuine 4-cycle-kernel connectivity limits (§23).
fn connected_components(matrix: &[Vec<f64>]) -> Vec<usize> {
    let k = matrix.len();
    let mut comp = vec![usize::MAX; k];
    let mut next = 0usize;
    for start in 0..k {
        if comp[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        comp[start] = next;
        while let Some(u) = stack.pop() {
            for v in 0..k {
                if comp[v] == usize::MAX && (matrix[u][v] > 0.0 || matrix[v][u] > 0.0) {
                    comp[v] = next;
                    stack.push(v);
                }
            }
        }
        next += 1;
    }
    comp
}

/// Exact-E local kernel (§22.4): from the base matrix, any transition
/// whose destination leaves the exact-`E` fiber collapses to the
/// diagonal.
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
        // Probability mass that stays inside the fiber (incl. self).
        let mut inside: f64 = 0.0;
        for j in 0..k {
            if states[j].0.len() == edge_target {
                p[i][j] = base[i][j];
                inside += base[i][j];
            }
        }
        let outside_mass = 1.0 - inside;
        p[i][i] = base[i][i] + outside_mass;
    }
    p
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

/// Every row must sum to 1 (tolerance 1e-10 against accumulated row
/// arithmetic).  When `fiber` is given, only fiber rows are checked
/// (non-fiber rows are outside the kernel's domain).
fn assert_row_sums(matrix: &[Vec<f64>], fiber: Option<&[usize]>) {
    for (i, row) in matrix.iter().enumerate() {
        if let Some(f) = fiber {
            if !f.contains(&i) {
                continue;
            }
        }
        let s: f64 = row.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-10,
            "row {i} sums to {s:.3e} (expected 1)"
        );
    }
}

/// Pairwise detailed balance on the indexed states with unnormalized
/// weights `w(x) = exp(log_weight[x])`: `w(x) P(x,y) == w(y) P(y,x)`.
fn assert_pairwise_detailed_balance(
    states: &[WeightedState],
    matrix: &[Vec<f64>],
    fiber: &[usize],
    label: &str,
) {
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
            assert!(
                rel < 1e-9,
                "{label}: DB violated {i}->{j}: w_i P_ij={lhs:.3e} vs w_j P_ji={rhs:.3e} (rel {rel:.3e})"
            );
        }
    }
}

/// Stationarity: normalized `pi` over the fiber is a fixed point of the
/// matrix restricted to the fiber (tolerance 1e-10 absolute).
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
    for &j in fiber {
        assert!(
            (next[j] - pi[j]).abs() < 1e-10,
            "{label}: stationarity violated at {j}: pi_P[j]={:.6} vs pi[j]={:.6}",
            next[j],
            pi[j]
        );
    }
}

/// Log family-degeneracy weights of the enumerated states (the ordinary
/// fixed-strength target `pi_s`).
fn fixed_strength_log_weights(states: &[WeightedState]) -> Vec<f64> {
    states.iter().map(|(_, w)| *w).collect()
}

/// Auxiliary bridge target weights `mu_lambda(t) ∝ pi_s(t) exp(−λ |E(t)
/// − E_target|)` in log space (§8).
fn aux_log_weights(states: &[WeightedState], lambda: f64, edge_target: usize) -> Vec<f64> {
    states
        .iter()
        .map(|(s, w)| w - lambda * s.len().abs_diff(edge_target) as f64)
        .collect()
}

/// Exact bridge matrix by dynamic propagation (§22.6):
///
/// - first auxiliary substep: in-fiber destinations abort into `B[x,x]`,
///   outside destinations start the excursion;
/// - further substeps: outside→outside stays active, outside→fiber
///   accumulates into `B[x,y]` and stops;
/// - at the maximum step all remaining outside mass becomes `B[x,x]`
///   (production undoes and restores the origin).
fn bridge_matrix(
    states: &[WeightedState],
    n: usize,
    family: OccupationFamily,
    self_loops: bool,
    edge_target: usize,
    lambda: f64,
    max_steps: usize,
) -> Vec<Vec<f64>> {
    let k = states.len();
    let aux_lw = aux_log_weights(states, lambda, edge_target);
    let aux = mh_matrix(states, n, family, self_loops, &aux_lw);
    let in_fiber = |i: usize| states[i].0.len() == edge_target;

    let mut b = vec![vec![0.0_f64; k]; k];
    for x in 0..k {
        if !in_fiber(x) {
            continue;
        }
        let mut mass = vec![0.0_f64; k];
        // Departure step: in-fiber destinations abort into B[x,x];
        // outside destinations start the excursion.
        for z in 0..k {
            let p = aux[x][z];
            if p == 0.0 {
                continue;
            }
            if in_fiber(z) {
                b[x][x] += p; // abort (held / rejected / delta-E=0)
            } else {
                mass[z] += p;
            }
        }
        // Substeps 2..=max_steps.
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

// ---------------------------------------------------------------------------
// Phase 4: exact-E local kernel detailed balance on tiny fibers (§22.4)
// ---------------------------------------------------------------------------

fn local_fixed_e_kernel_checks(family: OccupationFamily, so: &[OccNum], si: &[OccNum], sl: bool) {
    let states = enumerate_fiber(family, so, si, sl);
    assert!(
        !states.is_empty(),
        "enumeration returned no states for {family:?}"
    );
    let n = so.len();
    let base = mh_matrix(&states, n, family, sl, &fixed_strength_log_weights(&states));

    // Distinct feasible edge counts in the fiber.
    let mut edges: Vec<usize> = states.iter().map(|(s, _)| s.len()).collect();
    edges.sort_unstable();
    edges.dedup();
    assert!(!edges.is_empty());

    for &e in &edges {
        let fiber: Vec<usize> = states
            .iter()
            .enumerate()
            .filter(|(_, (s, _))| s.len() == e)
            .map(|(i, _)| i)
            .collect();
        let label = format!("{family:?} sl={sl} E={e}");
        let local = local_fixed_e_matrix(&base, &states, e);
        assert_row_sums(&local, Some(&fiber));
        assert_pairwise_detailed_balance(&states, &local, &fiber, &label);
        assert_stationarity(&states, &local, &fiber, &label);
    }
}

#[test]
fn local_fixed_e_kernel_exact_detailed_balance_tiny_fibers() {
    // Mandatory tiny cases (§23): families ME/B/W, self-loop policies,
    // symmetric and heterogeneous margins, all feasible edge targets.
    let cases: Vec<(OccupationFamily, Vec<OccNum>, Vec<OccNum>, bool)> = vec![
        // ME
        (OccupationFamily::ME, vec![2u64, 2], vec![2, 2], true),
        (OccupationFamily::ME, vec![2u64, 2], vec![2, 2], false),
        (OccupationFamily::ME, vec![3u64, 1], vec![1, 3], true),
        (OccupationFamily::ME, vec![2u64; 3], vec![2; 3], true),
        (OccupationFamily::ME, vec![2u64; 3], vec![2; 3], false),
        (OccupationFamily::ME, vec![3u64, 2, 1], vec![1, 2, 3], true),
        // B (M=2)
        (
            OccupationFamily::B { layers: 2 },
            vec![2u64, 2],
            vec![2, 2],
            true,
        ),
        (
            OccupationFamily::B { layers: 2 },
            vec![2u64; 3],
            vec![2; 3],
            false,
        ),
        // B (M=3)
        (
            OccupationFamily::B { layers: 3 },
            vec![3u64, 3, 3],
            vec![3, 3, 3],
            true,
        ),
        // W
        (
            OccupationFamily::W { layers: 1 },
            vec![2u64, 2],
            vec![2, 2],
            true,
        ),
        (
            OccupationFamily::W { layers: 1 },
            vec![2u64; 3],
            vec![2; 3],
            false,
        ),
        (
            OccupationFamily::W { layers: 2 },
            vec![3u64, 1],
            vec![1, 3],
            true,
        ),
        (
            OccupationFamily::W { layers: 2 },
            vec![3u64; 3],
            vec![3; 3],
            true,
        ),
    ];

    for (family, so, si, sl) in cases {
        local_fixed_e_kernel_checks(family, &so, &si, sl);
    }
}

#[test]
fn local_fixed_e_kernel_singleton_fiber_is_stationary() {
    // The §5 counterexample N=2, loops, s=[2,2], E=2: the fiber has
    // exactly two states and no in-fiber transition; the local kernel is
    // a pure self-loop on each — still exactly stationary.
    let family = OccupationFamily::ME;
    let states = enumerate_fiber(family, &[2, 2], &[2, 2], true);
    let n = 2;
    let base = mh_matrix(
        &states,
        n,
        family,
        true,
        &fixed_strength_log_weights(&states),
    );
    let e = 2usize;
    let fiber: Vec<usize> = states
        .iter()
        .enumerate()
        .filter(|(_, (s, _))| s.len() == e)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        fiber.len(),
        2,
        "expected exactly two states in the §5 fiber"
    );
    let local = local_fixed_e_matrix(&base, &states, e);
    assert_row_sums(&local, Some(&fiber));
    assert_pairwise_detailed_balance(&states, &local, &fiber, "§5 fiber");
    assert_stationarity(&states, &local, &fiber, "§5 fiber");
}

// ---------------------------------------------------------------------------
// Phase 5: edge-repair feasibility oracle (§24.1)
// ---------------------------------------------------------------------------

/// §24.1: for every enumerated start state **in the same connected
/// component** as the target fiber and every feasible edge target (derived
/// from the enumeration itself), the biased initialization repair must
/// reach the exact edge target preserving strengths, admissibility, and
/// family capacity.
///
/// Disconnected components are genuine 4-cycle-kernel limitations (e.g.
/// loopless perfect matchings, where every 4-cycle creates an
/// inadmissible self-loop); the plan documents those (§23/§41) and the
/// production orchestrator surfaces them as structured errors, so they
/// are silently skipped here and counted for the report.
#[test]
fn edge_repair_reaches_every_feasible_tiny_target() {
    use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
    use menobis_core::generation::microcanonical::occupation_mcmc::fixed_edges::{
        repair_to_edge_target, EdgeRepairConfig,
    };
    use menobis_core::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem;
    use menobis_core::generation::microcanonical::occupation_mcmc::state::StrengthState;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let cases: Vec<(OccupationFamily, Vec<OccNum>, Vec<OccNum>, bool)> = vec![
        (OccupationFamily::ME, vec![2u64, 2], vec![2, 2], true),
        (OccupationFamily::ME, vec![2u64, 2], vec![2, 2], false),
        (OccupationFamily::ME, vec![3u64, 1], vec![1, 3], true),
        (OccupationFamily::ME, vec![2u64; 3], vec![2; 3], true),
        (OccupationFamily::ME, vec![2u64; 3], vec![2; 3], false),
        (OccupationFamily::ME, vec![3u64, 2, 1], vec![1, 2, 3], true),
        (
            OccupationFamily::B { layers: 2 },
            vec![2u64, 2],
            vec![2, 2],
            true,
        ),
        (
            OccupationFamily::B { layers: 3 },
            vec![3u64, 3, 3],
            vec![3, 3, 3],
            true,
        ),
        (
            OccupationFamily::W { layers: 1 },
            vec![2u64, 2],
            vec![2, 2],
            true,
        ),
        (
            OccupationFamily::W { layers: 2 },
            vec![3u64; 3],
            vec![3; 3],
            true,
        ),
    ];

    let mut repaired_pairs = 0usize;
    let mut skipped_disconnected = 0usize;

    for (family, so, si, sl) in cases {
        let states = enumerate_fiber(family, &so, &si, sl);
        assert!(!states.is_empty(), "empty fiber for {family:?}");
        let mut edges: Vec<usize> = states.iter().map(|(s, _)| s.len()).collect();
        edges.sort_unstable();
        edges.dedup();

        let n = so.len();
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: sl,
        };
        let problem = FixedStrengthProblem::new(family, so.clone(), si.clone(), domain, vec![])
            .unwrap()
            .into_residual()
            .unwrap();

        // Connected components of the ordinary fixed-strength chain under
        // the 4-cycle proposal (positive probability in either direction).
        let base = mh_matrix(&states, n, family, sl, &fixed_strength_log_weights(&states));
        let comp = connected_components(&base);
        let k = states.len();

        for &e in &edges {
            // Components that contain at least one exact-E target state.
            let target_comps: std::collections::HashSet<usize> = (0..k)
                .filter(|&i| states[i].0.len() == e)
                .map(|i| comp[i])
                .collect();
            for (start_idx, (start_state, _)) in states.iter().enumerate() {
                if !target_comps.contains(&comp[start_idx]) {
                    // Genuinely disconnected under the 4-cycle kernel
                    // (e.g. loopless perfect matching components).
                    skipped_disconnected += 1;
                    continue;
                }
                let mut rng = StdRng::seed_from_u64(42 + start_idx as u64 * 7919);
                let mut state = StrengthState::new(n, start_state.clone());
                repair_to_edge_target(
                    &mut state,
                    &problem,
                    &mut rng,
                    e,
                    &EdgeRepairConfig::default(),
                )
                .unwrap_or_else(|err| {
                    panic!(
                        "repair failed: {family:?} sl={sl} start={start_state:?} target E={e}: {err}"
                    )
                });
                repaired_pairs += 1;
                assert_eq!(
                    state.occupied_count(),
                    e,
                    "{family:?} sl={sl} start={start_state:?} target E={e}"
                );
                // Exact strengths.
                let (mut co, mut ci) = (vec![0u64; n], vec![0u64; n]);
                for ((s, t), o) in state.iter_occupied() {
                    co[s as usize] += o;
                    ci[t as usize] += o;
                }
                assert_eq!(
                    co, so,
                    "{family:?} sl={sl} target E={e}: out-strength drift"
                );
                assert_eq!(ci, si, "{family:?} sl={sl} target E={e}: in-strength drift");
                // Domain validity.
                for ((s, t), _) in state.iter_occupied() {
                    assert!(
                        problem.domain.is_admissible(s, t),
                        "{family:?} sl={sl} target E={e}: inadmissible pair ({s},{t})"
                    );
                }
                // Family capacity.
                if let OccupationFamily::B { layers } = family {
                    for (_, o) in state.iter_occupied() {
                        assert!(
                            o <= layers as OccNum,
                            "{family:?} sl={sl} target E={e}: B occupation {o} > M={layers}"
                        );
                    }
                }
            }
        }
    }

    assert!(
        repaired_pairs > 0,
        "no (start, target) pair was repaired; the grid is vacuous"
    );
    eprintln!(
        "[edge-repair oracle] repaired {repaired_pairs} (start, target) pairs; \
         {skipped_disconnected} pairs skipped across disconnected components"
    );
}

// ---------------------------------------------------------------------------
// Phase 6: auxiliary kernel (§22.5) and bridge matrix (§22.6)
// ---------------------------------------------------------------------------

/// Shared mandatory tiny-case grid (§23).
fn mandatory_cases() -> Vec<(OccupationFamily, Vec<OccNum>, Vec<OccNum>, bool)> {
    vec![
        (OccupationFamily::ME, vec![2u64, 2], vec![2, 2], true),
        (OccupationFamily::ME, vec![2u64, 2], vec![2, 2], false),
        (OccupationFamily::ME, vec![3u64, 1], vec![1, 3], true),
        (OccupationFamily::ME, vec![2u64; 3], vec![2; 3], true),
        (OccupationFamily::ME, vec![2u64; 3], vec![2; 3], false),
        (OccupationFamily::ME, vec![3u64, 2, 1], vec![1, 2, 3], true),
        (
            OccupationFamily::B { layers: 2 },
            vec![2u64, 2],
            vec![2, 2],
            true,
        ),
        (
            OccupationFamily::B { layers: 3 },
            vec![3u64, 3, 3],
            vec![3, 3, 3],
            true,
        ),
        (
            OccupationFamily::W { layers: 1 },
            vec![2u64, 2],
            vec![2, 2],
            true,
        ),
        (
            OccupationFamily::W { layers: 2 },
            vec![3u64; 3],
            vec![3; 3],
            true,
        ),
    ]
}

/// §22.5: exact detailed balance of the auxiliary edge-biased MH kernel
/// `K_lambda` against `mu_lambda(t) ∝ pi_s(t) exp(−λ|E(t)−E_target|)`
/// across the **full** fixed-strength state space, for every feasible
/// edge target.
#[test]
fn auxiliary_kernel_exact_detailed_balance() {
    let lambda = 1.0;
    for (family, so, si, sl) in mandatory_cases() {
        let states = enumerate_fiber(family, &so, &si, sl);
        let n = so.len();
        let k = states.len();
        let mut edges: Vec<usize> = states.iter().map(|(s, _)| s.len()).collect();
        edges.sort_unstable();
        edges.dedup();
        for &e in &edges {
            let mu_log = aux_log_weights(&states, lambda, e);
            let aux = mh_matrix(&states, n, family, sl, &mu_log);
            let label = format!("aux {family:?} sl={sl} E={e}");
            assert_row_sums(&aux, None);
            // Pairwise DB over the full space (all pairs, not just fiber).
            for i in 0..k {
                let wi = mu_log[i].exp();
                for j in 0..k {
                    let wj = mu_log[j].exp();
                    let lhs = wi * aux[i][j];
                    let rhs = wj * aux[j][i];
                    if lhs == 0.0 && rhs == 0.0 {
                        continue;
                    }
                    let rel = (lhs - rhs).abs() / lhs.abs().max(rhs.abs());
                    assert!(
                        rel < 1e-9,
                        "{label}: auxiliary DB violated {i}->{j}: {lhs:.3e} vs {rhs:.3e} (rel {rel:.3e})"
                    );
                }
            }
        }
    }
}

/// §22.6: exact bridge matrix by dynamic propagation — row sums on the
/// fiber, pairwise detailed balance against the conditional target
/// `pi_(s,E)`, and the mandatory N=2 counterexample is connected.
#[test]
fn bridge_kernel_exact_detailed_balance() {
    let lambda = 1.0;
    let max_steps = 16usize;
    let mut connected_fibers = 0usize;
    for (family, so, si, sl) in mandatory_cases() {
        let states = enumerate_fiber(family, &so, &si, sl);
        let n = so.len();
        let mut edges: Vec<usize> = states.iter().map(|(s, _)| s.len()).collect();
        edges.sort_unstable();
        edges.dedup();
        for &e in &edges {
            let b = bridge_matrix(&states, n, family, sl, e, lambda, max_steps);
            let fiber: Vec<usize> = states
                .iter()
                .enumerate()
                .filter(|(_, (s, _))| s.len() == e)
                .map(|(i, _)| i)
                .collect();
            let label = format!("bridge {family:?} sl={sl} E={e}");
            assert_row_sums(&b, Some(&fiber));
            assert_pairwise_detailed_balance(&states, &b, &fiber, &label);
            if fiber.len() > 1 {
                connected_fibers += 1;
            }
        }
    }
    assert!(connected_fibers > 0, "no non-singleton fiber checked");
}

/// The mandatory §5 counterexample: ME N=2, self-loops, s=[2,2]/[2,2],
/// E=2.  The bridge matrix must connect the two fiber states (positive
/// probability in both directions), which the local kernel cannot.
#[test]
fn bridge_matrix_connects_counterexample() {
    let family = OccupationFamily::ME;
    let states = enumerate_fiber(family, &[2, 2], &[2, 2], true);
    let n = 2;
    let e = 2usize;
    let fiber: Vec<usize> = states
        .iter()
        .enumerate()
        .filter(|(_, (s, _))| s.len() == e)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(fiber.len(), 2, "§5 fiber must contain exactly two states");

    let b = bridge_matrix(&states, n, family, true, e, 1.0, 16);
    let (a, bb) = (fiber[0], fiber[1]);
    assert!(
        b[a][bb] > 0.0 && b[bb][a] > 0.0,
        "bridge must connect the §5 states: B[A][B]={:.4}, B[B][A]={:.4}",
        b[a][bb],
        b[bb][a]
    );

    // The local kernel alone must be a pure self-loop on this fiber.
    let base = mh_matrix(
        &states,
        n,
        family,
        true,
        &fixed_strength_log_weights(&states),
    );
    assert_eq!(base[a][bb], 0.0, "local kernel must not connect §5 states");
    assert_eq!(base[bb][a], 0.0);
}
