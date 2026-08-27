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
) -> Vec<Vec<f64>> {
    let k = states.len();
    let cap = family_capacity(family);
    let index = state_index_map(states);
    let mut p = vec![vec![0.0_f64; k]; k];

    for (si, (state, lw)) in states.iter().enumerate() {
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
                let log_alpha = (states[di].1 - lw) + q_rev.ln() - q_fwd.ln();
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
    let base = mh_matrix(&states, n, family, sl);

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
    let base = mh_matrix(&states, n, family, true);
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
