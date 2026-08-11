//! Heavy: exact Hastings-kernel verification for the occupied-cell 4-cycle MCMC.
//!
//! **Purpose:** Directly validate that `occupied_cycle4_step` implements the
//! correct Metropolis–Hastings kernel by comparing the *exact* transition
//! matrix (computed by enumerating all feasible proposals) against the
//! *empirical* transition matrix (measured by running the kernel 20 000
//! times per state).
//!
//! # Method
//!
//! 1. **Enumerate** all feasible occupation states (N=3, ME, self-loops,
//!    s_out=s_in=[2,2,2]) via recursive backtracking.
//! 2. **Exact kernel**: For each state S, enumerate every valid unordered
//!    pair of occupied cells ┌(a,b), (c,d)┐ with a≠c and b≠d (the 4-cycle
//!    proposal).  For each eligible pair compute:
//!    - q_fwd = (1/m)(1/v_ab + 1/v_cd) — the proposal probability (§22–§23)
//!    - Δlogπ — the four-cell target-log-weight change
//!    - q_rev = (1/m′)(1/v′_ad + 1/v′_cb) — reverse proposal probability
//!    - acceptance = min(1, exp(Δlogπ + log q_rev − log q_fwd))
//!    - Accumulate P_exact(S→S′).
//! 3. **Empirical kernel**: From each state S run the actual
//!    `occupied_cycle4_step` 20 000 times (seeded, reproducible) and count
//!    destination-state frequencies.
//! 4. **Assert** that every P_emp(S→S′) matches P_exact(S→S′) within
//!    absolute tolerance 0.01 **or** relative tolerance 10 % (whichever is
//!    larger, but only checked for transitions with P_exact ≥ 0.005).
//!
//! # Tolerances
//!
//! - `atol = 0.01` (±1 % absolute error is acceptable binomial noise).
//! - `rtol = 0.10` (10 % relative for small probabilities).
//! - Only tested for `P_exact ≥ 0.005` (below that we lack statistical
//!   power even with 20 000 trials).
//!
//! This test lives in `menobis-test-oracles` because it is heavy
//! (~22 states × 20 000 proposals = 440 000 MCMC steps per run).

use std::collections::HashMap;

use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::generation::microcanonical::occupation_mcmc::move_cycle::occupied_cycle4_step;
use menobis_core::generation::microcanonical::occupation_mcmc::state::StrengthState;
use menobis_core::generation::microcanonical::occupation_mcmc::target::StrengthTarget;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;
use rand::rngs::StdRng;
use rand::SeedableRng;

// --------------------------------------------------------------------------
// Type aliases
// --------------------------------------------------------------------------

/// A sorted list of positive-occupation pairs: `((src, tgt), occ)`.
type OccupiedState = Vec<((u64, u64), OccNum)>;

/// We index states by their index into the master list.
type StateIdx = usize;

// --------------------------------------------------------------------------
// 1.  State enumeration (reuses the pattern from fixed_strength_enumeration)
// --------------------------------------------------------------------------

/// Enumerate all feasible ME occupation states for the given strength
/// sequences and self-loop policy.
///
/// Returns `(state, log_microcanonical_weight)`.  For ME the weight is
/// `Σ −ln(t_ij!)` (the multinomial count degeneracy).
fn enumerate_me_states(
    s_out: &[OccNum],
    s_in: &[OccNum],
    self_loops: bool,
) -> Vec<(OccupiedState, f64)> {
    let n = s_out.len();
    let mut results = Vec::new();
    let cells: Vec<(u64, u64)> = (0..n as u64)
        .flat_map(|i| (0..n as u64).map(move |j| (i, j)))
        .filter(|&(i, j)| self_loops || i != j)
        .collect();

    fn recurse(
        idx: usize,
        cells: &[(u64, u64)],
        remaining_out: &mut [OccNum],
        remaining_in: &mut [OccNum],
        current: &mut OccupiedState,
        results: &mut Vec<(OccupiedState, f64)>,
    ) {
        if idx == cells.len() {
            if remaining_out.iter().all(|&s| s == 0) && remaining_in.iter().all(|&s| s == 0) {
                let log_weight: f64 = current
                    .iter()
                    .map(|&(_, occ)| -libm::lgamma((occ as f64) + 1.0))
                    .sum();
                results.push((current.clone(), log_weight));
            }
            return;
        }
        let (src, tgt) = cells[idx];
        let max_possible = remaining_out[src as usize].min(remaining_in[tgt as usize]);
        for occ in 0..=max_possible {
            remaining_out[src as usize] -= occ;
            remaining_in[tgt as usize] -= occ;
            if occ > 0 {
                current.push(((src, tgt), occ));
            }
            recurse(
                idx + 1,
                cells,
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
        &mut remaining_out,
        &mut remaining_in,
        &mut current,
        &mut results,
    );
    results
}

// --------------------------------------------------------------------------
// 2.  Exact transition kernel
// --------------------------------------------------------------------------

/// Row and column occupied-pair counts from an `OccupiedState`.
struct OccupancyCounts {
    row: Vec<usize>,
    col: Vec<usize>,
    total: usize,
}

fn occupancy_counts(state: &OccupiedState, n: usize) -> OccupancyCounts {
    let mut row = vec![0usize; n];
    let mut col = vec![0usize; n];
    for &((src, tgt), _) in state {
        row[src as usize] += 1;
        col[tgt as usize] += 1;
    }
    OccupancyCounts {
        total: state.len(),
        row,
        col,
    }
}

/// Build a hash-map from `(src, tgt)` to occupation for O(1) lookups.
fn occ_map(state: &OccupiedState) -> HashMap<(u64, u64), OccNum> {
    state.iter().copied().collect()
}

/// Compute the destination state after applying a fixed-direction 4-cycle
/// proposal: decrement (a,b) and (c,d); increment (a,d) and (c,b).
fn apply_proposal(state: &OccupiedState, a: u64, b: u64, c: u64, d: u64) -> OccupiedState {
    let mut map: HashMap<(u64, u64), i64> = HashMap::new();
    for &((src, tgt), occ) in state {
        map.insert((src, tgt), occ as i64);
    }
    // Apply deltas
    *map.entry((a, b)).or_insert(0) -= 1;
    *map.entry((c, d)).or_insert(0) -= 1;
    *map.entry((a, d)).or_insert(0) += 1;
    *map.entry((c, b)).or_insert(0) += 1;

    // Convert to OccupiedState (remove zeros, sort)
    let mut result: OccupiedState = map
        .into_iter()
        .filter(|&(_, v)| v > 0)
        .map(|(k, v)| (k, v as OccNum))
        .collect();
    result.sort_unstable();
    result
}

/// Compute the exact transition probability from `state` to every other
/// enumerated state, together with the exact self-loop probability.
///
/// **Hastings formula** (§23):
///
/// q_fwd = (1/m) * (1/v_ab + 1/v_cd)
/// α = Δlogπ + log(q_rev) − log(q_fwd)
/// A = min(1, exp(α))
///
/// where v_ab = m − row[a] − col[b] + 1 counts the available P2 candidates
/// for a given P1=(a,b), and m = number of occupied pairs.
fn exact_transition_probs(
    state: &OccupiedState,
    state_idx: StateIdx,
    all_states: &[OccupiedState],
    n: usize,
    family: OccupationFamily,
) -> Vec<f64> {
    let k = all_states.len();
    let mut probs = vec![0.0_f64; k];
    let occ = occupancy_counts(state, n);
    let m = occ.total;

    if m < 2 {
        // No valid proposal possible → chain stays put with probability 1.
        probs[state_idx] = 1.0;
        return probs;
    }

    let target = StrengthTarget::new(family);
    let map = occ_map(state);

    // Enumerate every unordered pair of distinct occupied cells.
    for i in 0..m {
        let ((a, b), occ_ab) = state[i];
        for j in (i + 1)..m {
            let ((c, d), occ_cd) = state[j];

            // The 4-cycle requires distinct sources and targets.
            if a == c || b == d {
                continue;
            }

            // --- v_ab, v_cd (availability counts for each ordering) ---
            let v_ab = m as i64 - occ.row[a as usize] as i64 - occ.col[b as usize] as i64 + 1;
            let v_cd = m as i64 - occ.row[c as usize] as i64 - occ.col[d as usize] as i64 + 1;

            // Both orderings must be feasible (mirrors the code guard).
            if v_ab <= 0 || v_cd <= 0 {
                continue;
            }

            // --- Compute Δlogπ ---
            // Decrement cells: (a,b): occ_ab → occ_ab-1, (c,d): occ_cd → occ_cd-1
            // Increment cells: (a,d): occ_ad → occ_ad+1, (c,b): occ_cb → occ_cb+1
            let occ_ad = map.get(&(a, d)).copied().unwrap_or(0);
            let occ_cb = map.get(&(c, b)).copied().unwrap_or(0);

            let new_ab = occ_ab - 1;
            let new_cd = occ_cd - 1;
            let new_ad = occ_ad + 1;
            let new_cb = occ_cb + 1;

            let delta_log_pi = target.delta_log_weight(a, b, occ_ab, new_ab).unwrap()
                + target.delta_log_weight(c, d, occ_cd, new_cd).unwrap()
                + target.delta_log_weight(a, d, occ_ad, new_ad).unwrap()
                + target.delta_log_weight(c, b, occ_cb, new_cb).unwrap();

            // --- Reverse availability counts ---
            // Compute m', row'_a, row'_c, col'_b, col'_d after the proposal.
            let leaves_ab = if occ_ab == 1 { 1 } else { 0 };
            let leaves_cd = if occ_cd == 1 { 1 } else { 0 };
            let enters_ad = if occ_ad == 0 { 1 } else { 0 };
            let enters_cb = if occ_cb == 0 { 1 } else { 0 };
            let leaves = leaves_ab + leaves_cd;
            let enters = enters_ad + enters_cb;

            // These mirror the code in move_cycle.rs step 8.
            let m_prime = m as i64 + enters - leaves;
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

            // --- Log-proposal probabilities ---
            let m_f = m as f64;
            let m_prime_f = m_prime as f64;
            let q_fwd = (1.0 / v_ab as f64 + 1.0 / v_cd as f64) / m_f;
            let q_rev = (1.0 / v_ad_prime as f64 + 1.0 / v_cb_prime as f64) / m_prime_f;

            // --- Hastings acceptance ---
            let log_alpha = delta_log_pi + q_rev.ln() - q_fwd.ln();
            let acceptance = if log_alpha >= 0.0 {
                1.0
            } else {
                log_alpha.exp()
            };

            let prob = q_fwd * acceptance;
            if prob <= 0.0 {
                continue;
            }

            // --- Destination state ---
            let dest = apply_proposal(state, a, b, c, d);
            let dest_idx = all_states
                .iter()
                .position(|s| s == &dest)
                .expect("destination state not in enumeration!");

            probs[dest_idx] += prob;
        }
    }

    // --- Self-loop probability ---
    let sum_trans = probs.iter().sum::<f64>();
    debug_assert!(
        sum_trans <= 1.0 + 1e-12,
        "transition probabilities sum to {sum_trans} > 1"
    );
    probs[state_idx] = 1.0 - sum_trans;

    probs
}

// --------------------------------------------------------------------------
// 3.  Empirical transition kernel
// --------------------------------------------------------------------------

/// Measure the empirical transition kernel by running
/// `occupied_cycle4_step` from `state` for `trials` independent seeds.
fn empirical_transition_probs(
    state: &OccupiedState,
    _state_idx: StateIdx,
    all_states: &[OccupiedState],
    n: usize,
    family: OccupationFamily,
    trials: usize,
    base_seed: u64,
) -> Vec<f64> {
    let k = all_states.len();
    let mut counts = vec![0u64; k];

    let domain = PairDomain::Complete {
        node_count: n,
        self_loops: true,
    };
    let target = StrengthTarget::new(family);

    for t in 0..trials {
        // Build a fresh StrengthState from the abstract state.
        let pairs: Vec<((u64, u64), OccNum)> = state.iter().map(|&(p, o)| (p, o)).collect();
        let mut s = StrengthState::new(n, pairs);
        let mut rng = StdRng::seed_from_u64(base_seed + t as u64 * 7919); // prime spacing

        let outcome = occupied_cycle4_step(&mut s, &target, &domain, &mut rng);

        // Convert the resulting state to OccupiedState for lookup.
        let mut state_after: OccupiedState = s.iter_occupied().collect();
        state_after.sort_unstable();

        let idx = all_states
            .iter()
            .position(|s| s == &state_after)
            .unwrap_or_else(|| {
                panic!(
                    "state after step not found in enumeration! state={state_after:?}, outcome={outcome:?}"
                );
            });
        counts[idx] += 1;
    }

    let trials_f = trials as f64;
    counts.into_iter().map(|c| c as f64 / trials_f).collect()
}

// --------------------------------------------------------------------------
// 4.  Main test
// --------------------------------------------------------------------------

#[test]
fn occupied_cell_hastings_kernel_exact_vs_empirical_n3() {
    let n = 3;
    let s_out = vec![2u64; 3];
    let s_in = vec![2u64; 3];
    let family = OccupationFamily::ME;

    // ---- 1. Enumerate all states ----
    let raw_states = enumerate_me_states(&s_out, &s_in, true);
    let all_states: Vec<OccupiedState> = raw_states.into_iter().map(|(s, _w)| s).collect();
    let k = all_states.len();
    assert!(k >= 3, "need at least 3 states, found {k}");

    eprintln!("[hastings-verify] enumerated {k} feasible states (N={n}, s=2)");

    // ---- 2. Compute exact transition matrix ----
    let mut exact = vec![vec![0.0_f64; k]; k];
    for (si, state) in all_states.iter().enumerate() {
        exact[si] = exact_transition_probs(state, si, &all_states, n, family);
    }

    // ---- 3. Compute empirical transition matrix (20k trials per state) ----
    let trials = 20_000;
    let mut empirical = vec![vec![0.0_f64; k]; k];
    for (si, state) in all_states.iter().enumerate() {
        eprintln!(
            "[hastings-verify] empirical: state {si}/{k} ({m} occupied pairs)",
            m = state.len()
        );
        empirical[si] = empirical_transition_probs(
            state,
            si,
            &all_states,
            n,
            family,
            trials,
            42 + si as u64 * 31337,
        );
    }

    // ---- 4. Assert agreement ----
    let atol = 0.01_f64; // absolute tolerance
    let rtol = 0.10_f64; // relative tolerance (10 %)
    let min_prob = 0.005; // skip transitions below this (insufficient power)

    let mut max_abs_err = 0.0_f64;
    let mut max_rel_err = 0.0_f64;
    let mut worst_pair = (0usize, 0usize);

    for si in 0..k {
        for sj in 0..k {
            let p_exact = exact[si][sj];
            let p_emp = empirical[si][sj];

            // Skip transitions too small to measure reliably.
            if p_exact < min_prob {
                continue;
            }

            let abs_err = (p_emp - p_exact).abs();
            let rel_err = abs_err / p_exact;

            if abs_err > max_abs_err {
                max_abs_err = abs_err;
                worst_pair = (si, sj);
            }
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }

            let allowed = atol.max(p_exact * rtol);
            assert!(
                abs_err <= allowed,
                "state {si} → {sj}: P_exact={p_exact:.6}, P_emp={p_emp:.6}, \
                 abs_err={abs_err:.6} > allowed={allowed:.6} (atol={atol}, rtol={rtol})"
            );
        }
    }

    eprintln!(
        "[hastings-verify] PASS: {k}×{k} transition matrix verified. \
         max_abs_err={max_abs_err:.6} at ({}→{}), max_rel_err={max_rel_err:.4}",
        worst_pair.0, worst_pair.1
    );
}
