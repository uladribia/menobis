//! 4-cycle symmetric MCMC kernel for fixed-strength sampling.
//!
//! The basic move selects two distinct source nodes \(a \ne c\) and two
//! distinct target nodes \(b \ne d\), chooses a sign \(s \in \{+1, -1\}\)
//! uniformly, and applies the occupation deltas:
//!
//! \[
//! t_{ab} \to t_{ab} + s,\quad
//! t_{cd} \to t_{cd} + s,\quad
//! t_{ad} \to t_{ad} - s,\quad
//! t_{cb} \to t_{cb} - s.
//! \]
//!
//! Every row and column receives one \(+s\) and one \(-s\), so out- and
//! in-strengths are preserved exactly.  The proposal is symmetric because
//! node indices are chosen uniformly, so the Metropolis acceptance ratio
//! depends only on the target probability ratio.

use rand::Rng;

use super::domain::PairDomain;
use super::state::StrengthState;
use super::target::StrengthTarget;
use crate::generation::microcanonical::mcmc::McmcOutcome;
use crate::OccNum;

/// A 4-cycle delta before validation.
struct Cycle4Proposal {
    deltas: [(u64, u64, i64); 4],
}

/// Build a 4-cycle proposal from two source nodes, two target nodes, and a sign.
fn build_cycle4(a: u64, c: u64, b: u64, d: u64, sign: i64) -> Cycle4Proposal {
    Cycle4Proposal {
        deltas: [(a, b, sign), (c, d, sign), (a, d, -sign), (c, b, -sign)],
    }
}

/// Validate a 4-cycle proposal against the current state and domain.
fn validate_cycle4(
    proposal: &Cycle4Proposal,
    state: &StrengthState,
    domain: &PairDomain,
    cap: OccNum,
) -> bool {
    let mut net_delta: std::collections::HashMap<(u64, u64), i64> =
        std::collections::HashMap::new();
    for &(src, tgt, d) in &proposal.deltas {
        *net_delta.entry((src, tgt)).or_insert(0) += d;
    }

    let mut is_noop = true;
    for (&(src, tgt), &d) in &net_delta {
        if d == 0 {
            continue;
        }
        is_noop = false;
        let old = state.get(src, tgt);
        let new = old as i64 + d;

        // Negative occupation check.
        if new < 0 {
            return false;
        }
        #[allow(clippy::cast_sign_loss)]
        let new_occ = new as OccNum;

        // Domain admissibility check.
        if !domain.is_admissible(src, tgt) {
            return false;
        }

        // Capacity check (B layers, etc.).
        if new_occ > cap {
            return false;
        }
    }

    // No-op check.
    if is_noop {
        return false;
    }

    true
}

/// Perform one 4-cycle MCMC step.
pub fn cycle4_step(
    state: &mut StrengthState,
    target: &StrengthTarget,
    domain: &PairDomain,
    rng: &mut impl Rng,
) -> McmcOutcome {
    let n = state.node_count;
    if n < 2 {
        return McmcOutcome::Held;
    }

    // Choose two distinct source nodes.
    let a = rng.random_range(0..n) as u64;
    let mut c = rng.random_range(0..n - 1) as u64;
    if c >= a {
        c += 1;
    }

    // Choose two distinct target nodes.
    let b = rng.random_range(0..n) as u64;
    let mut d = rng.random_range(0..n - 1) as u64;
    if d >= b {
        d += 1;
    }

    // Choose sign uniformly.
    let sign = if rng.random_bool(0.5) { 1i64 } else { -1i64 };

    let proposal = build_cycle4(a, c, b, d, sign);
    let cap = domain.capacity(target.family);

    if !validate_cycle4(&proposal, state, domain, cap) {
        return McmcOutcome::Held;
    }

    // Merge overlapping deltas.
    let mut merged: std::collections::HashMap<(u64, u64), i64> = std::collections::HashMap::new();
    for &(src, tgt, d) in &proposal.deltas {
        *merged.entry((src, tgt)).or_insert(0) += d;
    }

    // Compute Δlogπ.
    let mut delta_log_pi = 0.0f64;
    for (&(src, tgt), &d) in &merged {
        if d == 0 {
            continue;
        }
        let old = state.get(src, tgt);
        let new = (old as i64 + d) as OccNum;
        match target.delta_log_weight(src, tgt, old, new) {
            Some(w) => delta_log_pi += w,
            None => return McmcOutcome::Held,
        }
    }

    // Metropolis acceptance.
    if delta_log_pi < 0.0 {
        let log_u = (rng.random::<f64>() + f64::MIN_POSITIVE).ln();
        if log_u >= delta_log_pi {
            return McmcOutcome::Rejected;
        }
    }

    // Apply the move.
    let deltas_vec: Vec<_> = merged
        .into_iter()
        .filter(|&(_, d)| d != 0)
        .map(|((src, tgt), d)| (src, tgt, d))
        .collect();
    state.apply_deltas(&deltas_vec);

    McmcOutcome::Accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::OccupationFamily;
    use crate::generation::microcanonical::fixed_strength::initializer::initialize_table;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_state(
        n: usize,
        so: &[OccNum],
        si: &[OccNum],
        family: OccupationFamily,
        sl: bool,
    ) -> StrengthState {
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: sl,
        };
        let table = initialize_table(so, si, family, &domain).unwrap();
        StrengthState::new(n, table)
    }

    #[test]
    fn cycle4_preserves_strengths_me() {
        let n = 4;
        let so = vec![5u64, 3, 7, 2];
        let si = vec![4u64, 6, 3, 4];
        let mut state = make_state(n, &so, &si, OccupationFamily::Poisson, true);
        let target = StrengthTarget::new(OccupationFamily::Poisson);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            cycle4_step(&mut state, &target, &domain, &mut rng);
            assert_eq!(state.out_strengths, so, "out-strengths changed");
            assert_eq!(state.in_strengths, si, "in-strengths changed");
        }
    }

    #[test]
    fn cycle4_acceptance_rate_nonzero() {
        let n = 5;
        let so = vec![10u64; 5];
        let si = vec![10u64; 5];
        let mut state = make_state(n, &so, &si, OccupationFamily::Poisson, true);
        let target = StrengthTarget::new(OccupationFamily::Poisson);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(99);
        let mut accepted = 0u64;
        let trials = 500;
        for _ in 0..trials {
            if cycle4_step(&mut state, &target, &domain, &mut rng) == McmcOutcome::Accepted {
                accepted += 1;
            }
        }
        assert!(
            accepted > trials / 10,
            "acceptance rate too low: {accepted}/{trials}"
        );
    }

    #[test]
    fn cycle4_no_self_loops() {
        let n = 4;
        let so = vec![5u64; 4];
        let si = vec![5u64; 4];
        let mut state = make_state(n, &so, &si, OccupationFamily::Poisson, false);
        let target = StrengthTarget::new(OccupationFamily::Poisson);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: false,
        };
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..200 {
            cycle4_step(&mut state, &target, &domain, &mut rng);
            for &(src, tgt) in state.occupied_pairs() {
                assert_ne!(src, tgt, "self-loop appeared");
            }
        }
    }

    #[test]
    fn cycle4_b_capacity() {
        let n = 3;
        let layers = 3u32;
        let so = vec![4u64; 3];
        let si = vec![4u64; 3];
        let mut state = make_state(n, &so, &si, OccupationFamily::Binomial(layers), true);
        let target = StrengthTarget::new(OccupationFamily::Binomial(layers));
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..200 {
            cycle4_step(&mut state, &target, &domain, &mut rng);
            for (_, occ) in state.iter_occupied() {
                assert!(
                    occ <= layers as OccNum,
                    "B occupation {occ} exceeds M={layers}"
                );
            }
        }
    }

    #[test]
    fn cycle4_occupied_pairs_change() {
        let n = 4;
        let so = vec![3u64; 4];
        let si = vec![3u64; 4];
        let mut state = make_state(n, &so, &si, OccupationFamily::Poisson, true);
        let target = StrengthTarget::new(OccupationFamily::Poisson);
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let initial = state.occupied_count();
        for _ in 0..500 {
            cycle4_step(&mut state, &target, &domain, &mut rng);
        }
        assert!(
            state.occupied_count() != initial,
            "occupied pairs unchanged after 500 moves"
        );
    }
}
