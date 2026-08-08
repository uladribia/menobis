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
//!
//! # Hot path
//!
//! The four cells are always distinct (`a \ne c`, `b \ne d`), so no delta
//! merging is needed and the step performs **no heap allocation**.

use rand::Rng;

use super::domain::PairDomain;
use super::state::StrengthState;
use super::target::StrengthTarget;
use crate::generation::microcanonical::mcmc::McmcOutcome;
use crate::OccNum;

/// Four (cell, delta) pairs of one cycle proposal.
type Deltas = [(u64, u64, i64); 4];

/// Build a 4-cycle delta set from two sources, two targets, and a sign.
fn build_cycle4(a: u64, c: u64, b: u64, d: u64, sign: i64) -> Deltas {
    [(a, b, sign), (c, d, sign), (a, d, -sign), (c, b, -sign)]
}

/// Perform one 4-cycle MCMC step (allocation-free).
///
/// Validation is inlined: each affected cell is checked for negativity,
/// domain admissibility, and family capacity before the target ratio is
/// computed.  If valid and accepted, the four cells are written directly
/// via [`StrengthState::set`].
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

    let deltas = build_cycle4(a, c, b, d, sign);
    let cap = domain.capacity(target.family);

    // ---- Validation + target ratio in one pass (no allocation) ----
    let mut delta_log_pi = 0.0f64;
    let mut applied = false;
    for &(src, tgt, d) in &deltas {
        let old = state.get(src, tgt);
        let new = (old as i64 + d) as OccNum;
        if old as i64 + d < 0 {
            return McmcOutcome::Held;
        }
        if !domain.is_admissible(src, tgt) {
            return McmcOutcome::Held;
        }
        if new > cap {
            return McmcOutcome::Held;
        }
        match target.delta_log_weight(src, tgt, old, new) {
            Some(w) => delta_log_pi += w,
            None => return McmcOutcome::Held,
        }
        if d != 0 {
            applied = true;
        }
    }
    if !applied {
        return McmcOutcome::Held;
    }

    // ---- Metropolis acceptance ----
    if delta_log_pi < 0.0 {
        let log_u = (rng.random::<f64>() + f64::MIN_POSITIVE).ln();
        if log_u >= delta_log_pi {
            return McmcOutcome::Rejected;
        }
    }

    // ---- Apply directly (cells are distinct, no merging needed) ----
    for &(src, tgt, d) in &deltas {
        let old = state.get(src, tgt);
        state.set(src, tgt, (old as i64 + d) as OccNum);
    }

    McmcOutcome::Accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::microcanonical::occupation_mcmc::initializer::initialize_table;
    use crate::model::family::OccupationFamily;
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
        let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
        let target = StrengthTarget::new(OccupationFamily::ME);
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
        let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
        let target = StrengthTarget::new(OccupationFamily::ME);
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
        let mut state = make_state(n, &so, &si, OccupationFamily::ME, false);
        let target = StrengthTarget::new(OccupationFamily::ME);
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
        let mut state = make_state(n, &so, &si, OccupationFamily::B { layers }, true);
        let target = StrengthTarget::new(OccupationFamily::B { layers });
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
        let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
        let target = StrengthTarget::new(OccupationFamily::ME);
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

    #[test]
    fn cycle4_reproducible() {
        let n = 4;
        let so = vec![3u64; 4];
        let si = vec![3u64; 4];

        let run = |seed: u64| -> Vec<OccNum> {
            let mut state = make_state(n, &so, &si, OccupationFamily::ME, true);
            let target = StrengthTarget::new(OccupationFamily::ME);
            let domain = PairDomain::Complete {
                node_count: n,
                self_loops: true,
            };
            let mut rng = StdRng::seed_from_u64(seed);
            for _ in 0..100 {
                cycle4_step(&mut state, &target, &domain, &mut rng);
            }
            let mut pairs = state.iter_occupied().collect::<Vec<_>>();
            pairs.sort_unstable();
            pairs
                .into_iter()
                .flat_map(|((s, t), o)| vec![s, t, o])
                .collect()
        };

        assert_eq!(run(42), run(42));
    }
}
