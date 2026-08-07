//! Exact enumeration of small occupation-number state spaces.
//!
//! These functions enumerate every feasible occupation configuration for
//! tiny systems and compute exact (log) probabilities.  Use them to
//! validate MCMC samplers by comparing empirical frequencies against
//! exact enumeration.

use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;

/// A fully-specified occupation state with its log weight.
#[derive(Clone, Debug)]
pub struct WeightedState {
    /// Positive occupations, one per occupied pair.
    pub occupations: Vec<OccNum>,
    /// Log of the unnormalized degeneracy weight.
    pub log_weight: f64,
}

/// Enumerate all positive-occupation vectors for a fixed-(E,T) problem
/// under the given family.
///
/// Enumerates all compositions of `total` into `e` positive parts,
/// computes `∑ ln d_F(t_i)` for each.
///
/// # Panics
///
/// Panics if `total < e` or `e == 0`.
pub fn enumerate_fixed_total(
    family: OccupationFamily,
    e: usize,
    total: OccNum,
) -> Vec<WeightedState> {
    assert!(e > 0, "e must be positive");
    assert!(total >= e as OccNum, "total must be >= e");
    let mut results = Vec::new();
    let mut current = vec![0u64; e];
    enumerate_compositions(family, total, e, 0, &mut current, &mut results);
    results
}

fn enumerate_compositions(
    family: OccupationFamily,
    remaining: OccNum,
    slots: usize,
    idx: usize,
    current: &mut Vec<OccNum>,
    results: &mut Vec<WeightedState>,
) {
    if idx == slots - 1 {
        current[idx] = remaining;
        let log_weight: f64 = current.iter().map(|&t| family.log_base_measure(t)).sum();
        results.push(WeightedState {
            occupations: current.clone(),
            log_weight,
        });
        return;
    }
    // Minimum 1 per remaining slot
    let min_for_rest = (slots - idx - 1) as OccNum;
    let max_here = remaining - min_for_rest;
    for v in 1..=max_here {
        current[idx] = v;
        enumerate_compositions(family, remaining - v, slots, idx + 1, current, results);
    }
}

/// Compute the normalized exact probability of each state.
pub fn normalize_states(states: &[WeightedState]) -> Vec<f64> {
    let max_log = states
        .iter()
        .map(|s| s.log_weight)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = states
        .iter()
        .map(|s| (s.log_weight - max_log).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|w| w / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use menobis_core::model::family::OccupationFamily;

    #[test]
    fn enumerate_me_small() {
        // ME, E=2, T=3 → compositions (1,2), (2,1)
        let states = enumerate_fixed_total(OccupationFamily::ME, 2, 3);
        assert_eq!(states.len(), 2);

        let probs = normalize_states(&states);
        // For ME: d(t1,t2) = 1/(t1! t2!)
        // (1,2): 1/(1! 2!) = 1/2
        // (2,1): 1/(2! 1!) = 1/2
        // Both equal weight → each prob = 0.5
        for p in &probs {
            assert!((p - 0.5).abs() < 1e-12, "expected 0.5, got {p}");
        }
    }

    #[test]
    fn enumerate_b_small() {
        // B(3), E=2, T=3 → (1,2), (2,1) since max per cell is 3
        let states = enumerate_fixed_total(OccupationFamily::B { layers: 3 }, 2, 3);
        assert_eq!(states.len(), 2);

        let probs = normalize_states(&states);
        // d(t) = C(3,t)
        // (1,2): C(3,1)*C(3,2) = 3*3 = 9
        // (2,1): C(3,2)*C(3,1) = 3*3 = 9
        // Both equal → each 0.5
        for p in &probs {
            assert!((p - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn enumerate_w_small() {
        // W(2), E=2, T=3 → (1,2), (2,1)
        let states = enumerate_fixed_total(OccupationFamily::W { layers: 2 }, 2, 3);
        assert_eq!(states.len(), 2);

        let probs = normalize_states(&states);
        // d(t) = C(1+t, t) = t+1
        // (1,2): (1+1)*(2+1) = 2*3 = 6
        // (2,1): (2+1)*(1+1) = 3*2 = 6
        // Both equal → each 0.5
        for p in &probs {
            assert!((p - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn enumerate_me_three_cells() {
        // ME, E=3, T=4 → compositions of 4 into 3 positive parts
        let states = enumerate_fixed_total(OccupationFamily::ME, 3, 4);
        assert_eq!(states.len(), 3); // C(3,2)=3

        let probs = normalize_states(&states);
        // (1,1,2): 1/(1!1!2!) = 1/2
        // (1,2,1): 1/2
        // (2,1,1): 1/2
        // All equal → each 1/3
        for p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn validate_me_fixed_et_distribution() {
        // Use the oracle to validate the core's sample_me_fixed_et
        use menobis_core::generation::microcanonical::sample_me_fixed_et;
        use std::collections::HashMap;

        // N=3, self_loops=false, E=2, T=3
        // All admissible pairs: 3*2=6 ordered pairs
        // We enumerate all E=2 supports and occupations
        // For ME, support is uniform, occupations follow the enumeration above.
        // Since both compositions have equal weight, each support should
        // be equally likely regardless of occupation values.
        let trials = 6000;
        let mut counts: HashMap<Vec<(u64, u64)>, u64> = HashMap::new();

        for seed in 0..trials {
            let net = sample_me_fixed_et(3, false, 2, 3, seed).unwrap();
            let mut edges: Vec<_> = net.sources.iter().zip(net.targets.iter()).collect();
            edges.sort_unstable();
            let key: Vec<(u64, u64)> = edges.into_iter().map(|(&s, &t)| (s, t)).collect();
            *counts.entry(key).or_default() += 1;
        }

        // Number of possible supports: C(6,2) = 15
        let n_supports = 15;
        let expected = trials as f64 / n_supports as f64;
        for count in counts.values() {
            let ratio = *count as f64 / expected;
            assert!(
                ratio > 0.5 && ratio < 1.5,
                "support frequency {count} vs expected {expected:.1} (ratio {ratio:.2})"
            );
        }
    }
}
