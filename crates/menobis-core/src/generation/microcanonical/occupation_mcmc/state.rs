//! Sparse integer occupation state for fixed-strength MCMC.
//!
//! The state holds the current occupation matrix as a sparse map from
//! ordered pair `(src, tgt)` to occupation count, avoiding \(O(N^2)\)
//! memory.  Strength marginals are cached for validation.

use std::collections::HashMap;

use rand::Rng;

use crate::generation::output::SampledNetwork;
use crate::OccNum;

/// Sparse integer occupation state for a fixed-strength chain.
///
/// Maintains:
/// - `occupations`: `HashMap` from `(src, tgt)` to occupation count.
/// - `out_strengths`, `in_strengths`: cached marginals for validation.
/// - `occupied_pairs`: indexed vector of positive pairs for fast
///   iteration and uniform sampling.
///
/// Memory: \(O(N + E)\) where \(E\) is the number of occupied pairs.
#[derive(Clone, Debug)]
pub struct StrengthState {
    pub node_count: usize,
    /// Current out-strength per node (redundant with occupations; used
    /// for validation and diagnostics).
    pub out_strengths: Vec<OccNum>,
    /// Current in-strength per node.
    pub in_strengths: Vec<OccNum>,

    // Sparse occupation map: only positive occupations are stored.
    occupations: HashMap<(u64, u64), OccNum>,

    /// Index of positive pairs for fast iteration and uniform sampling.
    occupied_pairs: Vec<(u64, u64)>,
    /// Maps each positive pair to its index in `occupied_pairs`.
    occupied_positions: HashMap<(u64, u64), usize>,

    /// Number of occupied pairs per source node.
    /// Length = `node_count`.
    pub(super) row_occ_count: Vec<usize>,
    /// Number of occupied pairs per target node.
    /// Length = `node_count`.
    pub(super) col_occ_count: Vec<usize>,
}

impl StrengthState {
    /// Create a new state from a list of occupied pairs and their
    /// occupations.
    ///
    /// The marginals are recomputed from the input.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if:
    /// - any pair has occupation 0 (use `None` instead).
    /// - any node index is out of range.
    /// - out and in totals are unbalanced.
    pub fn new(node_count: usize, pairs: Vec<((u64, u64), OccNum)>) -> Self {
        let m = pairs.len();
        let mut occupations = HashMap::with_capacity(m);
        let mut occupied_pairs = Vec::with_capacity(m);
        let mut occupied_positions = HashMap::with_capacity(m);
        let mut out_strengths = vec![0u64; node_count];
        let mut in_strengths = vec![0u64; node_count];

        for (idx, ((src, tgt), occ)) in pairs.into_iter().enumerate() {
            debug_assert!(
                occ > 0,
                "occupation must be positive; remove zero entries before calling new()"
            );
            debug_assert!(
                (src as usize) < node_count && (tgt as usize) < node_count,
                "node index out of range"
            );
            out_strengths[src as usize] += occ;
            in_strengths[tgt as usize] += occ;
            occupations.insert((src, tgt), occ);
            occupied_pairs.push((src, tgt));
            occupied_positions.insert((src, tgt), idx);
        }

        debug_assert_eq!(
            out_strengths.iter().sum::<OccNum>(),
            in_strengths.iter().sum::<OccNum>(),
            "unbalanced strengths"
        );

        // Compute row/col occupied-pair counts from the initial pairs.
        let row_occ_count = {
            let mut counts = vec![0usize; node_count];
            for &(src, _) in &occupied_pairs {
                counts[src as usize] += 1;
            }
            counts
        };
        let col_occ_count = {
            let mut counts = vec![0usize; node_count];
            for &(_, tgt) in &occupied_pairs {
                counts[tgt as usize] += 1;
            }
            counts
        };

        Self {
            node_count,
            out_strengths,
            in_strengths,
            occupations,
            occupied_pairs,
            occupied_positions,
            row_occ_count,
            col_occ_count,
        }
    }

    /// Look up the occupation of a pair.  Returns `0` if not occupied.
    #[inline]
    pub fn get(&self, src: u64, tgt: u64) -> OccNum {
        self.occupations.get(&(src, tgt)).copied().unwrap_or(0)
    }

    /// Number of occupied pairs (positive occupations).
    #[inline]
    pub fn occupied_count(&self) -> usize {
        self.occupied_pairs.len()
    }

    /// Iterator over all occupied pairs and their occupations.
    pub fn iter_occupied(&self) -> impl Iterator<Item = ((u64, u64), OccNum)> + '_ {
        self.occupied_pairs
            .iter()
            .map(|&p| (p, self.occupations[&p]))
    }

    /// Set the occupation of a pair and update marginals.
    ///
    /// If `new_occ` is 0, the pair is removed from the sparse map.
    /// If `new_occ > 0` and the pair was absent, it is inserted.
    ///
    /// This is `pub(super)` so the cycle move can apply the 4-cell
    /// deltas directly without heap allocation.
    pub(super) fn set(&mut self, src: u64, tgt: u64, new_occ: OccNum) {
        let old_occ = self.get(src, tgt);
        if old_occ == new_occ {
            return;
        }

        // Update marginals.
        let delta = (new_occ as i64) - (old_occ as i64);
        self.out_strengths[src as usize] =
            (self.out_strengths[src as usize] as i64 + delta) as OccNum;
        self.in_strengths[tgt as usize] =
            (self.in_strengths[tgt as usize] as i64 + delta) as OccNum;

        match (old_occ, new_occ) {
            (0, new) if new > 0 => {
                // Insert.
                let idx = self.occupied_pairs.len();
                self.occupied_pairs.push((src, tgt));
                self.occupied_positions.insert((src, tgt), idx);
                self.occupations.insert((src, tgt), new);
                self.row_occ_count[src as usize] += 1;
                self.col_occ_count[tgt as usize] += 1;
            }
            (old, 0) if old > 0 => {
                // Remove (swap-remove).
                let idx = self.occupied_positions.remove(&(src, tgt)).unwrap();
                let last = self.occupied_pairs.pop().unwrap();
                if idx < self.occupied_pairs.len() {
                    self.occupied_pairs[idx] = last;
                    self.occupied_positions.insert(last, idx);
                }
                self.occupations.remove(&(src, tgt));
                self.row_occ_count[src as usize] -= 1;
                self.col_occ_count[tgt as usize] -= 1;
            }
            (_, new) => {
                // Update in place — occupied set unchanged.
                self.occupations.insert((src, tgt), new);
            }
        }
    }

    /// Convert to a [`SampledNetwork`] for output.
    ///
    /// The pairs are sorted by `(src, tgt)` for deterministic output.
    pub fn to_sampled_network(&self) -> SampledNetwork {
        let m = self.occupied_pairs.len();
        let mut sources = Vec::with_capacity(m);
        let mut targets = Vec::with_capacity(m);
        let mut occ_nums = Vec::with_capacity(m);

        let mut pairs = self.occupied_pairs.to_vec();
        pairs.sort_unstable();
        for (src, tgt) in pairs {
            sources.push(src);
            targets.push(tgt);
            occ_nums.push(self.occupations[&(src, tgt)]);
        }

        SampledNetwork {
            sources,
            targets,
            occ_nums,
        }
    }

    /// Debug validation: check internal consistency.
    #[cfg(debug_assertions)]
    pub fn debug_validate(&self) {
        // Check map and vec agreement.
        assert_eq!(self.occupations.len(), self.occupied_pairs.len());
        assert_eq!(self.occupations.len(), self.occupied_positions.len());

        for (idx, &(src, tgt)) in self.occupied_pairs.iter().enumerate() {
            assert_eq!(self.occupied_positions[&(src, tgt)], idx);
            assert!(self.occupations.contains_key(&(src, tgt)));
            assert!(self.occupations[&(src, tgt)] > 0);
        }

        // Check marginals.
        let mut check_out = vec![0u64; self.node_count];
        let mut check_in = vec![0u64; self.node_count];
        for (&(src, tgt), &occ) in &self.occupations {
            check_out[src as usize] += occ;
            check_in[tgt as usize] += occ;
        }
        assert_eq!(check_out, self.out_strengths);
        assert_eq!(check_in, self.in_strengths);

        // Check row/col occupied-pair counts.
        let mut check_row = vec![0usize; self.node_count];
        let mut check_col = vec![0usize; self.node_count];
        for &(src, tgt) in &self.occupied_pairs {
            check_row[src as usize] += 1;
            check_col[tgt as usize] += 1;
        }
        assert_eq!(check_row, self.row_occ_count, "row_occ_count mismatch");
        assert_eq!(check_col, self.col_occ_count, "col_occ_count mismatch");

        // Balanced totals.
        assert_eq!(
            check_out.iter().sum::<OccNum>(),
            check_in.iter().sum::<OccNum>()
        );
    }

    /// Debug validation skipped in release builds.
    #[cfg(not(debug_assertions))]
    pub fn debug_validate(&self) {}

    /// Return a reference to the list of occupied pairs.
    pub fn occupied_pairs(&self) -> &[(u64, u64)] {
        &self.occupied_pairs
    }

    /// Select an occupied pair uniformly at random.
    ///
    /// Returns `None` if there are no occupied pairs.
    pub fn choose_random_occupied(&self, rng: &mut impl Rng) -> Option<((u64, u64), OccNum)> {
        let m = self.occupied_pairs.len();
        if m == 0 {
            return None;
        }
        let idx = rng.random_range(0..m);
        let pair = self.occupied_pairs[idx];
        Some((pair, self.occupations[&pair]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn empty_state() {
        let state = StrengthState::new(3, vec![]);
        assert_eq!(state.occupied_count(), 0);
        assert_eq!(state.get(0, 1), 0);
        assert_eq!(state.out_strengths, vec![0, 0, 0]);
    }

    #[test]
    fn single_pair() {
        let state = StrengthState::new(3, vec![((0, 1), 5)]);
        assert_eq!(state.get(0, 1), 5);
        assert_eq!(state.get(0, 0), 0);
        assert_eq!(state.out_strengths[0], 5);
        assert_eq!(state.in_strengths[1], 5);
        assert_eq!(state.occupied_count(), 1);
    }

    #[test]
    fn set_increment() {
        let mut state = StrengthState::new(3, vec![((0, 1), 3)]);
        state.set(0, 1, 5);
        assert_eq!(state.get(0, 1), 5);
        assert_eq!(state.out_strengths[0], 5);
        assert_eq!(state.in_strengths[1], 5);
    }

    #[test]
    fn set_decrement() {
        let mut state = StrengthState::new(3, vec![((0, 1), 5)]);
        state.set(0, 1, 2);
        assert_eq!(state.get(0, 1), 2);
        assert_eq!(state.out_strengths[0], 2);
    }

    #[test]
    fn set_remove() {
        let mut state = StrengthState::new(3, vec![((0, 1), 3)]);
        state.set(0, 1, 0);
        assert_eq!(state.get(0, 1), 0);
        assert_eq!(state.occupied_count(), 0);
    }

    #[test]
    fn set_insert() {
        let mut state = StrengthState::new(3, vec![]);
        state.set(1, 2, 4);
        assert_eq!(state.get(1, 2), 4);
        assert_eq!(state.occupied_count(), 1);
    }

    #[test]
    fn to_sampled_network() {
        let state = StrengthState::new(3, vec![((0, 1), 3), ((2, 0), 7), ((1, 2), 2)]);
        let net = state.to_sampled_network();
        // Pairs should be sorted: (0,1), (1,2), (2,0)
        assert_eq!(net.sources, vec![0, 1, 2]);
        assert_eq!(net.targets, vec![1, 2, 0]);
        assert_eq!(net.occ_nums, vec![3, 2, 7]);
        let total: OccNum = net.occ_nums.iter().sum();
        assert_eq!(total, 12);
    }

    #[test]
    fn row_col_occ_counts_initial() {
        let state = StrengthState::new(3, vec![((0, 1), 3), ((2, 0), 7), ((1, 2), 2)]);
        assert_eq!(state.row_occ_count, vec![1, 1, 1]);
        assert_eq!(state.col_occ_count, vec![1, 1, 1]);
    }

    #[test]
    fn row_col_occ_counts_on_insert_remove_update() {
        let mut state = StrengthState::new(3, vec![((0, 1), 3)]);
        assert_eq!(state.row_occ_count, vec![1, 0, 0]);
        assert_eq!(state.col_occ_count, vec![0, 1, 0]);

        // Insert new pair
        state.set(2, 0, 5);
        assert_eq!(state.row_occ_count, vec![1, 0, 1]);
        assert_eq!(state.col_occ_count, vec![1, 1, 0]);

        // In-place update (no set change)
        state.set(2, 0, 7);
        assert_eq!(state.row_occ_count, vec![1, 0, 1]);
        assert_eq!(state.col_occ_count, vec![1, 1, 0]);

        // Remove pair
        state.set(2, 0, 0);
        assert_eq!(state.row_occ_count, vec![1, 0, 0]);
        assert_eq!(state.col_occ_count, vec![0, 1, 0]);

        // Remove last pair
        state.set(0, 1, 0);
        assert_eq!(state.row_occ_count, vec![0, 0, 0]);
        assert_eq!(state.col_occ_count, vec![0, 0, 0]);
    }

    #[test]
    fn choose_random_occupied_empty() {
        let state = StrengthState::new(3, vec![]);
        let mut rng = StdRng::seed_from_u64(42);
        assert!(state.choose_random_occupied(&mut rng).is_none());
    }

    #[test]
    fn choose_random_occupied_single() {
        let state = StrengthState::new(3, vec![((0, 1), 5)]);
        let mut rng = StdRng::seed_from_u64(42);
        let (pair, occ) = state.choose_random_occupied(&mut rng).unwrap();
        assert_eq!(pair, (0, 1));
        assert_eq!(occ, 5);
    }

    #[test]
    fn choose_random_occupied_uniform() {
        use std::collections::HashMap;
        let state = StrengthState::new(
            2,
            vec![((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)],
        );
        let mut counts = HashMap::new();
        let mut rng = StdRng::seed_from_u64(12345);
        for _ in 0..4000 {
            let (pair, _) = state.choose_random_occupied(&mut rng).unwrap();
            *counts.entry(pair).or_insert(0) += 1;
        }
        // Each of the 4 pairs should appear roughly 1000 times (within 3 std).
        for &count in counts.values() {
            assert!(
                count > 800 && count < 1200,
                "Expected ~1000, got {count}"
            );
        }
    }

    #[test]
    fn marginals_preserved_after_updates() {
        // All four cells of the 4-cycle must be initially occupied.
        let mut state =
            StrengthState::new(3, vec![((0, 0), 2), ((0, 2), 1), ((1, 0), 3), ((1, 2), 5)]);
        let out_before = state.out_strengths.clone();
        let in_before = state.in_strengths.clone();

        // Apply a 4-cycle: (0,0)+1, (1,2)+1, (0,2)-1, (1,0)-1
        state.set(0, 0, 3);
        state.set(1, 2, 6);
        state.set(0, 2, 0);
        state.set(1, 0, 2);

        // Marginals should be unchanged (each node gets +1 and -1).
        assert_eq!(state.out_strengths, out_before);
        assert_eq!(state.in_strengths, in_before);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn debug_validate_after_updates() {
        let mut state = StrengthState::new(2, vec![((0, 1), 4), ((1, 0), 2)]);
        state.set(0, 1, 3);
        state.set(1, 0, 3);
        state.debug_validate();
    }
}
