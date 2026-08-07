//! Sparse positive-occupation state for the fixed-total Gibbs chain.
//!
//! Holds exactly `E` positive occupations summing to `T`.  Memory is
//! `O(E)`.

use crate::OccNum;

/// Occupation vector for a fixed-total problem: `t_e ≥ 1`, `Σ t_e = T`.
#[derive(Clone, Debug)]
pub struct FixedTotalState {
    occupations: Vec<OccNum>,
    total: OccNum,
}

impl FixedTotalState {
    /// Create a state from a positive occupation vector.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if any occupation is zero.
    pub fn new(occupations: Vec<OccNum>) -> Self {
        debug_assert!(
            occupations.iter().all(|&t| t > 0),
            "fixed-total state requires strictly positive occupations"
        );
        let total = occupations.iter().sum();
        Self { occupations, total }
    }

    /// Number of occupied cells `E`.
    #[inline]
    pub fn len(&self) -> usize {
        self.occupations.len()
    }

    /// Whether the state is empty (`E = 0`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.occupations.is_empty()
    }

    /// Total occupation `T`.
    #[inline]
    pub fn total(&self) -> OccNum {
        self.total
    }

    /// Occupation of cell `idx`.
    #[inline]
    pub fn get(&self, idx: usize) -> OccNum {
        self.occupations[idx]
    }

    /// The occupation vector.
    #[inline]
    pub fn occupations(&self) -> &[OccNum] {
        &self.occupations
    }

    /// Write a new split into cells `i` and `j` (Gibbs step).
    #[inline]
    pub(crate) fn set_pair(&mut self, i: usize, j: usize, a: OccNum, b: OccNum) {
        self.occupations[i] = a;
        self.occupations[j] = b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_computed() {
        let s = FixedTotalState::new(vec![1, 2, 3]);
        assert_eq!(s.total(), 6);
        assert_eq!(s.len(), 3);
        assert_eq!(s.get(1), 2);
    }

    #[test]
    fn set_pair_keeps_total() {
        let mut s = FixedTotalState::new(vec![1, 3, 2]);
        s.set_pair(0, 1, 2, 2); // (1,3) -> (2,2): sum still 4
        assert_eq!(s.total(), 6);
        assert_eq!(s.occupations(), &[2, 2, 2]);
    }
}
