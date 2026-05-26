//! Sparse pair mask for fitting routines.
//!
//! Replaces the dense `Vec<bool>` of size N² with a sparse representation
//! that stores only the excluded pairs (self-loops + known/frozen pairs).
//! Memory usage: O(N + K) where K = number of known pairs, instead of O(N²).

use std::collections::HashSet;

/// Sparse representation of excluded (masked) node pairs.
///
/// A pair `(i, j)` is masked (excluded from fitting sums) if:
/// - `!self_loops && i == j` (diagonal), OR
/// - The pair was explicitly added via [`PairMask::add`] or construction.
///
/// # Performance
///
/// - `is_masked(i, j)`: O(1) via HashSet lookup
/// - `free_col_sum(j, values)`: O(K_j) where K_j = masked rows in column j
/// - `free_row_sum(i, values)`: O(K_i) where K_i = masked cols in row i
/// - `n_free()`: O(1)
/// - Memory: O(N + K) where K = total masked pairs beyond the diagonal
#[derive(Clone, Debug)]
pub struct PairMask {
    n: usize,
    self_loops: bool,
    /// Explicit masked pairs (excludes diagonal when !self_loops, those are implicit).
    pairs: HashSet<(usize, usize)>,
    /// Per-column: list of masked row indices (including diagonal).
    masked_rows: Vec<Vec<usize>>,
    /// Per-row: list of masked column indices (including diagonal).
    masked_cols: Vec<Vec<usize>>,
    /// Total number of free (non-masked) pairs.
    n_free: usize,
}

impl PairMask {
    /// Create a mask from self-loop policy and explicit known pairs.
    ///
    /// Known pairs are specified as parallel arrays of source/target indices.
    #[must_use]
    pub fn new(n: usize, self_loops: bool, known_src: &[u64], known_tgt: &[u64]) -> Self {
        let mut pairs = HashSet::with_capacity(known_src.len());
        let mut masked_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut masked_cols: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Self-loop diagonal
        if !self_loops {
            for i in 0..n {
                masked_rows[i].push(i);
                masked_cols[i].push(i);
            }
        }

        // Known pairs
        for (&s, &t) in known_src.iter().zip(known_tgt.iter()) {
            let si = s as usize;
            let ti = t as usize;
            if si < n && ti < n {
                // Skip if already masked (self-loop case)
                if !self_loops && si == ti {
                    continue;
                }
                if pairs.insert((si, ti)) {
                    masked_rows[ti].push(si);
                    masked_cols[si].push(ti);
                }
            }
        }

        let n_masked = if self_loops {
            pairs.len()
        } else {
            n + pairs.len()
        };
        let n_free = (n * n).saturating_sub(n_masked);

        Self {
            n,
            self_loops,
            pairs,
            masked_rows,
            masked_cols,
            n_free,
        }
    }

    /// Create a mask from self-loop policy only (no known pairs).
    #[must_use]
    pub fn from_self_loops(n: usize, self_loops: bool) -> Self {
        Self::new(n, self_loops, &[], &[])
    }

    /// Create a `PairMask` from a dense `Vec<bool>` (for transition from legacy code).
    ///
    /// Pairs where `dense[i * n + j]` is `true` are masked.
    #[must_use]
    pub fn from_dense(n: usize, dense: &[bool]) -> Self {
        let mut pairs = HashSet::new();
        let mut masked_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut masked_cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut n_masked = 0usize;

        for i in 0..n {
            for j in 0..n {
                if dense[i * n + j] {
                    n_masked += 1;
                    if i != j {
                        pairs.insert((i, j));
                    }
                    masked_rows[j].push(i);
                    masked_cols[i].push(j);
                }
            }
        }

        // Detect self_loops policy from diagonal
        let self_loops = n == 0 || !dense[0];

        let n_free = (n * n).saturating_sub(n_masked);

        Self {
            n,
            self_loops,
            pairs,
            masked_rows,
            masked_cols,
            n_free,
        }
    }

    /// Check if pair (i, j) is masked (excluded).
    #[inline]
    #[must_use]
    pub fn is_masked(&self, i: usize, j: usize) -> bool {
        if !self.self_loops && i == j {
            return true;
        }
        self.pairs.contains(&(i, j))
    }

    /// Number of nodes.
    #[inline]
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Whether self-loops are allowed.
    #[inline]
    #[must_use]
    pub fn self_loops(&self) -> bool {
        self.self_loops
    }

    /// Total number of free (non-masked) pairs.
    #[inline]
    #[must_use]
    pub fn n_free(&self) -> usize {
        self.n_free
    }

    /// Compute sum of `values[i]` over free rows in column `j`.
    ///
    /// Equivalent to: `(0..n).filter(|&i| !mask[i*n+j]).map(|i| values[i]).sum()`
    /// but runs in O(K_j) instead of O(N) when full_sum is precomputed.
    #[inline]
    #[must_use]
    pub fn free_col_sum(&self, j: usize, values: &[f64], full_sum: f64) -> f64 {
        let masked_sum: f64 = self.masked_rows[j].iter().map(|&i| values[i]).sum();
        full_sum - masked_sum
    }

    /// Compute sum of `values[j]` over free columns in row `i`.
    ///
    /// Equivalent to: `(0..n).filter(|&j| !mask[i*n+j]).map(|j| values[j]).sum()`
    /// but runs in O(K_i) instead of O(N) when full_sum is precomputed.
    #[inline]
    #[must_use]
    pub fn free_row_sum(&self, i: usize, values: &[f64], full_sum: f64) -> f64 {
        let masked_sum: f64 = self.masked_cols[i].iter().map(|&j| values[j]).sum();
        full_sum - masked_sum
    }

    /// Get masked row indices for a given column.
    #[inline]
    #[must_use]
    pub fn masked_rows_in_col(&self, j: usize) -> &[usize] {
        &self.masked_rows[j]
    }

    /// Get masked column indices for a given row.
    #[inline]
    #[must_use]
    pub fn masked_cols_in_row(&self, i: usize) -> &[usize] {
        &self.masked_cols[i]
    }

    /// Convert to a dense `Vec<bool>` for backward compatibility during migration.
    ///
    /// This is O(N²) and should only be used as a temporary bridge.
    #[must_use]
    pub fn to_dense(&self) -> Vec<bool> {
        let mut mask = vec![false; self.n * self.n];
        if !self.self_loops {
            for i in 0..self.n {
                mask[i * self.n + i] = true;
            }
        }
        for &(i, j) in &self.pairs {
            mask[i * self.n + j] = true;
        }
        mask
    }

    /// Add a masked pair. Returns true if the pair was newly added.
    pub fn add(&mut self, i: usize, j: usize) -> bool {
        if i >= self.n || j >= self.n {
            return false;
        }
        if !self.self_loops && i == j {
            return false; // already masked
        }
        if self.pairs.insert((i, j)) {
            self.masked_rows[j].push(i);
            self.masked_cols[i].push(j);
            self.n_free -= 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mask_with_self_loops() {
        let m = PairMask::from_self_loops(4, true);
        assert_eq!(m.n_free(), 16);
        assert!(!m.is_masked(0, 0));
        assert!(!m.is_masked(1, 2));
    }

    #[test]
    fn mask_without_self_loops() {
        let m = PairMask::from_self_loops(4, false);
        assert_eq!(m.n_free(), 12); // 16 - 4
        assert!(m.is_masked(0, 0));
        assert!(m.is_masked(3, 3));
        assert!(!m.is_masked(0, 1));
    }

    #[test]
    fn mask_with_known_pairs() {
        let src = vec![0u64, 1, 2];
        let tgt = vec![1u64, 2, 0];
        let m = PairMask::new(4, false, &src, &tgt);
        assert_eq!(m.n_free(), 12 - 3); // 9
        assert!(m.is_masked(0, 1));
        assert!(m.is_masked(1, 2));
        assert!(m.is_masked(2, 0));
        assert!(!m.is_masked(0, 2));
    }

    #[test]
    fn free_col_sum_matches_dense() {
        let src = vec![0u64, 2];
        let tgt = vec![1u64, 1]; // mask (0,1) and (2,1)
        let m = PairMask::new(4, false, &src, &tgt);
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let full_sum: f64 = values.iter().sum(); // 10.0

        // Column 1: masked rows = {1 (diagonal), 0, 2} → free = {3}
        let sparse_sum = m.free_col_sum(1, &values, full_sum);
        // Dense: filter rows where !mask[i*4+1] → i=3 → values[3]=4.0
        assert!((sparse_sum - 4.0).abs() < 1e-12);
    }

    #[test]
    fn free_row_sum_matches_dense() {
        let src = vec![1u64, 1];
        let tgt = vec![0u64, 2]; // mask (1,0) and (1,2)
        let m = PairMask::new(4, false, &src, &tgt);
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let full_sum: f64 = values.iter().sum(); // 10.0

        // Row 1: masked cols = {1 (diagonal), 0, 2} → free = {3}
        let sparse_sum = m.free_row_sum(1, &values, full_sum);
        assert!((sparse_sum - 4.0).abs() < 1e-12);
    }

    #[test]
    fn to_dense_matches_manual() {
        let src = vec![0u64, 2];
        let tgt = vec![1u64, 3];
        let m = PairMask::new(3, false, &src, &tgt);
        let dense = m.to_dense();
        // Diagonal masked
        assert!(dense[0]); // (0,0)
        assert!(dense[4]); // (1,1)
        assert!(dense[8]); // (2,2)
                           // Known pairs
        assert!(dense[1]); // (0,1)
                           // (2,3) is out of bounds for n=3, so not added
        assert!(!dense[6]); // (2,0) free
    }

    #[test]
    fn equivalence_with_dense_strength_sum() {
        // Verify that sparse mask gives same IPF denominators as dense
        let n = 5;
        let src = vec![0u64, 1, 3];
        let tgt = vec![2u64, 4, 0];
        let mask = PairMask::new(n, false, &src, &tgt);
        let dense = mask.to_dense();

        let x = vec![1.5, 2.3, 0.8, 3.1, 1.2];
        let y = vec![0.9, 1.7, 2.5, 0.4, 1.1];
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();

        for j in 0..n {
            let dense_sum: f64 = (0..n).filter(|&i| !dense[i * n + j]).map(|i| x[i]).sum();
            let sparse_sum = mask.free_col_sum(j, &x, sum_x);
            assert!(
                (dense_sum - sparse_sum).abs() < 1e-12,
                "col {j}: dense={dense_sum}, sparse={sparse_sum}"
            );
        }

        for i in 0..n {
            let dense_sum: f64 = (0..n).filter(|&j| !dense[i * n + j]).map(|j| y[j]).sum();
            let sparse_sum = mask.free_row_sum(i, &y, sum_y);
            assert!(
                (dense_sum - sparse_sum).abs() < 1e-12,
                "row {i}: dense={dense_sum}, sparse={sparse_sum}"
            );
        }
    }

    #[test]
    fn memory_savings_documented() {
        // Document memory characteristics at various N
        // Dense: N*N bytes; Sparse: ~(N + K) * (8+8) bytes for HashSet + Vecs
        for &n in &[100, 1000, 5000, 10000] {
            let dense_bytes = n * n; // Vec<bool> = 1 byte per entry
            let k = 50; // typical known pairs
                        // PairMask overhead: HashSet entries + 2*N Vecs with ~(N/N + K/N) entries each
                        // Approximate: pairs HashSet ~K*48 + masked_rows/cols Vecs ~N*(24+diagonal)
            let sparse_approx = k * 64 + n * 48; // generous upper bound
            let ratio = dense_bytes as f64 / sparse_approx as f64;
            // At N=10000: 100MB vs ~500KB = 200x savings
            assert!(
                ratio > 1.0,
                "Sparse should always use less memory than dense at N={n}"
            );

            // Actually construct and verify
            let src: Vec<u64> = (0..k as u64).collect();
            let tgt: Vec<u64> = (1..=k as u64).collect();
            let mask = PairMask::new(n, false, &src, &tgt);
            assert_eq!(mask.n_free(), n * n - n - k); // N² - diagonal - known
        }
    }

    #[test]
    fn from_dense_roundtrip() {
        let n = 6;
        let src = vec![0u64, 1, 3, 4];
        let tgt = vec![2u64, 4, 0, 1];
        let original = PairMask::new(n, false, &src, &tgt);
        let dense = original.to_dense();
        let reconstructed = PairMask::from_dense(n, &dense);

        assert_eq!(original.n_free(), reconstructed.n_free());
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    original.is_masked(i, j),
                    reconstructed.is_masked(i, j),
                    "mismatch at ({i},{j})"
                );
            }
        }
    }
}
