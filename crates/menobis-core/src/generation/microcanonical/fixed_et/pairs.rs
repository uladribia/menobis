//! On-the-fly index-to-pair mapping for fixed-(E,T) sampling.
//!
//! Avoids materialising the full N² pair list; O(1) arithmetic lookup.

/// Total number of admissible pairs given node count and self-loop policy.
pub fn total_admissible_pairs(n: usize, self_loops: bool) -> usize {
    if self_loops {
        n.saturating_mul(n)
    } else {
        n.saturating_mul(n.saturating_sub(1))
    }
}

/// Map a linear index `idx` in `[0, L)` to a source/target node pair `(i, j)`.
///
/// With self-loops: i = idx / n, j = idx % n.
/// Without self-loops: skip the diagonal.  i = idx / (n-1),
/// j = idx % (n-1) with j ≥ i shifted by +1.
pub fn linear_to_pair(idx: usize, n: usize, self_loops: bool) -> (usize, usize) {
    if self_loops {
        (idx / n, idx % n)
    } else {
        let i = idx / (n - 1);
        let j_raw = idx % (n - 1);
        let j = if j_raw >= i { j_raw + 1 } else { j_raw };
        (i, j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_self_loops() {
        let n = 4;
        assert_eq!(linear_to_pair(0, n, true), (0, 0));
        assert_eq!(linear_to_pair(3, n, true), (0, 3));
        assert_eq!(linear_to_pair(4, n, true), (1, 0));
        assert_eq!(linear_to_pair(15, n, true), (3, 3));
    }

    #[test]
    fn without_self_loops() {
        let n = 4;
        assert_eq!(linear_to_pair(0, n, false), (0, 1));
        assert_eq!(linear_to_pair(2, n, false), (0, 3));
        assert_eq!(linear_to_pair(3, n, false), (1, 0));
        assert_eq!(linear_to_pair(4, n, false), (1, 2));
        assert_eq!(linear_to_pair(11, n, false), (3, 2));
    }

    #[test]
    fn roundtrip() {
        for &sl in &[true, false] {
            for n in [2, 3, 5] {
                let l = if sl { n * n } else { n * (n - 1) };
                let mut found = std::collections::HashSet::new();
                for idx in 0..l {
                    let (i, j) = linear_to_pair(idx, n, sl);
                    assert!(i < n && j < n);
                    if !sl {
                        assert_ne!(i, j);
                    }
                    assert!(found.insert((i, j)));
                }
                assert_eq!(found.len(), l);
            }
        }
    }

    #[test]
    fn total_pairs() {
        assert_eq!(total_admissible_pairs(4, true), 16);
        assert_eq!(total_admissible_pairs(4, false), 12);
        assert_eq!(total_admissible_pairs(0, true), 0);
    }
}
