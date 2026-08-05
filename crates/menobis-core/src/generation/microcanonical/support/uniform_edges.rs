//! Shared support-sampling routines for fixed-(E,T) microcanonical families.

use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::Rng;

/// Sample `k` distinct indices from `[0, n)` uniformly without replacement.
///
/// Uses Floyd's algorithm with a `HashSet` for O(1) membership checks
/// (expected O(k) time, O(k) memory).  The collected set is sorted before
/// the final shuffle so the result is reproducible for a fixed seed (hash
/// iteration order is non-deterministic).
pub fn sample_uniform_support(n: usize, k: usize, rng: &mut StdRng) -> Vec<usize> {
    debug_assert!(k <= n, "Floyd: k={k} > n={n}");
    let mut selected: HashSet<usize> = HashSet::with_capacity(k);
    let start = n.saturating_sub(k);
    for j in start..n {
        let t = rng.random_range(0..=j);
        if selected.contains(&t) {
            selected.insert(j);
        } else {
            selected.insert(t);
        }
    }
    debug_assert_eq!(selected.len(), k);
    let mut result: Vec<usize> = selected.into_iter().collect();
    result.sort_unstable();
    shuffle_slice(&mut result, rng);
    result
}

/// Fisher-Yates shuffle.
pub fn shuffle_slice<T>(slice: &mut [T], rng: &mut StdRng) {
    let len = slice.len();
    for i in (1..len).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn correct_size_no_duplicates() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            let s = sample_uniform_support(10_000, 500, &mut rng);
            assert_eq!(s.len(), 500);
            let mut sorted = s.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 500);
            assert!(sorted.last().unwrap() < &10_000);
        }
    }

    #[test]
    fn reproducible() {
        let mut r1 = StdRng::seed_from_u64(123);
        let mut r2 = StdRng::seed_from_u64(123);
        assert_eq!(
            sample_uniform_support(10_000, 800, &mut r1),
            sample_uniform_support(10_000, 800, &mut r2),
        );
    }

    #[test]
    fn large_k() {
        // k close to n — exercises the HashSet path with large k.
        let mut rng = StdRng::seed_from_u64(7);
        let s = sample_uniform_support(20_000, 19_000, &mut rng);
        assert_eq!(s.len(), 19_000);
        let mut sorted = s.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 19_000);
    }
}
