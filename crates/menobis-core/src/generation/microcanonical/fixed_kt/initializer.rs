//! Greedy directed support initializer with randomized fallback.
//!
//! Constructs one feasible simple directed support graph for a given
//! out-degree / in-degree sequence pair.
//!
//! Algorithm:
//! 1. Sort nodes by residual out-degree descending.
//! 2. For the node with highest residual out-degree:
//!    a. Select the `d_out` distinct target nodes with the largest residual
//!       in-degrees, skipping self-loops and already-occupied pairs.
//!    b. Add directed edges.
//! 3. If construction fails, retry with randomized target selection
//!    (shuffle among tied in-degrees). This resolves cases where the
//!    deterministic tie-breaking makes sub-optimal choices.

use rand::Rng;
use rand::SeedableRng;
use std::collections::HashSet;

use super::errors::FixedKTError;
use super::state::DegreeSupportState;

/// The initialization method used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitializationMethod {
    GreedyDirected,
}

/// Maximum retries for the randomized fallback.
const MAX_RETRIES: usize = 20;

/// Build one feasible directed support graph using the greedy constructor.
///
/// Returns a `DegreeSupportState` or an error if construction fails.
pub fn greedy_directed_initialize(
    out_degrees: &[u32],
    in_degrees: &[u32],
    self_loops: bool,
) -> Result<DegreeSupportState, FixedKTError> {
    // Try deterministic first, then fall back to randomized attempts.
    if let Ok(state) = try_construct(out_degrees, in_degrees, self_loops, None) {
        return Ok(state);
    }

    // Randomized retries with different seeds
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for attempt in 0..MAX_RETRIES {
        let seed = rng.random::<u64>();
        if let Ok(state) = try_construct(out_degrees, in_degrees, self_loops, Some(seed)) {
            return Ok(state);
        }
        if attempt == MAX_RETRIES - 1 {
            return Err(FixedKTError::InitializationFailed(format!(
                "could not construct support after {} attempts",
                MAX_RETRIES
            )));
        }
    }

    Err(FixedKTError::InitializationFailed(
        "construction failed".into(),
    ))
}

/// Internal construction attempt.
///
/// If `rng_seed` is `None`, uses deterministic tie-breaking (by node index).
/// If `rng_seed` is `Some(seed)`, shuffles tied candidates randomly.
fn try_construct(
    out_degrees: &[u32],
    in_degrees: &[u32],
    self_loops: bool,
    rng_seed: Option<u64>,
) -> Result<DegreeSupportState, FixedKTError> {
    let n = out_degrees.len();
    let mut out_rem: Vec<u32> = out_degrees.to_vec();
    let mut in_rem: Vec<u32> = in_degrees.to_vec();
    let mut edge_set: HashSet<(u64, u64)> = HashSet::new();
    let mut edges: Vec<(u64, u64)> =
        Vec::with_capacity(out_degrees.iter().map(|&d| d as usize).sum());
    let mut rng = rng_seed.map(|s| rand::rngs::StdRng::seed_from_u64(s));

    // Active nodes with residual out-degree > 0
    let mut active: Vec<usize> = (0..n).filter(|&i| out_rem[i] > 0).collect();

    while !active.is_empty() {
        // Sort by residual out-degree descending
        active.sort_by(|&a, &b| out_rem[b].cmp(&out_rem[a]).then_with(|| a.cmp(&b)));
        let u = active[0];
        let d_out = out_rem[u];
        if d_out == 0 {
            active.remove(0);
            continue;
        }

        // Collect candidate targets
        let mut candidates: Vec<(u32, usize)> = (0..n)
            .filter(|&v| {
                in_rem[v] > 0
                    && (self_loops || u != v)
                    && !edge_set.contains(&(u as u64, v as u64))
            })
            .map(|v| (in_rem[v], v))
            .collect();

        if candidates.len() < d_out as usize {
            return Err(FixedKTError::InitializationFailed(format!(
                "node {u} needs {d_out} distinct targets, found {}",
                candidates.len()
            )));
        }

        // Sort candidates by in-degree descending
        candidates.sort_by(|a, b| b.0.cmp(&a.0));

        if let Some(ref mut rng) = rng {
            // Randomize among equal in-degrees (Fisher-Yates shuffle within ties)
            let mut i = 0;
            while i < candidates.len() {
                let mut j = i;
                while j < candidates.len() && candidates[j].0 == candidates[i].0 {
                    j += 1;
                }
                if j - i > 1 {
                    let range = &mut candidates[i..j];
                    for k in (1..range.len()).rev() {
                        let idx = rng.random_range(0..=k);
                        range.swap(k, idx);
                    }
                }
                i = j;
            }
        }

        // Take d_out candidates
        for &(_, v) in candidates.iter().take(d_out as usize) {
            let pair = (u as u64, v as u64);
            edge_set.insert(pair);
            edges.push(pair);
            in_rem[v] = in_rem[v].saturating_sub(1);
        }
        out_rem[u] = 0;

        active.retain(|&i| out_rem[i] > 0);
    }

    // Check in-degree satisfaction
    if let Some((j, &d)) = in_rem.iter().enumerate().find(|(_, &d)| d > 0) {
        return Err(FixedKTError::InitializationFailed(format!(
            "node {j} has residual in-degree {d} after construction"
        )));
    }

    let state = DegreeSupportState::new(n, edges, self_loops);
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directed_cycle() {
        let out = vec![1u32, 1, 1, 1];
        let inp = vec![1u32, 1, 1, 1];
        let state = greedy_directed_initialize(&out, &inp, false).unwrap();
        assert_eq!(state.edge_count(), 4);
        assert_eq!(state.out_degree_sequence(), out);
        for i in 0..4 {
            assert_eq!(state.in_degree(i), inp[i] as usize);
        }
    }

    #[test]
    fn out_star() {
        let n = 5;
        let mut out = vec![0u32; n];
        out[0] = (n - 1) as u32;
        let mut inp = vec![0u32; n];
        for i in 1..n {
            inp[i] = 1;
        }
        let state = greedy_directed_initialize(&out, &inp, false).unwrap();
        assert_eq!(state.edge_count(), (n - 1) as usize);
        assert_eq!(state.out_degree_sequence(), out);
        for i in 0..n {
            assert_eq!(state.in_degree(i), inp[i] as usize);
        }
    }

    #[test]
    fn complete_directed() {
        let n = 4;
        let out = vec![(n - 1) as u32; n];
        let inp = vec![(n - 1) as u32; n];
        let state = greedy_directed_initialize(&out, &inp, false).unwrap();
        assert_eq!(state.edge_count(), n * (n - 1));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(state.contains(&(i as u64, j as u64)));
                }
            }
        }
    }

    #[test]
    fn empty_degrees() {
        let out = vec![0u32; 3];
        let inp = vec![0u32; 3];
        let state = greedy_directed_initialize(&out, &inp, false).unwrap();
        assert_eq!(state.edge_count(), 0);
    }

    #[test]
    fn tricky_degrees() {
        // This sequence exercises the randomized fallback when deterministic
        // tie-breaking picks sub-optimal targets.
        let out = vec![2u32, 1, 1];
        let inp = vec![1u32, 2, 1];
        let state = greedy_directed_initialize(&out, &inp, false).unwrap();
        assert_eq!(state.out_degree_sequence(), out);
        for i in 0..3 {
            assert_eq!(state.in_degree(i), inp[i] as usize);
        }
    }

    #[test]
    fn deterministic_reproducibility() {
        let out = vec![2u32, 1, 1];
        let inp = vec![1u32, 2, 1];
        let a = greedy_directed_initialize(&out, &inp, false).unwrap();
        let b = greedy_directed_initialize(&out, &inp, false).unwrap();
        assert_eq!(a.edges, b.edges);
    }
}
