//! Greedy directed support initializer with randomized fallback.
//!
//! Constructs one feasible simple directed support graph for a given
//! out-degree / in-degree sequence pair.
//!
//! Algorithm:
//!
//! 1. Sort nodes by residual out-degree descending.
//! 2. For the node with highest residual out-degree, select the `d_out`
//!    distinct target nodes with the largest residual in-degrees, skipping
//!    self-loops and already-occupied pairs, then add directed edges.
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
    admissible_pairs: Option<&[(u64, u64)]>,
) -> Result<DegreeSupportState, FixedKTError> {
    let is_admissible = |s: u64, t: u64| {
        admissible_pairs
            .map(|ap| ap.contains(&(s, t)))
            .unwrap_or(true)
    };
    // Try deterministic first, then fall back to randomized attempts.
    if let Ok(state) = try_construct(
        out_degrees,
        in_degrees,
        self_loops,
        None::<&mut rand::rngs::StdRng>,
        is_admissible,
    ) {
        return Ok(state);
    }

    // Randomized retries with different seeds.
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for attempt in 0..MAX_RETRIES {
        let seed = rng.random::<u64>();
        let mut attempt_rng = rand::rngs::StdRng::seed_from_u64(seed);
        if let Ok(state) = try_construct(
            out_degrees,
            in_degrees,
            self_loops,
            Some(&mut attempt_rng),
            is_admissible,
        ) {
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

/// Build one feasible directed support graph from a caller-supplied
/// admissibility predicate and **caller RNG** (Gate B, §16).
///
/// Unlike [`greedy_directed_initialize`], there is no deterministic
/// first pass: tie-breaking is randomized from the very first attempt
/// and consumes the caller's `rng`, so repeated calls with the same
/// seed reproduce the same support and different seeds produce
/// different supports.  The (fixed-(s,k)) caller passes
/// `|src, tgt| residual.domain.is_admissible(src, tgt)`.
pub fn greedy_directed_initialize_with_admissibility<F>(
    out_degrees: &[u32],
    in_degrees: &[u32],
    self_loops: bool,
    rng: &mut impl Rng,
    is_admissible: F,
) -> Result<DegreeSupportState, FixedKTError>
where
    F: Fn(u64, u64) -> bool,
{
    try_construct(
        out_degrees,
        in_degrees,
        self_loops,
        Some(rng),
        is_admissible,
    )
}

/// Internal construction attempt.
///
/// If `rng` is `None`, uses deterministic tie-breaking (by node index).
/// If `rng` is `Some(rng)`, shuffles tied candidates randomly.
fn try_construct<F>(
    out_degrees: &[u32],
    in_degrees: &[u32],
    self_loops: bool,
    mut rng: Option<&mut impl Rng>,
    is_admissible: F,
) -> Result<DegreeSupportState, FixedKTError>
where
    F: Fn(u64, u64) -> bool,
{
    let n = out_degrees.len();
    let mut out_rem: Vec<u32> = out_degrees.to_vec();
    let mut in_rem: Vec<u32> = in_degrees.to_vec();
    let mut edge_set: HashSet<(u64, u64)> = HashSet::new();
    let mut edges: Vec<(u64, u64)> =
        Vec::with_capacity(out_degrees.iter().map(|&d| d as usize).sum());

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
                    && is_admissible(u as u64, v as u64)
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
        candidates.sort_by_key(|c| std::cmp::Reverse(c.0));

        if let Some(rng) = rng.as_mut() {
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
        let state = greedy_directed_initialize(&out, &inp, false, None).unwrap();
        assert_eq!(state.edge_count(), 4);
        assert_eq!(state.out_degree_sequence(), out);
        for (i, &d) in inp.iter().enumerate() {
            assert_eq!(state.in_degree(i), d as usize);
        }
    }

    #[test]
    fn out_star() {
        let n = 5;
        let mut out = vec![0u32; n];
        out[0] = (n - 1) as u32;
        let mut inp = vec![0u32; n];
        for item in inp.iter_mut().skip(1) {
            *item = 1;
        }
        let state = greedy_directed_initialize(&out, &inp, false, None).unwrap();
        assert_eq!(state.edge_count(), n - 1);
        assert_eq!(state.out_degree_sequence(), out);
        for (i, &d) in inp.iter().enumerate() {
            assert_eq!(state.in_degree(i), d as usize);
        }
    }

    #[test]
    fn complete_directed() {
        let n = 4;
        let out = vec![(n - 1) as u32; n];
        let inp = vec![(n - 1) as u32; n];
        let state = greedy_directed_initialize(&out, &inp, false, None).unwrap();
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
        let state = greedy_directed_initialize(&out, &inp, false, None).unwrap();
        assert_eq!(state.edge_count(), 0);
    }

    #[test]
    fn tricky_degrees() {
        // This sequence exercises the randomized fallback when deterministic
        // tie-breaking picks sub-optimal targets.
        let out = vec![2u32, 1, 1];
        let inp = vec![1u32, 2, 1];
        let state = greedy_directed_initialize(&out, &inp, false, None).unwrap();
        assert_eq!(state.out_degree_sequence(), out);
        for (i, &d) in inp.iter().enumerate() {
            assert_eq!(state.in_degree(i), d as usize);
        }
    }

    #[test]
    fn deterministic_reproducibility() {
        let out = vec![2u32, 1, 1];
        let inp = vec![1u32, 2, 1];
        let a = greedy_directed_initialize(&out, &inp, false, None).unwrap();
        let b = greedy_directed_initialize(&out, &inp, false, None).unwrap();
        assert_eq!(a.edges, b.edges);
    }
}

#[cfg(test)]
mod with_admissibility_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Randomized-from-first-attempt with a caller RNG: same seed must
    /// reproduce the same support; a different seed may differ (§16, §34).
    #[test]
    fn same_seed_reproduces_randomized_support() {
        // out=(2,2,0), in=(1,1,2), self-loops allowed: row 0 must pick
        // among two equal in-degree targets (tie), so seeds can explore
        // different supports.
        let out = vec![2u32, 2, 0];
        let inp = vec![1u32, 1, 2];
        let run = |seed: u64| -> Vec<(u64, u64)> {
            let mut rng = StdRng::seed_from_u64(seed);
            let s = greedy_directed_initialize_with_admissibility(
                &out,
                &inp,
                true,
                &mut rng,
                |_, _| true,
            )
            .unwrap();
            let mut edges = s.edges.clone();
            edges.sort_unstable();
            edges
        };
        assert_eq!(run(7), run(7), "same seed must reproduce the support");
        // Tie-breaking is randomized from the first attempt: across seeds
        // more than one support must be reachable.
        let distinct = [
            run(1),
            run(2),
            run(3),
            run(4),
            run(5),
            run(6),
            run(7),
            run(8),
        ]
        .iter()
        .collect::<HashSet<_>>()
        .len();
        assert!(distinct > 1, "expected more than one support across seeds");
    }

    /// An admissibility predicate excluding one coordinate must keep it
    /// out of the constructed support.
    #[test]
    fn admissibility_predicate_excludes_pairs() {
        let out = vec![1u32, 1, 1];
        let inp = vec![1u32, 1, 1];
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..20 {
            let state = greedy_directed_initialize_with_admissibility(
                &out,
                &inp,
                true,
                &mut rng,
                |s, t| (s, t) != (0, 1),
            )
            .unwrap();
            assert!(
                !state.contains(&(0, 1)),
                "excluded pair (0,1) must never be constructed"
            );
            assert_eq!(state.out_degree_sequence(), out);
            assert_eq!(
                state.in_degree(0),
                1,
                "node 0 must still receive in-degree 1 from another row"
            );
        }
    }
}
