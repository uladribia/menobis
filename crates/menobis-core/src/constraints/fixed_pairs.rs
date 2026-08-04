//! Fixed-pair accounting and residualization.
//!
//! Fixed (known) pairs are node pairs whose occupation number is frozen
//! before ensemble dispatch. Their contributions to every observable are
//! computed once, and residual constraints are what the free pairs must
//! satisfy. A fixed pair with occupation zero is represented explicitly,
//! not as a separate "prohibited pair" concept (thesis §2.2).

use crate::OccNum;
use std::collections::HashSet;

use super::mask::PairMask;

/// Fixed pairs with explicit occupation numbers.
///
/// `sources`/`targets`/`occ_nums` are parallel arrays. `occ_nums[i]` may be
/// zero: a fixed-zero pair is a valid occupation state, not a structural
/// absence. `mask` is the induced sparse pair mask.
#[derive(Clone, Debug)]
pub struct FixedPairs {
    pub sources: Vec<u64>,
    pub targets: Vec<u64>,
    pub occ_nums: Vec<OccNum>,
    pub mask: PairMask,
}

impl FixedPairs {
    /// Build `FixedPairs` from parallel arrays, computing the sparse mask.
    ///
    /// Duplicate coordinates are rejected; out-of-range indices are rejected
    /// when `node_count` is supplied.
    pub fn new(
        node_count: usize,
        self_loops: bool,
        sources: &[u64],
        targets: &[u64],
        occ_nums: &[OccNum],
    ) -> Result<Self, String> {
        if sources.len() != targets.len() || sources.len() != occ_nums.len() {
            return Err(
                "known_source, known_target, and known_occnum must have the same length".into(),
            );
        }
        let mut seen = HashSet::with_capacity(sources.len());
        for ((&s, &t), &o) in sources.iter().zip(targets.iter()).zip(occ_nums.iter()) {
            if s as usize >= node_count || t as usize >= node_count {
                return Err("known pair index out of range".into());
            }
            if !self_loops && s == t {
                return Err("known self-loop conflicts with self_loops=False".into());
            }
            if !seen.insert((s, t)) {
                return Err("duplicate known pair coordinates".into());
            }
            let _ = o;
        }
        let mask = PairMask::new(node_count, self_loops, sources, targets);
        Ok(Self {
            sources: sources.to_vec(),
            targets: targets.to_vec(),
            occ_nums: occ_nums.to_vec(),
            mask,
        })
    }

    /// Empty fixed pairs for the unmasked case.
    #[must_use]
    pub fn empty(node_count: usize, self_loops: bool) -> Self {
        let mask = PairMask::from_self_loops(node_count, self_loops);
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            occ_nums: Vec::new(),
            mask,
        }
    }

    /// The sparse pair mask induced by the self-loop policy and fixed pairs.
    #[must_use]
    pub fn mask(&self) -> &PairMask {
        &self.mask
    }
}

/// Contributions of fixed pairs to every observable.
///
/// Complexity `O(N + K)` where `K` is the number of fixed pairs.
#[derive(Clone, Debug, Default)]
pub struct FixedContributions {
    pub total_events: u64,
    pub total_edges: u64,
    pub strength_out: Vec<u64>,
    pub strength_in: Vec<u64>,
    pub degree_out: Vec<u64>,
    pub degree_in: Vec<u64>,
    pub total_cost: Option<f64>,
}

/// Compute fixed-pair contributions in one pass over the fixed pairs.
pub fn compute_fixed_contributions(
    node_count: usize,
    fixed: &FixedPairs,
    cost_provider: Option<&dyn crate::pairs::PairCostProvider>,
) -> FixedContributions {
    let mut out = FixedContributions {
        strength_out: vec![0_u64; node_count],
        strength_in: vec![0_u64; node_count],
        degree_out: vec![0_u64; node_count],
        degree_in: vec![0_u64; node_count],
        total_cost: if cost_provider.is_some() {
            Some(0.0)
        } else {
            None
        },
        ..Default::default()
    };
    for ((&s, &t), &occ) in fixed
        .sources
        .iter()
        .zip(fixed.targets.iter())
        .zip(fixed.occ_nums.iter())
    {
        let si = s as usize;
        let ti = t as usize;
        out.total_events += occ;
        if occ > 0 {
            out.total_edges += 1;
            out.strength_out[si] += occ;
            out.strength_in[ti] += occ;
            out.degree_out[si] += 1;
            out.degree_in[ti] += 1;
        }
        if let Some(total_cost) = out.total_cost.as_mut() {
            if let Some(provider) = cost_provider {
                if let Some(c) = provider.cost(si, ti) {
                    *total_cost += c * occ as f64;
                }
            }
        }
    }
    out
}

/// Residual constraints after subtracting fixed-pair contributions.
///
/// `None` fields mean the observable is not constrained. Borrowed slices are
/// not used here because residual vectors are freshly computed.
#[derive(Clone, Debug, Default)]
pub struct ResidualConstraints {
    pub total_events: Option<u64>,
    pub total_edges: Option<u64>,
    pub strength_out: Option<Vec<u64>>,
    pub strength_in: Option<Vec<u64>>,
    pub degree_out: Option<Vec<u64>>,
    pub degree_in: Option<Vec<u64>>,
    pub expected_cost: Option<f64>,
}

/// Subtract fixed contributions from full constraints.
///
/// Strength/degree residuals are clamped at zero after feasibility checks
/// (a negative residual means the fixed pairs over-satisfy the constraint,
/// which the caller must treat as infeasible).
#[allow(clippy::too_many_arguments)]
pub fn residualize(
    node_count: usize,
    fixed: &FixedContributions,
    full_total_events: Option<u64>,
    full_total_edges: Option<u64>,
    full_strength_out: Option<&[u64]>,
    full_strength_in: Option<&[u64]>,
    full_degree_out: Option<&[u64]>,
    full_degree_in: Option<&[u64]>,
    full_cost: Option<f64>,
) -> Result<ResidualConstraints, String> {
    let sub = |full: Option<&[u64]>, contrib: &[u64]| -> Result<Option<Vec<u64>>, String> {
        match full {
            None => Ok(None),
            Some(seq) => {
                if seq.len() != node_count {
                    return Err("constraint sequence length mismatch".into());
                }
                let mut res = Vec::with_capacity(node_count);
                for (f, c) in seq.iter().zip(contrib.iter()) {
                    if f < c {
                        return Err("fixed pairs over-satisfy the constraint".into());
                    }
                    res.push(f - c);
                }
                Ok(Some(res))
            }
        }
    };
    let strength_out = sub(full_strength_out, &fixed.strength_out)?;
    let strength_in = sub(full_strength_in, &fixed.strength_in)?;
    let degree_out = sub(full_degree_out, &fixed.degree_out)?;
    let degree_in = sub(full_degree_in, &fixed.degree_in)?;

    let total_events = match (full_total_events, fixed.total_events) {
        (Some(t), c) => {
            if t < c {
                return Err("fixed pairs exceed total_events".into());
            }
            Some(t - c)
        }
        (None, _) => None,
    };
    let total_edges = match (full_total_edges, fixed.total_edges) {
        (Some(e), c) => {
            if e < c {
                return Err("fixed pairs exceed total_edges".into());
            }
            Some(e - c)
        }
        (None, _) => None,
    };
    let expected_cost = match (full_cost, fixed.total_cost) {
        (Some(c), Some(fc)) => Some((c - fc).max(0.0)),
        (Some(c), None) => Some(c),
        (None, _) => None,
    };

    Ok(ResidualConstraints {
        total_events,
        total_edges,
        strength_out,
        strength_in,
        degree_out,
        degree_in,
        expected_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_contributions_accumulate_all_observables() {
        let fixed = FixedPairs::new(4, true, &[0, 1, 2], &[1, 2, 3], &[3, 0, 5]).expect("valid");
        let contrib = compute_fixed_contributions(4, &fixed, None);
        assert_eq!(contrib.total_events, 8); // 3 + 0 + 5
        assert_eq!(contrib.total_edges, 2); // only positive occupations
        assert_eq!(contrib.strength_out, vec![3, 0, 5, 0]);
        assert_eq!(contrib.strength_in, vec![0, 3, 0, 5]);
        assert_eq!(contrib.degree_out, vec![1, 0, 1, 0]);
        assert_eq!(contrib.degree_in, vec![0, 1, 0, 1]);
    }

    #[test]
    fn fixed_zero_pair_is_an_ordinary_fixed_pair() {
        let fixed = FixedPairs::new(3, true, &[1], &[2], &[0]).expect("valid");
        assert!(fixed.mask().is_masked(1, 2));
        let contrib = compute_fixed_contributions(3, &fixed, None);
        assert_eq!(contrib.total_edges, 0);
        assert_eq!(contrib.total_events, 0);
    }

    #[test]
    fn residualize_subtracts_fixed_contributions() {
        let fixed = FixedPairs::new(3, true, &[0, 1], &[1, 2], &[2, 4]).expect("valid");
        let contrib = compute_fixed_contributions(3, &fixed, None);
        let res = residualize(
            3,
            &contrib,
            Some(100),
            Some(10),
            Some(&[10, 20, 30]),
            Some(&[10, 20, 30]),
            Some(&[3, 4, 5]),
            Some(&[3, 4, 5]),
            None,
        )
        .expect("feasible");
        // strength residual: [10-2, 20-4, 30] = [8, 16, 30]
        assert_eq!(res.strength_out.as_deref(), Some(&[8, 16, 30][..]));
        // degree residual: fixed degrees 1+1 -> [2, 3, 5]
        assert_eq!(res.degree_out.as_deref(), Some(&[2, 3, 5][..]));
        assert_eq!(res.total_events, Some(100 - 6));
        assert_eq!(res.total_edges, Some(10 - 2));
    }

    #[test]
    fn residualize_rejects_over_satisfaction() {
        let fixed = FixedPairs::new(3, true, &[0], &[1], &[50]).expect("valid");
        let contrib = compute_fixed_contributions(3, &fixed, None);
        assert!(residualize(
            3,
            &contrib,
            None,
            None,
            Some(&[10, 20, 30]),
            Some(&[10, 20, 30]),
            None,
            None,
            None,
        )
        .is_err());
    }

    #[test]
    fn fixed_pairs_reject_duplicates_and_out_of_range() {
        assert!(FixedPairs::new(3, true, &[0, 0], &[1, 1], &[1, 2]).is_err());
        assert!(FixedPairs::new(3, true, &[0, 5], &[1, 1], &[1, 2]).is_err());
        assert!(FixedPairs::new(3, false, &[1], &[1], &[2]).is_err()); // self-loop conflict
        assert!(FixedPairs::new(3, true, &[0], &[1], &[1]).is_ok());
    }
}
