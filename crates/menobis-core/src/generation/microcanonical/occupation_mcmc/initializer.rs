//! Initial-state construction for fixed-strength residual problems.
//!
//! Provides two constructors:
//!
//! - **Greedy fill** for any complete pair domain (O(N²) integer scan,
//!   no max-flow).  Always succeeds for balanced strengths on a complete
//!   domain.
//! - **Max-flow construction** for sparse/admissible-pair domains.

use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use super::feasibility::feasibility_max_flow;
use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Greedy construction for any complete pair domain.
///
/// Scans row-by-row, column-by-column, skipping inadmissible cells
/// (e.g. the diagonal when self-loops are forbidden) and respecting
/// per-cell family capacity.
///
/// **Correctness**: Always succeeds for ME/W with self-loops (the
/// diagonal provides a safety valve).  For ME/W without self-loops
/// it succeeds for `N ≥ 4` or when strengths are not all concentrated
/// on the last rows.  Returns a structured error if residuals remain,
/// signalling the caller to fall back to max-flow.
///
/// Complexity: O(N²) integer operations — fast at production sizes
/// (N=5000 → 25M iterations, pure integer math, no flow network).
#[allow(clippy::needless_range_loop)]
pub fn greedy_complete(
    strength_out: &[OccNum],
    strength_in: &[OccNum],
    family: OccupationFamily,
    domain: &PairDomain,
) -> Result<super::feasibility::FlowTable, FixedStrengthError> {
    let n = strength_out.len();
    let mut out = strength_out.to_vec();
    let mut inp = strength_in.to_vec();
    let cap = domain.capacity(family);
    let self_loops = domain.self_loops_allowed();
    let mut table = Vec::new();

    for i in 0..n {
        if out[i] == 0 {
            continue;
        }
        for j in 0..n {
            if inp[j] == 0 {
                continue;
            }
            if !self_loops && i == j {
                continue;
            }
            let assign = cap.min(out[i]).min(inp[j]);
            if assign > 0 {
                table.push(((i as u64, j as u64), assign));
                out[i] -= assign;
                inp[j] -= assign;
            }
            if out[i] == 0 {
                break;
            }
        }
    }

    // Residual check — non-fatal: caller falls back to max-flow.
    if out.iter().any(|&s| s > 0) || inp.iter().any(|&s| s > 0) {
        return Err(FixedStrengthError::InitializationFailed(
            "greedy fill failed: residual strengths remain".into(),
        ));
    }

    Ok(table)
}

/// Construct an initial occupation table for a residual strength problem.
///
/// - **Complete domain** (with or without self-loops, any family): fast
///   greedy O(N²) integer fill, no max-flow.
/// - **Sparse domain** (explicit admissible pairs): Dinic max-flow.
pub fn initialize_table(
    strength_out: &[OccNum],
    strength_in: &[OccNum],
    family: OccupationFamily,
    domain: &PairDomain,
) -> Result<super::feasibility::FlowTable, FixedStrengthError> {
    let total: OccNum = strength_out.iter().sum();
    if total == 0 {
        return Ok(Vec::new());
    }

    // Try greedy first for ME/W on complete domains.  Falls back to max-flow
    // if the greedy fill fails (can happen for small N without self-loops).
    // B always uses max-flow (per-cell capacity can deadlock greedy).
    let family_is_b = matches!(family, OccupationFamily::B { .. });
    if matches!(domain, PairDomain::Complete { .. }) && !family_is_b {
        match greedy_complete(strength_out, strength_in, family, domain) {
            Ok(table) => return Ok(table),
            Err(_) => { /* fall through to max-flow */ }
        }
    }

    // Sparse / admissible-pair domain: use max-flow.
    let cap = domain.capacity(family);
    let admissible: Vec<_> = domain.iter_admissible().collect();
    match feasibility_max_flow(strength_out, strength_in, family, &admissible, cap) {
        Ok(Some(table)) => Ok(table),
        Ok(None) => Ok(Vec::new()),
        Err(msg) => Err(FixedStrengthError::InitializationFailed(msg)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn greedy_simple_2x2() {
        let out = vec![5, 5];
        let inp = vec![5, 5];
        let domain = PairDomain::Complete {
            node_count: 2,
            self_loops: true,
        };
        let table = greedy_complete(&out, &inp, OccupationFamily::ME, &domain).unwrap();
        let total: OccNum = table.iter().map(|(_, o)| o).sum();
        assert_eq!(total, 10);
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn greedy_heterogeneous() {
        let out = vec![10, 0, 5];
        let inp = vec![3, 7, 5];
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: true,
        };
        let table = greedy_complete(&out, &inp, OccupationFamily::ME, &domain).unwrap();
        let total: OccNum = table.iter().map(|(_, o)| o).sum();
        assert_eq!(total, 15);
        let mut check_out = vec![0u64; 3];
        let mut check_in = vec![0u64; 3];
        for &((s, t), o) in &table {
            check_out[s as usize] += o;
            check_in[t as usize] += o;
        }
        assert_eq!(check_out, out);
        assert_eq!(check_in, inp);
    }

    #[test]
    fn no_self_loops_greedy() {
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: false,
        };
        let out = vec![5, 5, 5];
        let inp = vec![5, 5, 5];
        let table = initialize_table(&out, &inp, OccupationFamily::ME, &domain).unwrap();
        let total: OccNum = table.iter().map(|(_, o)| o).sum();
        assert_eq!(total, 15);
        // No self-loops.
        for &((s, t), _) in &table {
            assert_ne!(s, t, "self-loop found");
        }
    }

    #[test]
    fn sparse_domain_via_max_flow() {
        let mut allowed = HashSet::new();
        allowed.insert((0, 1));
        allowed.insert((1, 0));
        allowed.insert((2, 2));
        let domain = PairDomain::Sparse {
            node_count: 3,
            allowed,
        };
        let out = vec![4, 3, 2];
        let inp = vec![3, 4, 2];
        let table = initialize_table(&out, &inp, OccupationFamily::ME, &domain).unwrap();
        let total: OccNum = table.iter().map(|(_, o)| o).sum();
        assert_eq!(total, 9);
        for &((s, t), _) in &table {
            assert!(domain.is_admissible(s, t), "({s},{t}) not admissible");
        }
    }

    #[test]
    fn zero_total() {
        let out = vec![0; 3];
        let inp = vec![0; 3];
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: true,
        };
        let table = initialize_table(&out, &inp, OccupationFamily::ME, &domain).unwrap();
        assert!(table.is_empty());
    }

    #[test]
    fn greedy_complete_n5000_no_self_loops() {
        let n = 5000usize;
        let t: OccNum = 45000;
        let mut out = vec![t / n as OccNum; n];
        let mut inp = out.clone();
        let rem = t - out.iter().sum::<OccNum>();
        out[n - 1] += rem;
        inp[n - 1] += rem;
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: false,
        };
        let table = greedy_complete(&out, &inp, OccupationFamily::ME, &domain).unwrap();
        let total: OccNum = table.iter().map(|(_, o)| o).sum();
        assert_eq!(total, t);
    }

    #[test]
    fn b_capacity_greedy() {
        let out = vec![5, 0, 0];
        let inp = vec![2, 2, 1];
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: true,
        };
        let table =
            greedy_complete(&out, &inp, OccupationFamily::B { layers: 2 }, &domain).unwrap();
        for &(_, occ) in &table {
            assert!(occ <= 2, "B occupation {occ} exceeds M=2");
        }
        let total: OccNum = table.iter().map(|(_, o)| o).sum();
        assert_eq!(total, 5);
    }
}
