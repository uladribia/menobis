//! Initial-state construction for fixed-strength residual problems.
//!
//! Provides two constructors:
//!
//! - **Greedy fill** for complete domains with self-loops allowed (all N²
//!   ordered pairs admissible).  Avoids building flow edges.
//! - **Max-flow construction** for restricted domains (no self-loops,
//!   masks, sparse admissible sets).

use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use super::feasibility::feasibility_max_flow;
use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Greedy construction for a complete pair domain with self-loops allowed.
///
/// All N² ordered pairs are admissible, which guarantees a feasible
/// assignment exists for any balanced strength pair.  The fill proceeds
/// row-by-row, column-by-column.
#[allow(clippy::needless_range_loop)]
pub fn greedy_complete(
    strength_out: &[OccNum],
    strength_in: &[OccNum],
    family: OccupationFamily,
) -> Result<super::feasibility::FlowTable, FixedStrengthError> {
    let n = strength_out.len();
    let mut out = strength_out.to_vec();
    let mut inp = strength_in.to_vec();
    let cap = PairDomain::Complete {
        node_count: n,
        self_loops: true,
    }
    .capacity(family);
    let mut table = Vec::new();

    for i in 0..n {
        if out[i] == 0 {
            continue;
        }
        for j in 0..n {
            if inp[j] == 0 {
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

    debug_assert!(
        out.iter().all(|&s| s == 0),
        "greedy constructor failed: residual out-strengths remain"
    );
    debug_assert!(
        inp.iter().all(|&s| s == 0),
        "greedy constructor failed: residual in-strengths remain"
    );

    Ok(table)
}

/// Construct an initial occupation table for a residual strength problem.
///
/// Uses the fast greedy fill for complete domains with self-loops and
/// max flow for all restricted domains (no self-loops, masks, etc.).
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

    // Fast path: greedy only for ME with self-loops (no per-cell capacity limit).
    let use_greedy = matches!(
        domain,
        PairDomain::Complete {
            self_loops: true,
            ..
        },
    ) && family == OccupationFamily::ME;

    if use_greedy {
        return greedy_complete(strength_out, strength_in, family);
    }

    // Generic path: max flow for all constrained cases.
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
        let table = greedy_complete(&out, &inp, OccupationFamily::ME).unwrap();
        let total: OccNum = table.iter().map(|(_, o)| o).sum();
        assert_eq!(total, 10);
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn greedy_heterogeneous() {
        let out = vec![10, 0, 5];
        let inp = vec![3, 7, 5];
        let table = greedy_complete(&out, &inp, OccupationFamily::ME).unwrap();
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
    fn no_self_loops_via_max_flow() {
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
    fn b_capacity_greedy() {
        let out = vec![5, 0, 0];
        let inp = vec![2, 2, 1];
        let table = greedy_complete(&out, &inp, OccupationFamily::B { layers: 2 }).unwrap();
        for &(_, occ) in &table {
            assert!(occ <= 2, "B occupation {occ} exceeds M=2");
        }
        let total: OccNum = table.iter().map(|(_, o)| o).sum();
        assert_eq!(total, 5);
    }
}
