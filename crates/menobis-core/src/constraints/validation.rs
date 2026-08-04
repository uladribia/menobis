//! Common validation shared by every ensemble backend.
//!
//! These checks run once before dispatch. Backends keep only
//! algorithm-specific checks (graphical realization, kernel availability,
//! optimizer convergence).

use crate::OccNum;

/// Balanced-strength feasibility: total out must equal total in.
pub fn validate_balanced_strengths(
    strength_out: &[u64],
    strength_in: &[u64],
) -> Result<(), String> {
    if strength_out.len() != strength_in.len() {
        return Err("strength_out and strength_in must have the same length".into());
    }
    let out: u64 = strength_out.iter().sum();
    let in_sum: u64 = strength_in.iter().sum();
    if out != in_sum {
        return Err(format!(
            "unbalanced strengths: total out {out} != total in {in_sum}"
        ));
    }
    Ok(())
}

/// Balanced-degree feasibility: total out-degree must equal total in-degree.
pub fn validate_balanced_degrees(degree_out: &[u64], degree_in: &[u64]) -> Result<(), String> {
    if degree_out.len() != degree_in.len() {
        return Err("degree_out and degree_in must have the same length".into());
    }
    let out: u64 = degree_out.iter().sum();
    let in_sum: u64 = degree_in.iter().sum();
    if out != in_sum {
        return Err(format!(
            "unbalanced degrees: total out {out} != total in {in_sum}"
        ));
    }
    Ok(())
}

/// Elementary strength-degree feasibility: every node strength >= degree.
pub fn validate_strength_degree_bounds(
    strength_out: &[u64],
    strength_in: &[u64],
    degree_out: &[u64],
    degree_in: &[u64],
) -> Result<(), String> {
    for (i, (&s, &k)) in strength_out.iter().zip(degree_out.iter()).enumerate() {
        if s < k {
            return Err(format!("out strength {s} < out degree {k} at node {i}"));
        }
    }
    for (i, (&s, &k)) in strength_in.iter().zip(degree_in.iter()).enumerate() {
        if s < k {
            return Err(format!("in strength {s} < in degree {k} at node {i}"));
        }
    }
    Ok(())
}

/// Event/edge consistency: total events >= total edges.
pub fn validate_events_at_least_edges(total_events: u64, total_edges: u64) -> Result<(), String> {
    if total_events < total_edges {
        return Err(format!(
            "total events {total_events} < total edges {total_edges}"
        ));
    }
    Ok(())
}

/// B family capacity: fixed occupation numbers must not exceed layer count M.
pub fn validate_b_occupation_capacity(layers: u32, occ_nums: &[OccNum]) -> Result<(), String> {
    for &o in occ_nums {
        if o > OccNum::from(layers) {
            return Err(format!(
                "fixed occupation {o} exceeds binomial layer capacity {layers}"
            ));
        }
    }
    Ok(())
}

/// Layer count sanity: M >= 1.
pub fn validate_layers(layers: u32) -> Result<(), String> {
    if layers < 1 {
        return Err("layers must be at least 1".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_strengths_pass_and_fail() {
        assert!(validate_balanced_strengths(&[10, 20], &[15, 15]).is_ok());
        assert!(validate_balanced_strengths(&[10, 20], &[15, 14]).is_err());
        assert!(validate_balanced_strengths(&[1], &[1, 2]).is_err());
    }

    #[test]
    fn strength_degree_bounds() {
        assert!(validate_strength_degree_bounds(&[5, 5], &[5, 5], &[2, 3], &[3, 2]).is_ok());
        assert!(validate_strength_degree_bounds(&[5, 5], &[5, 5], &[6, 1], &[3, 2]).is_err());
    }

    #[test]
    fn events_and_edges() {
        assert!(validate_events_at_least_edges(10, 5).is_ok());
        assert!(validate_events_at_least_edges(5, 10).is_err());
    }

    #[test]
    fn b_capacity_and_layers() {
        assert!(validate_b_occupation_capacity(3, &[0, 3]).is_ok());
        assert!(validate_b_occupation_capacity(3, &[0, 4]).is_err());
        assert!(validate_layers(1).is_ok());
        assert!(validate_layers(0).is_err());
    }
}
