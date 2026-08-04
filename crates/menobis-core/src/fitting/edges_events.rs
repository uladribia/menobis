//! Grand-canonical EDGES_EVENTS fitting.
//!
//! The `EDGES_EVENTS` constraint fixes total binary edges `E` and total
//! events `T` in expectation over `n_pairs` candidate node pairs. With only
//! two global constraints, the max-ent solution is symmetric: every pair
//! shares one zero-inflated distribution with a global occupation multiplier
//! `lam` and positive-support parameter `q`.
//!
//! Conditional mean (thesis §2.2 zero-inflated equations):
//!
//! ```text
//! E[t_ij | t_ij > 0] = q G'_F(q) / G_F(q)
//! ```
//!
//! with `G_ME(q)=exp(q)-1`, `G_B(q)=(1+q)^M-1`, `G_W(q)=(1-q)^(-M)-1`.
//! Setting the conditional mean to `T/E` and the occupation probability to
//! `E/n_pairs` decouples the two constraints:
//!
//! ```text
//! q G'_F(q) / G_F(q) = T / E
//! occupation = E / n_pairs
//! lam = occupation / ((1 - occupation) G_F(q))
//! ```

use crate::distribution::OccupationFamily;

/// Fitted scalar multipliers for the EDGES_EVENTS model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgesEventsFitResult {
    /// Positive-support parameter `q`.
    pub q: f64,
    /// Global occupation multiplier `lam`.
    pub lam: f64,
    /// Binary occupation probability `E[Theta(t_ij > 0)]`.
    pub occupation: f64,
    /// Conditional mean `E[t_ij | t_ij > 0] = T/E`.
    pub positive_mean: f64,
    /// Always true: the two scalar equations solve by bisection.
    pub converged: bool,
    /// Bisection iterations for the conditional-mean solve.
    pub iterations: usize,
}

/// Errors for EDGES_EVENTS feasibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgesEventsFitError {
    /// `E == 0`: no occupied pairs.
    ZeroEdges,
    /// `T < E`: every occupied pair needs at least one event.
    EventsBelowEdges,
    /// `E > n_pairs`: more occupied pairs than candidate pairs.
    EdgesAbovePairs,
    /// `T/E > M` for Binomial(M) or `T/E` outside the family support.
    MeanOutOfSupport,
}

impl std::fmt::Display for EdgesEventsFitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroEdges => write!(f, "total edges must be positive"),
            Self::EventsBelowEdges => {
                write!(f, "total events must be at least total edges")
            }
            Self::EdgesAbovePairs => {
                write!(f, "total edges cannot exceed the number of candidate pairs")
            }
            Self::MeanOutOfSupport => write!(
                f,
                "T/E conditional mean is outside the family occupation support"
            ),
        }
    }
}

impl std::error::Error for EdgesEventsFitError {}

/// Positive-support partition factor `G_F(q)`.
fn partition_factor(family: OccupationFamily, q: f64) -> f64 {
    match family {
        OccupationFamily::Poisson => q.exp_m1(),
        OccupationFamily::Binomial(m) => (1.0 + q).powi(m as i32) - 1.0,
        OccupationFamily::Geometric => {
            let qq = q.clamp(0.0, 1.0 - 1e-15);
            qq / (1.0 - qq)
        }
        OccupationFamily::NegativeBinomial(m) => {
            let qq = q.clamp(0.0, 1.0 - 1e-15);
            (1.0 - qq).powi(-(m as i32)) - 1.0
        }
    }
}

/// Conditional mean `q G'_F(q) / G_F(q)` for the given family.
fn conditional_mean(family: OccupationFamily, q: f64, _layers: u32) -> f64 {
    match family {
        OccupationFamily::Poisson => {
            if q <= 0.0 {
                1.0
            } else {
                q / (1.0 - (-q).exp())
            }
        }
        OccupationFamily::Binomial(m) => {
            if q <= 0.0 {
                1.0
            } else {
                let base = 1.0 + q;
                let mf = f64::from(m);
                mf * q * base.powi(m as i32 - 1) / (base.powi(m as i32) - 1.0)
            }
        }
        OccupationFamily::Geometric => {
            let qq = q.clamp(0.0, 1.0 - 1e-15);
            1.0 / (1.0 - qq)
        }
        OccupationFamily::NegativeBinomial(m) => {
            if q <= 0.0 {
                1.0
            } else {
                let qq = q.clamp(0.0, 1.0 - 1e-15);
                let mf = f64::from(m);
                mf * qq * (1.0 - qq).powi(-(m as i32) - 1) / ((1.0 - qq).powi(-(m as i32)) - 1.0)
            }
        }
    }
}

/// Solve `conditional_mean(q) = target` by bisection over `q`.
///
/// Poisson/Binomial have unbounded support; the bracket is expanded
/// exponentially until the conditional mean exceeds the target. W families
/// are bounded to `q in (0, 1)`.
fn solve_q(
    family: OccupationFamily,
    target: f64,
    layers: u32,
    max_iterations: usize,
) -> Result<(f64, usize), EdgesEventsFitError> {
    let valid: bool = match family {
        OccupationFamily::Poisson | OccupationFamily::Binomial(_) => true,
        OccupationFamily::Geometric | OccupationFamily::NegativeBinomial(_) => target > 1.0,
    };
    if !valid {
        return Err(EdgesEventsFitError::MeanOutOfSupport);
    }
    if let OccupationFamily::Binomial(m) = family {
        if target > f64::from(m) {
            return Err(EdgesEventsFitError::MeanOutOfSupport);
        }
    }
    if target == 1.0 {
        return Ok((0.0, 0));
    }
    let (mut lo, mut hi) = match family {
        OccupationFamily::Poisson | OccupationFamily::Binomial(_) => {
            let mut hi = 1.0;
            let mut guard = 0;
            while conditional_mean(family, hi, layers) < target && guard < 64 {
                hi *= 2.0;
                guard += 1;
            }
            (0.0, hi)
        }
        OccupationFamily::Geometric | OccupationFamily::NegativeBinomial(_) => (0.0, 1.0 - 1e-15),
    };
    for iter in 0..max_iterations {
        let mid = 0.5 * (lo + hi);
        if conditional_mean(family, mid, layers) < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-14 * hi.max(1.0) {
            return Ok((0.5 * (lo + hi), iter + 1));
        }
    }
    Ok((0.5 * (lo + hi), max_iterations))
}

/// Fit the grand-canonical EDGES_EVENTS model.
///
/// # Arguments
/// - `family`: ME / B / W occupation family.
/// - `total_edges`: expected binary edge count `E`.
/// - `total_events`: expected event count `T`.
/// - `n_pairs`: number of candidate ordered pairs (N² or N(N-1) by self-loops).
/// - `max_iterations`: bisection iteration cap.
pub fn fit_edges_events(
    family: OccupationFamily,
    total_edges: f64,
    total_events: u64,
    n_pairs: u64,
    max_iterations: usize,
) -> Result<EdgesEventsFitResult, EdgesEventsFitError> {
    let e = total_edges;
    let t = total_events as f64;
    if e <= 0.0 {
        return Err(EdgesEventsFitError::ZeroEdges);
    }
    if t < e {
        return Err(EdgesEventsFitError::EventsBelowEdges);
    }
    if e > n_pairs as f64 {
        return Err(EdgesEventsFitError::EdgesAbovePairs);
    }
    let layers = match family {
        OccupationFamily::Binomial(m) | OccupationFamily::NegativeBinomial(m) => m,
        _ => 1,
    };
    let positive_mean = t / e;
    let (q, iterations) = solve_q(family, positive_mean, layers, max_iterations)?;
    let occupation = e / n_pairs as f64;
    let g = partition_factor(family, q);
    let lam = if g > 0.0 && occupation < 1.0 {
        occupation / ((1.0 - occupation) * g)
    } else {
        f64::INFINITY
    };
    Ok(EdgesEventsFitResult {
        q,
        lam,
        occupation,
        positive_mean,
        converged: true,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n_pairs(n: usize, self_loops: bool) -> u64 {
        if self_loops {
            (n * n) as u64
        } else {
            (n * (n - 1)) as u64
        }
    }

    #[test]
    fn me_recovers_expected_edges_and_events() {
        // N=10 no self-loops -> 90 pairs. E=45, T=270 -> avg 3, occupation 0.5.
        let fit = fit_edges_events(
            OccupationFamily::Poisson,
            45.0,
            270,
            n_pairs(10, false),
            100,
        )
        .expect("feasible");
        let g = fit.q.exp_m1();
        let occ = fit.lam * g / (1.0 + fit.lam * g);
        let mean = fit.lam * fit.q * fit.q.exp() / (1.0 + fit.lam * g);
        assert!((occ - 45.0 / 90.0).abs() < 1e-10);
        assert!((mean * 90.0 - 270.0).abs() < 1e-9);
        assert!(fit.converged);
    }

    #[test]
    fn b_recovers_expected_edges_and_events() {
        let fit = fit_edges_events(
            OccupationFamily::Binomial(4),
            30.0,
            90,
            n_pairs(10, false),
            100,
        )
        .expect("feasible");
        let m = 4.0_f64;
        let g = (1.0 + fit.q).powi(4) - 1.0;
        let occ = fit.lam * g / (1.0 + fit.lam * g);
        let mean = fit.lam * m * fit.q * (1.0 + fit.q).powi(3) / (1.0 + fit.lam * g);
        assert!((occ - 30.0 / 90.0).abs() < 1e-10);
        assert!((mean * 90.0 - 90.0).abs() < 1e-9);
    }

    #[test]
    fn w_geometric_recovers_expected_edges_and_events() {
        let fit = fit_edges_events(
            OccupationFamily::Geometric,
            30.0,
            120,
            n_pairs(10, false),
            100,
        )
        .expect("feasible");
        let q = fit.q;
        let g = q / (1.0 - q);
        let occ = fit.lam * g / (1.0 + fit.lam * g);
        let mean = fit.lam * (q / (1.0 - q).powi(2)) / (1.0 + fit.lam * g);
        assert!((occ - 30.0 / 90.0).abs() < 1e-10);
        assert!((mean * 90.0 - 120.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_infeasible_cases() {
        // T < E
        assert_eq!(
            fit_edges_events(OccupationFamily::Poisson, 50.0, 10, n_pairs(10, false), 100),
            Err(EdgesEventsFitError::EventsBelowEdges)
        );
        // E > n_pairs
        assert_eq!(
            fit_edges_events(
                OccupationFamily::Poisson,
                100.0,
                100,
                n_pairs(10, false),
                100
            ),
            Err(EdgesEventsFitError::EdgesAbovePairs)
        );
        // B mean > M
        assert_eq!(
            fit_edges_events(
                OccupationFamily::Binomial(2),
                10.0,
                30,
                n_pairs(10, false),
                100
            ),
            Err(EdgesEventsFitError::MeanOutOfSupport)
        );
        // E = 0
        assert_eq!(
            fit_edges_events(OccupationFamily::Poisson, 0.0, 0, n_pairs(10, false), 100),
            Err(EdgesEventsFitError::ZeroEdges)
        );
    }
}
