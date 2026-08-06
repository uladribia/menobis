//! Diagnostics for the fixed-degree support MCMC.

use crate::generation::microcanonical::mcmc::McmcCounters;

/// Representation mode — direct or complement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepresentationMode {
    Direct,
    Complement,
}

/// Heterogeneity classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegreeHeterogeneity {
    Light,
    Heterogeneous,
    HubDominated,
}

/// Diagnostics collected during support MCMC.
///
/// Embeds the generic [`McmcCounters`] for proposal/acceptance tracking,
/// plus degree-specific diagnostics.
#[derive(Clone, Debug)]
pub struct FixedDegreeDiagnostics {
    /// Generic MCMC counters (proposals, accepted, held, rejected).
    pub mcmc: McmcCounters,
    pub representation: RepresentationMode,
    pub heterogeneity: DegreeHeterogeneity,
    pub self_loop_holds: u64,
    pub duplicate_holds: u64,
    pub no_op_holds: u64,
}

impl FixedDegreeDiagnostics {
    pub fn new() -> Self {
        Self {
            mcmc: McmcCounters::new(),
            representation: RepresentationMode::Direct,
            heterogeneity: DegreeHeterogeneity::Light,
            self_loop_holds: 0,
            duplicate_holds: 0,
            no_op_holds: 0,
        }
    }

    /// Overall acceptance rate (delegates to [`McmcCounters::acceptance_rate`]).
    pub fn acceptance_rate(&self) -> f64 {
        self.mcmc.acceptance_rate()
    }
}

impl Default for FixedDegreeDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

/// Classify degree-sequence heterogeneity.
pub fn classify_heterogeneity(out_degrees: &[u32], in_degrees: &[u32]) -> DegreeHeterogeneity {
    let e: f64 = out_degrees.iter().map(|&d| d as f64).sum::<f64>() / 2.0;
    if e <= 0.0 {
        return DegreeHeterogeneity::Light;
    }
    let max_out = *out_degrees.iter().max().unwrap_or(&0) as f64;
    let max_in = *in_degrees.iter().max().unwrap_or(&0) as f64;
    let rho_max_out = max_out * max_out / (2.0 * e);
    let rho_max_in = max_in * max_in / (2.0 * e);
    let rho_max = rho_max_out.max(rho_max_in);

    if rho_max < 0.1 {
        DegreeHeterogeneity::Light
    } else if rho_max < 1.0 {
        DegreeHeterogeneity::Heterogeneous
    } else {
        DegreeHeterogeneity::HubDominated
    }
}

/// Try to complement the degree sequence if it reduces working edge count.
pub fn maybe_complement(
    out_degrees: &[u32],
    in_degrees: &[u32],
    self_loops: bool,
) -> (
    Vec<u32>,
    Vec<u32>,
    RepresentationMode,
    usize, // working edge count
) {
    let n = out_degrees.len() as u32;
    let max_allowed = if self_loops { n } else { n - 1 };
    let direct_e: usize = out_degrees.iter().map(|&d| d as usize).sum();

    let comp_out: Vec<u32> = out_degrees
        .iter()
        .map(|&d| max_allowed.saturating_sub(d))
        .collect();
    let comp_in: Vec<u32> = in_degrees
        .iter()
        .map(|&d| max_allowed.saturating_sub(d))
        .collect();
    let comp_e: usize = comp_out.iter().map(|&d| d as usize).sum();

    if comp_e < direct_e {
        (comp_out, comp_in, RepresentationMode::Complement, comp_e)
    } else {
        (
            out_degrees.to_vec(),
            in_degrees.to_vec(),
            RepresentationMode::Direct,
            direct_e,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_classification() {
        // N=20, each node out=1, in=1 → E=20, max=1, rho_max=1/20=0.05 < 0.1
        let out = vec![1u32; 20];
        let inp = vec![1u32; 20];
        assert_eq!(
            classify_heterogeneity(&out, &inp),
            DegreeHeterogeneity::Light
        );
    }

    #[test]
    fn hub_dominated() {
        // Hub out-degree dominates: max_out=100, E=108, rho_max=100^2/108≈92.6 >> 1
        let out = vec![100u32, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let inp = vec![10u32, 10, 10, 10, 10, 10, 10, 10, 10, 8]; // sum=108
        assert_eq!(
            classify_heterogeneity(&out, &inp),
            DegreeHeterogeneity::HubDominated
        );
    }
}
