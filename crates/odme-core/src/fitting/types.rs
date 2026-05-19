//! Shared fitting result types.

#[derive(Clone, Debug)]
pub struct FitResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}

/// Fitted multipliers for exact ME fixed-strength-and-edge-count models.
#[derive(Clone, Debug)]
pub struct StrengthEdgesFitResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub lam: f64,
    pub converged: bool,
    pub iterations: usize,
}

/// Fitted multipliers for exact ME fixed-strength-degree models.
#[derive(Clone, Debug)]
pub struct StrengthDegreeFitResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub w: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}

/// Fitted multipliers for ME fixed-strength-and-cost models.
#[derive(Clone, Debug)]
pub struct StrengthCostFitResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub gamma: f64,
    pub converged: bool,
    pub iterations: usize,
}

/// Sparse rate table fitted under partial constraints.
#[derive(Clone, Debug)]
pub struct PartialFitResult {
    pub sources: Vec<u64>,
    pub targets: Vec<u64>,
    pub rates: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}
