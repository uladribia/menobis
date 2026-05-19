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

/// Solver status for conic W fitting routines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WFitStatus {
    /// Solver returned an optimal solution within requested tolerances.
    Solved,
    /// Solver stopped at the iteration limit or another nonfatal condition.
    Inaccurate,
    /// Solver reported primal/dual infeasibility or invalid inputs.
    Infeasible,
    /// Solver failed before returning usable multipliers.
    Failed,
}

/// Size metrics for lifted conic W fitting problems.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WProblemMetrics {
    pub variables: usize,
    pub auxiliary_variables: usize,
    pub exponential_cones: usize,
    pub power_cones: usize,
    pub linear_constraints: usize,
    pub sparse_nonzeros: usize,
}

/// Residual summary for fitted W strength constraints.
#[derive(Clone, Debug, Default)]
pub struct WStrengthResiduals {
    pub max_abs: f64,
    pub total_abs: f64,
    pub min_margin: f64,
    pub max_q: f64,
}

/// Options for conic W fitting routines.
#[derive(Clone, Copy, Debug)]
pub struct WConicFitOptions {
    pub self_loops: bool,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for WConicFitOptions {
    fn default() -> Self {
        Self {
            self_loops: false,
            tolerance: 1e-9,
            max_iterations: 1_000,
        }
    }
}

/// Fitted multipliers and diagnostics for independent W fixed-strength models.
#[derive(Clone, Debug)]
pub struct WStrengthFitResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub layers: u32,
    pub status: WFitStatus,
    pub objective: f64,
    pub iterations: usize,
    pub min_margin: f64,
    pub max_q: f64,
    pub max_strength_residual: f64,
    pub total_strength_residual: f64,
    pub metrics: WProblemMetrics,
}
