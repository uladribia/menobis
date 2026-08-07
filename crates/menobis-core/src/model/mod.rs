//! Model-level abstractions: family, constraints, problem, sampling plan.
//!
//! This module defines the shared types that encode the mathematical
//! structure of a MENoBiS model — the occupation family, hard/soft
//! constraints, the prepared problem, and the sampling-plan routing.
//!
//! The types here are the architectural "header": the vocabulary that
//! generation, filtering, and fitting modules use to communicate.
//! They are not tied to any particular backend.

pub mod family;
pub mod problem;
pub mod sampling_plan;

pub use family::OccupationFamily;
pub use problem::PreparedProblem;
pub use sampling_plan::SamplingPlan;
