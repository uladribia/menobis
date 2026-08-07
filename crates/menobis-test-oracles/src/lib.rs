//! MENoBiS test oracles — exact enumeration and reference algorithms.
//!
//! # Purpose
//!
//! This crate provides **expensive but exact** algorithms for validating
//! production samplers.  It is **development/test infrastructure only** and
//! must **never** become a production dependency of `menobis-core`.
//!
//! A minimal installation of MENoBiS (`menobis-core`) does **not** include
//! this crate.  Heavy enumeration-based tests live here so that
//! `cargo test -p menobis-core` runs only fast unit tests.
//!
//! # What belongs here
//!
//! - Exact enumeration of small occupation-number state spaces
//! - Exhaustive binary support enumeration
//! - Exact probability computation
//! - Detailed-balance / transition-matrix verification
//! - Multi-chain convergence and mixing validation
//! - Reference implementations of old exact samplers (for regression
//!   testing against new scalable backends)
//!
//! # What stays in `menobis-core`
//!
//! - Fast unit tests for formulas, invariants, and API behaviour
//! - Property-based tests that run in <1 s
//! - Small deterministic correctness checks

pub mod enumeration;
