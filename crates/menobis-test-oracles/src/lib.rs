//! MENoBiS test oracles — exact enumeration and reference algorithms.
//!
//! This crate provides expensive but exact algorithms for validating
//! production samplers.  It is development/test infrastructure only;
//! it must never become a production dependency of `menobis-core`.

pub mod enumeration;
