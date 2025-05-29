//! # Control-rs
//!
//! Control-rs is a numerical modeling and analysis library designed for embedded applications.
//! Inspired by MATLAB's control systems toolbox, this crate provides a structured approach to
//! system modeling, analysis and numerical design while maintaining a lightweight footprint suitable
//! for real-time and resource-constrained environments.
//!
//! ## Features
//! - **Modeling**: Support for Polynomial, Transfer Function, State-Space and custom representations
//! - **Analysis**: Tools for classical, modern and robust system analysis
//! - **Synthesis**: Direct and data-driven methods to create models
//! - **Simulation**: Easy model integration and data visualization
//!
//! ## Design Philosophy
//! This crate is structured around core numerical model representations, each implementing common traits to have
//! a consistent interface for simulation, analysis and synthesis. This is all done to provide a clean interface
//! for users to turn models into datasets and datasets into models.
//!
//! ***A lot of docs were written by throwing bullet points into ChatGPT, some hallucinations may have snuck in.
//! Please report any you find.***
#![warn(missing_docs, unused)]
#![cfg_attr(not(feature = "std"), no_std)]

pub mod assertf;

pub mod state_space;
pub use state_space::StateSpace;

pub mod transfer_function;
pub use transfer_function::TransferFunction;

pub mod frequency_tools;

// pub mod vector;
// // pub use vector::Vector;
//
// pub mod matrix;
// // pub use matrix::Matrix;

pub mod polynomial;
pub use polynomial::Polynomial;

pub mod integrators;

pub mod math;
pub use math::systems;
