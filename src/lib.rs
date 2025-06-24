//! # Control-rs
//!
//! Rust-native numerical modeling and synthesis library for embedded projects.
//!
//! The crate is `no_std` by default (but has a `std` feature flag for plotting) and intends to support
//! both fixed-point and floating-point numeric types.
//!
//! The goal is to make cargo templates for robotic components like ESCs, BMS and odometry
//! systems. These templates will use the
//! [awesome embedded rust crates](https://!github.com/rust-embedded/awesome-embedded-rust)
//! to provide detailed guides to implementing and operating the components.
//!
//! This list covers a few projects that are in the works:
//! - [ ] DC Motor lead-lag compensator
//! - [ ] BLDC ESC (FOC or fancy 6-stage commuter)
//! - [ ] Lipo Battery model adaptive estimator
//! - [ ] Quadcopter attitude/altitude controller (3-loop autopilot)
//! - [ ] Visual-Inertial Odometry
//!
//! ## Features
//! ### Model Types
//!
//! * [Polynomial] - Dense univariate polynomial.
//! * [`TransferFunction`] - Intended to be a laplace domain input/output model but could potentially
//!   be used as a rational function.
//! * [`StateSpace`] - Standard linear-algebra representation for a system of differential equations.
//! * [`NLModel`] - A trait for custom models that provides a more flexible structure/implementation.
//!
//! ### Analysis Tools
//!
//! * [`FrequencyTools`] - Classical frequency-response methods
//! * `RobustTools` - Hopefully coming soon
//!
//! ### Synthesis Tools
//!
//! * `LeastSquares` - Trait that's still in the works (should be available for the statically
//!   sized models).
//! * `GradientDescent` - Also in the works but should provide a trait to perform backpropagation of
//!   error on models.
//!
//! ### Simulation Tools
//!
//! * [integrators] - Various types of integration for precision simulations (RK4 + Dormand-Prince)
//! * `response` - Classic system response implementations: step, ramp, sine, impulse...
//!
//! # Example
//! ```
//! use control_rs::{TransferFunction, StateSpace, Polynomial};
//! use control_rs::{
//!     transfer_function::utils::tf2ss,
//!     polynomial::utils::convolution
//! };
//! ```
//!
//! The point of separating utilities from each model type is that a lot of the utilities are
//! re-used throughout the crate. This structure allows each model to exist independently of the
//! others.
//!
//! ***A lot of docs were written by throwing bullet points into `ChatGPT`, some hallucinations may have snuck in.
//! Please report any you find.***
//!
//! # References
//! * [Control Systems Wiki book](https://en.wikibooks.org/wiki/Control_Systems)
//! * [Feedback control of Dynamic Systems](https://mrce.in/ebooks/Feedback%20Control%20of%20Dynamic%20Systems%208th%20Ed.pdf)
#![no_std]
#![deny(
    clippy::all,
    clippy::todo,
    clippy::panic,
    clippy::nursery,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::unimplemented,
)]
#![warn(
    unused,
    missing_docs,
    rust_2018_idioms,
)]

pub mod frequency_tools;

pub mod integrators;

pub mod math;
pub use math::systems;

pub mod polynomial;
pub use polynomial::Polynomial;

pub mod state_space;
pub use state_space::StateSpace;

pub mod transfer_function;
pub use transfer_function::TransferFunction;
