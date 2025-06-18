//! # Control-rs
//!
//! `control-rs` is structured around core numerical model representations, each implementing common
//! traits to have a consistent interface for simulation, analysis and synthesis. This is all done
//! to provide a clean interface for users to turn models into datasets and datasets into models.
//!
//! Each numerical model type has its own module and utilities. The models are re-exported to be
//! available from the crate root while the utilities are only available through the specific
//! modules.
//!
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
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(rust_2018_idioms)]
#![warn(missing_docs, unused)]

pub mod state_space;
pub use state_space::StateSpace;

pub mod transfer_function;
pub use transfer_function::TransferFunction;

pub mod frequency_tools;

pub mod polynomial;
pub use polynomial::Polynomial;

pub mod integrators;

pub mod math;
pub use math::systems;
