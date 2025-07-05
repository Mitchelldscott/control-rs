#![doc = include_str!("../README.md")]
//!
//! The point of separating utilities from each model type is that a lot of the utilities are
//! re-used throughout the crate. This structure allows each model to exist independently of the
//! others.
#![no_std]
#![deny(
    clippy::all,
    clippy::todo,
    clippy::panic,
    clippy::nursery,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::unimplemented
)]
#![warn(unused, missing_docs, rust_2018_idioms)]

pub mod frequency_tools;

pub mod integrators;

pub mod math;
pub use math::systems;

pub mod polynomial;
pub use polynomial::Polynomial;

pub mod state_space;
pub use state_space::StateSpace;

pub mod static_storage;

pub mod transfer_function;
pub use transfer_function::TransferFunction;
