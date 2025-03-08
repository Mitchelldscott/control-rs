//! # Control-rs
//!
//! Control-rs is a numerical modeling and analysis library designed for embedded applications.
//! Inspired by MATLAB's control systems toolbox, this crate provides a structured approach to
//! system modeling, analysis, and numerical design while maintaining a lightweight footprint suitable
//! for real-time and resource-constrained environments.
//!
//! ## Features
//! - **Modeling:** Support for Polynomial, Transfer Function, State-Space, and custom representations
//! - **Analysis:** Tools for classical, modern and robust system analysis
//! - **Synthesis:** Direct and data-driven methods to create models
//! - **Simulation:** Easy model integration and data vizualization
//! 
//! ## Design Philosophy
//! This crate is structured around core numerical model representations, each implementing common traits to have
//! a consistent interface for simulation, analysis, and synthesis. This is all done to provide a clean interface
//! for users to turn models into datasets and datasets into models.
//!
//! ***A lot of docs were written by throwing bullet points into ChatGPT, some hallucinations may have snuck in.
//! Please report any you find.***
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub mod assertf;

pub mod state_space;
pub use state_space::StateSpace;

pub mod transfer_function;
pub use transfer_function::TransferFunction;

pub mod frequency_tools;

pub mod polynomial;
pub use polynomial::Polynomial;

pub mod integrators;

/// # Numerical Function trait
///
/// This trait provides a universal interface for evalutating numerical models.
///
/// model must be in the form:
/// <pre>
/// y = f(x)
/// </pre>
pub trait NumericalFunction<T> {
    /// Evaluates the function for the given input
    fn __evaluate(&self, x: T) -> T;
}

/// # Nonlinear Model
///
/// This allows users to implement a linearization of a nonlinear model. This also provides a
/// trait bound for algorithms that use linearization.
///
/// # Generic Arguments
///
/// * `T` - type of the state, input and output values
/// * `Input` - type of the input vector
/// * `State` - type of the state vector
/// * `Output` - type of the output vector
///
/// ## References
///
/// - *Nonlinear Systems*, Khalil, Ch. 2: Nonlinear Models.
///
/// ## TODO:
/// - [ ] move generics to type aliases, the <> are too full
/// - [ ] add generic linearization so users don't need to define a custom one (derive?)
/// - [ ] add LinearModel trait so custom models can be linearized to other forms (linear multivariate polynomial?)
pub trait NLModel<Input, State, Output, A, B, C, D>:
    DynamicModel<Input, State, Output> {
    /// Linearizes the system about a nominal state and input
    fn linearize(&self, x: State, u: Input) -> StateSpace<A, B, C, D>;
}

/// # Dynamic Model
///
/// This trait provides a universal interface for evalutating numerical models.
///
/// model must be in the form:
/// <pre>
/// ẋ = f(x, u)
/// y = h(x, u)
/// </pre>
///
/// # Generic Arguments
///
/// * `Input` - type of the input variable(s)
/// * `State` - type of the state variable(s)
/// * `Output` - type of the output variable(s)
pub trait DynamicModel<Input, State, Output> {
    /// Evaluates the dynamics of the state for the given state and input
    fn dynamics(&self, x: State, u: Input) -> State;
    /// Evaluates the model's output for the given state and input
    fn output(&self, x: State, u: Input) -> Output;
}