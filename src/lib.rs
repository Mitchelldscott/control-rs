//! # Control-rs
//!
//! Control-rs is a numerical modeling and control system library designed for embedded applications.
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
//! This crate is structured around core numerical model representations, each implementing traits that ensure
//! a consistent interface for simulation, analysis, and synthesis. This is all done to provide a clean interface
//! for users to turn models into datasets and datasets into models. There is no IDE or GUI, because this crate is
//! meant to be used in the field (In the future a separate crate may provide a visualization dashboard).
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

#[cfg(feature = "std")]
use std::ops::{Add, Div, Mul};

#[cfg(not(feature = "std"))]
use core::ops::{Add, Div, Mul};

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

/// trait for all dynamic systems (i.e. systems with state and output equations).
pub trait DynamicSystem<T> {
    /// a numerical function whose input is (state, input) and output is the state update
    type DynamicFunction: NumericalFunction<T>;
    /// a numerical function whose input is (state, input) and output is the systems output
    type OutputFunction: NumericalFunction<T>;
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
pub trait NLModel<T, Input, State, Output, const N: usize, const M: usize, const L: usize>:
    DynamicModel<T, Input, State, Output>
where
    Input: Copy,
    State: Copy,
    Output: Copy,
{
    /// Linearizes the system about a nominal state and input
    fn linearize(&self, x: State, u: Input) -> StateSpace<T, N, M, L>;
}

/// # Dynamic Model
///
/// This trait provides a universal interface for evalutating numerical models.
///
/// model must be in the form:
/// <pre>
/// dx = f(x, u)
/// y = h(x, u)
/// </pre>
///
/// # Generic Arguments
///
/// * `T` - type of the state, input and output values
/// * `Input` - type of the input vector
/// * `State` - type of the state vector
/// * `Output` - type of the output vector
pub trait DynamicModel<T, Input, State, Output>
where
    Input: Copy,
    State: Copy,
    Output: Copy,
{
    /// Evaluates the dynamics of the state for the given state and input
    fn f(&self, x: State, u: Input) -> State;
    /// Evaluates the model's output for the given state and input
    fn h(&self, x: State, u: Input) -> Output;

    /// Simulate the system for a given time interval
    ///
    /// The time interval is assumed to be small enough the input will be constant for the
    /// duration of the integration. For simulations with time-varying input call this repeatedly
    /// in a loop.
    ///
    /// # Arguments
    ///
    /// * `dt` - length of a step
    /// * `t0` - start time
    /// * `tf` - end time
    /// * `x0` - initial state
    /// * `u` - input
    ///
    /// # Returns
    ///
    /// * `x` - state at end time
    /// * `y` - system output at end time
    fn rk4(&self, dt: T, t0: T, tf: T, x0: State, u: Input) -> State
    where
        T: Copy + PartialOrd + PartialEq + From<u8> + Add<Output = T> + Div<Output = T>,
        State: Add<Output = State> + Mul<T, Output = State>,
    {
        let dt_2 = dt / T::from(2);
        let dt_3 = dt / T::from(3);
        let dt_6 = dt / T::from(6);

        let mut t = t0;
        let mut x = x0;

        while t < tf {
            let k1 = self.f(x, u);
            let k2 = self.f(x + k1 * dt_2, u);
            let k3 = self.f(x + k2 * dt_2, u);
            let k4 = self.f(x + k3 * dt_2, u);
            x = x + k1 * dt_6 + k2 * dt_3 + k3 * dt_3 + k4 * dt_6;
            t = t + dt;
        }
        x
    }
}
