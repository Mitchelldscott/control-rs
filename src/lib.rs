//!
//! Tools for numerical model synthesis and analysis
//!
//! This toolbox is inpired by MATLab but is intended to be used in embedded applications.
//!
//! # Controls Tools
//!
//! This crate intends to provide a data driven approach to system modelling and controller design.
//! The crate is built around a few numerical model formats (transfer function, state space and
//! nonlinear), each of these model formats will implement traits that create a common interface
//! for simulating, analyzing and synthesizing a model.
//!
//! ## Examples
//!
//! The goal is that all examples are implementations of a theoritical example in a textbook. There may also
//! be more practical examples that help users integrate these tools with embedded systems (like how to prepare
//! and store data for synthesis). More specific hardware integrated examples will be provided in a
//! cargo template (but thats a long ways off).
//!
//! ## Tests
//!
//! ### Unit
//!
//! Unit tests will be done inside of the module containing the unit that being tested (i.e. src/<module>/...).
//! Simple tests may be done in mod.rs as a reference, but more complex tests will be in their own files to minimize the
//! length of mod.rs. The current plan is to have a test module for each unit, test modules then provide many test cases.
//! Each test case should be short, independent and descriptive of the test being done.
//!
//! ### Integration
//!
//! Integration tests will be done in the tests/ directory. These tests will seem like examples but the models
//! they work with may appear trivial. The purpose of this is to provide a high level check that the tools are
//! working. An example of this is a test that confirms a simple model is identical across model formats.
//!
//! ## Docs
//!
//! The docs are meant to be as much of a theory reference as a user guide. Each module should have:
//! - short conceptual description
//! - links to theoretical references
//! - use cases (doctests)
//!
//! *Will hopefully split the docs into pure dev info and a book with theory/use cases that should be used together*
//!
//! ## TODO
//! - [ ] transfer function: improving tests and docs
//!   - [ ] system interconnections (add, sub, mul, feedback)
//!   - [ ] document FrequencyTools impl (should it move to frequency_tools?)
//!   - [ ] find more productive examples of dc gain, lhp and as monic from textbooks
//!   - [ ] use textbook examples to make edge case/usage tests for tf
//!   - [ ] example from textbook for transfer function struct docs (with model and problem)
//! - [ ] state space
//!   - [ ] frequency tools
//!   - [ ] system interconnections (add, sub, mul, feedback)
//!   - [ ] sigular_values
//!   - [ ] eigens
//!   - [ ] is_stable
//!   - [ ] is_controllable
//!   - [ ] is_observable
//!   - [ ] lqr
//!   - [ ] kalman
//!   - [ ] hinf
//!   - [ ] h2
//!   - [ ] pole placement
//!   - [ ] tests
//!   - [ ] docs
//! - [ ] nl model: modules w/ traits (lyapunov, bifurcation, auto-differentiation?...)

pub mod assertf;

pub mod state_space;
pub use state_space::StateSpace;

pub mod transfer_function;
pub use transfer_function::TransferFunction;

pub mod frequency_tools;

pub mod polynomial;

use std::ops::{Add, Div, Mul};

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
    /// in a loop (convenience funtion coming soon).
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
    /// * `x` - state at tf
    /// * `y` - system output at tf
    fn rk4(&self, dt: T, t0: T, tf: T, x0: State, u: Input) -> State
    where
        T: Copy + PartialEq + PartialOrd + From<u8> + Add<Output = T> + Div<Output = T>,
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
/// - [ ] add generic linearization so users don't need to define a custom one (derive?)
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
