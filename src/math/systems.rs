//! higher level traits for numerical models

use crate::state_space::StateSpace;
use core::ops::{Add, Div, Mul};
use num_traits::{One, Zero};

/// # Dynamical System
///
/// This trait provides a universal interface for propagating systems with internal states.
///
/// The interface implements the generic form:
/// <pre>
/// ẋ = f(x, u)
/// y = h(x, u)
/// </pre>
///
/// Where <pre>f</pre> is the dynamics function and <pre>h</pre> is the output function.
///
/// # Generic Arguments
/// * `Input` - Input variable(s)
/// * `State` - State variable(s)
/// * `Output` - Output variable(s)
pub trait DynamicalSystem<Input, State, Output> {
    /// Evaluates the dynamics of the state for the given state and input
    fn dynamics(&self, x: State, u: Input) -> State;
    /// Evaluates the model's output for the given state and input
    fn output(&self, x: State, u: Input) -> Output;
}

/// # Nonlinear Model
///
/// This allows users to implement a linearization of a nonlinear model. This also provides a
/// trait bound for algorithms that use linearization.
///
/// # Generic Arguments
/// * `Input` - input vector
/// * `State` - state vector
/// * `Output` - output vector
/// * `A` - state matrix
/// * `B` - input matrix
/// * `C` - output matrix
/// * `D` - feedthrough matrix
///
/// ## References
/// - *Nonlinear Systems*, Khalil, Ch. 2: Nonlinear Models.
///
pub trait NLModel<Input, State, Output, A, B, C, D>: DynamicalSystem<Input, State, Output> {
    // # TODO:
    //     * Generic linearization so users don't need to define a custom one (derive?)
    //     * LinearModel trait so custom models can be linearized to other forms (linear multivariate polynomial?)
    /// Linearizes the system about a nominal state and input
    fn linearize(&self, x: State, u: Input) -> StateSpace<A, B, C, D>;
}

/// Core trait for numerical models
///
/// This is used to interconnect and evaluate models
pub trait System: Clone {
    /// Create a "zero" system
    fn zero() -> Self;

    /// Create an "identity" system
    fn identity() -> Self;
}

/// Feedback of two systems
///
/// This function implements the feedback of two systems.
///
/// # Generic Arguments
/// * `T` - Type of the input variable(s)
pub fn feedback<T, G, H, GH, CL>(sys1: &G, sys2: &H, sign_in: T, sign_feedback: T) -> CL
where
    T: Clone + Zero + One + Mul<G, Output = G> + Add<GH, Output = GH> + Mul<GH, Output = GH>,
    G: System + Mul<H, Output = GH> + Div<GH, Output = CL>,
    GH: System,
    H: System,
{
    sign_in * sys1.clone() / (T::one() + sign_feedback * sys1.clone() * sys2.clone())
}

impl System for f32 {
    fn zero() -> Self {
        0.0f32
    }
    fn identity() -> Self {
        1.0f32
    }
}
impl System for f64 {
    fn zero() -> Self {
        0.0f64
    }
    fn identity() -> Self {
        1.0f64
    }
}
