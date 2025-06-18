//! higher level traits for numerical models

use core::ops::{Add, Div, Mul};
use num_traits::{One, Zero};
use crate::state_space::StateSpace;

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
pub fn feedback<T, G, H, GH, GH2, CL>(sys1: &G, sys2: &H, sign_in: T, sign_feedback: T) -> CL
where
    T: Clone + Zero + One + Mul<G, Output = G> + Mul<GH, Output = GH>,
    G: System + Mul<H, Output = GH> + Div<GH2, Output = CL>,
    GH: System + Add<Output = GH2>,
    H: System,
{
    sign_in * sys1.clone() / (GH::identity() + sign_feedback * sys1.clone() * sys2.clone())
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

#[cfg(test)]
mod feedback_test {
    use super::*;

    use crate::{
        assert_f32_eq,
        TransferFunction,
        polynomial::Constant,
    };

    #[test]
    fn zero_constant() {
        let p1 = Constant::zero();
        let p2 = Constant::new([2.0]);
        let p3: Constant<f64> = feedback(&p1, &p2, 1.0, -1.0);
        assert_eq!(
            p3.coefficient(0),
            Some(&0.0),
            "incorrect feedback polynomial {p3}"
        );
    }

    #[test]
    fn constant_constant() {
        // x = r - m where, m = p2(p1(x)) and r is an arbitrary input to the system
        let p1 = Constant::new([1.0]);
        let p2 = Constant::new([2.0]);
        let p3 = feedback(&p1, &p2, 1.0, -1.0);
        assert_eq!(
            p3.coefficient(0),
            Some(&-1.0),
            "incorrect feedback polynomial {p3}"
        );
    }

    #[test]
    fn tf_closed_loop() {
        let tf = TransferFunction::new([1.0], [1.0, 1.0]);
        let cl_tf = feedback(&tf, &1.0, 1.0, -1.0);
        assert_f32_eq!(cl_tf.numerator[0], 1.0);
        assert_f32_eq!(cl_tf.numerator[1], 1.0);
        assert_f32_eq!(cl_tf.numerator[2], 0.0);
        assert_f32_eq!(cl_tf.denominator[0], 0.0);
        assert_f32_eq!(cl_tf.denominator[1], 1.0);
        assert_f32_eq!(cl_tf.denominator[2], 1.0);
    }
}
