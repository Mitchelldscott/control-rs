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
pub trait NLModel<Input, State, Output, A, B, C, D>: DynamicModel<Input, State, Output> {
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

#[cfg(test)]
mod feedback_test {
    use num_traits::{One, Zero};
    use nalgebra::{U1, Scalar, DimName, RawStorageMut, allocator::Allocator, DefaultAllocator};

    #[cfg(feature = "std")]
    use std::ops::{Sub, Div, Mul};
    
    #[cfg(not(feature = "std"))]
    use core::ops::{Sub, Div, Mul};

    use crate::{Polynomial, TransferFunction};
    
    trait NumericalModel: Clone {
        fn zero() -> Self;
        fn identity() -> Self;
    }

    impl<T, D, S, N> NumericalModel for Polynomial<T, D, S, N> 
    where 
        T: Scalar + Zero + One,
        D: DimName,
        S: Clone + RawStorageMut<T, D, N>,
        N: DimName,
        DefaultAllocator: Allocator<D, N, Buffer<T> = S>,
    {
        fn zero() -> Self {
            Polynomial::from_element_generic(D::name(), N::name(), T::zero())
        }

        fn identity() -> Self {
            let mut ident = Polynomial::from_element_generic(D::name(), N::name(), T::zero());
            let num_coeff = ident.num_coefficients();
            for _ in 0..ident.num_equations() {
                ident[num_coeff - 1] = T::one();
            }
            ident
        }
    }

    impl<T, M, N, S1, S2> NumericalModel for TransferFunction<T, M, N, S1, S2> 
    where 
        T: Scalar + Zero + One + Copy,
        M: DimName,
        N: DimName,
        S1: Clone + RawStorageMut<T, M>,
        S2: Clone + RawStorageMut<T, N>,
        DefaultAllocator: Allocator<M, U1, Buffer<T> = S1>
            + Allocator<N, U1, Buffer<T> = S2>,
    {
        fn zero() -> Self {
            TransferFunction {
                numerator: Polynomial::zeros_generic(M::name(), U1),
                denominator: Polynomial::zeros_generic(N::name(), U1),
            }
        }

        fn identity() -> Self {
            let mut ident = TransferFunction::zero();
            let m = ident.numerator.num_coefficients() - 1;
            let n = ident.denominator.num_coefficients() - 1;
            ident.numerator[m] = T::one();
            ident.denominator[n] = T::one();

            ident
        }
    }

    fn feedback<T, G, H, GH, D>(
        sys1: &G,
        sys2: &H,
        sign_in: T,
        sign_feedback: T,
    ) -> D
    where
        T: Clone + Zero + One +  Mul<G, Output = G> + Mul<GH, Output = GH>,
        G: NumericalModel + Mul<H, Output = GH> + Div<GH, Output = D>,
        GH: NumericalModel + Sub<Output = GH>,
        H: NumericalModel,
    {
        sign_in.clone() * sys1.clone() / (GH::identity() - sign_feedback.clone() * sys1.clone() * sys2.clone())
    }

    // #[test]
    // fn polynomial_polynomial() {
    //     let p1 = Polynomial::new([1.0]);
    //     let p2 = Polynomial::new([2.0]);
    //     let p3 = feedback(&p1, &p2, 1.0, -1.0);
    //     assert_eq!(p3.coefficients, [-1.0], "incorrect feedback polynomial");
    // }

    #[test]
    fn tf_tf() {
        let tf1 = TransferFunction::new(
            [1.0],
            [1.0, 0.0]
        );
        let tf2 = TransferFunction::new(
            [1.0],
            [1.0, 0.0]
        );
        let tf3 = feedback(&tf1, &tf2, 1.0, -1.0);
    }
}