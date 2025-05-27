//! # Control-rs
//!
//! Control-rs is a numerical modeling and analysis library designed for embedded applications.
//! Inspired by MATLAB's control systems toolbox, this crate provides a structured approach to
//! system modeling, analysis and numerical design while maintaining a lightweight footprint suitable
//! for real-time and resource-constrained environments.
//!
//! ## Features
//! - **Modeling**: Support for Polynomial, Transfer Function, State-Space, and custom representations
//! - **Analysis**: Tools for classical, modern and robust system analysis
//! - **Synthesis**: Direct and data-driven methods to create models
//! - **Simulation**: Easy model integration and data visualization
//!
//! ## Design Philosophy
//! This crate is structured around core numerical model representations, each implementing common traits to have
//! a consistent interface for simulation, analysis, and synthesis. This is all done to provide a clean interface
//! for users to turn models into datasets and datasets into models.
//!
//! ***A lot of docs were written by throwing bullet points into ChatGPT, some hallucinations may have snuck in.
//! Please report any you find.***
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs, unused)]

pub mod assertf;

pub mod state_space;
pub use state_space::StateSpace;

pub mod transfer_function;
pub use transfer_function::TransferFunction;

pub mod frequency_tools;

pub mod vector;
// pub use vector::Vector;

pub mod matrix;
// pub use matrix::Matrix;

pub mod polynomial;
pub use polynomial::Polynomial;

pub mod integrators;

mod math;
pub use math::{num_traits, system_traits};

#[cfg(test)]
mod feedback_test {
    use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, RawStorageMut, Scalar, U1};
    use num_traits::{One, Zero};

    #[cfg(feature = "std")]
    use std::ops::{Div, Mul, Sub};

    #[cfg(not(feature = "std"))]
    use core::ops::{Div, Mul, Sub};

    use crate::{math, Polynomial, TransferFunction};

    trait NumericalModel: Clone {
        fn zero() -> Self;
        fn identity() -> Self;
    }

    impl<T, const N: usize> NumericalModel for Polynomial<T, N>
    where
        T: Copy + Clone + num_traits::Num,
    {
        fn zero() -> Self {
            Polynomial::from_element(T::zero())
        }

        fn identity() -> Self { Polynomial::from_constant(T::one()) }
    }

    // impl<T, M, N, S1, S2> NumericalModel for TransferFunction<T, M, N, S1, S2>
    // where
    //     T: Scalar + Zero + One + Copy,
    //     M: DimName,
    //     N: DimName,
    //     S1: Clone + RawStorageMut<T, M>,
    //     S2: Clone + RawStorageMut<T, N>,
    //     DefaultAllocator: Allocator<M, U1, Buffer<T> = S1> + Allocator<N, U1, Buffer<T> = S2>,
    // {
    //     fn zero() -> Self {
    //         TransferFunction {
    //             numerator: Polynomial::zeros_generic(M::name(), U1),
    //             denominator: Polynomial::zeros_generic(N::name(), U1),
    //         }
    //     }
    //
    //     fn identity() -> Self {
    //         let mut ident = TransferFunction::zero();
    //         let m = ident.numerator.num_coefficients() - 1;
    //         let n = ident.denominator.num_coefficients() - 1;
    //         ident.numerator[m] = T::one();
    //         ident.denominator[n] = T::one();
    //
    //         ident
    //     }
    // }

    fn feedback<T, G, H, GH, D>(sys1: &G, sys2: &H, sign_in: T, sign_feedback: T) -> D
    where
        T: Clone + Zero + One + Mul<G, Output = G> + Mul<GH, Output = GH>,
        G: NumericalModel + Mul<H, Output = GH> + Div<GH, Output = D>,
        GH: NumericalModel + Sub<Output = GH>,
        H: NumericalModel,
    {
        sign_in.clone() * sys1.clone()
            / (GH::identity() - sign_feedback.clone() * sys1.clone() * sys2.clone())
    }

    // #[test]
    // fn polynomial_polynomial() {
    //     // The input to p1 is the difference of the input to the system and
    //     // the output of p2. The output of p1 is the output of the system and
    //     // the input to p2.
    //     // x = r - m, m = p2(p1(x))
    //     let p1 = Polynomial::new([1.0]);
    //     let p2 = Polynomial::new([2.0]);
    //     let p3 = feedback(&p1, &p2, 1.0, -1.0);
    //     assert_eq!(p3.coefficients.0, [[-1.0]], "incorrect feedback polynomial");
    // }

    // #[test]
    // fn tf_tf() {
    //     let tf1 = TransferFunction::new(
    //         [1.0],
    //         [1.0, 0.0]
    //     );
    //     let tf2 = TransferFunction::new(
    //         [1.0],
    //         [1.0, 0.0]
    //     );
    //     let tf3 = feedback(&tf1, &tf2, 1.0, -1.0);
    // }
}
