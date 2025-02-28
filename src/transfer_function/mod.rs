//! # Transfer Function representation and tools
//!
//! "A transfer function is a convenient way to represent a linear, time-invariant system in terms
//! of its input-output relationship. It is obtained by applying a Laplace transform to the
//! differential equations describing system dynamics, assuming zero initial conditions. In the
//! absence of these equations, a transfer function can also be estimated from measured
//! input-output data.
//!
//! Transfer functions are frequently used in block diagram representations of systems and are
//! popular for performing time-domain and frequency-domain analyses and controller design. The
//! key advantage of transfer functions is that they allow engineers to use simple algebraic
//! equations instead of complex differential equations for analyzing and designing systems."
//!
//! > [MathWorks](https://www.mathworks.com/discovery/transfer-function.html)
//!

#[cfg(feature = "std")]
use std::{fmt, ops::Neg};

#[cfg(not(feature = "std"))]
use core::{fmt, ops::Neg};

use nalgebra::{ArrayStorage, Complex, Const, Dim, RawStorage, RealField, U1};
use num_traits::{Float, Num};

pub mod linear_tools;
pub use linear_tools::*;

use crate::{
    frequency_tools::{FrequencyResponse, FrequencyTools},
    polynomial::Polynomial,
};

/// # Transfer Function
///
/// <pre>
/// G(s) = b(s) / a(s)
/// a(s) = (s^N + a_N s^(N-1) + ... + a_1)
/// b(s) = (b_N s^(N-1) + b_(N-1) s^(N-2) + ... + b_1)
/// </pre>
///
/// Stores two nalgebra vectors representing coefficients of polynomials, one for the numerator
/// and one for the denominator.
///
/// # Generic Arguments
///
/// * `T` - type of the coefficients
/// * `M` - order of the numerator
/// * `N` - order of the denominator
///
/// ## References
///
/// - *Feedback Control of Dynamic Systems*, Franklin et al., Ch. 3.1
pub struct TransferFunction<T, M, N, S1, S2> {
    /// coefficients of the numerator `[bn, ... b2, b1]`
    pub numerator: Polynomial<T, M, S1>,
    /// coefficients of the denominator `[an, ... a1, a0]`
    pub denominator: Polynomial<T, N, S2>,
}

impl<T, const M: usize, const N: usize>
    TransferFunction<T, Const<M>, Const<N>, ArrayStorage<T, M, 1>, ArrayStorage<T, N, 1>>
{
    /// Create a new transfer function from arrays of coefficients
    ///
    /// # Arguments
    ///
    /// * `numerator` - coefficients of the numerator `[bm, ... b2, b1]`
    /// * `denominator` - coefficients of the denominator `[an, ... a1, a0]`
    ///
    /// # Returns
    ///
    /// * `TransferFunction` - a new TransferFunction object
    ///
    /// # Example
    ///
    /// ```
    /// use control_rs::TransferFunction;
    ///
    /// fn main() {
    ///     let tf = TransferFunction::new([1.0, 1.0], [1.0, 1.0, 1.0]);
    ///     println!("{tf}");
    /// }
    /// ```
    pub const fn new(numerator: [T; M], denominator: [T; N]) -> Self {
        TransferFunction {
            numerator: Polynomial::new("s", numerator),
            denominator: Polynomial::new("s", denominator),
        }
    }
}

impl<T, M, N, S1, S2> FrequencyTools<T, 1, 1> for TransferFunction<T, M, N, S1, S2>
where
    T: Float + RealField + From<i16>,
    M: Dim,
    N: Dim,
    S1: RawStorage<T, M, U1>,
    S2: RawStorage<T, N, U1>,
{
    fn frequency_response<const L: usize>(&self, response: &mut FrequencyResponse<T, L, 1, 1>) {
        // Evaluate the transfer function at each frequency
        response.frequencies[0]
            .iter()
            .enumerate()
            .for_each(|(i, frequency)| {
                let s = Complex::new(T::zero(), *frequency); // s = jω
                response.responses[0][i] =
                    self.numerator.evaluate(s) / self.denominator.evaluate(s);
            })
    }
}

impl<T, M, N, S1, S2> fmt::Display for TransferFunction<T, M, N, S1, S2>
where
    T: Copy + Num + PartialOrd + Neg<Output = T> + fmt::Display,
    M: Dim,
    N: Dim,
    S1: RawStorage<T, M>,
    S2: RawStorage<T, N>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let num_str = format!("{}", self.numerator);
        let den_str = format!("{}", self.denominator);

        let (n_align, d_align, d_bar) = match den_str.len() > num_str.len() {
            true => (
                " ".repeat((den_str.len() - num_str.len()).max(0) / 2),
                "".to_string(),
                "-".repeat(den_str.len()),
            ),
            false => (
                "".to_string(),
                " ".repeat((num_str.len() - den_str.len()).max(0) / 2),
                "-".repeat(num_str.len()),
            ),
        };

        write!(
            f,
            "TransferFunction:\n{n_align}{num_str}\n{d_bar}\n{d_align}{den_str}\n"
        )
    }
}

#[cfg(test)]
mod basic_tf_tests {
    //! Basic test cases to make sure the TransferFunction is usable
    use super::*;

    #[test]
    fn initialize_integrator() {
        let tf = TransferFunction::new([1.0], [1.0, 0.0]);
        assert_eq!(
            tf.numerator,
            Polynomial::new("s", [1.0]),
            "TF numerator incorrect"
        );
        assert_eq!(
            tf.denominator,
            Polynomial::new("s", [1.0, 0.0]),
            "TF denominator incorrect"
        );
    }

    #[test]
    fn tf_as_monic() {
        let tf = TransferFunction::new([2.0], [2.0, 0.0]);
        let monic_tf = as_monic(&tf);
        assert_eq!(
            monic_tf.numerator.coefficients(),
            [1.0],
            "TF numerator incorrect"
        );
        assert_eq!(
            monic_tf.denominator.coefficients(),
            [1.0, 0.0],
            "TF denominator incorrect"
        );
    }

    #[test]
    fn monic_tf_as_monic() {
        let tf = TransferFunction::new([1.0, 1.0], [1.0, 0.0]);
        let monic_tf = as_monic(&tf);
        assert_eq!(
            monic_tf.numerator.coefficients(),
            [1.0, 1.0],
            "TF numerator incorrect"
        );
        assert_eq!(
            monic_tf.denominator.coefficients(),
            [1.0, 0.0],
            "TF denominator incorrect"
        );
    }

    #[test]
    fn test_lhp() {
        let tf = TransferFunction::new([1.0, 1.0], [1.0, 1.0]);
        assert!(lhp(&tf), "TF is not LHP stable");
    }
}
