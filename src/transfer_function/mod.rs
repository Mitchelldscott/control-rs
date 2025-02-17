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


use std::fmt::Display;

use nalgebra::{ArrayStorage, Complex, RealField, SMatrix};
use num_traits::{Float, Zero};

pub mod linear_tools;
pub use linear_tools::*;

pub mod tf_edge_case_test;

use crate::frequency_tools::{FrequencyResponse, FrequencyTools};

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
/// * `N` - order of the denominator
/// * `M` - order of the numerator
///
/// ## References
///
/// - *Feedback Control of Dynamic Systems*, Franklin et al., Ch. 3.1
pub struct TransferFunction<T, const N: usize, const M: usize> {
    /// coefficients of the numerator `[bn, ... b2, b1]`
    pub numerator: SMatrix<T, M, 1>,
    /// coefficients of the denominator `[an, ... a1, a0]`
    pub denominator: SMatrix<T, N, 1>,
}

impl<T, const N: usize, const M: usize> TransferFunction<T, N, M>
where
    T: 'static + Copy + PartialEq + std::fmt::Debug,
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
            numerator: SMatrix::from_array_storage(ArrayStorage([numerator])),
            denominator: SMatrix::from_array_storage(ArrayStorage([denominator])),
        }
    }
}

impl<T, const N: usize, const M: usize> FrequencyTools<T, 1, 1> for TransferFunction<T, N, M>
where
    T: Float + RealField + From<i16>,
{
    fn frequency_response<const L: usize>(&self, response: &mut FrequencyResponse<T, L, 1, 1>) {
        // Evaluate the transfer function at each frequency
        response.frequencies[0]
            .iter()
            .enumerate()
            .for_each(|(i, frequency)| {
                let s = Complex::new(T::zero(), *frequency); // s = jω
                let numerator: Complex<T> = self
                    .numerator
                    .iter()
                    .rev()
                    .fold(Complex::zero(), |acc, &coeff| acc * s + coeff);
                let denominator: Complex<T> = self
                    .denominator
                    .iter()
                    .rev()
                    .fold(Complex::zero(), |acc, &coeff| acc * s + coeff);
                response.responses[0][i] = numerator / denominator;
            })
    }
}

impl<T, const N: usize, const M: usize> std::fmt::Display for TransferFunction<T, N, M>
where
    T: 'static + Copy + PartialEq + Zero + std::fmt::Debug + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        fn polynomial_term<T: Display + Zero>(coeff: T, var: &str, order: usize) -> Option<String> {
            match coeff.is_zero() {
                true => None,
                _ => {
                    match order {
                        0 => Some(format!("{coeff}")),
                        1 => Some(format!("{coeff}{var}")),
                        _ => Some(format!("{coeff}{var}^{order}")),
                    }
                }
            }
        }
        let num_str = (0..self.numerator.len())
            .filter_map(|i| polynomial_term(self.numerator[i], "s", M - i - 1))
            .collect::<Vec<_>>()
            .join(" + ");
        let den_str = (0..self.denominator.len())
            .filter_map(|i| polynomial_term(self.denominator[i], "s", N - i - 1))
            .collect::<Vec<_>>()
            .join(" + ");

        let (n_align, d_align, d_bar) = match den_str.len() > num_str.len() {
            true => (" ".repeat((den_str.len() - num_str.len()).max(0) / 2),"".to_string(), "-".repeat(den_str.len())),
            false => ("".to_string(), " ".repeat((num_str.len() - den_str.len()).max(0) / 2), "-".repeat(num_str.len())),
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
            nalgebra::Matrix1::new(1.0),
            "TF numerator incorrect"
        );
        assert_eq!(
            tf.denominator,
            nalgebra::Matrix2x1::new(1.0, 0.0),
            "TF denominator incorrect"
        );
    }

    #[test]
    fn tf_as_monic() {
        let tf = TransferFunction::new([2.0], [2.0, 0.0]);
        let (num, den) = as_monic(&tf);
        assert_eq!(num, [1.0], "TF numerator incorrect");
        assert_eq!(den, [1.0, 0.0], "TF denominator incorrect");
    }

    #[test]
    fn monic_tf_as_monic() {
        let tf = TransferFunction::new([1.0, 1.0], [1.0, 0.0]);
        let (num, den) = as_monic(&tf);
        assert_eq!(num, [1.0, 1.0], "TF numerator incorrect");
        assert_eq!(den, [1.0, 0.0], "TF denominator incorrect");
    }

    #[test]
    fn test_lhp() {
        let tf = TransferFunction::new([1.0, 1.0], [1.0, 1.0]);
        assert!(lhp(&tf), "TF is not LHP stable");
    }
}
