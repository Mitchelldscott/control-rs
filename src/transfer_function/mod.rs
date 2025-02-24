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

// #[cfg(feature = "std")]
// use std::fmt;

// #[cfg(not(feature = "std"))]
// use core::fmt;

use nalgebra::{ArrayStorage, Complex, Dim, RawStorage, RealField};
use num_traits::Float;

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
/// * `N` - order of the denominator
/// * `M` - order of the numerator
///
/// ## References
///
/// - *Feedback Control of Dynamic Systems*, Franklin et al., Ch. 3.1
pub struct TransferFunction<T, D1: Dim, D2: Dim, S1, S2> {
    /// coefficients of the numerator `[bn, ... b2, b1]`
    pub numerator: Polynomial<T, D1, S1>,
    /// coefficients of the denominator `[an, ... a1, a0]`
    pub denominator: Polynomial<T, D2, S2>,
}

impl<T, D1: Dim, D2: Dim, const N: usize, const M: usize>
    TransferFunction<T, D1, D2, ArrayStorage<T, M, 1>, ArrayStorage<T, N, 1>>
where
    T: 'static + Copy + PartialEq,
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
            numerator: Polynomial::new(numerator),
            denominator: Polynomial::new(denominator),
        }
    }
}

impl<T, D1: Dim, D2: Dim, S1, S2> FrequencyTools<T, 1, 1> for TransferFunction<T, D1, D2, S1, S2>
where
    T: Float + RealField + From<i16>,
    S1: RawStorage<T, D1>,
    S2: RawStorage<T, D2>,
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

// impl<T, const N1: usize, const M1: usize, const N2: usize, const M2: usize>
//     Mul for TransferFunction<T, N1, M1>
// where
//     T: Scalar + Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
// {
//     type Output = TransferFunction<T, { N1 + N2 - 1 }, { M1 + M2 - 1 }>;

//     fn mul(self, rhs: Self) -> Self::Output {
//         let mut new_numerator = SMatrix::<T, { M1 + M2 - 1 }, 1>::zeros();
//         let mut new_denominator = SMatrix::<T, { N1 + N2 - 1 }, 1>::zeros();

//         // Polynomial multiplication for numerator
//         for i in 0..M1 {
//             for j in 0..M2 {
//                 new_numerator[(i + j, 0)] += self.numerator[(i, 0)] * rhs.numerator[(j, 0)];
//             }
//         }

//         // Polynomial multiplication for denominator
//         for i in 0..N1 {
//             for j in 0..N2 {
//                 new_denominator[(i + j, 0)] += self.denominator[(i, 0)] * rhs.denominator[(j, 0)];
//             }
//         }

//         TransferFunction {
//             numerator: new_numerator,
//             denominator: new_denominator,
//         }
//     }
// }

// #[cfg(feature = "std")]
// impl<T, const N: usize, const M: usize> fmt::Display for TransferFunction<T, N, M>
// where
//     T: 'static + Copy + PartialEq + Zero + fmt::Debug + fmt::Display,
// {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         fn polynomial_term<T: fmt::Display + Zero>(
//             coeff: T,
//             var: &str,
//             order: usize,
//         ) -> Option<String> {
//             match coeff.is_zero() {
//                 true => None,
//                 _ => match order {
//                     0 => Some(format!("{coeff:.4}")),
//                     1 => Some(format!("{coeff:.4}{var}")),
//                     _ => Some(format!("{coeff:.4}{var}^{order}")),
//                 },
//             }
//         }
//         let num_str = (0..self.numerator.len())
//             .filter_map(|i| polynomial_term(self.numerator[i], "s", M - i - 1))
//             .collect::<Vec<_>>()
//             .join(" + ");
//         let den_str = (0..self.denominator.len())
//             .filter_map(|i| polynomial_term(self.denominator[i], "s", N - i - 1))
//             .collect::<Vec<_>>()
//             .join(" + ");

//         let (n_align, d_align, d_bar) = match den_str.len() > num_str.len() {
//             true => (
//                 " ".repeat((den_str.len() - num_str.len()).max(0) / 2),
//                 "".to_string(),
//                 "-".repeat(den_str.len()),
//             ),
//             false => (
//                 "".to_string(),
//                 " ".repeat((num_str.len() - den_str.len()).max(0) / 2),
//                 "-".repeat(num_str.len()),
//             ),
//         };

//         write!(
//             f,
//             "TransferFunction:\n{n_align}{num_str}\n{d_bar}\n{d_align}{den_str}\n"
//         )
//     }
// }

// #[cfg(not(feature = "std"))]
// impl<T, const N: usize, const M: usize> fmt::Display for TransferFunction<T, N, M>
// where
//     T: 'static + Copy + PartialEq + Zero + fmt::Debug + fmt::Display,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         // Format the numerator coefficients
//         write!(f, "Numerator: [")?;
//         for i in 0..M {
//             if i > 0 {
//                 write!(f, ", ")?; // Add a comma separator
//             }
//             write!(f, "{:.4}", self.numerator[(i, 0)])?;
//         }
//         write!(f, "]")?;

//         // Format the denominator coefficients
//         write!(f, " | Denominator: [")?;
//         for i in 0..N {
//             if i > 0 {
//                 write!(f, ", ")?; // Add a comma separator
//             }
//             write!(f, "{:.4}", self.denominator[(i, 0)])?;
//         }
//         write!(f, "]")?;

//         Ok(())
//     }
// }

#[cfg(test)]
mod basic_tf_tests {
    //! Basic test cases to make sure the TransferFunction is usable
    use nalgebra::{U1, U2};

    use super::*;

    #[test]
    fn initialize_integrator() {
        let tf: TransferFunction<f64, U1, U2, ArrayStorage<f64, 1, 1>, ArrayStorage<f64, 2, 1>> =
            TransferFunction::new([1.0], [1.0, 0.0]);
        assert_eq!(
            tf.numerator.coefficients,
            ArrayStorage([[1.0]]),
            "TF numerator incorrect"
        );
        assert_eq!(
            tf.denominator.coefficients,
            ArrayStorage([[1.0, 0.0]]),
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

#[cfg(test)]
pub mod tf_edge_case_test;
