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

use core::{
    fmt,
    ops::{Add, Div, Mul, Neg},
};
use nalgebra::{Complex, RealField};
use num_traits::{Float, One, Zero};

use crate::frequency_tools::{FrequencyResponse, FrequencyTools};

// ===============================================================================================
//      TransferFunction Tests
// ===============================================================================================

#[cfg(test)]
mod edge_case_test;

#[cfg(test)]
mod tf_frequency_tests;

// ===============================================================================================
//      TransferFunction Sub-modules
// ===============================================================================================

pub mod linear_tools;
pub use linear_tools::*;

// ===============================================================================================
//      TransferFunction Tests
// ===============================================================================================

/// # Transfer Function
///
/// <pre>
/// G(s) = b(s) / a(s)
/// a(s) = (a_0 * s^(N-1) + a_1 * s^(N-2) + ... + a_(N-1))
/// b(s) = (b_0 * s^(M-1) + b_1 * s^(M-2) + ... + b_(M-1))
/// </pre>
///
/// Stores two polynomials, one for the numerator and one for the denominator.
///
/// # Generic Arguments
/// * `T` - type of the coefficients
/// * `M` - number of coefficients in the numerator
/// * `N` - number of coefficients in the denominator
///
/// ## References
/// - *Feedback Control of Dynamic Systems*, Franklin et al., Ch. 3.1
///
/// TODO: Example + Integration Test
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TransferFunction<T, const M: usize, const N: usize> {
    /// coefficients of the numerator `[b_0, b_1, ... b_m]`
    pub numerator: [T; M],
    /// coefficients of the denominator `[a_0, a_1, ... a_n]`
    pub denominator: [T; N],
}

impl<T, const M: usize, const N: usize> TransferFunction<T, M, N> {
    /// Create a new transfer function from arrays of coefficients
    ///
    /// # Arguments
    /// * `numerator` - coefficients of the numerator `[b_m, ... b_1, b_0]`
    /// * `denominator` - coefficients of the denominator `[a_n, ... a_1, a_0]`
    ///
    /// # Returns
    /// * `TransferFunction` - static Transfer Function
    ///
    /// # Example
    ///
    /// ```
    /// use control_rs::TransferFunction;
    /// let tf = TransferFunction::new([1.0, 1.0], [1.0, 1.0, 1.0]);
    /// println!("{tf}");
    /// ```
    /// TODO: Unit Test
    pub const fn new(numerator: [T; M], denominator: [T; N]) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}
impl<T: Clone, const N: usize, const M: usize> TransferFunction<T, M, N> {
    /// TODO: Doc + Unit Test + Example
    pub fn evaluate<U>(&self, value: &U) -> U
    where
        U: Clone + Zero + Add<T, Output = U> + Mul<U, Output = U> + Div<Output = U>,
    {
        self.numerator
            .iter()
            .fold(U::zero(), |acc, a_i| acc * value.clone() + a_i.clone())
            / self
                .denominator
                .iter()
                .fold(U::zero(), |acc, a_i| acc * value.clone() + a_i.clone())
    }
}

impl<T: Float + RealField, const M: usize, const N: usize> FrequencyTools<T, 1, 1>
    for TransferFunction<T, M, N>
{
    /// TODO: Doc + Unit Test + Example
    fn frequency_response<const L: usize>(&self, response: &mut FrequencyResponse<T, L, 1, 1>) {
        // Evaluate the transfer function at each frequency
        response.frequencies[0]
            .iter()
            .enumerate()
            .for_each(|(i, frequency)| {
                // s = jω
                response.responses[0][i] = self.evaluate(&Complex::new(T::zero(), *frequency));
            });
    }
}
struct FmtLengthCounter {
    length: usize,
}

impl fmt::Write for FmtLengthCounter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.length += s.len();
        Ok(())
    }
}

fn formatted_length<T: fmt::Display>(value: &T) -> usize {
    use fmt::Write;
    let mut counter = FmtLengthCounter { length: 0 };
    write!(&mut counter, "{value}").unwrap();
    counter.length
}

/// TODO: Fix formating
impl<T, const M: usize, const N: usize> fmt::Display for TransferFunction<T, M, N>
where
    T: Copy + Zero + One + Neg<Output = T> + PartialOrd + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let num_len = formatted_length(&crate::Polynomial::new(self.numerator));
        let den_len = formatted_length(&crate::Polynomial::new(self.denominator));

        let (n_align, d_align, bar_len) = if den_len > num_len {
            ((den_len - num_len) / 2, 0, den_len)
        } else {
            (0, (num_len - den_len) / 2, num_len)
        };

        writeln!(f, "Transfer Function:")?;

        // Write numerator with padding
        for _ in 0..n_align {
            write!(f, " ")?;
        }
        writeln!(f, "{}", crate::Polynomial::new(self.numerator))?;

        // Write division bar
        for _ in 0..bar_len {
            write!(f, "-")?;
        }
        writeln!(f, " ")?;

        // Write denominator with padding
        for _ in 0..d_align {
            write!(f, " ")?;
        }
        writeln!(f, "{}", crate::Polynomial::new(self.denominator))?;

        Ok(())
    }
}

#[cfg(test)]
mod basic_tf_tests {
    //! Basic test cases to make sure the Transfer Function is usable
    use crate::{assert_f64_eq, transfer_function::*};

    #[test]
    fn initialize_integrator() {
        let tf = TransferFunction::new([1.0], [1.0, 0.0]);
        assert_f64_eq!(tf.numerator[0], 1.0);
        assert_f64_eq!(tf.denominator[0], 1.0);
        assert_f64_eq!(tf.denominator[1], 0.0);
    }

    #[test]
    fn tf_as_monic() {
        let tf = TransferFunction::new([2.0], [2.0, 0.0]);
        let monic_tf = as_monic(&tf);
        assert_f64_eq!(monic_tf.numerator[0], 1.0);
        assert_f64_eq!(monic_tf.denominator[0], 1.0);
        assert_f64_eq!(monic_tf.denominator[1], 0.0);
    }

    #[test]
    fn monic_tf_as_monic() {
        let tf = TransferFunction::new([1.0, 1.0], [1.0, 0.0]);
        let monic_tf = as_monic(&tf);
        assert_f64_eq!(monic_tf.numerator[0], 1.0);
        assert_f64_eq!(monic_tf.numerator[1], 1.0);
        assert_f64_eq!(monic_tf.denominator[0], 1.0);
        assert_f64_eq!(monic_tf.denominator[1], 0.0);
    }

    #[test]
    fn test_lhp() {
        let tf = TransferFunction::new([1.0, 1.0], [1.0, 1.0]);
        assert!(lhp(&tf), "TF is not LHP stable");
    }
}
