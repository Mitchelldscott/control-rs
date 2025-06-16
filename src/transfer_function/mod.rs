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
use nalgebra::{Complex, Const, DimMax, DimSub, RealField, U1};
use num_traits::{Float, One, Zero};

use crate::{
    polynomial::utils::{reverse_array, array_from_iterator_with_default, add_generic},
    frequency_tools::{FrequencyResponse, FrequencyTools}
};

// ===============================================================================================
//      TransferFunction Tests
// ===============================================================================================

#[cfg(test)]
mod basic_tf_tests;

#[cfg(test)]
mod edge_case_tests;

#[cfg(test)]
mod tf_frequency_tests;

#[cfg(test)]
mod tf_arithmatic_tests;

// ===============================================================================================
//      TransferFunction Sub-modules
// ===============================================================================================

pub mod linear_tools;

pub use linear_tools::*;
use crate::systems::System;
// ===============================================================================================
//      TransferFunction
// ===============================================================================================

/// # Transfer Function
///
/// <pre>
/// G(s) = b(s) / a(s)
/// a(s) = (a_n * s^n + a_1 * s^(n-1) + ... + a_0)
/// b(s) = (b_m * s^m + b_1 * s^(m-1) + ... + b_0)
/// </pre>
///
/// Stores the coefficients of two polynomials, one for the numerator and one for the denominator.
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
    /// Create a new transfer function from arrays of coefficients in degree-minor order
    ///
    /// # Arguments
    /// * `numerator` - coefficients of the numerator `[b_0, b_1, ... b_m]`
    /// * `denominator` - coefficients of the denominator `[a_0, a_1, ... a_n]`
    ///
    /// # Returns
    /// * `TransferFunction` - static Transfer Function
    ///
    /// # Example
    /// ```
    /// use control_rs::TransferFunction;
    /// // 1 / s^2
    /// let tf = TransferFunction::new([1.0], [0.0, 0.0, 1.0]);
    /// println!("{tf}");
    /// ```
    pub const fn from_data(numerator: [T; M], denominator: [T; N]) -> Self {
        Self { numerator, denominator }
    }
}

impl<T: Copy, const M: usize, const N: usize> TransferFunction<T, M, N> {
    /// Create a new transfer function from arrays of coefficients in degree-major order
    ///
    /// # Arguments
    /// * `numerator` - coefficients of the numerator `[b_m, ... b_1, b_0]`
    /// * `denominator` - coefficients of the denominator `[a_n, ... a_1, a_0]`
    ///
    /// # Returns
    /// * `TransferFunction` - static Transfer Function
    ///
    /// # Example
    /// ```
    /// use control_rs::TransferFunction;
    /// // s + 1 / s^2 + s + 1
    /// let tf = TransferFunction::new([1.0, 1.0], [1.0, 1.0, 1.0]);
    /// println!("{tf}");
    /// ```
    /// TODO: Unit Test
    pub const fn new(numerator: [T; M], denominator: [T; N]) -> Self {
        Self {
            numerator: reverse_array(numerator),
            denominator: reverse_array(denominator),
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
            .rfold(U::zero(), |acc, a_i| acc * value.clone() + a_i.clone())
            / self
                .denominator
                .iter()
                .rfold(U::zero(), |acc, a_i| acc * value.clone() + a_i.clone())
    }
}

// ===============================================================================================
//      Polynomial System traits
// ===============================================================================================

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

impl<T, const N: usize, const M: usize> System for TransferFunction<T, M, N>
where
    T: Copy + Clone + Zero + One,
    Const<N>: DimSub<U1>,
    Const<M>: DimSub<U1>,
{
    fn zero() -> Self {
        Self::new([T::zero(); M], [T::zero(); N])
    }

    fn identity() -> Self {
        Self::from_data(
            array_from_iterator_with_default([T::one()], T::zero()),
            array_from_iterator_with_default([T::one()], T::zero())
        )
    }
}

// ===============================================================================================
//      TransferFunction-Scalar Arithmetic
// ===============================================================================================

impl<T: Clone + Neg<Output = T>, const M: usize, const N: usize> Neg for TransferFunction<T, M, N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut neg_self = self;
        for b in &mut neg_self.numerator {
            *b = b.clone().neg();
        }
        for a in &mut neg_self.denominator {
            *a = a.clone().neg();
        }
        neg_self
    }
}

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Zero,  const M: usize, const N: usize, const L: usize> Add<T> for TransferFunction<T, M, N>
where
    Const<N>: DimMax<Const<M>, Output = Const<L>>,
{
    type Output = TransferFunction<T, L, N>;
    fn add(self, rhs: T) -> Self::Output {
        let mut scaled_denom = self.denominator.clone();
        #[allow(clippy::suspicious_arithmetic_impl)]
        for a in &mut scaled_denom {
            *a = a.clone() * rhs.clone();
        }

        TransferFunction::from_data(
            add_generic(&scaled_denom, &self.numerator),
            self.denominator
        )

    }
}

// ===============================================================================================
//      TransferFunction Formatters
// ===============================================================================================

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
