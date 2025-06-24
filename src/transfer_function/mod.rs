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
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use nalgebra::{Complex, Const, DimAdd, DimMax, DimSub, RealField, U1};
use num_traits::{Float, One, Zero};

use crate::{
    frequency_tools::{FrequencyResponse, FrequencyTools},
    polynomial::utils::{
        add_generic, array_from_iterator_with_default, convolution, reverse_array, sub_generic,
    },
    systems::System,
};

// ===============================================================================================
//      TransferFunction Tests
// ===============================================================================================

#[cfg(test)]
mod tests;

// ===============================================================================================
//      TransferFunction Sub-modules
// ===============================================================================================

pub mod utils;
pub use utils::*;
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
        Self {
            numerator,
            denominator,
        }
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
        let mut responses = [Complex::zero(); L];
        // Evaluate the transfer function at each frequency
        response.frequencies[0]
            .iter()
            .enumerate()
            .for_each(|(i, frequency)| {
                // s = jω
                responses[i] = self.evaluate(&Complex::new(T::zero(), *frequency));
            });
        response.responses = Some([responses]);
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
            array_from_iterator_with_default([T::one()], T::zero()),
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

impl<T, const M: usize, const N: usize, const M2: usize> Add<T> for TransferFunction<T, M, N>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
    Const<M>: DimMax<Const<N>, Output = Const<M2>>,
{
    type Output = TransferFunction<T, M2, N>;
    fn add(self, rhs: T) -> Self::Output {
        let mut scaled_denom = self.denominator.clone();
        #[allow(clippy::suspicious_arithmetic_impl)]
        for a in &mut scaled_denom {
            *a = a.clone().mul(rhs.clone());
        }

        TransferFunction::from_data(add_generic(self.numerator, scaled_denom), self.denominator)
    }
}

impl<T, const M: usize, const N: usize> AddAssign<T> for TransferFunction<T, M, N>
where
    T: Clone + AddAssign + Mul<Output = T>,
    Const<M>: DimMax<Const<N>, Output = Const<M>>,
{
    fn add_assign(&mut self, rhs: T) {
        let mut scaled_denom = self.denominator.clone();
        #[allow(clippy::suspicious_arithmetic_impl)]
        for a in &mut scaled_denom {
            *a = a.clone().mul(rhs.clone());
        }
        for (numerator, denominator) in self.numerator.iter_mut().zip(scaled_denom.into_iter()) {
            numerator.add_assign(denominator);
        }
    }
}

impl<T, const M: usize, const N: usize, const L: usize> Sub<T> for TransferFunction<T, M, N>
where
    T: Clone + Sub<Output = T> + Mul<Output = T> + Zero,
    Const<M>: DimMax<Const<N>, Output = Const<L>>,
{
    type Output = TransferFunction<T, L, N>;
    fn sub(self, rhs: T) -> Self::Output {
        let mut scaled_denom = self.denominator.clone();
        #[allow(clippy::suspicious_arithmetic_impl)]
        for a in &mut scaled_denom {
            *a = a.clone().mul(rhs.clone());
        }

        TransferFunction::from_data(sub_generic(self.numerator, scaled_denom), self.denominator)
    }
}

impl<T, const M: usize, const N: usize> SubAssign<T> for TransferFunction<T, M, N>
where
    T: Clone + SubAssign + Mul<Output = T>,
    Const<M>: DimMax<Const<N>, Output = Const<M>>,
{
    fn sub_assign(&mut self, rhs: T) {
        let mut scaled_denom = self.denominator.clone();
        #[allow(clippy::suspicious_arithmetic_impl)]
        for a in &mut scaled_denom {
            *a = a.clone().mul(rhs.clone());
        }
        for (numerator, denominator) in self.numerator.iter_mut().zip(scaled_denom.into_iter()) {
            numerator.sub_assign(denominator);
        }
    }
}

impl<T, const M: usize, const N: usize> Mul<T> for TransferFunction<T, M, N>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        let mut product = self;
        for numerator in &mut product.numerator {
            *numerator = numerator.clone().mul(rhs.clone());
        }
        product
    }
}

impl<T, const M: usize, const N: usize> MulAssign<T> for TransferFunction<T, M, N>
where
    T: Clone + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        for numerator in &mut self.numerator {
            numerator.mul_assign(rhs.clone());
        }
    }
}
impl<T, const M: usize, const N: usize> Div<T> for TransferFunction<T, M, N>
where
    T: Clone + Mul<Output = T> + Zero,
{
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        let mut product = self;
        for denominator in &mut product.denominator {
            *denominator = denominator.clone().mul(rhs.clone());
        }
        product
    }
}

impl<T, const M: usize, const N: usize> DivAssign<T> for TransferFunction<T, M, N>
where
    T: Clone + MulAssign,
{
    fn div_assign(&mut self, rhs: T) {
        for denominator in &mut self.denominator {
            denominator.mul_assign(rhs.clone());
        }
    }
}

macro_rules! impl_left_scalar_ops {
    ($($scalar:ty),*) => {
        $(
            impl<const M: usize, const N: usize, const M2: usize> Add<TransferFunction<$scalar, M, N>> for $scalar
            where
                Const<M>: DimMax<Const<N>, Output = Const<M2>>,
            {
                type Output = TransferFunction<$scalar, M2, N>;
                #[inline(always)]
                fn add(self, rhs: TransferFunction<$scalar, M, N>) -> Self::Output {
                    rhs.add(self)
                }
            }
            impl<const M: usize, const N: usize, const M2: usize> Sub<TransferFunction<$scalar, M, N>> for $scalar
            where
                Const<N>: DimMax<Const<M>, Output = Const<M2>>,
            {
                type Output = TransferFunction<$scalar, M2, N>;
                #[inline(always)]
                fn sub(self, rhs: TransferFunction<$scalar, M, N>) -> Self::Output {
                    let mut scaled_denom = rhs.denominator.clone();
                    for a in &mut scaled_denom {
                        *a = a.clone().mul(self.clone());
                    }
                    TransferFunction::from_data(sub_generic(scaled_denom, rhs.numerator), rhs.denominator)
                }
            }
            impl<const M: usize, const N: usize> Mul<TransferFunction<$scalar, M, N>> for $scalar
            where
                Const<N>: DimSub<U1>,
            {
                type Output = TransferFunction<$scalar, M, N>;
                #[inline(always)]
                fn mul(self, rhs: TransferFunction<$scalar, M, N>) -> Self::Output {
                    rhs.mul(self)
                }
            }
            impl<const M: usize, const N: usize> Div<TransferFunction<$scalar, M, N>> for $scalar
            where
                Const<N>: DimSub<U1>,
            {
                type Output = TransferFunction<$scalar, N, M>;
                #[inline(always)]
                fn div(self, rhs: TransferFunction<$scalar, M, N>) -> Self::Output {
                    let mut scaled_denom = rhs.denominator.clone();
                    for a in &mut scaled_denom {
                        *a = a.clone().mul(self.clone());
                    }
                    TransferFunction::from_data(scaled_denom, rhs.numerator)
                }
            }
        )*
    };
}

impl_left_scalar_ops!(i8, u8, i16, u16, i32, u32, isize, usize, f32, f64);

// ===============================================================================================
//      TransferFunction Arithmetic
// ===============================================================================================

impl<
        T,
        const M1: usize,
        const N1: usize,
        const M2: usize,
        const N2: usize,
        const S1: usize,
        const S2: usize,
        const M3: usize,
        const N3: usize,
    > Add<TransferFunction<T, M2, N2>> for TransferFunction<T, M1, N1>
where
    T: Clone + AddAssign + Mul<Output = T> + Zero,
    // Bound for self.numerator * rhs.denominator
    Const<M1>: DimAdd<Const<N2>>,
    <Const<M1> as DimAdd<Const<N2>>>::Output: DimSub<U1, Output = Const<S1>>,
    // Bound for rhs.numerator * self.denominator
    Const<M2>: DimAdd<Const<N1>>,
    <Const<M2> as DimAdd<Const<N1>>>::Output: DimSub<U1, Output = Const<S2>>,
    // Bound for (self.numerator * rhs.denominator) + (rhs.numerator * self.denominator)
    Const<S1>: DimMax<Const<S2>, Output = Const<M3>>,
    // Bound for self.denominator * rhs.denominator
    Const<N1>: DimAdd<Const<N2>>,
    <Const<N1> as DimAdd<Const<N2>>>::Output: DimSub<U1, Output = Const<N3>>,
{
    type Output = TransferFunction<T, M3, N3>;
    fn add(self, rhs: TransferFunction<T, M2, N2>) -> Self::Output {
        TransferFunction::from_data(
            add_generic(
                convolution(&self.numerator, &rhs.denominator),
                convolution(&rhs.numerator, &self.denominator),
            ),
            convolution(&self.denominator, &rhs.denominator),
        )
    }
}

impl<
        T,
        const M1: usize,
        const N1: usize,
        const M2: usize,
        const N2: usize,
        const S1: usize,
        const S2: usize,
        const M3: usize,
        const N3: usize,
    > Sub<TransferFunction<T, M2, N2>> for TransferFunction<T, M1, N1>
where
    T: Clone + AddAssign + Sub<Output = T> + Mul<Output = T> + Zero,
    // Bound for self.numerator * rhs.denominator
    Const<M1>: DimAdd<Const<N2>>,
    <Const<M1> as DimAdd<Const<N2>>>::Output: DimSub<U1, Output = Const<S1>>,
    // Bound for rhs.numerator * self.denominator
    Const<M2>: DimAdd<Const<N1>>,
    <Const<M2> as DimAdd<Const<N1>>>::Output: DimSub<U1, Output = Const<S2>>,
    // Bound for (self.numerator * rhs.denominator) + (rhs.numerator * self.denominator)
    Const<S1>: DimMax<Const<S2>, Output = Const<M3>>,
    // Bound for self.denominator * rhs.denominator
    Const<N1>: DimAdd<Const<N2>>,
    <Const<N1> as DimAdd<Const<N2>>>::Output: DimSub<U1, Output = Const<N3>>,
{
    type Output = TransferFunction<T, M3, N3>;
    fn sub(self, rhs: TransferFunction<T, M2, N2>) -> Self::Output {
        TransferFunction::from_data(
            sub_generic(
                convolution(&self.numerator, &rhs.denominator),
                convolution(&rhs.numerator, &self.denominator),
            ),
            convolution(&self.denominator, &rhs.denominator),
        )
    }
}

impl<
        T,
        const M1: usize,
        const N1: usize,
        const M2: usize,
        const N2: usize,
        const M3: usize,
        const N3: usize,
    > Mul<TransferFunction<T, M2, N2>> for TransferFunction<T, M1, N1>
where
    T: Clone + AddAssign + Mul<Output = T> + Zero,
    // Bound for self.numerator * rhs.numerator
    Const<M1>: DimAdd<Const<M2>>,
    <Const<M1> as DimAdd<Const<M2>>>::Output: DimSub<U1, Output = Const<M3>>,
    // Bound for self.denominator * rhs.denominator
    Const<N1>: DimAdd<Const<N2>>,
    <Const<N1> as DimAdd<Const<N2>>>::Output: DimSub<U1, Output = Const<N3>>,
{
    type Output = TransferFunction<T, M3, N3>;
    fn mul(self, rhs: TransferFunction<T, M2, N2>) -> Self::Output {
        TransferFunction::from_data(
            convolution(&self.numerator, &rhs.numerator),
            convolution(&self.denominator, &rhs.denominator),
        )
    }
}

impl<
        T,
        const M1: usize,
        const N1: usize,
        const M2: usize,
        const N2: usize,
        const M3: usize,
        const N3: usize,
    > Div<TransferFunction<T, M2, N2>> for TransferFunction<T, M1, N1>
where
    T: Clone + AddAssign + Mul<Output = T> + Zero,
    // Bound for self.numerator * rhs.denominator
    Const<M1>: DimAdd<Const<N2>>,
    <Const<M1> as DimAdd<Const<N2>>>::Output: DimSub<U1, Output = Const<M3>>,
    // Bound for self.denominator * rhs.numerator
    Const<N1>: DimAdd<Const<M2>>,
    <Const<N1> as DimAdd<Const<M2>>>::Output: DimSub<U1, Output = Const<N3>>,
{
    type Output = TransferFunction<T, M3, N3>;
    fn div(self, rhs: TransferFunction<T, M2, N2>) -> Self::Output {
        TransferFunction::from_data(
            convolution(&self.numerator, &rhs.denominator),
            convolution(&self.denominator, &rhs.numerator),
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

fn formatted_length<T: fmt::Display>(value: &T, f: &fmt::Formatter<'_>) -> Result<usize, fmt::Error> {
    use fmt::Write;
    let mut counter = FmtLengthCounter { length: 0 };
    if let Some(precision) = f.precision() {
        write!(&mut counter, "{value:precision$}")?;
    }
    else {
        write!(&mut counter, "{value}")?;
    }

    Ok(counter.length)
}

/// TODO: Fix formating
impl<T, const M: usize, const N: usize> fmt::Display for TransferFunction<T, M, N>
where
    T: Copy + Zero + One + Neg<Output = T> + PartialOrd + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let num = crate::Polynomial::from_data(self.numerator);
        let den = crate::Polynomial::from_data(self.denominator);
        let num_len = formatted_length(&num, f)?;
        let den_len = formatted_length(&den, f)?;

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
        if let Some(precision) = f.precision() {
            writeln!(f, "{num:.precision$}")?;
        }
        else {
            writeln!(f, "{num}")?;
        }

        // Write division bar
        for _ in 0..bar_len {
            write!(f, "-")?;
        }
        writeln!(f, " ")?;

        // Write denominator with padding
        for _ in 0..d_align {
            write!(f, " ")?;
        }
        if let Some(precision) = f.precision() {
            writeln!(f, "{den:.precision$}")?;
        }
        else {
            writeln!(f, "{den}")?;
        }


        Ok(())
    }
}
