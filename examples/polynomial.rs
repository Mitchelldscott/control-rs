
use core::ops::{Add, AddAssign, Sub, SubAssign, Mul, Div};
use num_traits::Zero;
use nalgebra::{Const, DimMax, DimAdd, DimSub, U1};

/// Statically sized univariate polynomial
///
/// This struct stores the coefficients of a polynomial a(x):
/// <pre>
/// a(x) = a_n * x^n + a_(n-1) * x^(n-1) + ... + a_1 * x + a_0
/// </pre>
/// where `n` is the degree of the polynomial.
///
/// The coefficients are stored in degree-minor order: `[a_0, a_1 ... a_n]` (index 0 is the lowest degree
/// or constant term, index n is the highest degree term).
///
/// # Generic Arguments
/// * `T` - Type of the coefficients in the polynomial.
/// * `N` - Capacity of the underlying array.
///
/// # Example
/// ```rust
/// // creates a quadratic monomial, y = x^2
/// let quadratic = control_rs::Polynomial::from_data([0, 0, 1]);
/// ```
/// # TODO: Demo + Integration Test
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Polynomial<T, const N: usize> {
    /// coefficients of the polynomial
    coefficients: [T; N],
}

impl<T: Copy + Zero, const N: usize> Polynomial<T, N> {
    pub fn monomial(coefficient: T, degree: usize) -> Self {
        let mut coefficients = [T::zero(); N];
        if degree < N {
            // SAFETY: `degree < N` is a valid index of [T; N]
            // unsafe { *coefficients.get_unchecked_mut(degree) = coefficient; }
            coefficients[degree] = coefficient;
        }
        Self { coefficients }
    }
}

impl<T: Zero, const N: usize> Polynomial<T, N> {

    pub fn degree(&self) -> Option<usize> {
        for i in (0..N).rev() {
            // SAFETY: `i < N` is a valid index of [T; N]
            // unsafe {
            //     if !self.coefficients.get_unchecked(i).is_zero() {
            //         return Some(i);
            //     }
            // }
            if !self.coefficients[i].is_zero() {
                return Some(i);
            }
        }
        None
    }
}

impl<T, const N: usize, const M: usize, const L: usize> Add<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Copy + Zero + AddAssign,
    Const<N>: DimMax<Const<M>, Output = Const<L>>,
{
    type Output = Polynomial<T, L>;

    /// Adds two polynomials
    ///
    /// The result is a polynomial with capacity `max(N, M)`. This function utilizes the 
    /// [Polynomial::addassign_polynomial_with_smaller_capacity] method to add the polynomials.
    /// This implementation avoids the need for a zero-filled or [MaybeUninit] array 
    /// initialization. This should also allow for compile time branch optimizations because 
    /// the compiler can determine which branch to take at compile time (not sure if this is 
    /// true).
    ///
    /// # Generic Arguments
    /// * `M` - Degree of the rhs polynomial.
    fn add(self, rhs: Polynomial<T, M>) -> Self::Output {
        let mut result = Polynomial { coefficients: [T::zero(); L] };
        for (i, c) in result.coefficients.iter_mut().enumerate() {
            // SAFETY: `i < N` is a valid index of [T; N]
            if i < N { unsafe { *c = self.coefficients.get_unchecked(i).clone() } }
            // SAFETY: `i < M` is a valid index of [T; M]
            if i < M { unsafe { *c += rhs.coefficients.get_unchecked(i).clone() } }
        }
        result
    }
}

impl<T, const N: usize> AddAssign<Polynomial<T, N>> for Polynomial<T, N>
where
    T: Copy + Zero + AddAssign,
{

    /// Adds two polynomials
    fn add_assign(&mut self, rhs: Polynomial<T, N>) {
        for (a, b) in self.coefficients.iter_mut().zip(rhs.coefficients.iter()) {
            *a += *b;
        }
    }
}

impl<T, const N: usize, const M: usize, const L: usize> Sub<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Copy + Zero + SubAssign,
    Const<N>: DimMax<Const<M>, Output = Const<L>>,
{
    type Output = Polynomial<T, L>;

    /// Subtracts two polynomials
    ///
    /// The result is a polynomial with capacity `max(N, M)`.
    ///
    /// # Generic Arguments
    /// * `M` - Degree of the rhs polynomial.
    fn sub(self, rhs: Polynomial<T, M>) -> Self::Output {
        let mut result = Polynomial { coefficients: [T::zero(); L] };
        for (i, c) in result.coefficients.iter_mut().enumerate() {
            // SAFETY: `i < N` is a valid index of [T; N]
            if i < N { unsafe { *c = self.coefficients.get_unchecked(i).clone() } }
            // SAFETY: `i < M` is a valid index of [T; M]
            if i < M { unsafe { *c -= rhs.coefficients.get_unchecked(i).clone() } }
        }
        result
    }
}

impl<T, const N: usize, const M: usize, const L: usize> Mul<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Copy + Zero + AddAssign + Mul<Output = T>,
    Const<N>: DimAdd<Const<M>>,
    <Const<N> as DimAdd<Const<M>>>::Output: DimSub<U1, Output = Const<L>>,
{
    type Output = Polynomial<T, L>;
    /// Multiplies two polynomials
    ///
    /// The result is a polynomial with capacity `N + M - 1`. This may be larger than the degree of the
    /// result polynomial, in which case the higher order coefficients are set to zero.
    ///
    /// # Generic Arguments
    /// * `M` - Degree of the rhs polynomial.
    /// * `L` - Degree of the result polynomial.
    ///
    /// # Example
    /// ```rust
    /// let p1 = Polynomial { coefficients: [1i32; 2] };
    /// let p2 = Polynomial { coefficients: [1i32; 2] };
    /// let p3 = p1 * p2;
    /// assert_eq!(p3.coefficients, [1i32, 1i32, 1i32], "wrong multiplication result");
    /// ```
    fn mul(self, rhs: Polynomial<T, M>) -> Self::Output {
        let mut result = Polynomial { coefficients: [T::zero(); L] };
        for i in 0..N {
            for j in 0..M {
                // SAFETY: `i + j = (N - 1) + (M - 1) < L` is a valid index of [T; L]
                // SAFETY: `i < N` is a valid index of [T; N]
                // SAFETY: `j < M` is a valid index of [T; M]
                // unsafe {
                //     *result.coefficients.get_unchecked_mut(i + j) += self.coefficients.get_unchecked(i).clone()
                //         * rhs.coefficients.get_unchecked(j).clone();
                // }
                result.coefficients[i + j] += self.coefficients[i] * rhs.coefficients[j];
            }
        }
        result
    }
}

/// Divides two polynomials
///
/// The result is a polynomial with capacity `N`.
///
/// # Generic Arguments
/// * `N` - Degree of the rhs polynomial.
///
/// # Example
/// ```rust
/// let p1 = Polynomial { coefficients: [1i32; 2] };
/// let p2 = Polynomial { coefficients: [1i32; 2] };
/// let p3 = p1 / p2;
/// assert_eq!(p3.coefficients, [1i32, 1i32], "wrong division result");
/// ```
///
/// # Algorithm
/// <pre>
/// function n / d is
/// while r ≠ 0 and degree(r) ≥ degree(d) do
///     t ← lead(r) / lead(d)       // Divide the leading terms
///     q ← q + t
///     r ← r − t × d
/// return (q, r)
///</pre>
impl<T, const N: usize> Div<Polynomial<T, N>> for Polynomial<T, N> 
where 
    T: Copy + Zero + Div<Output = T> + Mul<Output = T> + AddAssign + SubAssign + core::fmt::Debug,
    Const<N>: DimMax<Const<N>>,
{
    type Output = Polynomial<T, N>;
    fn div(self, rhs: Polynomial<T, N>) -> Self::Output {
        let mut quotient = Polynomial { coefficients: [T::zero(); N] };

        // Find actual degrees
        let deg_self = self.degree();
        let deg_rhs = rhs.degree();

        // degree of self and rhs exists
        if let Some(deg_self) = deg_self {
            if let Some(deg_rhs) = deg_rhs {
                let mut remainder = self.coefficients;
                let leading_divisor = rhs.coefficients[deg_rhs];
                
                for i in (deg_rhs..=deg_self).rev() {
                    if remainder[i].is_zero() {
                        continue;
                    }

                    let q_index = i - deg_rhs;
                    let leading_remainder = remainder[i];
                    let term_divisor = leading_remainder / leading_divisor;
                    quotient.coefficients[q_index] += term_divisor;
                    for j in 0..deg_rhs {
                        remainder[q_index + j] -= term_divisor * rhs.coefficients[j];
                    }
                }
            }
        }

        quotient
    }
}

fn main()  {
    let p1 = Polynomial { coefficients: [1i32; 0] };
    let p2 = Polynomial { coefficients: [0i32; 1] };

    assert_eq!((p1 + p2).coefficients, [0i32; 1], "wrong addition result");

    let p3 = Polynomial { coefficients: [1i32; 2] };
    let p4 = Polynomial { coefficients: [1i32; 1] };

    assert_eq!((p3 - p4).coefficients, [0i32, 1i32], "wrong subtraction result");

    let p5 = Polynomial { coefficients: [1i32; 2] };
    let p6 = Polynomial { coefficients: [1i32; 2] };

    assert_eq!((p5 * p6).coefficients, [1i32, 2i32, 1i32], "wrong multiplication result");

    let p7 = Polynomial { coefficients: [-4i32, 0i32, -2i32, 1i32] };
    let p8 = Polynomial { coefficients: [-3i32, 1i32, 0i32, 0i32] };

    assert_eq!((p7 / p8).coefficients, [3i32, 1i32, 1i32, 0i32], "wrong division result");
}