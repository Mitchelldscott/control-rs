//! Array-based univariate-polynomial
//!
//! This module contains a base implementation of a generic array polynomial. To guarantee safe and
//! deterministic implementations of all arithmetic, some implementations are only available for
//! certain specializations.
//!
//! # Examples
//!
//! ```rust
//! use control_rs::polynomial::{Polynomial, Constant, Line};
//!
//! let one = Constant::new([1.0]);
//! assert_eq!(one.degree(), Some(0));
//! assert_eq!(one.leading_coefficient(), Some(&1.0));
//!
//! let line = Line::new([1.0, 0.0]);
//! assert_eq!(line.degree(), Some(1));
//! assert_eq!(line.leading_coefficient(), Some(&1.0));
//! ```
//!
// TODO:
//  * calculus
//      * `compose(f: Polynomial, g: Polynomial) -> Polynomial`
//      * `from_roots(&mut self, roots: &[T]) -> Result<(), Polynomial>`
//      * `from_complex_roots(&mut self, roots: &[Complex<T>]) -> Result<(), Polynomial>`
//      * `roots(p: Polynomial, roots: &mut [T]) -> Result<(), PolynomialError>`
//      * `complex_roots(p: Polynomial, roots: &mut [Complex<T>]) -> Result<(), PolynomialError>`
//  * formatting
//      * Display precision option
//      * Latex / symbolic formatter
//  * Move internal function to separate files (keep private) so other mods can use them without polynomial
//      * roots
//      * derivative/integral
//      * from_array_initializers

use core::{
    array, fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
    slice,
};
use nalgebra::{
    allocator::Allocator, Complex, Const, DefaultAllocator, DimAdd, DimDiff, DimMax, DimSub,
    RealField, U1,
};
use num_traits::{Num, One, Zero};

// ===============================================================================================
//      Polynomial Submodules
// ===============================================================================================

pub mod utils;

// ===============================================================================================
//      Polynomial Specializations
// ===============================================================================================

mod aliases;
use crate::systems::System;
pub use aliases::{Constant, Cubic, Line, Quadratic, Quartic, Quintic};
// ===============================================================================================
//      Polynomial Tests
// ===============================================================================================

#[cfg(test)]
mod basic_polynomial_tests;

#[cfg(test)]
mod arithmetic_tests;
// ===============================================================================================
//      Polynomial
// ===============================================================================================

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

impl<T, const N: usize> Polynomial<T, N> {
    /// Creates a new polynomial from an array of coefficients.
    ///
    /// # Arguments
    /// * `coefficients` - An array of coefficients `[a_0, a_1 ... a_n]`,
    ///   where `a_0` is the constant and `a_n` the nth degree term.
    ///
    /// # Returns
    /// * `polynomial` - A polynomial with the given coefficients.
    ///
    /// # Example
    /// ```
    /// use control_rs::Polynomial;
    /// // creates a quadratic equation
    /// let p = Polynomial::from_data([0, 0, 1]);
    /// assert!(p.is_monic(), "Quadratic is not monic");
    /// assert_eq!(p.degree(), Some(2), "Quadratic degree was not 2");
    /// ```
    #[inline]
    pub const fn from_data(coefficients: [T; N]) -> Self {
        // SAFETY: The array is guaranteed to have `N` elements of `T`.
        Self { coefficients }
    }

    /// Creates a new polynomial from a function closure.
    ///
    /// This is a wrapper for [`array::from_fn`].
    ///
    /// # Arguments
    /// * `cb` - The generator function, which takes the degree as input and returns the
    ///   coefficient for that degree.
    ///
    /// # Returns
    /// * `polynomial` - A new instance with the generated coefficients.
    ///
    /// # Example
    /// ```
    /// use control_rs::Polynomial;
    /// // creates a quadratic equation
    /// let p: Polynomial<i32, 3> = Polynomial::from_fn(|_| 1);
    /// assert!(p.is_monic(), "Quadratic is not monic");
    /// assert_eq!(p.degree(), Some(2), "Quadratic degree was not 2");
    /// ```
    #[inline]
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self::from_data(array::from_fn(cb))
    }

    /// Checks if the capacity is zero
    ///
    /// # Returns
    /// * `bool` - true if the capacity is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use control_rs::Polynomial;
    /// let p = Polynomial::<f32, 0>::new([]);
    /// assert_eq!(p.is_empty(), true);
    /// ```
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Access the coefficients as a slice iter
    fn iter(&self) -> slice::Iter<'_, T> {
        self.coefficients.iter()
    }

    /// Access the coefficients as a mutable slice iter
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.coefficients.iter_mut()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Polynomial<T, N> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    /// Access the coefficients as a slice iter
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut Polynomial<T, N> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    /// Access the coefficients as a slice iter
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, const N: usize> Polynomial<T, N> {}

impl<T: Clone + Zero, const N: usize> Polynomial<T, N> {
    /// Creates a new polynomial from an [Iterator].
    ///
    /// If the iterator has more than N items, the trailing items will be ignored. If the iterator
    /// has fewer than N items, the remaining indices will be filled with zeros.
    ///
    /// # Arguments
    /// * `iterator` - [Iterator] over items of `T`
    ///
    /// # Returns
    /// * `polynomial` - A zero-padded polynomial with the given coefficients.
    ///
    /// ```
    /// use control_rs::Polynomial;
    /// let p = Polynomial::from_iterator(1..6);
    /// assert_eq!(p, Polynomial::from_data([1,2,3,4,5]));
    /// ```
    #[inline]
    pub fn from_iterator<I>(iterator: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_data(utils::array_from_iterator_with_default(iterator, T::zero()))
    }

    /// Create a polynomial from another polynomial with a different capacity.
    ///
    /// * Shrinking the capacity will drop higher order terms.
    /// * Extending the capacity will pad the end of the array with `T::zero()`.
    /// * Calling this on same sized polynomials will copy one into the other.
    ///
    /// # Arguments
    /// * `other` - the polynomial to resize.
    ///
    /// # Returns
    /// * `resized_polynomial` - a polynomial with capacity `N`.
    ///
    /// # Example
    /// ```
    /// use control_rs::Polynomial;
    /// let quadratic = Polynomial::new([1, 0, 0]);
    /// let extended_quadratic = quadratic.resize::<4>();
    /// assert_eq!(extended_quadratic, Polynomial::from_data([0, 0, 1, 0]));
    /// ```
    /// TODO: Unit Test
    #[inline]
    pub fn resize<const M: usize>(&self) -> Polynomial<T, M> {
        Polynomial::<T, M>::from_iterator(self.coefficients.clone())
    }

    /// Creates a monomial from a given coefficient.
    ///
    /// A monomial consists of a single non-zero leading coefficient. This is implemented by
    /// creating a zero polynomial with the specified size and setting the final element to the
    /// given constant.
    ///
    /// # Returns
    /// * `Polynomial` - a polynomial with a single non-zero element.
    ///
    /// # Example
    /// ```
    /// use control_rs::Polynomial;
    /// let quadratic = Polynomial::<f32, 3>::monomial(1.0);
    /// assert_eq!(*quadratic.coefficient(2).unwrap(), 1.0);
    /// ```
    #[inline]
    pub fn monomial(coefficient: T) -> Self {
        Self::from_iterator(core::iter::repeat_n(T::zero(), N - 1).chain([coefficient]))
    }
}

impl<T: Copy, const N: usize> Polynomial<T, N> {
    /// Creates a new polynomial with all coefficients set to the same element
    ///
    /// # Arguments
    /// * `element` - The value to be copied into the coefficient array.
    ///
    /// # Returns
    /// * `polynomial` - polynomial with all coefficients set to `element`.
    ///
    /// # Example
    /// ```
    /// use control_rs::Polynomial;
    /// let p_ones = Polynomial::<i8, 4>::from_element(1);
    /// ```
    #[inline]
    pub const fn from_element(element: T) -> Self {
        Self::from_data([element; N])
    }

    /// Creates a new polynomial from an array.
    ///
    /// Provides a more readable interface than [`Polynomial::from_data()`] to initialize
    /// polynomials with a degree-major array.
    ///
    /// # Arguments
    /// * `coefficients` - An array of coefficients in degree-major order `[a_n, ... a_1, a_0]`.
    ///
    /// # Returns
    /// * `polynomial` - polynomial with the given coefficients.
    ///
    /// # Example
    /// ```
    /// use control_rs::Polynomial;
    /// assert_eq!(Polynomial::<i8, 0>::new([]), Polynomial::from_iterator([])); // degenerate
    /// assert_eq!(Polynomial::new([1, 2, 3]), Polynomial::from_data([3, 2, 1])); // x^2 + 2x + 3
    /// assert_eq!(Polynomial::new([0.0, 1.0, 2.0]), Polynomial::from_iterator([2.0, 1.0])); // x + 2
    /// ```
    /// TODO: Unit Test
    #[inline]
    pub const fn new(coefficients: [T; N]) -> Self {
        Self::from_data(utils::reverse_array(coefficients))
    }
}

impl<T: Default, const N: usize> Default for Polynomial<T, N> {
    #[inline]
    fn default() -> Self {
        Self::from_fn(|_| T::default())
    }
}

impl<T: Zero, const N: usize> Polynomial<T, N> {
    /// Returns the degree of the polynomial
    ///
    /// The degree is found by iterating through the array of coefficients, from N to 0, and
    /// returning the power of the first non-zero term. This is slow for polynomials with
    /// lots of leading zeros.
    ///
    /// # Returns
    /// * `Option<usize>`
    ///     * `Some(degree)` - power of the highest order non-zero coefficient
    ///     * `None` - if the length is zero or all coefficients are zero
    /// # Example
    /// ```
    /// use control_rs::Polynomial;
    /// assert_eq!(Polynomial::new([1, 1]).degree(), Some(1));
    /// ```
    #[inline]
    pub fn degree(&self) -> Option<usize> {
        utils::largest_nonzero_index(&self.coefficients)
    }
}
impl<T: Clone, const N: usize> Polynomial<T, N> {
    /// Evaluate the polynomial using Horner's method.
    ///
    /// # Arguments
    /// * `value` - A variable that supports arithmetic with the polynomials coefficients.
    ///
    /// # Returns
    /// * `T` - The value of the polynomial at the given value.
    ///
    /// # Example
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    /// let p = Polynomial::new([1, 2, 3]);
    /// assert_eq!(p.evaluate(&2), 11);
    /// ```
    /// TODO: Unit Test
    #[inline]
    pub fn evaluate<U>(&self, value: &U) -> U
    where
        U: Clone + Zero + Add<T, Output = U> + Mul<U, Output = U>,
    {
        self.coefficients
            .iter()
            .rfold(U::zero(), |acc, a_i| acc * value.clone() + a_i.clone())
    }
}

impl<T: PartialEq + One + Zero, const N: usize> Polynomial<T, N> {
    /// Checks if a polynomial is monic
    ///
    /// # Returns
    /// * `bool` - true if the leading coefficient is one, false otherwise
    #[inline]
    pub fn is_monic(&self) -> bool {
        self.leading_coefficient().is_some_and(T::is_one)
    }
}

// ===============================================================================================
//      Generic Polynomial Coefficient Access
// ===============================================================================================

impl<T, const N: usize> Polynomial<T, N> {
    /// Returns a coefficient of the polynomial
    ///
    /// # Arguments
    /// * `index` - the degree/index of the coefficient to return
    ///
    /// # Returns
    /// * `&T` - the coefficient at the specified degree
    ///
    /// # Safety
    /// * `index` must be valid. That is, `0 <= index < N`. Failing to meet this condition will
    ///   result in dereferencing an out-of-bounds or otherwise invalid memory address, leading to
    ///   undefined behavior.
    /// * The memory backing `self.coefficients` must remain valid and unchanged for the lifetime
    ///   of the returned reference.
    #[inline]
    #[must_use]
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        // TODO: Setup benchmarks to compare performance of ptr vs slice access
        // &*self.coefficients.as_ptr().wrapping_add(index)
        self.coefficients.get_unchecked(index)
    }

    /// Returns a coefficient of the polynomial
    ///
    /// # Arguments
    /// * `index` - the degree/index of the coefficient to return
    ///
    /// # Returns
    /// * `&mut T` - the coefficient at the specified degree
    ///
    /// # Safety
    /// * `index` must be valid. That is, `0 <= index < N`. Failing to meet this condition will
    ///   result in dereferencing an out-of-bounds or otherwise invalid memory address, leading to
    ///   undefined behavior.
    /// * The memory backing `self.coefficients` must remain valid and unchanged for the lifetime
    ///   of the returned reference.
    #[inline]
    #[must_use]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // TODO: Setup benchmarks to compare performance of ptr vs slice access
        // &mut *self.coefficients.as_mut_ptr().wrapping_add(index)
        self.coefficients.get_unchecked_mut(index)
    }

    /// Returns a coefficient of the polynomial
    ///
    /// # Arguments
    /// * `degree` - the degree/index of the coefficient to return
    ///
    /// # Returns
    /// * `Option<&T>`
    ///     * `Some(coefficient)` - when N > degree
    ///     * `None` - when N <= degree
    ///
    /// TODO: Unit Test
    #[inline]
    #[must_use]
    pub fn coefficient(&self, degree: usize) -> Option<&T> {
        if N > degree {
            // SAFETY: the index is usize (>= 0) and less than N,
            unsafe { Some(self.get_unchecked(degree)) }
        } else {
            None
        }
    }

    /// Returns a coefficient of the polynomial
    ///
    /// # Arguments
    /// * `degree` - the degree/index of the coefficient to return
    ///
    /// # Returns
    /// * `Option<&mut T>`
    ///     * `Some(coefficient)` - when N > degree
    ///     * `None` - when N <= degree
    ///
    /// TODO: Unit Test + Example
    #[inline]
    #[must_use]
    pub fn coefficient_mut(&mut self, degree: usize) -> Option<&mut T> {
        if N > degree {
            // SAFETY: the index is usize (>= 0) and less than N
            unsafe { Some(self.get_unchecked_mut(degree)) }
        } else {
            None
        }
    }
}

impl<T: Zero, const N: usize> Polynomial<T, N> {
    /// Returns the highest order term of the polynomial.
    ///
    /// Leading zeros will be ignored, this is equivalent to `polynomial.coefficient(degree)`.
    ///
    /// # Returns
    /// * `Option<&T>`
    ///     * `Some(leading_coefficient)` - when N > 0
    ///     * `None` - when N == 0
    ///
    /// TODO: Unit Test
    #[inline]
    #[must_use]
    pub fn leading_coefficient(&self) -> Option<&T> {
        // SAFETY: degree exists and so is a valid index
        self.degree()
            .map(|degree| unsafe { self.get_unchecked(degree) })
    }

    /// Returns the highest order term of the polynomial.
    ///
    /// Leading zeros will be ignored, this is equivalent to `polynomial.coefficient_mut(degree)`.
    ///
    /// # Returns
    /// * `Option<&mut T>`
    ///     * `Some(leading_coefficient)` - when N > 0
    ///     * `None` - when N == 0
    ///
    /// TODO: Unit Test + Example
    #[inline]
    #[must_use]
    pub fn leading_coefficient_mut(&mut self) -> Option<&mut T> {
        // SAFETY: degree exists and so is a valid index
        self.degree()
            .map(|degree| unsafe { self.get_unchecked_mut(degree) })
    }
}

impl<T, const N: usize> Polynomial<T, N>
where
    Const<N>: DimSub<U1>,
{
    /// Returns the constant term of the polynomial
    ///
    /// # Returns
    /// * `Option<&T>`
    ///     * `Some(constant)` - when N > 0
    ///     * `None` - when N == 0
    #[inline]
    #[must_use]
    pub fn constant(&self) -> &T {
        // SAFETY: `N > 0` so this is valid
        unsafe { self.get_unchecked(0) }
    }

    /// Returns the constant term of the polynomial
    ///
    /// # Returns
    /// * `Option<&mut T>`
    ///     * `Some(constant)` - when N > 0
    ///     * `None` - when N == 0
    ///
    /// TODO: Unit Test + Example
    #[inline]
    #[must_use]
    pub fn constant_mut(&mut self) -> &mut T {
        // SAFETY: `N > 0` so this is valid
        unsafe { self.get_unchecked_mut(0) }
    }
}

// ===============================================================================================
//      Calculus
// ===============================================================================================

impl<T: Clone + AddAssign + Zero + One, const N: usize> Polynomial<T, N> {
    /// Computes the derivative of a polynomial.
    ///
    /// See [`utils::differentiate()`] for more.
    ///
    /// # Returns
    /// * `Polynomial` - a polynomial with capacity `N - 1`
    ///
    /// # Examples
    /// ```
    /// use control_rs::polynomial::Polynomial;
    /// let p1 = Polynomial::new([3_i32, 2_i32, 1_i32]); // Represents 3x^2 + 2x + 1
    /// assert_eq!(
    ///     p1.derivative(),
    ///     Polynomial::from_data([2_i32, 6_i32]), // 6x + 2
    ///     "Incorrect polynomial derivative"
    /// );
    /// ```
    // TODO: Unit test
    #[inline]
    pub fn derivative<const M: usize>(&self) -> Polynomial<T, M>
    where
        Const<N>: DimSub<U1, Output = Const<M>> + nalgebra::ToTypenum,
    {
        Polynomial::from_data(utils::differentiate(&self.coefficients))
    }
}

impl<T: Clone + Zero + One + AddAssign + Div<Output = T>, const N: usize> Polynomial<T, N> {
    /// Computes the indefinite integral of a polynomial.
    ///
    /// See [`utils::integrate()`] for more.
    ///
    /// # Returns
    /// * `Polynomial` - a polynomial with capacity `N + 1`
    ///
    /// # Examples
    /// ```
    /// use control_rs::polynomial::Polynomial;
    /// let p1 = Polynomial::from_data([2_i32, 6_i32]); // 6x + 2
    /// assert_eq!(
    ///     p1.integral(1i32),
    ///     Polynomial::from_data([1_i32, 2_i32, 3_i32]), // 3x^2 + 2x + 1
    ///     "Incorrect polynomial integral"
    /// );
    /// ```
    // TODO: Unit test
    #[inline]
    pub fn integral<const M: usize>(&self, constant: T) -> Polynomial<T, M>
    where
        Const<N>: DimAdd<U1, Output = Const<M>>,
    {
        Polynomial::from_data(utils::integrate::<T, N, M>(&self.coefficients, constant))
    }
}

impl<T: Copy + Zero + One + Neg<Output = T> + Div<Output = T>, const N: usize> Polynomial<T, N> {
    /// Computes the Frobenius companion matrix of a polynomial.
    ///
    /// # Example
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::new([1.0, -6.0, 11.0, -6.0]); // x^3 - 6x^2 + 11x - 6
    /// assert_eq!(p.companion(), [[6.0, -11.0, 6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "incorrect companion");
    /// ```
    // TODO: Unit test
    pub fn companion<const M: usize>(&self) -> [[T; M]; M]
    where
        Const<N>: DimSub<U1, Output = Const<M>>,
    {
        utils::companion::<T, N, M>(&self.coefficients)
    }
}

impl<T, const N: usize> Polynomial<T, N>
where
    T: Clone + Num + Neg<Output = T> + RealField,
{
    /// Computes the roots of the polynomial
    ///
    /// # Example
    /// ```
    /// use control_rs::polynomial::Polynomial;
    /// let p = Polynomial::new([1.0, -6.0, 11.0, -6.0]);
    /// let roots = p.roots();
    /// ```
    /// # Errors
    /// * `NoRoots` - the polynomial has no roots, or the function could not compute them
    // TODO: Unit test
    pub fn roots<const M: usize>(&self) -> Result<[Complex<T>; M], utils::NoRoots>
    where
        T: Copy
            + Zero
            + One
            + Neg<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + PartialOrd
            + fmt::Debug
            + RealField,
        Const<N>: DimSub<U1, Output = Const<M>>,
        Const<M>: DimSub<U1>,
        DefaultAllocator:
            Allocator<Const<M>, DimDiff<Const<M>, U1>> + Allocator<DimDiff<Const<M>, U1>>,
    {
        utils::roots::<T, N, M>(&self.coefficients)
    }
}

// ===============================================================================================
//      Polynomial System traits
// ===============================================================================================

impl<T, const N: usize> System for Polynomial<T, N>
where
    T: Copy + Clone + Zero + One,
{
    fn zero() -> Self {
        Self::from_element(T::zero())
    }

    fn identity() -> Self {
        Self::from_iterator([T::one()])
    }
}

// ===============================================================================================
//      Polynomial-Scalar Arithmetic
// ===============================================================================================

/// # -Polynomial<T, N>
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([1, 2, 3]);
/// let p2 = -p1; // Negate p1
/// assert_eq!(p2.constant(), &-3);
/// assert_eq!(*p2.leading_coefficient().unwrap(), -1);
/// ```
impl<T: Clone + Neg<Output = T>, const N: usize> Neg for Polynomial<T, N> {
    type Output = Self;

    /// Negates all coefficients in the polynomial
    #[inline]
    fn neg(self) -> Self::Output {
        let mut neg_self = self;
        for a in &mut neg_self.coefficients {
            *a = a.clone().neg();
        }
        neg_self
    }
}

/// # Polynomial<T, N> + T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([0]);
/// let p2 = p1 + 1;
/// assert_eq!(p2.constant(), &1);
/// ```
impl<T: Clone + Add<Output = T>, const N: usize> Add<T> for Polynomial<T, N>
where
    Const<N>: DimSub<U1>, // N > 0
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        let mut result = self;
        // SAFETY: `N > 0` so the index is valid
        unsafe {
            *result.get_unchecked_mut(0) = result.get_unchecked(0).clone().add(rhs);
        }
        result
    }
}

/// # Polynomial<T, N> += T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([0]);
/// p1 += 1;
/// assert_eq!(p1.constant(), &1);
/// ```
impl<T: AddAssign, const N: usize> AddAssign<T> for Polynomial<T, N>
where
    Const<N>: DimSub<U1>, // N > 0
{
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        // SAFETY: `N > 0` so the index is valid
        unsafe {
            self.get_unchecked_mut(0).add_assign(rhs);
        }
    }
}

/// # Polynomial<T, N> - T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([0]);
/// let p2 = p1 - 1;
/// assert_eq!(p2.constant(), &-1);
/// ```
impl<T: Clone + Sub<Output = T>, const N: usize> Sub<T> for Polynomial<T, N>
where
    Const<N>: DimSub<U1>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        let mut result = self;
        // SAFETY: `N > 0` so the index is valid
        unsafe {
            *result.get_unchecked_mut(0) = result.get_unchecked(0).clone().sub(rhs);
        }
        result
    }
}

/// # Polynomial<T, N> -= T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([0]);
/// p1 -= 1;
/// assert_eq!(p1.constant(), &-1);
/// ```
impl<T: SubAssign, const N: usize> SubAssign<T> for Polynomial<T, N>
where
    Const<N>: DimSub<U1>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        // SAFETY: `N > 0` so the index is valid
        unsafe {
            self.get_unchecked_mut(0).sub_assign(rhs);
        }
    }
}

/// # Polynomial<T, N> * T
///
/// # Example
/// ```
/// use control_rs::{Polynomial, polynomial::Quadratic};
/// let p1 = Polynomial::new([0i32]);
/// let p2 = p1 * 1;
/// assert_eq!(p2.constant(), &0);
/// let p3 = Quadratic::<i32>::from_element(1);
/// assert_eq!(p3 * 2, Quadratic::<i32>::from_element(2));
/// ```
impl<T: Clone + Mul<Output = T>, const N: usize> Mul<T> for Polynomial<T, N>
where
    Const<N>: DimSub<U1>,
{
    type Output = Self;

    /// Returns a new polynomial with all coefficients scaled by rhs
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        let mut product = self;
        for a in product.iter_mut() {
            *a = a.clone().mul(rhs.clone());
        }
        product
    }
}

/// # Polynomial<T, N> *= T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([1]);
/// p1 *= 2;
/// assert_eq!(p1.constant(), &2);
/// ```
impl<T: Clone + MulAssign, const N: usize> MulAssign<T> for Polynomial<T, N>
where
    Const<N>: DimSub<U1>,
{
    fn mul_assign(&mut self, rhs: T) {
        for a_i in self.iter_mut() {
            a_i.mul_assign(rhs.clone());
        }
    }
}

/// # Polynomial<T, N> / T
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([3, 6, 9]);
/// let p2 = p1 / 3;
/// assert_eq!(p2, Polynomial::new([1, 2, 3]));
/// ```
impl<T: Clone + Div<Output = T>, const N: usize> Div<T> for Polynomial<T, N>
where
    Const<N>: DimSub<U1>,
{
    type Output = Self;

    /// Returns a new polynomial with all coefficients scaled by 1 / rhs
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        let mut quotient = self;
        for a in quotient.iter_mut() {
            *a = a.clone().div(rhs.clone());
        }
        quotient
    }
}

/// # Polynomial<T, N> /= T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([3, 6, 9]);
/// p1 /= 3;
/// assert_eq!(p1, Polynomial::new([1, 2, 3]));
/// ```
impl<T: Clone + DivAssign, const N: usize> DivAssign<T> for Polynomial<T, N>
where
    Const<N>: DimSub<U1>,
{
    fn div_assign(&mut self, rhs: T) {
        for a_i in &mut self.coefficients {
            a_i.div_assign(rhs.clone());
        }
    }
}

/// # Polynomial<T, N> % T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([1, 2, 3]);
/// assert_eq!(p1 % 2, Polynomial::new([1, 0, 1]));
/// ```
/// TODO: Unit Test
impl<T: Clone + Rem<Output = T>, const N: usize> Rem<T> for Polynomial<T, N>
where
    Const<N>: DimSub<U1>,
{
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        let mut remainder = self;
        for a in remainder.iter_mut() {
            *a = a.clone().rem(rhs.clone());
        }
        remainder
    }
}

/// # Polynomial<T, N> %= T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([1, 2, 3]);
/// p1 %= 2;
/// assert_eq!(p1, Polynomial::new([1, 0, 1]));
/// ```
/// TODO: Unit Test
impl<T: Clone + RemAssign, const N: usize> RemAssign<T> for Polynomial<T, N>
where
    Const<N>: DimSub<U1>,
{
    fn rem_assign(&mut self, rhs: T) {
        for a_i in self.iter_mut() {
            a_i.rem_assign(rhs.clone());
        }
    }
}

macro_rules! impl_left_scalar_ops {
    ($($scalar:ty),*) => {
        $(
            impl<const N: usize> Add<Polynomial<$scalar, N>> for $scalar
            where
                Const<N>: DimSub<U1>,
            {
                type Output = Polynomial<$scalar, N>;
                #[inline(always)]
                fn add(self, rhs: Polynomial<$scalar, N>) -> Self::Output {
                    rhs.add(self)
                }
            }
            impl<const N: usize> Sub<Polynomial<$scalar, N>> for $scalar
            where
                Const<N>: DimSub<U1>,
            {
                type Output = Polynomial<$scalar, N>;
                #[inline(always)]
                fn sub(self, rhs: Polynomial<$scalar, N>) -> Self::Output {
                    let mut result = Self::Output::from_iterator([self]);
                    for (a, b) in result.iter_mut().zip(rhs.iter()) {
                        *a = a.clone().sub(b.clone());
                    }
                    result
                }
            }
            impl<const N: usize> Mul<Polynomial<$scalar, N>> for $scalar
            where
                Const<N>: DimSub<U1>,
            {
                type Output = Polynomial<$scalar, N>;
                #[inline(always)]
                fn mul(self, rhs: Polynomial<$scalar, N>) -> Self::Output {
                    rhs.mul(self)
                }
            }
            impl<const N: usize> Div<Polynomial<$scalar, N>> for $scalar
            where
                Const<N>: DimSub<U1>,
            {
                type Output = Polynomial<$scalar, N>;
                #[inline(always)]
                fn div(self, rhs: Polynomial<$scalar, N>) -> Self::Output {
                    let mut result = rhs;
                    for a_i in result.iter_mut() {
                        *a_i = self.clone().div(a_i.clone());
                    }
                    result
                }
            }
        )*
    };
}

impl_left_scalar_ops!(i8, u8, i16, u16, i32, u32, isize, usize, f32, f64);

// ===============================================================================================
//      Polynomial-Polynomial Arithmetic
// ===============================================================================================

/// # Polynomial<T, N> + Polynomial<T, M>
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([1, 2, 3]);
/// assert_eq!(p1 + Polynomial::new([-1, -2, -3]), Polynomial::from_element(0));
/// ```
impl<T, const N: usize, const M: usize, const L: usize> Add<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Clone + Add<Output = T> + Zero,
    Const<N>: DimMax<Const<M>, Output = Const<L>> + DimSub<U1>,
    Const<M>: DimSub<U1>,
{
    type Output = Polynomial<T, L>;

    /// Adds the coefficients of a polynomial together
    #[inline]
    fn add(self, rhs: Polynomial<T, M>) -> Self::Output {
        Polynomial::from_data(utils::add_generic(self.coefficients, rhs.coefficients))
    }
}

/// # Polynomial<T, N> += Polynomial<T, M>
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let mut p1 = Polynomial::new([1, 2, 3]);
/// p1 += Polynomial::new([-1, -2, -3]);
/// assert_eq!(p1, Polynomial::from_element(0));
/// ```
impl<T, const N: usize, const M: usize> AddAssign<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Clone + AddAssign,
    Const<N>: DimMax<Const<M>, Output = Const<N>> + DimSub<U1>,
    Const<M>: DimSub<U1>,
{
    /// Adds the coefficients of a polynomial together
    ///
    /// Only available if N >= M
    #[inline]
    fn add_assign(&mut self, rhs: Polynomial<T, M>) {
        for (lhs, rhs) in self.iter_mut().zip(rhs.iter()) {
            *lhs += rhs.clone();
        }
    }
}

/// # Polynomial<T, N> - Polynomial<T, M>
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([1, 2, 3]);
/// assert_eq!(p1 - p1, Polynomial::from_element(0));
/// ```
impl<T, const N: usize, const M: usize, const L: usize> Sub<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Clone + Sub<Output = T> + Zero,
    Const<N>: DimMax<Const<M>, Output = Const<L>> + DimSub<U1>,
    Const<M>: DimSub<U1>,
{
    type Output = Polynomial<T, L>;

    /// Subtracts the coefficients of a polynomial from each other
    #[inline]
    fn sub(self, rhs: Polynomial<T, M>) -> Self::Output {
        Polynomial::from_data(utils::sub_generic(self.coefficients, rhs.coefficients))
    }
}

/// # Polynomial<T, N> -= Polynomial<T, M>
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let mut p1 = Polynomial::new([1, 2, 3]);
/// p1 -= Polynomial::new([1, 2, 3]);
/// assert_eq!(p1, Polynomial::from_element(0));
/// ```
impl<T, const N: usize, const M: usize> SubAssign<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Clone + SubAssign,
    Const<N>: DimMax<Const<M>, Output = Const<N>> + DimSub<U1>,
    Const<M>: DimSub<U1>,
{
    /// Subtracts the coefficients of a polynomial from each other
    ///
    /// Only available if N >= M
    #[inline]
    fn sub_assign(&mut self, rhs: Polynomial<T, M>) {
        for (a, b) in self.iter_mut().zip(rhs.iter()) {
            *a -= b.clone();
        }
    }
}

/// # Polynomial<T, N> * Polynomial<T, M>
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([1, 0, 1]);
/// assert_eq!(p1 * Polynomial::new([2, 7]), Polynomial::new([2, 7, 2, 7]));
/// ```
impl<T, const N: usize, const M: usize, const L: usize> Mul<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Clone + AddAssign + Mul<Output = T> + Zero,
    Const<M>: DimSub<U1>,
    Const<N>: DimAdd<Const<M>> + DimSub<U1>,
    <Const<N> as DimAdd<Const<M>>>::Output: DimSub<U1, Output = Const<L>>,
{
    type Output = Polynomial<T, L>;

    /// Multiplies the coefficients of a polynomial, also known as a convolution
    #[inline]
    fn mul(self, rhs: Polynomial<T, M>) -> Self::Output {
        Polynomial::from_data(utils::convolution(self.coefficients, rhs.coefficients))
    }
}

/// # Polynomial<T, N> / Polynomial<T, M>
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([1, 1, 1, 1]);
/// assert_eq!(p1 / Polynomial::new([1]), Polynomial::new([1, 1, 1, 1]));
/// ```
impl<T, const N: usize, const M: usize> Div<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Clone + Zero + Div<Output = T> + Mul<Output = T> + AddAssign + SubAssign,
    Const<N>: DimMax<Const<M>, Output = Const<N>> + DimSub<U1>,
    Const<M>: DimSub<U1>,
{
    type Output = Self;

    /// Performs long division with the two polynomials
    #[inline]
    fn div(self, divisor: Polynomial<T, M>) -> Self::Output {
        Self::from_data(utils::long_division(
            self.coefficients,
            divisor.coefficients,
        ))
    }
}

// ===============================================================================================
//      Polynomial Display Implementation
// ===============================================================================================

impl<T, const N: usize> fmt::Display for Polynomial<T, N>
where
    T: Clone + Zero + One + PartialOrd + Neg<Output = T> + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(crate::math::DEFAULT_PRECISION);
        let mut n = N;
        for (i, a_i) in self.iter().enumerate().rev() {
            if a_i.is_zero() {
                n = n.saturating_sub(1);
                continue;
            }
            if n > 1 && n > i + 1 {
                write!(f, " {} ", if *a_i >= T::zero() { "+" } else { "-" })?;
            } else if *a_i < T::zero() {
                write!(f, "-")?;
            }
            let abs_a_i = if *a_i < T::zero() {
                a_i.clone().neg()
            } else {
                a_i.clone()
            };
            if !abs_a_i.is_one() || i == 0 {
                write!(f, "{abs_a_i:.precision$}")?;
            }
            if i > 0 {
                write!(f, "x")?;
                if i > 1 {
                    write!(f, "^{i}")?;
                }
            }
        }
        Ok(())
    }
}
