//! A safe and statically sized univariate-polynomial
//!
//! TODO:
//!     * constructors
//!         * from_fn<F>(cb: F): needs example and tests
//!         * from_element(element: T): needs example
//!         * new(coefficients; [T; N]): needs example
//!         * from_constant(constant: T): needs example
//!         * monomial(coefficient: T): needs example
//!         * resize(): needs example and tests
//!         * compose(f: Polynomial, g: Polynomial) -> Polynomial: Needs investigation
//!         * to_monic(&self) -> Self
//!         * zero(): needs tests and example
//!         * one(): needs tests and example
//!     * accessors
//!         * degree(): needs tests and example
//!         * is_monic(): needs tests and example
//!         * is_zero(): needs tests and example
//!         * is_one(): needs tests and example
//!         * coefficient(): needs tests and example
//!         * coefficient_mut(): needs tests and example
//!         * constant(): needs tests and example
//!         * constant_mut(): needs tests and example
//!         * leading_coefficient(): needs tests and example
//!         * leading_coefficient_mut(): needs tests and example
//!         * as_array(): needs implementation, tests and example
//!     * arithmatic
//!         * Zero() -> Self: needs tests and example
//!         * One() -> Self: needs tests and example
//!         * Neg() -> Self: needs tests and example
//!         * Add/AddAssign(&self, rhs: T): needs tests and example
//!         * Sub/SubAssign(&self, rhs: T): needs tests and example
//!         * Mul/MulAssign(&self, rhs: T): needs tests and example
//!         * Div/DivAssign(&self, rhs: T) (Euclidean division): needs test and example
//!         * Rem/RemAssign(&self, rhs: T) (remainder): needs test and example
//!         * Add/AddAssign(&self, rhs: Polynomial<T, M>): needs tests and example
//!         * Sub/SubAssign(&self, rhs: Polynomial<T, M>): needs tests and example
//!         * Mul/MulAssign(&self, rhs: Polynomial<T, M>): Can't implement safely
//!         * Div/DivAssign(&self, rhs: Polynomial<T, M>) (Euclidean division): Can't implement safely
//!         * Rem/RemAssign(&self, rhs: Polynomial<T, M>) (remainder): Can't implement safely
//!     * calculus
//!         * evaluate<U>(x: U) -> U: needs test and example
//!         * derivative(p_src: &Polynomial<T, M>) -> Self: cant implement safely
//!         * integral(p_src: &Polynomial<T, M>) -> Self: cant implement safely
//!         * foil_roots(&mut self, roots: &[T]) -> Result<(), Polynomial>
//!         * foil_complex_roots(&mut self, roots: &[Complex<T>]) -> Result<(), Polynomial>
//!         * real_roots(p: Polynomial, roots: &mut [T]) -> Result<(), PolynomialError>
//!         * complex_roots(p: Polynomial, roots: &mut [Complex<T>]) -> Result<(), PolynomialError>
//!     * formatting
//!         * Display with precision option
//!         * Latex / symbolic formatter (optional)

#[cfg(feature = "std")]
use std::{iter, mem::MaybeUninit, ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg, Rem, RemAssign}, array};

#[cfg(not(feature = "std"))]
use core::{iter, mem::MaybeUninit, ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg, Rem, RemAssign}, array};
use nalgebra::Const;
use num_traits::{Zero, One};

// ===============================================================================================
//      Polynomial Tests
// ===============================================================================================

#[cfg(test)]
mod basic_tests;

#[cfg(test)]
mod arithmatic_tests;

/// Helper function to reverse arrays given to [Polynomial::new()]
#[inline(always)]
const fn reverse_array<T: Copy, const N: usize>(input: [T; N]) -> [T; N] {
    let mut output = input;
    let mut i = 0;
    while i < N / 2 {
        let tmp = output[i];
        output[i] = output[N - 1 - i];
        output[N - 1 - i] = tmp;
        i += 1;
    }

    output
}

/// Statically sized univariate polynomial
///
/// This struct stores the coefficients of a polynomial a(x):
/// <pre>
/// a(s) = a_n * x^n + a_(n-1) * x^(n-1) + ... + a_1 * x + a_0
/// </pre>
/// where `n` is the degree of the polynomial.
///
/// The coefficients are stored in ascending degree order (i.e., index 0 is the lowest degree
/// or constant term, index N-1 is the highest degree term).
///
/// # Generic Arguments
/// * `T` - Type of the coefficients in the polynomial
/// * `N` - Capacity of the underlying array
///
/// # Example
/// ```rust
/// let quadratic = control_rs::Polynomial::new([1, 0, 0]);
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Polynomial<T, const N: usize> {
    /// coefficients of the polynomial `[a_0, a_1 ... a_n]`, lowest to highest degree order
    coefficients: [T; N],
}

impl<T, const N: usize> Polynomial<T, N> {
    /// Creates a new polynomial from a function closure.
    ///
    /// This is a wrapper for [array::from_fn].
    ///
    /// # Arguments
    /// * `cb` - The generator function, which takes the degree (`usize`) as input and returns the
    /// coefficient (`T`) for that degree
    ///
    /// # Returns
    /// * `polynomial` - A new instance with the generated coefficients
    ///
    /// TODO: Test + Example
    #[inline]
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> T
    {
        Self { coefficients: array::from_fn(cb) }
    }

    /// Checks if the capacity is zero
    ///
    /// TODO: Test + Example
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool { N == 0 }
}

impl<T: Zero, const N: usize> Polynomial<T, N> {
    /// Creates a new polynomial from an [Iterator].
    ///
    /// This function copies items from the iterator into an array of [MaybeUninit<T>]. If the
    /// iterator has more items than N, the trailing items will be ignored. If the iterator has
    /// less than N items, the remaining indices will be filled with zeros.
    ///
    /// # Arguments
    /// * `iter` - [Iterator] with item type `T`
    ///
    /// # Returns
    /// * `polynomial` - A new instance with the given coefficients
    ///
    /// TODO: Test + Example
    #[inline]
    pub fn from_iterator<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut uninit_coefficients: [MaybeUninit<T>; N] =
            unsafe { MaybeUninit::uninit().assume_init() };

        let mut degree = 0;
        // copy all items from the iterator into the uninitialized array.
        // SAFETY: will only copy N.min(iter.len()) elements.
        for (c, d) in uninit_coefficients.iter_mut().zip(iter.into_iter()) {
            *c = MaybeUninit::new(d);
            degree += 1;
        }
        // pad zeros if the iterator was not long enough
        for c in uninit_coefficients.iter_mut().skip(degree) {
            *c = MaybeUninit::new(T::zero());
        }

        // Safely convert `[MaybeUninit<T>; MAX_POLYNOMIAL_CAPACITY]` to `[T; MAX_POLYNOMIAL_CAPACITY]`
        // This is safe because we have ensured all elements are initialized.
        let initialized_coefficients = unsafe {
            // Get a pointer to the `uninit_coefficients` array.
            // Cast it to a pointer to an array of `T`.
            // Then `read()` the value from that pointer.
            // This is equivalent to transmute from `[MaybeUninit<T>; N]` to `[T; N]`.
            (uninit_coefficients.as_ptr() as *const [T; N]).read()
        };

        // SAFETY: all elements have been initialized
        Self {
            coefficients: initialized_coefficients,
        }
    }

    /// Create a polynomial from another polynomial with a different capacity.
    ///
    /// * Shrinking the capacity will drop higher order terms.
    /// * Extending the capacity will pad the array with T::zero().
    ///
    /// # Arguments
    /// * `other` - the polynomial to resize
    ///
    /// # Returns
    /// * `resized_polynomial` - a polynomial with a new capacity
    ///
    /// TODO: Test + Example
    #[inline]
    pub fn resize<const M: usize>(self, other: Polynomial<T, M>) -> Polynomial<T, N> {
        Self::from_iterator(other.coefficients)
    }

    /// Creates a polynomial with all coefficients except the constant term set to zero
    ///
    /// If N == 0, this will return an empty polynomial.
    ///
    /// # Arguments
    /// * `constant` - The value of the constant term
    ///
    /// # Returns
    /// * `Polynomial` - a polynomial with only the trailing term
    ///
    /// TODO: Test + Example
    #[inline]
    pub fn from_constant(constant: T) -> Self {
        Self::from_iterator([constant])
    }

    /// Creates a monomial from a given constant.
    ///
    /// A monomial consists of a single non-zero leading coefficient. This is implemented by
    /// creating a zero polynomial with the specified size and setting the final element to the
    /// given constant.
    ///
    /// # Returns
    /// * `Polynomial` - a polynomial with only the trailing term
    /// TODO: Cleanup + Test + Example
    #[inline]
    pub fn monomial(coefficient: T) -> Self {
        let mut polynomial = Self::from_fn(|_| T::zero());
        if let Some(a_n) = polynomial.leading_coefficient_mut() { *a_n = coefficient }
        polynomial
    }
}

impl<T: Clone, const N: usize> Polynomial<T, N> {
    /// Creates a new polynomial from an [Iterator].
    ///
    /// This function copies items from the iterator into an array of [MaybeUninit<T>]. If the
    /// iterator has more items than N, the trailing items will be ignored. If the iterator has
    /// less than N items, the remaining indices will be filled with the default value.
    ///
    /// # Arguments
    /// * `iter` - [Iterator] with item type `T`
    ///
    /// # Returns
    /// * `polynomial` - A new instance with the given coefficients
    ///
    /// TODO: Test + Example
    ///     * could make default a closure so it doesn't have to be Clone
    #[inline]
    pub fn from_iterator_with_default<I>(iter: I, default: T) -> Self
    where
        I: IntoIterator<Item=T>,
    {
        let mut uninit_coefficients: [MaybeUninit<T>; N] =
            unsafe { MaybeUninit::uninit().assume_init() };

        let mut degree = 0;
        // copy all items from the iterator into the uninitialized array.
        // SAFETY: will only copy N.min(iter.len()) elements.
        for (c, d) in uninit_coefficients.iter_mut().zip(iter.into_iter()) {
            *c = MaybeUninit::new(d);
            degree += 1;
        }
        // pad zeros if the iterator was not long enough
        for c in uninit_coefficients.iter_mut().skip(degree) {
            *c = MaybeUninit::new(default.clone());
        }

        // Safely convert `[MaybeUninit<T>; MAX_POLYNOMIAL_CAPACITY]` to `[T; MAX_POLYNOMIAL_CAPACITY]`
        // This is safe because we have ensured all elements are initialized.
        let initialized_coefficients = unsafe {
            // Get a pointer to the `uninit_coefficients` array.
            // Cast it to a pointer to an array of `T`.
            // Then `read()` the value from that pointer.
            // This is equivalent to transmute from `[MaybeUninit<T>; N]` to `[T; N]`.
            (uninit_coefficients.as_ptr() as *const [T; N]).read()
        };

        // SAFETY: all elements have been initialized
        Self {
            coefficients: initialized_coefficients,
        }
    }
}

impl<T: Copy, const N: usize> Polynomial<T, N> {
    /// Creates a new polynomial with all coefficients set to the same element
    ///
    /// # Arguments
    /// * `element` - The value to be copied into the coefficient array
    ///
    /// # Returns
    /// * `polynomial` - polynomial with all coefficients set to `element`
    ///
    /// TODO: Test + Example
    #[inline]
    pub const fn from_element(element: T) -> Self {
        Self { coefficients:  [element; N] }
    }

    /// Creates a new polynomial from an array.
    ///
    /// Expects an array of coefficients sorted highest to lowest degree.
    ///
    /// # Arguments
    /// * `coefficients` - The coefficient array in descending degree order (highest -> lowest)
    ///
    /// # Returns
    /// * `polynomial` - polynomial with the given coefficients
    ///
    /// TODO: Test + Example
    #[inline]
    pub const fn new(coefficients: [T; N]) -> Self {
        Self { coefficients:  reverse_array(coefficients) }
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
    /// The degree is found by iterating through the array of coefficients, from 0 to N, and
    /// returning the power of the first non-zero term. This is slow for polynomials with
    /// lots of leading zeros.
    ///
    /// # Returns
    /// * `Option<usize>`
    ///     * `Some(degree)` - power of the highest order non-zero coefficient
    ///     * `None` - if the length is zero or all coefficients are zero
    ///
    /// TODO: Test + Example
    #[inline]
    pub fn degree(&self) -> Option<usize> {
        for (i, a_i) in self.coefficients.iter().enumerate().rev() {
            if !a_i.is_zero() { return Some(i); }
        }
        None
    }
}
impl<T: Clone, const N: usize> Polynomial<T, N> {
    /// Evaluate the polynomial at the given value.
    ///
    /// # Arguments
    /// * `value` - The value at which to evaluate the polynomial.
    ///
    /// # Returns
    /// * `T` - The value of the polynomial at the given value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::new([1, 2, 3]);
    /// let result = p.evaluate(2);
    /// ```
    ///
    /// TODO: Test
    #[inline]
    pub fn evaluate<U>(&self, value: U) -> U
    where
        U: Clone + Zero + Add<T, Output = U> + Mul<U, Output = U>,
    {
        self.coefficients.iter().rev().fold(U::zero(), |acc, a_i| acc * value.clone() + a_i.clone())
    }
}
impl<T: PartialEq + One, const N: usize> Polynomial<T, N> {
    /// Checks if a polynomial is monic
    ///
    /// # Returns
    /// * `bool` - true if the leading coefficient is one, false otherwise
    ///
    /// TODO: Test + Example
    #[inline]
    pub fn is_monic(&self) -> bool {
        if self.is_empty() { false }
        else {
            // SAFETY: N > 0 so N-1 is valid, and the reference drops before the function returns
            unsafe {
                if self.get_unchecked(N-1).is_one() {true}
                else { false }
            }
        }
    }
}

// ===============================================================================================
//      Polynomial Coefficient Accessors
// ===============================================================================================

impl<T, const N: usize> Polynomial<T, N> {
    /// Returns a coefficient of the polynomial
    ///
    /// # Arguments
    /// * `index` - the degree of the coefficient to return
    ///
    /// # Returns
    /// * `&T` - the coefficient at the specified degree
    ///
    /// # Safety
    /// 1.  `index` must be valid. That is, `0 <= index < N`. Failing to meet this condition will
    /// result in dereferencing an out-of-bounds or otherwise invalid memory address, leading to
    /// undefined behavior.
    /// 2.  The memory backing `self.coefficients` must remain valid and unchanged for the lifetime
    /// of the returned reference.
    ///
    /// TODO: Test
    #[inline]
    #[must_use]
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        &*self.coefficients.as_ptr().wrapping_add(index)
    }

    /// Returns a coefficient of the polynomial
    ///
    /// # Arguments
    /// * `index` - the degree of the coefficient to return
    ///
    /// # Returns
    /// * `&mut T` - the coefficient at the specified degree
    ///
    /// # Safety
    /// 1.  `index` must be valid. That is, `0 <= index < N`. Failing to meet this condition will
    /// result in dereferencing an out-of-bounds or otherwise invalid memory address, leading to
    /// undefined behavior.
    /// 2.  The memory backing `self.coefficients` must remain valid and unchanged for the lifetime
    /// of the returned reference.
    ///
    /// TODO: Test
    #[inline]
    #[must_use]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        &mut *self.coefficients.as_mut_ptr().wrapping_add(index)
    }
}

impl<T, const N: usize> Polynomial<T, N> {
    /// Returns a coefficient of the polynomial
    ///
    /// # Arguments
    /// * `degree` - the degree of the coefficient to return
    ///
    /// # Returns
    /// * `Option<&T>`
    ///     * `Some(constant)` - when N > degree: coefficient at the specified degree
    ///     * `None` - when N <= degree
    ///
    /// TODO: Test + Example
    #[inline]
    #[must_use]
    pub fn coefficient(&self, degree: usize) -> Option<&T> {
        if N > degree {
            // SAFETY: the index is usize (>= 0) and less than N, the reference will also have the
            // same lifetime as self.
            unsafe { Some(self.get_unchecked(degree)) }
        } else { None }
    }

    /// Returns the constant term of the polynomial
    ///
    /// # Returns
    /// * `Option<&T>`
    ///     * `Some(constant)` - when N > 0: coefficient at the start of the array
    ///     * `None` - when N == 0
    ///
    /// TODO: Test + Example
    #[inline]
    #[must_use]
    pub fn constant(&self) -> Option<&T> {
        self.coefficient(0)
    }

    /// Returns the highest order term of the polynomial
    ///
    /// # Returns
    /// * `Option<&T>`
    ///     * `Some(constant)` - when N > 0: coefficient at the end of the array
    ///     * `None` - when N == 0
    ///
    /// TODO: Test + Example
    #[inline]
    #[must_use]
    pub fn leading_coefficient(&self) -> Option<&T> {
        self.coefficient(N-1)
    }

    /// Returns a coefficient of the polynomial
    ///
    /// # Arguments
    /// * `degree` - the degree of the coefficient to return
    ///
    /// # Returns
    /// * `Option<&mut T>`
    ///     * `Some(coefficient)` - when N > degree: coefficient at the specified degree
    ///     * `None` - when N <= degree
    ///
    /// TODO: Test + Example
    #[inline]
    #[must_use]
    pub fn coefficient_mut(&mut self, degree: usize) -> Option<&mut T> {
        if N > degree {
            // SAFETY: the index is usize (>= 0) and less than N, the reference will also have the
            // same lifetime as self.
            unsafe { Some(self.get_unchecked_mut(degree)) }
        }
        else { None }
    }

    /// Returns the constant term of the polynomial
    ///
    /// # Returns
    /// * `Option<&mut T>`
    ///     * `Some(constant)` - when N > 0: coefficient at the start of the array
    ///     * `None` - when N == 0
    ///
    /// TODO: Test + Example
    #[inline]
    #[must_use]
    pub fn constant_mut(&mut self) -> Option<&mut T> {
        self.coefficient_mut(0)
    }

    /// Returns the highest order term of the polynomial
    ///
    /// # Returns
    /// * `Option<&mut T>`
    ///     * `Some(leading_coefficient)` - when N > 0: coefficient at the end of the array
    ///     * `None` - when N == 0
    ///
    /// TODO: Test + Example
    #[inline]
    #[must_use]
    pub fn leading_coefficient_mut(&mut self) -> Option<&mut T> {
        self.coefficient_mut(N-1)
    }
}

impl<T: Zero, const N: usize> Zero for Polynomial<T, N> {
    /// TODO: Doc + Test + Example
    #[inline]
    fn zero() -> Self {
        Self::from_fn(|_| T::zero())
    }

    /// Checks if coefficients of a polynomial are zero
    ///
    /// # Returns
    /// * `bool` - false if any coefficients are non-zero or the array is empty, otherwise true
    ///
    /// TODO: Doc + Test + Example
    #[inline]
    fn is_zero(&self) -> bool {
        if N > 0 {
            for c in &self.coefficients {
                if !c.is_zero() { return false }
            }
        }
        true
    }
}

/// TODO: Doc + Test + Example
impl<T: Zero + PartialEq + One, const N: usize> One for Polynomial<T, N> {
    #[inline]
    fn one() -> Self {
        Self::from_fn(|_| T::one())
    }

    /// Checks if coefficients of a polynomial are zero and the constant is one
    ///
    /// # Returns
    /// * `bool` - true if all coefficients are zero and the constant is one, false otherwise
    ///
    #[inline]
    fn is_one(&self) -> bool {
        if self.degree() == Some(0) {
            if let Some(constant) = self.constant() { return constant.is_one() }
            else { false }
        } else { false }
    }
}

// ===============================================================================================
//      Polynomial-Scalar Arithmatic
// ===============================================================================================

/// TODO: Doc + Test + Example
impl<T: Clone + Neg<Output = T>, const N: usize> Neg for Polynomial<T, N> {
    type Output = Self;

    /// Negates all coefficients in the polynomial
    #[inline]
    fn neg(self) -> Self::Output {
        Self::from_fn(|i| unsafe{ -self.get_unchecked(i).clone() })
        // from iterator requires T: Zero, even if it is the right size
        // Self::from_iterator(self.coefficients.iter().map(|a_i| -a_i.clone()))
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T: Clone + Add<Output = T>, const N: usize, const M: usize> Add<T> for Polynomial<T, N>
where
    Const<N>: nalgebra::DimMax<nalgebra::U1, Output = Const<M>>,
{
    type Output = Polynomial<T, M>;

    fn add(self, rhs: T) -> Self::Output {
        // alternatively sum the constants and store as the default, then pass
        // coefficients.iter().take(N-1) as the iterator
        Polynomial::from_iterator_with_default(self.coefficients.into_iter().enumerate().map(|(i, a_i)| {
            if i > 0 { a_i }
            else { a_i + rhs.clone() }
        }), rhs.clone())
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T: AddAssign, const N: usize> AddAssign<T> for Polynomial<T, N> {
    fn add_assign(&mut self, rhs: T) {
        if let Some(constant) = self.constant_mut() {
            *constant += rhs;
        }
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T: Clone + Sub<Output = T>, const N: usize> Sub<T> for Polynomial<T, N> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        let mut result = self.clone();
        if let Some(constant) = result.constant_mut() {
            *constant = constant.clone() - rhs;
        }
        result
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T: SubAssign, const N: usize> SubAssign<T> for Polynomial<T, N> {
    fn sub_assign(&mut self, rhs: T) {
        if let Some(constant) = self.constant_mut() {
            *constant -= rhs;
        }
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T: Clone + Mul<Output = T>, const N: usize> Mul<T> for Polynomial<T, N> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::from_iterator_with_default(self.coefficients.into_iter().map(|a_i| a_i * rhs.clone()), rhs.clone())
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T: Clone + MulAssign, const N: usize> MulAssign<T> for Polynomial<T, N> {
    fn mul_assign(&mut self, rhs: T) {
        for a_i in self.coefficients.iter_mut() {
            *a_i *= rhs.clone()
        }
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T: Clone + Div<Output = T>, const N: usize> Div<T> for Polynomial<T, N> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self::from_iterator_with_default(self.coefficients.into_iter().map(|a_i| a_i / rhs.clone()), rhs.clone())
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T, const N: usize> DivAssign<T> for Polynomial<T, N>
where
    T: DivAssign + Copy,
{
    fn div_assign(&mut self, rhs: T) {
        for a_i in self.coefficients.iter_mut() {
            *a_i /= rhs.clone()
        }
    }
}

/// TODO: Repair + Doc + Test + Example
impl<T: Clone + Rem<Output = T>, const N: usize> Rem<T> for Polynomial<T, N> {
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        Self::from_iterator_with_default(self.coefficients.into_iter().map(|a_i| a_i % rhs.clone()), rhs.clone())
    }
}

/// TODO: Repair + Doc + Test + Example
impl<T: Clone + RemAssign, const N: usize> RemAssign<T> for Polynomial<T, N> {
    fn rem_assign(&mut self, rhs: T) {
        for a_i in self.coefficients.iter_mut() {
            *a_i %= rhs.clone()
        }
    }
}

// ===============================================================================================
//      Polynomial-Polynomial Arithmatic
//
// These functions are intentionally only available for polynomials with the same capacity. This
// is meant to force user implementations to provide size checking and guaranteed operations.
// ===============================================================================================

/// TODO: Cleanup + Doc + Test + Example
impl<T: Clone + Add<Output = T> + Zero, const N: usize, const M: usize, const L: usize> Add<Polynomial<T, M>> for Polynomial<T, N>
where
    nalgebra::Const<N>: nalgebra::DimMax<nalgebra::Const<M>, Output = nalgebra::Const<L>>,
{
    type Output = Polynomial<T, L>;

    fn add(self, rhs: Polynomial<T, M>) -> Self::Output {
        Self::Output::from_fn(|degree|
            match (degree < N, degree < M) {
                (true, true) => self.coefficients[degree].clone() + rhs.coefficients[degree].clone(),
                (true, false) => self.coefficients[degree].clone(),
                (false, true) => rhs.coefficients[degree].clone(),
                (false, false) => T::zero(),
            }
        )
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T: Clone + AddAssign, const N: usize, const M: usize> AddAssign<Polynomial<T, M>> for Polynomial<T, N>
where
    nalgebra::Const<N>: nalgebra::DimMax<nalgebra::Const<M>, Output = nalgebra::Const<N>>,
{
    fn add_assign(&mut self, rhs: Polynomial<T, M>) {
        for i in 0..M {
            // safe because N > M and 0 <= i < M
            self.coefficients[i] += rhs.coefficients[i].clone();
        }
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T: Clone + Sub<Output = T>, const N: usize> Sub for Polynomial<T, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Polynomial::from_fn(|i| self.coefficients[i].clone() - rhs.coefficients[i].clone())
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T: Clone + SubAssign, const N: usize> SubAssign for Polynomial<T, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coefficients[i] -= rhs.coefficients[i].clone();
        }
    }
}

/// TODO: Cleanup + Doc + Test + Example
impl<T, const N: usize, const M: usize, const L: usize> Mul<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Add<Output = T> + Mul<Output = T> + Zero + Copy, // Need Zero for initialization
    nalgebra::Const<N>: nalgebra::DimMul<nalgebra::Const<M>, Output = nalgebra::Const<L>>,
{
    type Output = Polynomial<T, L>;

    fn mul(self, rhs: Polynomial<T, M>) -> Self::Output {
        let mut result = [T::zero(); L];

        // Standard polynomial multiplication algorithm (Cauchy product)
        for i in 0..N { // Iterate through terms of self
            for j in 0..N { // Iterate through terms of rhs
                // Coefficient of x^(i+j) in the product is (self.coefficients[i] * rhs.coefficients[j])
                result[i + j] = result[i + j] + self.coefficients[i] * rhs.coefficients[j];
            }
        }

        Polynomial::new( result )
    }
}

impl<T, const N: usize, const M: usize> MulAssign<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Add<Output = T> + Mul<Output = T> + Zero + Copy + PartialEq, // Add PartialEq for degree()
    Polynomial<T, N>: Mul<Polynomial<T, M>, Output = Polynomial<T, N>>, // Ensure the output size matches N
{
    fn mul_assign(&mut self, rhs: Polynomial<T, M>) {
        // This only works if `N` is exactly `self.degree() + rhs.degree() + 1` after multiplication.
        // If `N` is not `N_self + M - 1`, this will either truncate or zero-pad,
        // which might not be the desired behavior for `MulAssign`.
        //
        // A common pattern for `MulAssign` with fixed-size arrays is to
        // panic if the result won't fit, or require N >= N + M - 1,
        // which means N must be very large or M must be 1.
        //
        // For simplicity and adherence to trait, we'll assume the result fits.
        // It's likely more practical for a `Polynomial` using `Vec<T>`.

        // Calculate the product into a temporary polynomial
        let product = self.clone() * rhs; // Requires `Clone` if `self` is consumed by `Mul`

        // Check if the product fits into `self` (N should be `product.len()`)
        // If N != product.len(), we have a problem for `MulAssign` semantics.
        // For this skeleton, we assume N is appropriate or truncate.
        // If N < product.len(), we truncate higher terms.
        // If N > product.len(), we zero-pad.

        let mut new_coeffs = [T::zero(); N];
        let common_len = N.min(product.len());
        for i in 0..common_len {
            new_coeffs[i] = product.coefficients[i];
        }
        self.coefficients = new_coeffs;
    }
}


// Div<Polynomial<T, M>> (Euclidean division)
// This is significantly more complex and requires specific properties for T (e.g., field, or Euclidean domain).
// The algorithm typically involves long division.
// It also needs to return a new polynomial.
impl<T, const N: usize, const M: usize> Div<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Zero + One + Copy + PartialEq, // Extensive bounds for division
// T also needs to support inverses if it's a field (e.g. f64) or be an integer type with proper division behavior.
// The divisor (rhs) must not be the zero polynomial.
{
    type Output = Polynomial<T, { N.saturating_sub(M) + 1 }>; // Degree of quotient is deg(N) - deg(M). Size: deg(quotient) + 1

    fn div(self, rhs: Polynomial<T, M>) -> Self::Output {
        // Handle division by zero polynomial
        if rhs.is_zero() {
            panic!("Division by zero polynomial");
        }

        // Implementation of polynomial long division (Euclidean division)
        // This is a complex algorithm.
        // For integer coefficients, you might need a way to handle non-divisible terms (e.g., using fractions).
        // For floating-point coefficients, this is more direct.
        //
        // Algorithm sketch:
        // 1. Determine degrees of self and rhs.
        // 2. If deg(self) < deg(rhs), quotient is 0.
        // 3. Loop:
        //    a. Divide leading term of current dividend by leading term of rhs.
        //    b. This gives a term for the quotient.
        //    c. Multiply that term by rhs and subtract from dividend.
        //    d. Update dividend.
        // 4. Collect quotient terms.
        todo!("Implement polynomial Euclidean division (long division algorithm)");
    }
}

// DivAssign<Polynomial<T, M>>
impl<T, const N: usize, const M: usize> DivAssign<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Zero + One + Copy + PartialEq,
    Polynomial<T, N>: Div<Polynomial<T, M>, Output = Polynomial<T, N>>, // Ensure output size matches N
{
    fn div_assign(&mut self, rhs: Polynomial<T, M>) {
        // This is problematic for fixed-size arrays if the quotient's degree
        // doesn't match N.
        // Similar to MulAssign, this typically implies N must be exact for the quotient,
        // or truncation/padding occurs.
        let quotient = self.clone() / rhs; // Requires `Clone`

        let mut new_coeffs = [T::zero(); N];
        let common_len = N.min(quotient.len());
        for i in 0..common_len {
            new_coeffs[i] = quotient.coefficients[i];
        }
        self.coefficients = new_coeffs;
    }
}

// Rem<Polynomial<T, M>> (Remainder of Euclidean division)
impl<T, const N: usize, const M: usize> Rem<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Zero + One + Copy + PartialEq,
{
    // The degree of the remainder is always less than the degree of the divisor.
    // So, the size of the remainder polynomial will be M.
    type Output = Polynomial<T, M>;

    fn rem(self, rhs: Polynomial<T, M>) -> Self::Output {
        if rhs.is_zero() {
            panic!("Remainder by zero polynomial");
        }

        // Implementation of polynomial long division to get the remainder.
        // You'll essentially run the division algorithm and return the final dividend.
        todo!("Implement polynomial remainder (long division algorithm remainder part)");
    }
}

// RemAssign<Polynomial<T, M>>
impl<T, const N: usize, const M: usize> RemAssign<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Zero + One + Copy + PartialEq,
    Polynomial<T, N>: Rem<Polynomial<T, M>, Output = Polynomial<T, N>>, // Ensure output size matches N
{
    fn rem_assign(&mut self, rhs: Polynomial<T, M>) {
        // Similar issues to MulAssign and DivAssign regarding fixed size.
        // The remainder's degree is always less than the divisor's (M).
        // If N < M, this is fine. If N > M, we're zeroing out higher terms.
        let remainder = self.clone() % rhs; // Requires `Clone`

        let mut new_coeffs = [T::zero(); N];
        let common_len = N.min(remainder.len()); // remainder.len() is M here
        for i in 0..common_len {
            new_coeffs[i] = remainder.coefficients[i];
        }
        self.coefficients = new_coeffs;
    }
}

// ===============================================================================================
//      Polynomial Display Implementation
// ===============================================================================================

// impl<T, const N: usize> fmt::Display for Polynomial<T, N>
// where
//     T: Copy + Number + PartialOrd + Neg<Output = T> + fmt::Display,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         let num_coefficients = self.degree() + 1;
//         for i in 0..num_coefficients {
//             let coefficient = self.coefficient(i);
//             if coefficient == T::zero() {
//                 continue;
//             }
//
//             if i > 0 {
//                 write!(f, " {} ", if coefficient >= T::zero() { "+" } else { "-" })?;
//             } else if coefficient < T::zero() {
//                 write!(f, "-")?;
//             }
//
//             let abs_coefficient = if coefficient < T::zero() { -coefficient } else { coefficient };
//             let exp = num_coefficients - 1 - i;
//
//             if abs_coefficient != T::one() || exp == 0 {
//                 write!(f, "{}", abs_coefficient)?;
//             }
//
//             if exp > 0 {
//                 write!(f, "x")?;
//                 if exp > 1 {
//                     write!(f, "^{}", exp)?;
//                 }
//             }
//         }
//
//         Ok(())
//     }
// }