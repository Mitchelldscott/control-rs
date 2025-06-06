//! Array-based univariate-polynomial
//!
//! This module contains a base implementation of a generic array polynomial. To guarantee safe and
//! deterministic implementations of all arithmatic, some implementations are only available for
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
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
    slice,
};
use nalgebra::{
    allocator::Allocator, Complex, Const, DefaultAllocator, DimDiff, DimName, DimSub, OMatrix,
    RealField, U1,
};
use num_traits::{Float, Num, One, Zero};
// ===============================================================================================
//      Polynomial Specializations
// ===============================================================================================

mod constant;
pub use constant::Constant;

mod line;
pub use line::Line;

// ===============================================================================================
//      Polynomial Tests
// ===============================================================================================

#[cfg(test)]
mod basic_tests;

#[cfg(test)]
mod arithmatic_tests;
// ===============================================================================================
//      Polynomial
// ===============================================================================================

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

/// Initialize an array from an iterator.
///
/// # Arguments
/// * `iterator` - An [Iterator] over a collection of `T`.
/// * `default` - the default value to use if the iterator is not long enough.
///
/// # Returns
/// * `initialized_array` - An array with all elements initialized
///
/// # Safety
/// This function uses `MaybeUninit` and raw pointer casting to avoid requiring `T: Default + Copy`.
/// The safety relies on:
/// - Fully initializing all elements of the `[MaybeUninit<T>; N]` array before calling `read()`
/// - Not reading from or dropping uninitialized memory
fn initialize_array_from_iterator_with_default<I, T, const N: usize>(
    iterator: I,
    default: T,
) -> [T; N]
where
    T: Clone,
    I: IntoIterator<Item = T>,
{
    // SAFETY: `[MaybeUninit<T>; N]` is valid.
    let mut uninit_array: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut degree = 0;

    // SAFETY: zip() will only iterate over `N.min(iter.len())` elements so no out-of-bounds access
    // can occur.
    for (c, d) in uninit_array.iter_mut().zip(iterator.into_iter()) {
        *c = MaybeUninit::new(d);
        degree += 1;
    }

    // SAFETY: `T: Clone`, and we are initializing all remaining uninitialized slots.
    for c in uninit_array.iter_mut().skip(degree) {
        *c = MaybeUninit::new(default.clone());
    }

    // SAFETY:
    // - All `N` elements of `uninit_array` have now been initialized.
    // - `MaybeUninit<T>` does not drop its content, so no double-drop will occur.
    // - We can safely transmute it to `[T; N]` by reading the pointer.
    unsafe {
        // Get a pointer to the `uninit_array` array.
        // Cast it to a pointer to an array of `T`.
        // Then `read()` the value from that pointer.
        // This is equivalent to transmute from `[MaybeUninit<T>; N]` to `[T; N]`.
        (uninit_array.as_ptr() as *const [T; N]).read()
    }
}

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
    /// where `a_0` is the constant and `a_n` the nth degree term.
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
    /// This is a wrapper for [array::from_fn].
    ///
    /// # Arguments
    /// * `cb` - The generator function, which takes the degree as input and returns the
    /// coefficient for that degree.
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
    /// let p = Polynomial::new([]);
    /// assert_eq!(p.is_empty(), true);
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Access the coefficients as a slice iter
    pub fn iter(&self) -> slice::Iter<T> {
        self.coefficients.iter()
    }

    /// Access the coefficients as a mutable slice iter
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.coefficients.iter_mut()
    }
}

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
    /// let p = Polynomial::from_iterator([1, 2, 3, 4, 5]);
    /// assert_eq!(p.degree(), Some(4));
    /// assert_eq!(p.constant(), Some(&1));
    /// ```
    #[inline]
    pub fn from_iterator<I>(iterator: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_data(initialize_array_from_iterator_with_default(
            iterator,
            T::zero(),
        ))
    }

    /// Create a polynomial from another polynomial with a different capacity.
    ///
    /// * Shrinking the capacity will drop higher order terms.
    /// * Extending the capacity will pad the end of the array with T::zero().
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
    /// assert_eq!(*extended_quadratic.degree().unwrap(), 2);
    /// assert_eq!(*extended_quadratic.leading_coefficient().unwrap(), 0);
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
        Self::from_iterator(
            core::iter::repeat(T::zero())
                .take(N - 1)
                .chain([coefficient]),
        )
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
    /// Provides a more readable interface than [Polynomial::from_data] to initialize polynomials with
    /// a degree-major array.
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
    /// let p = Polynomial::new([1, 0, 0]);
    /// assert_eq!(p.degree(), Some(2));
    /// assert_eq!(p.leading_coefficient(), Some(&1));
    /// ```
    ///
    /// TODO: Unit Test
    #[inline]
    pub const fn new(coefficients: [T; N]) -> Self {
        Self::from_data(reverse_array(coefficients))
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
    #[inline]
    pub fn degree(&self) -> Option<usize> {
        for (i, a_i) in self.coefficients.iter().enumerate().rev() {
            if !a_i.is_zero() {
                return Some(i);
            }
        }
        None
    }
}
impl<T: Clone, const N: usize> Polynomial<T, N> {
    /// Evaluate the polynomial using Horner's method.
    ///
    /// # Arguments
    /// * `value` - A variable that supports arithmatic with the polynomials coefficients.
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
    /// TODO: Unit Test
    #[inline]
    pub fn evaluate<U>(&self, value: U) -> U
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
        if let Some(degree) = self.degree() {
            // SAFETY: degree < N so degree is valid
            unsafe { self.get_unchecked(degree).is_one() }
        } else {
            false
        }
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
    /// result in dereferencing an out-of-bounds or otherwise invalid memory address, leading to
    /// undefined behavior.
    /// * The memory backing `self.coefficients` must remain valid and unchanged for the lifetime
    /// of the returned reference.
    #[inline(always)]
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
    /// result in dereferencing an out-of-bounds or otherwise invalid memory address, leading to
    /// undefined behavior.
    /// * The memory backing `self.coefficients` must remain valid and unchanged for the lifetime
    /// of the returned reference.
    #[inline(always)]
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

    /// Returns the constant term of the polynomial
    ///
    /// # Returns
    /// * `Option<&T>`
    ///     * `Some(constant)` - when N > 0
    ///     * `None` - when N == 0
    #[inline]
    #[must_use]
    pub fn constant(&self) -> Option<&T> {
        self.coefficient(0)
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
    pub fn constant_mut(&mut self) -> Option<&mut T> {
        self.coefficient_mut(0)
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
        if let Some(degree) = self.degree() {
            // SAFETY: degree < N so degree is valid
            unsafe { Some(self.get_unchecked(degree)) }
        } else {
            None
        }
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
        if let Some(degree) = self.degree() {
            // SAFETY: degree < N so degree is valid
            unsafe { Some(self.get_unchecked_mut(degree)) }
        } else {
            None
        }
    }
}

// ===============================================================================================
//      Calculus
// ===============================================================================================

/// Result of taking the derivative of a polynomial.
///
/// This enum represents the possible outcomes when computing the derivative of a polynomial:
/// - A zero polynomial (when differentiating a constant)
/// - A valid polynomial of the same capacity
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PolynomialDerivative<T, const N: usize> {
    /// Represents a zero polynomial with the given zero value.
    /// This variant is returned when differentiating a constant polynomial.
    Zero,
    /// Represents a valid polynomial resulting from the derivative operation.
    Ok(Polynomial<T, N>),
}

impl<T: Clone + Zero + One + Add<Output = T> + Mul<Output = T>, const N: usize> Polynomial<T, N> {
    /// Calculates the derivative of a polynomial using the power rule.
    ///
    /// This function performs the core derivative calculation by applying the power rule
    /// `d/dx(ax^n) = n * ax^{n-1}` to each term of the polynomial.
    /// It iterates through the coefficients, effectively shifting them to a lower exponent
    /// and multiplying by the original exponent.
    ///
    /// This function is safe for any size polynomial (`N`) and correctly handles
    /// polynomials of any degree, including first-order and constant terms.
    /// The constant term (coefficient of x^0) is implicitly handled by `skip(1)`,
    /// as its derivative is zero and thus not included in the resulting polynomial coefficients.
    ///
    /// This function is private because calling it on a zero or empty polynomial will have no
    /// effect. Users should call [Polynomial::derivative] to ensure the base case is handled
    /// properly.
    ///
    /// # Examples
    ///
    /// If `self` represents the polynomial $3x^3 + 2x^2 + 5x + 10$:
    /// The resulting polynomial from `derivative_internal` would represent $9x^2 + 4x + 5$.
    #[inline]
    fn derivative_internal(&self) -> Self {
        let mut exponent = T::zero();
        Polynomial::from_iterator(self.iter().skip(1).map(|a_i| {
            exponent = exponent.clone() + T::one();
            a_i.clone() * exponent.clone()
        }))
    }

    /// Computes the derivative of a polynomial.
    ///
    /// This function determines if the polynomial is constant or empty (degree does not exist).
    /// If the polynomial has a degree (meaning it's not empty), it delegates to
    /// `derivative_internal` to perform the actual power rule calculations.
    /// If the polynomial is empty or represents a constant (e.g., $f(x) = C$),
    /// its derivative is the zero polynomial.
    ///
    /// # Returns
    ///
    /// - `PolynomialDerivative::Zero`: If the polynomial is constant (degree 0) or empty (degree
    /// None)
    /// - `PolynomialDerivative::Ok(Polynomial<T, N>)`: If the polynomial's degree is
    ///   greater than 0, containing the coefficients of the resulting derivative polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use control_rs::polynomial::{Polynomial, PolynomialDerivative}; // Assuming these traits/structs are in your crate
    /// use num_traits::{Zero, One};
    /// // Example for a polynomial of degree 2
    /// let p1 = Polynomial::new([1_i32, 2_i32, 3_i32]); // Represents 3x^2 + 2x + 1
    /// assert_eq!(
    ///     p1.derivative(),
    ///     PolynomialDerivative::Ok(Polynomial::from_data([2_i32, 6_i32, 0_i32])),
    ///     "Expected a valid polynomial derivative"
    /// );
    ///
    /// // Example for a constant polynomial
    /// let p2 = Polynomial::new([5_i32]); // Represents 5
    /// assert_eq!(p2.derivative(), PolynomialDerivative::Zero, "Expected a zero polynomial derivative");
    ///
    /// // Example for an empty polynomial (assuming it can be constructed like this)
    /// let p3 = Polynomial::from_iterator(std::iter::empty()); // Represents an empty polynomial
    /// assert_eq!(p3.derivative(), PolynomialDerivative::Zero, "Expected a zero polynomial derivative")
    /// ```
    #[inline]
    pub fn derivative(&self) -> PolynomialDerivative<T, N> {
        if let Some(_) = self.degree() {
            PolynomialDerivative::Ok(self.derivative_internal())
        } else {
            PolynomialDerivative::Zero
        }
    }
}

/// Result of integrating a polynomial.
///
/// This enum represents the possible outcomes when computing the integral of a polynomial:
/// - A constant (when integrating a zero or empty polynomial)
/// - A valid polynomial with the same capacity
/// - A truncated polynomial (when the result would exceed the maximum capacity N)
pub enum PolynomialIntegral<T, const N: usize> {
    /// Represents a constant value resulting from integrating a zero polynomial.
    /// This variant contains the integration constant.
    Constant(Polynomial<T, 1>),
    /// Represents a valid polynomial resulting from the integration operation.
    /// This variant is returned when the integral fits within the polynomial's capacity.
    Ok(Polynomial<T, N>),
    /// Represents a truncated polynomial resulting from the integration operation.
    /// This variant is returned when the integral would exceed the polynomial's capacity (N),
    /// so the result is truncated to fit.
    Truncated(Polynomial<T, N>),
}

impl<T: Clone + Zero + One + Add<Output = T> + Div<Output = T>, const N: usize> Polynomial<T, N> {
    /// Calculates the indefinite integral of a polynomial using the power rule.
    ///
    /// This internal function computes the indefinite integral of the polynomial by
    /// applying the reverse power rule ($\int ax^n dx = \frac{a}{n+1}x^{n+1}$) to each term.
    /// It also incorporates the provided `constant` of integration as the new constant term.
    ///
    /// The process involves:
    /// 1. Prepending the `constant` of integration as the coefficient of $x^0$.
    /// 2. Iterating through the original polynomial's coefficients $a_i$ (corresponding to $x^i$).
    /// 3. For each $a_i$, calculating the new coefficient as $a_i / (i+1)$, where
    ///    $(i+1)$ is the new exponent.
    ///
    /// This function will not resize a polynomial. If there is not enough space, it will truncate
    /// the highest order term.
    ///
    /// # Arguments
    /// * `constant`: The constant of integration, $C$. This will become the coefficient
    ///   of $x^0$ in the resulting polynomial.
    ///
    // TODO: Examples
    #[inline]
    fn integral_internal(&self, constant: T) -> Self {
        let mut exponent = T::one();
        Polynomial::from_iterator([constant].into_iter().chain(self.iter().enumerate().map(
            |(_, a_i)| {
                exponent = exponent.clone() + T::one();
                a_i.clone() / exponent.clone()
            },
        )))
    }

    /// Computes the indefinite integral of a polynomial.
    ///
    /// If the polynomial has a degree (meaning it's not empty), it delegates to
    /// `integral_internal` to perform the actual power rule calculations. If the
    /// polynomial is empty teh result is a constant polynomial. If the degree is
    /// `N-1` then the result will truncate the highest order coefficient.
    #[inline]
    pub fn integral(&self, constant: T) -> PolynomialIntegral<T, N> {
        if let Some(degree) = self.degree() {
            let integral = self.integral_internal(constant);
            if degree + 1 == N {
                PolynomialIntegral::Truncated(integral)
            } else {
                PolynomialIntegral::Ok(integral)
            }
        } else {
            PolynomialIntegral::Constant(Polynomial::from_data([constant]))
        }
    }
}

impl<T, const N: usize> Polynomial<T, N>
where
    T: 'static + Clone + Num + Neg<Output = T> + fmt::Debug,
    Const<N>: DimSub<U1>,
    DimDiff<Const<N>, U1>: DimName,
    DefaultAllocator: Allocator<DimDiff<Const<N>, U1>, DimDiff<Const<N>, U1>>,
{
    /// Constructs the companion matrix of the polynomial.
    ///
    /// The companion matrix is a square matrix whose eigenvalues are the roots of the polynomial.
    /// It is constructed from an identity matrix, a zero column and a row of the polynomial's
    /// coefficients scaled by the highest term.
    ///
    /// <pre>
    /// |  a[0...n-1] / a_n  |
    /// |     I      0     |
    /// </pre>
    /// <pre>
    /// |  -a_(n-1)/a_n  -a_(n-2)/a_n  -a_(n-3)/a_n  ...  -a_1/a_0  -a_0/a_n  |
    /// |     1         0         0      ...       0            0         |
    /// |     0         1         0      ...       0            0         |
    /// |     0         0         1      ...       0            0         |
    /// |    ...       ...       ...     ...      ...          ...        |
    /// |     0         0         0      ...       1            0         |
    /// </pre>
    ///
    /// # Returns
    /// * `OMatrix<T, DimDiff<Const<N>, U1>, DimDiff<Const<N>, U1>>` - The companion matrix
    /// representation of the polynomial.
    ///
    /// # Example
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::new([1.0, -6.0, 11.0, -6.0]);
    /// let companion_matrix = p.companion();
    /// ```
    pub fn companion(&self) -> OMatrix<T, DimDiff<Const<N>, U1>, DimDiff<Const<N>, U1>> {
        // return companion;
        if let Some(leading_coefficient) = self.leading_coefficient() {
            OMatrix::<T, DimDiff<Const<N>, U1>, DimDiff<Const<N>, U1>>::from_fn(|i, j| {
                if i == 0 {
                    // SAFETY: the index j is less than N
                    unsafe { -self.get_unchecked(N - j - 1).clone() / leading_coefficient.clone() }
                } else {
                    if i == j + 1 {
                        T::one()
                    } else {
                        T::zero()
                    }
                }
            })
        } else {
            OMatrix::<T, DimDiff<Const<N>, U1>, DimDiff<Const<N>, U1>>::zeros()
        }
    }
}

impl<T, const N: usize> Polynomial<T, N>
where
    T: 'static + Clone + Num + Neg<Output = T> + fmt::Debug + RealField + Float,
    Const<N>: DimSub<U1>,
    DimDiff<Const<N>, U1>: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<DimDiff<Const<N>, U1>, DimDiff<Const<N>, U1>>
        + Allocator<DimDiff<Const<N>, U1>, DimDiff<DimDiff<Const<N>, U1>, U1>>
        + Allocator<DimDiff<DimDiff<Const<N>, U1>, U1>>
        + Allocator<DimDiff<Const<N>, U1>>,
{
    /// Computes the roots of the polynomial.
    ///
    /// Edge cases:
    /// - if the leading coefficient is zero, this should reduce the order and recurse, having
    /// issues with trait bounds (currently returns NaN)
    /// - all coefficients are zero: all roots are infinite
    /// - if there are two coefficients and the lead is non-zero: the root is
    /// `-coefficient[1]/coefficient[0]`
    /// - if all but the first coefficient are zero: all roots are zero
    ///
    /// For very high-order polynomial's this may be inefficient, especially for degenerate cases.
    ///
    /// # Returns
    /// * `OMatrix<Complex<T>, Const<D>, U1>` - A column vector containing the computed roots.
    ///
    /// # Example
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::new([1.0, -6.0, 11.0, -6.0]);
    /// let roots = p.roots();
    /// ```
    pub fn roots(&self) -> OMatrix<Complex<T>, DimDiff<Const<N>, U1>, U1> {
        if !self.coefficients[0].is_zero() {
            if N == 2 {
                OMatrix::<Complex<T>, DimDiff<Const<N>, U1>, U1>::from_element(Complex::new(
                    -self.coefficients[1].clone() / self.coefficients[0].clone(),
                    T::zero(),
                ))
            } else {
                let num_zeros = (0..N).fold(0, |acc, i| match self.coefficients[i] == T::zero() {
                    true => acc + 1,
                    false => acc,
                });

                if num_zeros == N {
                    // zero/degenerate polynomial, all infinite roots
                    OMatrix::<Complex<T>, DimDiff<Const<N>, U1>, U1>::from_element(Complex::new(
                        T::infinity(),
                        T::infinity(),
                    ))
                } else if num_zeros == N - 1 {
                    // unit case, all zero roots
                    OMatrix::<Complex<T>, DimDiff<Const<N>, U1>, U1>::from_element(Complex::new(
                        T::zero(),
                        T::zero(),
                    ))
                } else {
                    // need to know more specifics about what matrices work with complex_eigenvalues,
                    // the current cases fixed an infinite loop in the test, but certainly not a guaranteed solution
                    self.companion().complex_eigenvalues()
                }
            }
        } else {
            // should be able to reduce order and keep trying, but having issues with recursive trait bounds
            OMatrix::<Complex<T>, DimDiff<Const<N>, U1>, U1>::from_element(Complex::new(
                T::nan(),
                T::nan(),
            ))
        }
    }
}

// ===============================================================================================
//      Generic Polynomial-Scalar Arithmatic
//
//  The following operations are provided for polynomials of any capacity:
//      * Neg
//      * Mul<T> & Mul<Polynomial<T,N>> for T
//      * MulAssign<T>
//      * Div<T>
//      * DivAssign<T>
//      * Rem<T>
//      * RemAssign<T>
//
// ===============================================================================================

/// # -Polynomial<T, N>
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([1, 2, 3]);
/// let p2 = -p1; // Negate p1
/// assert_eq!(*p2.constant().unwrap(), -3);
/// assert_eq!(*p2.leading_coefficient().unwrap(), -1);
/// ```
impl<T: Clone + Neg<Output = T>, const N: usize> Neg for Polynomial<T, N> {
    type Output = Self;

    /// Negates all coefficients in the polynomial
    #[inline]
    fn neg(self) -> Self::Output {
        Self::from_fn(|i| {
            // SAFETY: The index is usize (>= 0) and less than N
            unsafe { self.get_unchecked(i).clone().neg() }
        })
    }
}

/// # Polynomial<T, N> * T
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([]);
/// let p2 = p1 * 1;
/// assert_eq!(*p2.constant(), None);
/// ```
impl<T: Clone + Mul<Output = T>, const N: usize> Mul<T> for Polynomial<T, N> {
    type Output = Self;

    /// Returns a new polynomial with all coefficients scaled by rhs
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::from_fn(|i| {
            // SAFETY: The index is usize (>= 0) and less than N
            unsafe { self.get_unchecked(i).clone() * rhs.clone() }
        })
    }
}

/// # Polynomial<T, N> *= T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([]);
/// p1 *= 2;
/// assert_eq!(*p1.constant(), None);
/// ```
impl<T: Clone + MulAssign, const N: usize> MulAssign<T> for Polynomial<T, N> {
    fn mul_assign(&mut self, rhs: T) {
        for a_i in self.coefficients.iter_mut() {
            *a_i *= rhs.clone();
        }
    }
}

/// # Polynomial<T, N> / T
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([]);
/// let p2 = p1 / 1;
/// assert_eq!(*p2.constant(), None);
/// ```
impl<T: Clone + Div<Output = T>, const N: usize> Div<T> for Polynomial<T, N> {
    type Output = Self;

    /// Returns a new polynomial with all coefficients scaled by 1 / rhs
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::Output::from_fn(|i| {
            // SAFETY: The index is usize (>= 0) and less than N
            unsafe { self.get_unchecked(i).clone() / rhs.clone() }
        })
    }
}

/// # Polynomial<T, N> /= T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([]);
/// p1 /= 2;
/// assert_eq!(*p1.constant(), None);
/// ```
impl<T: Clone + DivAssign, const N: usize> DivAssign<T> for Polynomial<T, N> {
    fn div_assign(&mut self, rhs: T) {
        for a_i in self.coefficients.iter_mut() {
            *a_i /= rhs.clone();
        }
    }
}

/// # Polynomial<T, N> % T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([]);
/// let p2 = p1 % 2;
/// assert_eq!(*p2.constant().unwrap(), None);
/// ```
/// TODO: Unit Test
impl<T: Clone + Rem<Output = T>, const N: usize> Rem<T> for Polynomial<T, N> {
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        Self::from_fn(|i|
            // SAFETY: The index is usize (>= 0) and less than N
            unsafe {
                self.get_unchecked(i).clone() % rhs.clone()
            })
    }
}

/// # Polynomial<T, N> %= T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([]);
/// p1 %= 2;
/// assert_eq!(*p1.constant().unwrap(), None);
/// ```
/// TODO: Unit Test
impl<T: Clone + RemAssign, const N: usize> RemAssign<T> for Polynomial<T, N> {
    fn rem_assign(&mut self, rhs: T) {
        for a_i in self.coefficients.iter_mut() {
            *a_i %= rhs.clone();
        }
    }
}

macro_rules! impl_generic_left_scalar_mul {
    ($($scalar:ty),*) => {
        $(
            impl<const N: usize> Mul<Polynomial<$scalar, N>> for $scalar {
                type Output = Polynomial<$scalar, N>;

                fn mul(self, rhs: Polynomial<$scalar, N>) -> Self::Output {
                    Self::Output::from_iterator(rhs.iter().map(|a_i| self.clone() * a_i.clone()))
                }
            }
        )*
    };
}

impl_generic_left_scalar_mul!(i8, u8, i16, u16, i32, u32, isize, usize, f32, f64);

// ===============================================================================================
//      Empty Polynomial-Scalar Arithmatic
// ===============================================================================================

/// # Polynomial<T, 0> + T
///
/// This is a base case implementation where a scalar is added to an empty polynomial. The result
/// is a polynomial with a constant term equal to the scalar.
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([]);
/// let p2 = p1 + 1;
/// assert_eq!(*p2.constant().unwrap(), 1);
/// ```
impl<T> Add<T> for Polynomial<T, 0> {
    type Output = Polynomial<T, 1>;

    /// Returns a new polynomial with the given constant term
    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Polynomial::from_data([rhs])
    }
}

/// # Polynomial<T, 0> - T
///
/// This is a base case implementation where a scalar is subtracted from an empty polynomial. The
/// result is a polynomial with a constant term equal to 0 - rhs.
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([]);
/// let p2 = p1 - 1;
/// assert_eq!(*p2.constant().unwrap(), -1);
/// ```
impl<T: Neg<Output = T>> Sub<T> for Polynomial<T, 0> {
    type Output = Polynomial<T, 1>;

    /// Returns a new polynomial with the constant term equal to -rhs
    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Self::Output::from_data([rhs.neg()])
    }
}

macro_rules! impl_base_case_left_scalar_arithmatic {
    ($($scalar:ty),*) => {
        $(
            impl Add<Polynomial<$scalar, 0>> for $scalar {
                type Output = Polynomial<$scalar, 1>;

                fn add(self, _: Polynomial<$scalar, 0>) -> Self::Output {
                    Polynomial::from_data([self.clone()])
                }
            }
            impl AddAssign<Polynomial<$scalar, 0>> for $scalar {
                fn add_assign(&mut self, _rhs: Polynomial<$scalar, 0>) {}
            }
            impl Sub<Polynomial<$scalar, 0>> for $scalar {
                type Output = Polynomial<$scalar, 1>;

                fn sub(self, _: Polynomial<$scalar, 0>) -> Self::Output {
                    Polynomial::from_data([self.clone()])
                }
            }
            impl SubAssign<Polynomial<$scalar, 0>> for $scalar {
                fn sub_assign(&mut self, _rhs: Polynomial<$scalar, 0>) {}
            }
        )*
    };
}

impl_base_case_left_scalar_arithmatic!(i8, u8, i16, u16, i32, u32, isize, usize, f32, f64);

// ===============================================================================================
//      Empty Polynomial-Generic Polynomial Arithmatic
//
//  Assignment operators are not implemented because it would lead to unexpected behavior (i.e.
// `p1 += p2` would have a different result than `p3 = p1 + p2`). Similarly, multiplication and
// division with an empty polynomial is logically invalid.
// ===============================================================================================

/// # Polynomial<T, 0> + Polynomial<T, N>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([]);
/// let p2 = Polynomial::new([1]);
/// let p3 = p1 + p2;
/// assert_eq!(*p3.constant().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: Clone, const N: usize> Add<Polynomial<T, N>> for Polynomial<T, 0> {
    type Output = Polynomial<T, N>;

    fn add(self, rhs: Polynomial<T, N>) -> Self::Output {
        rhs.clone()
    }
}

/// # Polynomial<T, 0> - Polynomial<T, N>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([]);
/// let p2 = Polynomial::new([1]);
/// let p3 = p1 - p2;
/// assert_eq!(*p3.constant().unwrap(), -1);
/// ```
/// TODO: Unit Test
impl<T: Clone + Neg<Output = T>, const N: usize> Sub<Polynomial<T, N>> for Polynomial<T, 0> {
    type Output = Polynomial<T, N>;

    fn sub(self, rhs: Polynomial<T, N>) -> Self::Output {
        rhs.clone().neg()
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
                write!(f, "{}", abs_a_i)?;
            }
            if i > 0 {
                write!(f, "x")?;
                if i > 1 {
                    write!(f, "^{}", i)?;
                }
            }
        }
        Ok(())
    }
}
