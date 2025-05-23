//! A safe and statically sized univariate-polynomial
//!
//! TODO:
//!     * constructors
//!         * from_fn<F>(cb: F): needs example and tests
//!         * from_element(element: T): needs example
//!         * new(coefficients; [T; N]): needs example
//!         * from_constant(constant: T): needs example
//!         * monomial(): needs example
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
use std::{ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg, Rem, RemAssign}, array};

#[cfg(not(feature = "std"))]
use core::{ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg, Rem, RemAssign}, array};

use num_traits::{Zero, One};

// ===============================================================================================
//      Polynomial Tests
// ===============================================================================================

#[cfg(test)]
mod basic_tests;

#[cfg(test)]
mod arithmatic_tests;

// ===============================================================================================
//      Polynomial Errors
// ===============================================================================================


/// Errors that can occur during polynomial initialization.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PolynomialInitError {
    /// Indicates that a slice provided to create a polynomial had an incorrect length.
    LengthMismatch { expected: usize, actual: usize },
}

/// Errors that can occur during polynomial operations.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PolynomialOpError {
    /// Indicates that an attempt was made to access or set a coefficient for a power
    /// greater than the polynomial's degree.
    PowerOutOfBounds { requested_power: usize, max_degree: usize },
    // Note: Arithmetic overflows (e.g., integer overflow) are dependent on the type `T`
    // and its operator implementations. This struct itself does not add overflow checks
    // beyond what T provides. For safety-critical applications, use `T` with
    // defined overflow behavior (e.g., saturating or checked arithmetic).
}

/// Helper function to reverse arrays given to [Polynomial::new()]
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
    #[inline]
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> T
    {
        Self { coefficients: array::from_fn(cb) }
    }

    /// Checks if the capacity is zero
    #[inline]
    pub fn is_empty(&self) -> bool { N == 0 }
}

impl<T: Copy, const N: usize> Polynomial<T, N> {
    /// Creates a new polynomial from an array
    ///
    /// Expects an array of coefficients sorted highest to lowest degree.
    ///
    /// # Arguments
    /// * `coefficients` - The coefficient array in descending degree order (highest -> lowest)
    ///
    /// # Returns
    /// * `polynomial` - polynomial with the given coefficients
    #[inline]
    pub const fn new(coefficients: [T; N]) -> Self {
        Self { coefficients:  reverse_array(coefficients) }
    }

    /// Creates a new polynomial with all coefficients set to the same element
    ///
    /// # Arguments
    /// * `element` - The value to be copied into the coefficient array
    ///
    /// # Returns
    /// * `polynomial` - polynomial with all coefficients set to `element`
    #[inline]
    pub const fn from_element(element: T) -> Self {
        Self::new([element; N])
    }
}

impl<T: Default, const N: usize> Default for Polynomial<T, N> {
    #[inline]
    fn default() -> Self {
        Self::from_fn(|_| T::default())
    }
}

impl<T: Clone + Zero, const N: usize> Polynomial<T, N> {
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
    #[inline]
    pub fn resize<const M: usize>(self, other: Polynomial<T, M>) -> Polynomial<T, N> {
        Self::from_fn(|i| {
            if i < M { other.coefficients[i].clone() }
            else { T::zero() }
        })
    }

    /// Creates a polynomial with all coefficients except the constant term set to zero
    ///
    /// If N == 0 this will return an empty polynomial.
    ///
    /// # Arguments
    /// * `constant` - The value of the constant term
    ///
    /// # Returns
    /// * `Polynomial` - a polynomial with only the trailing term
    pub fn from_constant(constant: T) -> Self {
        let mut polynomial = Self::from_fn(|_| T::zero());
        if !polynomial.is_empty() { polynomial.coefficients[0] = constant }
        polynomial
    }
}
impl<T: Zero + One, const N: usize> Polynomial<T, N> {
    /// Creates a polynomial with all coefficients but the constant set to zero
    ///
    /// # Returns
    /// * `Polynomial` - a polynomial with only the trailing term
    pub fn monomial() -> Self {
        let mut polynomial = Self::from_fn(|_| T::zero());
        if !polynomial.is_empty() { polynomial.coefficients[0] = T::one() }
        polynomial
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
    pub fn evaluate<U>(&self, value: U) -> U
    where
        U: Clone + Zero + Add<T, Output = U> + Mul<U, Output = U>,
    {
        (0..N).rev().fold(U::zero(), |acc, i| {
            acc * value.clone() + self.coefficients[i].clone()
        })
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
    pub fn degree(&self) -> Option<usize> {
        for i in (0..N).rev() {
            if !self.coefficients[i].is_zero() {
                return Some(i);
            }
        }
        None
    }
}
impl<T: PartialEq + One, const N: usize> Polynomial<T, N> {
    /// Checks if a polynomial is monic
    ///
    /// # Returns
    /// * `bool` - true if the leading coefficient is one, false otherwise
    pub fn is_monic(&self) -> bool {
        if self.is_empty() { false }
        else if self.coefficients[0].is_one() { true }
        else { false }
    }
}

// ===============================================================================================
//      Polynomial Coefficient Accessors
// ===============================================================================================

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
    pub fn coefficient(&self, degree: usize) -> Option<&T> {
        if N > degree { Some(&self.coefficients[degree]) }
        else { None }
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
    pub fn coefficient_mut(&mut self, degree: usize) -> Option<&T> {
        if N > degree { Some(&mut self.coefficients[N - degree - 1]) }
        else { None }
    }

    /// Returns the constant term of the polynomial
    ///
    /// # Returns
    /// * `Option<&T>`
    ///     * `Some(constant)` - when N > 0: coefficient at the start of the array
    ///     * `None` - when N == 0
    pub fn constant(&self) -> Option<&T> {
        if self.is_empty() { None }
        else { Some(&self.coefficients[0]) }
    }

    /// Returns the constant term of the polynomial
    ///
    /// # Returns
    /// * `Option<&mut T>`
    ///     * `Some(constant)` - when N > 0: coefficient at the start of the array
    ///     * `None` - when N == 0
    pub fn constant_mut(&mut self) -> Option<&mut T> {
        if self.is_empty() { None }
        else { Some(&mut self.coefficients[0]) }
    }

    /// Returns the highest order term of the polynomial
    ///
    /// # Returns
    /// * `Option<&T>`
    ///     * `Some(constant)` - when N > 0: coefficient at the end of the array
    ///     * `None` - when N == 0
    pub fn leading_coefficient(&self) -> Option<&T> {
        if self.is_empty() { None }
        else { Some(&self.coefficients[N-1]) }
    }

    /// Returns the highest order term of the polynomial
    ///
    /// # Returns
    /// * `Option<&mut T>`
    ///     * `Some(leading_coefficient)` - when N > 0: coefficient at the end of the array
    ///     * `None` - when N == 0
    pub fn leading_coefficient_mut(&mut self) -> Option<&mut T> {
        if self.is_empty() { None }
        else { Some(&mut self.coefficients[N-1]) }
    }
}

impl<T: Zero, const N: usize> Zero for Polynomial<T, N> {
    fn zero() -> Self {
        Self::from_fn(|_| T::zero())
    }

    /// Checks if coefficients of a polynomial are zero
    ///
    /// # Returns
    /// * `bool` - false if any coefficients are non-zero or the array is empty, otherwise true
    fn is_zero(&self) -> bool {
        if N > 0 {
            for c in &self.coefficients {
                if !c.is_zero() { return false }
            }
        }
        true
    }
}

impl<T: Zero + PartialEq + One, const N: usize> One for Polynomial<T, N> {
    fn one() -> Self {
        Self::from_fn(|_| T::one())
    }

    /// Checks if coefficients of a polynomial are zero and the constant is one
    ///
    /// # Returns
    /// * `bool` - true if all coefficients are zero and the constant is one, false otherwise
    fn is_one(&self) -> bool {
        if self.is_empty() { return false }
        else if !self.coefficients[N-1].is_one() { return false }
        for c in &self.coefficients {
            if !c.is_zero() { return false }
        }
        true
    }
}

// ===============================================================================================
//      Polynomial-Scalar Arithmatic
// ===============================================================================================

impl<T: Clone + Neg<Output = T>, const N: usize> Neg for Polynomial<T, N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_fn(|i| -self.coefficients[i].clone())
    }
}

impl<T: Clone + Add<Output = T>, const N: usize> Add<T> for Polynomial<T, N> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let mut result = self.clone();
        if N > 0 {
            // Add `rhs` to the constant term (x^0)
            result.coefficients[0] = result.coefficients[0].clone() + rhs;
        }
        result
    }
}

impl<T: AddAssign, const N: usize> AddAssign<T> for Polynomial<T, N> {
    fn add_assign(&mut self, rhs: T) {
        if N > 0 {
            // Add `rhs` to the constant term (x^0)
            self.coefficients[0] += rhs;
        }
    }
}

impl<T: Clone + Sub<Output = T>, const N: usize> Sub<T> for Polynomial<T, N> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        let mut result = self.clone();
        if N > 0 {
            // Subtract `rhs` from the constant term (x^0)
            result.coefficients[0] = result.coefficients[0].clone() - rhs;
        }
        result
    }
}

impl<T: SubAssign, const N: usize> SubAssign<T> for Polynomial<T, N> {
    fn sub_assign(&mut self, rhs: T) {
        if N > 0 {
            // Subtract `rhs` from the constant term (x^0)
            self.coefficients[0] -= rhs;
        }
    }
}

impl<T: Clone + Mul<Output = T>, const N: usize> Mul<T> for Polynomial<T, N> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut result = self.clone();
        for i in 0..N {
            result.coefficients[i] = result.coefficients[i].clone() * rhs.clone();
        }
        result
    }
}

impl<T: Clone + MulAssign, const N: usize> MulAssign<T> for Polynomial<T, N> {
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..N {
            self.coefficients[i] *= rhs.clone();
        }
    }
}

impl<T: Clone + Div<Output = T>, const N: usize> Div<T> for Polynomial<T, N> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut result = self.clone();
        for i in 0..N {
            result.coefficients[i] = result.coefficients[i].clone() / rhs.clone();
        }
        self
    }
}

impl<T, const N: usize> DivAssign<T> for Polynomial<T, N>
where
    T: DivAssign + Copy,
{
    fn div_assign(&mut self, rhs: T) {
        for i in 0..N {
            self.coefficients[i] /= rhs;
        }
    }
}

impl<T: Clone + Rem<Output = T>, const N: usize> Rem<T> for Polynomial<T, N> {
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        let mut result = self.clone();
        for i in 0..N {
            result.coefficients[i] = result.coefficients[i].clone() % rhs.clone();
        }
        self
    }
}

impl<T: Clone + RemAssign, const N: usize> RemAssign<T> for Polynomial<T, N> {
    fn rem_assign(&mut self, rhs: T) {
        for i in 0..N {
            self.coefficients[i] %= rhs.clone();
        }
    }
}

// ===============================================================================================
//      Polynomial-Polynomial Arithmatic
//
// These functions are intentionally only available for polynomials with the same capacity. This
// is meant to force user implementations to provide size checking and guaranteed operations.
// ===============================================================================================

impl<T: Clone + Add<Output = T>, const N: usize> Add for Polynomial<T, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_fn(|degree| self.coefficients[degree].clone() + rhs.coefficients[degree].clone())
    }
}

impl<T: Clone + AddAssign, const N: usize> AddAssign for Polynomial<T, N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coefficients[i] += rhs.coefficients[i].clone();
        }
    }
}

impl<T: Clone + Sub<Output = T>, const N: usize> Sub for Polynomial<T, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Polynomial::from_fn(|i| self.coefficients[i].clone() - rhs.coefficients[i].clone())
    }
}

impl<T: Clone + SubAssign, const N: usize> SubAssign for Polynomial<T, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coefficients[i] -= rhs.coefficients[i].clone();
        }
    }
}

// Mul<Polynomial<T, M>>
// Polynomial multiplication: the degree of the product is deg(P) + deg(Q).
// So, the size of the resulting array is N + M - 1.
// We need `T` to have `Add` and `Mul` traits.
impl<T, const N: usize> Mul for Polynomial<T, N>
where
    T: Add<Output = T> + Mul<Output = T> + Zero + Copy, // Need Zero for initialization
{
    type Output = Polynomial<T, N * N>; // Size of result: (N-1) + (M-1) + 1 = N+M-1

    fn mul(self, rhs: Polynomial<T, M>) -> Self::Output {
        // Handle edge cases where N or M is 0 (empty polynomials)
        if N == 0 || M == 0 {
            return Polynomial { coefficients: [T::zero(); {N + M - 1}] }; // Result is zero polynomial
        }

        const RESULT_SIZE: usize = N + M - 1;
        let mut result_coeffs = [T::zero(); RESULT_SIZE];

        // Standard polynomial multiplication algorithm (Cauchy product)
        for i in 0..N { // Iterate through terms of self
            for j in 0..M { // Iterate through terms of rhs
                // Coefficient of x^(i+j) in the product is (self.coeffs[i] * rhs.coeffs[j])
                result_coeffs[i + j] = result_coeffs[i + j] + self.coefficients[i] * rhs.coefficients[j];
            }
        }
        Polynomial { coefficients: result_coeffs }
    }
}

// MulAssign<Polynomial<T, M>>
// Note: MulAssign for polynomials is complex because the result often has a different degree.
// This means the array size changes. It's generally not recommended for `[T; N]` based polynomials
// unless you are willing to truncate or resize, which is outside the scope of fixed-size arrays.
// For a `Vec<T>` based polynomial, this would be easier.
// For fixed-size arrays, it implies `N` must be large enough to hold `N + M - 1` terms.
// We'll implement it by assigning the product, which means `N` must be the product's size.
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