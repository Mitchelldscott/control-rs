//! A safe and statically sized univariate-polynomial
//! TODO:
//!     * calculus
//!         * evaluate<U>(x: U) -> U
//!         * derivative(p_src: &Polynomial<T, M>) -> Self
//!         * integral(p_src: &Polynomial<T, M>) -> Self
//!         * foil_roots(&mut self, roots: &[T]) -> Result<(), Polynomial>
//!         * foil_complex_roots(&mut self, roots: &[Complex<T>]) -> Result<(), Polynomial>
//!         * real_roots(p: Polynomial, roots: &mut [T]) -> Result<(), PolynomialError>
//!         * complex_roots(p: Polynomial, roots: &mut [Complex<T>]) -> Result<(), PolynomialError>
//!     * formatting
//!         * Display with precision option
//!         * Latex / symbolic formatter (optional)

#[cfg(feature = "std")]
use std::{mem::MaybeUninit, ops::{Add, Sub, Mul, MulAssign, Div, DivAssign, Neg, Rem, RemAssign}, array};

#[cfg(not(feature = "std"))]
use core::{mem::MaybeUninit, ops::{Add, Sub, Mul, MulAssign, Div, DivAssign, Neg, Rem, RemAssign}, array};

use num_traits::{Zero, One};

// ===============================================================================================
//      Polynomial Specializations
// ===============================================================================================

pub mod constant;
pub use constant::Constant;

mod line;
// pub use line::Line;

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
/// * `iterator` - An [Iterator] over a collection of `T`
/// * `default` - the default value to use if the iterator is not long enough
///
/// # Returns
/// * `initialized_array` - An array with all elements initialized
///
/// # Safety
/// This function uses `MaybeUninit` and raw pointer casting to avoid requiring `T: Default + Copy`.
/// The safety relies on:
/// - Fully initializing all elements of the `[MaybeUninit<T>; N]` array before calling `read()`
/// - Not reading from or dropping uninitialized memory
fn initialize_array_from_iterator_with_default<I, T, const N: usize>(iterator: I, default: T) -> [T; N]
where
    T: Clone,
    I: IntoIterator<Item = T>,
{
    // SAFETY: `[MaybeUninit<T>; N]` is valid.
    let mut uninit_array: [MaybeUninit<T>; N] =
        unsafe { MaybeUninit::uninit().assume_init() };
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
/// # TODO: Demo + Integration Test
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Polynomial<T, const N: usize> {
    /// coefficients of the polynomial `[a_0, a_1 ... a_n]`, in degree-minor order
    /// (lowest to highest)
    coefficients: [T; N],
}

impl<T, const N: usize> Polynomial<T, N> {
    /// Creates a new polynomial from an array of coefficients.
    ///
    /// This function assumes the array is sorted in degree-minor order (i.e. `[a_0, a_1 ... a_n]`,
    /// where `a_0` is the constant and `a_n` the nth degree term).
    ///
    /// # Arguments
    /// * `coefficients` - An array of coefficients
    ///
    /// # Returns
    /// * `polynomial` - A polynomial with the given coefficients
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
    /// * `cb` - The generator function, which takes the degree (`usize`) as input and returns the
    /// coefficient (`T`) for that degree
    ///
    /// # Returns
    /// * `polynomial` - A new instance with the generated coefficients
    ///
    /// # Example
    /// ```
    /// use control_rs::Polynomial;
    /// // creates a quadratic equation
    /// let p: Polynomial<i32, 3> = Polynomial::from_fn(|i| 1);
    /// assert!(p.is_monic(), "Quadratic is not monic");
    /// assert_eq!(p.degree(), Some(2), "Quadratic degree was not 2");
    /// ```
    #[inline]
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> T
    {
        Self::from_data(array::from_fn(cb))
    }

    /// Checks if the capacity is zero
    ///
    /// # Example
    ///
    /// ```
    /// use control_rs::Polynomial;
    /// // creates a quadratic equation
    /// let p = Polynomial::new([]);
    /// assert_eq!(p.is_empty(), true);
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool { N == 0 }
}

impl<T: Clone + Zero, const N: usize> Polynomial<T, N> {
    /// Creates a new polynomial from an [Iterator].
    ///
    /// If the iterator has more items than N, the trailing items will be ignored. If the iterator
    /// has less than N items, the remaining indices will be filled with zeros.
    ///
    /// # Arguments
    /// * `iterator` - [Iterator] over items of `T`
    ///
    /// # Returns
    /// * `polynomial` - A zero-padded polynomial with the given coefficients
    ///
    /// ```
    /// use control_rs::Polynomial;
    /// let p = Polynomial::from_iterator([1, 2, 3, 4, 5]);
    /// assert_eq!(p.degree(), Some(4));
    /// ```
    #[inline]
    pub fn from_iterator<I>(iterator: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_data(initialize_array_from_iterator_with_default(iterator, T::zero()))
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
    /// * `resized_polynomial` - a polynomial with capacity `N`
    ///
    /// TODO: Unit Test + Example
    #[inline]
    pub fn resize<const M: usize>(other: Polynomial<T, M>) -> Self {
        Self::from_iterator(other.coefficients)
    }

    /// Creates a monomial from a given constant.
    ///
    /// A monomial consists of a single non-zero leading coefficient. This is implemented by
    /// creating a zero polynomial with the specified size and setting the final element to the
    /// given constant.
    ///
    /// # Returns
    /// * `Polynomial` - a polynomial with only the trailing term
    /// TODO: Cleanup + Example
    #[inline]
    pub fn monomial(coefficient: T) -> Self {
        let mut polynomial = Self::from_fn(|_| T::zero());
        if let Some(a_n) = polynomial.leading_coefficient_mut() { *a_n = coefficient }
        polynomial
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
    /// TODO: Example
    #[inline]
    pub const fn from_element(element: T) -> Self {
        Self::from_data([element; N])
    }

    /// Creates a new polynomial from an array.
    ///
    /// Expects an array of coefficients sorted highest to lowest degree .
    ///
    /// # Arguments
    /// * `coefficients` - An array of coefficients in descending degree order `[a_n, ... a_1, a_0]`
    ///
    /// # Returns
    /// * `polynomial` - polynomial with the given coefficients
    ///
    /// TODO: Unit Test + Example
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
    /// The degree is found by iterating through the array of coefficients, from 0 to N, and
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
    /// TODO: Unit Test
    #[inline]
    pub fn evaluate<U>(&self, value: U) -> U
    where
        U: Clone + Zero + Add<T, Output = U> + Mul<U, Output = U>,
    {
        self.coefficients.iter().rfold(U::zero(), |acc, a_i| acc * value.clone() + a_i.clone())
    }
}
impl<T: PartialEq + One, const N: usize> Polynomial<T, N> {
    /// Checks if a polynomial is monic
    ///
    /// # Returns
    /// * `bool` - true if the leading coefficient is one, false otherwise
    #[inline]
    pub fn is_monic(&self) -> bool {
        if self.is_empty() { false }
        else {
            // SAFETY: N > 0 so N-1 is valid
            unsafe { self.get_unchecked(N-1).is_one() }
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
    /// * `index` - the degree of the coefficient to return
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
    /// TODO: Unit Test + Example
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
    /// TODO: Example
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
    /// TODO: Unit Test + Example
    #[inline]
    #[must_use]
    pub fn leading_coefficient(&self) -> Option<&T> {
        if self.is_empty() { None }
        else {
            // SAFETY: N > 0 so N-1 is valid
            unsafe{
                Some(self.get_unchecked(N-1))
            }
        }
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
    /// TODO: Unit Test + Example
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
    /// TODO: Unit Test + Example
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
    /// TODO: Unit Test + Example
    #[inline]
    #[must_use]
    pub fn leading_coefficient_mut(&mut self) -> Option<&mut T> {
        if self.is_empty() { None }
        else {
            // SAFETY: N > 0 so N-1 is valid
            unsafe{
                Some(self.get_unchecked_mut(N-1))
            }
        }
    }
}

// ===============================================================================================
//      Polynomial-Scalar Arithmatic
// ===============================================================================================

/// Implementation of [Neg] for Polynomial<T, N>
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

/// Implementation of [Add] for Polynomial<T, 0> and T
///
/// This is a base case implementation where a scalar is added to an empty polynomial. The result
/// is a polynomial with a constant term equal to the scalar.
///
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([]);
/// let p2 = p1 + 1;
/// assert_eq!(*p2.constant().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T> Add<T> for Polynomial<T, 0> {
    type Output = Polynomial<T, 1>;

    /// Returns a new polynomial with the given constant term
    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Polynomial::from_data([rhs])
    }
}


/// Implementation of [Sub] for Polynomial<T, 0> and T
///
/// This is a base case implementation where a scalar is subtracted from an empty polynomial. The
/// result is a polynomial with a constant term equal to 0 - the scalar.
///
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([]);
/// let p2 = p1 - 1;
/// assert_eq!(*p2.constant().unwrap(), -1);
/// ```
/// TODO: Unit Test
impl<T: Zero + Sub<Output = T>> Sub<T> for Polynomial<T, 0> {
    type Output = Polynomial<T, 1>;

    /// Returns a new polynomial with the constant term equal to 0 - the scalar
    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Self::Output::from_data([T::zero() - rhs])
    }
}

/// Implementation of [Mul] for Polynomial<T, N> and T
///
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([]);
/// let p2 = p1 * 1;
/// assert_eq!(*p2.constant(), None);
/// ```
/// TODO: Unit Test
impl<T: Clone + Mul<Output = T>, const N: usize> Mul<T> for Polynomial<T, N> {
    type Output = Self;

    /// Returns a new polynomial with all coefficients scaled by rhs
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::from_fn(|i| {
            // SAFETY: The index is usize (>= 0) and less than N
            unsafe {
                self.get_unchecked(i).clone() * rhs.clone()
            }
        })
    }
}

/// Implementation of [MulAssign] for Polynomial<T, N> and T
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([]);
/// p1 *= 2;
/// assert_eq!(*p1.constant(), None);
/// ```
/// TODO: Unit Test
impl<T: Clone + MulAssign, const N: usize> MulAssign<T> for Polynomial<T, N> {
    fn mul_assign(&mut self, rhs: T) {
        for a_i in self.coefficients.iter_mut() {
            *a_i *= rhs.clone();
        }
    }
}

/// Implementation of [Div] for Polynomial<T, N> and T
///
/// ```
/// use control_rs::Polynomial;
/// let p1 = Polynomial::new([]);
/// let p2 = p1 / 1;
/// assert_eq!(*p2.constant(), None);
/// ```
/// TODO: Unit Test
impl<T: Clone + Div<Output = T>, const N: usize> Div<T> for Polynomial<T, N> {
    type Output = Self;

    /// Returns a new polynomial with all coefficients scaled by 1 / rhs
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::Output::from_fn(|i| {
            // SAFETY: The index is usize (>= 0) and less than N
            unsafe {
                self.get_unchecked(i).clone() / rhs.clone()
            }
        })
    }
}

/// Implementation of [DivAssign] for Polynomial<T, N> and T
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([]);
/// p1 /= 2;
/// assert_eq!(*p1.constant(), None);
/// ```
/// TODO: Unit Test
impl<T: Clone + DivAssign, const N: usize> DivAssign<T> for Polynomial<T, N> {
    fn div_assign(&mut self, rhs: T) {
        for a_i in self.coefficients.iter_mut() {
            *a_i /= rhs.clone();
        }
    }
}

/// Implementation of [Rem] for Polynomial<T, 1> and T
///
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
            }
        )
    }
}

/// Implementation of [RemAssign] for Polynomial<T, 1> and T
///
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