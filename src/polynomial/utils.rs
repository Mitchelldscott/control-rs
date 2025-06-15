//! Provides core logic of a Polynomial without edge case handling and explicit return types.
//!
//! This is used by other parts of `control_rs` that perform their own specialized edge cases and
//! error handling. Users should call the provided Polynomial interface.

use core::{
    array, fmt, iter,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};
use nalgebra::{
    allocator::Allocator, ArrayStorage, Complex, Const, DefaultAllocator, DimAdd, DimDiff, DimMax,
    DimSub, RealField, SMatrix, U1,
};
use num_traits::{One, Zero};

/// Helper function to reverse arrays given to `Polynomial::new()`
#[inline]
pub const fn reverse_array<T: Copy, const N: usize>(input: [T; N]) -> [T; N] {
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
pub(super) fn array_from_iterator_with_default<I, T, const N: usize>(
    iterator: I,
    default: T,
) -> [T; N]
where
    T: Clone,
    I: IntoIterator<Item = T>,
{
    // SAFETY: `[MaybeUninit<T>; N]` is valid.
    let mut uninit_array: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut count = 0;

    // zip() will only iterate over `N.min(iterator_len)` items so no OOB access will occur.
    for (c, d) in uninit_array.iter_mut().zip(iterator.into_iter()) {
        *c = MaybeUninit::new(d);
        count += 1;
    }

    // `T: Clone`, and we are initializing all remaining uninitialized slots.
    for c in uninit_array.iter_mut().skip(count) {
        *c = MaybeUninit::new(default.clone());
    }

    // SAFETY:
    // * All `N` elements of `uninit_array` have been initialized.
    // * `MaybeUninit<T>` and T are guaranteed to have the same layout.
    // * `MaybeUninit` does not drop, so there are no double-frees.
    unsafe {
        // Get a pointer to the `uninit_array` array.
        // Cast it to a pointer to an array of `T`.
        // Then `read()` the value from that pointer.
        // This is equivalent to transmute from `[MaybeUninit<T>; N]` to `[T; N]`.
        uninit_array.as_ptr().cast::<[T; N]>().read()
    }
}

// /// # Safety
// ///
// /// This function performs a raw memory copy of `N` elements from the input slice `src`
// /// into a newly constructed `[T; N]` array using [`copy_nonoverlapping`].
// ///
// /// ## Preconditions
// /// To avoid undefined behavior, the caller **must** ensure the following:
// /// * `src.len() >= N`: The slice must contain **at least** `N` valid elements.
// /// * `src.as_ptr()` must be properly aligned for type `T` (This is guaranteed for slices of `T`,
// ///   unless manually constructed unsafely).
// #[inline]
// const unsafe fn array_from_slice_copy<T: Copy, const N: usize>(src: &[T]) -> [T; N] {
//     let mut dst: MaybeUninit<[T; N]> = MaybeUninit::uninit();
//     // Behavior is undefined if any of the following conditions are violated:
//     // * `src` must be [valid] for reads of `count * size_of::<T>()` bytes.
//     // * `dst` must be [valid] for writes of `count * size_of::<T>()` bytes.
//     // * Both `src` and `dst` must be properly aligned.
//     copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().cast::<T>(), N);
//     // SAFETY: All elements are initialized
//     dst.assume_init()
// }

/// Finds the **last** non-zero value in an array.
///
/// # Returns
/// * `Option<usize>`
///     * `Some(index)` - largets non-zero index
///     * `None` - if the length is zero or all elements are zero
///
/// # Example
/// ```
/// use control_rs::polynomial::utils::largest_nonzero_index;
/// assert_eq!(largest_nonzero_index::<u8>(&[]), None);
/// assert_eq!(largest_nonzero_index(&[1]), Some(0));
/// assert_eq!(largest_nonzero_index(&[0, 1]), Some(1));
/// assert_eq!(largest_nonzero_index(&[1, 0]), Some(0));
/// ```
#[inline]
pub fn largest_nonzero_index<T: Zero>(coefficients: &[T]) -> Option<usize> {
    for (i, element) in coefficients.iter().enumerate().rev() {
        if !element.is_zero() {
            return Some(i);
        }
    }
    None
}

/// Computes the derivative of a polynomial.
///
/// This function performs the core derivative calculation by applying the power rule
/// `d/dx(ax^n) = n * ax^{n-1}` to each term of the polynomial.
/// It iterates through the coefficients, effectively shifting them to a lower exponent
/// and multiplying by the original exponent.
///
/// # Examples
/// ```
/// use control_rs::polynomial::utils::differentiate;
/// assert_eq!(differentiate(&[0]), []); // d/dt(1) = 0 base case is an empty array
/// assert_eq!(differentiate(&[1, 2]), [2]); // d/dt(2x + 1) = 2
/// assert_eq!(differentiate(&[1, 1, 1]), [1, 2]); // d/dt(x^2 + x + 1) = 2x + 1
/// ```
#[inline]
pub fn differentiate<T, const N: usize, const M: usize>(coefficients: &[T; N]) -> [T; M]
where
    T: Clone + AddAssign + Zero + One,
    Const<N>: DimSub<U1, Output = Const<M>>,
{
    let mut exponent = T::zero();
    array_from_iterator_with_default(
        coefficients.iter().skip(1).map(|a_i| {
            exponent += T::one();
            a_i.clone() * exponent.clone()
        }),
        T::zero(),
    )
}

/// Computes the indefinite integral of a polynomial.
///
/// This internal function computes the indefinite integral of the polynomial by
/// applying the reverse power rule `\int ax^n dx = (a / n) * x^n` to each term.
/// It also incorporates the provided `constant` of integration as the new constant term.
///
/// The process involves:
/// 1. Prepending the `constant` of integration as the coefficient of `x^0`.
/// 2. For each `a_i`, calculating the new coefficient as `a_i = da_i / (i+1)`
///
/// # Example
/// ```
/// use control_rs::polynomial::utils::integrate;
/// assert_eq!(integrate(&[], 0), [0]); // d/dt(1) = 0 base case is an empty array
/// assert_eq!(integrate(&[2], 1), [1, 2]); // d/dt(2x + 1) = 2
/// assert_eq!(integrate(&[1, 2], 1), [1, 1, 1]); // d/dt(x^2 + x + 1) = 2x + 1
/// ```
#[inline]
pub fn integrate<T, const N: usize, const M: usize>(coefficients: &[T; N], constant: T) -> [T; M]
where
    T: Clone + AddAssign + Div<Output = T> + Zero + One,
    Const<N>: DimAdd<U1, Output = Const<M>>,
{
    let mut exponent = T::zero();
    array_from_iterator_with_default(
        iter::once(constant).chain(coefficients.iter().map(|a_i| {
            exponent += T::one();
            a_i.clone() / exponent.clone()
        })),
        T::zero(),
    )
}

/// Computes the Frobenius companion matrix of a polynomial.
///
/// The companion matrix is a square matrix whose eigenvalues are the roots of the polynomial.
/// It is constructed from an identity matrix, a zero column and a row of the polynomial's
/// coefficients scaled by the highest term.
///
/// **Note**: if the leading coefficient is zero, the result will be incorrect.
///
/// <pre>
/// companion(a_n * x^n + ... + a_1 * x + a_0) =
///     |  -a_(n-1)/a_n  -a_(n-2)/a_n  -a_(n-3)/a_n  ...  -a_1/a_0  -a_0/a_n |
///     |     1             0             0          ...     0         0     |
///     |     0             1             0          ...     0         0     |
///     |     0             0             1          ...     0         0     |
///     |    ...           ...           ...         ...    ...       ...    |
///     |     0             0             0          ...     1         0     |
/// </pre>
///
/// # Example
/// ```
/// use control_rs::polynomial::utils::companion;
/// let coefficients = [6, -7, 0, 1]; // p(x) = x^3 - 7x + 6
/// assert_eq!(companion(&coefficients), [[0, 7, -6], [1, 0, 0], [0, 1, 0]]);
/// ```
pub fn companion<T, const N: usize, const M: usize>(coefficients: &[T; N]) -> [[T; M]; M]
where
    T: Clone + Zero + One + Neg<Output = T> + Div<Output = T>,
    Const<N>: DimSub<U1, Output = Const<M>>,
{
    // assert!(coefficients.len() == M+1, "M is not a valid index to coefficients");
    let mut companion = array::from_fn(|_| array::from_fn(|_| T::zero()));

    // SAFETY: `M` is a valid index because Const<N> impl DimSub<U1, Output = Const<M>> so `N > 0`
    // and `N - 1 = M`
    let leading_coefficient_neg = unsafe { coefficients.get_unchecked(M).clone().neg() };
    if !leading_coefficient_neg.is_zero() {
        let mut companion_iter = companion.iter_mut();
        if let Some(row) = companion_iter.next() {
            for (companion_i, coefficient) in row.iter_mut().rev().zip(coefficients.iter()) {
                *companion_i = coefficient.clone() / leading_coefficient_neg.clone();
            }
            for (i, row) in companion_iter.enumerate() {
                row[i] = T::one();
            }
        }
    }
    companion
}

/// Temporary marker struct indicating that the roots function was not able to calculate a
/// solution.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct NoRoots;

/// Computes the roots of the polynomial.
/// # Errors
/// * `NoRoots` - the function was not able to compute roots for the given polynomial
///
/// # Example
/// ```
/// use nalgebra::Complex;
/// use control_rs::{polynomial::Polynomial, assert_f64_eq};
/// let p = Polynomial::new([1.0, -6.0, 11.0, -6.0]); // x^3 - 6x^2 + 11x - 6
/// let roots = p.roots().expect("Failed to calculate roots");
/// assert_f64_eq!(roots[0].re, 3.0, 1.5e-14); // having precision issues...
/// assert_f64_eq!(roots[1].re, 2.0, 1e-14);
/// assert_f64_eq!(roots[2].re, 1.0);
/// ```
/// TODO: Docs + Unit Tests + better return object + use recursion for leading zero cases
pub fn roots<T, const N: usize, const M: usize>(
    coefficients: &[T; N],
) -> Result<[Complex<T>; M], NoRoots>
where
    T: Clone
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
    DefaultAllocator: Allocator<Const<M>, DimDiff<Const<M>, U1>> + Allocator<DimDiff<Const<M>, U1>>,
{
    if let Some(degree) = largest_nonzero_index(coefficients) {
        let mut roots = array::from_fn(|_| Complex::zero());
        if degree == 0 {
            return Err(NoRoots);
        } else if degree == 1 {
            if coefficients[1].is_zero() {
                return Err(NoRoots);
            }
            roots[0].re = coefficients[0].clone().neg()
                / coefficients[1].clone();
        } else if degree == 2 {
            let a = coefficients[2].clone();
            if a.is_zero() {
                if coefficients[1].is_zero() {
                    return Err(NoRoots);
                }
                roots[0].re = coefficients[0].clone().neg()
                    / coefficients[1].clone();
            }
            let b = coefficients[1].clone();
            let ac = coefficients[0].clone() * a.clone();
            let discriminant = b.clone().powi(2) - (ac.clone() + ac.clone() + ac.clone() + ac);
            let two_a = a.clone() + a;
            let b_neg = b.neg();
            if discriminant < T::zero() {
                let real_part = b_neg / two_a.clone();
                let imag_part = (-discriminant).sqrt() / two_a;
                roots[0] = Complex { re: real_part.clone(), im: imag_part.clone() };
                roots[1] = Complex { re: real_part, im: imag_part.neg() };
            }
            else {
                let discriminant_sqrt = discriminant.sqrt();
                roots[0].re = (b_neg.clone() + discriminant_sqrt.clone()) / two_a.clone();
                roots[1].re = (b_neg - discriminant_sqrt) / two_a;
            }
        } else {
            let matrix = SMatrix::from_data(ArrayStorage(companion::<T, N, M>(coefficients)));
            for (eigenvalue, root) in matrix
                .complex_eigenvalues()
                .into_iter()
                .zip(roots.iter_mut())
            {
                *root = eigenvalue.clone();
            }
        }
        Ok(roots)
    } else {
        Err(NoRoots)
    }
}

/// Adds two polynomials.
///
/// This function does not put any requirements on the shape of the inputs and outputs
pub fn add_generic<T, const N: usize, const M: usize, const L: usize>(lhs: &[T; N], rhs: &[T; M]) -> [T; L]
where
    T: Clone + Add<Output = T> + Zero,
    Const<N>: DimMax<Const<M>, Output = Const<L>>,
{
    let mut result: [T; L] = array_from_iterator_with_default(lhs.iter().cloned(), T::zero());
    for (a, b) in result.iter_mut().zip(rhs.iter()) {
        *a = a.clone() + b.clone();
    }
    result
}

/// Subtracts two polynomials
///
/// This function does not put any requirements on the shape of the inputs and outputs
pub fn sub_generic<T, const N: usize, const M: usize, const L: usize>(lhs: &[T; N], rhs: &[T; M]) -> [T; L]
where
    T: Clone + Sub<Output = T> + Zero,
    Const<N>: DimMax<Const<M>, Output = Const<L>>,
{
    let mut result: [T; L] = array_from_iterator_with_default(lhs.iter().cloned(), T::zero());
    for (a, b) in result.iter_mut().zip(rhs.iter()) {
        *a = a.clone() - b.clone();
    }
    result
}

/// Multiplies two polynomials.
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
/// use control_rs::polynomial::utils::convolution;
/// let p1 = [1i32; 2];
/// let p2 = [1i32; 2];
/// let mut p3 = [0i32; 3];
/// assert_eq!(convolution(&p1, &p2), [1i32, 2i32, 1i32], "wrong convolution result");
/// ```
pub fn convolution<T, const N: usize, const M: usize, const L: usize>(
    lhs: &[T; N],
    rhs: &[T; M],
) -> [T; L]
where
    T: Clone + AddAssign + Mul<Output = T> + Zero,
    Const<N>: DimAdd<Const<M>>,
    <Const<N> as DimAdd<Const<M>>>::Output: DimSub<U1, Output = Const<L>>,
{
    let mut result = array::from_fn(|_| T::zero());
    for i in 0..N {
        for j in 0..M {
            // SAFETY: `i + j = (N - 1) + (M - 1) < L` is a valid index of [T; L]
            // SAFETY: `i < N` is a valid index of [T; N]
            // SAFETY: `j < M` is a valid index of [T; M]
            unsafe {
                *result.get_unchecked_mut(i + j) +=
                    lhs.get_unchecked(i).clone() * rhs.get_unchecked(j).clone();
            }
        }
    }
    result
}

/// Divides two polynomials.
///
/// The result is a polynomial with capacity `N`. This may be larger than the degree of the result,
/// in which case the higher order terms will be `T::zero()`.
///
/// # Example
/// ```rust
/// use control_rs::polynomial::utils::long_division;
/// let p1 = [1i32; 2];
/// let p2 = [1i32; 2];
/// assert_eq!(long_division(&p1, &p2), [1i32, 0i32], "wrong division result");
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
pub fn long_division<T, const N: usize, const M: usize>(
    dividend: &[T; N],
    divisor: &[T; M],
) -> [T; N]
where
    T: Clone + Zero + Div<Output = T> + Mul<Output = T> + AddAssign + SubAssign,
    Const<N>: DimMax<Const<M>, Output = Const<N>>,
{
    let mut quotient = array::from_fn(|_| T::zero());
    // Find actual degrees
    let dividend_order = largest_nonzero_index(dividend);
    let divisor_order = largest_nonzero_index(divisor);

    // degree of self and rhs exists
    if let Some(dividend_order) = dividend_order {
        if let Some(divisor_order) = divisor_order {
            let mut remainder = dividend.clone();
            // SAFETY: divisor_order is less than the capacity of divisor
            let leading_divisor = unsafe { divisor.get_unchecked(divisor_order) };

            for i in (divisor_order..=dividend_order).rev() {
                // SAFETY: index is less than the capacity of dividend, remainder has the same
                // capacity
                let rem_i = unsafe { remainder.get_unchecked(i) };
                if rem_i.is_zero() {
                    continue;
                }

                let q_index = i - divisor_order;
                let term_divisor = rem_i.clone() / leading_divisor.clone();
                // SAFETY: q_index is less than the capacity of dividend, quotient has the same
                // capacity
                unsafe {
                    *quotient.get_unchecked_mut(q_index) += term_divisor.clone();
                }
                for (rem, div) in remainder.iter_mut().skip(q_index).zip(divisor.iter()) {
                    *rem -= term_divisor.clone() * div.clone();
                }
            }
        }
    }
    quotient
}
