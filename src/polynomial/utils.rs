//! Provides core logic of a [Polynomial] without edge case handling and explicit return types.

use core::{
    fmt,
    mem::MaybeUninit,
    ops::{AddAssign, Div, Mul, Neg, Sub, SubAssign},
};
use nalgebra::{
    allocator::Allocator, ArrayStorage, Complex, Const, DefaultAllocator, DimAdd, DimDiff, DimMax,
    DimSub, RealField, SMatrix, U1,
};
use num_traits::{One, Zero};

/// Helper function to reverse arrays given to [Polynomial::new()]
#[inline(always)]
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
pub(super) fn initialize_array_from_iterator_with_default<I, T, const N: usize>(
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

/// Finds the **last** non-zero value in an array.
///
/// # Returns
/// * `Option<usize>`
///     * `Some(index)` - largets non-zero index
///     * `None` - if the length is zero or all elements are zero
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
/// // d/dt(x^2 + x + 1) = 2x + 1
/// assert_eq!(differentiate(&[1, 1, 1]), [1, 2]);
/// ```
#[inline]
pub fn differentiate<T, const N: usize, const M: usize>(coefficients: &[T; N]) -> [T; M]
where
    T: Clone + AddAssign + Zero + One,
    Const<N>: DimSub<U1, Output = Const<M>>,
{
    let mut exponent = T::zero();
    initialize_array_from_iterator_with_default(
        coefficients.iter().cloned().map(|a_i| {
            exponent += T::one();
            a_i * exponent.clone()
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
/// // d/dx(x^2 + x + 1) = 2x + 1
/// assert_eq!(integrate(&[1, 2], 1), [1, 1, 1]);
/// ```
#[inline]
pub fn integrate<T, const N: usize, const M: usize>(coefficients: &[T; N], constant: T) -> [T; M]
where
    T: Clone + AddAssign + Div<Output = T> + Zero + One,
    Const<N>: DimAdd<U1, Output = Const<M>>,
{
    let mut exponent = T::zero();
    initialize_array_from_iterator_with_default(
        [constant]
            .into_iter()
            .chain(coefficients.iter().enumerate().map(|(_, a_i)| {
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
/// let coefficients = [20.0, -14.0, 2.0]; // p(x) = 2x^2 - 14x + 20
/// assert_eq!(companion(&coefficients), [[7.0, -10.0], [1.0, 0.0]]);
/// ```
pub fn companion<T, const N: usize, const M: usize>(coefficients: &[T; N]) -> [[T; M]; M]
where
    T: Copy + Zero + One + Neg<Output = T> + Div<Output = T>,
    Const<N>: DimSub<U1, Output = Const<M>>,
{
    let mut companion = [[T::zero(); M]; M];

    // SAFETY: `N - 1` is a valid index if Const<N> impl DimSub<U1> `N > 0`
    let leading_coefficient = unsafe { coefficients.get_unchecked(M).clone() };
    if M > 0 && !leading_coefficient.is_zero() {
        for i in 0..M {
            // SAFETY: `i < M` i is a valid index for either dimension of companion
            // SAFETY: `M < N` i is a valid index of coefficients
            unsafe {
                *companion.get_unchecked_mut(0).get_unchecked_mut(i) =
                    -coefficients.get_unchecked(M - i).clone() / leading_coefficient.clone();
            }
        }
        for i in 1..M {
            // SAFETY: `0 < i < M` i is a valid index for either dimension of companion
            unsafe {
                *companion.get_unchecked_mut(i).get_unchecked_mut(i - 1) = T::one();
            }
        }
    }
    companion
}

/// Temporary marker struct indicating that the roots function was not able to calculate a
/// solution.
pub struct NoRoots;

/// Computes the roots of the polynomial.
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p = Polynomial::new([1.0, -6.0, 11.0, -6.0]); // x^3 - 6x^2 + 11x - 6
/// // assert_eq!(p.roots(), Ok([1.0, 1.0, 1.0]), "Incorrect polynomial roots");
/// ```
pub fn roots<T, const N: usize, const M: usize>(
    coefficients: &[T; N],
) -> Result<[Complex<T>; M], NoRoots>
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
    DefaultAllocator: Allocator<Const<M>, DimDiff<Const<M>, U1>> + Allocator<DimDiff<Const<M>, U1>>,
{
    if let Some(degree) = largest_nonzero_index(coefficients) {
        let mut roots = [Complex::zero(); M];
        if degree == 0 {
            return Err(NoRoots);
        } else if degree == 1 {
            // SAFETY: the array has a non-zero element at 1 so 1 and 0 are valid indices
            unsafe {
                roots.get_unchecked_mut(0).re =
                    coefficients.get_unchecked(0).neg() / *coefficients.get_unchecked(1);
            }
        } else if degree == 2 {
            let a = unsafe { coefficients.get_unchecked(2).clone() };
            let b = unsafe { coefficients.get_unchecked(1).clone() };
            let c = unsafe { coefficients.get_unchecked(0).clone() };
            let discriminant = -(0..4).fold(b * b, |acc, _| acc - (a.clone() * c.clone()));
            if discriminant < T::zero() {
                return Err(NoRoots);
            } else {
                // SAFETY: the array has a non-zero element at 2 so 2, 1 and 0 are valid indices
                let root =
                    b * discriminant.clone().sqrt() / (0..1).fold(a.clone(), |acc, _| acc + a);
                unsafe {
                    roots.get_unchecked_mut(0).re = root.clone().neg();
                    roots.get_unchecked_mut(1).re = root;
                }
            }
        } else {
            let matrix = SMatrix::from_data(ArrayStorage(companion::<T, N, M>(coefficients)));
            for (eigenvalue, root) in matrix.complex_eigenvalues().iter().zip(roots.iter_mut()) {
                *root = *eigenvalue;
            }
        }
        Ok(roots)
    } else {
        Err(NoRoots)
    }
}

/// Adds two polynomials.
///
/// This requires that the left-hand side has a larger or equal capacity.
pub fn add_generic<T, const N: usize, const M: usize>(lhs: &[T; N], rhs: [T; M]) -> [T; N]
where
    T: Clone + AddAssign,
    Const<N>: DimMax<Const<M>, Output = Const<N>>,
{
    let mut result = lhs.clone();
    add_assign_generic(&mut result, rhs);
    result
}

/// Adds two polynomials storing the result in the left polynomial.
///
/// This requires that the left-hand side has a larger or equal capacity.
pub fn add_assign_generic<T, const N: usize, const M: usize>(lhs: &mut [T; N], rhs: [T; M])
where
    T: Clone + AddAssign,
    Const<N>: DimMax<Const<M>, Output = Const<N>>,
{
    for (a, b) in lhs.iter_mut().zip(rhs.iter()) {
        *a += b.clone();
    }
}

/// Subtracts two polynomials
///
/// This requires that the left-hand side has a larger or equal capacity.
pub fn sub_generic<T, const N: usize, const M: usize>(lhs: &[T; N], rhs: [T; M]) -> [T; N]
where
    T: Clone + SubAssign,
    Const<N>: DimMax<Const<M>, Output = Const<N>>,
{
    let mut result = lhs.clone();
    sub_assign_generic(&mut result, rhs);
    result
}

/// Subtracts two polynomials
///
/// This requires that the left-hand side has a larger or equal capacity.
pub fn sub_assign_generic<T, const N: usize, const M: usize>(lhs: &mut [T; N], rhs: [T; M])
where
    T: Clone + SubAssign,
    Const<N>: DimMax<Const<M>, Output = Const<N>>,
{
    for (a, b) in lhs.iter_mut().zip(rhs.iter()) {
        *a -= b.clone();
    }
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
/// use control_rs::polynomial::utils::mul_generic;
/// let p1 = [1i32; 2];
/// let p2 = [1i32; 2];
/// let mut p3 = [0i32; 3];
/// mul_generic(&mut p3, &p1, p2);
/// assert_eq!(p3, [1i32, 1i32, 1i32], "wrong multiplication result");
/// ```
pub fn mul_generic<T, const N: usize, const M: usize, const L: usize>(
    result: &mut [T; L],
    lhs: &[T; N],
    rhs: [T; M],
) where
    T: Clone + Zero + AddAssign + Mul<Output = T>,
    Const<N>: DimAdd<Const<M>>,
    <Const<N> as DimAdd<Const<M>>>::Output: DimSub<U1, Output = Const<L>>,
{
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
}

/// Divides two polynomials.
///
/// The result is a polynomial with capacity `N`. This may be larger than the degree of the result,
/// in which case the higher order terms will be [T::zero()].
///
/// # Example
/// ```rust
/// use control_rs::polynomial::utils::div_generic;
/// let p1 = [1i32; 2];
/// let p2 = [1i32; 2];
/// let mut p3 = [0i32; 2];
/// div_generic(&mut p3, &p1, &p2);
/// assert_eq!(p3, [1i32, 1i32], "wrong division result");
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
pub fn div_generic<T, const N: usize, const M: usize>(
    quotient: &mut [T; N],
    numerator: &[T; N],
    denominator: &[T; M],
) where
    T: Clone + Zero + Div<Output = T> + Mul<Output = T> + AddAssign + SubAssign,
    Const<N>: DimMax<Const<M>, Output = Const<N>>,
{
    // Find actual degrees
    let numerator_order = largest_nonzero_index(numerator);
    let denominator_order = largest_nonzero_index(denominator);

    // degree of self and rhs exists
    if let Some(numerator_order) = numerator_order {
        if let Some(denominator_order) = denominator_order {
            let mut remainder = numerator.clone();
            // SAFETY: denominator_order is less than the capacity of denominator
            let leading_denominator = unsafe { denominator.get_unchecked(denominator_order) };

            for i in (denominator_order..=numerator_order).rev() {
                // SAFETY: index is less than the capacity of numerator, remainder has the same
                // capacity
                let rem_i = unsafe { remainder.get_unchecked(i) };
                if rem_i.is_zero() {
                    continue;
                }

                let q_index = i - denominator_order;
                let term_denominator = rem_i.clone() / leading_denominator.clone();
                // SAFETY: q_index is less than the capacity of numerator, quotient has the same
                // capacity
                unsafe {
                    *quotient.get_unchecked_mut(q_index) += term_denominator.clone();
                }
                for (rem, div) in remainder.iter_mut().skip(q_index).zip(denominator.iter()) {
                    *rem -= term_denominator.clone() * div.clone();
                }
            }
        }
    }
}
