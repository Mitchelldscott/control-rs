//! Provides core logic of a Polynomial without edge case handling and explicit return types.
//!
//! This is used by other parts of `control_rs` that perform their own specialized edge cases and
//! error handling. Users should call the provided Polynomial interface.

use core::{
    array, fmt, iter,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};
use nalgebra::{
    allocator::Allocator, ArrayStorage, Complex, Const, DefaultAllocator, DimAdd, DimDiff, DimMax,
    DimSub, RealField, SMatrix, U1,
};
use num_traits::{Float, One, Zero};

use crate::static_storage::{array_from_iterator, array_from_iterator_with_default};

/// Finds the largest index of a non-zero value in a slice.
///
/// This function iterates through a slice in reverse and checks if each value `.is_zero()`. It
/// will return after the first time the condition is **false**.
///
/// <div class="warning">
///
/// The logical correctness of the result is dependent on the correctness of `.is_zero()`.
///
/// </div>
/// <div class="warning">
///
/// If the slice has more elements than `usize::MAX`, the result will be incorrect (this should not
/// be possible without heap allocation).
///
/// </div>
///
/// # Generic Arguments
/// * `T` - field type of the array, which must implement [Zero].
///
/// # Arguments
/// * `coefficients` - a slice of `T`.
///
/// # Returns
/// * `Option<usize>`
///     * `Some(index)` - The largest index containing a non-zero value.
///     * `None` - If the slice is empty or all elements are zero.
///
/// # Panics
/// This function does not panic.
///
/// # Safety
/// This function does not use `unsafe` code.
///
/// # Example
/// ```
/// use control_rs::polynomial::utils::largest_nonzero_index;
/// assert_eq!(largest_nonzero_index::<u8>(&[]), None);
/// assert_eq!(largest_nonzero_index(&[0, 1]), Some(1));
/// assert_eq!(largest_nonzero_index(&[1, 0]), Some(0));
/// assert_eq!(largest_nonzero_index(&[0, 0]), None);
/// assert_eq!(largest_nonzero_index(&[1]), Some(0));
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
///
/// <div class="warning">
///
/// If `N == 1`, the result will be an empty array, even if the element is non-zero. This may be
/// unexpected.
///
/// </div>
/// <div class="warning">
///
/// If `T` cannot represent up-to `N`, there will be an overflow calculating the exponent that
/// scales the elements of the source array ([`DimSub`] is only implemented for `Const<N> < 128`).
///
/// </div>
///
/// # Generic Arguments
/// * `T` - Field type of the coefficients.
/// * `N` - Capacity of the source coefficients array (restricted to the range [1..127].
/// * `M` - Capacity of the destination derivative array.
///
/// # Arguments
/// * `coefficients` - A degree minor polynomial array.
///
/// # Returns
/// * `derivative` - A degree minor polynomial array with length `N-1`.
///
/// # Panics
/// This function does not panic.
///
/// # Safety
/// This function uses an unsafe block to initialize the derivative array from an iterator. The
/// length of the new array is inferred from the return type which is guaranteed to be equal to
/// `N-1`. The iterator given to the initializing function has `N` items and will skip the first
/// item, so its length is also guaranteed to be `N-1`.
///
/// # Examples
/// ```
/// use control_rs::polynomial::utils::differentiate;
/// assert_eq!(differentiate([1u8]), [0u8; 0]); // derivative of constants is an empty array
/// assert_eq!(differentiate([1, 2]), [2]); // d/dt(2x + 1) = 2
/// assert_eq!(differentiate([1, 1, 1]), [1, 2]); // d/dt(x^2 + x + 1) = 2x + 1
/// ```
#[inline]
pub fn differentiate<T, const N: usize, const M: usize>(coefficients: [T; N]) -> [T; M]
where
    T: Clone + AddAssign + Zero + One,
    Const<N>: DimSub<U1, Output = Const<M>>,
{
    let mut exponent = T::zero();
    // Safety: the iterator will have 1 less element than coefficients, which has more than 0
    // elements because Const<N> impls DimSub
    unsafe {
        array_from_iterator(coefficients.into_iter().skip(1).map(|a_i| {
            exponent += T::one();
            a_i * exponent.clone()
        }))
    }
}

/// Computes the indefinite integral of a polynomial.
///
/// This internal function computes the indefinite integral of the polynomial by
/// applying the reverse power rule `∫ax^n dx = a * x^(n+1) / (n+1)` to each term.
/// It incorporates the provided constant of integration as the new constant term (x^0).
///
/// <div class="warning">
///
/// If `T` cannot represent up-to `N`, there will be an overflow calculating the exponent that
/// scales the elements of the source array ([`DimSub`] is only implemented for `Const<N> < 128`).
///
/// </div>
///
/// # Generic Arguments
/// * `T` - The field type of the coefficients.
/// * `N` - The capacity of the source coefficients array.
/// * `M` - The capacity of the destination integral array.
///
/// # Arguments
/// * `coefficients` - A degree-minor polynomial array.
/// * `constant` - The constant of integration.
///
/// # Returns
/// * `integral` - a degree minor polynomial array with length `N+1`.
///
/// # Panics
/// This function does not panic.
///
/// # Safety
/// This function uses an unsafe block to initialize the new array from an iterator. The length of
/// the new array is inferred from the return type which is guaranteed to be `N+1`. The iterator
/// given to the initializer is created by chaining a single item (the constant) to the source
/// array, this is guaranteed to have length `N+1`.
///
/// # Example
/// ```
/// use control_rs::polynomial::utils::integrate;
/// assert_eq!(integrate([], 1), [1]); // d/dt(1) = [] base case is an empty array
/// assert_eq!(integrate([2], 1), [1, 2]); // d/dt(2x + 1) = 2
/// assert_eq!(integrate([1, 2], 1), [1, 1, 1]); // d/dt(x^2 + x + 1) = 2x + 1
/// ```
#[inline]
pub fn integrate<T, const N: usize, const M: usize>(coefficients: [T; N], constant: T) -> [T; M]
where
    T: Clone + AddAssign + Div<Output = T> + Zero + One,
    Const<N>: DimAdd<U1, Output = Const<M>>,
{
    let mut exponent = T::zero();
    // Safety: the iterator will have 1 more element than coefficients
    unsafe {
        array_from_iterator(
            iter::once(constant).chain(coefficients.into_iter().map(|a_i| {
                exponent += T::one();
                a_i / exponent.clone()
            })),
        )
    }
}

/// Computes the Frobenius companion matrix of a polynomial.
///
/// The companion matrix is a square matrix whose eigenvalues are the roots of the polynomial.
/// It is constructed from an identity matrix, a zero column and a row of the polynomial's
/// coefficients scaled by the leading coefficient.
///
/// <pre>
/// companion(a_n * x^n + ... + a_1 * x + a_0) =
///     |  -a_(n-1)/a_n -a_(n-2)/a_n -a_(n-3)/a_n ...  -a_1/a_0  -a_0/a_n |
///     |    1            0            0          ...     0         0     |
///     |    0            1            0          ...     0         0     |
///     |    0            0            1          ...     0         0     |
///     |   ...          ...          ...         ...    ...       ...    |
///     |    0            0            0          ...     1         0     |
/// </pre>
///
/// <div class="warning">
///
/// The polynomial is assumed to not have any leading zeros. If it does, the first row of the
/// companion will be all zeros.
///
/// </div>
///
/// # Generic Arguments
/// * `T` - Field type of the array and companion matrix.
/// * `N` - Capacity of the coefficient array.
/// * `M` - Number of rows and columns of the companion matrix.
///
/// # Arguments
/// * `coefficients` - The degree minor polynomial array.
///
/// # Returns
/// * `companion` - An array of arrays, both dimensions equal to `N-1`.
///
/// # Panics
/// The function does not panic.
///
/// # Safety
/// This function makes an unsafe call to access the leading coefficient of the polynomial. The
/// polynomial is assumed to not have any leading zero coefficients and so the leading coefficient
/// is at the largest index `N-1`. It is guaranteed that `M=N-1` so `M` is a safe index.
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
    let mut companion = array::from_fn(|_| array::from_fn(|_| T::zero()));

    // SAFETY: `M` is a valid index because Const<N> impl DimSub<U1, Output = Const<M>> so `N > 0`
    // and `N - 1 = M`
    let leading_coefficient_neg = unsafe { coefficients.get_unchecked(M).clone().neg() };
    let mut companion_row_iter = companion.iter_mut();
    if let Some(first_row) = companion_row_iter.next() {
        if !leading_coefficient_neg.is_zero() {
            for (companion_i, coefficient) in first_row.iter_mut().rev().zip(coefficients.iter()) {
                *companion_i = coefficient.clone() / leading_coefficient_neg.clone();
            }
        }
        for (i, row) in companion_row_iter.enumerate() {
            // this could also be a get_unchecked(), r == c, i < r.
            row[i] = T::one();
        }
    }
    companion
}

/// Temporary marker struct indicating that the roots function was not able to calculate a
/// solution.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct NoRoots;

/// Computes the root of a line.
///
/// This function calculates the x intercept of the standard slope intercept form `y=m*x+b`.
///
/// # Generic Arguments
/// * `T` - Field type of the line's coefficients.
///
/// # Arguments
/// * `m` - slope of the line.
/// * `b` - offset of the line.
///
/// # Returns
/// * `Result`
///     * `Ok(root)` - The x-intercept of the line.
///     * `NoRoots` - The slope is 0 or undefined, so there is no intercept.
///
/// # Errors
/// * `NoRoots` - the function was not able to compute a solution for the line
///
/// # Example
/// ```
/// use control_rs::{polynomial::utils::x_intercept, assert_f64_eq};
/// assert_eq!(x_intercept(1, 0), Ok(0));
/// ```
pub fn x_intercept<T: Zero + Neg<Output = T> + Div<Output = T>>(m: T, b: T) -> Result<T, NoRoots> {
    Ok(b.neg() / m)
}

/// Computes the root of a quadratic.
///
/// # Errors
/// * `NoRoots` - the function was not able to compute a solution for the quadratic
///
/// # Example
/// ```
/// use control_rs::{polynomial::utils::quadratic_roots, assert_f64_eq};
/// let roots = quadratic_roots(1.0, 0.0, 0.0).expect("failed to compute roots");
/// assert_f64_eq!(roots[0].re, 0.0, 1.5e-14); // having precision issues...
/// assert_f64_eq!(roots[1].re, 0.0, 1e-14);
/// ```
/// TODO: Fixed Point support
pub fn quadratic_roots<T>(a: T, b: T, c: T) -> Result<[Complex<T>; 2], NoRoots>
where
    T: Clone
        + PartialOrd
        + Zero
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>
        + RealField,
{
    let ac = a.clone() * c;
    let discriminant = (b.clone() * b.clone()) - (ac.clone() + ac.clone() + ac.clone() + ac);
    let two_a = a.clone() + a;
    let b_neg = b.neg();
    if discriminant < T::zero() {
        let real_part = b_neg / two_a.clone();
        let imag_part = (-discriminant).sqrt() / two_a;
        Ok([
            Complex {
                re: real_part.clone(),
                im: imag_part.clone(),
            },
            Complex {
                re: real_part,
                im: imag_part.neg(),
            },
        ])
    } else {
        let discriminant_sqrt = discriminant.sqrt();
        Ok([
            Complex {
                re: (b_neg.clone() + discriminant_sqrt.clone()) / two_a.clone(),
                im: T::zero(),
            },
            Complex {
                re: (b_neg - discriminant_sqrt) / two_a,
                im: T::zero(),
            },
        ])
    }
}

/// Computes the roots of the polynomial.
///
/// This function returns an array of `Complex<T>` that may have a larger capacity than the
/// polynomial has roots. By default elements of the returned array have a real and imaginary part
/// set to NaN. This prevents the function from being available for integer types.
///
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
/// TODO: fixed point support
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
        + RealField
        + Float,
    Const<N>: DimSub<U1, Output = Const<M>>,
    Const<M>: DimSub<U1>,
    DefaultAllocator: Allocator<Const<M>, DimDiff<Const<M>, U1>> + Allocator<DimDiff<Const<M>, U1>>,
{
    let Some(degree) = largest_nonzero_index(coefficients) else {
        return Err(NoRoots);
    };

    let mut roots = array::from_fn(|_| Complex::new(T::nan(), T::nan()));

    if degree == 0 {
        return Err(NoRoots);
    } else if degree == 1 {
        roots[0].re = x_intercept(coefficients[1], coefficients[0])?;
        roots[0].im = T::zero();
    } else if degree == 2 {
        let a = coefficients[2];
        if a.is_zero() {
            roots[0].re = x_intercept(coefficients[1], coefficients[0])?;
        } else {
            for (q_root, root) in quadratic_roots(a, coefficients[1], coefficients[0])?
                .into_iter()
                .zip(roots.iter_mut())
            {
                *root = q_root;
            }
        }
    } else {
        // TODO:
        //  * Edge-case where coefficients represent a monomial
        //  * Edge-case where polynomial has no constant (root at zero, can deflate)
        //  * Edge-case where polynomial has a leading zero, companion will have a row of zeros
        let companion = if degree == M {
            companion(coefficients)
        } else {
            let mut shifted_coefficients = [T::zero(); N];
            for i in 0..=degree {
                shifted_coefficients[M - i] = coefficients[degree - i];
            }
            companion(&shifted_coefficients)
        };
        let matrix = SMatrix::from_data(ArrayStorage(companion));
        for (eigenvalue, root) in matrix
            .complex_eigenvalues()
            .into_iter()
            .zip(roots.iter_mut())
            .take(degree)
        {
            *root = *eigenvalue;
        }
    }
    Ok(roots)
}

/// Adds two polynomials.
///
/// This function takes ownership of the given arrays
pub fn add_generic<T, const N: usize, const M: usize, const L: usize>(
    lhs: [T; N],
    rhs: [T; M],
) -> [T; L]
where
    T: Clone + Add<Output = T> + Zero,
    Const<N>: DimMax<Const<M>, Output = Const<L>>,
{
    let mut result: [T; L] = array_from_iterator_with_default(lhs, T::zero());
    for (a, b) in result.iter_mut().zip(rhs.into_iter()) {
        *a = a.clone().add(b);
    }
    result
}

/// Subtracts two polynomials
///
/// This function takes ownership of the given arrays
pub fn sub_generic<T, const N: usize, const M: usize, const L: usize>(
    lhs: [T; N],
    rhs: [T; M],
) -> [T; L]
where
    T: Clone + Sub<Output = T> + Zero,
    Const<N>: DimMax<Const<M>, Output = Const<L>>,
{
    let mut result: [T; L] = array_from_iterator_with_default(lhs, T::zero());
    for (a, b) in result.iter_mut().zip(rhs.into_iter()) {
        *a = a.clone().sub(b);
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
/// assert_eq!(long_division(p1, &p2), [1i32, 0i32], "wrong division result");
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
    dividend: [T; N],
    divisor: &[T; M],
) -> [T; N]
where
    T: Clone + Zero + Div<Output = T> + Mul<Output = T> + AddAssign + SubAssign,
    Const<N>: DimMax<Const<M>, Output = Const<N>>,
{
    let mut quotient = array::from_fn(|_| T::zero());
    // Find actual degrees
    let dividend_order = largest_nonzero_index(&dividend);
    let divisor_order = largest_nonzero_index(divisor);

    // degree of self and rhs exists
    if let Some(dividend_order) = dividend_order {
        if let Some(divisor_order) = divisor_order {
            let mut remainder = dividend;
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
