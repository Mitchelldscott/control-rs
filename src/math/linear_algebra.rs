//! Helpful functions for linear algebra operations
//!
//! Should consider sharing this with `nalgebra`.

use core::ops::{Add, Mul, Neg, Sub};
use num_traits::{One, Zero};
use crate::static_storage::array_from_iterator;

/// Computes the vector dot product of two vectors.
///
/// This function calculates the dot product of two vectors: `x = a * b + c`. Where `a` has N
/// columns, `b` has N rows and x and c are scalars.
///
/// Zips the two input vectors, calls [`fold()`] on the iterator and returns the result.
///
/// # Generic Arguments
/// * `T` - Element type of the vectors that form a commutative ring.
/// * `N` - Length of each input vector.
///
/// # Arguments
/// * `rhs` - 1D array representing a row vector (`a`).
/// * `lhs` - 1D array representing a colum vector (`b`).
/// * `fold_init` - Initial value for the fold operation (`c`).
///
/// # Returns
/// `x` - The scalar value resulting from the dot product of the two input vectors.
///
/// # Panics
/// * While debug assertions are enabled, this may:
///     * `attempt to multiply with overflow` if `a[i] * b[i] < T::MIN || T::MAX < a[i] * b[i]`.
///     * `attempt to add with overflow` if `sum(a[i] * b[i] for i in 0..n) + c < T::MIN ||
///        T::MAX < sum(a[i] * b[i] for i in 0..n) + c` is violated for any `n in 0..N`.
///
/// # Safety
/// This function does not call `unsafe` code.
pub fn vector_dot_product<T, const N: usize>(rhs: &[T; N], lhs: &[T; N], fold_init: T) -> T
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    rhs
        .iter()
        .zip(lhs.iter())
        .fold(fold_init, |acc, (ai, bi)| acc + ai.clone() * bi.clone())
}

/// Computes the resulting vector from multiplying a matrix and vector.
///
/// This function computes the vector `b` in `b = A * x + c`. Where `A` is an `MxN` matrix, `x` is
/// a column vector with `N` elements and `b` and `c` are column vectors with `M` elements.
///
/// This function zips the rows of `A` with the elements of `c` and then uses
/// ['vector_dot_product()`] to find each element of `b`.
///
/// # Generic Arguments
/// * `T` - Element type of the matrix and vector that forms a commutative ring.
/// * `M` - Number of columns in `A` (length of a row) and rows in `x` (length of a column).
/// * `N` - Number of rows in `A` (length of a column) and rows in `b` and `c` (length of a column).
///
/// # Arguments
/// `a` - 2D array on the left-hand side of the operator.
/// `x` - 1D array on the right-hand side of the operator.
/// `c` - 1D array of offsets for the result vector.
///
/// # Returns
/// * `b` - 1D array representing a column vector resulting from the multiplication.
///
/// # Panics
/// * This function has the same panics as [`vector_dot_product()`].
///
/// # Safety
/// * This function makes an `unsafe` call to initialize an array from an iterator. The safety
///   criterion is satisfied because the iterator is guaranteed to have `M` elements.
/// TODO: prove that [`array_from_iterator()`] is not called, or at-least will not initialize any
///     elements when constructing the iterator fails.
fn matrix_vector_mul<T, const M: usize, const N: usize>(a: &[[T; N]; M], b: &[T; N], c: [T; M]) -> [T; M]
where
    T: Clone + Add<Output = T> + Mul<Output = T>
{
    unsafe {
        array_from_iterator(
            a.iter().zip(c.into_iter()).map(|(row, offset)|
                vector_dot_product(row, b, offset)
            )
        )
    }
}

/// Computes the resulting matrix from multiplying two matrices.
///
/// The matrices must have a common inner dimension. I.e., `C = A * B` means `shape(C) = (rows(A),
/// cols(B))` and `cols(A) == rows(B)`.
///
/// # Generic Arguments
/// * `T` - Element type of the matrices that form a commutative ring.
/// * `M` - Number of rows of `A` and `C` (length of a column).
/// * `N` - Number of columns of `A` and rows of `B`.
/// * `K` - Number of columns of `B` and `C` (length of a row).
///
/// # Arguments
/// * `a` - Matrix on the right-hand side of the operator.
/// * `b` - Matrix on the left-hand side of the operator.
/// * `c` - Offset Matrix to add to the result.
///
/// # Returns
/// * `a * b + c` - The result of the multiplication plus the offset.
///
/// TODO:
///   * Unit test
///   * Cost of cache miss? when should this switch to transpose(b)? time of cache miss > time to
///     transpose?
///   * Map efficiency, should this use [`array_from_iterator()`] or manually create and initialize
///     a 2D array?
fn matrix_matrix_mul<T, const N: usize, const K: usize, const M: usize>(a: &[[T; K]; N], b: &[[T; M]; K], c: [[T; M]; N]) -> [[T; M]; N]
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T>,
{
    unsafe {
        // Safety: The iterator is guaranteed to have `N` elements.
        array_from_iterator(
            c.into_iter().enumerate().map(|(i, c_i)|
                // Safety: The iterator is guaranteed to have `M` elements.
                array_from_iterator(
                    c_i.into_iter().enumerate().map(|(j, c_ij)|
                        (0..K).fold(c_ij, |acc, k|
                            acc + a[i][k].clone() * b[k][j].clone()
                        )
                    )
                )
            )
        )
    }
}

/// Calculates the matrix exponential using the power method.
///
/// # Generic Arguments
/// * `T` - Element type of the matrix and exponent that forms a commutative ring.
/// * `N` - Number of rows and columns in `A`.
///
fn matrix_exp<T, const N: usize>(a: &[[T; N]; N], k: usize) -> [[T; N]; N]
where
    T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    // if k == 0 {
    //     // R * I * C = dot(R, C)
    //     return r.iter().zip(c).fold(T::zero(), |acc, (ri, ci)| acc + ri.clone() * ci.clone());
    // }
    //
    // let mut vec = c.clone();
    // for _ in 0..k {
    //     let mut next = [T::zero(); N];
    //     for i in 0..N {
    //         for j in 0..N {
    //             next[i] = next[i].clone() + a[i][j].clone() * vec[j].clone();
    //         }
    //     }
    //     vec = next;
    // }
    //
    // r.iter().zip(&vec).fold(T::zero(), |acc, (ri, vi)| acc + ri.clone() * vi.clone())
    [[T::zero(); N]; N]
}

// // Build Toeplitz matrix T0
// fn build_toeplitz<T, const N: usize>(
//     a: &[[T; N]; N],
// ) -> [[T; N + 1]; N + 1]
// where
//     T: Clone + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
// {
//     let mut t = [[T::zero(); N + 1]; N + 1];
//     let a11 = a[0][0].clone();
//
//     for i in 0..=N {
//         for j in 0..=N {
//             if i == j {
//                 t[i][j] = T::one();
//             } else if j + 1 == i {
//                 t[i][j] = -a11.clone();
//             } else if j + 2 <= i {
//                 let r = &a[0][1..N];
//                 let c: Vec<_> = (1..N).map(|i| a[i][0].clone()).collect();
//                 let mut rc_prod = power_multiply(
//                     &a[1..].iter().map(|row| row[1..].to_vec()).collect::<Vec<_>>()
//                         .iter()
//                         .map(|v| v.clone().try_into().unwrap())
//                         .collect::<Vec<[T; N - 1]>>()
//                         .try_into()
//                         .unwrap(),
//                     &r.try_into().unwrap(),
//                     &c.try_into().unwrap(),
//                     i - 2,
//                 );
//                 t[i][j] = -rc_prod;
//             }
//         }
//     }
//
//     t
// }

// // Recursive characteristic polynomial computation
// pub fn characteristic_poly<T, const N: usize>(a: [[T; N]; N]) -> [T; N + 1]
// where
//     T: Clone
//     + Zero
//     + One
//     + Add<Output = T>
//     + Sub<Output = T>
//     + Mul<Output = T>
//     + Neg<Output = T>,
// {
//     if N == 1 {
//         return [T::one(), -a[0][0].clone()];
//     }
//
//     let a1 = {
//         let mut sub = [[T::zero(); N - 1]; N - 1];
//         for i in 1..N {
//             for j in 1..N {
//                 sub[i - 1][j - 1] = a[i][j].clone();
//             }
//         }
//         sub
//     };
//
//     let p1 = characteristic_poly::<T, { N - 1 }>(a1);
//     let t0 = build_toeplitz(&a);
//
//     matvec(&t0, &p1)
// }
