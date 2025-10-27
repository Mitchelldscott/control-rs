use nalgebra::{Const, DimAdd, DimMin, DimSub, RealField, U1};
use num_traits::Zero;

/// Transposes the contents of a 2D array.
///
/// This will copy every (i,j)th element of one array into the (j,i)th element
/// of a buffer. The buffer is assumed to be fully initialized to zero.
///
/// # Arguments
/// * `src`: `n x m` array
/// * `dst`: `m x n` array (initialized to zero)
///
/// # Safety
/// * This function requires that src and dst are fully initialized.
pub fn transpose_unchecked<T: Clone, const N: usize, const M: usize>(
    src: &[[T; N]; M],
    dst: &mut [[T; M]; N],
) {
    for i in 0..N {
        for j in 0..M {
            // SAFETY: The buffer and source are fully initialized and their sizes
            // are checked at compile time.
            unsafe {
                dst.get_unchecked_mut(j)
                    .get_unchecked_mut(i)
                    .clone_from(src.get_unchecked(i).get_unchecked(j));
            }
        }
    }
}

pub fn transpose<T: Copy + Zero, const N: usize, const M: usize>(src: &[[T; N]; M]) -> [[T; M]; N] {
    let mut dst = [[T::zero(); M]; N];
    transpose_unchecked(src, &mut dst);
    dst
}

fn split_columns<T, const N: usize, const P: usize, const NP: usize>(
    g: &[[T; NP]; N],
) -> ([[T; N]; N], [[T; P]; N])
where
    T: Zero + Copy,
    Const<N>: DimAdd<Const<P>, Output = Const<NP>>,
{
    let mut a = [[T::zero(); N]; N];
    let mut b = [[T::zero(); P]; N];

    for (src_row, (a_row, b_row)) in g.iter().zip(a.iter_mut().zip(b.iter_mut())) {
        a_row.copy_from_slice(&src_row[..N]);
        b_row.copy_from_slice(&src_row[N..]);
    }

    (a, b)
}

use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

// Define a trait bound for numeric types that support basic arithmetic operations
// This is necessary for a generic implementation.
pub trait Numeric:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + PartialOrd
    + Debug
    + From<f64>
    + Into<f64>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
}

// Example implementation for f64
impl Numeric for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

// --- SVD Component Structs ---
// U: Left singular vectors (m x m)
// S: Singular values (diagonal of an m x n matrix Σ)
// Vt: Transpose of right singular vectors (n x n)
#[derive(Debug)]
pub struct SVD<T: Numeric, const M: usize, const N: usize, const MN: usize>
where
    Const<M>: DimMin<Const<N>, Output = Const<MN>>,
{
    pub u: [[T; M]; M],
    pub s: [T; MN], // Vector of singular values
    pub vt: [[T; N]; N],
}

// --- Placeholder for a Matrix type or 2D Array ---
// In a real implementation, you'd likely have a dedicated Matrix struct.
// For this example, we use [[T; N]; M] as requested.

/**
 * Computes the Singular Value Decomposition (SVD) of a matrix A.
 * A = U * Σ * Vᵀ
 *
 * Where:
 * - A is the input matrix [M x N]
 * - U is an [M x M] orthogonal matrix of left singular vectors (forming a basis for the codomain).
 * - Σ is an [M x N] diagonal matrix containing the singular values (σ_i).
 * - Vᵀ is a transposed [N x N] orthogonal matrix of right singular vectors (forming a basis for the domain).
 *
 * This function outlines the steps; a robust implementation requires careful handling
 * of numerical stability (e.g., using Golub-Kahan bidiagonalization and QR iteration).
 */
pub fn svd<T: Numeric, const M: usize, const N: usize, const MN: usize>(
    a: &[[T; N]; M],
) -> Result<SVD<T, M, N, MN>, &'static str>
where
    Const<M>: DimMin<Const<N>, Output = Const<MN>>,
{
    // A robust SVD implementation is non-trivial. The standard approach involves:

    // --- Step 1: Bidiagonalization ---
    // Reduce the matrix A to an upper bidiagonal matrix B using Householder reflections.
    // A = Q_L * B * Q_Rᵀ
    // Where Q_L and Q_R are orthogonal matrices built from Householder reflectors.
    // U will be derived from Q_L and Vᵀ from Q_Rᵀ.
    let (b, u_accum, v_accum) = bi_diagonalize(a)?;

    // --- Step 2: Iterative Diagonalization (e.g., Golub-Kahan-Reinsch QR Algorithm) ---
    // Iteratively apply QR decompositions with implicit shifts to the bidiagonal matrix B
    // until it converges to a diagonal matrix (whose diagonal elements are the singular values).
    // B_k -> Σ as k -> ∞
    // The orthogonal transformations used in each QR step are accumulated into U and V.

    let max_iterations = 100; // Set a limit for convergence
    let mut singular_values = [T::zero(); MN];

    for _iter in 0..max_iterations {
        // a) Find a suitable shift (e.g., Wilkinson shift) to speed up convergence.

        // b) Perform an implicit QR step (Givens rotations) on B.
        //    This chases the "bulge" down the bidiagonal.

        // c) Accumulate these rotations into the U and V matrices.
        //    u_accum = u_accum * G_L
        //    v_accum = G_Rᵀ * v_accum

        // d) Check for convergence:
        //    - If any super-diagonal element is close to zero, we can "deflate"
        //      the problem by splitting B into smaller subproblems.
        //    - If all super-diagonal elements are effectively zero, B is diagonal.

        // [Pseudo-code for iteration]
        // if converged(&b) { break; }
        // let shift = calculate_shift(&b);
        // (b, givens_l, givens_r) = implicit_qr_step(&b, shift);
        // u_accum = matrix_multiply_m_m_m(&u_accum, &givens_l);
        // v_accum = matrix_multiply_n_n_n(&givens_r, &v_accum);
    }

    // --- Step 3: Finalization ---
    // a) Ensure all singular values are non-negative.
    //    If σ_i is negative, flip its sign and the corresponding column in U.
    //    (The QR algorithm might produce negative values on the diagonal).

    // b) Sort the singular values and corresponding singular vectors in descending order.

    // c) Extract the diagonal elements of the converged B into the singular_values vector.
    //    (This is a simplified placeholder)
    let n_values = if M < N { M } else { N };
    for i in 0..n_values {
        singular_values[i] = b[i][i]; // Assuming b converged
    }

    // Placeholder result
    Ok(SVD {
        u: u_accum, // U = Q_L * (product of left Givens rotations)
        s: singular_values,
        vt: v_accum, // Vᵀ = (product of right Givens rotations) * Q_Rᵀ
    })
}

/// Solves the linear system A * x = b using the precomputed SVD.
///
/// For the user's request: Solves Ωᵀ * Gᵀ = X'ᵀ for Gᵀ
/// Where A = Ωᵀ, x = Gᵀ, b = X'ᵀ
/// The solution is found using the pseudoinverse (Moore-Penrose inverse):
/// <pre> x = A⁺ b = V Σ⁺ Uᵀ b </pre>
/// Σ⁺ (pseudoinverse of Σ) is an [N x M] matrix where the non-zero diagonal
/// elements σ_i are replaced by 1/σ_i. Values smaller than `tolerance` are
/// treated as zero to handle singularity.
pub fn solve_svd<T: Numeric, const M: usize, const N: usize, const K: usize, const MN: usize>(
    svd_of_a: &SVD<T, M, N, MN>,
    b: &[[T; K]; M],
    tolerance: T,
) -> Result<[[T; K]; N], &'static str>
where
    Const<M>: DimMin<Const<N>, Output = Const<MN>>,
{
    // Returns x (Gᵀ) [N x K]

    let min_dim = if M < N { M } else { N };
    let mut sigma_pseudo_inv_diag = [T::zero(); MN];

    // 1. Calculate Σ⁺ (as a diagonal vector)
    for (i, &s) in svd_of_a.s.iter().enumerate() {
        if s.abs() > tolerance {
            sigma_pseudo_inv_diag[i] = T::one() / s;
        } else {
            sigma_pseudo_inv_diag[i] = T::zero(); // Treat as singular
        }
    }

    // 2. Compute x = V * Σ⁺ * Uᵀ * b
    //    x = (svd_of_a.vt)ᵀ * Σ⁺ * (svd_of_a.u)ᵀ * b

    // 2a. c = Uᵀ * b
    let u_transpose = transpose_m_m(&svd_of_a.u);
    let c = matrix_multiply_m_n_k(&u_transpose, b)?; // [M x M] * [M x K] -> [M x K]

    // 2b. d = Σ⁺ * c
    // This is a diagonal matrix multiplication, so we just scale rows.
    let mut d = [[T::zero(); K]; N];
    for i in 0..N {
        for j in 0..K {
            if i < min_dim {
                d[i][j] = sigma_pseudo_inv_diag[i] * c[i][j];
            }
            // else d[i][j] remains zero (from initialization)
        }
    }

    // 2c. x = V * d = (Vᵀ)ᵀ * d
    let v = transpose_n_n(&svd_of_a.vt);
    let x = matrix_multiply_m_n_k(&v, &d)?; // [N x N] * [N x K] -> [N x K]

    Ok(x)
}

// --- Helper Function Skeletons (Required for a full implementation) ---
#[allow(clippy::type_complexity)]
fn bi_diagonalize<T: Numeric, const M: usize, const N: usize>(
    _a: &[[T; N]; M],
) -> Result<
    (
        InputMatrix<T, M, N>,
        StateTransitionMatrix<T, M>,
        StateTransitionMatrix<T, N>,
    ),
    &'static str,
> {
    // Placeholder: This function would perform Householder bidiagonalization.
    // It returns (B, U_accum, V_accum)
    // B is the bidiagonal matrix#[warn(clippy::type_complexity)]
    // U_accum is the accumulator for U (initialized to identity or Q_L)
    // V_accum is the accumulator for Vᵀ (initialized to identity or Q_Rᵀ)
    println!("Step 1: Bi-diagonalizing... (placeholder)");
    Ok((
        [[T::zero(); N]; M],   // B
        identity_matrix_m_m(), // U
        identity_matrix_n_n(), // Vᵀ
    ))
}

// --- Matrix Math Helpers (Placeholders) ---

fn transpose_m_n<T: Numeric, const M: usize, const N: usize>(mat: &[[T; N]; M]) -> [[T; M]; N] {
    let mut out = [[T::zero(); M]; N];
    for (i, row) in mat.iter().enumerate() {
        for (j, o) in out.iter_mut().enumerate() {
            o[i] = row[j];
        }
    }
    out
}
fn transpose_m_m<T: Numeric, const M: usize>(mat: &[[T; M]; M]) -> [[T; M]; M] {
    transpose_m_n(mat)
}
fn transpose_n_n<T: Numeric, const N: usize>(mat: &[[T; N]; N]) -> [[T; N]; N] {
    transpose_m_n(mat)
}

fn matrix_multiply_m_n_k<T: Numeric, const M: usize, const N: usize, const K: usize>(
    a: &[[T; N]; M],
    b: &[[T; K]; N],
) -> Result<[[T; K]; M], &'static str> {
    let mut out = [[T::zero(); K]; M];
    for i in 0..M {
        for j in 0..K {
            for (l, b) in b.iter().enumerate() {
                out[i][j] = out[i][j] + a[i][l] * b[j];
            }
        }
    }
    Ok(out)
}

fn identity_matrix_m_m<T: Numeric, const M: usize>() -> [[T; M]; M] {
    let mut ident = [[T::zero(); M]; M];
    for (i, id) in ident.iter_mut().enumerate() {
        id[i] = T::one();
    }
    ident
}

fn identity_matrix_n_n<T: Numeric, const N: usize>() -> [[T; N]; N] {
    let mut ident = [[T::zero(); N]; N];
    for (i, id) in ident.iter_mut().enumerate() {
        id[i] = T::one();
    }
    ident
}

type StateTransitionMatrix<T, const N: usize> = [[T; N]; N];
type InputMatrix<T, const N: usize, const M: usize> = [[T; M]; N];

/// Implements Dynamic Mode Decomposition with Control (DMDc).
///
/// This function discovers the best-fit linear system matrices (A, B) that
/// approximate the dynamics `x' ≈ Ax + Bu` given a time-series history of
/// state vectors `x_history` and control vectors `u_history`.
///
/// `x_history` is an `n x m` matrix, where `n` is the state dimension and `m` is the
/// number of samples.
/// `u_history` is a `p x m` matrix, where `p` is the control dimension and `m` is the
/// number of samples.
///
/// This is a practical application of the concepts above:
/// 1.  It solves a linear least-squares problem to find `[A, B]`.
/// 2.  It uses Singular Value Decomposition (SVD), which is built from
///     eigenvectors, to find the pseudoinverse robustly.
/// 3.  The resulting `A` and `B` matrices represent the **Linear Transformations**
///     that govern the system's dynamics.
fn dmdc<T, const N: usize, const P: usize, const NP: usize, const M: usize, const M1: usize>(
    x: [[T; N]; M],
    u: [[T; P]; M],
) -> Result<(StateTransitionMatrix<T, N>, InputMatrix<T, N, P>), &'static str>
where
    T: Copy + RealField,
    Const<M>: DimSub<U1, Output = Const<M1>>,
    Const<M1>: DimSub<U1>,
    Const<N>: DimAdd<Const<P>, Output = Const<NP>>,
    Const<NP>: DimMin<Const<M1>>,
{
    // 1. Create update sample set: X'
    // X = [x(t1), x(t2), ..., x(t{m-1})]
    // X' = [x(t2), x(t3), ..., x(tm)]
    let mut x_prime = [[T::zero(); N]; M1];
    x_prime.copy_from_slice(&x[1..]);

    // 2. Form the augmented matrix Ω = [X; U]
    let _omega: [[T; NP]; M1] = std::array::from_fn(|k| {
        let mut row = [T::zero(); NP]; // row does not need to be initialized
        row[0..N].copy_from_slice(&x[k]);
        row[N..NP].copy_from_slice(&u[k]);
        row
    });

    // 3. Solve the least-squares problem X' ≈ G * Ω, where G = [A, B]
    // We want to find G that minimizes ||X' - GΩ||.
    // The solution is G = X' * Ω⁺ (where Ω⁺ is the pseudoinverse).
    // This is solved most stably using SVD.

    // SVD computes the pseudoinverse, which is related to the **Matrix Inverse**.
    // The SVD itself finds the principal components, which form a **Basis**
    // for the input space.
    // Solves Ωᵀ * Gᵀ = X'ᵀ for Gᵀ
    // let g_transpose = solve_svd(&svd(&transpose(&omega)), &transpose(&x_prime), 1e-10) // 1e-10 is a tolerance for singular values
    //     .map_err(|_| "SVD solve failed. Matrix may be singular.")?;

    // G = (Gᵀ)ᵀ
    // This uses the property (Aᵀ)ᵀ = A.
    // let g = transpose(&g_transpose);
    let g = [[T::zero(); NP]; N];
    // 5. Extract A and B from G = [A, B]
    // `A` is the first `n` columns, `B` is the next `p` columns.
    // The `A` and `B` matrices are the **Matrix of a Transformation**.
    Ok(split_columns::<T, N, P, NP>(&g))
}

// Example Usage (requires `nalgebra` crate in Cargo.toml):
fn main() {
    // n=2 states, p=1 control, m=6 samples
    let x_hist = [
        [1.0, 0.0], // x1
        [1.0, 0.1], // x2
        [0.9, 0.2], // x3
        [0.7, 0.3], // x4
        [0.4, 0.4], // x5
        [0.0, 0.5], // x6
    ];

    let u_hist = [[0.5], [0.5], [0.5], [0.5], [0.5], [0.5]]; // u1 to u6

    match dmdc::<f64, 2, 1, 3, 6, 5>(x_hist, u_hist) {
        Ok((a, b)) => {
            println!("Found A matrix:\n{:#?}", a);
            println!("Found B matrix:\n{:#?}", b);
        }
        Err(e) => {
            println!("{}", e);
        }
    }
}
