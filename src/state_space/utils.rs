//! Utilities for creating, converting and analyzing SS models
//!

use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

use crate::static_storage::array_from_iterator;
use crate::{StateSpace, TransferFunction};
use nalgebra::{Const, DimSub, SMatrix, Scalar, U1};
use num_traits::{One, Zero};

/// Create a new SISO state-space model from the numerator and denominator coefficients of
/// a monic transfer function
///
/// The transfer function should be proper and its denominator must be monic. The function still
/// runs if this criterion is not met, but the results will be incorrect.
///
/// <pre>
/// G(s) = b(s) / a(s)
/// a(s) = (s^N + a_N s^(N-1) + ... + a_1)
/// b(s) = (b_N s^(N-1) + b_(N-1) s^(N-2) + ... + b_1)
///
/// A = [  0    1    0  ...  0   ]
///     [  0    0    1  ...  0   ]
///     [  ...                   ]
///     [ -a_1 -a_2 -a_3... -a_n ]
///  
/// - the final row of A is the denominator coefficients
///   where j = 0 is the constant and j = N - 1 is the second-highest order coefficient
/// - the derivative of each state variable i is the state variable `i + 1`
///
/// B = [ 0 ]
///     [...]
///     [ 1 ]
///
/// - the input is a term in the derivative of x_n
///
/// C = [ b_1 b_2 ... b_n]
///
/// D = [ 0 ]
/// </pre>
///
/// # Arguments
/// * `b` - coefficients of a transfer function's numerator `[b_n ... b_1]`
/// * `a` - coefficients of a transfer function's denominator `[a_n ... a_1]`
///
/// # Returns
/// * `StateSpace` - state-space model in control canonical form
///
/// # Example
///
/// ```
/// use control_rs::state_space::{StateSpace, utils::control_canonical};
/// let ss = control_canonical(&[1.0], &[0.1, 0.0]);
/// println!("{ss}");
/// ```
/// # TODO: repair docs
pub fn control_canonical<T, const N: usize, const M: usize, const L: usize>(
    b: &[T; M],
    a: &[T; L],
) -> StateSpace<SMatrix<T, N, N>, SMatrix<T, N, 1>, SMatrix<T, 1, N>, SMatrix<T, 1, 1>>
where
    T: Scalar + Clone + Zero + One + Neg<Output = T> + Div<Output = T>,
    Const<L>: DimSub<U1, Output = Const<N>>,
{
    StateSpace {
        a: SMatrix::from_fn(|i, j| {
            if i == N - 1 {
                a[j].clone().neg() / a[L - 1].clone()
            } else if i + 1 == j {
                T::one()
            } else {
                T::zero()
            }
        }),
        b: SMatrix::from_fn(|i, _| if i == N - 1 { T::one() } else { T::zero() }),
        c: SMatrix::from_fn(|_, j| if j < M { b[j].clone() } else { T::zero() }),
        d: SMatrix::from_fn(|_, _| T::zero()),
    }
}

/// Discretizes the given `StateSpace` model
///
/// This discretization applies a zero-order-hold (zoh) to the continuous state space matrices.
/// zoh relies on the matrix exponent operation e^At which is approximated using a tenth order
/// taylor-series expansion. The implementation is based on an example from "Digital Control of
/// Dynamic Systems" by Gene F. Franklin, J. David Powell and Michael Workman (Ch 4.3, pg 107).
///
/// 1. Select sampling period T
/// 2. **I** = Identity
/// 3. **Psi** = Identity
/// 4. k = 10
/// 5. for i in 0..k-1
///     - **Psi** = **I** + **A** * T * **Psi** / (k - i)
/// 6. **G** = T * **Psi** * **B**
/// 7. **F** = **I** + **F** * T * **Psi**
///
/// # Arguments
/// * `&self` - the dynamic system to linearize
/// * `ts` - sampling time of the system
/// * `x` - state vector
/// * `u` - input vector
///
/// # Returns
/// * `StateSpace` - a discretized version of the state space model
///
/// # Example
/// ```
/// use control_rs::state_space::utils::{control_canonical, zoh};
/// let ss = control_canonical(&[1.0, 1.0], &[1.0, 1.0, 1.0]);
/// let ssd = zoh(&ss, 0.1_f32);
/// println!("{ssd}");
/// ```
///
/// ## TODO:
/// - [ ] compare with [`nalgebra::Matrix::exp`]
pub fn zoh<T, A, B, C, D>(ss: &StateSpace<A, B, C, D>, ts: T) -> StateSpace<A, B, C, D>
where
    T: Clone + Zero + One + Sub<Output = T>,
    A: Clone
        + One
        + Add<Output = A>
        + Mul<T, Output = A>
        + Div<T, Output = A>
        + Mul<A, Output = A>
        + Mul<B, Output = B>
        + Default,
    B: Clone + Mul<T, Output = B>,
    C: Clone,
    D: Clone,
{
    let k = 10;
    // TODO: Really need constants to remove this garbage
    let k_as_t = (0..k).fold(T::zero(), |t, _| T::one() + t.clone());
    let identity = A::one();
    // let psi = (0..k - 1).fold(identity.clone(), |psi, i| {
    //     identity.clone() + ss.a.clone() * ts.clone() * psi / T::from(k - i)
    // });
    let (_, psi) = (0..k - 1).fold((k_as_t, identity.clone()), |(k, psi), _| {
        (
            k.clone() - T::one(),
            identity.clone() + ss.a.clone() * ts.clone() * psi / k,
        )
    });
    StateSpace {
        a: identity + ss.a.clone() * ts.clone() * psi.clone(),
        b: psi * ss.b.clone() * ts,
        c: ss.c.clone(),
        d: ss.d.clone(),
    }
}

/// Converts a state-space representation to a transfer function matrix.
///
/// This function takes a MIMO (Multiple-Input, Multiple-Output) state-space model
/// defined by matrices (A, B, C, D) and calculates the equivalent matrix of
/// transfer functions, H(s).
///
/// The conversion is performed using the well-known formula:
/// <pre> H(s) = C * (sI - A)^-1 * B + D </pre>>
/// where `I` is the identity matrix and `s` is the complex frequency variable.
///
/// # Generic Arguments
/// * `T`: The numeric type of the matrix elements (e.g., `f64`).
/// * `N`: The number of states in the system (dimension of the A matrix).
/// * `M`: The number of inputs to the system (number of columns in B and D).
/// * `L`: The number of outputs from the system (number of rows in C and D).
/// * `N2`: The required size of the denominator coefficient array for each transfer function.
/// * `M2`: The required size of the numerator coefficient array for each transfer function.
///
/// # Arguments
/// * `ss`: A `StateSpace` object representing the system, containing:
///     * `a`: The `NxN` state matrix.
///     * `b`: The `NxM` input matrix.
///     * `c`: The `LxN` output matrix.
///     * `d`: The `LxM` feedthrough matrix.
///
/// # Returns
/// * An `MxL` matrix (a 2D array of `M` rows by `L` columns) of `TransferFunction` objects.
///
/// **Important Note on Dimensions**: The returned `[[T; L]; M]` matrix is organized
/// such that the element at `H[i][j]` represents the transfer function from the
/// **i-th input** to the **j-th output**.
///
/// ### Size of Each Transfer Function
/// Each `TransferFunction<T, M2, N2>` in the returned matrix is a SISO (Single-Input,
/// Single-Output) system. The size of its numerator and denominator coefficient arrays
/// (`M2` and `N2` respectively) is determined by `N`, the number of states in the system.
///
/// * **Denominator Size (`N2`)**: The denominator of every transfer function in the
///     matrix is the characteristic polynomial of the A matrix, given by `det(sI - A)`.
///     This is a polynomial of degree `N`. To store its `N + 1` coefficients (from s⁰ to sⁿ),
///     the constant `N2` **must be `N + 1`**.
///
/// * **Numerator Size (`M2`)**: The numerator of each transfer function is derived from the
///     expression `C * adj(sI - A) * B + D`, where `adj` is the adjugate matrix. The
///     polynomials in `adj(sI - A)` have a maximum degree of `N - 1`. Therefore, the resulting
///     numerator polynomial will have a maximum degree of `N`. To store its `N + 1` coefficients,
///     the constant `M2` **must be `N + 1`**. If the D matrix is zero, the maximum numerator
///     degree is `N-1`, requiring `N` coefficients, but `N+1` provides a safe upper bound for all
///     cases.
///
/// # Reference
/// * [Swarthmore: Transformations](https://lpsa.swarthmore.edu/Representations/SysRepTransformations/TF2SS.html)
/// * [Wikipedia Faddeev-LaVerrier](https://en.wikipedia.org/wiki/Faddeev%E2%80%93LeVerrier_algorithm)
/// * [Matlab ss2tf](https://www.mathworks.com/help/matlab/ref/ss2tf.html)
pub fn ss2tf<T, const N: usize, const M: usize, const L: usize, const N2: usize, const M2: usize>(
    ss: StateSpace<SMatrix<T, N, N>, SMatrix<T, N, M>, SMatrix<T, L, N>, SMatrix<T, L, M>>,
) -> [[TransferFunction<T, M2, N2>; L]; M]
where
    T: 'static
        + Copy
        + Zero
        + One
        + PartialEq
        + core::fmt::Debug
        + AddAssign
        + MulAssign
        + Neg<Output = T>
        + Div<Output = T>,
    Const<L>: DimSub<U1, Output = Const<N>>,
{
    // --- 1. Faddeev-LeVerrier Algorithm ---
    // This algorithm computes the coefficients of the characteristic polynomial det(sI - A)
    // and the coefficient matrices of the adjugate matrix adj(sI - A).

    let mut r_matrices: [SMatrix<T, N, N>; N] = [SMatrix::zeros(); N];
    let mut p_coeffs: [T; N] = [T::zero(); N]; // p1, p2, ..., pN
    let identity = SMatrix::<T, N, N>::identity();

    let mut r_prev = identity;
    let mut k = T::one();
    for i in 1..=N {
        let m_k = &ss.a * &r_prev;
        let p_k = -m_k.trace() / k;
        let r_k = &m_k + &identity * p_k;

        r_matrices[i] = r_prev; // Stores R_{k-1} for adjugate matrix calculation
        p_coeffs[i] = p_k;
        r_prev = r_k;
        k += T::one();
    }

    // --- 2. Construct Common Denominator Polynomial ---
    // The characteristic polynomial is s^N + p1*s^(N-1) + ... + pN.
    // We store coefficients from highest power (s^N) to lowest (s^0). // this is wrong should be flipped
    // This requires the const generic N2 to be N + 1.
    let mut den_coeffs = SMatrix::<T, N2, 1>::zeros();
    den_coeffs[0] = T::one(); // Coefficient of s^N
    for i in 0..N {
        den_coeffs[i + 1] = p_coeffs[i];
    }

    // --- 3. Calculate Numerator for Each Input-Output Pair ---
    // Pre-calculate the matrix products C * R_k * B for efficiency.
    // Safety: There are exactly N matrices in r_matrices.
    let crb_matrices: [SMatrix<T, L, M>; N] =
        unsafe { array_from_iterator(r_matrices.iter().map(|r_k| &ss.c * r_k * &ss.b)) };

    let mut result = [[TransferFunction::new([T::zero(); M2], [T::zero(); N2]); L]; M];

    // Loop through each output `i` and input `j` to build the H_ij(s) transfer function.
    for i in 0..L {
        // Output index
        for j in 0..M {
            // Input index
            // This requires the const generic M2 to be N + 1.

            // a) Calculate numerator part from C * adj(sI - A) * B
            // The adjugate polynomial is R_0*s^(N-1) + R_1*s^(N-2) + ... + R_{N-1}
            // The resulting numerator polynomial coefficients are the (i,j) elements of the crb_matrices.
            let mut num_coeffs = SMatrix::<T, M2, 1>::zeros();
            for k in 0..N {
                // The coefficient for s^(N-1-k) is (crb_k)_ij.
                // In our vector (highest power first), this corresponds to index k+1.
                num_coeffs[k + 1] = crb_matrices[k][(i, j)];
            }

            // b) Calculate numerator part from D * det(sI - A)
            // let d_ij = ss.d[(i, j)];
            // let num_part2: SMatrix<T, M2, 1> = den_coeffs.map(|c| d_ij * c);

            // c) Combine to get the final numerator polynomial
            // let num_coeffs = num_part1 + num_part2;

            // d) Create the TransferFunction and place it in the result matrix (transposed)
            let tf = TransferFunction::new(num_coeffs.data.0[0], den_coeffs.data.0[0].clone());
            result[j][i] = tf;
        }
    }

    result
}
