//! Miscellaneous tools to help work with transfer functions
//!
use core::{
    fmt,
    ops::{Div, Neg, Sub},
};

use nalgebra::{
    Complex, Const, DefaultAllocator, DimAdd, DimDiff, DimMin, DimMinimum, DimSub, OMatrix,
    RealField, SMatrix, Scalar, ToTypenum, U1, allocator::Allocator,
};
use num_traits::{Float, One, Zero};

use crate::{
    StateSpace, TransferFunction,
    frequency_tools::FrequencyResponse,
    polynomial::utils::{RootFindingError, largest_nonzero_index, unchecked_roots},
    state_space::utils::control_canonical,
};

/// Computes the DC gain of a continuous transfer function.
///
/// The DC gain is the value of the transfer function as the frequency approaches zero.
///
/// <pre>
/// DC Gain = b_m / a_n
/// </pre>
///
/// Where `b_m` is the constant term of the numerator, and `a_n` is the constant term
/// of the denominator.
///
/// # Arguments
/// * `tf` - the transfer function to analyze
///
/// # Returns
/// * `Option<T>` - The DC gain if both numerator and denominator have a non-zero constant term;
///   `None` otherwise
///
/// # Generic Arguments
/// * `T` - Scalar type for the coefficients (e.g., `f32`, `f64`)
/// * `N` - number of coefficients in the denominator
/// * `M` - number of coefficients in the numerator
///
/// # Example
///
/// ```rust
/// use control_rs::transfer_function::{TransferFunction, dc_gain};
/// // Transfer function: G(s) = (2s + 4) / (s^2 + 3s + 2)
/// let tf = TransferFunction::new([2.0, 4.0], [1.0, 3.0, 2.0]);
/// let gain = dc_gain(&tf);
/// println!("DC Gain: {gain:.2}");
/// ```
pub fn dc_gain<T: Float, const M: usize, const N: usize>(tf: &TransferFunction<T, M, N>) -> T {
    if N > 0 {
        if tf.denominator[0].is_zero() {
            T::infinity()
        } else if M > 0 {
            tf.numerator[0] / tf.denominator[0]
        } else {
            T::nan()
        }
    } else {
        T::nan()
    }
}

/// Compute the roots of the characteristic equation
///
/// Calculates the eigen values of a companion matrix constructed from the denominator.
///
/// # Arguments
///  * `tf` - the transfer function to check the poles of
///
/// # Returns
///  * `[complex<T>; N]` - poles of the tf
///
/// # Example
///
/// ```rust
/// use control_rs::transfer_function::{TransferFunction, poles};
/// // Transfer function: G(s) = (2s + 4) / (s^2 + 3s + 2)
/// let tf = TransferFunction::new([2.0, 4.0], [1.0, 3.0, 2.0]);
/// let poles = poles(&tf);
/// ```
/// # Errors
/// * `NoRoots` - the function was not able to find any roots for the denominator
pub fn poles<T, const M: usize, const N: usize, const L: usize>(
    tf: &TransferFunction<T, M, N>,
) -> Result<[Complex<T>; L], RootFindingError>
where
    T: Copy
        + Zero
        + One
        + Neg<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + PartialOrd
        + fmt::Debug
        + RealField
        + Float,
    Const<N>: ToTypenum + DimSub<U1, Output = Const<L>>,
    Const<L>: DimSub<U1>,
    DefaultAllocator: Allocator<Const<L>, DimDiff<Const<L>, U1>> + Allocator<DimDiff<Const<L>, U1>>,
{
    unchecked_roots(&tf.denominator)
}

/// Compute the roots of the transfer function's numerator
///
/// Calculates the eigen values of a companion matrix constructed from the numerator.
///
/// # Arguments
///  * `tf` - the transfer function to check the zeros of
///
/// # Returns
///  * `[complex<T>; N]` - zeros of the tf
///
/// # Example
///
/// ```rust
/// use control_rs::transfer_function::{TransferFunction, zeros};
/// // Transfer function: G(s) = (2s + 4) / (s^2 + 3s + 2)
/// let tf = TransferFunction::new([2.0, 4.0], [1.0, 3.0, 2.0]);
/// let zeros = zeros(&tf);
/// ```
/// # Errors
/// * `NoRoots` - the function was not able to find any roots for the denominator
pub fn zeros<T, const M: usize, const N: usize, const L: usize>(
    tf: &TransferFunction<T, M, N>,
) -> Result<[Complex<T>; L], RootFindingError>
where
    T: Copy
        + Zero
        + One
        + Neg<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + PartialOrd
        + fmt::Debug
        + RealField
        + Float,
    Const<M>: ToTypenum + DimSub<U1, Output = Const<L>>,
    Const<L>: DimSub<U1>,
    DefaultAllocator: Allocator<Const<L>, DimDiff<Const<L>, U1>> + Allocator<DimDiff<Const<L>, U1>>,
{
    unchecked_roots(&tf.numerator)
}

/// Check if the system's poles lie on the left-half plane (LHP), a condition for stability.
///
/// Calculates the roots of the transfer function's denominator. If all roots of the characteristic
/// equation lie on the left half-plane (real part <= 0), then the transfer function is stable.
///
/// # Arguments
///  * `tf` - the transfer function to check the poles of
///
/// # Returns
///  * `bool` - if the transfer functions poles are all <= 0
///
/// # Example
/// ```
/// use control_rs::transfer_function::{TransferFunction, lhp};
/// // Transfer function: G(s) = (2s + 4) / (s^2 + 3s + 2)
/// let tf = TransferFunction::new([2.0, 4.0], [1.0, 3.0, 2.0]);
/// if lhp(&tf) {
///     println!("{tf} has stable poles");
/// } else {
///     println!("{tf} has unstable poles");
/// }
/// ```
///
/// ## References
/// - *Feedback Control of Dynamic Systems*, Franklin et al., Ch. 5: Stability Criteria
pub fn lhp<T, const M: usize, const N: usize, const L: usize>(
    tf: &TransferFunction<T, M, N>,
) -> bool
where
    T: Copy
        + Zero
        + One
        + Neg<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + PartialOrd
        + fmt::Debug
        + Float
        + RealField,
    Const<N>: ToTypenum + DimSub<U1, Output = Const<L>>,
    Const<L>: DimSub<U1>,
    DefaultAllocator: Allocator<Const<L>, DimDiff<Const<L>, U1>> + Allocator<DimDiff<Const<L>, U1>>,
{
    poles::<T, M, N, L>(tf).is_ok_and(|roots| {
        roots
            .iter()
            .all(|&pole| !pole.re.is_nan() && pole.re < T::zero())
    })
}

/// Helper function to create a state space model from a transfer function
///
/// Scales each of the coefficients by the highest order coefficient in the
/// denominator.
///
/// # Arguments
/// * `tf` - the transfer function that will be converted to monic arrays
///
/// # Returns
/// * `TransferFunction` - transfer function scaled by `self.denominator[0]`
///
/// # Example
/// ```
/// use control_rs::{assert_f64_eq, transfer_function::{TransferFunction, as_monic}};
/// let tf = TransferFunction::new([2.0, 1.0], [3.0, 1.0, 1.0]);
/// let monic_tf = as_monic(&tf);
/// assert_f64_eq!(monic_tf.numerator[1], 2.0 / 3.0);
/// ```
///
/// TODO: utilize polynomial utils
pub fn as_monic<T, const M: usize, const N: usize>(
    tf: &TransferFunction<T, M, N>,
) -> TransferFunction<T, M, N>
where
    T: Clone + Zero + Sub<Output = T> + Div<Output = T> + PartialOrd,
{
    let mut numerator = tf.numerator.clone();
    let mut denominator = tf.denominator.clone();

    if let Some(den_deg) = largest_nonzero_index(&denominator) {
        let leading_denominator = denominator[den_deg].clone();
        for b_i in &mut numerator {
            *b_i = b_i.clone() / leading_denominator.clone();
        }
        for a_i in &mut denominator {
            *a_i = a_i.clone() / leading_denominator.clone();
        }
    }

    TransferFunction {
        numerator,
        denominator,
    }
}

/// Converts a transfer function to a state space model.
///
/// TODO: finish docs
pub fn tf2ss<T, const N: usize, const M: usize, const L: usize>(
    tf: &TransferFunction<T, M, L>,
) -> StateSpace<SMatrix<T, N, N>, SMatrix<T, N, 1>, SMatrix<T, 1, N>, SMatrix<T, 1, 1>>
where
    T: Scalar
        + Clone
        + Zero
        + One
        + Neg<Output = T>
        + Div<Output = T>
        + Sub<Output = T>
        + PartialOrd,
    Const<L>: ToTypenum + DimSub<U1, Output = Const<N>>,
    Const<N>: DimSub<U1>,
{
    let tf_as_monic = as_monic(tf);
    control_canonical(&tf_as_monic.numerator, &tf_as_monic.denominator)
}

/// Fits a single-input, single-output (SISO) transfer function to frequency response data
/// using a least-squares algorithm.
///
/// This function solves the equation H(s) = B(s)/A(s) for the coefficients of B(s) and A(s),
/// where H(s) is the complex frequency response. The problem is formulated as a linear
/// system `H(s)A(s) = B(s)`.
///
/// We rewrite this as `B(s) - H(s)A(s) = 0`. To avoid the trivial solution of all zeros,
/// the leading denominator coefficient `a_n` is typically fixed to 1.
///
/// The equation for each frequency `w_k` is:
/// <pre>(b_0 + b_1*s + ... + b_m*s^m) - H(s)*(a_0 + a_1*s + ... + a_{n-1}*s^{n-1}) = H(s)*s^n</pre>
/// where `s = j*w_k`.
///
/// This can be set up as a linear system Ax = b, where x contains the unknown coefficients.
///
/// # Generic Arguments
/// * `M` - Degree of the numerator.
/// * `N` - Degree of the denominator.
/// * `K` - Number of frequency points.
///
/// # Arguments
/// * `freq_response`: The frequency response data for a single channel (input 0 to output 0).
/// * `m_order`: The desired order of the numerator polynomial (m).
/// * `n_order`: The desired order of the denominator polynomial (n).
///
/// # Returns
/// * `Ok(TransferFunction)`: The fitted transfer function if successful.
/// * `Err(&str)`: An error message if fitting fails (e.g., no response data).
///
/// # Errors
/// * The function will return a &str if the least squares solution fails.
///
/// # Example
/// ```
/// use control_rs::{
///     assert_f64_eq,
///     transfer_function::{fit, TransferFunction},
///     frequency_tools::{FrequencyResponse, FrequencyTools}
/// };
/// let tf = TransferFunction::new([1.0], [1.0, 1.0]);
/// let mut fr = FrequencyResponse::logspace(-1.0, 100.0);
/// tf.frequency_response::<100>(&mut fr);
/// let fitted_tf: TransferFunction<f64, 1, 2> = fit(&fr).expect("failed to fit fr data");
/// assert_f64_eq!(fitted_tf.numerator[0], 1.0, 1e-14);
/// assert_f64_eq!(fitted_tf.denominator[0], 1.0, 1e-14);
/// assert_f64_eq!(fitted_tf.denominator[1], 1.0, 1e-14);
/// ```
pub fn fit<T: Clone + RealField, const M: usize, const N: usize, const K: usize, const NM: usize>(
    freq_response: &FrequencyResponse<T, 1, 1, K>, // Example with K=100 points
) -> Result<TransferFunction<T, M, N>, &'static str>
where
    Const<M>: DimSub<U1>,
    Const<N>: DimSub<U1>,
    Const<K>: DimSub<U1> + DimMin<Const<NM>>,
    Const<NM>: DimMin<Const<K>, Output = Const<NM>> + DimSub<U1>,
    <Const<N> as DimSub<U1>>::Output: DimAdd<Const<M>, Output = Const<NM>>,
    DimMinimum<Const<K>, Const<NM>>: DimSub<U1>,
    DefaultAllocator: Allocator<Const<K>, Const<NM>>
        + Allocator<Const<NM>>
        + Allocator<Const<K>>
        + Allocator<DimDiff<DimMinimum<Const<K>, Const<NM>>, U1>>
        + Allocator<DimMinimum<Const<K>, Const<NM>>, Const<NM>>
        + Allocator<Const<K>, DimMinimum<Const<K>, Const<NM>>>
        + Allocator<DimMinimum<Const<K>, Const<NM>>>,
{
    // Ensure we have response data to work with.
    // This implementation focuses on a Single-Input Single-Output (SISO) system.
    let responses = if let Some(res) = &freq_response.responses {
        &res[0][0]
    } else {
        return Err("Frequency response data is missing.");
    };

    let frequencies = &freq_response.frequencies;
    let num_points = frequencies.len();

    // Construct the matrix A for the Least-Squares problem Ax = b.
    // Each row corresponds to a frequency point.
    // The columns correspond to the coefficients [b_0, ..., b_{M-1}, a_0, ..., a_{N-2}].
    let mut a_mat = OMatrix::<Complex<T>, Const<K>, Const<NM>>::zeros();

    // Construct the vector b.
    let mut b_vec = OMatrix::<Complex<T>, Const<K>, U1>::zeros();

    for k in 0..num_points {
        let w = frequencies[k].clone();
        let s = Complex::new(T::zero(), w);
        let h_s = responses[k].clone();

        // Fill with the part of the row for numerator coefficients (B(s))
        for i in 0..M {
            a_mat[(k, i)] = s.powu(u32::try_from(i).unwrap_or(0u32));
        }

        // Fill with the part of the row for denominator coefficients (A(s))
        for i in 0..(N - 1) {
            a_mat[(k, M + i)] = -h_s.clone() * s.powu(u32::try_from(i).unwrap_or(0u32));
        }

        // The right-hand side is H(s) * s^n, since we fixed a_n = 1.
        b_vec[k] = h_s * s.powu(u32::try_from(N - 1).unwrap_or(0u32));
    }

    // Solve the Least-Squares problem A*x = b.
    // The SVD decomposition is a robust way to solve this.
    let svd = a_mat.svd(true, true);
    let x = svd
        .solve(&b_vec, T::default_epsilon()) // 1e-10 is the tolerance
        .map_err(|_| "Least-squares solution failed.")?;

    // Extract coefficients from the solution vector x.
    let numerator: [T; M] = core::array::from_fn(|i| x[i].re.clone());
    let mut denominator: [T; N] = core::array::from_fn(|i| x[M + i - 1].re.clone());
    denominator[0] = T::one(); // The fixed coefficient

    Ok(TransferFunction {
        numerator,
        denominator,
    })
}
