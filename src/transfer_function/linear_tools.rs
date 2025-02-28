//! Miscellaneous tools to help work with transfer functions
//!
use nalgebra::{
    allocator::Allocator, Complex, DefaultAllocator, Dim, DimDiff, DimName, DimSub, OMatrix,
    RawStorage, RawStorageMut, RealField, U1,
};
use num_traits::{Float, Zero};

use crate::TransferFunction;

/// Computes the DC gain of a continuous transfer function.
///
/// The DC gain is the value of the transfer function as the frequency approaches zero.
///
/// <pre>
/// DC Gain = b_0 / a_0
/// </pre>
///
/// where \(b_0\) is the constant term of the numerator, and \(a_0\) is the constant term
/// of the denominator.
///
/// # Arguments
///
/// * `tf` - the transfer function to analyze
///
/// # Returns
///
/// * `Option<T>` - The DC gain if both numerator and denominator have a constant term;
///                 `None` otherwise
///
/// # Generic Arguments
///
/// * `T` - Scalar type for the coefficients (e.g., `f32`, `f64`)
/// * `N` - number of coefficients in the denominator
/// * `M` - number of coefficients in the numerator
///
/// # Example
///
/// ```rust
/// use control_rs::transfer_function::{TransferFunction, dcgain};
///
/// fn main() {
///     // Transfer function: G(s) = (2s + 4) / (s^2 + 3s + 2)
///     let tf = TransferFunction::new([2.0, 4.0], [1.0, 3.0, 2.0]);
///     let gain = dcgain(&tf);
///     println!("DC Gain: {gain:.2}");
/// }
/// ```
pub fn dcgain<T, M, N, S1, S2>(tf: &TransferFunction<T, M, N, S1, S2>) -> T
where
    T: Float,
    M: Dim + DimSub<U1>,
    N: Dim + DimSub<U1>,
    S1: RawStorage<T, M>,
    S2: RawStorage<T, N>,
{
    let denominator_constant = tf.denominator.constant();
    if denominator_constant.is_zero() {
        T::infinity()
    } else {
        tf.numerator.constant() / denominator_constant
    }
}

/// Compute the roots of the characteristic equation
///
/// Calculates the eigen values of the companion matrix of the transfer function's denominator.
/// The eigen values of the companion matrix are the roots of the characteristic equation (tf
/// denominator).
///
/// # Arguments
///
///  * `tf` - the transfer function to check the poles of
///
/// # Returns
///
///  * `[complex<T>; N]` - poles of the tf
///
/// # Example
///
/// ```rust
/// use control_rs::transfer_function::{TransferFunction, poles};
///
/// fn main() {
///     // Transfer function: G(s) = (2s + 4) / (s^2 + 3s + 2)
///     let tf = TransferFunction::new([2.0, 4.0], [1.0, 3.0, 2.0]);
///     let poles = poles(&tf); // contains 1 more element than it should
/// }
/// ```
///
/// ## References
///
/// - *Feedback Control of Dynamic Systems*, Franklin et al., Ch. 5: Stability Criteria
pub fn poles<T, M, N, S1, S2>(
    tf: &TransferFunction<T, M, N, S1, S2>,
) -> OMatrix<Complex<T>, DimDiff<N, U1>, U1>
where
    T: Copy + Zero + Float + RealField,
    N: DimSub<U1>,
    DimDiff<N, U1>: DimName + DimSub<U1>,
    S2: RawStorage<T, N>,
    DefaultAllocator: Allocator<DimDiff<N, U1>, DimDiff<N, U1>>
        + Allocator<DimDiff<N, U1>, DimDiff<DimDiff<N, U1>, U1>>
        + Allocator<DimDiff<DimDiff<N, U1>, U1>>
        + Allocator<DimDiff<N, U1>>,
{
    tf.denominator.roots()
}

/// Check if the system's poles lie in the left-half plane (LHP), a condition for stability
///
/// Calculates the eigen values of the companion matrix of the transfer function's denominator.
/// The eigen values of the companion matrix are the roots of the characteristic equation (tf
/// denominator). If all roots of the characteristic equation lie in the left half plane (real
/// part <= 0) then the transfer function is stable.
///
/// # Arguments
///
///  * `tf` - the transfer function to check the poles of
///
/// # Returns
///
///  * `bool` - if the transfer functions poles are all <= 0
///
/// # Example
///
/// ```rust
/// use control_rs::transfer_function::{TransferFunction, lhp};
///
/// fn main() {
///     // Transfer function: G(s) = (2s + 4) / (s^2 + 3s + 2)
///     let tf = TransferFunction::new([2.0, 4.0], [1.0, 3.0, 2.0]);
///
///     if lhp(&tf) {
///         println!("{tf} has stable poles");
///     } else {
///         println!("{tf} has unstable poles");
///     }
/// }
/// ```
///
/// ## References
///
/// - *Feedback Control of Dynamic Systems*, Franklin et al., Ch. 5: Stability Criteria
pub fn lhp<T, M, N, S1, S2>(tf: &TransferFunction<T, M, N, S1, S2>) -> bool
where
    T: Copy + Zero + Float + RealField,
    N: DimSub<U1>,
    DimDiff<N, U1>: DimName + DimSub<U1>,
    S2: RawStorage<T, N>,
    DefaultAllocator: Allocator<DimDiff<N, U1>, DimDiff<N, U1>>
        + Allocator<DimDiff<N, U1>, DimDiff<DimDiff<N, U1>, U1>>
        + Allocator<DimDiff<DimDiff<N, U1>, U1>>
        + Allocator<DimDiff<N, U1>>,
{
    poles(&tf)
        .iter()
        .all(|&pole| !pole.re.is_nan() && pole.re < T::zero())
}

/// Helper function to create a state space model from a transfer function
///
/// Scales each of the coefficients by the highest order coefficient in the
/// denominator.
///
/// # Arguments
///
/// * `tf` - the transfer function that will be converted to monic arrays
///
/// # Returns
///
/// * `TransferFunction` - transferfunction scaled by denominator[0]
///
/// # Example
///
/// ```
/// use control_rs::transfer_function::{TransferFunction, as_monic};
///
/// fn main() {
///     let tf = TransferFunction::new([1.0, 1.0], [1.0, 1.0, 1.0]);
///     let monic_tf = as_monic(&tf);
///     let (num, den) = (monic_tf.numerator.coefficients, monic_tf.denominator.coefficients);
/// }
/// ```
pub fn as_monic<T, M, N, S1, S2>(
    tf: &TransferFunction<T, M, N, S1, S2>,
) -> TransferFunction<T, M, N, S1, S2>
where
    T: Copy + Zero + Float,
    N: Dim,
    M: Dim,
    S1: RawStorageMut<T, M> + Copy,
    S2: RawStorageMut<T, N> + Copy,
{
    TransferFunction {
        numerator: tf.numerator / tf.denominator[0],
        denominator: tf.denominator / tf.denominator[0],
    }
}
