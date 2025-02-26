//! Miscellaneous tools to help work with transfer functions
//!
use nalgebra::{
    allocator::Allocator, Complex, Const, DefaultAllocator, DimDiff, DimSub, OMatrix, RealField, U1,
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
pub fn dcgain<T, const M: usize, const N: usize>(tf: &TransferFunction<T, M, N>) -> T
where
    T: Float,
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
pub fn poles<T, const M: usize, const N: usize>(
    tf: &TransferFunction<T, M, N>,
) -> Option<OMatrix<Complex<T>, Const<N>, U1>>
where
    T: Copy + Zero + Float + RealField,
    Const<N>: DimSub<U1>,
    DefaultAllocator: Allocator<Const<N>, DimDiff<Const<N>, U1>>
        + Allocator<DimDiff<Const<N>, U1>>
        + Allocator<Const<N>, Const<N>>
        + Allocator<Const<N>>,
{
    // tf.denominator.roots()
    None
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
pub fn lhp<T, const M: usize, const N: usize>(tf: &TransferFunction<T, M, N>) -> bool
where
    T: Copy + Zero + Float + RealField,
    Const<N>: DimSub<U1>,
    DefaultAllocator: Allocator<Const<N>, DimDiff<Const<N>, U1>>
        + Allocator<DimDiff<Const<N>, U1>>
        + Allocator<Const<N>, Const<N>>
        + Allocator<Const<N>>,
{
    if N == 1 {
        false
    } else if let Some(poles) = poles(&tf) {
        poles
            .iter()
            .take(N - 1)
            .all(|&pole| !pole.re.is_nan() && pole.re < T::zero())
    } else {
        false
    }
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
pub fn as_monic<T, const M: usize, const N: usize>(
    tf: &TransferFunction<T, M, N>,
) -> TransferFunction<T, M, N>
where
    T: Copy + Zero + Float,
{
    TransferFunction {
        numerator: tf.numerator / tf.denominator[0],
        denominator: tf.denominator / tf.denominator[0],
    }
}
