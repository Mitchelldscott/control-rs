use nalgebra::{Complex, RealField};
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
///
///     if let Some(gain) = dcgain(&tf) {
///         println!("DC Gain: {:.2}", gain);
///     } else {
///         println!("DC Gain could not be computed.");
///     }
/// }
/// ```
pub fn dcgain<T, const N: usize, const M: usize>(tf: &TransferFunction<T, N, M>) -> T
where
    T: Float,
{
    let num_constant = tf.numerator[M - 1]; // Get constant term of numerator
    let denom_constant = tf.denominator[N - 1]; // Get constant term of denominator

    if denom_constant.is_zero() {
        T::infinity()
    } else {
        num_constant / denom_constant
    }
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
pub fn lhp<T, const N: usize, const M: usize>(tf: &TransferFunction<T, N, M>) -> bool
where
    T: Copy + Zero + Float +  RealField,
{
    if N <= 1 {
        return false;   
    }

    let mut poles = [Complex::new(T::zero(), T::zero()); N]; // actually only ever need N-1...
    crate::polynomial::roots(tf.denominator.as_slice(), &mut poles);

    poles.iter().take(N - 1).all(|&pole| !pole.re.is_nan() && pole.re < T::zero())
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
/// * (`num`, `den`) - coefficients of the numerator and denominator
///
/// # Example
///
/// ```
/// use control_rs::transfer_function::{TransferFunction, as_monic};
///
/// fn main() {
///     let tf = TransferFunction::new([1.0, 1.0], [1.0, 1.0, 1.0]);
///     let (num, den) = as_monic(&tf);
/// }
/// ```
pub fn as_monic<T, const N: usize, const M: usize>(
    tf: &TransferFunction<T, N, M>,
) -> ([T; M], [T; N])
where
    T: Copy + Zero + Float,
{
    let mut num = [T::zero(); M];
    let mut den = [T::zero(); N];
    // Ensure the leading coefficient of the denominator is non-zero
    let lead_den = tf.denominator[0];

    // Scale numerator coefficients by the leading denominator coefficient
    for (i, coeff) in tf.numerator.iter().enumerate() {
        num[i] = *coeff / lead_den;
    }

    // Scale denominator coefficients by the leading denominator coefficient
    for (i, coeff) in tf.denominator.iter().enumerate() {
        den[i] = *coeff / lead_den;
    }

    return (num, den);
}
