//! Miscellaneous tools to help work with transfer functions
//!
use core::{
    fmt,
    ops::{Div, Neg, Sub},
};

use nalgebra::{
    allocator::Allocator, Complex, Const, DefaultAllocator, DimDiff, DimSub, RealField, SMatrix,
    Scalar, U1,
};
use num_traits::{Float, One, Zero};

use crate::{state_space::utils::control_canonical, StateSpace, TransferFunction};

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
/// let poles = poles(&tf); // contains 1 more element than it should
/// ```
/// # Errors
/// * `NoRoots` - the function was not able to find any roots for the denominator
///
/// ## References
/// - *Feedback Control of Dynamic Systems*, Franklin et al., Ch. 5: Stability Criteria
pub fn poles<T, const M: usize, const N: usize, const L: usize>(
    tf: &TransferFunction<T, M, N>,
) -> Result<[Complex<T>; L], crate::polynomial::utils::NoRoots>
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
    Const<N>: DimSub<U1, Output = Const<L>>,
    Const<L>: DimSub<U1>,
    DefaultAllocator: Allocator<Const<L>, DimDiff<Const<L>, U1>> + Allocator<DimDiff<Const<L>, U1>>,
{
    crate::polynomial::utils::roots::<T, N, L>(&crate::polynomial::utils::reverse_array(
        tf.denominator,
    ))
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
    Const<N>: DimSub<U1, Output = Const<L>>,
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
/// TODO: move to polynomial utils
pub fn as_monic<T, const M: usize, const N: usize>(
    tf: &TransferFunction<T, M, N>,
) -> TransferFunction<T, M, N>
where
    T: Clone + Zero + Sub<Output = T> + Div<Output = T> + PartialOrd,
{
    let mut numerator = tf.numerator.clone();
    let mut denominator = tf.denominator.clone();

    if N > 0 && !denominator[N-1].is_zero() {
        let leading_denominator = if denominator[N-1] > T::zero() {denominator[N-1].clone()} else {T::zero() - denominator[N-1].clone()};
        numerator
            .iter_mut()
            .for_each(|b_i| *b_i = b_i.clone() / leading_denominator.clone());
        denominator
            .iter_mut()
            .for_each(|a_i| *a_i = a_i.clone() / leading_denominator.clone());
    }

    TransferFunction {
        numerator,
        denominator,
    }
}

/// Converts the transfer function to a state space model.
///
///
pub fn tf2ss<T, const N: usize, const M: usize, const L: usize>(
    tf: &TransferFunction<T, M, L>,
) -> StateSpace<SMatrix<T, N, N>, SMatrix<T, N, 1>, SMatrix<T, 1, N>, SMatrix<T, 1, 1>>
where
    T: Scalar + Clone + Zero + One + Neg<Output = T> + Div<Output = T> + Sub<Output = T> + PartialOrd,
    Const<L>: DimSub<U1, Output = Const<N>>,
{
    let tf_as_monic = as_monic(tf);
    control_canonical(&tf_as_monic.numerator, &tf_as_monic.denominator)
}
